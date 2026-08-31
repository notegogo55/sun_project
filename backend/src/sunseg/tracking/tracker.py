"""ติดตาม active region ข้ามเฟรมด้วย Hungarian matching + differential rotation

แนวคิดคล้าย SORT tracker ในงาน computer vision ทั่วไป แต่แทนที่จะใช้ Kalman filter
ทำนายการเคลื่อนที่ เราใช้ **ฟิสิกส์ของการหมุนของดวงอาทิตย์** ซึ่งแม่นกว่ามาก เพราะ
active region แทบไม่มีการเคลื่อนที่เฉพาะตัว — มันแค่ถูกพาไปกับการหมุนของดวงอาทิตย์

วงจรชีวิตของ track::

    tentative  --(เห็นครบ n_confirm เฟรม)-->  confirmed
        |                                          |
        +--(หายไป)--> deleted    (หายเกิน max_age_frames) --> lost

track ที่ยังเป็น ``tentative`` แล้วหายไปจะถูกลบทิ้งทันที เพื่อไม่ให้ false positive
ชั่วคราวจาก U-Net กลายเป็น track ปลอมที่ค้างอยู่
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection, mask_iou
from .rotation import angular_separation, rotate_longitude, wrap_longitude

logger = logging.getLogger(__name__)


@dataclass
class TrackPoint:
    """สถานะของ track ณ เวลาหนึ่ง"""

    time: datetime
    lon: float
    lat: float
    centroid_x: float
    centroid_y: float
    area_px: int
    total_unsigned_flux: float
    max_abs_field: float = 0.0
    mean_field: float = 0.0

    #: ความเข้มแสงราย channel ที่ติดมากับ detection ของจุดนี้ (ดู Detection.intensities)
    intensities: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "lon": None if np.isnan(self.lon) else round(self.lon, 3),
            "lat": None if np.isnan(self.lat) else round(self.lat, 3),
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "area_px": int(self.area_px),
            "total_unsigned_flux": float(self.total_unsigned_flux),
            "max_abs_field": float(self.max_abs_field),
            "mean_field": float(self.mean_field),
        }


@dataclass
class Track:
    """เส้นทางการเคลื่อนที่ของ active region หนึ่งดวงตลอดช่วงเวลา"""

    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    state: str = "tentative"  # tentative | confirmed | lost
    hits: int = 0
    age_since_update: int = 0

    #: mask ล่าสุด ใช้คำนวณ IoU กับเฟรมถัดไป
    last_mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def last(self) -> TrackPoint:
        return self.points[-1]

    @property
    def duration_hours(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].time - self.points[0].time).total_seconds() / 3600.0

    def predict(self, target_time: datetime) -> tuple[float, float]:
        """ทำนายตำแหน่ง (lon, lat) ณ เวลาที่กำหนด โดยใช้การหมุนของดวงอาทิตย์"""
        last = self.last
        delta_days = (target_time - last.time).total_seconds() / 86400.0
        predicted_lon = float(rotate_longitude(last.lon, last.lat, delta_days))
        return predicted_lon, last.lat

    def update(self, detection: Detection, time: datetime) -> None:
        self.points.append(
            TrackPoint(
                time=time,
                lon=detection.lon,
                lat=detection.lat,
                centroid_x=detection.centroid_x,
                centroid_y=detection.centroid_y,
                area_px=detection.area_px,
                total_unsigned_flux=detection.total_unsigned_flux,
                max_abs_field=detection.max_abs_field,
                mean_field=detection.mean_field,
                intensities=dict(detection.intensities),
            )
        )
        self.last_mask = detection.mask
        self.hits += 1
        self.age_since_update = 0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state,
            "n_points": len(self.points),
            "duration_hours": round(self.duration_hours, 2),
            "start_time": self.points[0].time.isoformat() if self.points else None,
            "end_time": self.points[-1].time.isoformat() if self.points else None,
            "max_area_px": max((p.area_px for p in self.points), default=0),
            "points": [p.to_dict() for p in self.points],
        }


class ARTracker:
    """จับคู่ detection กับ track ที่มีอยู่ ด้วย Hungarian algorithm

    ฟังก์ชันต้นทุนรวมสามสัญญาณเข้าด้วยกัน::

        cost = w_dist * (ระยะเชิงมุม / ระยะสูงสุด)
             + w_area * |ΔA| / max(A)
             + w_iou  * (1 - IoU)

    * **ระยะเชิงมุม** วัดจากตำแหน่งที่ *ทำนายไว้* หลังชดเชยการหมุนแล้ว
    * **พื้นที่** ช่วยแยก AR ใหญ่กับเล็กที่บังเอิญอยู่ใกล้กัน
    * **IoU** มีประโยชน์มากเมื่อเฟรมห่างกันไม่มาก แต่จะเป็น 0 เมื่อเฟรมห่างกันเกินไป
      จนไม่ทับซ้อนกัน — จึงถ่วงน้ำหนักไว้ต่ำกว่าระยะทาง
    """

    def __init__(
        self,
        w_dist: float = 1.0,
        w_area: float = 0.3,
        w_iou: float = 0.5,
        max_cost: float = 1.2,
        max_match_dist_deg: float = 12.0,
        n_confirm: int = 2,
        max_age_frames: int = 3,
    ) -> None:
        self.w_dist = w_dist
        self.w_area = w_area
        self.w_iou = w_iou
        self.max_cost = max_cost
        self.max_match_dist_deg = max_match_dist_deg
        self.n_confirm = n_confirm
        self.max_age_frames = max_age_frames

        self.tracks: list[Track] = []
        self.finished: list[Track] = []
        self._id_counter = itertools.count(1)

    # ------------------------------------------------------------------ #

    def _cost_matrix(
        self, detections: list[Detection], time: datetime
    ) -> tuple[np.ndarray, np.ndarray]:
        """สร้างเมทริกซ์ต้นทุน (n_tracks x n_detections) พร้อมหน้ากากคู่ที่เป็นไปไม่ได้"""
        n_tracks, n_dets = len(self.tracks), len(detections)
        cost = np.full((n_tracks, n_dets), np.inf)
        feasible = np.zeros((n_tracks, n_dets), dtype=bool)

        for i, track in enumerate(self.tracks):
            pred_lon, pred_lat = track.predict(time)

            for j, det in enumerate(detections):
                if not det.has_heliographic or np.isnan(pred_lon):
                    continue

                distance = float(angular_separation(pred_lon, pred_lat, det.lon, det.lat))
                if distance > self.max_match_dist_deg:
                    continue

                area_a, area_b = track.last.area_px, det.area_px
                area_term = abs(area_a - area_b) / max(area_a, area_b, 1)
                iou_term = 1.0 - mask_iou(track.last_mask, det.mask)

                total = (
                    self.w_dist * (distance / self.max_match_dist_deg)
                    + self.w_area * area_term
                    + self.w_iou * iou_term
                )
                if total <= self.max_cost:
                    cost[i, j] = total
                    feasible[i, j] = True

        return cost, feasible

    def update(self, detections: list[Detection], time: datetime) -> list[Track]:
        """ประมวลผลหนึ่งเฟรม คืนรายการ track ที่ยังทำงานอยู่

        ต้องเรียกตามลำดับเวลาจากเก่าไปใหม่
        """
        if not self.tracks:
            for det in detections:
                self._start_track(det, time)
            return self.active_tracks

        cost, feasible = self._cost_matrix(detections, time)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if feasible.any():
            # แทน inf ด้วยค่าโทษสูงจำกัด เพราะ linear_sum_assignment รับ inf ไม่ได้
            solvable = np.where(feasible, cost, self.max_cost * 1000.0)
            rows, cols = linear_sum_assignment(solvable)
            for i, j in zip(rows, cols, strict=True):
                if feasible[i, j]:
                    self.tracks[i].update(detections[j], time)
                    matched_tracks.add(i)
                    matched_dets.add(j)

        # track ที่ไม่ถูกจับคู่ในเฟรมนี้
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.age_since_update += 1

        # detection ที่ไม่ถูกจับคู่ = AR ที่เพิ่งโผล่ (หรือเพิ่งหมุนเข้ามาจากขอบตะวันออก)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._start_track(det, time)

        self._update_states()
        return self.active_tracks

    def _start_track(self, detection: Detection, time: datetime) -> None:
        track = Track(track_id=next(self._id_counter))
        track.update(detection, time)
        self.tracks.append(track)

    def _update_states(self) -> None:
        """เลื่อนสถานะของ track และเก็บกวาดตัวที่หายไปนานเกิน"""
        still_active: list[Track] = []

        for track in self.tracks:
            if track.state == "tentative" and track.hits >= self.n_confirm:
                track.state = "confirmed"

            if track.age_since_update > self.max_age_frames:
                # tentative ที่หายไป = false positive ชั่วคราว ทิ้งได้เลย
                if track.state == "confirmed":
                    track.state = "lost"
                    self.finished.append(track)
                continue

            still_active.append(track)

        self.tracks = still_active

    # ------------------------------------------------------------------ #

    @property
    def active_tracks(self) -> list[Track]:
        return list(self.tracks)

    @property
    def all_tracks(self) -> list[Track]:
        """ทุก track ทั้งที่ยังทำงานและที่จบไปแล้ว เรียงตาม id"""
        return sorted(self.tracks + self.finished, key=lambda t: t.track_id)

    def confirmed_tracks(self, min_points: int = 2) -> list[Track]:
        """เฉพาะ track ที่ยืนยันแล้วและมีจุดมากพอ — ใช้แสดงผลและวิเคราะห์"""
        return [
            t
            for t in self.all_tracks
            if t.state in ("confirmed", "lost") and len(t.points) >= min_points
        ]

    def finalise(self) -> list[Track]:
        """ปิดการติดตาม ย้าย track ที่ยังทำงานอยู่ทั้งหมดไปยังรายการที่เสร็จแล้ว"""
        for track in self.tracks:
            if track.state == "confirmed":
                self.finished.append(track)
        self.tracks = []
        return sorted(self.finished, key=lambda t: t.track_id)

    def summary(self) -> str:
        tracks = self.confirmed_tracks()
        if not tracks:
            return "ไม่มี track ที่ยืนยันได้"

        durations = [t.duration_hours for t in tracks]
        return "\n".join(
            [
                f"track ที่ยืนยันแล้ว {len(tracks)} เส้น",
                f"  ระยะเวลาติดตาม: เฉลี่ย {np.mean(durations):.1f} ชม. "
                f"สูงสุด {np.max(durations):.1f} ชม.",
                f"  จำนวนจุดเฉลี่ย: {np.mean([len(t.points) for t in tracks]):.1f} จุดต่อ track",
            ]
        )


def build_tracker(cfg) -> ARTracker:
    """สร้าง tracker จาก ``configs/tracking.yaml``"""
    return ARTracker(
        w_dist=cfg.w_dist,
        w_area=cfg.w_area,
        w_iou=cfg.w_iou,
        max_cost=cfg.max_cost,
        max_match_dist_deg=cfg.max_match_dist_deg,
        n_confirm=cfg.n_confirm,
        max_age_frames=cfg.max_age_frames,
    )


__all__ = ["ARTracker", "Track", "TrackPoint", "build_tracker", "wrap_longitude"]
