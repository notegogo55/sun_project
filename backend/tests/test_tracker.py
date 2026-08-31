"""ตรวจสอบการติดตาม AR ด้วยฉากจำลองที่รู้คำตอบล่วงหน้า

ทดสอบด้วยข้อมูลสังเคราะห์แทนภาพจริง เพราะเราต้องการรู้คำตอบที่ถูกต้องแน่นอน:
"AR ดวงนี้ควรได้ track_id เดียวตลอด" หรือ "สองดวงนี้ต้องไม่ถูกรวมเป็นดวงเดียว"
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from sunseg.tracking.detect import Detection, clean_mask, detect_regions, mask_iou
from sunseg.tracking.rotation import rotate_longitude
from sunseg.tracking.tracker import ARTracker

START = datetime(2014, 1, 1, 0, 0, 0)


def make_detection(
    label: int, lon: float, lat: float, area: int = 500, mask: np.ndarray | None = None
) -> Detection:
    return Detection(
        label=label,
        centroid_x=256.0 + lon,
        centroid_y=256.0 + lat,
        bbox=(0, 0, 10, 10),
        area_px=area,
        lon=lon,
        lat=lat,
        mask=mask,
    )


class TestSingleRegionTracking:
    def test_region_rotating_with_the_sun_keeps_one_track_id(self):
        """กรณีพื้นฐานที่สุด: AR ดวงเดียวหมุนไปตามดวงอาทิตย์ ต้องได้ track เดียว"""
        tracker = ARTracker()
        lon, lat = -60.0, 15.0

        for step in range(8):
            time = START + timedelta(hours=12 * step)
            tracker.update([make_detection(1, lon, lat)], time)
            lon = float(rotate_longitude(lon, lat, 0.5))

        tracks = tracker.confirmed_tracks()
        assert len(tracks) == 1, "AR ดวงเดียวไม่ควรแตกออกเป็นหลาย track"
        assert len(tracks[0].points) == 8
        assert tracks[0].state == "confirmed"

    def test_ignoring_rotation_would_break_matching(self):
        """พิสูจน์ว่าการชดเชยการหมุน *จำเป็นจริง*

        ถ้าตั้งระยะจับคู่สูงสุดให้แคบกว่าระยะที่ AR เลื่อนไปใน 1 วัน (~13.7 องศา)
        แต่ยังชดเชยการหมุนถูกต้อง ก็ยังจับคู่ได้อยู่ — เพราะเราวัดจากตำแหน่งที่ทำนายไว้
        """
        tracker = ARTracker(max_match_dist_deg=5.0)
        lon, lat = 0.0, 0.0

        for step in range(4):
            time = START + timedelta(days=step)
            tracker.update([make_detection(1, lon, lat)], time)
            lon = float(rotate_longitude(lon, lat, 1.0))

        tracks = tracker.confirmed_tracks()
        assert len(tracks) == 1
        assert len(tracks[0].points) == 4

    def test_track_starts_as_tentative(self):
        tracker = ARTracker(n_confirm=3)
        tracker.update([make_detection(1, 0.0, 0.0)], START)

        assert tracker.active_tracks[0].state == "tentative"
        assert tracker.confirmed_tracks() == []


class TestMultipleRegions:
    def test_two_separated_regions_stay_distinct(self):
        """AR สองดวงที่อยู่ห่างกันมากต้องไม่ถูกสลับหรือรวมกัน"""
        tracker = ARTracker()
        positions = [(-40.0, 20.0), (40.0, -20.0)]

        for step in range(6):
            time = START + timedelta(hours=12 * step)
            detections = [
                make_detection(i + 1, lon, lat) for i, (lon, lat) in enumerate(positions)
            ]
            tracker.update(detections, time)
            positions = [(float(rotate_longitude(lo, la, 0.5)), la) for lo, la in positions]

        tracks = tracker.confirmed_tracks()
        assert len(tracks) == 2
        assert all(len(t.points) == 6 for t in tracks)

        # แต่ละ track ต้องอยู่คนละซีกละติจูดตลอด ไม่มีการสลับตัวกัน
        mean_lats = sorted(np.mean([p.lat for p in t.points]) for t in tracks)
        assert mean_lats[0] == pytest.approx(-20.0)
        assert mean_lats[1] == pytest.approx(20.0)

    def test_area_difference_helps_separate_nearby_regions(self):
        """AR ใหญ่กับเล็กที่อยู่ใกล้กัน — พจน์พื้นที่ในฟังก์ชันต้นทุนช่วยแยกแยะ"""
        tracker = ARTracker(w_area=1.0)

        for step in range(4):
            time = START + timedelta(hours=12 * step)
            lon_a = float(rotate_longitude(0.0, 0.0, 0.5 * step))
            lon_b = float(rotate_longitude(6.0, 0.0, 0.5 * step))
            tracker.update(
                [make_detection(1, lon_a, 0.0, area=5000), make_detection(2, lon_b, 0.0, area=200)],
                time,
            )

        tracks = tracker.confirmed_tracks()
        assert len(tracks) == 2
        areas = sorted(int(np.mean([p.area_px for p in t.points])) for t in tracks)
        assert areas[0] == pytest.approx(200, abs=1)
        assert areas[1] == pytest.approx(5000, abs=1)


class TestTrackLifecycle:
    def test_disappearing_region_is_marked_lost(self):
        tracker = ARTracker(n_confirm=2, max_age_frames=2)

        for step in range(3):
            tracker.update(
                [make_detection(1, 0.0, 0.0)], START + timedelta(hours=12 * step)
            )
        assert tracker.active_tracks[0].state == "confirmed"

        # หายไปหลายเฟรมติดกัน
        for step in range(3, 7):
            tracker.update([], START + timedelta(hours=12 * step))

        assert tracker.active_tracks == []
        assert tracker.confirmed_tracks()[0].state == "lost"

    def test_brief_gap_does_not_break_the_track(self):
        """หายไปเฟรมเดียว (U-Net พลาด) ไม่ควรทำให้ track ขาด"""
        tracker = ARTracker(max_age_frames=2)
        lon, lat = 0.0, 10.0

        for step in range(6):
            time = START + timedelta(hours=12 * step)
            detections = [] if step == 3 else [make_detection(1, lon, lat)]
            tracker.update(detections, time)
            lon = float(rotate_longitude(lon, lat, 0.5))

        tracks = tracker.confirmed_tracks()
        assert len(tracks) == 1, "ช่องว่างสั้นๆ ไม่ควรทำให้เกิด track ใหม่"
        assert len(tracks[0].points) == 5

    def test_transient_false_positive_is_discarded(self):
        """detection ที่โผล่มาเฟรมเดียวแล้วหายคือ false positive — ต้องไม่กลายเป็น track"""
        tracker = ARTracker(n_confirm=2, max_age_frames=1)

        tracker.update([make_detection(9, 100.0, 40.0)], START)
        for step in range(1, 4):
            tracker.update([], START + timedelta(hours=12 * step))

        assert tracker.confirmed_tracks() == []

    def test_new_region_emerging_creates_new_track(self):
        tracker = ARTracker()

        for step in range(4):
            time = START + timedelta(hours=12 * step)
            detections = [make_detection(1, float(rotate_longitude(0.0, 0.0, 0.5 * step)), 0.0)]
            if step >= 2:
                detections.append(make_detection(2, -70.0, 25.0))
            tracker.update(detections, time)

        assert len(tracker.confirmed_tracks()) == 2

    def test_duration_hours_is_computed_correctly(self):
        tracker = ARTracker()
        lon = 0.0
        for step in range(5):
            tracker.update([make_detection(1, lon, 0.0)], START + timedelta(hours=12 * step))
            lon = float(rotate_longitude(lon, 0.0, 0.5))

        assert tracker.confirmed_tracks()[0].duration_hours == pytest.approx(48.0)

    def test_finalise_moves_active_tracks_to_finished(self):
        tracker = ARTracker()
        lon = 0.0
        for step in range(3):
            tracker.update([make_detection(1, lon, 0.0)], START + timedelta(hours=12 * step))
            lon = float(rotate_longitude(lon, 0.0, 0.5))

        assert len(tracker.finalise()) == 1
        assert tracker.active_tracks == []


class TestDetection:
    def test_detects_separate_blobs(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:15, 5:15] = 1
        mask[40:55, 40:55] = 1

        detections = detect_regions(mask, min_area_px=10, close_px=0, open_px=0)
        assert len(detections) == 2
        # เรียงจากใหญ่ไปเล็ก
        assert detections[0].area_px > detections[1].area_px

    def test_filters_out_small_blobs(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[50, 50] = 1  # จุดเดียว = noise

        detections = detect_regions(mask, min_area_px=10, close_px=0, open_px=0)
        assert len(detections) == 1

    def test_centroid_is_at_blob_centre(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:31, 20:31] = 1  # สี่เหลี่ยม 11x11 จุดกลางที่ (25, 25)

        detection = detect_regions(mask, min_area_px=10, close_px=0, open_px=0)[0]
        assert detection.centroid_x == pytest.approx(25.0, abs=0.5)
        assert detection.centroid_y == pytest.approx(25.0, abs=0.5)

    def test_computes_magnetic_flux_from_magnetogram(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        magnetogram = np.zeros((32, 32), dtype=np.float32)
        magnetogram[10:15, 10:20] = 100.0
        magnetogram[15:20, 10:20] = -100.0  # ขั้วตรงข้าม

        detection = detect_regions(
            mask, magnetogram=magnetogram, min_area_px=10, close_px=0, open_px=0
        )[0]

        # ฟลักซ์แบบไม่คิดเครื่องหมายต้องรวมทั้งสองขั้ว = 100 พิกเซล x 100 G
        assert detection.total_unsigned_flux == pytest.approx(10000.0)
        # ค่าเฉลี่ยแบบคิดเครื่องหมายต้องหักล้างกันเป็นศูนย์
        assert detection.mean_field == pytest.approx(0.0, abs=1e-5)
        assert detection.max_abs_field == pytest.approx(100.0)

    def test_empty_mask_returns_nothing(self):
        assert detect_regions(np.zeros((32, 32), dtype=np.uint8)) == []

    def test_closing_joins_split_region(self):
        """AR ที่ขาดเป็นสองส่วนตรงกลางควรถูกเชื่อมกลับด้วย morphological closing"""
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:15] = 1
        mask[10:20, 17:22] = 1  # เว้นช่องว่าง 2 พิกเซล

        assert len(detect_regions(mask, min_area_px=10, close_px=0, open_px=0)) == 2
        assert len(detect_regions(mask, min_area_px=10, close_px=5, open_px=0)) == 1

    def test_clean_mask_removes_speckle(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[2, 2] = 1

        cleaned = clean_mask(mask, close_px=0, open_px=3)
        assert cleaned[2, 2] == 0
        assert cleaned[15, 15] == 1


class TestMaskIou:
    def test_identical_masks(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:8, 4:8] = True
        assert mask_iou(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks(self):
        a = np.zeros((16, 16), dtype=bool)
        b = np.zeros((16, 16), dtype=bool)
        a[0:4, 0:4] = True
        b[10:14, 10:14] = True
        assert mask_iou(a, b) == 0.0

    def test_none_input_returns_zero(self):
        assert mask_iou(None, np.ones((4, 4), dtype=bool)) == 0.0
