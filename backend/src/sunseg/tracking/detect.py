"""แปลง mask ที่ได้จาก U-Net ให้เป็นรายการ active region พร้อมคุณสมบัติ

ขั้นตอน: ทำความสะอาดรูปทรงด้วย morphology -> หา connected component -> คัดทิ้ง
blob เล็กเกินไป -> คำนวณคุณสมบัติของแต่ละดวง (จุดศูนย์กลาง, พื้นที่, ฟลักซ์แม่เหล็ก)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """active region หนึ่งดวงที่ตรวจพบในเฟรมเดียว"""

    label: int
    centroid_x: float
    centroid_y: float
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) โดย x1/y1 เป็นแบบไม่รวมปลาย
    area_px: int

    #: ตำแหน่งเฮลิโอกราฟิก — เติมเมื่อมีข้อมูล WCS ของภาพ
    lon: float = float("nan")
    lat: float = float("nan")

    #: ฟลักซ์แม่เหล็กรวมแบบไม่คิดเครื่องหมาย (หน่วยตาม magnetogram ที่ป้อนเข้ามา)
    total_unsigned_flux: float = 0.0
    mean_field: float = 0.0
    max_abs_field: float = 0.0

    #: mask ไบนารีเฉพาะดวงนี้ ใช้คำนวณ IoU ตอนจับคู่ (ไม่บันทึกลงไฟล์)
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)

    #: ความเข้มแสงราย channel ที่วัดจากพิกเซลใน mask นี้ — ``{"171": {"mean": ..}, ..}``
    #:
    #: ``detect_regions`` ไม่ได้เติมค่านี้เอง (มันไม่รู้จักภาพ AIA) ผู้เรียกเป็นคนเติมทีหลัง
    #: ด้วย ``sunseg.data.aia.region_intensities`` มีไว้เพื่อให้ค่าติดไปกับ detection ตอนที่
    #: tracker จับคู่ข้ามเฟรม — ไม่งั้นจะไม่มีทางรู้ว่าค่าไหนเป็นของ AR ดวงไหนหลังจับคู่แล้ว
    intensities: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def has_heliographic(self) -> bool:
        return not (np.isnan(self.lon) or np.isnan(self.lat))

    def to_dict(self, include_mask: bool = False) -> dict:
        data = {
            "label": self.label,
            "centroid_x": round(self.centroid_x, 2),
            "centroid_y": round(self.centroid_y, 2),
            "bbox": list(self.bbox),
            "area_px": self.area_px,
            "lon": None if np.isnan(self.lon) else round(self.lon, 3),
            "lat": None if np.isnan(self.lat) else round(self.lat, 3),
            "total_unsigned_flux": float(self.total_unsigned_flux),
            "mean_field": float(self.mean_field),
            "max_abs_field": float(self.max_abs_field),
        }
        if include_mask and self.mask is not None:
            data["mask"] = self.mask
        return data


def clean_mask(mask: np.ndarray, close_px: int = 3, open_px: int = 2) -> np.ndarray:
    """ปิดรูเล็กๆ ภายใน AR แล้วลบจุด noise ที่กระจัดกระจาย

    ลำดับสำคัญ: **closing ก่อน opening** — closing เชื่อมส่วนของ AR เดียวกันที่ขาด
    จากกัน (ซึ่งมักเกิดตรงเส้นกลับขั้วที่สนามอ่อน) ถ้าทำ opening ก่อน ชิ้นส่วนเล็กๆ
    เหล่านั้นจะถูกลบทิ้งไปเสียก่อนจนเชื่อมไม่ได้
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)

    if close_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if open_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary


def detect_regions(
    mask: np.ndarray,
    magnetogram: np.ndarray | None = None,
    min_area_px: int = 12,
    close_px: int = 3,
    open_px: int = 2,
    solar_map=None,
    keep_masks: bool = True,
) -> list[Detection]:
    """หา active region ทั้งหมดใน mask หนึ่งเฟรม

    Parameters
    ----------
    mask
        mask ไบนารี (หรือความน่าจะเป็นที่ threshold แล้ว) รูปทรง ``(H, W)``
    magnetogram
        ภาพ magnetogram ค่าดิบรูปทรงเดียวกัน ใช้คำนวณฟลักซ์ (ไม่บังคับ)
    solar_map
        ``sunpy.map.GenericMap`` ที่มี WCS — ถ้าให้มาจะเติมพิกัด lon/lat ให้
    keep_masks
        เก็บ mask รายดวงไว้สำหรับคำนวณ IoU ตอน tracking (ใช้หน่วยความจำเพิ่ม)

    Returns
    -------
    รายการ :class:`Detection` เรียงจากพื้นที่มากไปน้อย
    """
    binary = clean_mask(mask, close_px, open_px)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    detections: list[Detection] = []
    # เริ่มที่ 1 เพราะ label 0 คือพื้นหลัง
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        component = labels == label_id
        detection = Detection(
            label=label_id,
            centroid_x=float(centroids[label_id, 0]),
            centroid_y=float(centroids[label_id, 1]),
            bbox=(x0, y0, x0 + width, y0 + height),
            area_px=area,
            mask=component if keep_masks else None,
        )

        if magnetogram is not None:
            values = np.asarray(magnetogram)[component]
            finite = values[np.isfinite(values)]
            if finite.size:
                detection.total_unsigned_flux = float(np.abs(finite).sum())
                detection.mean_field = float(finite.mean())
                detection.max_abs_field = float(np.abs(finite).max())

        detections.append(detection)

    if solar_map is not None and detections:
        _attach_heliographic(detections, solar_map)

    detections.sort(key=lambda d: d.area_px, reverse=True)
    logger.debug(
        "ตรวจพบ AR %d ดวง (จาก component ทั้งหมด %d, เกณฑ์พื้นที่ขั้นต่ำ %d พิกเซล)",
        len(detections),
        n_labels - 1,
        min_area_px,
    )
    return detections


def _attach_heliographic(detections: list[Detection], solar_map) -> None:
    """เติมพิกัด Stonyhurst ให้ทุก detection ในครั้งเดียว (เร็วกว่าแปลงทีละดวงมาก)"""
    from .rotation import pixel_to_stonyhurst

    xs = np.array([d.centroid_x for d in detections])
    ys = np.array([d.centroid_y for d in detections])

    try:
        lons, lats = pixel_to_stonyhurst(xs, ys, solar_map)
    except Exception as exc:  # noqa: BLE001 — การแปลงพิกัดล้มเหลวไม่ควรทำให้ทั้ง pipeline พัง
        logger.warning("แปลงพิกัดเฮลิโอกราฟิกไม่สำเร็จ: %s", exc)
        return

    for detection, lon, lat in zip(detections, lons, lats, strict=True):
        detection.lon = float(lon)
        detection.lat = float(lat)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """IoU ระหว่าง mask สองอัน คืน 0.0 ถ้าอันใดอันหนึ่งไม่มี"""
    if a is None or b is None:
        return 0.0
    intersection = float(np.logical_and(a, b).sum())
    if intersection == 0:
        return 0.0
    return intersection / float(np.logical_or(a, b).sum())


def summarise_detections(detections: list[Detection]) -> str:
    if not detections:
        return "ไม่พบ active region"

    areas = [d.area_px for d in detections]
    lines = [f"พบ active region {len(detections)} ดวง (พื้นที่รวม {sum(areas):,} พิกเซล)"]
    for d in detections[:5]:
        position = (
            f"lon={d.lon:+7.2f} lat={d.lat:+6.2f}"
            if d.has_heliographic
            else f"pix=({d.centroid_x:.0f},{d.centroid_y:.0f})"
        )
        lines.append(f"  #{d.label:<3} พื้นที่ {d.area_px:>6,} px   {position}")
    if len(detections) > 5:
        lines.append(f"  ... และอีก {len(detections) - 5} ดวง")
    return "\n".join(lines)
