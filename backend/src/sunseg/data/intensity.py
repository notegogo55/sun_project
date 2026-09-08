"""ดึงความเข้มแสงราย active region จาก AIA ผ่าน mask ที่ U-Net ทำนาย แล้วจับคู่กับ HARP

ส่วนหนึ่งของงานเปรียบเทียบ SHARP-only กับ SHARP+intensity (ดู
``.scratch/lstm-intensity-comparison``) — ประกอบ pipeline เต็มเส้น:

    U-Net ทำนาย mask -> แยกเป็น AR รายดวง (``sunseg.tracking.detect``) -> วัดความเข้มแสง
    จากพิกเซล AIA ในแต่ละดวง (``sunseg.data.aia.region_intensities``) -> normalise เทียบ
    quiet Sun เพื่อลบ instrument degradation -> จับคู่แต่ละดวงเข้ากับ HARPNUM ผ่านแผนที่
    ระบุตัวตนจาก ``sunseg.data.build_masks.build_fulldisk_identity_map``

**สำคัญ**: ความเข้มแสงที่ได้เป็นของ **AR ดวงเดียว** ไม่ใช่ค่าเฉลี่ยทั้งจาน — SHARP
parameters ที่โปรเจคนี้ใช้เป็นค่าราย active region การเฉลี่ยทั้งจานจะให้ค่าเดียวกันสำหรับ
ทุก AR ณ เวลานั้น ซึ่งไม่มีข้อมูลแยกแยะดวงที่จะปะทุออกจากดวงที่ไม่ปะทุ
"""

from __future__ import annotations

import logging

import numpy as np

from ..tracking.detect import Detection, detect_regions
from .aia import region_intensities

logger = logging.getLogger(__name__)

#: สถิติทั้ง 5 ค่าที่ region_intensities คำนวณต่อหนึ่งช่อง — ลำดับนี้ใช้ตั้งชื่อคอลัมน์
STATS: tuple[str, ...] = ("mean", "median", "p95", "total", "n_pixels")


def quiet_sun_normalise(
    channel_array: np.ndarray, ar_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    """หารค่าทั้งภาพด้วยค่ากลางของพิกเซล quiet Sun (พิกเซลบนจานที่อยู่นอก ``ar_mask``)

    ลบผลการเสื่อมของกล้อง (instrument degradation) ที่คูณอยู่ทั้งภาพออกไปทั้งเศษทั้งส่วน —
    README เตือนว่าค่า DN/s ดิบ "เทียบกันได้เฉพาะภายในเฟรมเดียวกัน ห้ามเทียบข้ามปี" วิธีนี้
    ทำให้เทียบข้ามปีได้โดยไม่ต้องเพิ่ม dependency ``aiapy`` ที่ README จงใจเลี่ยงไว้

    คุณสมบัติที่ทดสอบได้ตรง ๆ และเป็นเหตุผลที่วิธีนี้แก้ degradation ได้: ถ้า
    ``channel_array`` ทั้งภาพถูกคูณด้วยค่าคงที่บวก (จำลอง degradation ที่คูณสม่ำเสมอทั้ง
    ภาพ) ค่าหลัง normalise ต้องเท่าเดิมทุกพิกเซล เพราะค่าคงที่นั้นถูกหารออกไปทั้งเศษ
    (ค่าของ AR) ทั้งส่วน (ระดับ quiet Sun)

    Returns
    -------
    ``(normalised, quiet_sun_level)`` — ``normalised`` มีหน่วยเป็น "เท่าของ quiet Sun"
    ``quiet_sun_level`` เป็น NaN ถ้าไม่มีพิกเซล quiet Sun ที่ใช้ได้เลย (เช่น mask คลุมทั้งจาน)
    """
    array = np.asarray(channel_array, dtype=np.float64)
    ar = np.asarray(ar_mask).astype(bool)
    quiet = np.isfinite(array) & ~ar

    if not quiet.any():
        return np.full(array.shape, np.nan), float("nan")

    level = float(np.median(array[quiet]))
    if not np.isfinite(level) or level == 0:
        return np.full(array.shape, np.nan), float("nan")

    return array / level, level


def match_detection_to_harp(detection: Detection, identity_map: np.ndarray) -> int | None:
    """จับคู่ AR หนึ่งดวงที่ U-Net ตรวจพบเข้ากับหมายเลข HARP จากแผนที่ระบุตัวตน

    ใช้เสียงข้างมาก (plurality) ของค่า ``identity_map`` ที่ไม่ใช่ 0 ภายใน ``detection.mask``
    **AR ที่ไม่ซ้อนทับกับ HARP ใดเลยจะไม่ถูกจับคู่** (คืน ``None``) แทนที่จะถูกยัดให้ดวงที่
    ใกล้ที่สุด — การจับคู่ผิดจะปนความ noise เข้าไปในผลการทดลองจนแยกไม่ออกว่าผลลบมาจาก
    intensity ไม่มีประโยชน์จริง หรือมาจากการจับคู่พลาด (ดู spec หัวข้อ "การจับ AR เข้ากับ HARP")
    """
    mask = np.asarray(detection.mask).astype(bool)
    identity = np.asarray(identity_map)
    if mask.shape != identity.shape:
        return None

    values = identity[mask]
    values = values[values != 0]
    if values.size == 0:
        return None

    unique, counts = np.unique(values, return_counts=True)
    return int(unique[np.argmax(counts)])


def extract_frame_intensities(
    timestamp: str,
    magnetogram: np.ndarray,
    predicted_mask: np.ndarray,
    identity_map: np.ndarray,
    channel_arrays: dict[str, np.ndarray | None],
    min_area_px: int = 12,
    solar_map=None,
) -> list[dict]:
    """สกัดความเข้มแสงราย HARP ของเฟรมหนึ่ง — ครบทั้ง detect, normalise, และจับคู่

    Parameters
    ----------
    channel_arrays
        ``{channel_key: อาร์เรย์ DN/s ดิบ หรือ None ถ้าเฟรมนี้ไม่มีช่องนั้น}`` — ฟังก์ชันนี้
        normalise เทียบ quiet Sun ให้เอง ไม่ต้องแปลงมาก่อน

    Returns
    -------
    หนึ่ง dict ต่อหนึ่ง AR ที่จับคู่กับ HARP ได้สำเร็จเท่านั้น (AR ที่จับคู่ไม่ได้ถูกทิ้ง — ไม่
    ถูกเติมด้วยค่า sentinel) แต่ละ dict มี ``HARPNUM``, ``issue_time``, ``lon``, ``lat``,
    ``area_px`` และสถิติ 5 ค่า (mean/median/p95/total/n_pixels) คูณทุกช่องใน
    ``channel_arrays`` — ช่องที่ไม่มีข้อมูลในเฟรมนี้ได้ ``NaN`` ไม่ใช่ error พร้อมระดับ
    quiet Sun ที่ใช้ normalise ของแต่ละช่อง (เก็บไว้ตรวจด่านเรื่องการลบ drift ข้ามปี)
    """
    detections = detect_regions(
        predicted_mask, magnetogram=magnetogram, min_area_px=min_area_px, solar_map=solar_map
    )
    if not detections:
        return []

    normalised_by_channel: dict[str, tuple[np.ndarray, float]] = {
        channel: quiet_sun_normalise(raw, predicted_mask)
        for channel, raw in channel_arrays.items()
        if raw is not None
    }

    rows: list[dict] = []
    for detection in detections:
        harpnum = match_detection_to_harp(detection, identity_map)
        if harpnum is None:
            continue

        row: dict = {
            "HARPNUM": harpnum,
            "issue_time": timestamp,
            "area_px": detection.area_px,
            "lon": None if np.isnan(detection.lon) else float(detection.lon),
            "lat": None if np.isnan(detection.lat) else float(detection.lat),
        }

        for channel in channel_arrays:
            data = normalised_by_channel.get(channel)
            if data is None:
                for stat in STATS:
                    row[f"{channel}_{stat}"] = float("nan")
                row[f"{channel}_quiet_sun"] = float("nan")
                continue

            normalised, quiet_level = data
            stats = region_intensities([detection], normalised)[0]
            for stat in STATS:
                row[f"{channel}_{stat}"] = stats[stat]
            row[f"{channel}_quiet_sun"] = quiet_level

        rows.append(row)

    return rows
