"""สร้าง ground-truth mask ของ active region จาก SHARP bitmap

**ที่มาของ label**: JSOC คำนวณ ``bitmap`` segment ให้ทุก SHARP patch อยู่แล้ว ซึ่ง
ระบุว่าพิกเซลไหนอยู่ในขอบเขตของ active region เราจึงได้ label คุณภาพสูงหลายหมื่น
ภาพโดยไม่ต้อง annotate เอง

**ขั้นตอน** สำหรับภาพเต็มดวงหนึ่งเฟรม::

    1. หา SHARP patch ทุกอันที่เวลาเดียวกัน
    2. reproject bitmap ของแต่ละ patch ไปวางบนกริด 512x512 ที่ย่อจากภาพเต็มดวง (ใช้ WCS)
    3. รวมเป็น mask ไบนารีอันเดียว
    4. ย่อภาพเต็มดวงเหลือ 512x512 แล้วบันทึกทั้งคู่เป็น .npy
    5. ลบไฟล์ FITS ทิ้ง (ดิสก์เหลือน้อย)

.. warning::
   ก่อนเทรน **ต้องตรวจด้วยตา** ว่า mask วางทับ active region ตรงตำแหน่งจริง
   ใช้ ``python backend/scripts/plot_masks.py`` — ถ้า WCS ผิด mask จะเลื่อนไปทั้งภาพ
   และการเทรนจะเสียเวลาไปเปล่าๆ
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: ค่าที่เป็นไปได้ของ SHARP bitmap ตามเอกสารของ JSOC
#: 1 = สนามอ่อน นอกเส้นขอบเขต, 2 = สนามแรง นอกเส้นขอบเขต
#: 33 = สนามอ่อน ในเส้นขอบเขต, 34 = สนามแรง ในเส้นขอบเขต
BITMAP_WEAK_OUTSIDE = 1
BITMAP_STRONG_OUTSIDE = 2
BITMAP_WEAK_INSIDE = 33
BITMAP_STRONG_INSIDE = 34

#: ค่าตั้งแต่นี้ขึ้นไปถือว่าอยู่ในขอบเขต active region
DEFAULT_AR_THRESHOLD = BITMAP_WEAK_INSIDE


def assert_bitmap_encoding(bitmap: np.ndarray, source: str = "") -> dict[int, int]:
    """ตรวจว่าค่าใน bitmap ตรงกับที่เอกสารระบุ แล้วคืนฮิสโทแกรมของค่า

    เรียกครั้งแรกที่ประมวลผลเสมอ — ไม่ควร hardcode ค่า 33 โดยไม่เคยเห็นข้อมูลจริง
    ถ้า JSOC เปลี่ยนรูปแบบการเข้ารหัส เราต้องรู้ทันทีแทนที่จะได้ mask ว่างเปล่าเงียบๆ
    """
    values, counts = np.unique(bitmap[np.isfinite(bitmap)].astype(int), return_counts=True)
    histogram = {int(v): int(c) for v, c in zip(values, counts, strict=True)}

    expected = {BITMAP_WEAK_OUTSIDE, BITMAP_STRONG_OUTSIDE, BITMAP_WEAK_INSIDE, BITMAP_STRONG_INSIDE}
    unexpected = set(histogram) - expected - {0}

    logger.info("ฮิสโทแกรมค่า bitmap%s: %s", f" ({source})" if source else "", histogram)
    if unexpected:
        logger.warning(
            "พบค่า bitmap ที่ไม่คาดคิด: %s (คาดหวัง %s) — "
            "รูปแบบการเข้ารหัสของ JSOC อาจเปลี่ยนไป ตรวจสอบเอกสารก่อนใช้งานต่อ",
            sorted(unexpected),
            sorted(expected),
        )
    if not (set(histogram) & {BITMAP_WEAK_INSIDE, BITMAP_STRONG_INSIDE}):
        logger.warning(
            "ไม่พบพิกเซลที่อยู่ในขอบเขต AR (ค่า >= %d) เลยใน bitmap นี้ — "
            "mask ที่ได้จะว่างเปล่า",
            DEFAULT_AR_THRESHOLD,
        )
    return histogram


def bitmap_to_binary(bitmap: np.ndarray, threshold: int = DEFAULT_AR_THRESHOLD) -> np.ndarray:
    """แปลง SHARP bitmap เป็น mask ไบนารี (1 = อยู่ในขอบเขต active region)"""
    data = np.asarray(bitmap)
    return (np.nan_to_num(data, nan=0.0) >= threshold).astype(np.uint8)


def binned_wcs_from(wcs, shape: tuple[int, int], target_size: int):
    """คืน ``(wcs, shape)`` ของกริดที่ถูกย่อเหลือ ``target_size`` พิกเซล

    ใช้การ slice WCS ของ astropy (``wcs[::f, ::f]``) แทนการแก้ CDELT/CRPIX เอง
    เพราะ astropy รู้ว่าต้องจัดการกับ PC matrix, CD matrix หรือ CROTA อย่างไร
    (HMI ใช้ PC matrix — การคูณ CDELT ตรงๆ จะได้ผลผิดถ้า header เปลี่ยนรูปแบบ)

    แยกออกมาจาก :func:`binned_wcs` เพื่อให้เรียกได้เมื่อมีแค่ WCS กับ shape โดยไม่มี
    ตัว Map — กรณีที่เกิดขึ้นตอนวางภาพ AIA ลงกริดของเฟรมที่เก็บเป็น ``.npy`` เปล่า
    (FITS ต้นฉบับถูกลบไปแล้ว WCS จึงมาจาก ``FrameWcsStore`` แทน)
    """
    ny, nx = shape
    if ny % target_size or nx % target_size:
        raise ValueError(
            f"ขนาดภาพ {ny}x{nx} หารด้วย target_size {target_size} ไม่ลงตัว"
        )
    factor_y, factor_x = ny // target_size, nx // target_size
    return wcs[::factor_y, ::factor_x], (target_size, target_size)


def binned_wcs(fulldisk_map, target_size: int):
    """เหมือน :func:`binned_wcs_from` แต่รับ sunpy Map โดยตรง"""
    return binned_wcs_from(fulldisk_map.wcs, fulldisk_map.data.shape, target_size)


def reproject_patch_to_grid(
    patch_map, target_wcs, target_shape: tuple[int, int], threshold: int = DEFAULT_AR_THRESHOLD
) -> np.ndarray:
    """วาง bitmap ของ SHARP patch หนึ่งอันลงบนกริดเป้าหมายที่ระบุ

    ใช้ ``reproject_interp`` ซึ่งอ่าน WCS ของทั้งสองภาพแล้วจับคู่พิกัดท้องฟ้าให้
    ถูกต้อง วิธีนี้ทำงานได้กับทั้ง series พิกัด CCD และ CEA โดยไม่ต้องเขียนสูตร
    แปลงพิกัดเอง (ซึ่งเป็นจุดที่พลาดง่ายมาก)

    ``target_wcs`` เป็นกริดใดก็ได้ — ปกติคือกริด 512 ที่ได้จาก :func:`binned_wcs`
    การ reproject ลงกริดปลายทางโดยตรงถูกกว่าการ reproject ลง 4096 แล้วค่อยย่อ
    **56 เท่า** (162 วิ -> 2.9 วิ ต่อเฟรม วัดจาก 15 patch จริง) เพราะต้นทุนแปรผัน
    ตามจำนวนพิกเซลปลายทาง ซึ่งลดลง 64 เท่า ผลลัพธ์แทบไม่ต่างกัน (IoU 0.9969 เทียบ
    กับวิธีเดิม, ได้ patch จำนวนเท่ากัน) — วัดแล้วว่า aliasing ไม่เป็นปัญหาจริง
    เพราะขอบเขต AR ใหญ่และเรียบเมื่อเทียบกับการย่อ 8 เท่า

    Returns
    -------
    mask ไบนารีขนาด ``target_shape``
    """
    from reproject import reproject_interp

    # ทำ threshold *ก่อน* reproject: การ interpolate ค่า 1/2/33/34 ตรงๆ จะสร้างค่ากลาง
    # ที่ไม่มีความหมาย (เช่น 17.5) แต่การ interpolate ค่า 0/1 แล้วปัดเป็นการหาขอบเขต
    # ที่ถูกต้อง
    binary = bitmap_to_binary(patch_map.data, threshold)

    import sunpy.map

    binary_map = sunpy.map.Map(binary.astype(np.float32), patch_map.meta)

    with warnings.catch_warnings():
        # reproject เตือนเรื่องพิกเซลนอกขอบเขตเป็นเรื่องปกติ (patch เล็กกว่าจานมาก)
        warnings.simplefilter("ignore")
        reprojected, _ = reproject_interp(binary_map, target_wcs, shape_out=target_shape)

    # ค่าที่ interpolate แล้ว >= 0.5 ถือว่าอยู่ใน AR; NaN (นอกขอบ patch) = 0
    return (np.nan_to_num(reprojected, nan=0.0) >= 0.5).astype(np.uint8)


def build_fulldisk_mask(
    fulldisk_map,
    patch_maps: list,
    threshold: int = DEFAULT_AR_THRESHOLD,
    target_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """รวม SHARP patch ทุกอันในเฟรมเดียวกันเป็น mask อันเดียว

    Parameters
    ----------
    target_size
        ถ้าระบุ จะสร้าง mask ที่ความละเอียดนี้โดยตรง (แนะนำ — เร็วกว่ามาก ดู
        :func:`reproject_patch_to_grid`) ถ้าเป็น ``None`` จะได้ mask ขนาดเท่า
        ภาพเต็มดวง ซึ่งต้องนำไปย่อเองด้วย ``downsample(..., is_mask=True)``

    Returns
    -------
    ``(mask, n_patches_used)``
    """
    if target_size is None:
        target_wcs, target_shape = fulldisk_map.wcs, fulldisk_map.data.shape
    else:
        target_wcs, target_shape = binned_wcs(fulldisk_map, target_size)

    mask = np.zeros(target_shape, dtype=np.uint8)
    n_used = 0

    for patch_map in patch_maps:
        try:
            patch_mask = reproject_patch_to_grid(patch_map, target_wcs, target_shape, threshold)
        except Exception as exc:  # noqa: BLE001 — patch เดียวพังไม่ควรทำให้ทั้งเฟรมเสีย
            logger.warning(
                "reproject patch HARPNUM=%s ไม่สำเร็จ: %s",
                patch_map.meta.get("harpnum", "?"),
                exc,
            )
            continue

        if patch_mask.any():
            mask |= patch_mask
            n_used += 1

    return mask, n_used


def build_fulldisk_identity_map(
    fulldisk_map,
    patch_maps: list,
    threshold: int = DEFAULT_AR_THRESHOLD,
    target_size: int | None = None,
) -> tuple[np.ndarray, int]:
    """เหมือน :func:`build_fulldisk_mask` แต่คืน**แผนที่ระบุตัวตน**ของ HARP แทน mask
    ไบนารีตรง ๆ — ใช้บอกว่าพิกเซลไหนของ mask เป็นของ active region ดวงไหน

    ``build_fulldisk_mask`` รวม patch ทุกดวงด้วย ``mask |= patch_mask`` ซึ่งโยน
    HARPNUM ทิ้งไปตลอดกาล ฟังก์ชันนี้ทำ reproject ผ่าน :func:`reproject_patch_to_grid`
    ตัวเดียวกันบน ``patch_maps`` ชุดเดียวกัน จึงแปลงกลับเป็น mask ไบนารีได้ด้วย
    ``identity_map > 0`` ซึ่งเท่ากับผลของ ``build_fulldisk_mask`` ทุกพิกเซลเสมอ
    (พิสูจน์ด้วย cross-check test ใน ``test_build_masks.py``) — **ไม่แก้
    ``build_fulldisk_mask`` เลย** เพื่อไม่ให้กระทบ mask ที่ U-Net เทรนไปแล้ว

    กติกาตัดสินพิกเซลที่ patch สองดวงซ้อนกัน: **เรียง patch ตาม HARPNUM จากน้อยไป
    มากก่อนเสมอ แล้วให้ดวงที่ประมวลผลก่อนครองพิกเซลนั้น** (พิกเซลที่ถูกจับจองแล้ว
    จะไม่ถูกเขียนทับ) ทำให้ผลลัพธ์ไม่ขึ้นกับลำดับที่ ``patch_maps`` ถูกส่งเข้ามา

    Returns
    -------
    ``(identity_map, n_patches_used)`` — ``identity_map`` ค่า 0 หมายถึงไม่ใช่ AR
    ดวงใด ค่าอื่นคือหมายเลข HARP (``int32``)
    """
    if target_size is None:
        target_wcs, target_shape = fulldisk_map.wcs, fulldisk_map.data.shape
    else:
        target_wcs, target_shape = binned_wcs(fulldisk_map, target_size)

    identity = np.zeros(target_shape, dtype=np.int32)
    n_used = 0

    for patch_map in sorted(patch_maps, key=_patch_harpnum):
        try:
            patch_mask = reproject_patch_to_grid(patch_map, target_wcs, target_shape, threshold)
        except Exception as exc:  # noqa: BLE001 — patch เดียวพังไม่ควรทำให้ทั้งเฟรมเสีย
            logger.warning(
                "reproject patch HARPNUM=%s ไม่สำเร็จ: %s",
                patch_map.meta.get("harpnum", "?"),
                exc,
            )
            continue

        if patch_mask.any():
            # เติมเฉพาะพิกเซลที่ยังไม่มีเจ้าของ — เรียง patch ตาม HARPNUM มาก่อนแล้ว
            # จึงทำให้ดวงที่ HARPNUM น้อยกว่าชนะพิกเซลที่ซ้อนทับกับดวงอื่น เสมอไม่ว่า
            # patch_maps จะถูกส่งเข้ามาด้วยลำดับใด
            claim = patch_mask.astype(bool) & (identity == 0)
            identity[claim] = _patch_harpnum(patch_map)
            n_used += 1

    return identity, n_used


def _patch_harpnum(patch_map) -> int:
    """อ่าน HARPNUM จาก meta ของ patch — คืน -1 ถ้าไม่มีหรืออ่านไม่ได้ (ไม่ควรเกิดกับ
    ข้อมูลจริงจาก JSOC แต่กันไว้ไม่ให้ทั้งเฟรมพังเพราะ patch เดียว)
    """
    try:
        return int(patch_map.meta.get("harpnum", -1))
    except (TypeError, ValueError):
        return -1


def downsample(
    image: np.ndarray, target_size: int, is_mask: bool = False
) -> np.ndarray:
    """ย่อภาพจาก 4096x4096 เหลือ ``target_size``

    ใช้วิธีต่างกันตามชนิดข้อมูล:

    * **magnetogram** — ``INTER_AREA`` (เฉลี่ยพื้นที่) ซึ่งรักษาค่าฟลักซ์รวมไว้ได้ดี
      และลด aliasing เมื่อย่อขนาดมากๆ
    * **mask** — เฉลี่ยพื้นที่แล้ว threshold ที่ 0.5 เท่ากับ "พิกเซลใหม่เป็น AR ถ้า
      พื้นที่เดิมเกินครึ่งเป็น AR" ซึ่งตรงไปตรงมากว่าการใช้ nearest ที่ทำให้ AR เล็ก
      หายไปแบบสุ่ม
    """
    import cv2

    data = np.asarray(image, dtype=np.float32)
    resized = cv2.resize(data, (target_size, target_size), interpolation=cv2.INTER_AREA)

    if is_mask:
        return (resized >= 0.5).astype(np.uint8)
    return resized


def save_frame(
    out_dir: Path,
    timestamp: str,
    magnetogram: np.ndarray,
    mask: np.ndarray,
    identity_map: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """บันทึกคู่ (ภาพ, mask) ที่ย่อแล้ว พร้อมแผนที่ระบุตัวตนของ HARP ถ้ามี

    magnetogram เก็บเป็น float16 เพื่อประหยัดพื้นที่ — ความละเอียดราว 3 หลักสำคัญ
    ซึ่งเกินพอสำหรับค่าสนามแม่เหล็กที่มีความไม่แน่นอนจากการวัดสูงกว่านั้นมาก
    (512x512 float16 = 512 KB ต่อภาพ เทียบกับ FITS ต้นฉบับ ~30 MB)

    ``identity_map`` เป็น parameter เสริม (ค่าเริ่มต้น ``None``) เพื่อไม่ให้กระทบ
    เฟรมเก่าหรือโค้ดเรียกเดิมที่ยังไม่มีแผนที่นี้ — ดู :func:`build_fulldisk_identity_map`
    """
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    image_path = image_dir / f"{timestamp}.npy"
    mask_path = mask_dir / f"{timestamp}.npy"

    np.save(image_path, magnetogram.astype(np.float16))
    np.save(mask_path, mask.astype(np.uint8))

    if identity_map is not None:
        identity_dir = out_dir / "harp_ids"
        identity_dir.mkdir(parents=True, exist_ok=True)
        np.save(identity_dir / f"{timestamp}.npy", identity_map.astype(np.int32))

    return image_path, mask_path


def mask_statistics(mask: np.ndarray) -> dict[str, float]:
    """สถิติของ mask — ใช้ตรวจจับเฟรมที่ผิดปกติ"""
    total = mask.size
    positive = int(mask.sum())
    return {
        "area_fraction": positive / total,
        "n_positive_px": positive,
    }


def validate_frame(
    magnetogram: np.ndarray, mask: np.ndarray, min_area_frac: float = 1e-5
) -> tuple[bool, str]:
    """ตรวจคู่ (ภาพ, mask) ก่อนบันทึก คืน ``(ผ่านหรือไม่, เหตุผล)``"""
    if magnetogram.shape != mask.shape:
        return False, f"ขนาดไม่ตรงกัน: ภาพ {magnetogram.shape} vs mask {mask.shape}"

    finite_frac = float(np.isfinite(magnetogram).mean())
    if finite_frac < 0.5:
        return False, f"ภาพมีค่า NaN มากเกินไป (มีค่าปกติเพียง {finite_frac:.1%})"

    area = mask.mean()
    if area < min_area_frac:
        return False, f"mask แทบว่างเปล่า (สัดส่วนพื้นที่ {area:.2e})"
    if area > 0.5:
        return False, f"mask ครอบคลุมภาพมากผิดปกติ (สัดส่วนพื้นที่ {area:.1%})"

    return True, "ok"
