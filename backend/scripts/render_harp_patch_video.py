"""ดึง SHARP patch พื้นเมือง (native resolution) ของ HARP หนึ่งดวงตรงจาก JSOC แล้วเรนเดอร์
วิดีโอ segmentation 4 ช่องเรียงแถวนอน (HMI magnetogram + AIA 1600/304/171) คล้ายภาพตัวอย่าง
SHARP quicklook ของ JSOC (NOAA .. / HARP .. .. continuum | B_los | AIA ..)

    python backend/scripts/render_harp_patch_video.py --harp 11149 --start 2024-04-30 --end 2024-05-16

ต้องมี ``SUNSEG_JSOC_EMAIL`` ใน ``.env`` (export segment ต้องเข้าคิว JSOC ~20-60 วินาที)
ดึงข้อมูลทั้งช่วงเวลาเป็น **export เดียว** (ไม่ทีละเฟรมแบบ ``download_images.py``) เพราะ
SHARP รับ HARPNUM เดี่ยวเป็นตัวกรองได้ตรง ๆ ในหนึ่ง recordset จึงไม่ต้องพึ่งภาพเต็มดวง
หรือ SHARP bitmap ของ HARP อื่นเลย

.. important::
   **ต่างจาก** ``render_segmentation_video.py --harp`` **ตรงไหน**: สคริปต์นั้นครอบตัด
   จากภาพเต็มดวงที่ถูกย่อเหลือ 512px ไว้แล้ว (~4 arcsec/px) — เร็วเพราะข้อมูลมีอยู่แล้ว
   แต่หยาบ ส่วนสคริปต์นี้ export segment ``magnetogram``/``bitmap`` ของ
   ``hmi.sharp_720s`` ตรง ๆ ที่ความละเอียดดิบ (~0.5 arcsec/px) จึงเห็นจุดมืดคมชัดกว่า
   มาก ตรงกับภาพตัวอย่าง SHARP quicklook — แลกมาด้วยการดาวน์โหลดใหม่ทั้งหมด (ช้ากว่า)

   ภาพ AIA สามช่องดึงผ่าน**คิว export ของ JSOC โดยตรง** (``aia.lev1_euv_12s`` สำหรับ
   304/171, ``aia.lev1_uv_24s`` สำหรับ 1600 — ความละเอียดดิบเต็ม ไม่ใช่คลัง synoptic ที่ย่อ
   ไว้แล้วเหลือ 2.4 arcsec/px) แล้ว reproject ลง WCS ของ patch นี้โดยตรง จึงคมชัดใกล้เคียง
   กับ magnetogram patch มาก — แลกมาด้วยไฟล์ที่ใหญ่กว่ามาก (ภาพเต็มดวง ~30-60 MB/ไฟล์ x 3
   ช่อง x ทุกเฟรม) และคิว export ที่นานกว่า synoptic ทั่วไป export ทั้งช่วงเวลาเป็น
   **หนึ่งคำขอต่อช่อง** (ไม่ใช่ทีละเฟรม) เหมือน SHARP patch ด้านบน

   **การทำนายของ U-Net**: โมเดลถูกเทรนที่สเกล ~4 arcsec/px (ภาพเต็มดวงย่อ 8 เท่าจาก
   4096->512) ป้อน patch ความละเอียดดิบตรง ๆ จะทำให้ AR ดูใหญ่กว่าที่โมเดลเคยเห็นถึง 8
   เท่าในหน่วยพิกเซล ผลทำนายจะแย่ สคริปต์นี้จึงย่อ patch ลง 8 เท่า (อัตราส่วนเดียวกับตอน
   เทรน) ก่อนป้อนเข้าโมเดล แล้วขยายผลทำนายกลับด้วย nearest-neighbor ทับภาพความละเอียดดิบ
   — เส้นขอบทำนายจะดู "เป็นขั้นบันได" กว่าเส้นขอบ SHARP ground truth (มาจาก bitmap
   ความละเอียดดิบจริง โดยตรง ไม่ผ่านการย่อ) เพราะเหตุนี้ ไม่ใช่บั๊ก

   **ขนาด patch เปลี่ยนทุกเฟรม** (กรอบ SHARP ขยับตามขอบเขต AR ที่วิวัฒนาการจริง) แต่
   วิดีโอต้องมีขนาดเฟรมคงที่ สคริปต์นี้จึงทำสองรอบ: รอบแรกดึง+ประมวลผลทุกเฟรมเก็บไว้ใน
   หน่วยความจำเพื่อหาขนาดใหญ่สุด รอบสองบุ padding สีดำให้ทุกเฟรมมีขนาดเท่ากัน (patch อยู่
   กึ่งกลางเสมอ) แล้วค่อยเข้ารหัสวิดีโอ
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.aia import (  # noqa: E402
    MASK_COLOR_AIA,
    _colormap_for,
    draw_mask_overlay,
    exposure_normalise,
    reproject_to_frame,
    tai_to_utc,
)
from sunseg.data.build_masks import assert_bitmap_encoding, bitmap_to_binary  # noqa: E402
from sunseg.data.jsoc_client import JsocClient, to_drms_time  # noqa: E402
from sunseg.inference.segment import SegmentationService  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.tracking.detect import detect_regions  # noqa: E402

logger = logging.getLogger("render_harp_patch_video")

COLOR_PREDICTED_MAG = (255, 90, 40)  # ส้ม — ผลทำนายของ U-Net บนพื้น magnetogram (RGB)
COLOR_TRUTH = (60, 220, 90)  # เขียว — SHARP bitmap ground truth (RGB)

AIA_LAYERS = {
    "1600": "AIA 1600A (photosphere)",
    "304": "AIA 304A (chromosphere)",
    "171": "AIA 171A (corona)",
}

#: series ของ JSOC ที่ต้อง export ต่อช่อง — UV (1600/1700) กับ EUV (ที่เหลือ) อยู่คนละ series
AIA_SERIES = {"1600": "aia.lev1_uv_24s", "304": "aia.lev1_euv_12s", "171": "aia.lev1_euv_12s"}

#: อัตราส่วนย่อที่ U-Net ถูกเทรนด้วย (ภาพเต็มดวง 4096 -> 512) ใช้ย่อ patch ก่อนป้อนโมเดล
FULLDISK_DOWNSAMPLE = 8

#: ชื่อไฟล์ export ของ JSOC (protocol=fits): <series>.<harpnum>.<T_REC>_TAI.<segment>.fits
_PATCH_FILENAME_RE = re.compile(r"\.(\d{8}_\d{6})_TAI\.(magnetogram|bitmap)\.fits$", re.IGNORECASE)

#: ชื่อไฟล์ export ของ aia.lev1_euv_12s/aia.lev1_uv_24s: <series>.<ISO8601Z>.<wavelength>.image_lev1.fits
#: (วัดจากไฟล์จริง — segment "image" ของสอง series นี้ได้ต่อท้ายด้วย "_lev1" เสมอ)
_AIA_FILENAME_RE = re.compile(r"\.(\d{4}-\d{2}-\d{2}T\d{6}Z)\.(\d+)\.image(?:_lev1)?\.fits$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--harp", type=int, required=True, help="HARPNUM")
    p.add_argument("--start", required=True, help="วันเริ่มต้น YYYY-MM-DD")
    p.add_argument("--end", required=True, help="วันสิ้นสุด YYYY-MM-DD (ไม่รวม)")
    p.add_argument("--cadence-hours", type=int, default=12, help="ความถี่เฟรม (ชั่วโมง)")
    p.add_argument("--out", default=None, help="path ไฟล์วิดีโอผลลัพธ์")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--scale", type=int, default=1, help="ขยายภาพจากขนาดต้นฉบับกี่เท่า (ปกติไม่ต้องขยาย — patch เป็นความละเอียดดิบอยู่แล้ว)")
    p.add_argument("--vmax", type=float, default=1000.0, help="ค่าสนามแม่เหล็กสูงสุดของสเกลสีเทา (G)")
    p.add_argument("--min-area-px", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-fits", action="store_true", help="ไม่ลบ FITS ของ patch หลังประมวลผล")
    p.add_argument("--limit", type=int, default=None, help="จำกัดจำนวนเฟรม (ใช้ตอนทดสอบ)")
    return p.parse_args()


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


# --------------------------------------------------------------------------- #
# ขั้นดึงข้อมูล
# --------------------------------------------------------------------------- #


def fetch_sharp_patches(
    client: JsocClient, series: str, harpnum: int, start: date, end: date,
    cadence_hours: int, out_dir: Path,
) -> dict[str, dict[str, Path]]:
    """export segment magnetogram+bitmap ของ HARP นี้ทั้งช่วงเวลาในคำขอเดียว

    คืน ``{T_REC เช่น "20240501_000000": {"magnetogram": path, "bitmap": path}}``
    """
    recordset = (
        f"{series}[{harpnum}][{to_drms_time(_midnight(start))}-"
        f"{to_drms_time(_midnight(end))}@{cadence_hours}h]"
    )
    logger.info("export SHARP patch (magnetogram+bitmap): %s", recordset)
    client.export_segments(recordset, segments=["magnetogram", "bitmap"], out_dir=out_dir)

    groups: dict[str, dict[str, Path]] = {}
    unmatched = 0
    for f in sorted(out_dir.glob("*.fits")):
        match = _PATCH_FILENAME_RE.search(f.name)
        if not match:
            unmatched += 1
            continue
        stamp, segment = match.group(1), match.group(2).lower()
        groups.setdefault(stamp, {})[segment] = f

    if unmatched:
        logger.warning(
            "มีไฟล์ %d ไฟล์ที่ชื่อไม่ตรงรูปแบบที่คาด (ข้ามไป) — ตรวจ regex ถ้า JSOC เปลี่ยนชื่อไฟล์",
            unmatched,
        )
    complete = {k: v for k, v in groups.items() if {"magnetogram", "bitmap"} <= v.keys()}
    incomplete = len(groups) - len(complete)
    if incomplete:
        logger.warning("มี %d เฟรมที่ export segment มาไม่ครบคู่ — ข้ามไป", incomplete)
    return complete


def fetch_native_aia(
    client: JsocClient, channel_key: str, start: date, end: date, cadence_hours: int, out_dir: Path,
) -> list[tuple[datetime, Path]]:
    """export ภาพเต็มดวง AIA ความละเอียดดิบของช่องนี้ทั้งช่วงเวลาในคำขอเดียว (ไม่ใช่คลัง synoptic)

    คืนรายการ ``(เวลา UTC จากชื่อไฟล์, path)`` เรียงตามเวลา — ไฟล์ใหญ่กว่า synoptic มาก
    (~30-60 MB/ไฟล์) จึงให้ ``timeout``/``read_timeout`` นานกว่าเดิม
    """
    series = AIA_SERIES[channel_key]
    recordset = (
        f"{series}[{to_drms_time(_midnight(start))}-{to_drms_time(_midnight(end))}"
        f"@{cadence_hours}h][{int(channel_key)}]"
    )
    logger.info("export AIA native ช่อง %s: %s", channel_key, recordset)
    client.export_segments(
        recordset, segments=["image"], out_dir=out_dir, timeout_s=1800, read_timeout_s=600
    )

    items: list[tuple[datetime, Path]] = []
    unmatched = 0
    for f in sorted(out_dir.glob("*.fits")):
        match = _AIA_FILENAME_RE.search(f.name)
        if not match:
            unmatched += 1
            continue
        moment = datetime.strptime(match.group(1), "%Y-%m-%dT%H%M%SZ")
        items.append((moment, f))

    if unmatched:
        logger.warning(
            "ช่อง %s: มีไฟล์ %d ไฟล์ที่ชื่อไม่ตรงรูปแบบที่คาด (ข้ามไป)", channel_key, unmatched
        )
    return sorted(items)


def nearest_aia_file(
    items: list[tuple[datetime, Path]], moment_utc: datetime, max_gap_minutes: float = 30.0
) -> Path | None:
    """ไฟล์ AIA ที่ใกล้เวลานี้ที่สุด — ``None`` ถ้าไม่มีไฟล์ไหนใกล้พอ"""
    if not items:
        return None
    best_moment, best_path = min(items, key=lambda it: abs((it[0] - moment_utc).total_seconds()))
    if abs((best_moment - moment_utc).total_seconds()) > max_gap_minutes * 60:
        return None
    return best_path


def reproject_aia_file(path: Path, target_wcs, target_shape: tuple[int, int]) -> np.ndarray:
    """อ่านไฟล์ AIA เต็มดวง แปลง DN/s แล้ว reproject ลง WCS ของ patch นี้โดยตรง"""
    import sunpy.map

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aia_map = sunpy.map.Map(str(path))

    exptime = float(aia_map.meta.get("exptime", 0.0))
    normalised = exposure_normalise(aia_map.data, exptime)
    return reproject_to_frame(sunpy.map.Map(normalised, aia_map.meta), target_wcs, target_shape)


# --------------------------------------------------------------------------- #
# เรนเดอร์ภาพพื้นแต่ละชนิด (เหมือน render_segmentation_video.py)
# --------------------------------------------------------------------------- #


def mag_panel_rgb(magnetogram: np.ndarray, vmax: float) -> np.ndarray:
    raw = np.asarray(magnetogram, dtype=np.float32)
    off_disk = ~np.isfinite(raw)
    scaled = np.clip(np.nan_to_num(raw) / vmax, -1.0, 1.0)
    grey = ((scaled + 1.0) * 127.5).astype(np.uint8)
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    rgb[off_disk] = 0
    return rgb


def _aia_normalized(image: np.ndarray, vmax: float) -> np.ndarray:
    from astropy.visualization import AsinhStretch, ImageNormalize

    array = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0)
    norm = ImageNormalize(vmin=0.0, vmax=float(vmax), stretch=AsinhStretch(0.02), clip=True)
    return norm(array)


def aia_panel_rgb(image: np.ndarray, wavelength: int, vmax: float) -> np.ndarray:
    return (_colormap_for(wavelength)(_aia_normalized(image, vmax))[..., :3] * 255).astype(np.uint8)


def with_overlay(rgb, predicted_mask, truth_mask, detections, predicted_color) -> np.ndarray:
    if truth_mask is not None:
        rgb = draw_mask_overlay(rgb, truth_mask, COLOR_TRUTH, detections=None)
    return draw_mask_overlay(rgb, predicted_mask, predicted_color, detections=detections)


def add_caption(rgb: np.ndarray, text: str) -> np.ndarray:
    rgb = rgb.copy()
    cv2.putText(rgb, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(rgb, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def compose_frame(rgb: np.ndarray, scale: int, label_lines: list[str]) -> np.ndarray:
    if scale > 1:
        size = (rgb.shape[1] * scale, rgb.shape[0] * scale)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST)

    bar_h = 22 * len(label_lines) + 14
    canvas = np.zeros((rgb.shape[0] + bar_h, rgb.shape[1], 3), dtype=np.uint8)
    canvas[: rgb.shape[0]] = rgb
    for i, line in enumerate(label_lines):
        cv2.putText(
            canvas, line, (10, rgb.shape[0] + 20 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def pad_to(array: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """บุขอบด้วย 0 ให้ได้ขนาด ``target_hw`` โดยวางของเดิมไว้กึ่งกลาง"""
    th, tw = target_hw
    h, w = array.shape[:2]
    top, left = (th - h) // 2, (tw - w) // 2
    canvas = np.zeros((th, tw, *array.shape[2:]), dtype=array.dtype)
    canvas[top : top + h, left : left + w, ...] = array
    return canvas


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_data_config()
    start, end = _as_date(args.start), _as_date(args.end)

    try:
        email = cfg.jsoc.email
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    seg = SegmentationService(
        cfg.paths.artifacts / "models" / "unet.pt",
        frames_dir=cfg.paths.processed / "frames",
        device=args.device,
    )
    if not seg.available:
        logger.error(
            "ยังไม่มีโมเดล U-Net ที่ใช้งานได้ (%s) — รัน backend/scripts/train_unet.py ก่อน",
            seg.checkpoint_path,
        )
        return 1

    work_dir = cfg.paths.raw / "harp_patch_tmp"
    patch_dir = work_dir / f"harp{args.harp}"
    aia_dir = work_dir / "aia"
    patch_dir.mkdir(parents=True, exist_ok=True)
    aia_dir.mkdir(parents=True, exist_ok=True)

    client = JsocClient(email=email, max_retries=cfg.jsoc.max_retries, retry_backoff_s=cfg.jsoc.retry_backoff_s)

    # ------------------------------------------------------------------ #
    logger.info("[1/4] ดึง SHARP patch (magnetogram+bitmap) ของ HARP %d", args.harp)
    expected = max(1, int((_midnight(end) - _midnight(start)).total_seconds() // 3600 // args.cadence_hours))
    groups = fetch_sharp_patches(
        client, cfg.jsoc.sharp_bitmap_series, args.harp, start, end, args.cadence_hours, patch_dir
    )
    if len(groups) < 0.5 * expected:
        # เคยเจอจริง: เน็ตสะดุดตอน export ทำให้ได้ group ไม่ครบ (เช่น 8 จาก 31 ที่ควรได้)
        # ทั้งที่ export ตัวมันเองรายงานว่าสำเร็จ — ลองใหม่อีกครั้งก่อนยอมแพ้
        logger.warning(
            "ได้ patch แค่ %d จากที่คาดไว้ราว %d เฟรม — อาจเป็นปัญหาเน็ตชั่วคราวตอน export ลองใหม่อีกครั้ง",
            len(groups), expected,
        )
        groups = fetch_sharp_patches(
            client, cfg.jsoc.sharp_bitmap_series, args.harp, start, end, args.cadence_hours, patch_dir
        )
    if not groups:
        logger.error("ไม่ได้ patch ของ HARP %d เลยในช่วง %s ถึง %s", args.harp, start, end)
        return 1
    stamps = sorted(groups)
    if args.limit:
        stamps = stamps[: args.limit]
    logger.info("พบ %d เฟรม", len(stamps))

    # ------------------------------------------------------------------ #
    logger.info("[2/4] ดึงภาพ AIA native ทั้ง %d ช่อง (%s) — คนละ export ต่อช่อง", len(AIA_LAYERS), ", ".join(AIA_LAYERS))
    aia_native_files: dict[str, list[tuple[datetime, Path]]] = {}
    for key in AIA_LAYERS:
        channel_dir = aia_dir / key
        channel_dir.mkdir(parents=True, exist_ok=True)
        aia_native_files[key] = fetch_native_aia(client, key, start, end, args.cadence_hours, channel_dir)
        if len(aia_native_files[key]) < 0.5 * expected:
            logger.warning(
                "ช่อง %s ได้แค่ %d ไฟล์จากที่คาดไว้ราว %d — อาจเป็นปัญหาเน็ตชั่วคราวตอน export ลองใหม่อีกครั้ง",
                key, len(aia_native_files[key]), expected,
            )
            aia_native_files[key] = fetch_native_aia(client, key, start, end, args.cadence_hours, channel_dir)
        logger.info("  ช่อง %s: ได้ %d ไฟล์", key, len(aia_native_files[key]))
        if not aia_native_files[key]:
            logger.error("ไม่ได้ไฟล์ AIA ช่อง %s เลย — หยุดทำงาน", key)
            return 1

    # ------------------------------------------------------------------ #
    logger.info("[3/4] ประมวลผลทีละเฟรม (magnetogram -> ย่อ 8x -> U-Net -> ขยายกลับ, AIA reproject)")
    import sunpy.map

    bitmap_checked = False
    frames_data: list[dict] = []
    max_h, max_w = 0, 0

    for index, stamp in enumerate(stamps, start=1):
        paths = groups[stamp]
        moment_tai = datetime.strptime(stamp, "%Y%m%d_%H%M%S")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mag_map = sunpy.map.Map(str(paths["magnetogram"]))
                bitmap_map = sunpy.map.Map(str(paths["bitmap"]))

            if not bitmap_checked:
                assert_bitmap_encoding(bitmap_map.data, source=paths["bitmap"].name)
                bitmap_checked = True

            magnetogram = np.asarray(mag_map.data, dtype=np.float32)
            truth_mask = bitmap_to_binary(bitmap_map.data)
            h, w = magnetogram.shape

            small = cv2.resize(
                magnetogram, (max(1, w // FULLDISK_DOWNSAMPLE), max(1, h // FULLDISK_DOWNSAMPLE)),
                interpolation=cv2.INTER_AREA,
            )
            prob_small = seg.segment(small)
            pred_small = (prob_small >= seg.threshold).astype(np.uint8)
            predicted_mask = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

            moment_utc = tai_to_utc(moment_tai)
            aia_images: dict[str, np.ndarray] = {}
            missing = []
            for key in AIA_LAYERS:
                aia_path = nearest_aia_file(aia_native_files[key], moment_utc)
                if aia_path is None:
                    missing.append(key)
                    continue
                aia_images[key] = reproject_aia_file(aia_path, mag_map.wcs, (h, w))
            if missing:
                logger.warning("[%d/%d] %s: ไม่มีภาพ AIA ช่อง %s ใกล้พอ — ข้ามเฟรมนี้", index, len(stamps), stamp, ", ".join(missing))
                continue

            frames_data.append(
                dict(
                    stamp=stamp, moment=moment_tai, magnetogram=magnetogram,
                    truth_mask=truth_mask, predicted_mask=predicted_mask, aia=aia_images,
                )
            )
            max_h, max_w = max(max_h, h), max(max_w, w)
            logger.info(
                "[%d/%d] %s: patch %dx%d, mask ทำนาย %.2f%%, SHARP %.2f%%",
                index, len(stamps), stamp, h, w, 100 * predicted_mask.mean(), 100 * truth_mask.mean(),
            )

        except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
            logger.error("[%d/%d] %s ล้มเหลว: %s", index, len(stamps), stamp, exc)

        finally:
            if not args.keep_fits:
                for p in paths.values():
                    p.unlink(missing_ok=True)

    if not frames_data:
        logger.error("ไม่มีเฟรมใดประมวลผลได้สำเร็จเลย")
        return 1

    # ------------------------------------------------------------------ #
    logger.info("[4/4] บุขนาดให้เท่ากัน (%dx%d) แล้วเรนเดอร์วิดีโอ", max_h, max_w)

    suffix = f"harp{args.harp}_native_"
    out_path = Path(args.out) if args.out else (
        cfg.paths.artifacts / "figures" / f"segmentation_{suffix}{args.start}_{args.end}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_proc: subprocess.Popen | None = None
    first_size = None
    n_written = 0

    for data in frames_data:
        target = (max_h, max_w)
        magnetogram = pad_to(data["magnetogram"], target)
        truth_mask = pad_to(data["truth_mask"], target)
        predicted_mask = pad_to(data["predicted_mask"], target)
        aia_images = {k: pad_to(v, target) for k, v in data["aia"].items()}

        # หมุน 180° แก้ทิศ CCD ดิบเหมือนภาพเต็มดวง (ดู docstring บนสุดของ
        # render_segmentation_video.py) — patch มาจากเซนเซอร์ตัวเดียวกัน ทิศเดียวกัน
        magnetogram = np.rot90(magnetogram, 2)
        truth_mask = np.rot90(truth_mask, 2)
        predicted_mask = np.rot90(predicted_mask, 2)
        aia_images = {k: np.rot90(v, 2) for k, v in aia_images.items()}

        detections = detect_regions(predicted_mask, magnetogram=magnetogram, min_area_px=args.min_area_px)
        pretty = data["moment"].strftime("%Y-%m-%d %H:%M UT (TAI)")

        mag_base = with_overlay(
            mag_panel_rgb(magnetogram, args.vmax), predicted_mask, truth_mask, detections, COLOR_PREDICTED_MAG
        )
        panels = [add_caption(mag_base, "HMI magnetogram")]
        for key in ("1600", "304", "171"):
            channel = cfg.aia.channel(key)
            panel = aia_panel_rgb(aia_images[key], channel.wavelength, channel.display_vmax)
            panel = with_overlay(panel, predicted_mask, truth_mask, detections, MASK_COLOR_AIA)
            panels.append(add_caption(panel, AIA_LAYERS[key]))

        base = np.concatenate(panels, axis=1)
        label_lines = [
            f"HARP {args.harp}   {pretty}   U-Net prediction (orange/cyan) vs SHARP ground truth (green)",
            f"AR regions: {len(detections)}   patch native: {data['magnetogram'].shape[1]}x{data['magnetogram'].shape[0]}px",
        ]
        frame = compose_frame(base, args.scale, label_lines)

        if ffmpeg_proc is None:
            first_size = (frame.shape[1], frame.shape[0])
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
                    "-video_size", f"{first_size[0]}x{first_size[1]}", "-framerate", str(args.fps),
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out_path),
                ],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )

        ffmpeg_proc.stdin.write(frame.tobytes())
        n_written += 1

    if not args.keep_fits:
        shutil.rmtree(work_dir, ignore_errors=True)

    if ffmpeg_proc is None:
        logger.error("ไม่มีเฟรมใดเรนเดอร์ได้สำเร็จเลย")
        return 1

    ffmpeg_proc.stdin.close()
    stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        logger.error("ffmpeg ล้มเหลว (exit %d):\n%s", ffmpeg_proc.returncode, stderr[-2000:])
        return 1

    logger.info("-" * 62)
    logger.info("เขียนวิดีโอ %d เฟรม (%d fps, %dx%d) ไปที่ %s", n_written, args.fps, *first_size, out_path)
    logger.info("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
