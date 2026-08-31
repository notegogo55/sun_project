"""ดาวน์โหลดภาพ HMI แล้วสร้างคู่ (magnetogram, mask) สำหรับเทรน U-Net

    # ทดสอบด้วยไม่กี่เฟรมก่อนเสมอ
    python backend/scripts/download_images.py --start 2014-10-20 --end 2014-10-28 --limit 5

    # ดึงเต็มช่วงตาม configs/data.yaml (ใช้เวลาหลายชั่วโมง)
    python backend/scripts/download_images.py

    # ดึงหน้าต่าง case study (2024-05 @ 12 ชม. = 62 เฟรม) เป็นชุดทดสอบ out-of-sample
    python backend/scripts/download_images.py --case-study

**กลยุทธ์ประหยัดพื้นที่**: ดิสก์เหลือน้อย จึงทำงานแบบ streaming — ดาวน์โหลด FITS
ทีละเฟรม, ย่อเหลือ 512x512, บันทึกเป็น ``.npy`` (float16), แล้ว **ลบ FITS ทันที**
ภาพเต็มดวงต้นฉบับมีขนาด ~30 MB ต่อไฟล์ ส่วนที่เก็บจริงเหลือเพียง ~512 KB

รันซ้ำได้ปลอดภัย — เฟรมที่ประมวลผลแล้วจะถูกข้าม
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.build_masks import (  # noqa: E402
    assert_bitmap_encoding,
    build_fulldisk_identity_map,
    downsample,
    save_frame,
    validate_frame,
)
from sunseg.data.jsoc_client import JsocClient, to_drms_time  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("download_images")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", type=_as_date, help="วันเริ่มต้น YYYY-MM-DD")
    p.add_argument("--end", type=_as_date, help="วันสิ้นสุด YYYY-MM-DD")
    p.add_argument("--limit", type=int, help="จำกัดจำนวนเฟรม (ใช้ตอนทดสอบ)")
    p.add_argument(
        "--case-study",
        action="store_true",
        help="ใช้หน้าต่าง case_study จาก config แทน time_range (ชุดทดสอบ out-of-sample)",
    )
    p.add_argument(
        "--cadence-hours", type=int, help="ทับค่า cadence ของภาพเต็มดวง (ชั่วโมง)"
    )
    p.add_argument(
        "--keep-fits", action="store_true", help="ไม่ลบไฟล์ FITS หลังประมวลผล (กินพื้นที่มาก)"
    )
    return p.parse_args()


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "download_images.log")

    # --case-study สลับไปใช้หน้าต่างประเมิน ซึ่งมี cadence ถี่กว่าชุดเทรนมาก
    window = cfg.case_study if args.case_study else cfg.time_range
    default_cadence = (
        cfg.case_study.cadence_hours if args.case_study else cfg.fulldisk.cadence_hours
    )
    start = args.start or window.start
    end = args.end or window.end
    cadence_hours = args.cadence_hours or default_cadence

    frames_dir = cfg.paths.processed / "frames"
    work_dir = cfg.paths.raw / "fits_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        "ดาวน์โหลดภาพและสร้าง mask: %s ถึง %s (cadence %d ชม.)%s",
        start,
        end,
        cadence_hours,
        "  [CASE STUDY]" if args.case_study else "",
    )
    logger.info("=" * 70)

    try:
        email = cfg.jsoc.email
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    client = JsocClient(
        email=email,
        max_retries=cfg.jsoc.max_retries,
        retry_backoff_s=cfg.jsoc.retry_backoff_s,
    )

    # ------------------------------------------------------------------ #
    logger.info("[1/2] ค้นหาเวลาของภาพเต็มดวงที่มีอยู่")
    times = client.query_fulldisk_times(
        series=cfg.jsoc.fulldisk_series,
        start=start,
        end=end,
        cadence_hours=cadence_hours,
    )
    if times.empty:
        logger.error("ไม่พบภาพเต็มดวงในช่วงเวลานี้")
        return 1

    times["t_rec"] = client.parse_trec(times["T_REC"])
    times = times[times["QUALITY"].astype(str).str.strip().isin({"0", "0x00000000"})]
    logger.info("พบภาพคุณภาพดี %d เฟรม", len(times))

    existing = {p.stem for p in (frames_dir / "images").glob("*.npy")}
    pending = [
        (row.T_REC, row.t_rec)
        for row in times.itertuples()
        if row.t_rec.strftime("%Y%m%d_%H%M%S") not in existing
    ]
    n_skipped = len(times) - len(pending)
    if n_skipped:
        logger.info(
            "ข้ามเฟรมที่ประมวลผลแล้ว %d เฟรม (มีบนดิสก์ทั้งหมด %d)", n_skipped, len(existing)
        )
    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        logger.info("ไม่มีเฟรมใหม่ที่ต้องประมวลผล")
        return 0

    # ------------------------------------------------------------------ #
    logger.info("[2/2] ประมวลผล %d เฟรม", len(pending))
    logger.info("      (ดาวน์โหลด -> สร้าง mask -> ย่อเหลือ %dpx -> ลบ FITS)",
                cfg.fulldisk.target_size)

    n_done = 0
    n_failed = 0
    bitmap_checked = False

    for index, (_trec_raw, moment) in enumerate(pending, start=1):
        tag = moment.strftime("%Y%m%d_%H%M%S")
        frame_dir = work_dir / tag
        drms_time = to_drms_time(moment)

        try:
            import sunpy.map

            # --- ภาพเต็มดวง ---
            client.export_segments(
                f"{cfg.jsoc.fulldisk_series}[{drms_time}]",
                segments=["magnetogram"],
                out_dir=frame_dir,
            )
            fits_files = sorted(frame_dir.glob("*.fits"))
            if not fits_files:
                raise RuntimeError("ไม่ได้ไฟล์ FITS ของภาพเต็มดวง")
            fulldisk_map = sunpy.map.Map(fits_files[0])

            # --- SHARP bitmap ทุก patch ที่เวลาเดียวกัน ---
            patch_dir = frame_dir / "patches"
            client.export_segments(
                f"{cfg.jsoc.sharp_bitmap_series}[][{drms_time}]",
                segments=["bitmap"],
                out_dir=patch_dir,
            )
            patch_files = sorted(patch_dir.glob("*.fits"))
            if not patch_files:
                logger.warning("[%d/%d] %s: ไม่มี SHARP patch — ข้าม", index, len(pending), tag)
                n_failed += 1
                continue

            patch_maps = [sunpy.map.Map(p) for p in patch_files]

            # ตรวจการเข้ารหัส bitmap กับข้อมูลจริงครั้งแรกเท่านั้น
            if not bitmap_checked:
                assert_bitmap_encoding(patch_maps[0].data, source=patch_files[0].name)
                bitmap_checked = True

            # สร้างแผนที่ระบุตัวตนของ HARP ที่ความละเอียดปลายทางโดยตรง — ถูกกว่าการ
            # reproject ลงกริด 4096 แล้วค่อยย่อ 56 เท่า (ดู reproject_patch_to_grid)
            # mask ไบนารีที่ใช้เทรน U-Net ได้จาก identity_small > 0 ตรงๆ ซึ่งเท่ากับ
            # ผลของ build_fulldisk_mask() เดิมทุกพิกเซล (ดู build_fulldisk_identity_map)
            size = cfg.fulldisk.target_size
            identity_small, n_patches = build_fulldisk_identity_map(
                fulldisk_map,
                patch_maps,
                threshold=cfg.fulldisk.bitmap_ar_value,
                target_size=size,
            )
            mask_small = (identity_small > 0).astype(np.uint8)

            # --- ย่อภาพและตรวจสอบ ---
            image_small = downsample(fulldisk_map.data, size, is_mask=False)

            ok, reason = validate_frame(image_small, mask_small)
            if not ok:
                logger.warning("[%d/%d] %s: ไม่ผ่านการตรวจสอบ (%s)", index, len(pending), tag, reason)
                n_failed += 1
                continue

            save_frame(frames_dir, tag, image_small, mask_small, identity_map=identity_small)
            n_done += 1
            logger.info(
                "[%d/%d] %s: ใช้ %d patch, mask ครอบคลุม %.2f%%",
                index, len(pending), tag, n_patches, 100 * mask_small.mean(),
            )

        except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
            logger.error("[%d/%d] %s ล้มเหลว: %s", index, len(pending), tag, exc)
            n_failed += 1

        finally:
            if not args.keep_fits:
                shutil.rmtree(frame_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    if not args.keep_fits:
        shutil.rmtree(work_dir, ignore_errors=True)

    total = len(list((frames_dir / "images").glob("*.npy")))
    size_mb = sum(p.stat().st_size for p in frames_dir.rglob("*.npy")) / 1024**2

    logger.info("=" * 70)
    logger.info("สำเร็จ %d เฟรม, ล้มเหลว %d เฟรม", n_done, n_failed)
    logger.info("รวมทั้งหมดบนดิสก์: %d เฟรม (%.1f MB)", total, size_mb)
    logger.info("ขั้นถัดไป:")
    logger.info("  1) ตรวจด้วยตา:  python backend/scripts/plot_masks.py")
    logger.info("  2) เทรนโมเดล:   python backend/scripts/train_unet.py")
    logger.info("=" * 70)
    return 0 if n_done else 1


if __name__ == "__main__":
    raise SystemExit(main())
