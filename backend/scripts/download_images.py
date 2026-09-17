"""ดาวน์โหลดภาพ HMI แล้วสร้างคู่ (magnetogram, mask) สำหรับเทรน U-Net

    # ทดสอบด้วยไม่กี่เฟรมก่อนเสมอ
    python backend/scripts/download_images.py --start 2014-10-20 --end 2014-10-28 --limit 5

    # ดึงเต็มช่วงตาม configs/data.yaml (ใช้เวลาหลายชั่วโมง)
    python backend/scripts/download_images.py

**กลยุทธ์ประหยัดพื้นที่**: ดิสก์เหลือน้อย จึงทำงานแบบ streaming — ดาวน์โหลด FITS
ทีละเฟรม, ย่อเหลือ 512x512, บันทึกเป็น ``.npy`` (float16), แล้ว **ลบ FITS ทันที**
ภาพเต็มดวงต้นฉบับมีขนาด ~30 MB ต่อไฟล์ ส่วนที่เก็บจริงเหลือเพียง ~512 KB

**ประมวลผลหลายเฟรมพร้อมกัน** (``--workers``, ปริยาย 4): เวลาเกือบทั้งหมดต่อเฟรมหมดไป
กับการรอ JSOC ตอบ (export_fast + keyword query ~2-20 วิ/ครั้ง เครือข่ายล้วนๆ ไม่ใช้ CPU)
ซึ่งปล่อย GIL อยู่แล้วระหว่างรอ ทำให้เฟรมที่ต่างกันประมวลผลพร้อมกันได้จริงโดยไม่ต้องล็อก —
แต่ละเฟรมเขียนคนละไดเรกทอรี/ไฟล์เสมอ ("fits_tmp/<tag>", "images/<tag>.npy",
"harp_ids/<tag>.npy") จึงไม่มีจุดชนกัน ยกเว้นการตรวจ bitmap encoding ครั้งแรกที่ล็อกไว้
เพราะเป็นแค่ log ตรวจสอบครั้งเดียว ไม่ใช่ correctness

รันซ้ำได้ปลอดภัย — เฟรมที่ประมวลผลแล้วจะถูกข้าม
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from sunseg.data.frame_wcs import (  # noqa: E402
    fetch_frame_wcs,
    fetch_sharp_wcs,
    map_from_as_is,
)
from sunseg.data.jsoc_client import JsocClient, to_drms_time  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("download_images")


class FrameSkipped(Exception):
    """เฟรมนี้ไม่มีข้อมูลพอจะประมวลผลได้ (ไม่ใช่ error รุนแรง) — log เป็น WARNING ไม่ใช่ ERROR"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", type=_as_date, help="วันเริ่มต้น YYYY-MM-DD")
    p.add_argument("--end", type=_as_date, help="วันสิ้นสุด YYYY-MM-DD")
    p.add_argument("--limit", type=int, help="จำกัดจำนวนเฟรม (ใช้ตอนทดสอบ)")
    p.add_argument(
        "--cadence-hours", type=int, help="ทับค่า cadence ของภาพเต็มดวง (ชั่วโมง)"
    )
    p.add_argument(
        "--keep-fits", action="store_true", help="ไม่ลบไฟล์ FITS หลังประมวลผล (กินพื้นที่มาก)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="จำนวนเฟรมที่ประมวลผลพร้อมกัน (ปริยาย 4 — คอขวดคือการรอ JSOC ตอบ)",
    )
    return p.parse_args()


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def process_frame(
    client: JsocClient,
    cfg,
    moment: datetime,
    work_dir: Path,
    keep_fits: bool,
    bitmap_check_lock: threading.Lock,
    bitmap_checked: list[bool],
) -> tuple[int, float]:
    """ประมวลผลเฟรมเดียวให้ครบ (ดาวน์โหลด -> mask -> ย่อ -> เซฟ) คืน (n_patches, mask_coverage_pct)

    โยน :class:`FrameSkipped` เมื่อข้อมูลไม่พอ (ไม่ใช่ error รุนแรง) หรือ ``Exception``
    อื่นเมื่อพังจริง — ผู้เรียกแยก log level จากชนิด exception
    """
    tag = moment.strftime("%Y%m%d_%H%M%S")
    frame_dir = work_dir / tag
    drms_time = to_drms_time(moment)

    try:
        # --- ภาพเต็มดวง ---
        # export_fast ใช้ url_quick/as-is (ไม่เข้าคิว export) — pixel data เหมือน
        # export_segments()/protocol=fits ทุกประการ แค่ header เปล่า จึงต้องต่อ WCS
        # เองจาก keyword query แยกต่างหาก (ดู frame_wcs.py)
        client.export_fast(
            f"{cfg.jsoc.fulldisk_series}[{drms_time}]",
            segments=["magnetogram"],
            out_dir=frame_dir,
        )
        fits_files = sorted(frame_dir.glob("*.fits"))
        if not fits_files:
            raise RuntimeError("ไม่ได้ไฟล์ FITS ของภาพเต็มดวง")

        fulldisk_wcs = fetch_frame_wcs(client, cfg.jsoc.fulldisk_series, [tag])
        if fulldisk_wcs.empty:
            raise RuntimeError("ดึง WCS ของภาพเต็มดวงไม่สำเร็จ")
        fulldisk_map = map_from_as_is(fits_files[0], fulldisk_wcs.iloc[0])

        # --- SHARP bitmap ทุก patch ที่เวลาเดียวกัน ---
        patch_dir = frame_dir / "patches"
        patch_result = client.export_fast(
            f"{cfg.jsoc.sharp_bitmap_series}[][{drms_time}]",
            segments=["bitmap"],
            out_dir=patch_dir,
        )
        patch_files = sorted(patch_dir.glob("*.fits"))
        if not patch_files:
            raise FrameSkipped("ไม่มี SHARP patch")

        sharp_wcs = fetch_sharp_wcs(client, cfg.jsoc.sharp_bitmap_series, moment)
        if sharp_wcs.empty:
            raise FrameSkipped("ดึง WCS ของ SHARP patch ไม่สำเร็จ")
        # drop=False: ต้องเก็บคอลัมน์ HARPNUM ไว้ด้วย ไม่ใช่แค่ทำเป็น index —
        # map_from_as_is อ่าน HARPNUM จาก row["HARPNUM"] เพื่อใส่ลง header ต่อ
        sharp_wcs = sharp_wcs.set_index("HARPNUM", drop=False)

        # จับคู่ไฟล์ที่ export_fast ดาวน์โหลดมากับแถว keyword ด้วย HARPNUM ที่อ่าน
        # จากคอลัมน์ record ("hmi.sharp_720s[812][...]") — จับคู่ด้วยชื่อไฟล์ล้วน
        # (ไม่ใช่ Path เต็ม) เพราะ row.download กับผลจาก glob() อาจไม่ผ่านการ
        # normalize มาเหมือนกัน (relative vs absolute) แต่ชื่อไฟล์ไม่ซ้ำกันแน่อยู่
        # แล้วภายในโฟลเดอร์เดียว
        # row.download เป็น None ได้ถ้าไฟล์นั้นดาวน์โหลดไม่สำเร็จ (เช่น เน็ตสะดุด
        # กลางทางแค่บาง segment) — ข้ามแถวนั้นแทนที่จะพังทั้งเฟรม (พบจริง: Path(None)
        # โยน TypeError)
        harpnum_by_filename = {
            Path(row.download).name: int(re.search(r"\[(\d+)\]", row.record).group(1))
            for row in patch_result.itertuples()
            if row.download is not None
        }

        patch_maps = []
        for patch_file in patch_files:
            harpnum = harpnum_by_filename.get(patch_file.name)
            if harpnum is None or harpnum not in sharp_wcs.index:
                logger.debug("%s: ไม่พบ keyword ของ HARPNUM=%s — ข้าม patch นี้", tag, harpnum)
                continue
            patch_maps.append(map_from_as_is(patch_file, sharp_wcs.loc[harpnum]))

        if not patch_maps:
            raise FrameSkipped("ไม่มี SHARP patch ที่ต่อ WCS ได้เลย")

        # ตรวจการเข้ารหัส bitmap กับข้อมูลจริงครั้งแรกเท่านั้น (ล็อกเพราะหลายเฟรม
        # อาจมาถึงจุดนี้พร้อมกัน — เป็นแค่ log ตรวจสอบครั้งเดียว ไม่ใช่ correctness)
        with bitmap_check_lock:
            if not bitmap_checked[0]:
                assert_bitmap_encoding(
                    patch_maps[0].data, source=f"HARPNUM={patch_maps[0].meta.get('harpnum')}"
                )
                bitmap_checked[0] = True

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
            raise FrameSkipped(f"ไม่ผ่านการตรวจสอบ ({reason})")

        save_frame(cfg.paths.processed / "frames", tag, image_small, mask_small, identity_map=identity_small)
        return n_patches, 100 * float(mask_small.mean())

    finally:
        if not keep_fits:
            shutil.rmtree(frame_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "download_images.log")

    start = args.start or cfg.time_range.start
    end = args.end or cfg.time_range.end
    cadence_hours = args.cadence_hours or cfg.fulldisk.cadence_hours

    frames_dir = cfg.paths.processed / "frames"
    work_dir = cfg.paths.raw / "fits_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(
        "ดาวน์โหลดภาพและสร้าง mask: %s ถึง %s (cadence %d ชม.)",
        start, end, cadence_hours,
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

    # "เสร็จแล้ว" ต้องมีทั้งภาพ**และ**แผนที่ระบุตัวตน ไม่ใช่แค่ภาพ — เฟรมจากยุคก่อนที่
    # save_frame() จะรองรับ identity_map (เป็น parameter เสริมมาตั้งแต่แรกเพื่อไม่ให้
    # กระทบโค้ดเรียกเดิม) มีแต่ภาพกับ mask ไม่มี harp_ids ถ้าเช็คแค่ images/ เฟรมพวกนี้
    # จะถูกข้ามไปตลอดกาลทั้งที่ไม่มีแผนที่ระบุตัวตนจริง — ต้องดาวน์โหลดใหม่เท่านั้น เพราะ
    # FITS ต้นฉบับถูกลบไปแล้วหลังประมวลผลครั้งก่อน สร้างย้อนหลังจากเฟรมที่มีอยู่ไม่ได้
    existing_images = {p.stem for p in (frames_dir / "images").glob("*.npy")}
    existing_identity = {p.stem for p in (frames_dir / "harp_ids").glob("*.npy")}
    existing = existing_images & existing_identity
    n_incomplete = len(existing_images - existing_identity)
    if n_incomplete:
        logger.info(
            "พบเฟรมเก่าที่มีภาพแต่ไม่มีแผนที่ระบุตัวตน %d เฟรม — จะดาวน์โหลดใหม่ให้ครบ",
            n_incomplete,
        )
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
    workers = max(1, args.workers)
    logger.info("[2/2] ประมวลผล %d เฟรม (workers=%d)", len(pending), workers)
    logger.info("      (ดาวน์โหลด -> สร้าง mask -> ย่อเหลือ %dpx -> ลบ FITS)",
                cfg.fulldisk.target_size)

    n_done = 0
    n_failed = 0
    bitmap_check_lock = threading.Lock()
    bitmap_checked = [False]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_frame, client, cfg, moment, work_dir, args.keep_fits,
                bitmap_check_lock, bitmap_checked,
            ): moment
            for _trec_raw, moment in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            moment = futures[future]
            tag = moment.strftime("%Y%m%d_%H%M%S")
            try:
                n_patches, coverage_pct = future.result()
                n_done += 1
                logger.info(
                    "[%d/%d] %s: ใช้ %d patch, mask ครอบคลุม %.2f%%",
                    index, len(pending), tag, n_patches, coverage_pct,
                )
            except FrameSkipped as exc:
                n_failed += 1
                logger.warning("[%d/%d] %s: %s", index, len(pending), tag, exc)
            except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
                n_failed += 1
                logger.error("[%d/%d] %s ล้มเหลว: %s", index, len(pending), tag, exc)

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
