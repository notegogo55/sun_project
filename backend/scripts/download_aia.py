"""ดาวน์โหลดภาพ AIA สามชั้นบรรยากาศ แล้ววางลงกริดเดียวกับเฟรม HMI ที่มีอยู่

    # 0) ดึง WCS ของเฟรมก่อน (เร็ว ไม่ต้องใช้อีเมล JSOC) — จำเป็นก่อนขั้นอื่นเสมอ
    python backend/scripts/download_aia.py --wcs-only

    # 1) ทดสอบด้วยไม่กี่เฟรมก่อนเสมอ
    python backend/scripts/download_aia.py --start 2014-10-20 --end 2014-10-28 --limit 5

    # 2) ตรวจการจัดตำแหน่งด้วยตา << ห้ามข้าม
    python backend/scripts/plot_aia_alignment.py

    # 3) ดึงเต็มช่วง (~1.2 GB, ราว 45 นาทีด้วย 6 worker)
    python backend/scripts/download_aia.py

**ดึงเฉพาะเฟรมที่มีอยู่แล้ว** — สคริปต์นี้ไม่สร้างเฟรมใหม่ มันเดินตามรายการ
``data/processed/frames/images/*.npy`` ที่ ``download_images.py`` สร้างไว้ แล้วเติมภาพ
AIA ของเวลาเดียวกันเข้าไป

**แหล่งข้อมูล**: คลัง synoptic ของ JSOC ผ่าน HTTP ธรรมดา ไม่ต้องใช้อีเมลที่ลงทะเบียน
และไม่ต้องเข้าคิว export (ดูเหตุผลเต็มใน ``sunseg/data/aia.py``)

ถ้าคลัง synoptic ใช้ไม่ได้ ยังดึงจากคิว export ของ JSOC ได้ผ่าน ``JsocClient.export_segments``
โดยใช้ ``aia.lev1_euv_12s`` (171/304) กับ ``aia.lev1_uv_24s`` (1600) segment ``image``
และ ``protocol="fits"`` — ส่วน reproject ใช้โค้ดเดิมได้เลยเพราะ ``reproject_interp`` อ่านมุมหมุน
จาก WCS ให้เอง แต่ **ยังไม่ได้เขียนไว้ในสคริปต์นี้** เพราะช้ากว่ามาก (~40 GB และหลายสิบชั่วโมง
เทียบกับ ~1.2 GB และไม่ถึงชั่วโมง) จึงยังไม่คุ้มที่จะดูแลโค้ดสองเส้นทาง

รันซ้ำได้ปลอดภัย — ภาพที่ประมวลผลแล้วจะถูกข้าม
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.aia import (  # noqa: E402
    AiaFrameStore,
    download_fits,
    exposure_normalise,
    find_nearest_url,
    read_synoptic,
    reproject_to_frame,
    tai_to_utc,
)
from sunseg.data.frame_wcs import (  # noqa: E402
    FrameWcsStore,
    fetch_frame_wcs,
    merge_and_save,
)
from sunseg.data.jsoc_client import JsocClient  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("download_aia")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", type=_as_date, help="วันเริ่มต้น YYYY-MM-DD")
    p.add_argument("--end", type=_as_date, help="วันสิ้นสุด YYYY-MM-DD")
    p.add_argument("--limit", type=int, help="จำกัดจำนวนเฟรม (ใช้ตอนทดสอบ)")
    p.add_argument(
        "--channels",
        help="ระบุเฉพาะบางช่อง คั่นด้วยจุลภาค เช่น 171,304 (ค่าปริยาย: ทุกช่องใน config)",
    )
    p.add_argument(
        "--wcs-only",
        action="store_true",
        help="ดึงเฉพาะ WCS ของเฟรม (เร็ว ไม่โหลดภาพ) — ต้องรันก่อนดึงภาพเสมอ",
    )
    p.add_argument(
        "--refresh-wcs",
        action="store_true",
        help="ดึง WCS ใหม่ทับของเดิม (ปกติจะดึงเฉพาะเฟรมที่ยังไม่มี)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="จำนวนเฟรมที่ประมวลผลพร้อมกัน (ค่าปริยาย 6 — คอขวดคือการรอ network)",
    )
    return p.parse_args()


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def _frame_moment(stem: str) -> datetime:
    return datetime.strptime(stem, "%Y%m%d_%H%M%S")


def select_frames(
    frames_dir: Path, start: date, end: date, limit: int | None
) -> list[str]:
    """เฟรมที่มีอยู่บนดิสก์ในช่วงเวลาที่เลือก"""
    image_dir = frames_dir / "images"
    if not image_dir.is_dir():
        return []

    stems = sorted(p.stem for p in image_dir.glob("*.npy"))
    selected = [s for s in stems if start <= _frame_moment(s).date() <= end]
    return selected[:limit] if limit else selected


def ensure_wcs(cfg, wcs_path: Path, stems: list[str], refresh: bool) -> FrameWcsStore:
    """ดึง WCS ของเฟรมที่ยังไม่มีในดัชนี แล้วคืน store ที่โหลดใหม่แล้ว"""
    store = FrameWcsStore(wcs_path)
    missing = stems if refresh else [s for s in stems if s not in store]

    if not missing:
        logger.info("WCS ครบทุกเฟรมแล้ว (%d เฟรม)", len(stems))
        return store

    logger.info("ดึง WCS ของ %d เฟรมที่ยังไม่มี (จากทั้งหมด %d)", len(missing), len(stems))
    client = JsocClient(
        max_retries=cfg.jsoc.max_retries, retry_backoff_s=cfg.jsoc.retry_backoff_s
    )
    fresh = fetch_frame_wcs(client, cfg.jsoc.fulldisk_series, missing)

    if fresh.empty:
        logger.error("ดึง WCS ไม่ได้เลยสักเฟรม — ตรวจการเชื่อมต่อไปยัง JSOC")
        return store

    merge_and_save(wcs_path, fresh)
    return FrameWcsStore(wcs_path)


def process_frame(
    cfg,
    store: AiaFrameStore,
    wcs_store: FrameWcsStore,
    stem: str,
    channels: list,
    work_dir: Path,
) -> tuple[int, dict[str, float]]:
    """ดึงและประมวลผลทุกช่องของเฟรมเดียว คืน ``(จำนวนที่สำเร็จ, p99.9 ต่อช่อง)``"""
    target_wcs = wcs_store.binned_wcs(stem, cfg.aia.target_size)
    if target_wcs is None:
        logger.warning("%s: ไม่มี WCS — ข้าม (รัน --wcs-only ก่อน)", stem)
        return 0, {}

    target_shape = (cfg.aia.target_size, cfg.aia.target_size)
    moment_utc = tai_to_utc(_frame_moment(stem))

    n_done = 0
    peaks: dict[str, float] = {}

    for channel in channels:
        if store.path_for(stem, channel.key).exists():
            n_done += 1
            continue

        try:
            found = find_nearest_url(cfg.aia.synoptic_base_url, moment_utc, channel.key)
            if found is None:
                logger.warning("%s ช่อง %s: ไม่พบไฟล์ในคลัง", stem, channel.key)
                continue

            stamp, url = found
            offset_s = abs((stamp - moment_utc).total_seconds())
            # ชื่อไฟล์ชั่วคราวต้องผูกกับ (stem, channel) เอง ไม่ใช่แค่ชื่อจาก URL — สอง
            # เฟรมที่ห่างกันแต่ URL ที่ใกล้ที่สุดชี้ไปไฟล์เดียวกันได้ (คลัง synoptic มี
            # cadence ต่ำกว่าที่ frame ต้องการ) ถ้าใช้ชื่อจาก URL ตรงๆ เธรดที่รันพร้อมกัน
            # (--workers > 1) จะแย่งเขียน/ลบไฟล์เดียวกัน
            tmp_path = work_dir / f"{stem}_{channel.key}_{Path(url).name}"

            # ไฟล์ FITS ที่เพิ่งเขียนเสร็จบน Windows บางทีถูกโปรแกรมสแกนไฟล์ (เช่น
            # antivirus) ล็อกไว้ชั่วครู่ก่อนอ่านได้ ทำให้เจอ WinError 32 "ใช้งานโดย
            # โปรเซสอื่น" หรืออ่านได้ไฟล์ไม่ครบ (Errno 22) เป็นครั้งคราว — ปัญหาชั่วคราว
            # ระดับ OS ไม่ใช่ข้อมูลเสีย จึง retry สั้นๆ พอให้ล็อกปล่อยแทนที่จะเสียทั้ง
            # ช่องนั้นไปฟรีๆ (พบจริงจากการรันด้วย --workers 3 บน Windows)
            last_exc: Exception | None = None
            for attempt in range(2):
                try:
                    fits_path = download_fits(url, tmp_path)
                    try:
                        aia_map = read_synoptic(fits_path)
                        exptime = float(aia_map.meta.get("exptime", 0.0))
                        normalised = exposure_normalise(aia_map.data, exptime)

                        import sunpy.map

                        grid = reproject_to_frame(
                            sunpy.map.Map(normalised, aia_map.meta), target_wcs, target_shape
                        )
                    finally:
                        fits_path.unlink(missing_ok=True)
                    last_exc = None
                    break
                except OSError as exc:
                    last_exc = exc
                    if attempt == 0:
                        time.sleep(2.0)
            if last_exc is not None:
                raise last_exc

            finite = grid[np.isfinite(grid)]
            if finite.size == 0:
                logger.warning("%s ช่อง %s: ภาพว่างเปล่าหลัง reproject", stem, channel.key)
                continue

            store.save(stem, channel.key, grid)
            peaks[channel.key] = float(np.percentile(finite, 99.9))
            n_done += 1

            logger.debug(
                "%s ช่อง %s: dt=%.0fs exp=%.3fs median=%.1f DN/s",
                stem, channel.key, offset_s, exptime, float(np.median(finite)),
            )

        except Exception as exc:  # noqa: BLE001 — ช่องเดียวพังไม่ควรหยุดทั้งงาน
            logger.error("%s ช่อง %s ล้มเหลว: %s", stem, channel.key, exc)

    return n_done, peaks


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "download_aia.log")

    start = args.start or cfg.time_range.start
    end = args.end or cfg.time_range.end

    frames_dir = cfg.paths.processed / "frames"
    wcs_path = cfg.paths.interim / "frame_wcs.parquet"
    work_dir = cfg.paths.raw / "aia_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ดาวน์โหลดภาพ AIA: %s ถึง %s", start, end)
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    stems = select_frames(frames_dir, start, end, args.limit)
    if not stems:
        logger.error(
            "ไม่พบเฟรมในช่วงเวลานี้ที่ %s — รัน backend/scripts/download_images.py ก่อน",
            frames_dir / "images",
        )
        return 1
    logger.info("พบเฟรมที่ต้องเติมภาพ AIA %d เฟรม", len(stems))

    # ------------------------------------------------------------------ #
    logger.info("[1/2] ตรวจ WCS ของเฟรม")
    wcs_store = ensure_wcs(cfg, wcs_path, stems, args.refresh_wcs)

    if args.wcs_only:
        logger.info("=" * 70)
        logger.info("ดึง WCS เสร็จแล้ว (%d เฟรมในดัชนี)", wcs_store.info()["n_frames"])
        logger.info("ขั้นถัดไป: python backend/scripts/download_aia.py --limit 5")
        logger.info("=" * 70)
        return 0

    if not wcs_store.available:
        logger.error("ไม่มี WCS ในดัชนีเลย — รัน --wcs-only ให้สำเร็จก่อน")
        return 1

    # ------------------------------------------------------------------ #
    channels = cfg.aia.channels
    if args.channels:
        wanted = {c.strip() for c in args.channels.split(",")}
        channels = [c for c in channels if c.key in wanted]
        if not channels:
            logger.error("ไม่มีช่องที่ตรงกับ --channels %s", args.channels)
            return 1

    logger.info(
        "[2/2] ประมวลผล %d เฟรม x %d ช่อง (%s)",
        len(stems), len(channels), ", ".join(c.key for c in channels),
    )
    logger.info("      (ดาวน์โหลด -> normalise ด้วย EXPTIME -> reproject ลงกริด %dpx -> ลบ FITS)",
                cfg.aia.target_size)

    store = AiaFrameStore(cfg.aia.root)
    all_peaks: dict[str, list[float]] = {c.key: [] for c in channels}
    n_frames_done = 0

    # ประมวลผลหลายเฟรมพร้อมกัน: เวลาเกือบทั้งหมดหมดไปกับการรอ network (วัดได้ ~14 วินาที
    # ต่อไฟล์) ซึ่งปล่อย GIL อยู่แล้ว ส่วน reproject ก็อยู่ใน scipy ที่ปล่อย GIL เช่นกัน
    # แต่ละเฟรมเขียนคนละไฟล์และชื่อไฟล์ชั่วคราวไม่ซ้ำกัน จึงไม่ต้องล็อกอะไร
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_frame, cfg, store, wcs_store, stem, channels, work_dir): stem
            for stem in stems
        }
        for index, future in enumerate(as_completed(futures), start=1):
            stem = futures[future]
            try:
                n_ok, peaks = future.result()
            except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
                logger.error("[%d/%d] %s ล้มเหลว: %s", index, len(stems), stem, exc)
                continue

            for key, value in peaks.items():
                all_peaks[key].append(value)
            if n_ok == len(channels):
                n_frames_done += 1
            logger.info("[%d/%d] %s: %d/%d ช่อง", index, len(stems), stem, n_ok, len(channels))

    # ------------------------------------------------------------------ #
    import shutil

    shutil.rmtree(work_dir, ignore_errors=True)

    final = AiaFrameStore(cfg.aia.root)
    size_mb = sum(p.stat().st_size for p in cfg.aia.root.rglob("*.npy")) / 1024**2

    logger.info("=" * 70)
    logger.info("เฟรมที่ครบทุกช่อง: %d/%d", n_frames_done, len(stems))
    logger.info("รวมบนดิสก์: %d เฟรม, %d ช่อง (%.1f MB)",
                final.n_frames, len(final.channels), size_mb)

    # p99.9 ใช้ปรับ aia.channels[].display_vmax ใน configs/data.yaml ถ้าภาพดูจืดหรือ saturate
    measured = {k: v for k, v in all_peaks.items() if v}
    if measured:
        logger.info("p99.9 ที่วัดได้ (เทียบกับ display_vmax ใน config):")
        for channel in channels:
            values = measured.get(channel.key)
            if values:
                logger.info(
                    "  %4s: p99.9 = %8.1f DN/s  (config vmax = %.0f)",
                    channel.key, float(np.median(values)), channel.display_vmax,
                )

    logger.info("ขั้นถัดไป:")
    logger.info("  1) ตรวจการจัดตำแหน่งด้วยตา:  python backend/scripts/plot_aia_alignment.py")
    logger.info("  2) เปิดเว็บ:                  uvicorn app.main:app --app-dir backend --reload")
    logger.info("=" * 70)
    return 0 if final.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
