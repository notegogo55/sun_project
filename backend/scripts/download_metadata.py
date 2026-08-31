"""ดาวน์โหลด metadata ทั้งหมดที่ต้องใช้เทรน LSTM (ไม่แตะไฟล์ภาพ)

ขั้นตอนนี้เบามาก (ไม่กี่ร้อย MB) และเร็วกว่าการดาวน์โหลดภาพหลายเท่า จึงควรรัน
ให้ผ่านก่อนเสมอ เพื่อยืนยันว่าการเชื่อมต่อ JSOC/HEK และ pipeline การ label ถูกต้อง

ตัวอย่างการใช้งาน::

    # ทดสอบด้วยช่วงสั้นๆ ก่อน (แนะนำให้ทำครั้งแรกเสมอ)
    python backend/scripts/download_metadata.py --start 2014-01-01 --end 2014-03-01

    # ดึงเต็มช่วงตามที่กำหนดใน configs/data.yaml
    python backend/scripts/download_metadata.py

    # เพิ่มหน้าต่าง case study เข้าไปในไฟล์เดิม (ไม่เขียนทับชุดเทรน)
    python backend/scripts/download_metadata.py --case-study

ผลลัพธ์ทั้งหมดถูก cache เป็นก้อนย่อย รันซ้ำจะทำต่อจากที่ค้างไว้
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.flare_catalog import (  # noqa: E402
    load_flares,
    map_flares_to_harps,
    summarise_flares,
)
from sunseg.data.harp_noaa_map import (  # noqa: E402
    build_noaa_to_harp,
    download_harp_noaa_table,
    load_harp_noaa_pairs,
)
from sunseg.data.jsoc_client import JsocClient  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("download_metadata")


def _write_parquet(
    df: pd.DataFrame, path: Path, key: list[str], append: bool, label: str
) -> pd.DataFrame:
    """เขียน parquet แบบ atomic และรวมกับไฟล์เดิมได้ตามต้องการ

    การรวมจำเป็นสำหรับโหมด case study: ช่วง 2024-05 ต้องอยู่ในไฟล์เดียวกับชุดเทรน
    2011-2017 เพราะ build_sequences.py และ splits.py อ่านไฟล์เดียว การเขียนทับตรงๆ
    จะทำให้ข้อมูลหลายปีที่ใช้เวลาดาวน์โหลดเป็นชั่วโมงหายไปโดยไม่มีคำเตือน

    คืนค่า DataFrame หลังรวมแล้ว เพื่อให้ผู้เรียกรายงานตัวเลขที่ตรงกับไฟล์จริง
    """
    if append and path.exists():
        old = pd.read_parquet(path)
        merged = (
            pd.concat([old, df], ignore_index=True)
            .drop_duplicates(subset=key, keep="last")
            .sort_values(key)
            .reset_index(drop=True)
        )
        logger.info(
            "รวม %s เข้ากับไฟล์เดิม: เดิม %d + ใหม่ %d -> %d แถว (ตัดซ้ำแล้ว)",
            label,
            len(old),
            len(df),
            len(merged),
        )
        df = merged

    # เขียนลงไฟล์ชั่วคราวก่อนแล้วค่อย rename — ถ้าถูกขัดจังหวะกลางคัน ไฟล์เดิมยังอยู่ครบ
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=_as_date, help="วันเริ่มต้น YYYY-MM-DD (ทับค่าใน config)")
    p.add_argument("--end", type=_as_date, help="วันสิ้นสุด YYYY-MM-DD (ทับค่าใน config)")
    p.add_argument("--skip-sharp", action="store_true", help="ข้ามการดึง SHARP keywords")
    p.add_argument("--skip-flares", action="store_true", help="ข้ามการดึง flare catalog")
    p.add_argument(
        "--case-study",
        action="store_true",
        help="ใช้หน้าต่าง case_study จาก config แทน time_range (เปิด --append ให้อัตโนมัติ)",
    )
    p.add_argument(
        "--hek-chunk-days",
        type=int,
        help="ทับขนาด chunk ของ HEK (ลดลงถ้าเจอช่วงที่ flare ถี่มากจนได้ตารางเปล่า)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="รวมกับ parquet เดิมแทนการเขียนทับ",
    )
    p.add_argument(
        "--flare-source",
        choices=["auto", "ngdc", "hek"],
        default="auto",
        help="แหล่งรายการ flare (auto = ใช้ NGDC ถ้าช่วงเวลาอยู่ในคลัง 1975-2017)",
    )
    p.add_argument("--force-harp-table", action="store_true", help="ดาวน์โหลดตาราง HARP-NOAA ใหม่")
    return p.parse_args()


def _as_date(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    cfg.paths.mkdirs()

    setup_logging(log_file=cfg.paths.artifacts / "logs" / "download_metadata.log")

    window = cfg.case_study if args.case_study else cfg.time_range
    start = args.start or window.start
    end = args.end or window.end
    # case study ต้อง append เสมอ มิฉะนั้น metadata ของชุดเทรนจะถูกเขียนทับ
    append = args.append or args.case_study
    if start >= end:
        logger.error("วันเริ่มต้น (%s) ต้องมาก่อนวันสิ้นสุด (%s)", start, end)
        return 1

    interim = cfg.paths.interim
    cache = cfg.paths.raw / "cache"
    logger.info("=" * 70)
    logger.info(
        "ดาวน์โหลด metadata: %s ถึง %s%s%s",
        start,
        end,
        "  [CASE STUDY]" if args.case_study else "",
        "  [APPEND]" if append else "",
    )
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    logger.info("[1/3] ตาราง HARPNUM <-> NOAA AR")
    table_path = download_harp_noaa_table(
        cfg.paths.raw / "all_harps_with_noaa_ars.txt", force=args.force_harp_table
    )
    pairs = load_harp_noaa_pairs(table_path)
    pairs.to_parquet(interim / "harp_noaa_pairs.parquet", index=False)

    # ------------------------------------------------------------------ #
    sharp_path = interim / "sharp_keywords.parquet"
    if args.skip_sharp and sharp_path.exists():
        logger.info("[2/3] ข้ามการดึง SHARP keywords (ใช้ไฟล์เดิม)")
    else:
        logger.info("[2/3] SHARP keywords จาก %s", cfg.jsoc.sharp_series)
        client = JsocClient(
            max_retries=cfg.jsoc.max_retries, retry_backoff_s=cfg.jsoc.retry_backoff_s
        )
        sharp = client.query_sharp_series(
            series=cfg.jsoc.sharp_series,
            start=start,
            end=end,
            keys=cfg.sharp.all_keys,
            cadence_hours=cfg.sharp.cadence_hours,
            cache_dir=cache / "sharp",
        )
        if sharp.empty:
            logger.error("JSOC ไม่คืนข้อมูล SHARP เลย — ตรวจสอบการเชื่อมต่อหรือชื่อ series")
            return 1

        # แปลง T_REC เป็น datetime ตั้งแต่ตอนนี้ เพื่อไม่ต้องทำซ้ำในทุกขั้นถัดไป
        sharp["t_rec"] = client.parse_trec(sharp["T_REC"])
        sharp = _write_parquet(
            sharp,
            sharp_path,
            key=["HARPNUM", "T_REC"],
            append=append,
            label="SHARP keywords",
        )
        logger.info(
            "บันทึก SHARP keywords: %s (%d แถว, %d HARP)",
            sharp_path.name,
            len(sharp),
            sharp["HARPNUM"].nunique(),
        )

    # ------------------------------------------------------------------ #
    flares_path = interim / "flares.parquet"
    if args.skip_flares and flares_path.exists():
        logger.info("[3/3] ข้ามการดึง flare catalog (ใช้ไฟล์เดิม)")
    else:
        logger.info("[3/3] รายการ flare (แหล่งที่ขอ: %s)", args.flare_source)
        flares = load_flares(
            source=args.flare_source,
            start=start,
            end=end,
            cache_dir=cache,
            hek_chunk_days=args.hek_chunk_days or cfg.flare.hek_chunk_days,
        )
        if flares.empty:
            logger.error("ไม่ได้รับ flare event เลย — รัน backend/scripts/check_connectivity.py เพื่อวินิจฉัย")
            return 1

        logger.info("สรุปรายการ flare:\n%s", summarise_flares(flares))

        noaa_to_harp = build_noaa_to_harp(pairs)
        flares_with_harp = map_flares_to_harps(flares, noaa_to_harp)
        flares_with_harp = _write_parquet(
            flares_with_harp,
            flares_path,
            key=["HARPNUM", "start_time", "goes_class"],
            append=append,
            label="flare",
        )
        logger.info("บันทึก flare ที่จับคู่ HARP แล้ว: %s (%d แถว)", flares_path.name, len(flares_with_harp))

    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("เสร็จสิ้น — ไฟล์ที่ได้:")
    for path in sorted(interim.glob("*.parquet")):
        size_mb = path.stat().st_size / 1024**2
        # อ่านจำนวนแถวจาก footer metadata — ไม่ต้องโหลดข้อมูลจริงเข้าหน่วยความจำ
        n_rows = pq.ParquetFile(path).metadata.num_rows
        logger.info("  %-28s %8.1f MB  %9d แถว", path.name, size_mb, n_rows)
    logger.info("ขั้นถัดไป:  python backend/scripts/build_sequences.py")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
