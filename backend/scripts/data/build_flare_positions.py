"""จับคู่ตำแหน่ง flare จาก PositionFlare เข้ากับแคตตาล็อกที่โมเดลใช้ทำ label

ผลลัพธ์ ``data/processed/flare_positions.parquet`` (+ ``.json`` ข้อมูลประกอบ) คือ flare ทุกดวงใน
``data/interim/flares.parquet`` ระดับ C ขึ้นไป พร้อมพิกัดจาก PositionFlare ที่หาคู่เจอ — แผนที่
ตำแหน่ง flare ในหน้าแรกของเว็บอ่านจากไฟล์นี้ (ดูเหตุผลของกติกาการจับคู่ใน
``sunseg.data.flare_positions``)

ตัวอย่างการใช้งาน::

    python backend/scripts/data/build_flare_positions.py
    python backend/scripts/data/build_flare_positions.py --csv D:/position_flare/backend/source_data/out/flares_all_cycles.csv
    python backend/scripts/data/build_flare_positions.py --tolerance-min 5

ต้องมี ``data/interim/flares.parquet`` ก่อน (สร้างด้วย download_metadata.py) ส่วน CSV ของ
PositionFlare อ่านอย่างเดียว ไม่แก้ไขอะไรในโปรเจคนั้น — รันซ้ำได้ทุกเมื่อที่ฝั่งใดฝั่งหนึ่งอัปเดต
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.flare_positions import (  # noqa: E402
    MAX_FLUX_RATIO,
    match_positions,
    meta_path_for,
    model_catalog_events,
    read_position_flare_csv,
)
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("build_flare_positions")


def _log_summary(result: pd.DataFrame, position_flare: pd.DataFrame) -> None:
    n = len(result)
    matched = int(result["matched"].sum())
    located = int(result["lat"].notna().sum())
    logger.info("flare ของโมเดล (C+)        : %d ดวง", n)
    logger.info("จับคู่ PositionFlare ได้    : %d (%.1f%%)", matched, 100 * matched / max(n, 1))
    logger.info("มีพิกัดวางบนแผนที่ได้      : %d (%.1f%%)", located, 100 * located / max(n, 1))

    pairs = result[result["matched"]]
    letter_diff = (pairs["goes_class"].str[0] != pairs["pf_goes_class"].str[0]).sum()
    logger.info(
        "ตัวอักษรคลาสต่างกัน         : %d คู่ (ปกติ — สเกล NGDC เดิม vs science ดู docstring ของ"
        " sunseg.data.flare_positions)", letter_diff,
    )
    exact = int((pairs["match_dt_min"].abs() == 0).sum())
    logger.info("เวลาพีคตรงกันเป๊ะ          : %d / %d คู่", exact, matched)

    start, end = result["peak_time"].min(), result["peak_time"].max()
    in_range = position_flare[(position_flare["peak"] >= start) & (position_flare["peak"] <= end)]
    unused = len(in_range) - matched
    logger.info(
        "PositionFlare ในช่วงเดียวกันที่ไม่อยู่ในแคตตาล็อกโมเดล: %d ดวง (ไม่ขึ้นแผนที่ — "
        "แผนที่แสดงเฉพาะชุดที่โมเดลใช้)", unused,
    )

    by_year = result.assign(year=pd.to_datetime(result["peak_time"]).dt.year).groupby("year").agg(
        flares=("goes_class", "size"),
        matched=("matched", "sum"),
        located=("lat", lambda s: s.notna().sum()),
    )
    logger.info("รายปี:\n%s", by_year.to_string())


def main() -> int:
    setup_logging()
    config = load_data_config()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv", type=Path, default=config.position_flare.csv,
        help="flares_all_cycles.csv ของ PositionFlare (ค่าเริ่มต้นจาก data.yaml / SUNSEG_POSITION_FLARE_CSV)",
    )
    parser.add_argument(
        "--tolerance-min", type=float, default=config.position_flare.match_tolerance_min,
        help="เวลาพีคห่างกันได้สูงสุดกี่นาที",
    )
    parser.add_argument(
        "--out", type=Path, default=config.paths.processed / "flare_positions.parquet",
        help="ไฟล์ผลลัพธ์",
    )
    args = parser.parse_args()

    catalog_path = config.paths.interim / "flares.parquet"
    if not catalog_path.exists():
        logger.error("ไม่พบแคตตาล็อกของโมเดลที่ %s — รัน backend/scripts/data/download_metadata.py ก่อน", catalog_path)
        return 1
    if not args.csv.exists():
        logger.error(
            "ไม่พบ CSV ของ PositionFlare ที่ %s — ตั้ง SUNSEG_POSITION_FLARE_CSV ใน .env "
            "หรือส่ง --csv ให้ชี้ไปที่ flares_all_cycles.csv", args.csv,
        )
        return 1

    logger.info("=" * 62)
    logger.info("จับคู่ตำแหน่ง flare: %s  x  %s", catalog_path.name, args.csv)
    logger.info("tolerance เวลาพีค ±%.1f นาที · อัตราส่วนฟลักซ์ ≤ %.0f เท่า", args.tolerance_min, MAX_FLUX_RATIO)
    logger.info("=" * 62)

    events = model_catalog_events(pd.read_parquet(catalog_path))
    position_flare = read_position_flare_csv(args.csv)
    logger.info("อ่าน PositionFlare: %d ดวง (%s → %s)", len(position_flare),
                position_flare["peak"].min(), position_flare["peak"].max())

    result = match_positions(events, position_flare, tolerance_min=args.tolerance_min)
    _log_summary(result, position_flare)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".part")
    result.to_parquet(tmp, index=False)
    tmp.replace(args.out)  # atomic — แอปที่เปิดอยู่จะไม่เจอไฟล์ครึ่ง ๆ กลาง ๆ

    meta = {
        "built_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_csv": str(args.csv),
        "model_catalog": str(catalog_path),
        "tolerance_min": args.tolerance_min,
        "max_flux_ratio": MAX_FLUX_RATIO,
        "n": len(result),
        "n_matched": int(result["matched"].sum()),
        "n_located": int(result["lat"].notna().sum()),
    }
    meta_path_for(args.out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("เขียนแล้ว: %s (+ %s) — รีสตาร์ต backend เพื่อโหลดใหม่", args.out, meta_path_for(args.out).name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
