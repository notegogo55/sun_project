"""สร้าง flare catalog ฉบับแก้เลข NOAA AR (``data/interim/flares_v2.parquet``) จากของที่มีบนดิสก์แล้ว

    python backend/scripts/data/rebuild_flares_v2.py

**ทำไมต้องมี**: ``flares.parquet`` เดิมสร้างก่อนแก้บั๊กใน ``hek_client`` (ดู ``normalise_noaa_ar``) —
ส่วนที่มาจาก HEK (2015 เป็นต้นไป) ทิ้ง flare ที่ SWPC ใส่ AR = 0 และที่ SSW Latest Events เขียนเลข
4 หลักไปเงียบ ๆ ปี 2022 จับคู่ HARP ได้แค่ 22 จาก 193 ครั้งของ M+ · ปี 2023 ได้ 56 จาก 382

**ทำไมเป็นไฟล์แยก ไม่เขียนทับ**: ผลเดิมทุกชุด (model_comparison, lstm_feature_ablation, โมเดล production)
ถูกเทรนด้วย label จากไฟล์เดิม — เขียนทับแล้วจะสร้างซ้ำไม่ได้ งานใหม่ชี้มาที่ไฟล์นี้ผ่าน
``study/build_dataset.py --flares-path``

**ไม่แตะเครือข่าย**: ส่วน HEK อ่านจาก cache รายก้อน (``data/raw/cache/hek/``) ผ่าน ``normalise_noaa_ar`` +
``deduplicate_flares`` ตัวเดียวกับ ``fetch_flare_events`` ส่วน NGDC (เลข AR 5 หลักถูกอยู่แล้ว) เอาแถวเดิม
จาก ``flares.parquet`` มาทั้งหมด แล้วรวมสองแหล่งโดยตัด flare ซ้ำข้ามแหล่ง (HARP เดียวกัน คลาสเดียวกัน
peak ห่างไม่เกิน ``--dup-minutes`` นาที) ให้แถวของ NGDC ชนะ
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.flare_catalog import map_flares_to_harps  # noqa: E402
from sunseg.data.harp_noaa_map import build_noaa_to_harp  # noqa: E402
from sunseg.data.hek_client import deduplicate_flares, normalise_noaa_ar  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("rebuild_flares_v2")

NGDC = "NOAA/NGDC"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, help="ปริยาย data/interim/flares_v2.parquet")
    p.add_argument("--dup-minutes", type=float, default=10.0, help="peak ห่างไม่เกินเท่านี้ถือเป็น flare เดียวกันข้ามแหล่ง")
    return p.parse_args()


def collapse_near_duplicates(df: pd.DataFrame, minutes: float) -> pd.DataFrame:
    """ตัดแถวที่ HARP เดียวกัน คลาสเดียวกัน และ peak ห่างกันไม่เกิน ``minutes`` — เก็บแถวแรกตาม ``_rank``

    ``deduplicate_flares`` จับได้แค่ peak นาทีเดียวกันเป๊ะ แต่ NGDC กับ HEK หรือ SWPC กับ SSW รายงาน
    peak ต่างกันได้ 1-7 นาที (วัดจากข้อมูลจริง) — label ไม่เปลี่ยน แต่จำนวน flare ย้อนหลังจะนับซ้ำ
    """
    ordered = df.sort_values(["HARPNUM", "goes_class", "peak_time", "_rank"]).reset_index(drop=True)
    gap = ordered.groupby(["HARPNUM", "goes_class"])["peak_time"].diff()
    new_event = gap.isna() | (gap > pd.Timedelta(minutes=minutes))
    ordered["_event"] = new_event.cumsum()
    kept = ordered.sort_values(["_event", "_rank"]).drop_duplicates("_event", keep="first")
    return kept.drop(columns="_event")


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_data_config()
    interim = cfg.paths.interim
    out = args.out or interim / "flares_v2.parquet"

    old = pd.read_parquet(interim / "flares.parquet")
    ngdc = old[old["frm_name"] == NGDC]
    logger.info("แถว NGDC จากไฟล์เดิม: %d (จาก %d)", len(ngdc), len(old))

    cache_files = sorted((cfg.paths.raw / "cache" / "hek").glob("hek_flares_*.parquet"))
    if not cache_files:
        logger.error("ไม่พบ cache ของ HEK ที่ %s", cfg.paths.raw / "cache" / "hek")
        return 1
    raw = pd.concat([pd.read_parquet(path) for path in cache_files], ignore_index=True)
    hek = deduplicate_flares(normalise_noaa_ar(raw))
    logger.info("HEK จาก cache %d ก้อน: %d event หลังตัดซ้ำ", len(cache_files), len(hek))

    pairs = pd.read_parquet(interim / "harp_noaa_pairs.parquet")
    hek_harp = map_flares_to_harps(hek, build_noaa_to_harp(pairs))

    columns = list(old.columns)
    combined = pd.concat(
        [ngdc.assign(_rank=0), hek_harp.reindex(columns=columns).assign(_rank=1)], ignore_index=True
    )
    for column in ("start_time", "peak_time", "end_time"):
        combined[column] = pd.to_datetime(combined[column])
    before = len(combined)
    combined = collapse_near_duplicates(combined, args.dup_minutes)
    logger.info("ตัด flare ซ้ำข้ามแหล่ง (≤ %.0f นาที): %d -> %d คู่ (HARP, flare)", args.dup_minutes, before, len(combined))

    combined = combined.drop(columns="_rank").sort_values(["peak_time", "HARPNUM"]).reset_index(drop=True)
    combined["noaa_ar"] = combined["noaa_ar"].astype("int64")
    combined["HARPNUM"] = combined["HARPNUM"].astype("int64")
    combined.to_parquet(out, index=False)

    def per_year(df: pd.DataFrame, min_flux: float = 0.0) -> pd.Series:
        sub = df[df["peak_flux"] >= min_flux]
        return sub.groupby(pd.to_datetime(sub["peak_time"]).dt.year).size()

    table = pd.DataFrame(
        {
            "ทุกคลาส เดิม": per_year(old),
            "ทุกคลาส v2": per_year(combined),
            "M+ เดิม": per_year(old, 1e-5),
            "M+ v2": per_year(combined, 1e-5),
        }
    ).fillna(0).astype(int)
    logger.info("คู่ (HARP, flare) รายปี:\n%s", table.to_string())
    logger.info("เขียน %s (%d แถว)", out, len(combined))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
