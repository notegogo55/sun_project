"""ด่านตรวจ ticket 04 — พิสูจน์ว่าสร้าง sequence dataset ที่มี SHARP + X-ray บนกริด
12 ชม. ได้จริง โดยยังไม่ต้องมีเฟรมภาพเลย (เดินคู่ขนานกับการดาวน์โหลดเฟรมของ ticket 03)

    python backend/scripts/study/xray_sequence_gate_check.py

ใช้ SHARP keywords รายชั่วโมงที่มีอยู่แล้ว (``data/interim/sharp_keywords.parquet``)
subsample ลงกริด 12 ชม. ด้วย ``build_sequences()`` ตัวเดียวกับ production (คนละ
config object ที่ override ``cadence_hours``/``sequence.length`` เฉพาะสำเนานี้ —
ไม่แตะ ``configs/data.yaml`` หรือ artifacts ของ production) แล้วต่อคอลัมน์ X-ray
เข้าไปด้วย ``XrayFluxStore.bin_series()``

**การต่อคอลัมน์**: กริดของแต่ละ HARP ใน ``build_sequences()`` ยึดจาก ``t_first`` ของ
HARP นั้นเอง (ดู ``_regular_grid``) จึงมี phase ต่างกันไปคนละ HARP ไม่ตรงกับกริดกลาง
ของ X-ray ที่ยึด 00:00 UTC เป๊ะ — ใช้ ``pd.merge_asof`` แบบ nearest ด้วย tolerance
ครึ่งหนึ่งของ cadence (6 ชม.) รับประกันว่าทุกจุดหา bin ที่ใกล้ที่สุดของมันเจอเสมอ
วิธีนี้เป็นทางออกชั่วคราวสำหรับด่านตรวจนี้เท่านั้น — ticket 06 (unified dataset) จะ
รวมกริดของทุกอย่าง (SHARP, intensity, X-ray) ให้เป็นกริดเดียวกันเป๊ะแทน
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.build_sequences import (  # noqa: E402
    build_flare_lookup,
    build_sequences,
    clean_sharp_frame,
)
from sunseg.data.splits import (  # noqa: E402
    apply_splits,
    assign_harp_splits,
    describe_split_balance,
    harp_lifespans,
    verify_no_harp_overlap,
)
from sunseg.data.study_dataset import STUDY_CADENCE_HOURS, STUDY_SEQUENCE_LENGTH  # noqa: E402
from sunseg.data.xray_flux import XrayFluxStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("xray_sequence_gate_check")

REPORT_DIR_NAME = "lstm_feature_ablation_gate_check"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tolerance-hours",
        type=float,
        default=STUDY_CADENCE_HOURS / 2,
        help="ระยะห่างสูงสุดที่ยอมให้จับคู่จุดกริดของ HARP กับ bin ของ X-ray ที่ใกล้ที่สุด",
    )
    return p.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_data_config()

    interim = cfg.paths.interim
    sharp_path = interim / "sharp_keywords.parquet"
    flares_path = interim / "flares.parquet"
    for path in (sharp_path, flares_path):
        if not path.exists():
            logger.error("ไม่พบ %s — รัน backend/scripts/data/download_metadata.py ก่อน", path)
            return 1

    xray_store = XrayFluxStore(cfg.paths.raw / "xrs")
    if not xray_store.available:
        logger.error("ไม่พบไฟล์ XRS ที่ %s — รัน backend/scripts/data/download_xray.py ก่อน", xray_store.root)
        return 1

    logger.info("=" * 70)
    logger.info(
        "ด่านตรวจ ticket 04: SHARP + X-ray บนกริด %d ชม. (seq_len=%d)",
        STUDY_CADENCE_HOURS, STUDY_SEQUENCE_LENGTH,
    )
    logger.info("=" * 70)

    # config สำเนาเฉพาะด่านนี้ — ไม่แตะ configs/data.yaml หรือ artifacts ของ production
    study_cfg = cfg.model_copy(deep=True)
    study_cfg.sharp.cadence_hours = STUDY_CADENCE_HOURS
    study_cfg.sequence.length = STUDY_SEQUENCE_LENGTH

    # ------------------------------------------------------------------ #
    logger.info("[1/4] ทำความสะอาดตาราง SHARP + สร้าง sequence บนกริด %d ชม.", STUDY_CADENCE_HOURS)
    sharp_raw = pd.read_parquet(sharp_path)
    sharp = clean_sharp_frame(sharp_raw, study_cfg)

    flares = pd.read_parquet(flares_path)
    lookup = build_flare_lookup(flares, study_cfg.flare.positive_goes_class)

    x, y, meta = build_sequences(sharp, lookup, study_cfg)
    logger.info("  ได้ %d sample (SHARP ล้วน, %d feature)", len(meta), x.shape[-1])

    # ------------------------------------------------------------------ #
    logger.info("[2/4] แบ่ง train/val/test แบบ HARP-disjoint (เหมือน production)")
    lifespans = harp_lifespans(sharp)
    assigned = assign_harp_splits(lifespans, study_cfg.split)
    meta_indexed = meta.reset_index(names="_row")
    meta_split = apply_splits(meta_indexed, assigned)
    keep_idx = meta_split["_row"].to_numpy()
    x, y = x[keep_idx], y[keep_idx]
    meta = meta_split.drop(columns="_row").reset_index(drop=True)
    verify_no_harp_overlap(meta)
    logger.info("  เหลือ %d sample หลังแบ่ง split", len(meta))

    # ------------------------------------------------------------------ #
    logger.info("[3/4] คำนวณ X-ray bin แล้วต่อเข้ากับ meta ด้วย merge_asof (nearest)")
    start, end = meta["issue_time"].min(), meta["issue_time"].max()
    xray_bins = xray_store.bin_series(start, end, cadence_hours=STUDY_CADENCE_HOURS)
    logger.info(
        "  bin ทั้งหมด %d ช่วง %s..%s — มีข้อมูลจริง %d bin (%.1f%%)",
        len(xray_bins), start.date(), end.date(),
        xray_bins["xray_median"].notna().sum(),
        100 * xray_bins["xray_median"].notna().mean(),
    )

    meta_sorted = meta.sort_values("issue_time").reset_index(drop=True)
    merged = pd.merge_asof(
        meta_sorted,
        xray_bins.sort_values("t_rec"),
        left_on="issue_time",
        right_on="t_rec",
        direction="nearest",
        tolerance=pd.Timedelta(hours=args.tolerance_hours),
    )
    n_matched = merged["xray_median"].notna().sum()
    logger.info(
        "  sample ที่จับคู่ bin ของ X-ray ได้ (ในระยะ %.1f ชม.): %d / %d (%.1f%%)",
        args.tolerance_hours, n_matched, len(merged), 100 * n_matched / max(len(merged), 1),
    )
    if n_matched == 0:
        logger.error("  [FAIL] จับคู่ X-ray ไม่ได้เลยสักแถว — ตรวจช่วงวันที่ของไฟล์ XRS บนดิสก์")
        return 1

    # ------------------------------------------------------------------ #
    logger.info("[4/4] รายงานผล")
    complete = merged.dropna(subset=["xray_median"]).reset_index(drop=True)
    balance = describe_split_balance(complete)
    logger.info("สมดุลของแต่ละ split (เฉพาะแถวที่มี X-ray ครบ):\n%s", balance.to_string(index=False))

    out_dir = cfg.paths.artifacts / REPORT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "xray_sequence_report.md"

    lines = [
        "# ด่านตรวจ ticket 04 — SHARP + X-ray บนกริด 12 ชม.",
        "",
        f"**cadence:** {STUDY_CADENCE_HOURS} ชม. · **seq_len:** {STUDY_SEQUENCE_LENGTH} "
        f"(ประวัติ {STUDY_CADENCE_HOURS * STUDY_SEQUENCE_LENGTH / 24:.0f} วัน)",
        f"**ช่วงเวลา:** {start} ถึง {end}",
        "",
        f"sample จาก SHARP ล้วน (ก่อนกรอง X-ray): {len(meta):,}",
        f"sample ที่มี X-ray ครบ (ระยะจับคู่ไม่เกิน {args.tolerance_hours:.1f} ชม.): "
        f"{len(complete):,} ({100 * len(complete) / max(len(meta), 1):.1f}%)",
        "",
        "## สมดุลของแต่ละ split (เฉพาะแถวที่มี X-ray ครบ)",
        "",
        balance.to_string(index=False),
        "",
        "## หมายเหตุ",
        "",
        "- นี่คือด่านตรวจ ไม่ใช่ dataset สุดท้าย — กริดของแต่ละ HARP ยึดจาก t_first ของ "
        "ตัวเอง (คนละ phase กัน) จับคู่กับ X-ray ด้วย nearest-match ชั่วคราว ticket 06 "
        "จะรวมกริดของ SHARP/intensity/X-ray ให้ตรงกันเป๊ะแทน",
        "- ยังไม่มีคอลัมน์ intensity เพราะยังไม่มีเฟรมภาพ (ticket 03 กำลังดาวน์โหลดอยู่)",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("เขียนรายงานที่ %s", report_path)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
