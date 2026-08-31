"""แปลง metadata ที่ดาวน์โหลดไว้ เป็น sequence dataset พร้อมเทรน LSTM

    python backend/scripts/build_sequences.py

อ่านจาก ``data/interim/`` แล้วเขียนลง ``data/processed/sequences/``::

    X.npy            (N, L, F) float32 — SHARP parameters ย้อนหลัง L ชั่วโมง
    y.npy            (N,)      uint8   — 1 = มี flare >= M1.0 ใน 24 ชม. ถัดไป
    meta.parquet     ข้อมูลกำกับรายตัวอย่าง (HARPNUM, เวลาออกพยากรณ์, split, ...)
    norm_stats.npz   mean/std ที่คำนวณจาก **train เท่านั้น**
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.build_sequences import (  # noqa: E402
    apply_normalisation,
    build_flare_lookup,
    build_sequences,
    clean_sharp_frame,
    compute_normalisation,
)
from sunseg.data.splits import (  # noqa: E402
    apply_splits,
    assign_harp_splits,
    describe_split_balance,
    harp_lifespans,
    verify_no_harp_overlap,
)
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("build_sequences")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="สร้างและรายงานสถิติ แต่ไม่เขียนไฟล์ผลลัพธ์",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help="ข้ามการแบ่ง train/val/test (ใช้ตอนทดสอบด้วยข้อมูลช่วงสั้น)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "build_sequences.log")

    interim = cfg.paths.interim
    out_dir = cfg.paths.processed / "sequences"

    sharp_path = interim / "sharp_keywords.parquet"
    flares_path = interim / "flares.parquet"
    for path in (sharp_path, flares_path):
        if not path.exists():
            logger.error("ไม่พบ %s — รัน backend/scripts/download_metadata.py ก่อน", path)
            return 1

    logger.info("=" * 70)
    logger.info("สร้าง sequence dataset")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    logger.info("[1/5] ทำความสะอาดตาราง SHARP")
    sharp_raw = pd.read_parquet(sharp_path)
    sharp = clean_sharp_frame(sharp_raw, cfg)

    # ------------------------------------------------------------------ #
    logger.info("[2/5] สร้างตารางค้นหา flare")
    flares = pd.read_parquet(flares_path)
    lookup = build_flare_lookup(flares, cfg.flare.positive_goes_class)

    # ------------------------------------------------------------------ #
    logger.info("[3/5] เลื่อนหน้าต่างสร้าง sequence")
    x, y, meta = build_sequences(sharp, lookup, cfg)

    # ------------------------------------------------------------------ #
    if args.no_split:
        logger.info("[4/5] ข้ามการแบ่ง split ตามที่ระบุ")
        meta["split"] = "train"
        keep = np.ones(len(meta), dtype=bool)
    else:
        logger.info("[4/5] แบ่ง train/val/test แบบ HARP-disjoint")
        lifespans = harp_lifespans(sharp)
        assigned = assign_harp_splits(lifespans, cfg.split)

        meta_before = len(meta)
        meta_indexed = meta.reset_index(names="_row")
        meta_split = apply_splits(meta_indexed, assigned)
        keep = np.zeros(meta_before, dtype=bool)
        keep[meta_split["_row"].to_numpy()] = True

        x, y = x[keep], y[keep]
        meta = meta_split.drop(columns="_row").reset_index(drop=True)
        verify_no_harp_overlap(meta)

    if len(x) == 0:
        logger.error(
            "ไม่เหลือ sample หลังแบ่ง split — ช่วงเวลาของข้อมูลอาจสั้นเกินกว่าเส้นแบ่งใน "
            "data.yaml (ลองใช้ --no-split เมื่อทดสอบด้วยข้อมูลไม่กี่เดือน)"
        )
        return 1

    balance = describe_split_balance(meta)
    logger.info("สมดุลของแต่ละ split:\n%s", balance.to_string(index=False))

    # ------------------------------------------------------------------ #
    logger.info("[5/5] คำนวณ normalisation statistics (จาก train เท่านั้น)")
    train_mask = (meta["split"] == "train").to_numpy()
    if not train_mask.any():
        logger.warning("ไม่มี sample ใน train — ใช้ทั้งชุดคำนวณแทน (โหมดทดสอบเท่านั้น)")
        train_mask = None
    stats = compute_normalisation(x, train_mask)

    # ตรวจว่าการ normalise ได้ผลจริง: หลังแปลงแล้ว train ควรมี mean~0 std~1
    normalised = apply_normalisation(x if train_mask is None else x[train_mask], stats)
    logger.info(
        "  หลัง normalise: mean=%.3f (ควรใกล้ 0), std=%.3f (ควรใกล้ 1), "
        "ช่วงค่า [%.1f, %.1f]",
        float(normalised.mean()),
        float(normalised.std()),
        float(normalised.min()),
        float(normalised.max()),
    )

    _report_sanity(y, meta, cfg)

    # ------------------------------------------------------------------ #
    if args.dry_run:
        logger.info("โหมด --dry-run: ไม่เขียนไฟล์")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X.npy", x)
    np.save(out_dir / "y.npy", y)
    meta.to_parquet(out_dir / "meta.parquet", index=False)
    np.savez(
        out_dir / "norm_stats.npz",
        mean=stats["mean"],
        std=stats["std"],
        features=np.array(cfg.sharp.features),
        # บันทึกชื่อการแปลงไว้ด้วย เพื่อให้ฝั่ง inference ยืนยันได้ว่าใช้สูตรเดียวกัน
        transform=np.array("signed_log1p"),
    )

    size_mb = (x.nbytes + y.nbytes) / 1024**2
    logger.info("=" * 70)
    logger.info("บันทึกแล้วที่ %s (%.1f MB)", out_dir, size_mb)
    logger.info("  X.npy          %s", x.shape)
    logger.info("  y.npy          %s  (positive %.2f%%)", y.shape, 100 * y.mean())
    logger.info("  meta.parquet   %d แถว", len(meta))
    logger.info("ขั้นถัดไป:  python backend/scripts/train_lstm.py")
    logger.info("=" * 70)
    return 0


def _report_sanity(y: np.ndarray, meta: pd.DataFrame, cfg) -> None:
    """เทียบสัดส่วน positive กับค่าที่งานวิจัยรายงาน — จับ label ที่ผิดพลาดร้ายแรง"""
    pos_rate = float(y.mean())
    logger.info("-" * 70)
    logger.info("ตรวจความสมเหตุสมผลของ label")
    logger.info(
        "  สัดส่วน positive: %.2f%%  (คาดหวัง ~1-5%% สำหรับ >= %s ใน %d ชม.)",
        100 * pos_rate,
        cfg.flare.positive_goes_class,
        cfg.flare.horizon_hours,
    )

    n_harp_pos = meta.loc[meta["label"] == 1, "HARPNUM"].nunique()
    n_harp = meta["HARPNUM"].nunique()
    logger.info(
        "  HARP ที่มีหน้าต่าง positive อย่างน้อยหนึ่ง: %d จาก %d (%.1f%%)",
        n_harp_pos,
        n_harp,
        100 * n_harp_pos / max(n_harp, 1),
    )

    if pos_rate == 0:
        logger.error("  [FAIL] ไม่มี positive เลย — การจับคู่ flare กับ HARP น่าจะผิดพลาด")
    elif pos_rate > 0.20:
        logger.warning("  [WARN] positive สูงผิดปกติ — ตรวจสอบเกณฑ์ความแรงและหน้าต่างเวลา")
    elif pos_rate < 0.002:
        logger.warning("  [WARN] positive น้อยผิดปกติ — อาจสูญเสีย flare ตอนจับคู่ AR")
    else:
        logger.info("  [ OK ] อยู่ในช่วงที่คาดหวัง")
    logger.info("-" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
