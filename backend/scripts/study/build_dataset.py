"""สร้าง dataset ก้อนเดียวของงานเปรียบเทียบ feature 5 แบบ (ticket 06 ของ
``.scratch/lstm-feature-ablation/``) — รวม SHARP + intensity + X-ray บนกริดเวลาเดียวกัน

    python backend/scripts/study/build_dataset.py

อ่าน SHARP keywords รายชั่วโมงที่มีอยู่แล้ว (``data/interim/sharp_keywords.parquet``)
subsample ลงกริด ``--cadence-hours`` (ปริยาย 12) ต่อกับ intensity ทุกไฟล์ที่เจอใต้
``artifacts/intensity_study/**/intensity*.parquet`` และ X-ray จาก ``XrayFluxStore.bin_series()``

**คนละ path จาก production เสมอ** — ไม่เขียนทับ ``data/processed/sequences/`` เดิม
เขียนลง ``data/processed/study_sequences/`` แทน (override ด้วย ``--out-dir``)

ประมวลผลช่วง ``--main-start``..``--main-end`` เป็นหน้าต่างเวลาต่อเนื่องเดียว (ดู
``sunseg.data.study_dataset``) ถ้ายังไม่มี intensity ครบทั้งช่วงจะได้ 0 sample โดยอัตโนมัติ
ไม่ error — เกณฑ์ "ต้องมี intensity+X-ray ครบทุก timestep" กรองออกเอง
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.build_sequences import (  # noqa: E402
    build_flare_lookup,
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
from sunseg.data.study_dataset import (  # noqa: E402
    STUDY_CADENCE_HOURS,
    STUDY_SEQUENCE_LENGTH,
    build_unified_window,
)
from sunseg.data.xray_flux import XrayFluxStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("build_study_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-start", default="2011-01-01", help="จุดเริ่มของช่วงข้อมูล")
    p.add_argument("--main-end", default="2026-01-01", help="จุดสิ้นสุดของช่วงข้อมูล")
    p.add_argument("--out-dir", default=None, help="override path ที่จะเขียนผลลัพธ์")
    p.add_argument(
        "--no-split",
        action="store_true",
        help="ข้ามการแบ่ง train/val/test แบบ HARP-disjoint เตรียม pool เต็มไว้ใช้กับ "
        "study/train.py --cv (ไม่ทิ้ง HARP ตามเส้นแบ่งเดียว เพราะ CV ต้องลองหลายเส้นแบ่ง)",
    )
    p.add_argument(
        "--flares-path", type=Path,
        help="flare catalog ที่ใช้ทำ label (ปริยาย data/interim/flares.parquet) — "
        "data/interim/flares_v2.parquet คือฉบับแก้เลข NOAA AR (ดู data/rebuild_flares_v2.py)",
    )
    p.add_argument(
        "--intensity-dir", type=Path,
        help="โฟลเดอร์ที่หา intensity*.parquet (ปริยาย artifacts/intensity_study)",
    )
    p.add_argument(
        "--extra-aia-channels", default="",
        help="ช่อง AIA ที่เพิ่มต่อท้ายหลัง X-ray คั่นด้วยจุลภาค เช่น 94,131 — ต้องมีในตาราง intensity ครบทุก timestep",
    )
    p.add_argument(
        "--positive-class",
        help="คลาส GOES ต่ำสุดที่นับเป็น positive (ปริยาย flare.positive_goes_class ใน data.yaml = M1.0) — "
        "เช่น X1.0 สำหรับแบบจำลองที่เตือน flare ระดับ X · แถวข้อมูลและ split ไม่เปลี่ยน เปลี่ยนแค่ label",
    )
    p.add_argument(
        "--flare-history", action="store_true",
        help="เพิ่มคอลัมน์ประวัติ flare ของ HARP นั้นเอง (ar_flare_*) ต่อท้ายสุด จาก catalog เดียวกับ label",
    )
    return p.parse_args()


def _find_intensity_tables(artifacts_dir: Path, study_dir: Path | None = None) -> pd.DataFrame:
    """รวมทุกไฟล์ ``intensity*.parquet`` ใต้ ``artifacts/intensity_study/`` เป็นตารางเดียว
    **ยกเว้นของ ``gate_check/``**

    ไม่ผูกชื่อไฟล์ตายตัว เพื่อไม่ต้องแก้สคริปต์นี้เมื่อ ticket 05 เพิ่มไฟล์ของช่วง
    2011-2017 เข้ามา — แค่วางไว้ใต้โฟลเดอร์เดียวกันแล้วรันใหม่

    ตารางของด่านเดือนเดียว (``gate_check/intensity_may2024.parquet``) ถูกข้ามโดยตั้งใจ: มันถูก
    สร้างก่อนแก้บั๊กที่คืนหนึ่งแถวต่อ blob (222 จาก 876 แถวเป็นเศษของ AR ที่ U-Net แตกเป็นหลาย
    ชิ้น — ดู ``extract_frame_intensities``) ส่วน ``full_range/`` ของ ticket 05 สกัด พ.ค. 2024
    ใหม่ครบทุกเฟรมด้วยโค้ดที่แก้แล้ว ถ้ารวมเข้ามา ``drop_duplicates(keep="last")`` ด้านล่างจะ
    เลือกแถวของ gate_check (path เรียงหลัง full_range) ซึ่งคือเศษชิ้นเล็ก
    """
    study_dir = study_dir or artifacts_dir / "intensity_study"
    files = sorted(path for path in study_dir.rglob("intensity*.parquet") if "gate_check" not in path.parts)
    if not files:
        logger.warning(
            "ไม่พบไฟล์ intensity ใต้ %s — ทุกแถวจะไม่มี intensity ครบ "
            "(sample ของหน้าต่างนั้นจะได้ 0 แถว รอ ticket 05)", study_dir,
        )
        return pd.DataFrame()

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        logger.info("  ใช้ intensity table: %s (%d แถว)", path, len(df))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["HARPNUM", "issue_time"], keep="last")
    if len(combined) != before:
        logger.warning("พบ intensity ซ้ำ (HARPNUM, issue_time) %d แถว — เก็บอันหลังสุด", before - len(combined))
    return combined


def main() -> int:
    setup_logging()
    args = parse_args()
    cfg = load_data_config()

    study_cfg = cfg.model_copy(deep=True)
    study_cfg.sharp.cadence_hours = STUDY_CADENCE_HOURS
    study_cfg.sequence.length = STUDY_SEQUENCE_LENGTH
    if args.positive_class:
        study_cfg.flare.positive_goes_class = args.positive_class

    interim = cfg.paths.interim
    sharp_path = interim / "sharp_keywords.parquet"
    flares_path = args.flares_path or interim / "flares.parquet"
    extra_channels = tuple(c.strip() for c in args.extra_aia_channels.split(",") if c.strip())
    for path in (sharp_path, flares_path):
        if not path.exists():
            logger.error("ไม่พบ %s — รัน backend/scripts/data/download_metadata.py ก่อน", path)
            return 1

    logger.info("=" * 70)
    logger.info("สร้าง unified study dataset (cadence %d ชม., seq_len %d)", STUDY_CADENCE_HOURS, STUDY_SEQUENCE_LENGTH)
    logger.info("=" * 70)

    logger.info("[1/5] โหลด+ทำความสะอาด SHARP, flare catalog")
    sharp = clean_sharp_frame(pd.read_parquet(sharp_path), study_cfg)
    flares = pd.read_parquet(flares_path)
    logger.info("  flare catalog: %s (%d แถว)", flares_path, len(flares))
    flare_lookup = build_flare_lookup(flares, study_cfg.flare.positive_goes_class)
    flare_history = None
    if args.flare_history:
        # flare ทุกคลาสจาก catalog เดียวกับ label — HARP เดียวกันจับคู่ผ่านเลข NOAA AR ชุดเดียวกัน
        flare_history = {
            int(harpnum): (
                pd.to_datetime(group["peak_time"]).to_numpy(dtype="datetime64[ns]"),
                group["peak_flux"].to_numpy(dtype=np.float64),
            )
            for harpnum, group in flares.sort_values("peak_time").groupby("HARPNUM")
        }
        logger.info("  ประวัติ flare ราย HARP: %d ดวง", len(flare_history))

    logger.info("[2/5] โหลด intensity table ทั้งหมดที่มีอยู่")
    intensity_all = _find_intensity_tables(cfg.paths.artifacts, args.intensity_dir)

    xray_store = XrayFluxStore(cfg.paths.raw / "xrs")
    if not xray_store.available:
        logger.error("ไม่พบไฟล์ XRS ที่ %s — รัน backend/scripts/data/download_xray.py ก่อน", xray_store.root)
        return 1

    start, end = pd.Timestamp(args.main_start), pd.Timestamp(args.main_end)

    logger.info("[3/5] สร้าง unified sequence")
    sharp_window = sharp[(sharp["t_rec"] >= start) & (sharp["t_rec"] < end)]
    if sharp_window.empty:
        logger.error("ไม่มี SHARP ในช่วง %s..%s", start, end)
        return 1

    intensity_window = (
        pd.DataFrame() if intensity_all.empty
        else intensity_all[
            pd.to_datetime(intensity_all["issue_time"], format="%Y%m%d_%H%M%S").between(start, end)
        ]
    )
    xray_bins = xray_store.bin_series(start, end, cadence_hours=STUDY_CADENCE_HOURS)

    x, y, meta, feature_names = build_unified_window(
        sharp_window, flare_lookup, intensity_window, xray_bins, start, end, study_cfg,
        extra_intensity_channels=extra_channels, flare_history=flare_history,
    )
    if len(x) == 0:
        logger.error("ได้ 0 sample — รอ intensity ครบทั้งช่วงก่อน")
        return 1
    logger.info("  %d sample, positive %d (%.2f%%)", len(x), int(y.sum()), 100 * y.mean())

    if args.no_split:
        logger.info("[4/5] ข้าม split ตามที่ระบุ — เตรียม pool เต็มสำหรับ --cv")
        meta["split"] = "train"
    else:
        logger.info("[4/5] แบ่ง train/val/test แบบ HARP-disjoint (เหมือน production)")
        lifespans = harp_lifespans(sharp)
        assigned = assign_harp_splits(lifespans, study_cfg.split)
        meta_indexed = meta.reset_index(names="_row")
        meta_split = apply_splits(meta_indexed, assigned)
        keep_idx = meta_split["_row"].to_numpy()
        x, y = x[keep_idx], y[keep_idx]
        meta = meta_split.drop(columns="_row").reset_index(drop=True)
        verify_no_harp_overlap(meta)

    train_mask = (meta["split"] == "train").to_numpy()
    if not train_mask.any():
        logger.error(
            "ได้ %d sample แต่ 0 แถวอยู่ใน train split — คำนวณ normalisation stats ไม่ได้ "
            "(ยังไม่เขียนไฟล์ใด ๆ) ตรวจสอบว่า intensity ครอบคลุม HARP ของ train ตาม "
            "split.train_end/val_end ใน data.yaml หรือไม่",
            len(x),
        )
        return 1
    if args.no_split:
        logger.info(
            "หมายเหตุ: norm_stats.npz ที่กำลังจะเขียนคำนวณจาก pool ทั้งก้อน (ไม่ตัด "
            "val/test ออก) ใช้ตรงๆ กับ fold ไหนไม่ได้ — study/train.py --cv คำนวณสถิติใหม่ "
            "จาก train ของแต่ละ fold เองเสมอ ไฟล์นี้แค่ทำให้ loader อ่านไฟล์ได้ครบ"
        )

    logger.info("[5/5] เขียนผลลัพธ์")
    out_dir = Path(args.out_dir) if args.out_dir else cfg.paths.processed / "study_sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", x.astype(np.float32))
    np.save(out_dir / "y.npy", y.astype(np.uint8))
    meta.to_parquet(out_dir / "meta.parquet", index=False)

    stats = compute_normalisation(x, train_mask)
    np.savez(
        out_dir / "norm_stats.npz",
        mean=stats["mean"],
        std=stats["std"],
        features=np.array(feature_names, dtype=object),
        transform="signed_log1p",
    )

    balance = describe_split_balance(meta)

    report = {
        "flares_path": str(flares_path),
        "extra_aia_channels": list(extra_channels),
        "flare_history": bool(args.flare_history),
        "positive_goes_class": study_cfg.flare.positive_goes_class,
        "n_samples": int(len(x)),
        "n_features": len(feature_names),
        "features": feature_names,
        "cadence_hours": STUDY_CADENCE_HOURS,
        "sequence_length": STUDY_SEQUENCE_LENGTH,
        "balance_by_split": balance.to_dict(orient="records"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info("=" * 70)
    logger.info("เขียน %d sample (%d feature) ไปที่ %s", len(x), len(feature_names), out_dir)
    logger.info("สมดุลของแต่ละ split:\n%s", balance.to_string(index=False))
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
