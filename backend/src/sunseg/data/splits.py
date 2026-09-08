"""แบ่งข้อมูลเป็น train/val/test แบบกัน data leakage

นี่คือจุดที่งาน flare forecasting จำนวนมากทำพลาด และเป็นสาเหตุที่ตัวเลขในบางเปเปอร์
สูงเกินจริง มีสองแหล่ง leakage ที่ต้องปิดพร้อมกัน:

**1. Autocorrelation ภายใน HARP เดียวกัน**
    SHARP parameters ของ AR ดวงหนึ่งที่ห่างกัน 1 ชั่วโมงแทบจะเหมือนกันทุกประการ
    ถ้าสุ่มแบ่ง sample โมเดลจะเจอ "AR ดวงเดียวกัน คนละชั่วโมง" ทั้งใน train และ test
    ซึ่งเท่ากับจำคำตอบได้ → เราจึง assign **ทั้ง HARP** ไปยัง split เดียวเท่านั้น

**2. การมองอนาคต (temporal leakage)**
    label ของ sample ที่เวลา t มองไปข้างหน้า 24 ชม. ดังนั้น sample สุดท้ายของ train
    จึงรู้เรื่องที่เกิดหลัง train_end → เราเว้น **ช่วง gap** ระหว่าง split และทิ้ง HARP
    ที่คร่อมเส้นแบ่งทั้งดวง

ผลคือเสียข้อมูลไปเล็กน้อย (HARP มีอายุราว 2 สัปดาห์ ส่วน split ยาวเป็นปี) แต่ได้
ตัวเลขที่เชื่อถือได้
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from ..config import SplitConfig

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def harp_lifespans(df: pd.DataFrame, time_col: str = "t_rec") -> pd.DataFrame:
    """หาเวลาที่ HARP แต่ละดวงปรากฏครั้งแรกและครั้งสุดท้าย"""
    if df.empty:
        return pd.DataFrame(columns=["HARPNUM", "t_first", "t_last", "n_obs"])

    grouped = df.groupby("HARPNUM")[time_col].agg(["min", "max", "count"])
    return (
        grouped.rename(columns={"min": "t_first", "max": "t_last", "count": "n_obs"})
        .reset_index()
        .sort_values("t_first")
        .reset_index(drop=True)
    )


def assign_harp_splits(lifespans: pd.DataFrame, cfg: SplitConfig) -> pd.DataFrame:
    """กำหนด split ให้ HARP แต่ละดวง — HARP ที่คร่อมเส้นแบ่งจะได้ ``"drop"``

    Returns
    -------
    ``lifespans`` เดิม บวกคอลัมน์ ``split`` ที่มีค่า train/val/test/drop
    """
    if lifespans.empty:
        return lifespans.assign(split=pd.Series(dtype="object"))

    gap = timedelta(days=cfg.gap_days)
    train_end = pd.Timestamp(cfg.train_end) + timedelta(days=1)  # inclusive ถึงสิ้นวัน
    val_end = pd.Timestamp(cfg.val_end) + timedelta(days=1)

    out = lifespans.copy()
    t_first = pd.to_datetime(out["t_first"])
    t_last = pd.to_datetime(out["t_last"])

    is_train = t_last <= train_end
    is_val = (t_first >= train_end + gap) & (t_last <= val_end)
    is_test = t_first >= val_end + gap

    out["split"] = "drop"
    out.loc[is_train, "split"] = "train"
    out.loc[is_val, "split"] = "val"
    out.loc[is_test, "split"] = "test"

    _log_split_summary(out)
    _assert_disjoint(out)
    return out


def _log_split_summary(assigned: pd.DataFrame) -> None:
    counts = assigned["split"].value_counts()
    total = len(assigned)
    logger.info("แบ่ง HARP ทั้งหมด %d ดวง:", total)
    for name in (*SPLIT_NAMES, "drop"):
        n = int(counts.get(name, 0))
        subset = assigned[assigned["split"] == name]
        if n and name != "drop":
            logger.info(
                "  %-5s : %4d HARP (%5.1f%%)  %s ถึง %s",
                name,
                n,
                100 * n / total,
                pd.to_datetime(subset["t_first"]).min().date(),
                pd.to_datetime(subset["t_last"]).max().date(),
            )
        else:
            logger.info("  %-5s : %4d HARP (%5.1f%%)  [คร่อมเส้นแบ่ง — ทิ้ง]", name, n, 100 * n / total)


def _assert_disjoint(assigned: pd.DataFrame) -> None:
    """ยืนยันว่าไม่มี HARP ดวงใดถูก assign ไปมากกว่าหนึ่ง split"""
    dupes = assigned[assigned.duplicated("HARPNUM", keep=False)]
    if not dupes.empty:
        raise AssertionError(
            f"HARP ต่อไปนี้ถูก assign ซ้ำมากกว่าหนึ่ง split: "
            f"{sorted(dupes['HARPNUM'].unique().tolist())}"
        )


def apply_splits(df: pd.DataFrame, assigned: pd.DataFrame) -> pd.DataFrame:
    """ติดคอลัมน์ ``split`` ให้ทุกแถวตาม HARPNUM แล้วตัดแถวที่ถูก drop ออก"""
    mapping = assigned.set_index("HARPNUM")["split"]
    out = df.copy()
    out["split"] = out["HARPNUM"].map(mapping).fillna("drop")

    before = len(out)
    out = out[out["split"].isin(SPLIT_NAMES)].reset_index(drop=True)
    logger.info("คัดแถวตาม split: %d -> %d แถว", before, len(out))
    return out


def verify_no_harp_overlap(df: pd.DataFrame) -> None:
    """ตรวจซ้ำอีกชั้นบนตารางผลลัพธ์สุดท้าย — ต้องไม่มี HARP ปรากฏในสอง split"""
    per_split = df.groupby("split")["HARPNUM"].apply(set)
    names = list(per_split.index)

    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = per_split[a] & per_split[b]
            if overlap:
                raise AssertionError(
                    f"ตรวจพบ data leakage: HARP {sorted(overlap)} "
                    f"ปรากฏทั้งใน split '{a}' และ '{b}'"
                )
    logger.info("ตรวจสอบแล้ว: ไม่มี HARP ซ้ำข้าม split")


def describe_split_balance(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """สรุปจำนวน sample และสัดส่วน positive ของแต่ละ split"""
    rows = []
    for name in SPLIT_NAMES:
        subset = df[df["split"] == name]
        if subset.empty:
            rows.append({"split": name, "n_samples": 0, "n_harp": 0, "n_positive": 0, "pos_rate": 0.0})
            continue
        n_pos = int(subset[label_col].sum())
        rows.append(
            {
                "split": name,
                "n_samples": len(subset),
                "n_harp": subset["HARPNUM"].nunique(),
                "n_positive": n_pos,
                "pos_rate": n_pos / len(subset),
            }
        )
    return pd.DataFrame(rows)


def default_boundaries(start: date, end: date, val_frac: float = 0.15, test_frac: float = 0.15):
    """คำนวณเส้นแบ่งเวลาเริ่มต้นจากสัดส่วน (ใช้เมื่อไม่ได้กำหนดวันที่ตายตัวใน config)"""
    span_days = (end - start).days
    train_days = int(span_days * (1 - val_frac - test_frac))
    val_days = int(span_days * val_frac)
    return start + timedelta(days=train_days), start + timedelta(days=train_days + val_days)
