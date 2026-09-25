"""วางแผน fold ของ rolling/blocked-window cross-validation ให้พร้อมใช้งาน

รวม logic ที่ต้องทำเหมือนกันทุกสคริปต์ที่รองรับ ``--cv`` (``study/train.py``,
``forecast/train.py``): สร้างขอบเขต fold, คำนวณ split ต่อแถวของแต่ละ fold, แล้วตัด fold ที่
val/test มี positive น้อยเกินไปทิ้งอัตโนมัติพร้อม log เหตุผล — เกณฑ์นี้อ้างจากประสบการณ์จริง
ตอนเลือกเส้นแบ่งของ single split (ดู comment ยาวใน ``configs/data.yaml`` ที่ ``split:``):
val ที่มี positive แค่ 40/24 ตัวเคยถูกตัดสินว่าเลือก threshold ไม่นิ่ง
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from ..config import DataConfig, SplitConfig
from .splits import assign_harp_splits, describe_split_balance, harp_lifespans, rolling_folds

logger = logging.getLogger(__name__)


@dataclass
class FoldPlan:
    """ผลวางแผน fold หนึ่ง — ``split_labels`` เรียงตรงกับแถวของ ``meta`` ที่ส่งเข้าไป"""

    index: int
    bounds: SplitConfig
    split_labels: pd.Series
    balance: pd.DataFrame
    valid: bool
    reason: str | None


def _count(balance: pd.DataFrame, split: str, column: str) -> int:
    by_split = balance.set_index("split")
    return int(by_split.loc[split, column]) if split in by_split.index else 0


def plan_folds(data_cfg: DataConfig, meta: pd.DataFrame, time_col: str = "issue_time") -> list[FoldPlan]:
    """สร้างแผน fold ทั้งหมดจาก ``data_cfg.split.cv`` พร้อมตัดสินแล้วว่า fold ไหนใช้ได้

    ``meta`` ต้องมีคอลัมน์ ``HARPNUM``, ``time_col``, ``label`` — มาจาก pool ที่สร้างด้วย
    ``--no-split`` (ไม่มี HARP ถูกทิ้งจากเส้นแบ่งเดียวมาก่อน ต่างจาก dataset ปกติ)
    """
    cv = data_cfg.split.cv
    if cv is None:
        raise ValueError("configs/data.yaml ไม่มี split.cv — ใส่พารามิเตอร์ก่อนใช้ --cv")

    lifespans = harp_lifespans(meta, time_col=time_col)
    if lifespans.empty:
        raise ValueError("ไม่มี HARP ให้วางแผน fold เลย")

    bounds_list = rolling_folds(data_cfg.split, cv, lifespans["t_first"].min(), lifespans["t_last"].max())
    plans: list[FoldPlan] = []
    for i, bounds in enumerate(bounds_list):
        assigned = assign_harp_splits(lifespans, bounds)
        split_labels = meta["HARPNUM"].map(assigned.set_index("HARPNUM")["split"]).fillna("drop")
        balance = describe_split_balance(meta.assign(split=split_labels))

        train_n, val_n, test_n = (_count(balance, s, "n_samples") for s in ("train", "val", "test"))
        train_pos = _count(balance, "train", "n_positive")
        val_pos = _count(balance, "val", "n_positive")
        test_pos = _count(balance, "test", "n_positive")

        reason = None
        if train_n == 0 or val_n == 0 or test_n == 0:
            reason = f"ว่างเปล่า (train={train_n}, val={val_n}, test={test_n})"
        elif train_pos == 0:
            reason = "train ไม่มี positive เลย"
        elif val_pos < cv.min_val_positive:
            reason = f"val มี positive {val_pos} ตัว < ขั้นต่ำ {cv.min_val_positive}"
        elif test_pos < cv.min_test_positive:
            reason = f"test มี positive {test_pos} ตัว < ขั้นต่ำ {cv.min_test_positive}"

        plans.append(
            FoldPlan(
                index=i,
                bounds=bounds,
                split_labels=split_labels,
                balance=balance,
                valid=reason is None,
                reason=reason,
            )
        )

    n_valid = sum(p.valid for p in plans)
    logger.info("วางแผน CV ได้ %d fold (ใช้ได้ %d, ตัดทิ้ง %d)", len(plans), n_valid, len(plans) - n_valid)
    for p in plans:
        status = "OK" if p.valid else f"SKIP ({p.reason})"
        logger.info(
            "  fold %d: train %s..%s · val ..%s · test ..%s -> %s",
            p.index, p.bounds.train_start, p.bounds.train_end, p.bounds.val_end, p.bounds.test_end, status,
        )
    if n_valid == 0:
        logger.error("ไม่มี fold ไหนผ่านเกณฑ์ขั้นต่ำเลย — ลองปรับ split.cv ใน configs/data.yaml")
    return plans
