"""โหลด dataset ของงานเปรียบเทียบชุด feature แล้วหั่นเป็นชุดข้อมูลของ "แบบ" หนึ่ง
(ticket 07 ของ ``.scratch/lstm-feature-ablation/``)

ต่อจาก ``sunseg.data.study_dataset`` ที่สร้าง ``X`` ก้อนเดียวเก็บทุกคอลัมน์ผู้สมัคร — โมดูลนี้
คือขั้น "เลือกคอลัมน์ตามชื่อ → normalise → แบ่ง split" ที่ทุกแบบผ่านเหมือนกันทุกประการ
แกนแถวไม่ถูกแตะเลย ทุกแบบจึงเห็น sample ชุดเดียวกันเรียงเหมือนกันโดยโครงสร้าง

**ทำไมหั่น mean/std ตามคอลัมน์ได้ตรง ๆ โดยไม่ต้องคำนวณใหม่**: ``compute_normalisation``
คำนวณ mean/std แยกอิสระต่อคอลัมน์ และ ``signed_log1p`` เป็นการแปลงรายตัว ค่าที่ normalise แล้ว
ของคอลัมน์หนึ่งจึงไม่ขึ้นกับว่ามีคอลัมน์อื่นอยู่ด้วยหรือไม่ — ทดสอบไว้ใน ``test_study_splits.py``
เพราะถ้าไม่จริง feature เดียวกันจะมีค่าไม่เท่ากันในแต่ละแบบ แล้วผลต่างจะไม่ได้มาจาก feature
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.build_sequences import apply_normalisation, compute_normalisation
from ..data.study_dataset import select_columns
from .sequence import SequenceDataset, SequenceSplits

logger = logging.getLogger(__name__)

REQUIRED_FILES: tuple[str, ...] = ("X.npy", "y.npy", "meta.parquet", "norm_stats.npz")


@dataclass
class StudyArrays:
    """dataset ก้อนเดียวของงานศึกษา — ``x`` เป็นค่าดิบ ``(N, L, F)`` ยังไม่ normalise"""

    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    stats: dict[str, np.ndarray]
    features: list[str]


def load_study_arrays(processed_dir: Path) -> StudyArrays:
    """โหลดสิ่งที่ ``backend/scripts/study/build_dataset.py`` เขียนไว้ พร้อมตรวจความสอดคล้อง"""
    processed_dir = Path(processed_dir)
    missing = [name for name in REQUIRED_FILES if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {missing} ใน {processed_dir}\nรัน backend/scripts/study/build_dataset.py ก่อน"
        )

    x = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    meta = pd.read_parquet(processed_dir / "meta.parquet")
    # features ถูกเก็บเป็น object array (ดู study/build_dataset.py) จึงต้อง allow_pickle
    npz = np.load(processed_dir / "norm_stats.npz", allow_pickle=True)
    features = [str(name) for name in npz["features"]]
    stats = {"mean": np.asarray(npz["mean"]), "std": np.asarray(npz["std"])}

    if not (len(x) == len(y) == len(meta)):
        raise ValueError(f"จำนวนแถวไม่ตรงกัน: X {len(x)} · y {len(y)} · meta {len(meta)}")
    if x.shape[-1] != len(features) or stats["mean"].shape != (len(features),):
        raise ValueError(
            f"จำนวน feature ไม่ตรงกัน: X {x.shape[-1]} · features {len(features)} · "
            f"mean {stats['mean'].shape}"
        )
    for column in ("split", "label", "HARPNUM", "issue_time"):
        if column not in meta.columns:
            raise ValueError(f"meta.parquet ไม่มีคอลัมน์ {column!r}")
    if not np.array_equal(meta["label"].to_numpy(), y):
        raise ValueError("meta['label'] ไม่ตรงกับ y.npy — meta กับอาร์เรย์เรียงไม่ตรงกัน")

    return StudyArrays(x=x, y=y, meta=meta, stats=stats, features=features)


def variant_splits(arrays: StudyArrays, columns: list[str]) -> SequenceSplits:
    """ชุดข้อมูล train/val/test ของแบบที่ใช้คอลัมน์ ``columns`` (ตามลำดับที่ขอ)

    ชื่อที่ไม่มีอยู่ทำให้ได้ ``KeyError`` ที่บอกชื่อนั้น (จาก :func:`select_columns`)
    """
    idx = select_columns(arrays.features, list(columns))
    stats = {"mean": arrays.stats["mean"][idx], "std": arrays.stats["std"][idx]}
    x = apply_normalisation(arrays.x[:, :, idx], stats)

    datasets = {}
    for name in ("train", "val", "test"):
        mask = (arrays.meta["split"] == name).to_numpy()
        datasets[name] = SequenceDataset(x[mask], arrays.y[mask])

    return SequenceSplits(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        meta=arrays.meta,
        stats=stats,
        features=list(columns),
    )


def fold_variant_splits(arrays: StudyArrays, columns: list[str], split_labels: pd.Series) -> SequenceSplits:
    """เหมือน :func:`variant_splits` แต่รับ ``split_labels`` ของ fold หนึ่งแทน
    ``arrays.meta["split"]`` — ใช้ตอน cross-validation ที่ split เปลี่ยนไปทุก fold

    normalisation stats ถูกคำนวณใหม่จาก **train ของ fold นี้เท่านั้น** (ไม่ใช้
    ``arrays.stats`` ซึ่งผูกกับ split เดียวของ pool ทั้งก้อน — ถ้าเผลอใช้ค่านั้นแทน
    เท่ากับ normalise ด้วยสถิติที่เห็น val/test ของ fold นี้แล้ว เป็น leakage ทันที)
    """
    idx = select_columns(arrays.features, list(columns))
    x_cols = arrays.x[:, :, idx]

    train_mask = (split_labels == "train").to_numpy()
    if not train_mask.any():
        raise ValueError("fold นี้ไม่มี sample ใน train — คำนวณ normalisation stats ไม่ได้")
    stats = compute_normalisation(x_cols, train_mask)
    x = apply_normalisation(x_cols, stats)

    datasets = {}
    for name in ("train", "val", "test"):
        mask = (split_labels == name).to_numpy()
        datasets[name] = SequenceDataset(x[mask], arrays.y[mask])

    meta = arrays.meta.assign(split=split_labels.to_numpy())
    return SequenceSplits(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        meta=meta,
        stats=stats,
        features=list(columns),
    )
