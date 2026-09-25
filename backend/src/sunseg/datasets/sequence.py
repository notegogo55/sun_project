"""PyTorch Dataset สำหรับ sequence ของ SHARP parameters

ชุดข้อมูลทั้งหมดมีขนาดเพียงไม่กี่ร้อย MB จึงโหลดเข้าหน่วยความจำทั้งก้อนได้ ทำให้
เทรนเร็วมาก (ไม่ต้องรอ I/O) ต่างจาก dataset ของ U-Net ที่ต้องอ่านจากดิสก์
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.build_sequences import apply_normalisation, compute_normalisation

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """ห่อ ``(X, y)`` ที่ normalise แล้วให้เป็น PyTorch Dataset"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        if len(x) != len(y):
            raise ValueError(f"จำนวน sample ไม่ตรงกัน: X มี {len(x)} แต่ y มี {len(y)}")

        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean())

    @property
    def n_features(self) -> int:
        return int(self.x.shape[-1])


@dataclass
class SequenceSplits:
    """ชุดข้อมูลที่แบ่งแล้ว พร้อมข้อมูลประกอบที่ใช้ตอนเทรนและ inference"""

    train: SequenceDataset
    val: SequenceDataset
    test: SequenceDataset
    meta: pd.DataFrame
    stats: dict[str, np.ndarray]
    features: list[str]

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def seq_len(self) -> int:
        """ความยาวหน้าต่างเวลา — สถาปัตยกรรมที่มีพารามิเตอร์ผูกกับตำแหน่ง
        (Transformer, DA-RNN) ต้องรู้ค่านี้ตอนสร้าง"""
        return int(self.train.x.shape[1])

    def summary(self) -> str:
        rows = []
        for name in ("train", "val", "test"):
            ds: SequenceDataset = getattr(self, name)
            n_pos = int(ds.y.sum())
            rows.append(
                f"  {name:<6} {len(ds):>8,} sample   positive {n_pos:>6,} "
                f"({100 * ds.positive_rate:5.2f}%)"
            )
        return "\n".join(rows)


def load_sequence_splits(processed_dir: Path) -> SequenceSplits:
    """โหลด dataset ที่ ``backend/scripts/data/build_sequences.py`` สร้างไว้ แล้ว normalise

    การ normalise ทำที่นี่ (ไม่ใช่ตอน build) เพื่อให้ไฟล์ ``X.npy`` เก็บค่าดิบไว้
    ตรวจสอบย้อนกลับได้ และเพื่อให้เปลี่ยนวิธี normalise ได้โดยไม่ต้องสร้าง dataset ใหม่
    """
    required = ["X.npy", "y.npy", "meta.parquet", "norm_stats.npz"]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {missing} ใน {processed_dir}\n"
            "รัน backend/scripts/data/build_sequences.py ก่อน"
        )

    x = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    meta = pd.read_parquet(processed_dir / "meta.parquet")
    npz = np.load(processed_dir / "norm_stats.npz", allow_pickle=False)

    stats = {"mean": npz["mean"], "std": npz["std"]}
    features = [str(f) for f in npz["features"]]

    transform = str(npz["transform"]) if "transform" in npz else "unknown"
    if transform != "signed_log1p":
        logger.warning(
            "norm_stats.npz ระบุการแปลงเป็น %r แต่โค้ดปัจจุบันใช้ signed_log1p — "
            "ควรสร้าง dataset ใหม่เพื่อให้สอดคล้องกัน",
            transform,
        )

    if len(meta) != len(x):
        raise ValueError(f"meta มี {len(meta)} แถว แต่ X มี {len(x)} sample")

    logger.info("normalise ข้อมูลด้วยสถิติจาก train set")
    x_norm = apply_normalisation(x, stats)

    datasets = {}
    for name in ("train", "val", "test"):
        mask = (meta["split"] == name).to_numpy()
        datasets[name] = SequenceDataset(x_norm[mask], y[mask])

    splits = SequenceSplits(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        meta=meta,
        stats=stats,
        features=features,
    )
    logger.info("โหลด dataset สำเร็จ:\n%s", splits.summary())
    return splits


@dataclass
class SequencePool:
    """dataset ก้อนเดียวที่ยังไม่แบ่ง split — ``x`` เป็นค่าดิบ ``(N, L, F)`` ยังไม่ normalise

    มาจาก ``build_sequences.py --no-split`` (ไม่มี HARP ถูกทิ้งเพราะคร่อมเส้นแบ่งใดๆ) ใช้ตอน
    cross-validation ที่ต้องลองหลายขอบเขตเวลา — ต่างจาก :func:`load_sequence_splits` ที่ split
    ถูก freeze มาจากตอนสร้างไฟล์แล้ว
    """

    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    features: list[str]


def load_sequence_pool(processed_dir: Path) -> SequencePool:
    """โหลด pool เต็มที่ ``build_sequences.py --no-split`` เขียนไว้ — ไม่ normalise ที่นี่

    ไม่อ่านค่าใน ``norm_stats.npz`` เพราะมันถูกคำนวณจาก pool ทั้งก้อน (ไม่ตัด val/test ออก)
    ใช้ไม่ได้กับ fold ไหนเลย — normalisation ของแต่ละ fold ต้องคำนวณใหม่จาก train ของ fold
    นั้นเท่านั้น (ดู :func:`fold_sequence_splits`)
    """
    required = ["X.npy", "y.npy", "meta.parquet", "norm_stats.npz"]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ {missing} ใน {processed_dir}\n"
            "รัน backend/scripts/data/build_sequences.py --no-split --out-dir <dir นี้> ก่อน"
        )

    x = np.load(processed_dir / "X.npy")
    y = np.load(processed_dir / "y.npy")
    meta = pd.read_parquet(processed_dir / "meta.parquet")
    npz = np.load(processed_dir / "norm_stats.npz", allow_pickle=False)
    features = [str(f) for f in npz["features"]]

    if len(meta) != len(x) or len(y) != len(x):
        raise ValueError(f"จำนวนแถวไม่ตรงกัน: X {len(x)} · y {len(y)} · meta {len(meta)}")
    return SequencePool(x=x, y=y, meta=meta, features=features)


def fold_sequence_splits(pool: SequencePool, split_labels: pd.Series) -> SequenceSplits:
    """เหมือน :func:`load_sequence_splits` แต่รับ ``split_labels`` ของ fold หนึ่งแทน split ที่
    freeze ไว้ในไฟล์ — normalisation คำนวณใหม่จาก train ของ fold นี้เท่านั้น (สำคัญ: ใช้สถิติ
    จากไฟล์ pool ตรงๆ แทนจะเป็น leakage เพราะสถิตินั้นเห็น val/test ของทุก fold ปนอยู่แล้ว)
    """
    train_mask = (split_labels == "train").to_numpy()
    if not train_mask.any():
        raise ValueError("fold นี้ไม่มี sample ใน train — คำนวณ normalisation stats ไม่ได้")
    stats = compute_normalisation(pool.x, train_mask)
    x_norm = apply_normalisation(pool.x, stats)

    datasets = {}
    for name in ("train", "val", "test"):
        mask = (split_labels == name).to_numpy()
        datasets[name] = SequenceDataset(x_norm[mask], pool.y[mask])

    meta = pool.meta.assign(split=split_labels.to_numpy())
    return SequenceSplits(
        train=datasets["train"],
        val=datasets["val"],
        test=datasets["test"],
        meta=meta,
        stats=stats,
        features=pool.features,
    )


def make_loaders(
    splits: SequenceSplits,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """สร้าง DataLoader ทั้งสามชุด

    ใช้ ``num_workers=0`` เป็นค่าเริ่มต้นเพราะข้อมูลอยู่ในหน่วยความจำแล้ว การแยก
    process ย่อยมีแต่จะเพิ่ม overhead (และบน Windows การ spawn process แพงมาก)
    """
    common = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available()}

    return (
        DataLoader(splits.train, batch_size=batch_size, shuffle=True, drop_last=False, **common),
        DataLoader(splits.val, batch_size=batch_size * 4, shuffle=False, **common),
        DataLoader(splits.test, batch_size=batch_size * 4, shuffle=False, **common),
    )
