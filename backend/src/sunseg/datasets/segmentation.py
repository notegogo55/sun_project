"""PyTorch Dataset สำหรับ segmentation ของ active region

ต่างจาก sequence dataset ตรงที่ภาพมีขนาดใหญ่เกินกว่าจะเก็บในหน่วยความจำทั้งหมด
(หลายพันภาพ x 512x512) จึงอ่านจากดิสก์ทีละภาพตอนเทรน ไฟล์ ``.npy`` แต่ละไฟล์
มีขนาดเพียง ~512 KB จึงอ่านได้เร็วพอ
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import AugmentConfig
from ..models.unet import normalise_magnetogram

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """คู่ (magnetogram, mask) ที่อ่านจากดิสก์

    Parameters
    ----------
    timestamps
        รายชื่อเฟรม (ชื่อไฟล์ไม่รวมนามสกุล) ที่อยู่ใน split นี้
    augment
        การเพิ่มความหลากหลายของข้อมูล — ใช้เฉพาะกับ train เท่านั้น
    """

    def __init__(
        self,
        frames_dir: Path,
        timestamps: list[str],
        norm_scale: float = 300.0,
        augment: AugmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.image_dir = Path(frames_dir) / "images"
        self.mask_dir = Path(frames_dir) / "masks"
        self.timestamps = list(timestamps)
        self.norm_scale = norm_scale
        self.augment = augment
        self._rng = np.random.default_rng(seed)

        if not self.timestamps:
            raise ValueError(f"ไม่มีเฟรมใน {frames_dir}")

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        timestamp = self.timestamps[index]

        image = np.load(self.image_dir / f"{timestamp}.npy").astype(np.float32)
        mask = np.load(self.mask_dir / f"{timestamp}.npy").astype(np.float32)
        image = np.nan_to_num(image, nan=0.0)

        if self.augment is not None:
            image, mask = self._apply_augment(image, mask)

        # normalise หลัง augment เสมอ — การหมุน/พลิกทำงานกับค่าดิบได้ตรงไปตรงมากว่า
        image = normalise_magnetogram(image, self.norm_scale)

        return (
            torch.from_numpy(np.ascontiguousarray(image))[None],  # (1, H, W)
            torch.from_numpy(np.ascontiguousarray(mask))[None],
        )

    def _apply_augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """เพิ่มความหลากหลายของข้อมูลโดยไม่ทำลายความหมายทางฟิสิกส์

        .. important::
           **ห้ามกลับเครื่องหมายของสนามแม่เหล็ก** ขั้วบวก/ลบไม่ใช่แค่สีในภาพ แต่เป็น
           ปริมาณทางฟิสิกส์ที่กำหนดโครงสร้างของ active region การพลิกเครื่องหมายจะ
           สอนให้โมเดลเรียนรู้สิ่งที่ไม่มีอยู่จริงในธรรมชาติ
        """
        cfg = self.augment
        assert cfg is not None

        if cfg.hflip and self._rng.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if cfg.vflip and self._rng.random() < 0.5:
            image, mask = image[::-1, :], mask[::-1, :]

        if cfg.rotate_deg > 0 and self._rng.random() < 0.5:
            import cv2

            angle = float(self._rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
            centre = (image.shape[1] / 2, image.shape[0] / 2)
            matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            size = (image.shape[1], image.shape[0])

            image = cv2.warpAffine(
                np.ascontiguousarray(image), matrix, size, flags=cv2.INTER_LINEAR, borderValue=0.0
            )
            # mask ใช้ nearest เพื่อให้ยังเป็นค่า 0/1 ไม่เกิดค่ากลาง
            mask = cv2.warpAffine(
                np.ascontiguousarray(mask), matrix, size, flags=cv2.INTER_NEAREST, borderValue=0.0
            )

        if cfg.intensity_jitter > 0:
            # จำลองความไม่แน่นอนของการปรับเทียบเครื่องมือ — คูณด้วยตัวเลขใกล้ 1
            scale = 1.0 + float(self._rng.uniform(-cfg.intensity_jitter, cfg.intensity_jitter))
            image = image * scale

        return np.ascontiguousarray(image), np.ascontiguousarray(mask)

    def positive_fraction(self, sample_size: int = 50) -> float:
        """สัดส่วนพิกเซลที่เป็น AR โดยประมาณ — ใช้ตั้งค่า ``pos_weight`` ของ loss"""
        indices = self._rng.choice(
            len(self.timestamps), size=min(sample_size, len(self.timestamps)), replace=False
        )
        fractions = [
            float(np.load(self.mask_dir / f"{self.timestamps[i]}.npy").mean()) for i in indices
        ]
        return float(np.mean(fractions))


def split_frames_by_time(
    timestamps: list[str], train_end: str, val_end: str, gap_days: int = 14
) -> dict[str, list[str]]:
    """แบ่งเฟรมตามเวลา พร้อมเว้นช่วง gap ระหว่าง split

    ใช้หลักการเดียวกับการแบ่ง sequence: เฟรมที่อยู่ในช่วง gap ถูกทิ้ง เพื่อไม่ให้
    active region ดวงเดียวกันปรากฏคาบเกี่ยวสอง split (AR หนึ่งดวงอยู่บนจานได้ราว
    2 สัปดาห์)
    """
    from datetime import datetime, timedelta

    def parse(name: str) -> datetime:
        return datetime.strptime(name, "%Y%m%d_%H%M%S")

    train_boundary = datetime.fromisoformat(train_end) + timedelta(days=1)
    val_boundary = datetime.fromisoformat(val_end) + timedelta(days=1)
    gap = timedelta(days=gap_days)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": [], "drop": []}
    for name in sorted(timestamps):
        moment = parse(name)
        if moment <= train_boundary:
            splits["train"].append(name)
        elif moment >= train_boundary + gap and moment <= val_boundary:
            splits["val"].append(name)
        elif moment >= val_boundary + gap:
            splits["test"].append(name)
        else:
            splits["drop"].append(name)

    logger.info(
        "แบ่งเฟรม: train %d, val %d, test %d (ทิ้งในช่วง gap %d)",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
        len(splits["drop"]),
    )
    return splits


def make_segmentation_loaders(
    frames_dir: Path,
    splits: dict[str, list[str]],
    batch_size: int,
    norm_scale: float,
    augment: AugmentConfig,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """สร้าง DataLoader ทั้งสามชุด (augment เฉพาะ train)"""
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_ds = SegmentationDataset(frames_dir, splits["train"], norm_scale, augment, seed)
    val_ds = SegmentationDataset(frames_dir, splits["val"], norm_scale, None, seed)
    test_ds = SegmentationDataset(frames_dir, splits["test"], norm_scale, None, seed)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common),
    )
