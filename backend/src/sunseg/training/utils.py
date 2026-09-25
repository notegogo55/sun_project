"""ยูทิลิตี้ที่ใช้ร่วมกันระหว่างการเทรน LSTM และ U-Net"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """ตรึงค่าสุ่มของทุกไลบรารีเพื่อให้ผลลัพธ์ทำซ้ำได้

    ``deterministic=True`` จะบังคับให้ cuDNN ใช้อัลกอริทึมที่ผลลัพธ์คงที่ ซึ่งช้าลง
    พอสมควร จึงเปิดเฉพาะตอนต้องการเปรียบเทียบผลอย่างเคร่งครัด
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # ให้ cuDNN เลือกอัลกอริทึมที่เร็วที่สุดสำหรับขนาด input ที่คงที่
        torch.backends.cudnn.benchmark = True


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            "ใช้ GPU: %s (VRAM %.1f GB, compute %d.%d)",
            props.name,
            props.total_memory / 1024**3,
            props.major,
            props.minor,
        )
        return device

    logger.warning("ไม่พบ CUDA — เทรนบน CPU (ช้ากว่ามากสำหรับ U-Net)")
    return torch.device("cpu")


def cosine_warmup_lambda(total_epochs: int, warmup_epochs: int = 0, min_factor: float = 0.01):
    """ตัวคูณ learning rate: อุ่นเครื่องเชิงเส้น แล้วลดแบบ cosine

    ช่วง warmup จำเป็นเมื่อใช้ batch เล็กหรือข้อมูลไม่สมดุล เพราะ gradient ในช่วง
    ต้นมีความแปรปรวนสูงและอาจผลักโมเดลไปในทิศทางที่แย่อย่างถาวร
    """

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return lr_lambda


def build_scheduler(optimizer, cfg, total_epochs: int):
    """สร้าง learning-rate scheduler ตามที่ระบุใน config"""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            cosine_warmup_lambda(total_epochs, getattr(cfg, "warmup_epochs", 0)),
        )
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    return None


@dataclass
class EarlyStopping:
    """หยุดเทรนเมื่อตัวชี้วัดบน validation ไม่ดีขึ้นติดต่อกันหลาย epoch

    เก็บ ``best_state`` ไว้ในหน่วยความจำเสมอ เพื่อให้ย้อนกลับไปยัง checkpoint ที่ดี
    ที่สุดได้ แม้จะเทรนเลยจุดนั้นไปแล้ว
    """

    patience: int = 10
    mode: str = "max"
    min_delta: float = 1e-5

    best_score: float = field(init=False)
    best_epoch: int = field(default=-1, init=False)
    best_state: dict[str, Any] | None = field(default=None, init=False)
    counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.best_score = -math.inf if self.mode == "max" else math.inf

    def _improved(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def step(self, score: float, epoch: int, model: torch.nn.Module) -> bool:
        """อัปเดตสถานะ คืน ``True`` เมื่อควรหยุดเทรน"""
        if self._improved(score):
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: torch.nn.Module) -> None:
        if self.best_state is None:
            logger.warning("ไม่มี checkpoint ที่ดีที่สุดให้กู้คืน")
            return
        model.load_state_dict(self.best_state)
        logger.info(
            "กู้คืนน้ำหนักจาก epoch %d (คะแนน %.4f)", self.best_epoch + 1, self.best_score
        )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """บันทึกน้ำหนักพร้อม config และผลการวัด — ทำให้ checkpoint อธิบายตัวเองได้"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
        **(extra or {}),
    }
    torch.save(payload, path)
    logger.info("บันทึก checkpoint: %s (%.1f MB)", path, path.stat().st_size / 1024**2)


def save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """เขียนผลการวัดเป็น JSON ที่อ่านด้วยตาได้"""
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("บันทึกผลการวัด: %s", path)


def save_predictions(path: Path, df: pd.DataFrame) -> None:
    """บันทึกค่าทำนายราย sample เป็น parquet — ฐานข้อมูลของแผง confusion matrix

    เก็บแยกจาก metrics.json ที่มีแต่ตัวเลขสรุป เพื่อให้เอาค่าทำนายไปวิเคราะห์ต่อได้
    โดยไม่ต้องเทรนใหม่
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("บันทึกค่าทำนายราย sample: %s (%d แถว)", path, len(df))
