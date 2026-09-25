"""ฟังก์ชัน loss ที่รับมือกับข้อมูลไม่สมดุลอย่างรุนแรง

ทั้งสองงานในโปรเจคนี้มี positive class เป็นส่วนน้อยมาก:

* **segmentation** — active region กินพื้นที่เพียง ~1-3% ของ pixel บนจานสุริยะ
* **forecasting** — หน้าต่าง 24 ชม. ที่เกิด flare M+ มีเพียง ~1-3% ของ sample

``BCELoss`` ธรรมดาจะลู่เข้าสู่คำตอบ "ทายลบทุกครั้ง" ซึ่งได้ loss ต่ำแต่ไร้ประโยชน์
โมดูลนี้จึงมี loss ที่ถ่วงน้ำหนักหรือปรับ gradient ให้เน้น positive
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) — ใช้กับ LSTM forecasting

    ลดน้ำหนักของตัวอย่างที่โมเดลทายถูกอยู่แล้วด้วยตัวคูณ ``(1-p_t)^gamma`` ทำให้
    gradient ไปกระจุกที่ตัวอย่างยาก แทนที่จะถูกกลบด้วย negative ที่ง่ายเป็นแสนตัว

    Parameters
    ----------
    alpha
        น้ำหนักของ positive class (0.75 = ให้ความสำคัญ positive มากกว่า negative 3 เท่า)
    gamma
        ยิ่งสูงยิ่งกดตัวอย่างง่าย — 2.0 เป็นค่ามาตรฐานที่ใช้ได้ผลดี
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha ต้องอยู่ระหว่าง 0 ถึง 1 ได้รับ {alpha}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).to(logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = ความน่าจะเป็นที่โมเดลให้กับ "คลาสที่ถูกต้อง" — คำนวณจาก bce เพื่อความเสถียรเชิงตัวเลข
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """1 - Dice coefficient — ใช้กับ segmentation

    ปรับ overlap ของทั้งภาพโดยตรง จึงไม่สนใจว่า background จะมีกี่ pixel ต่างจาก
    BCE ที่รวม loss ของทุก pixel เท่าๆ กัน (แล้วถูก background กลบ)
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # ยุบทุกมิติยกเว้น batch เพื่อให้ Dice คิดแยกรายภาพแล้วค่อยเฉลี่ย
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).to(probs.dtype)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class TverskyLoss(nn.Module):
    """Tversky loss — Dice เวอร์ชันที่ปรับสมดุล FP/FN ได้

    ตั้ง ``beta > alpha`` เพื่อลงโทษ false negative หนักกว่า ซึ่งเหมาะเมื่อ "พลาด
    active region ไปทั้งดวง" แย่กว่า "ระบายเกินขอบไปหน่อย"
    ``alpha = beta = 0.5`` จะได้ Dice loss พอดี
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).reshape(logits.size(0), -1)
        targets = targets.reshape(targets.size(0), -1).to(probs.dtype)

        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


class BceDiceLoss(nn.Module):
    """BCE ถ่วงน้ำหนัก + Dice — loss เริ่มต้นของ U-Net ในโปรเจคนี้

    ทั้งสองส่วนเสริมกัน: BCE ให้ gradient ที่ชัดเจนรายพิกเซล (ช่วยตอนเริ่มเทรน)
    ส่วน Dice ปรับรูปร่างของ mask โดยรวม (ช่วยตอนใกล้ลู่เข้า) การใช้อย่างเดียว
    อย่างใดอย่างหนึ่งมักได้ผลด้อยกว่าอย่างเห็นได้ชัด
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = 20.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()
        # pos_weight คูณเข้ากับพจน์ของ positive ใน BCE — ชดเชยที่ AR มีพิกเซลน้อย
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight) if pos_weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(logits.dtype)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(logits.device, logits.dtype)

        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


def build_forecast_loss(cfg) -> nn.Module:
    """สร้าง loss ของ LSTM จาก ``configs/forecast.yaml``"""
    if cfg.type == "focal":
        return FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    if cfg.type == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"ไม่รู้จัก loss ชนิด {cfg.type!r}")


def build_segmentation_loss(cfg) -> nn.Module:
    """สร้าง loss ของ U-Net จาก ``configs/unet.yaml``"""
    return BceDiceLoss(
        bce_weight=cfg.bce_weight,
        dice_weight=cfg.dice_weight,
        pos_weight=cfg.pos_weight,
    )
