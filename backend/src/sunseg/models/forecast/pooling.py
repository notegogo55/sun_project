"""วิธียุบลำดับเวลาเป็นเวกเตอร์เดียว + หัวจำแนก — ใช้ร่วมกันทุกโมเดลพยากรณ์

แยกออกมาจาก ``lstm.py`` เพราะการเลือกวิธี pooling เป็น hyperparameter ของ **ทุก**
สถาปัตยกรรม ไม่ใช่ของ LSTM ตัวเดียว — ดู ``docs/adr/0001-tune-per-architecture.md``:
ถ้าปล่อยให้ LSTM ตัวเดียวทิ้ง attention ได้ (ซึ่งเป็นสิ่งที่การค้นหารอบแรกเลือกจริง ๆ)
แต่ตัวอื่นถูกบังคับให้ใช้ attention การเปรียบเทียบจะเอนไปทางเดียวอย่างเป็นระบบ

ทุกตัวคืนค่าเป็นคู่ ``(context, weights)`` รูปทรงเดียวกันเสมอ — ``weights`` มีไว้ให้รายงาน
และหน้าเว็บวาดได้โดยไม่ต้องรู้ว่าข้างในเป็น pooling แบบไหน ตัวที่ไม่มีน้ำหนักจริงคืนค่าที่
ตีความตรงตัวได้ (one-hot ที่ timestep สุดท้าย หรือค่าเท่ากันทุก timestep) ไม่ใช่ศูนย์ล้วน
ซึ่งอ่านผิดได้ง่ายว่า "โมเดลไม่สนใจอะไรเลย"
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: ชื่อวิธี pooling ที่ใช้ได้ — ตรงกับค่าที่ยอมรับใน ``configs/forecast.yaml`` และ
#: ``configs/study/architectures.yaml``
POOLING_KINDS: tuple[str, ...] = ("attention", "last", "mean")


class AttentionPooling(nn.Module):
    """ยุบลำดับเวลาเป็นเวกเตอร์เดียวด้วยผลรวมถ่วงน้ำหนักที่เรียนรู้ได้

    ดีกว่าการหยิบ hidden state ตัวสุดท้ายสองข้อ: (1) ไม่บังคับให้ข้อมูลทั้งหมดต้อง
    บีบผ่าน timestep สุดท้าย และ (2) น้ำหนักที่ได้ตีความได้ว่า "ชั่วโมงไหนสำคัญ"
    ซึ่งนำไปแสดงบน webapp ได้
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence: (B, L, H)
        scores = self.score(sequence)                    # (B, L, 1)
        weights = torch.softmax(scores, dim=1)           # (B, L, 1)
        context = (weights * sequence).sum(dim=1)        # (B, H)
        return context, weights.squeeze(-1)              # (B, H), (B, L)


class LastStepPooling(nn.Module):
    """หยิบ timestep สุดท้าย — ไม่มีพารามิเตอร์

    น้ำหนักที่คืนเป็น one-hot ที่ตำแหน่งสุดท้าย ซึ่งเป็นความจริงของสิ่งที่เกิดขึ้น
    (เดิม ``FlareLSTM`` คืนศูนย์ล้วนในกรณีนี้ ซึ่งอ่านผิดได้)
    """

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.zeros(sequence.shape[:2], device=sequence.device, dtype=sequence.dtype)
        weights[:, -1] = 1.0
        return sequence[:, -1, :], weights


class MeanPooling(nn.Module):
    """เฉลี่ยทุก timestep เท่ากัน — ไม่มีพารามิเตอร์"""

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = sequence.size(1)
        weights = sequence.new_full(sequence.shape[:2], 1.0 / length)
        return sequence.mean(dim=1), weights


def build_pooling(kind: str, hidden_size: int) -> nn.Module:
    if kind == "attention":
        return AttentionPooling(hidden_size)
    if kind == "last":
        return LastStepPooling()
    if kind == "mean":
        return MeanPooling()
    raise ValueError(f"pooling {kind!r} ไม่รู้จัก (ต้องเป็นหนึ่งใน {list(POOLING_KINDS)})")


def build_head(hidden_size: int, dropout: float) -> nn.Sequential:
    """หัวจำแนกที่ทุกสถาปัตยกรรมใช้ร่วมกัน — คืน logit หนึ่งค่าต่อ sample

    ใช้ตัวเดียวกันทุกสถาปัตยกรรมโดยตั้งใจ: สิ่งที่เปรียบเทียบคือวิธีอ่านลำดับเวลา
    ไม่ใช่วิธีแปลงเวกเตอร์สรุปเป็นความน่าจะเป็น ถ้าหัวต่างกันด้วยจะแยกไม่ออกว่าผลต่าง
    มาจากส่วนไหน
    """
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_size // 2, 1),
    )


def check_input(x: torch.Tensor, n_features: int) -> None:
    """ตรวจรูปทรง input ให้ทุกสถาปัตยกรรมพ่น error ข้อความเดียวกัน"""
    if x.dim() != 3:
        raise ValueError(f"คาดหวัง input 3 มิติ (B, L, F) แต่ได้รูปทรง {tuple(x.shape)}")
    if x.size(-1) != n_features:
        raise ValueError(f"จำนวน feature ไม่ตรง: โมเดลคาดหวัง {n_features} แต่ได้ {x.size(-1)}")
