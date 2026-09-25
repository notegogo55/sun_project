"""ทะเบียนโมเดลพยากรณ์ — จุดเดียวที่รู้จักชื่อสถาปัตยกรรมทั้งหมด

เพิ่มสถาปัตยกรรมใหม่ = เพิ่มไฟล์โมเดลหนึ่งไฟล์ + หนึ่งรายการใน :data:`ARCHITECTURES`
+ หนึ่ง config class ใน ``sunseg.config`` แล้วทุกอย่างที่เหลือ (สคริปต์เทรน, inference,
หน้าเว็บ, งานเปรียบเทียบ, รายงาน) ตามมาเอง

**ไม่ import optuna ที่นี่โดยตั้งใจ** — ขอบเขตการค้นหาของแต่ละสถาปัตยกรรมอยู่ใน
``backend/scripts/study/tune.py`` เพราะเป็นเรื่องของการทดลอง ไม่ใช่ของไลบรารี
และ ``sunseg`` ที่ production ใช้ไม่ควรลากไลบรารีสำหรับ tune ติดไปด้วย

**ทุกตัวรับ ``seq_len``** แม้ ``FlareLSTM`` จะไม่ต้องใช้ (รับความยาวเท่าใดก็ได้) เพื่อให้
ผู้เรียกไม่ต้องรู้ว่าตัวไหนผูกกับความยาว — ``FlareTransformer`` และ ``FlareDARNN`` ผูก
เพราะมีพารามิเตอร์ที่ผูกกับตำแหน่ง
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from .darnn import FlareDARNN
from .lstm import FlareLSTM
from .tcn import FlareTCN
from .transformer import FlareTransformer


@dataclass(frozen=True)
class Architecture:
    """สิ่งที่ระบบต้องรู้เกี่ยวกับสถาปัตยกรรมหนึ่งตัว

    ``label`` คือชื่อที่ไปปรากฏในตารางและรูปของรายงาน — เก็บไว้ที่นี่เพื่อให้ชื่อใน
    ผลลัพธ์ตรงกันทุกที่โดยไม่ต้องประกาศซ้ำใน YAML
    """

    kind: str
    label: str
    build: Callable[..., nn.Module]
    note: str


def _build_lstm(n_features: int, seq_len: int, cfg) -> FlareLSTM:
    del seq_len  # LSTM รับความยาวลำดับเท่าใดก็ได้
    return FlareLSTM(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=False,
        pooling=cfg.pooling,
    )


def _build_tcn(n_features: int, seq_len: int, cfg) -> FlareTCN:
    return FlareTCN(
        n_features=n_features,
        seq_len=seq_len,
        channels=cfg.channels,
        kernel_size=cfg.kernel_size,
        dilations=cfg.dilations,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )


def _build_transformer(n_features: int, seq_len: int, cfg) -> FlareTransformer:
    return FlareTransformer(
        n_features=n_features,
        seq_len=seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        ff_dim=cfg.ff_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )


def _build_darnn(n_features: int, seq_len: int, cfg) -> FlareDARNN:
    return FlareDARNN(
        n_features=n_features,
        seq_len=seq_len,
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
    )


ARCHITECTURES: dict[str, Architecture] = {
    "lstm": Architecture(
        kind="lstm",
        label="LSTM",
        build=_build_lstm,
        note="สถานะส่งต่อทีละก้าว — สถาปัตยกรรมอ้างอิงของทั้งตาราง",
    ),
    "tcn": Architecture(
        kind="tcn",
        label="TCN",
        build=_build_tcn,
        note="convolution causal ที่ขยาย dilation — ไม่มีสถานะ เห็นทั้งหน้าต่างพร้อมกัน",
    ),
    "transformer": Architecture(
        kind="transformer",
        label="Transformer",
        build=_build_transformer,
        note="self-attention — ระยะระหว่างทุกคู่ timestep เท่ากันหมด",
    ),
    "darnn": Architecture(
        kind="darnn",
        label="DA-RNN",
        build=_build_darnn,
        note="LSTM ที่ถ่วงน้ำหนักราย feature — ตอบได้ว่าพารามิเตอร์ตัวไหนถูกใช้",
    ),
}


def build_architecture(kind: str, n_features: int, seq_len: int, cfg) -> nn.Module:
    """ประกอบโมเดลหนึ่งตัวจากชื่อสถาปัตยกรรมและ config ของมัน"""
    if kind not in ARCHITECTURES:
        raise ValueError(f"สถาปัตยกรรม {kind!r} ไม่รู้จัก (ที่มี: {sorted(ARCHITECTURES)})")
    return ARCHITECTURES[kind].build(n_features, seq_len, cfg)


def architecture_label(kind: str) -> str:
    if kind not in ARCHITECTURES:
        raise ValueError(f"สถาปัตยกรรม {kind!r} ไม่รู้จัก (ที่มี: {sorted(ARCHITECTURES)})")
    return ARCHITECTURES[kind].label
