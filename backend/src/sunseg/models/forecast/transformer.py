"""Transformer encoder จิ๋วสำหรับพยากรณ์ flare จากลำดับเวลา

**ลักษณะเฉพาะ**: self-attention ให้ทุก timestep มองเห็นกันเอง
ทั้งหมดในชั้นเดียว ระยะทางระหว่างชั่วโมงที่ 1 กับชั่วโมงที่ 24 เท่ากับระหว่างชั่วโมงที่
23 กับ 24 — ต่างจาก LSTM ที่ข้อมูลต้องเดินผ่านสถานะทีละก้าว และต่างจาก TCN ที่ระยะ
ถูกจำกัดด้วย receptive field

**ต้องเล็กมาก**: effective sample size จริงคือจำนวน HARP ที่เคยปะทุ (~114 ดวง) ไม่ใช่
จำนวนแถว — เหตุผลเดียวกับที่ ``configs/forecast.yaml`` อธิบายไว้ ค่าปริยายที่นี่ (1 ชั้น,
2 head, d_model 32) ให้ราว 8k พารามิเตอร์ ใกล้เคียง LSTM ที่ระบบใช้อยู่

**pre-norm ไม่ใช่ post-norm**: ``norm_first=True`` ทำให้เทรนได้โดยไม่ต้องมี warmup
ยาว ๆ ซึ่งสำคัญเมื่อ epoch ทั้งหมดมีแค่ 60 และ early stopping ตัดที่ราว 38

**ความยาวลำดับคงที่**: positional embedding เป็นพารามิเตอร์ที่เรียนรู้ได้ จึงผูกกับ
``seq_len`` ต่างจาก ``FlareLSTM`` ที่รับความยาวเท่าใดก็ได้ — ลำดับที่ยาวกว่าที่ประกาศไว้
จะ raise แทนที่จะเงียบ ๆ ใช้ตำแหน่งผิด
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import build_head, build_pooling, check_input


class FlareTransformer(nn.Module):
    """input projection + positional embedding + encoder + pooling + หัวจำแนก"""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int = 32,
        n_heads: int = 2,
        ff_dim: int = 16,
        n_layers: int = 1,
        dropout: float = 0.3,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} ต้องหารด้วยจำนวน head {n_heads} ลงตัว")

        self.n_features = n_features
        self.seq_len = seq_len

        self.project = nn.Linear(n_features, d_model)
        # เริ่มที่ศูนย์: ชั้นแรกจึงเห็นลำดับแบบไม่มีอคติเรื่องตำแหน่ง แล้วค่อยเรียนว่า
        # ชั่วโมงไหนต่างกันอย่างไร — ดีกว่าสุ่มค่าเริ่มต้นเมื่อข้อมูลน้อย
        self.positions = nn.Parameter(torch.zeros(1, seq_len, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor ใช้กับ norm_first ไม่ได้อยู่แล้ว — ปิดไว้เพื่อไม่ให้เตือนทุกครั้งที่สร้าง
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

        self.pool = build_pooling(pooling, d_model)
        self.head = build_head(d_model, dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        check_input(x, self.n_features)
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"ความยาวลำดับไม่ตรง: โมเดลผูกกับ {self.seq_len} timestep (positional embedding) "
                f"แต่ได้ {x.size(1)}"
            )
        sequence = self.encoder(self.project(x) + self.positions)   # (B, L, d_model)
        context, attention = self.pool(sequence)
        logits = self.head(context).squeeze(-1)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
