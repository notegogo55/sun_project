"""TCN — convolution แบบ causal ที่ขยาย dilation สำหรับพยากรณ์ flare จากลำดับเวลา

**ลักษณะเฉพาะ**: LSTM อ่านลำดับด้วยสถานะที่ส่งต่อทีละก้าว
ส่วน TCN อ่านด้วยหน้าต่างคงที่ที่กว้างขึ้นเรื่อย ๆ ตามชั้น — ไม่มีสถานะ ไม่มีการลืม
และเห็นทุก timestep ในหน้าต่างพร้อมกัน เป็น inductive bias คนละแบบจริง ๆ ไม่ใช่
LSTM ที่เปลี่ยนชื่อ (ซึ่งเป็นเหตุผลที่ GRU ถูกตัดออกจากงานนี้)

**causal อย่างเคร่งครัด**: pad ทางซ้ายเท่านั้นแล้วตัดส่วนเกินทางขวาทิ้ง timestep t
จึงเห็นได้แค่ t และก่อนหน้า ข้อนี้ไม่ใช่เรื่องความถูกต้องของ label (ทั้งหน้าต่างอยู่ใน
อดีตอยู่แล้ว) แต่เป็นเรื่องความหมายทางฟิสิกส์แบบเดียวกับที่ ``FlareLSTM`` ตั้ง
``bidirectional=False``

**receptive field ต้องคลุมทั้งหน้าต่าง**: RF = 1 + Σ (kernel−1)·dilation ถ้าไม่ถึง
``seq_len`` โมเดลจะมองไม่เห็นต้นลำดับเลย และจะแพ้ในตารางด้วยเหตุผลที่ไม่เกี่ยวกับ
สถาปัตยกรรม — ``__init__`` จึงคำนวณแล้ว raise ถ้าไม่พอ แทนที่จะปล่อยผ่าน

**และไม่ควรเกินหน้าต่างมากนัก**: ชั้น dilation ที่ยื่นเลยลำดับไปเห็นแต่ padding
น้ำหนักของมันจึงแทบไม่ได้ทำงาน — ความยาวหน้าต่างต่างกันตาม dataset: production ยาว
24 timestep (``sequence.length`` ใน ``configs/data.yaml``) ส่วน dataset ของงานเปรียบเทียบ
ยาว 8 timestep ที่ cadence 12 ชม. (``STUDY_SEQUENCE_LENGTH`` ใน ``sunseg.data.study_dataset``)
ที่ความยาว 8 kernel 2 กับ dilation 1/2/4 ให้ RF 8 พอดี
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .pooling import build_head, build_pooling, check_input


def receptive_field(kernel_size: int, dilations: Sequence[int]) -> int:
    """จำนวน timestep ที่ผลลัพธ์ ณ ตำแหน่งสุดท้ายมองเห็นย้อนหลัง"""
    return 1 + sum((kernel_size - 1) * d for d in dilations)


class CausalBlock(nn.Module):
    """conv causal หนึ่งชั้น + ReLU + dropout พร้อม residual

    residual มีไว้ให้ gradient ไหลถึงชั้นล่างได้ตรง ๆ — ที่ความลึก 4 ชั้นบนข้อมูลที่มี
    positive ไม่กี่ร้อยตัว การเทรนไม่นิ่งเป็นปัญหาจริง (เคยมี fold/seed ที่เทรนได้ TSS ติดลบ
    ซึ่งเป็นเหตุผลที่ objective ของการค้นหาลงโทษ SD ด้วย — ดู ``scripts/study/tune.py``)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        # ปรับจำนวนช่องให้ตรงกันเมื่อ residual ข้ามชั้นที่เปลี่ยนความกว้าง
        self.project = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        length = x.size(-1)
        y = self.conv(x)[:, :, :length]        # ตัด padding ทางขวาทิ้ง -> เหลือ causal ล้วน
        y = self.drop(self.act(y))
        residual = x if self.project is None else self.project(x)
        return y + residual


class FlareTCN(nn.Module):
    """TCN + pooling + หัวจำแนก

    Parameters
    ----------
    n_features
        จำนวน feature ต่อ timestep (ต่างกันตามแบบ: 18 / 19 / 22)
    seq_len
        ความยาวหน้าต่าง ใช้ตรวจว่า receptive field คลุมพอ
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        channels: int = 24,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        dropout: float = 0.3,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        dilations = tuple(int(d) for d in dilations)
        field = receptive_field(kernel_size, dilations)
        if field < seq_len:
            raise ValueError(
                f"receptive field {field} ไม่คลุมหน้าต่างยาว {seq_len} — "
                f"เพิ่ม dilation หรือ kernel_size (ตอนนี้ kernel={kernel_size}, dilations={list(dilations)})"
            )

        self.n_features = n_features
        self.seq_len = seq_len
        self.receptive_field = field

        blocks, in_channels = [], n_features
        for dilation in dilations:
            blocks.append(CausalBlock(in_channels, channels, kernel_size, dilation, dropout))
            in_channels = channels
        self.blocks = nn.Sequential(*blocks)

        self.pool = build_pooling(pooling, channels)
        self.head = build_head(channels, dropout)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        check_input(x, self.n_features)
        sequence = self.blocks(x.transpose(1, 2)).transpose(1, 2)   # (B, L, C)
        context, attention = self.pool(sequence)
        logits = self.head(context).squeeze(-1)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
