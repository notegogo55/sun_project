"""LSTM สำหรับพยากรณ์ flare จากลำดับเวลาของ SHARP magnetic parameters

รับ SHARP parameters ย้อนหลังหนึ่งหน้าต่าง (production: 24 timesteps x 18 features) แล้วทำนาย
ความน่าจะเป็นที่ active region ดวงนั้นจะปะทุ flare ระดับ >= M1.0 ภายใน 24 ชม. ถัดไป

**ทำไมใช้ลำดับเวลา ไม่ใช่แค่ค่า ณ เวลาปัจจุบัน**: งานวิจัยชี้ว่า *อัตราการเปลี่ยนแปลง*
ของ magnetic helicity และ free energy บอกการปะทุได้ดีกว่าค่าสัมบูรณ์ AR ที่มีสนาม
แม่เหล็กแรงแต่นิ่งมีโอกาสปะทุน้อยกว่า AR ที่สนามกำลังบิดตัวเร็ว — ข้อมูลนี้อยู่ใน
มิติเวลาเท่านั้น (สคริปต์เทรนจึงเทียบกับ logistic regression ที่ใช้ค่า ณ เวลาเดียว
เสมอ เพื่อพิสูจน์ว่ามิติเวลาช่วยจริง)

**ลักษณะเฉพาะ**: สถานะส่งต่อทีละก้าว ข้อมูลจากต้นหน้าต่างต้องเดินผ่านทุก timestep จึงจะถึง
ปลายหน้าต่าง — ต่างจาก TCN/Transformer ที่เห็นทั้งหน้าต่างพร้อมกัน
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import build_head, build_pooling, check_input

__all__ = ["FlareLSTM"]


class FlareLSTM(nn.Module):
    """LSTM + pooling + หัวจำแนก

    Parameters
    ----------
    n_features
        จำนวน feature ต่อ timestep (production: 18 SHARP parameters ตาม ``sharp.features``)
    hidden_size, num_layers, dropout
        พารามิเตอร์ของชั้น LSTM
    bidirectional
        ควรเป็น ``False`` — การมองย้อนกลับไม่มีความหมายทางฟิสิกส์สำหรับการพยากรณ์
        (แม้ในทางเทคนิคจะไม่รั่วข้อมูลอนาคต เพราะทั้งหน้าต่างอยู่ในอดีตแล้วก็ตาม)
    pooling
        วิธียุบลำดับเป็นเวกเตอร์เดียว — ดู :data:`.pooling.POOLING_KINDS`
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch ใช้ dropout ระหว่างชั้นเท่านั้น จึงไม่มีผลถ้ามีชั้นเดียว
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_size = hidden_size * (2 if bidirectional else 1)
        self.pool = build_pooling(self.pooling, out_size)
        self.head = build_head(out_size, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """ตั้งค่าเริ่มต้นแบบ orthogonal ให้ recurrent weights — ช่วยให้ gradient ไม่ระเบิด/หายไป"""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # ตั้ง forget gate bias = 1 เพื่อให้จำข้อมูลระยะยาวได้ตั้งแต่ต้น (Jozefowicz 2015)
                hidden = self.hidden_size
                param.data[hidden : 2 * hidden].fill_(1.0)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x
            รูปทรง ``(B, L, n_features)`` — ต้อง normalise มาแล้ว

        Returns
        -------
        logits รูปทรง ``(B,)`` (ยังไม่ผ่าน sigmoid) และถ้า ``return_attention=True``
        จะคืนน้ำหนัก attention รูปทรง ``(B, L)`` มาด้วย
        """
        check_input(x, self.n_features)

        sequence, _ = self.lstm(x)                       # (B, L, H)
        context, attention = self.pool(sequence)
        logits = self.head(context).squeeze(-1)          # (B,)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """ความน่าจะเป็นในช่วง [0, 1] — ใช้ตอน inference"""
        self.eval()
        return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
