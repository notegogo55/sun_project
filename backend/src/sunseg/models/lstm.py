"""LSTM สำหรับพยากรณ์ flare จากลำดับเวลาของ SHARP magnetic parameters

รับ SHARP parameters ย้อนหลัง 24 ชั่วโมง (24 timesteps x 18 features) แล้วทำนาย
ความน่าจะเป็นที่ active region ดวงนั้นจะปะทุ flare ระดับ >= M1.0 ภายใน 24 ชม. ถัดไป

**ทำไม LSTM ไม่ใช่แค่ใช้ค่า ณ เวลาปัจจุบัน**: งานวิจัยชี้ว่า *อัตราการเปลี่ยนแปลง*
ของ magnetic helicity และ free energy บอกการปะทุได้ดีกว่าค่าสัมบูรณ์ AR ที่มีสนาม
แม่เหล็กแรงแต่นิ่งมีโอกาสปะทุน้อยกว่า AR ที่สนามกำลังบิดตัวเร็ว — ข้อมูลนี้อยู่ใน
มิติเวลาเท่านั้น (สคริปต์เทรนจึงเทียบกับ logistic regression ที่ใช้ค่า ณ เวลาเดียว
เสมอ เพื่อพิสูจน์ว่ามิติเวลาช่วยจริง)
"""

from __future__ import annotations

import torch
import torch.nn as nn


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


class FlareLSTM(nn.Module):
    """LSTM + attention pooling + classification head

    Parameters
    ----------
    n_features
        จำนวน SHARP parameters ต่อ timestep (ต้องตรงกับ ``sharp.features`` ใน config)
    hidden_size, num_layers, dropout
        พารามิเตอร์ของชั้น LSTM
    bidirectional
        ควรเป็น ``False`` — การมองย้อนกลับไม่มีความหมายทางฟิสิกส์สำหรับการพยากรณ์
        (แม้ในทางเทคนิคจะไม่รั่วข้อมูลอนาคต เพราะทั้งหน้าต่างอยู่ในอดีตแล้วก็ตาม)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.use_attention = use_attention

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
        self.pool = AttentionPooling(out_size) if use_attention else None

        self.head = nn.Sequential(
            nn.LayerNorm(out_size),
            nn.Dropout(dropout),
            nn.Linear(out_size, out_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_size // 2, 1),
        )

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
        if x.dim() != 3:
            raise ValueError(f"คาดหวัง input 3 มิติ (B, L, F) แต่ได้รูปทรง {tuple(x.shape)}")
        if x.size(-1) != self.n_features:
            raise ValueError(
                f"จำนวน feature ไม่ตรง: โมเดลคาดหวัง {self.n_features} แต่ได้ {x.size(-1)}"
            )

        sequence, _ = self.lstm(x)                       # (B, L, H)

        if self.pool is not None:
            context, attention = self.pool(sequence)
        else:
            context = sequence[:, -1, :]                 # hidden state ตัวสุดท้าย
            attention = torch.zeros(x.size(0), x.size(1), device=x.device)

        logits = self.head(context).squeeze(-1)          # (B,)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """ความน่าจะเป็นในช่วง [0, 1] — ใช้ตอน inference"""
        self.eval()
        return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm(n_features: int, cfg) -> FlareLSTM:
    """สร้างโมเดลจาก ``configs/lstm.yaml``"""
    return FlareLSTM(
        n_features=n_features,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
        use_attention=cfg.use_attention,
    )
