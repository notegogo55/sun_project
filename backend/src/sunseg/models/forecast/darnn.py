"""DA-RNN — LSTM ที่ถ่วงน้ำหนัก *ราย feature* ก่อนป้อนเข้าแต่ละ timestep

ดัดแปลงจาก dual-stage attention RNN (Qin et al. 2017) ให้เหลือขนาดที่ข้อมูลชุดนี้รับไหว

**ลักษณะเฉพาะ** — และทำไมไม่ใช่แค่ "LSTM + attention อีกแบบ":
``FlareLSTM`` ถ่วงน้ำหนักข้าม *เวลา* จึงตอบได้แค่ "ชั่วโมงไหนสำคัญ" ส่วนตัวนี้ถ่วงน้ำหนัก
ข้าม *feature* ในทุก timestep จึงตอบได้ว่า "**พารามิเตอร์ตัวไหน**สำคัญ" ซึ่งเป็นคำถามที่
ตาราง TSS ตอบไม่ได้: ถ้าเติม X-ray แล้ว TSS ไม่ขยับ ตารางบอกไม่ได้ว่าเพราะ X-ray ไม่มี
ข้อมูล หรือเพราะโมเดลไม่ได้ใช้มัน — น้ำหนักจาก :meth:`feature_attention` แยกสองอย่างนี้ออก

**ที่ย่อจากเปเปอร์**: ขั้นที่สองของเปเปอร์เป็น decoder ที่มี temporal attention เต็มรูป
ที่นี่ใช้ attention pooling ตัวเดียวกับสถาปัตยกรรมอื่นแทน เพื่อให้ส่วนที่ต่างกันระหว่าง
แถวในตารางคือ input attention เท่านั้น ไม่ใช่ทั้งท่อ

**ราคาที่ต้องรู้**: ``alpha`` ขึ้นกับ hidden state ของก้าวก่อนหน้า จึงต้องวนทีละ timestep
ด้วย ``LSTMCell`` — เร็วกว่านี้ไม่ได้โดยธรรมชาติ และช้ากว่า ``nn.LSTM`` ที่ cuDNN fuse
ทั้งลำดับให้ราวสิบเท่า พจน์ ``U_e x^k`` ไม่ขึ้นกับเวลาจึงคำนวณครั้งเดียวนอกลูป

**ขนาดโตตาม feature พอ ๆ กับ LSTM**: ``U_e`` เป็น T×T จึงไม่ขึ้นกับจำนวน feature เลย
มีแค่ ``weight_ih`` ของ ``LSTMCell`` ที่โตตาม — วัดจริงที่ 18 -> 22 feature ได้ +5.6%
เทียบกับ LSTM +6.6% และ Transformer +1.6% (ดู ``docs/adr/0001-tune-per-architecture.md``)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .pooling import build_head, build_pooling, check_input


class InputAttention(nn.Module):
    """คะแนนความสำคัญของแต่ละ feature ณ timestep หนึ่ง จากสถานะของก้าวก่อนหน้า

    ``e_t^k = v_e . tanh(W_e [h_{t-1}; s_{t-1}] + U_e x^k)`` ตามเปเปอร์ โดย ``x^k`` คือ
    อนุกรมทั้งเส้นของ feature ที่ ``k`` — ซึ่งแปลว่าน้ำหนักของ feature หนึ่งขึ้นกับรูปร่าง
    ของอนุกรมนั้นทั้งเส้น ไม่ใช่แค่ค่า ณ เวลานั้น
    """

    def __init__(self, hidden_size: int, seq_len: int) -> None:
        super().__init__()
        self.state = nn.Linear(2 * hidden_size, seq_len, bias=False)   # W_e
        self.series = nn.Linear(seq_len, seq_len, bias=False)          # U_e
        self.score = nn.Linear(seq_len, 1, bias=False)                 # v_e

    def series_term(self, x: torch.Tensor) -> torch.Tensor:
        """``U_e x^k`` ของทุก feature — ไม่ขึ้นกับ t จึงคำนวณครั้งเดียวต่อ forward"""
        return self.series(x.transpose(1, 2))                          # (B, F, L)

    def forward(self, series_term: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        state = self.state(torch.cat([hidden, cell], dim=-1)).unsqueeze(1)   # (B, 1, L)
        scores = self.score(torch.tanh(state + series_term)).squeeze(-1)     # (B, F)
        return torch.softmax(scores, dim=-1)


class FlareDARNN(nn.Module):
    """input attention + LSTM + pooling + หัวจำแนก"""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_size: int = 26,
        dropout: float = 0.3,
        pooling: str = "attention",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        self.attention = InputAttention(hidden_size, seq_len)
        self.cell = nn.LSTMCell(n_features, hidden_size)
        self.pool = build_pooling(pooling, hidden_size)
        self.head = build_head(hidden_size, dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """orthogonal ให้ recurrent weights — เหตุผลเดียวกับ ``FlareLSTM``"""
        nn.init.xavier_uniform_(self.cell.weight_ih)
        nn.init.orthogonal_(self.cell.weight_hh)
        for bias in (self.cell.bias_ih, self.cell.bias_hh):
            nn.init.zeros_(bias)
        # forget gate bias = 1 (Jozefowicz 2015) — LSTMCell เรียง gate เป็น i, f, g, o
        hidden = self.hidden_size
        self.cell.bias_ih.data[hidden : 2 * hidden].fill_(1.0)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """คืน ``(hidden states (B, L, H), น้ำหนักราย feature (B, L, F))``"""
        batch = x.size(0)
        hidden = x.new_zeros(batch, self.hidden_size)
        cell = x.new_zeros(batch, self.hidden_size)
        series_term = self.attention.series_term(x)

        states, alphas = [], []
        for t in range(x.size(1)):
            alpha = self.attention(series_term, hidden, cell)      # (B, F)
            hidden, cell = self.cell(alpha * x[:, t], (hidden, cell))
            states.append(hidden)
            alphas.append(alpha)
        return torch.stack(states, dim=1), torch.stack(alphas, dim=1)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        check_input(x, self.n_features)
        if x.size(1) != self.seq_len:
            raise ValueError(
                f"ความยาวลำดับไม่ตรง: โมเดลผูกกับ {self.seq_len} timestep (U_e, W_e) แต่ได้ {x.size(1)}"
            )
        sequence, _ = self._encode(x)
        context, attention = self.pool(sequence)
        logits = self.head(context).squeeze(-1)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def feature_attention(self, x: torch.Tensor) -> torch.Tensor:
        """น้ำหนักราย feature รูปทรง ``(B, L, F)`` — รวมกันได้ 1 ในทุก (sample, timestep)

        นี่คือสิ่งที่สถาปัตยกรรมอื่นในตารางให้ไม่ได้ ใช้ตอบว่าโมเดล *ใช้* feature ที่เพิ่ม
        เข้ามาจริงหรือไม่ แยกจากคำถามว่ามันทำให้ TSS ดีขึ้นหรือไม่
        """
        self.eval()
        check_input(x, self.n_features)
        return self._encode(x)[1]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.sigmoid(self(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
