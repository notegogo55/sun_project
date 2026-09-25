"""สถาปัตยกรรมโมเดลของระบบ

- :mod:`.unet` — U-Net แบ่งส่วน active region จากภาพ magnetogram เต็มดวง
- :mod:`.forecast` — โมเดลพยากรณ์ flare สี่ตัวที่อยู่ในระดับเดียวกัน
  (LSTM, TCN, Transformer, DA-RNN) ประกอบผ่าน :func:`build_architecture`
"""

from .forecast import (
    ARCHITECTURES,
    Architecture,
    FlareDARNN,
    FlareLSTM,
    FlareTCN,
    FlareTransformer,
    architecture_label,
    build_architecture,
)
from .unet import UNet

__all__ = [
    "ARCHITECTURES",
    "Architecture",
    "FlareDARNN",
    "FlareLSTM",
    "FlareTCN",
    "FlareTransformer",
    "UNet",
    "architecture_label",
    "build_architecture",
]
