"""โมเดลพยากรณ์ flare >= M1.0 ใน 24 ชม. — สี่สถาปัตยกรรมที่อยู่ในระดับเดียวกัน

| kind          | คลาส               | วิธีอ่านลำดับเวลา                                     |
|---------------|--------------------|------------------------------------------------------|
| ``lstm``        | :class:`FlareLSTM`        | สถานะส่งต่อทีละก้าว                              |
| ``tcn``         | :class:`FlareTCN`         | convolution causal ที่ขยาย dilation              |
| ``transformer`` | :class:`FlareTransformer` | self-attention ระหว่างทุกคู่ timestep            |
| ``darnn``       | :class:`FlareDARNN`       | LSTM ที่ถ่วงน้ำหนักราย feature ก่อนป้อนทุกก้าว |

ทุกตัวรับ ``(B, L, F)`` คืน logit ``(B,)`` และคืนน้ำหนักราย timestep ได้ด้วย
``return_attention=True`` จึงใช้แทนกันได้ทุกที่ (สคริปต์เทรน, inference, หน้าเว็บ,
งานเปรียบเทียบ) — ประกอบผ่าน :func:`build_architecture` จากชื่อ ``kind`` เสมอ
ไม่เรียกคลาสตรง ๆ ค่า hyperparameter ของแต่ละตัวอยู่ที่ ``configs/forecast.yaml``
"""

from .darnn import FlareDARNN
from .lstm import FlareLSTM
from .pooling import POOLING_KINDS
from .registry import ARCHITECTURES, Architecture, architecture_label, build_architecture
from .tcn import FlareTCN
from .transformer import FlareTransformer

__all__ = [
    "ARCHITECTURES",
    "POOLING_KINDS",
    "Architecture",
    "FlareDARNN",
    "FlareLSTM",
    "FlareTCN",
    "FlareTransformer",
    "architecture_label",
    "build_architecture",
]
