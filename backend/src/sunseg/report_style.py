"""สไตล์ของรูปประกอบรายงาน — ใช้ร่วมกันระหว่าง ``scripts/study/plot_figures.py`` และ
``scripts/forecast/plot_*.py`` ให้รูปทุกรูปในรายงานหน้าตาเป็นชุดเดียวกัน

**import โมดูลนี้ก่อน ``matplotlib.pyplot`` เสมอ** — มันตั้ง backend เป็น ``Agg`` (เขียนไฟล์
อย่างเดียว ไม่เปิดหน้าต่าง) แล้วตั้งธีมของ seaborn

ธีม/จานสีเป็นของ seaborn ตรง ๆ ไม่ได้ตั้งเอง: ``whitegrid`` + ``colorblind`` (4 ช่องแรกผ่าน
``validate_palette.js`` ของ dataviz skill ทุกด่าน มีคำเตือน contrast ที่ช่องส้ม ซึ่งชดเชยด้วย legend +
ตาราง CSV คู่รูป) · ข้อความในรูปเป็นภาษาอังกฤษทั้งหมด จึงใช้ฟอนต์ปริยายได้โดยไม่ต้องหาฟอนต์ไทย
รูปตั้งใจทำโหมดสว่างอย่างเดียว — ปลายทางคือรายงานที่พิมพ์ลงกระดาษ
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.15)
plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight", "axes.titleweight": "bold"})

#: จานสีตามตัวตน — ลำดับคงที่ (ช่องที่ i = สถาปัตยกรรมที่ i ใน config) ไม่วนสี
PALETTE = sns.color_palette("colorblind")
BLUE, ORANGE, GREEN, RED = (PALETTE[i] for i in range(4))
GREY = "0.55"
THRESHOLD = "0.15"  # เส้นเกณฑ์ตัดสินใจ — สีกลาง ไม่ชนกับสีตัวตนของ series ใด
SEQUENTIAL = "Blues"   # ขนาด (confusion matrix)
DIVERGING = "vlag"     # ขั้วบวก/ลบ มีจุดกลางเป็นสีกลาง (interaction)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def true_runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """ช่วง index ที่ต่อเนื่องกันของค่า True — ใช้แรเงาเป็นแถบเดียวแทนแถบละจุด"""
    spans, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(flags) - 1))
    return spans
