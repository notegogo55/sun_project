"""ตรวจว่าภาพ AIA ถูกวางทับกริดของเฟรม HMI ตรงตำแหน่งจริง

    python backend/scripts/checks/plot_aia_alignment.py

.. warning::
   **ห้ามข้ามขั้นตอนนี้** ด้วยเหตุผลเดียวกับ ``plot_masks.py`` — การแปลงพิกัดที่ผิดจะให้
   ภาพ AIA เต็มดวงที่สวยงามพร้อมเส้นขอบในตำแหน่งที่ดูน่าเชื่อถือ แต่ตัวเลขความเข้มแสง
   ราย AR ที่ได้กลับถูกสุ่มมาจาก quiet Sun ทั้งหมด ความล้มเหลวแบบนี้เงียบสนิท

**สิ่งที่ต้องเห็นในภาพ**: จุดสว่าง (plage) ของ **1600 A ต้องทับพิกเซลสนามแม่เหล็กเข้ม
ของ magnetogram พอดี** ไม่ใช่ใกล้ ๆ — การอยู่ร่วมตำแหน่งนี้เป็นสิ่งที่ฟิสิกส์รับประกัน
(บริเวณสนามแม่เหล็กเข้มให้ความร้อนกับบรรยากาศเหนือมันโดยตรง) จึงเป็นตัวตรวจที่แม่นยำ

**การตรวจเชิงตัวเลข**: คำนวณ contrast ratio

    mean(AIA ในพิกเซลที่ |B| > 150 G) / median(AIA ทั้งจาน)

ทั้งของภาพปกติและของภาพที่หมุน 180 องศา ถ้าภาพที่หมุนได้คะแนนสูงกว่า แปลว่า WCS ผิดทิศ
(ค่าที่วัดได้จริงต่างกันราว 7 เท่า — ตัวตรวจนี้จึงชี้ขาดได้ ไม่คลุมเครือ)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.aia import AiaFrameStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("plot_aia_alignment")

#: เกณฑ์สนามแม่เหล็กเข้ม (เกาส์) ที่ใช้เลือกพิกเซลอ้างอิง
STRONG_FIELD_GAUSS = 150.0

#: contrast ต่ำกว่านี้ = น่าสงสัย
#:
#: ตั้งไว้ต่ำเพราะแต่ละช่องมี contrast ตามธรรมชาติไม่เท่ากัน: 304/171 ได้ 4-8 เท่าเพราะ
#: โคโรนาสงบมืด ส่วน **1600 A ได้เพียงราว 2.3 เท่าโดยปกติ** เนื่องจากเห็น continuum ของ
#: โฟโตสเฟียร์ซึ่งสว่างทั่วทั้งจานอยู่แล้ว plage จึงโดดออกมาน้อยกว่า — ไม่ใช่ความผิดพลาด
WARN_BELOW = 1.8

#: ภาพปกติต้องชนะภาพที่หมุน 180 องศาอย่างน้อยกี่เท่า
#:
#: ตัวชี้ขาดที่แท้จริงคืออัตราส่วนนี้ ไม่ใช่ค่าสัมบูรณ์ข้างบน: WCS ที่ผิดทิศให้ค่าราว 1.0
#: ทั้งสองฝั่ง ส่วนที่ถูกต้องให้ 2 เท่า (1600) ถึง 7 เท่า (171)
MIN_MARGIN = 1.5

#: ชื่อชั้นบรรยากาศแบบ ASCII — matplotlib ไม่มีฟอนต์ไทยติดมาด้วย ข้อความไทยจะกลายเป็น
#: สี่เหลี่ยมว่างบนรูป จึงใช้อังกฤษบนรูปแล้วเก็บภาษาไทยไว้ใน log กับหน้าเว็บแทน
REGION_ASCII = {
    "1600": "photosphere / TR",
    "304": "chromosphere",
    "171": "quiet corona",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--limit", type=int, default=3, help="จำนวนเฟรมที่วาด (ค่าปริยาย 3)")
    p.add_argument("--out", type=Path, help="ที่บันทึกรูป (ค่าปริยาย artifacts/figures/)")
    return p.parse_args()


def contrast_ratio(image: np.ndarray, selection: np.ndarray) -> float:
    """mean ของพิกเซลที่เลือก หารด้วย median ของทั้งภาพ"""
    finite = np.isfinite(image)
    baseline = float(np.median(image[finite])) if finite.any() else 0.0
    if baseline <= 0:
        return float("nan")

    values = image[selection & finite]
    return float(values.mean()) / baseline if values.size else float("nan")


def check_frame(
    magnetogram: np.ndarray, channels: dict[str, np.ndarray]
) -> list[tuple[str, float, float]]:
    """คืน ``[(ช่อง, contrast ปกติ, contrast เมื่อหมุน 180), ...]``"""
    strong = np.abs(np.nan_to_num(magnetogram)) > STRONG_FIELD_GAUSS

    results = []
    for key, image in channels.items():
        results.append(
            (key, contrast_ratio(image, strong), contrast_ratio(np.rot90(image, 2), strong))
        )
    return results


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize

    from sunseg.data.aia import _colormap_for

    store = AiaFrameStore(cfg.aia.root)
    if not store.available:
        logger.error(
            "ยังไม่มีภาพ AIA ที่ %s — รัน backend/scripts/data/download_aia.py ก่อน", cfg.aia.root
        )
        return 1

    frames_dir = cfg.paths.processed / "frames"
    stems = sorted({p.stem for p in cfg.aia.root.rglob("*.npy")})[: args.limit]
    logger.info("ตรวจ %d เฟรม: %s", len(stems), ", ".join(stems))

    n_cols = 1 + len(cfg.aia.channels)
    fig, axes = plt.subplots(
        len(stems), n_cols, figsize=(4 * n_cols, 4 * len(stems)), squeeze=False
    )

    all_results: list[tuple[str, str, float, float]] = []

    for row, stem in enumerate(stems):
        magnetogram = np.load(frames_dir / "images" / f"{stem}.npy").astype(np.float32)
        mask_path = frames_dir / "masks" / f"{stem}.npy"
        mask = np.load(mask_path).astype(np.uint8) if mask_path.exists() else None

        # --- คอลัมน์แรก: magnetogram ---
        ax = axes[row][0]
        ax.imshow(np.nan_to_num(magnetogram), cmap="gray", vmin=-1500, vmax=1500, origin="lower")
        if mask is not None:
            ax.contour(mask, levels=[0.5], colors="#ff5a28", linewidths=0.7)
        ax.set_title(f"{stem}\nHMI magnetogram", fontsize=9)
        ax.axis("off")

        channels: dict[str, np.ndarray] = {}
        for col, channel in enumerate(cfg.aia.channels, start=1):
            image = store.load(stem, channel.key)
            ax = axes[row][col]

            if image is None:
                ax.text(0.5, 0.5, f"no data: {channel.key} A", ha="center", va="center")
                ax.axis("off")
                continue

            channels[channel.key] = image
            norm = ImageNormalize(
                vmin=0, vmax=channel.display_vmax, stretch=AsinhStretch(0.02), clip=True
            )
            ax.imshow(
                np.nan_to_num(image), cmap=_colormap_for(channel.wavelength),
                norm=norm, origin="lower",
            )
            if mask is not None:
                ax.contour(mask, levels=[0.5], colors="#00e5ff", linewidths=0.7)
            ax.set_title(
                f"AIA {channel.key} A\n{REGION_ASCII.get(channel.key, '')}", fontsize=9
            )
            ax.axis("off")

        for key, normal, rotated in check_frame(magnetogram, channels):
            all_results.append((stem, key, normal, rotated))

    # ข้อความบนรูปเป็นอังกฤษล้วน (ดูหมายเหตุที่ REGION_ASCII)
    fig.suptitle(
        "AIA alignment check - 1600 A plage must sit exactly on strong-field pixels",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = args.out or (cfg.paths.artifacts / "figures" / "aia_alignment_check.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("บันทึกรูปที่ %s", out_path)

    # ------------------------------------------------------------------ #
    logger.info("-" * 70)
    logger.info("contrast ratio = mean(AIA ที่ |B| > %.0f G) / median(ทั้งจาน)", STRONG_FIELD_GAUSS)
    logger.info(
        "%-18s %6s %9s %9s %7s   %s", "เฟรม", "ช่อง", "ปกติ", "หมุน180", "อัตรา", "ผล"
    )

    n_failed = 0
    n_warned = 0
    for stem, key, normal, rotated in all_results:
        if not np.isfinite(normal) or not np.isfinite(rotated):
            verdict, n_warned = "ประเมินไม่ได้", n_warned + 1
        elif rotated > normal:
            verdict, n_failed = "!!! กลับหัว !!!", n_failed + 1
        elif normal < MIN_MARGIN * rotated:
            verdict, n_failed = f"!!! margin ต่ำ (< {MIN_MARGIN}x) !!!", n_failed + 1
        elif normal < WARN_BELOW:
            verdict, n_warned = f"น่าสงสัย (< {WARN_BELOW})", n_warned + 1
        else:
            verdict = "ok"
        ratio = normal / rotated if rotated else float("inf")
        logger.info(
            "%-18s %6s %9.2f %9.2f %7.1fx   %s", stem, key, normal, rotated, ratio, verdict
        )

    logger.info("-" * 70)

    if n_failed:
        logger.error(
            "ล้มเหลว %d รายการ — ภาพที่หมุน 180 องศาได้ contrast สูงกว่า แปลว่า WCS ผิดทิศ "
            "ห้ามใช้ข้อมูลชุดนี้ต่อ",
            n_failed,
        )
        return 1

    if n_warned:
        logger.warning(
            "มี %d รายการที่ contrast ต่ำผิดปกติ — ตรวจรูปด้วยตาก่อนใช้งานต่อ", n_warned
        )
    else:
        logger.info("ผ่านทุกรายการ — เปิดรูปดูด้วยตาอีกครั้งเพื่อยืนยัน")

    logger.info("เปิดดู: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
