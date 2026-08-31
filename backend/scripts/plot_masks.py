"""ตรวจด้วยตาว่า mask วางทับ active region ตรงตำแหน่งจริง

    python backend/scripts/plot_masks.py

**นี่คือประตูตรวจที่ห้ามข้าม** ก่อนเทรน U-Net การสร้าง mask ต้องแปลงพิกัดจาก SHARP
patch ไปยังภาพเต็มดวงผ่าน WCS ถ้าการแปลงผิด mask จะเลื่อนไปจากตำแหน่งจริงทั้งภาพ
โมเดลจะยังเทรนได้ (loss ลดลงสวยงาม) แต่เรียนรู้สิ่งที่ผิด และเราจะไม่มีทางรู้เลย
จนกว่าจะเปิดดูภาพ

สิ่งที่ต้องเห็นในภาพผลลัพธ์: **เส้นขอบสีส้มล้อมรอบบริเวณที่มีสนามแม่เหล็กเข้ม**
(จุดขาว/ดำจัด) ถ้าเส้นขอบไปอยู่บนพื้นเทาเรียบ แปลว่าการแปลงพิกัดผิด
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402
import matplotlib.font_manager  # noqa: E402

matplotlib.use("Agg")

# ฟอนต์เริ่มต้น (DejaVu Sans) ไม่มี glyph ภาษาไทย — ข้อความในภาพจะกลายเป็นกล่องสี่เหลี่ยม
# ภาพนี้คือด่านตรวจก่อนเทรน ถ้าอ่านหัวข้อไม่ออกก็ตรวจไม่ได้ จึงเลือกฟอนต์ที่มี glyph ไทย
# ตัวแรกที่ติดตั้งอยู่จริง แล้ว fallback กลับไป DejaVu Sans ถ้าไม่เจอสักตัว (เช่นบน Linux)
_THAI_FONTS = ["Leelawadee UI", "Tahoma", "Angsana New", "Noto Sans Thai"]
_INSTALLED = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
matplotlib.rcParams["font.family"] = [
    *[f for f in _THAI_FONTS if f in _INSTALLED],
    "DejaVu Sans",
]
# ฟอนต์ไทยส่วนใหญ่ไม่มี glyph เครื่องหมายลบของ Unicode (U+2212) — ใช้ ASCII แทน
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sunseg.config import load_data_config  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.tracking.detect import detect_regions  # noqa: E402

logger = logging.getLogger("plot_masks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=8, help="จำนวนเฟรมที่จะพล็อต")
    p.add_argument("--seed", type=int, default=0, help="ค่าสุ่มสำหรับเลือกเฟรม")
    p.add_argument("--vmax", type=float, default=1000.0, help="ค่าสนามแม่เหล็กสูงสุดของสเกลสี (G)")
    p.add_argument("--frames", nargs="*", help="ระบุชื่อเฟรมเองแทนการสุ่ม")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_data_config()
    setup_logging()

    frames_dir = cfg.paths.processed / "frames"
    image_dir = frames_dir / "images"
    mask_dir = frames_dir / "masks"

    if not image_dir.exists() or not any(image_dir.glob("*.npy")):
        logger.error(
            "ไม่พบเฟรมภาพใน %s\nรัน `python backend/scripts/download_images.py` ก่อน", image_dir
        )
        return 1

    available = sorted(p.stem for p in image_dir.glob("*.npy"))
    logger.info("มีเฟรมทั้งหมด %d เฟรม", len(available))

    if args.frames:
        missing = [f for f in args.frames if f not in available]
        if missing:
            logger.error("ไม่พบเฟรม: %s", missing)
            return 1
        chosen = args.frames
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.n, len(available))
        chosen = [available[i] for i in sorted(rng.choice(len(available), n, replace=False))]

    # ------------------------------------------------------------------ #
    n_cols = min(4, len(chosen))
    n_rows = (len(chosen) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    stats = []
    for ax, timestamp in zip(axes, chosen, strict=False):
        image = np.load(image_dir / f"{timestamp}.npy").astype(np.float32)
        mask = np.load(mask_dir / f"{timestamp}.npy").astype(np.uint8)

        # สเกลเทาแบบ HMI: ขั้วบวกขาว ขั้วลบดำ quiet Sun เทากลาง
        ax.imshow(
            np.nan_to_num(image), cmap="gray", vmin=-args.vmax, vmax=args.vmax, origin="lower"
        )
        # วาดเฉพาะเส้นขอบ เพื่อไม่ให้บังโครงสร้างสนามแม่เหล็กที่อยู่ข้างใต้
        ax.contour(mask, levels=[0.5], colors="#ff6b2c", linewidths=1.0)

        detections = detect_regions(
            mask, magnetogram=image, min_area_px=cfg.fulldisk.min_ar_area_px
        )
        area_pct = 100 * mask.mean()
        stats.append((timestamp, len(detections), area_pct))

        pretty = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
        ax.set_title(f"{pretty}\nAR {len(detections)} ดวง · {area_pct:.2f}% ของภาพ", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.suptitle(
        "ตรวจสอบ mask: เส้นขอบสีส้มต้องล้อมรอบบริเวณสนามแม่เหล็กเข้ม (จุดขาว/ดำจัด)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = cfg.paths.artifacts / "figures" / "mask_check.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    logger.info("-" * 62)
    logger.info("%-20s %10s %14s", "เฟรม", "จำนวน AR", "พื้นที่ mask")
    for timestamp, n_ar, area in stats:
        logger.info("%-20s %10d %13.2f%%", timestamp, n_ar, area)

    areas = [s[2] for s in stats]
    logger.info("-" * 62)
    logger.info("พื้นที่ mask เฉลี่ย %.2f%% (คาดหวังราว 0.5-3%% สำหรับ active region)", np.mean(areas))

    if np.mean(areas) > 10:
        logger.warning("[WARN] mask ครอบคลุมพื้นที่มากผิดปกติ — ตรวจสอบเกณฑ์ค่า bitmap")
    elif np.mean(areas) < 0.05:
        logger.warning("[WARN] mask เล็กผิดปกติ — การแปลงพิกัดหรือเกณฑ์อาจผิด")

    logger.info("")
    logger.info("บันทึกภาพแล้ว: %s", out_path)
    logger.info(">>> เปิดไฟล์นี้ดูด้วยตาก่อนเทรน U-Net <<<")
    logger.info("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
