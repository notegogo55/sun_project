"""เรนเดอร์ผล segmentation ของ U-Net เป็นวิดีโอ (mp4) สำหรับช่วงเวลาที่กำหนด

    python backend/scripts/render_segmentation_video.py --start 2024-05-01 --end 2024-06-01
    python backend/scripts/render_segmentation_video.py --layer 171
    python backend/scripts/render_segmentation_video.py --layer grid
    python backend/scripts/render_segmentation_video.py --layer stack
    python backend/scripts/render_segmentation_video.py --layer row --harp 11149  # Gannon storm (NOAA 13664)

แต่ละเฟรมคือภาพพื้นหลัง (magnetogram หรือช่อง AIA) พร้อมเส้นขอบ mask ทับสองเส้น:
  - **ส้ม/ฟ้า** = ผลทำนายของ U-Net (ส้มบนพื้น magnetogram, ฟ้าบนพื้น AIA — สีเดียวกับที่
    หน้าเว็บใช้)
  - **เขียว**   = mask จริงจาก SHARP (ถ้าเฟรมนั้นมีกำกับไว้) ไว้เทียบด้วยตา

``--layer grid`` เรนเดอร์ทั้งสี่เลเยอร์ (magnetogram + AIA 1600/304/171) เรียงเป็นตาราง
2x2 ในเฟรมเดียวกัน ให้เห็นภาพรวมทุกชั้นบรรยากาศพร้อมกัน

``--layer row`` เรนเดอร์เนื้อหาเดียวกับ ``grid`` แต่เรียงทั้งสี่ภาพต่อกันเป็นแถวนอนเดียว
(HMI | AIA 1600 | AIA 304 | AIA 171) แทนตาราง 2x2 — ให้ผลลัพธ์กว้างกว่าแต่เตี้ยกว่า

``--harp <HARPNUM>`` ครอบตัดทุกเลเยอร์ให้เหลือแค่หน้าต่างสี่เหลี่ยมขนาดคงที่รอบ HARP
ดวงนั้นดวงเดียว (ศูนย์กลางมาจาก ``LAT_FWT``/``LON_FWT`` ใน ``sharp_keywords.parquet`` แปลง
เป็นพิกเซลผ่าน WCS จริงของเฟรม) แทนที่จะโชว์ทั้งจาน — ใช้ดูวิวัฒนาการของ AR ดวงเดียวข้าม
เวลาแบบใกล้ชิด (เช่น HARP 11149 = NOAA 13664/13668 ต้นกำเนิด Gannon storm พ.ค. 2024)
เฟรมที่ HARP นั้นไม่มีข้อมูล SHARP ใกล้พอ (เกิน ``--harp-max-gap-hours``) หรือเลย
``--harp-max-abs-lon`` องศาไปแล้ว (ใกล้ขอบจานเกินไป) จะถูกข้าม ต้องมี
``data/interim/frame_wcs.parquet`` อยู่ก่อน (รัน ``download_aia.py --wcs-only``)

``--layer stack`` ผสมสามช่อง AIA เข้าเป็นภาพสีเดียว (tri-color composite แบบที่ SDO ใช้):
แดง=304 (โครโมสเฟียร์) เขียว=171 (โคโรนา) น้ำเงิน=1600 (โฟโตสเฟียร์) — คนละช่องสี ไม่ใช่เอา
colormap ของแต่ละช่องมาซ้อนทับกัน จึงยังแยกได้ว่าโครงสร้างสว่างจุดไหนมาจากชั้นบรรยากาศไหน
ไม่รวม magnetogram เข้าไปในภาพสี (ลองแล้วด้วย screen blend สีล้างกันจนแยกไม่ออก — เทากลาง
ของ magnetogram ดันโทนทั้งภาพไปทางขาว) ใช้เส้นขอบ mask จาก U-Net/SHARP บอกตำแหน่งสนาม
แม่เหล็กแรงแทน

ใช้ ``ffmpeg`` เข้ารหัสเป็น H.264 ผ่าน pipe (frame ดิบทาง stdin) แทน
``cv2.VideoWriter`` เพราะตัวเข้ารหัส mp4v ของ OpenCV บน Windows มักเล่นไม่ได้ในเบราว์เซอร์/
เครื่องเล่นทั่วไป — ต้องมี ffmpeg อยู่ใน PATH (ตรวจสอบด้วย ``ffmpeg -version``)

.. important::
   เฟรม ``.npy`` ทั้ง magnetogram และ AIA เก็บไว้เป็นทิศ CCD ดิบของ ``hmi.M_720s``
   (``CROTA2 ≈ 180°`` — ดูคอมเมนต์ใน ``backend/app/routers/segment.py::_solar_map_for``
   ภาพ AIA ถูก reproject ลงกริดเดียวกันจึงมีปัญหาเดียวกัน) ส่วนอื่นของโปรเจคใช้ค่านี้แก้
   พิกัด lon/lat ผ่าน WCS เท่านั้น แต่ไม่เคยหมุน **ภาพพิกเซล** ที่ใช้แสดงผล ทำให้ภาพนิ่งดูได้
   (ไม่มีอะไรให้เทียบทิศ) แต่พอเรียงเป็นวิดีโอ AR จะดูเหมือนเคลื่อนจากขวาไปซ้าย ซึ่งกลับทิศ
   จากการหมุนจริงของดวงอาทิตย์ (ตะวันออก→ตะวันตก คือซ้าย→ขวา เมื่อเหนือชี้ขึ้น) ตรวจยืนยัน
   ด้วยตำแหน่ง centroid ของ AR3664 ข้ามเฟรม: ก่อนหมุนแก้ ``cx`` ลดจาก 318→94 ตลอดสัปดาห์
   (ขวา→ซ้าย ผิด) หลังหมุน 180° กลายเป็น 194→418 (ซ้าย→ขวา ถูก) — จึงหมุนภาพ 180° ก่อน
   เรนเดอร์ทุกเฟรมในสคริปต์นี้ (ทุกเลเยอร์)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sunseg.config import load_data_config, load_tracking_config  # noqa: E402
from sunseg.data.aia import AiaFrameStore, MASK_COLOR_AIA, _colormap_for, draw_mask_overlay  # noqa: E402
from sunseg.data.frame_wcs import FrameWcsStore  # noqa: E402
from sunseg.inference.segment import SegmentationService, parse_frame_timestamp  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.tracking.detect import detect_regions  # noqa: E402
from sunseg.tracking.rotation import stonyhurst_to_pixel  # noqa: E402

logger = logging.getLogger("render_segmentation_video")

COLOR_PREDICTED_MAG = (255, 90, 40)  # ส้ม — ผลทำนายของ U-Net บนพื้น magnetogram (RGB)
COLOR_TRUTH = (60, 220, 90)  # เขียว — ground truth จาก SHARP (RGB) ใช้ทุกเลเยอร์

#: ช่อง AIA ที่รองรับ พร้อมป้ายกำกับภาษาอังกฤษ (cv2.putText วาดอักษรไทยไม่ได้)
AIA_LAYERS = {
    "1600": "AIA 1600A (photosphere)",
    "304": "AIA 304A (chromosphere)",
    "171": "AIA 171A (corona)",
}
LAYER_CHOICES = ["mag", *AIA_LAYERS, "grid", "row", "stack"]
COLOR_PREDICTED_STACK = (255, 255, 255)  # ขาว — เด่นชัดบนพื้นผสมที่มีแต่โทนอุ่น


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--start", default="2024-05-01", help="เวลาเริ่มต้น (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-06-01", help="เวลาสิ้นสุด (YYYY-MM-DD, ไม่รวม)")
    p.add_argument(
        "--layer",
        choices=LAYER_CHOICES,
        default="mag",
        help='เลเยอร์พื้นหลัง: "mag" (magnetogram), ช่อง AIA (171/304/1600), "grid" (ทั้งสี่ชั้นเรียง 2x2), '
        '"row" (ทั้งสี่ชั้นเรียงแถวนอน) หรือ "stack" (ผสมทั้งสี่ชั้นเป็นภาพเดียวด้วย screen blend)',
    )
    p.add_argument("--out", default=None, help="path ไฟล์วิดีโอผลลัพธ์ (default: artifacts/figures/segmentation_<layer>_<start>_<end>.mp4)")
    p.add_argument("--fps", type=int, default=6, help="เฟรมต่อวินาทีของวิดีโอ")
    p.add_argument("--scale", type=int, default=None, help="ขยายภาพจากขนาดต้นฉบับกี่เท่า (default: 2 ปกติ, 1 สำหรับ grid)")
    p.add_argument("--vmax", type=float, default=1000.0, help="ค่าสนามแม่เหล็กสูงสุดของสเกลสีเทา magnetogram (G)")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--harp", type=int, default=None,
        help="ครอบตัดทุกเลเยอร์เหลือแค่หน้าต่างรอบ HARP นี้ดวงเดียว (ดูตัวอย่างการใช้งานด้านบน)",
    )
    p.add_argument("--harp-window", type=int, default=220, help="ขนาดหน้าต่างครอบตัด (พิกเซล บนกริดเต็มดวง 512)")
    p.add_argument("--harp-max-gap-hours", type=float, default=3.0, help="ข้ามเฟรมถ้าไม่มีข้อมูล SHARP ของ HARP ใกล้กว่านี้")
    p.add_argument("--harp-max-abs-lon", type=float, default=88.0, help="ข้ามเฟรมถ้า HARP อยู่เกินองศานี้จากศูนย์กลางจาน")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# เรนเดอร์ภาพพื้นแต่ละชนิด (ยังไม่ resize, ไม่มีแถบข้อความ)
# --------------------------------------------------------------------------- #


def mag_panel_rgb(magnetogram: np.ndarray, vmax: float) -> np.ndarray:
    raw = np.asarray(magnetogram, dtype=np.float32)
    off_disk = ~np.isfinite(raw)
    scaled = np.clip(np.nan_to_num(raw) / vmax, -1.0, 1.0)
    grey = ((scaled + 1.0) * 127.5).astype(np.uint8)
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    rgb[off_disk] = 0
    return rgb


def _aia_normalized(image: np.ndarray, vmax: float) -> np.ndarray:
    """ยืด dynamic range ของภาพ AIA ด้วย AsinhStretch เดียวกับ ``aia_to_png`` คืนค่า 0..1"""
    from astropy.visualization import AsinhStretch, ImageNormalize

    array = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0)
    norm = ImageNormalize(vmin=0.0, vmax=float(vmax), stretch=AsinhStretch(0.02), clip=True)
    return norm(array)


def aia_panel_rgb(image: np.ndarray, wavelength: int, vmax: float) -> np.ndarray:
    """ย้อมสีภาพ AIA ตามธรรมเนียมของ SDO — ตรรกะเดียวกับ ``aia_to_png`` ในเว็บแอป

    คัดลอกมาแทนที่จะเรียก ``aia_to_png`` ตรง ๆ เพราะที่นี่ต้องได้ RGB ดิบ (ไม่ผ่าน PNG/base64)
    และต้องวาดเส้นขอบสองเส้น (ทำนาย + ground truth) ซึ่ง ``aia_to_png`` รับ mask ได้แค่อันเดียว
    """
    return (_colormap_for(wavelength)(_aia_normalized(image, vmax))[..., :3] * 255).astype(np.uint8)


def tri_color_composite(images: dict[str, np.ndarray], aia_cfg) -> np.ndarray:
    """รวมสามช่อง AIA เป็นภาพสีเดียว: แดง=304 (โครโมสเฟียร์) เขียว=171 (โคโรนา) น้ำเงิน=1600
    (โฟโตสเฟียร์) — วิธีมาตรฐานที่ภาพ tri-color ของ SDO ใช้กัน (แต่ละช่องคือ *ช่องสี* ไม่ใช่
    เอา colormap เดิมของแต่ละช่องมา screen blend กัน)

    ไม่รวม magnetogram เข้าไปด้วย: magnetogram เป็นสนามแม่เหล็กแนวสายตา (เทากลางคือ 0) ไม่ใช่
    ความสว่าง ผสมเข้ากับภาพความสว่างสามช่องนี้ตรง ๆ แล้วดันโทนสีทั้งภาพไปทางขาวจนโครงสร้าง
    ของแต่ละชั้นแยกไม่ออก (ลองแล้วด้วย screen blend) — ใช้เส้นขอบ mask ที่มาจาก U-Net/SHARP
    แทนเพื่อบอกตำแหน่งสนามแม่เหล็กแรงแทน
    """
    r = _aia_normalized(images["304"], aia_cfg.channel("304").display_vmax)
    g = _aia_normalized(images["171"], aia_cfg.channel("171").display_vmax)
    b = _aia_normalized(images["1600"], aia_cfg.channel("1600").display_vmax)
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255).astype(np.uint8)


def with_overlay(
    rgb: np.ndarray,
    predicted_mask: np.ndarray,
    truth_mask: np.ndarray | None,
    detections,
    predicted_color: tuple[int, int, int],
) -> np.ndarray:
    if truth_mask is not None:
        rgb = draw_mask_overlay(rgb, truth_mask, COLOR_TRUTH, detections=None)
    return draw_mask_overlay(rgb, predicted_mask, predicted_color, detections=detections)


def add_caption(rgb: np.ndarray, text: str) -> np.ndarray:
    """ป้ายชื่อเลเยอร์มุมซ้ายบน — ใช้ในโหมด grid ให้แยกแต่ละช่องออกจากกัน"""
    rgb = rgb.copy()
    cv2.putText(rgb, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(rgb, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def load_harp_track(cfg, harpnum: int) -> pd.DataFrame:
    """แถวทั้งหมดของ HARP นี้จาก ``sharp_keywords.parquet`` ที่มีตำแหน่งใช้ได้ เรียงตามเวลา"""
    path = cfg.paths.interim / "sharp_keywords.parquet"
    df = pd.read_parquet(path, columns=["HARPNUM", "t_rec", "LAT_FWT", "LON_FWT", "NOAA_ARS"])
    sub = df[(df["HARPNUM"] == harpnum) & df["LAT_FWT"].notna() & df["LON_FWT"].notna()]
    return sub.sort_values("t_rec").reset_index(drop=True)


def harp_pixel_center(
    track: pd.DataFrame,
    moment,
    frame_wcs: FrameWcsStore,
    frame_name: str,
    size: int,
    max_gap_hours: float,
    max_abs_lon: float,
) -> tuple[float, float] | None:
    """พิกเซลกึ่งกลาง HARP ในเฟรมนี้ **หลังหมุน 180°** แล้ว — ``None`` ถ้าไม่มีข้อมูลใกล้พอ

    ใช้ ``LAT_FWT``/``LON_FWT`` จริงของ HARP (ไม่ใช่ detection ที่ทำนาย) เป็นศูนย์กลางตัด
    เพราะแม่นกว่าและไม่ขึ้นกับความผิดพลาดของ U-Net — แปลงผ่าน WCS จริงของเฟรมได้พิกเซลใน
    ทิศ CCD ดิบ (เหมือน ``_solar_map_for`` ใน backend/app/routers/segment.py) จึงต้องกลับพิกัด
    ``size - 1 - x`` ให้ตรงกับภาพที่สคริปต์นี้หมุน 180° ไปแล้วก่อนเรนเดอร์
    """
    if track.empty:
        return None

    deltas = (track["t_rec"] - moment).abs()
    best = deltas.idxmin()
    row = track.loc[best]
    gap_hours = deltas.loc[best].total_seconds() / 3600
    if gap_hours > max_gap_hours or abs(float(row["LON_FWT"])) > max_abs_lon:
        return None

    solar_map = frame_wcs.solar_map(frame_name, np.zeros((size, size), dtype=np.float32))
    if solar_map is None:
        return None
    x_raw, y_raw = stonyhurst_to_pixel(float(row["LON_FWT"]), float(row["LAT_FWT"]), solar_map)
    return size - 1 - float(x_raw), size - 1 - float(y_raw)


def compose_frame(rgb: np.ndarray, scale: int, label_lines: list[str]) -> np.ndarray:
    """ขยายภาพแล้วแปะแถบข้อความด้านล่าง (กันข้อความไปทับโครงสร้างในภาพ)"""
    if scale > 1:
        size = (rgb.shape[1] * scale, rgb.shape[0] * scale)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_NEAREST)

    bar_h = 22 * len(label_lines) + 14
    canvas = np.zeros((rgb.shape[0] + bar_h, rgb.shape[1], 3), dtype=np.uint8)
    canvas[: rgb.shape[0]] = rgb
    for i, line in enumerate(label_lines):
        cv2.putText(
            canvas,
            line,
            (10, rgb.shape[0] + 20 + 22 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    args = parse_args()
    setup_logging()
    cfg = load_data_config()
    tracking_cfg = load_tracking_config()
    if args.scale is not None:
        scale = args.scale
    elif args.harp is not None:
        scale = 3
    elif args.layer in ("grid", "row"):
        scale = 1
    else:
        scale = 2
    needs_all_aia = args.layer in ("grid", "row", "stack")

    seg = SegmentationService(
        cfg.paths.artifacts / "models" / "unet.pt",
        frames_dir=cfg.paths.processed / "frames",
        device=args.device,
    )
    if not seg.available:
        logger.error(
            "ยังไม่มีโมเดล U-Net ที่ใช้งานได้ (%s) — รัน backend/scripts/train_unet.py ก่อน",
            seg.checkpoint_path,
        )
        return 1

    aia_store = AiaFrameStore(cfg.aia.root)
    needs_aia = args.layer in AIA_LAYERS or needs_all_aia
    if needs_aia and not aia_store.available:
        logger.error(
            "ยังไม่มีภาพ AIA ที่ %s — รัน backend/scripts/download_aia.py ก่อน หรือใช้ --layer mag",
            aia_store.root,
        )
        return 1

    harp_track: pd.DataFrame | None = None
    frame_wcs: FrameWcsStore | None = None
    if args.harp is not None:
        frame_wcs = FrameWcsStore(cfg.paths.interim / "frame_wcs.parquet")
        if not frame_wcs.available:
            logger.error(
                "ไม่มี WCS จริงของเฟรมที่ %s — รัน backend/scripts/download_aia.py --wcs-only ก่อน",
                frame_wcs.path,
            )
            return 1
        harp_track = load_harp_track(cfg, args.harp)
        if harp_track.empty:
            logger.error("ไม่พบข้อมูล SHARP ของ HARP %d ใน sharp_keywords.parquet", args.harp)
            return 1
        noaa_ars = harp_track["NOAA_ARS"].dropna().iloc[0] if harp_track["NOAA_ARS"].notna().any() else "?"
        logger.info(
            "ครอบตัดเฉพาะ HARP %d (NOAA %s) — มีข้อมูลตำแหน่งตั้งแต่ %s ถึง %s",
            args.harp, noaa_ars, harp_track["t_rec"].min(), harp_track["t_rec"].max(),
        )

    frames = seg.list_frames()
    selected = [
        name
        for name in frames
        if args.start <= f"{name[:4]}-{name[4:6]}-{name[6:8]}" < args.end
    ]
    if not selected:
        logger.error("ไม่พบเฟรมในช่วง %s ถึง %s (มีทั้งหมด %d เฟรม)", args.start, args.end, len(frames))
        return 1
    logger.info("พบ %d เฟรมในช่วง %s ถึง %s (เลเยอร์: %s)", len(selected), args.start, args.end, args.layer)

    layer_suffix = "" if args.layer == "mag" else f"{args.layer}_"
    harp_suffix = f"harp{args.harp}_" if args.harp is not None else ""
    out_path = Path(args.out) if args.out else (
        cfg.paths.artifacts / "figures" / f"segmentation_{harp_suffix}{layer_suffix}{args.start}_{args.end}.mp4"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    min_area_px = tracking_cfg.detect.min_area_px
    first_size = None
    ffmpeg_proc: subprocess.Popen | None = None
    n_written = 0
    n_skipped_no_aia = 0
    dice_scores: list[float] = []

    for name in selected:
        try:
            magnetogram, predicted_mask, _detections, dice, has_truth = seg.frame_masks(
                name, use_ground_truth=False, min_area_px=min_area_px
            )
        except (RuntimeError, FileNotFoundError) as exc:
            logger.warning("ข้ามเฟรม %s: %s", name, exc)
            continue

        _, truth_mask = seg.load_frame(name)
        if dice is not None:
            dice_scores.append(dice)

        # หมุน 180° แก้ทิศ CCD ดิบของ HMI/AIA ก่อนเรนเดอร์ (ดู docstring บนสุดของไฟล์) —
        # ต้องหมุน mask ทำนายด้วยแล้วหา detection ใหม่ ไม่งั้นเลขกำกับ AR จะเยื้องตำแหน่ง
        magnetogram = np.rot90(magnetogram, 2)
        predicted_mask = np.rot90(predicted_mask, 2)
        truth_mask = np.rot90(truth_mask, 2) if truth_mask is not None else None
        detections = detect_regions(predicted_mask, magnetogram=magnetogram, min_area_px=min_area_px)
        truth_for_overlay = truth_mask if has_truth else None

        aia_images: dict[str, np.ndarray] = {}
        if needs_aia:
            missing = []
            for key in AIA_LAYERS if needs_all_aia else [args.layer]:
                image = aia_store.load(name, key)
                if image is None:
                    missing.append(key)
                    continue
                aia_images[key] = np.rot90(image, 2)
            if missing:
                logger.warning("ข้ามเฟรม %s: ไม่มีภาพ AIA ช่อง %s", name, ", ".join(missing))
                n_skipped_no_aia += 1
                continue

        moment = parse_frame_timestamp(name)

        if harp_track is not None:
            size = magnetogram.shape[0]
            center = harp_pixel_center(
                harp_track, moment, frame_wcs, name, size,
                args.harp_max_gap_hours, args.harp_max_abs_lon,
            )
            if center is None:
                logger.debug("ข้ามเฟรม %s: HARP %d ไม่อยู่บนจาน (หรือไม่มีข้อมูล SHARP ใกล้พอ)", name, args.harp)
                continue
            cx, cy = center
            window = min(args.harp_window, size)
            half = window // 2
            x0 = int(np.clip(cx - half, 0, size - window))
            y0 = int(np.clip(cy - half, 0, size - window))
            x1, y1 = x0 + window, y0 + window

            magnetogram = magnetogram[y0:y1, x0:x1]
            predicted_mask = predicted_mask[y0:y1, x0:x1]
            truth_mask = truth_mask[y0:y1, x0:x1] if truth_mask is not None else None
            for key in aia_images:
                aia_images[key] = aia_images[key][y0:y1, x0:x1]
            detections = detect_regions(predicted_mask, magnetogram=magnetogram, min_area_px=min_area_px)
            truth_for_overlay = truth_mask if has_truth else None

        pretty = moment.strftime("%Y-%m-%d %H:%M UT")
        dice_txt = f"   Dice vs SHARP: {dice:.3f}" if dice is not None else ""

        if args.layer == "mag":
            base = mag_panel_rgb(magnetogram, args.vmax)
            base = with_overlay(base, predicted_mask, truth_for_overlay, detections, COLOR_PREDICTED_MAG)
            label_lines = [
                f"{pretty}   HMI magnetogram   U-Net prediction (orange)"
                + ("  vs SHARP ground truth (green)" if has_truth else ""),
                f"AR regions: {len(detections)}   predicted area: {100 * predicted_mask.mean():.2f}%{dice_txt}",
            ]
            frame = compose_frame(base, scale, label_lines)

        elif args.layer in AIA_LAYERS:
            channel = cfg.aia.channel(args.layer)
            base = aia_panel_rgb(aia_images[args.layer], channel.wavelength, channel.display_vmax)
            base = with_overlay(base, predicted_mask, truth_for_overlay, detections, MASK_COLOR_AIA)
            label_lines = [
                f"{pretty}   {AIA_LAYERS[args.layer]}   U-Net prediction (cyan)"
                + ("  vs SHARP ground truth (green)" if has_truth else ""),
                f"AR regions: {len(detections)}   predicted area: {100 * predicted_mask.mean():.2f}%{dice_txt}",
            ]
            frame = compose_frame(base, scale, label_lines)

        elif args.layer == "stack":
            composite = tri_color_composite(aia_images, cfg.aia)
            composite = with_overlay(
                composite, predicted_mask, truth_for_overlay, detections, COLOR_PREDICTED_STACK
            )
            label_lines = [
                f"{pretty}   tri-color: R=304 G=171 B=1600   U-Net prediction (white)"
                + ("  vs SHARP ground truth (green)" if has_truth else ""),
                f"AR regions: {len(detections)}   predicted area: {100 * predicted_mask.mean():.2f}%{dice_txt}",
            ]
            frame = compose_frame(composite, scale, label_lines)

        else:  # grid / row
            mag_base = with_overlay(
                mag_panel_rgb(magnetogram, args.vmax),
                predicted_mask, truth_for_overlay, detections, COLOR_PREDICTED_MAG,
            )
            panels = [add_caption(mag_base, "HMI magnetogram")]
            for key in ("1600", "304", "171"):
                channel = cfg.aia.channel(key)
                panel = aia_panel_rgb(aia_images[key], channel.wavelength, channel.display_vmax)
                panel = with_overlay(panel, predicted_mask, truth_for_overlay, detections, MASK_COLOR_AIA)
                panels.append(add_caption(panel, AIA_LAYERS[key]))

            if args.layer == "row":
                base = np.concatenate(panels, axis=1)
            else:
                top = np.concatenate([panels[0], panels[1]], axis=1)
                bottom = np.concatenate([panels[2], panels[3]], axis=1)
                base = np.concatenate([top, bottom], axis=0)
            label_lines = [
                f"{pretty}   U-Net prediction (orange/cyan)"
                + ("  vs SHARP ground truth (green)" if has_truth else ""),
                f"AR regions: {len(detections)}   predicted area: {100 * predicted_mask.mean():.2f}%{dice_txt}",
            ]
            frame = compose_frame(base, scale, label_lines)

        if ffmpeg_proc is None:
            first_size = (frame.shape[1], frame.shape[0])  # (width, height)
            ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-pixel_format", "rgb24",
                    "-video_size", f"{first_size[0]}x{first_size[1]}",
                    "-framerate", str(args.fps),
                    "-i", "-",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(out_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        ffmpeg_proc.stdin.write(frame.tobytes())
        n_written += 1
        if n_written % 20 == 0:
            logger.info("เรนเดอร์แล้ว %d/%d เฟรม", n_written, len(selected))

    if ffmpeg_proc is None:
        logger.error("ไม่มีเฟรมใดเรนเดอร์ได้สำเร็จเลย")
        return 1

    ffmpeg_proc.stdin.close()
    stderr = ffmpeg_proc.stderr.read().decode("utf-8", errors="replace")
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        logger.error("ffmpeg ล้มเหลว (exit %d):\n%s", ffmpeg_proc.returncode, stderr[-2000:])
        return 1

    logger.info("-" * 62)
    logger.info("เขียนวิดีโอ %d เฟรม (%d fps, %dx%d) ไปที่ %s", n_written, args.fps, *first_size, out_path)
    if n_skipped_no_aia:
        logger.info("ข้ามไป %d เฟรมเพราะไม่มีภาพ AIA ครบทุกช่อง", n_skipped_no_aia)
    if dice_scores:
        logger.info(
            "Dice เทียบ SHARP เฉลี่ย %.4f (n=%d เฟรมที่มี ground truth)",
            float(np.mean(dice_scores)),
            len(dice_scores),
        )
    logger.info("-" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
