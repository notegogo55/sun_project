"""บริการ segmentation + tracking — ห่อ U-Net checkpoint ให้ webapp เรียกใช้"""

from __future__ import annotations

import base64
import io
import logging
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ..data.aia import MASK_COLOR_MAG, draw_mask_overlay
from ..metrics import dice_coefficient
from ..models.unet import UNet, normalise_magnetogram
from ..tracking.detect import Detection, detect_regions

#: จำนวนเฟรมที่เก็บผลการ segment ไว้ในหน่วยความจำ
#:
#: การสลับ *เลเยอร์* ที่แสดง (magnetogram / AIA) ไม่ได้เปลี่ยน mask เลย — mask มาจาก
#: U-Net ที่มองแค่ magnetogram เท่านั้น ถ้าไม่ cache ไว้ การกดสลับเลเยอร์แต่ละครั้งจะ
#: รัน U-Net ใหม่ทั้งที่ได้ผลเดิมเป๊ะ  ~4 MB ต่อรายการ (mask เต็มเฟรมของทุก detection
#: เป็นตัวกินที่พักหลัก) เพดานจึงราว 32 MB
_CACHE_SIZE = 8

logger = logging.getLogger(__name__)


class SegmentationService:
    """โหลด U-Net ที่เทรนแล้ว และให้บริการ segment ภาพ magnetogram"""

    def __init__(
        self,
        checkpoint_path: Path,
        frames_dir: Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.frames_dir = Path(frames_dir)
        self.device = torch.device(device)
        self.available = False

        self.model: UNet | None = None
        self.threshold: float = 0.5
        self.image_size: int = 512
        self.norm_scale: float = 300.0
        self.metrics: dict = {}

        # (timestamp, use_ground_truth) -> ผลของ frame_masks()
        self._frame_cache: OrderedDict[tuple[str, bool], tuple] = OrderedDict()

        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            logger.warning(
                "ไม่พบ checkpoint ของ U-Net ที่ %s — ฟีเจอร์ segmentation จะปิดใช้งาน "
                "(รัน backend/scripts/segmentation/train.py เพื่อสร้าง)",
                self.checkpoint_path,
            )
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = payload["config"]

            self.threshold = float(config.get("threshold", 0.5))
            self.image_size = int(config.get("image_size", 512))
            self.norm_scale = float(config.get("norm_scale_gauss", 300.0))
            self.metrics = payload.get("metrics", {})

            model_cfg = config["model"]
            self.model = UNet(
                in_channels=model_cfg["in_channels"],
                out_channels=model_cfg["out_channels"],
                base_channels=model_cfg["base_channels"],
                depth=model_cfg["depth"],
                norm=model_cfg["norm"],
                bilinear_upsample=model_cfg["bilinear_upsample"],
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device).eval()
            self.available = True

            dice = self.metrics.get("test", {}).get("dice")
            logger.info(
                "โหลดโมเดล segmentation สำเร็จ: %dpx, threshold %.2f%s",
                self.image_size,
                self.threshold,
                f", test Dice {dice:.4f}" if dice is not None else "",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("โหลด checkpoint ของ U-Net ไม่สำเร็จ: %s", exc)
            self.available = False

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def segment(self, magnetogram: np.ndarray) -> np.ndarray:
        """คืนแผนที่ความน่าจะเป็นขนาดเท่ากับ input"""
        if not self.available or self.model is None:
            raise RuntimeError(
                "ยังไม่มีโมเดล segmentation ที่ใช้งานได้ — รัน backend/scripts/segmentation/train.py ก่อน"
            )

        data = normalise_magnetogram(np.asarray(magnetogram, dtype=np.float32), self.norm_scale)
        tensor = torch.from_numpy(np.nan_to_num(data))[None, None].to(self.device)

        logits = self.model(tensor)
        return torch.sigmoid(logits)[0, 0].cpu().numpy()

    def segment_and_detect(
        self,
        magnetogram: np.ndarray,
        min_area_px: int = 12,
        solar_map=None,
    ) -> tuple[np.ndarray, list[Detection]]:
        """segment แล้วแยกเป็นรายดวงพร้อมคุณสมบัติ"""
        probability = self.segment(magnetogram)
        mask = (probability >= self.threshold).astype(np.uint8)
        detections = detect_regions(
            mask,
            magnetogram=magnetogram,
            min_area_px=min_area_px,
            solar_map=solar_map,
        )
        return probability, detections

    # ------------------------------------------------------------------ #

    def frame_masks(
        self,
        timestamp: str,
        use_ground_truth: bool,
        min_area_px: int = 12,
        solar_map=None,
    ) -> tuple[np.ndarray, np.ndarray, list[Detection], float | None, bool]:
        """โหลดเฟรม + หา mask และ AR ทั้งหมด โดย cache ผลไว้ให้เรียกซ้ำได้ถูก ๆ

        เป็นทางเข้าเดียวที่ endpoint ควรใช้ เพราะทั้งภาพที่แสดงและตัวเลขความเข้มแสง
        ต้องมาจาก mask **อันเดียวกัน** ถ้าแยกกันคำนวณจะมีโอกาสที่เส้นขอบบนภาพกับแท่ง
        กราฟไม่ตรงกัน

        Parameters
        ----------
        use_ground_truth
            ``True`` ใช้ mask จาก SHARP (ใช้ได้แม้ยังไม่ได้เทรน U-Net)
            ``False`` ใช้ผลทำนายของ U-Net

        Returns
        -------
        ``(magnetogram, mask, detections, dice, has_ground_truth)`` โดย ``dice`` เป็น
        ``None`` เมื่อเทียบไม่ได้ (ไม่มี ground truth หรือกำลังแสดง ground truth อยู่)
        """
        key = (timestamp, bool(use_ground_truth))
        if key in self._frame_cache:
            self._frame_cache.move_to_end(key)
            return self._frame_cache[key]

        magnetogram, truth = self.load_frame(timestamp)

        if use_ground_truth:
            if truth is None:
                raise RuntimeError(f"เฟรม {timestamp} ไม่มี mask จริงกำกับ")
            mask = truth
            dice = None
        else:
            if not self.available:
                raise RuntimeError(
                    "ยังไม่มีโมเดล segmentation — รัน backend/scripts/segmentation/train.py ก่อน "
                    "หรือใช้ use_ground_truth=true เพื่อดู mask จาก SHARP"
                )
            probability = self.segment(magnetogram)
            mask = (probability >= self.threshold).astype(np.uint8)
            dice = float(dice_coefficient(mask, truth)) if truth is not None else None

        detections = detect_regions(
            mask,
            magnetogram=magnetogram,
            min_area_px=min_area_px,
            solar_map=solar_map,
        )

        result = (magnetogram, mask, detections, dice, truth is not None)
        self._frame_cache[key] = result
        if len(self._frame_cache) > _CACHE_SIZE:
            self._frame_cache.popitem(last=False)
        return result

    # ------------------------------------------------------------------ #

    def list_frames(self) -> list[str]:
        """รายการเวลาของเฟรมที่ประมวลผลไว้แล้วบนดิสก์"""
        image_dir = self.frames_dir / "images"
        if not image_dir.exists():
            return []
        return sorted(p.stem for p in image_dir.glob("*.npy"))

    def load_frame(self, timestamp: str) -> tuple[np.ndarray, np.ndarray | None]:
        """โหลดคู่ (magnetogram, mask ที่เป็น ground truth) ของเวลาที่ระบุ"""
        image_path = self.frames_dir / "images" / f"{timestamp}.npy"
        if not image_path.exists():
            raise FileNotFoundError(f"ไม่พบเฟรมเวลา {timestamp} ที่ {image_path}")

        magnetogram = np.load(image_path).astype(np.float32)

        mask_path = self.frames_dir / "masks" / f"{timestamp}.npy"
        mask = np.load(mask_path).astype(np.uint8) if mask_path.exists() else None
        return magnetogram, mask

    def info(self) -> dict:
        return {
            "available": self.available,
            "checkpoint": str(self.checkpoint_path),
            "image_size": self.image_size,
            "threshold": round(self.threshold, 4),
            "n_frames": len(self.list_frames()),
            "metrics": self.metrics,
        }


# --------------------------------------------------------------------------- #
# การเรนเดอร์ภาพสำหรับส่งไปแสดงบนเว็บ
# --------------------------------------------------------------------------- #


def magnetogram_to_png(
    magnetogram: np.ndarray,
    vmax: float = 1500.0,
    mask: np.ndarray | None = None,
    detections=None,
) -> str:
    """เรนเดอร์ magnetogram เป็น PNG (เข้ารหัส base64) พร้อมขอบ mask ทับ

    ใช้สเกลเทาแบบมาตรฐานของ HMI: ขั้วบวกเป็นสีขาว ขั้วลบเป็นสีดำ และ quiet Sun
    เป็นสีเทากลาง เมื่อมี mask จะวาดเฉพาะ **เส้นขอบ** ไม่ระบายทึบ เพื่อไม่ให้บัง
    โครงสร้างของสนามแม่เหล็กที่อยู่ข้างใต้

    ``detections`` ถ้าระบุ จะเขียนเลขลำดับกำกับแต่ละดวง ให้โยงกับแท่งกราฟความเข้มแสง
    ในหน้าเว็บได้ — ใช้ตัววาดตัวเดียวกับเลเยอร์ AIA เพื่อให้ทั้งสองฝั่งหน้าตาตรงกัน

    **พื้นที่นอกจานถูกวาดเป็นสีดำ** ไม่ใช่เทากลาง: HMI ใส่ ``NaN`` ไว้นอกขอบจานอยู่แล้ว
    (ราว 25% ของภาพ) ถ้าแปลงเป็นศูนย์ตรง ๆ มันจะได้สีเดียวกับ quiet Sun พอดี ทำให้ขอบจาน
    หายไปและภาพดูเป็นสี่เหลี่ยมเทาทึบ — ต่างจากเลเยอร์ AIA ที่เห็นเป็นดวงกลมชัดเจน
    """
    import cv2
    from PIL import Image

    raw = np.asarray(magnetogram, dtype=np.float32)
    off_disk = ~np.isfinite(raw)

    scaled = np.clip(np.nan_to_num(raw) / vmax, -1.0, 1.0)
    grey = ((scaled + 1.0) * 127.5).astype(np.uint8)
    rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    rgb[off_disk] = 0

    rgb = draw_mask_overlay(rgb, mask, MASK_COLOR_MAG, detections)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_frame_timestamp(timestamp: str) -> datetime:
    """แปลงชื่อไฟล์เฟรม (``20140101_000000``) กลับเป็น datetime"""
    return datetime.strptime(timestamp, "%Y%m%d_%H%M%S")


def format_frame_timestamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%d_%H%M%S")
