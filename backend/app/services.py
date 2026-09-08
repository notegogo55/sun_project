"""ตัวรวมบริการทั้งหมดที่แอปใช้ — โหลดครั้งเดียวตอน startup

ออกแบบให้ **ทำงานต่อได้แม้ส่วนประกอบบางอย่างยังไม่พร้อม** ผู้ใช้ที่เพิ่งเริ่มโปรเจค
จะยังไม่มีโมเดลที่เทรนแล้ว แอปจึงต้องเปิดได้และบอกอย่างชัดเจนว่าต้องรันสคริปต์ไหน
ต่อ แทนที่จะล่มตั้งแต่ import
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sunseg.config import DataConfig, load_data_config, load_tracking_config
from sunseg.data.aia import AiaFrameStore
from sunseg.data.frame_wcs import FrameWcsStore
from sunseg.data.proton_flux import ProtonFluxStore
from sunseg.data.xray_flux import XrayFluxStore
from sunseg.inference.forecast import ForecastService, SequenceStore
from sunseg.inference.predictions_store import PredictionsStore
from sunseg.inference.segment import SegmentationService

logger = logging.getLogger(__name__)


class AppServices:
    """รวมโมเดล ข้อมูล และ config ไว้ที่เดียว"""

    def __init__(self, device: str = "cpu") -> None:
        self.config: DataConfig = load_data_config()
        self.tracking_config = load_tracking_config()
        self.device = device

        artifacts = self.config.paths.artifacts
        processed = self.config.paths.processed

        self.forecast = ForecastService(artifacts / "models" / "lstm.pt", device=device)
        self.predictions = PredictionsStore(
            artifacts / "metrics" / "predictions.parquet", artifacts / "metrics" / "lstm.json"
        )
        self.segmentation = SegmentationService(
            artifacts / "models" / "unet.pt",
            frames_dir=processed / "frames",
            device=device,
        )
        self.sequences = SequenceStore(processed / "sequences")
        self.flares = self._load_flares(self.config.paths.interim / "flares.parquet")
        self.proton = ProtonFluxStore(self.config.proton.root)
        self.xray = XrayFluxStore(self.config.paths.raw / "xrs")
        self.frame_wcs = FrameWcsStore(self.config.paths.interim / "frame_wcs.parquet")
        self.aia = AiaFrameStore(self.config.aia.root)

        self._log_readiness()

    @staticmethod
    def _load_flares(path: Path) -> pd.DataFrame | None:
        if not path.exists():
            logger.warning("ไม่พบรายการ flare ที่ %s — แผง GOES จะไม่มีข้อมูล", path)
            return None

        df = pd.read_parquet(path)
        df["peak_time"] = pd.to_datetime(df["peak_time"])
        # ไฟล์ถูกคลี่เป็นคู่ (flare, HARP) — ยุบกลับเป็น event ที่ไม่ซ้ำสำหรับกราฟเส้นเวลา
        logger.info("โหลดรายการ flare: %d คู่ (flare, HARP)", len(df))
        return df.sort_values("peak_time").reset_index(drop=True)

    def _log_readiness(self) -> None:
        checks = [
            (
                "โมเดลพยากรณ์ flare (LSTM)",
                self.forecast.available,
                "รัน backend/scripts/train_lstm.py",
            ),
            (
                "โมเดล segmentation (U-Net)",
                self.segmentation.available,
                "รัน backend/scripts/train_unet.py",
            ),
            (
                "ข้อมูล sequence ย้อนหลัง",
                self.sequences.available,
                "รัน backend/scripts/build_sequences.py",
            ),
            (
                "ค่าทำนายราย sample (แผง confusion matrix)",
                self.predictions.available,
                "รัน backend/scripts/train_lstm.py",
            ),
            (
                "รายการ flare (GOES)",
                self.flares is not None,
                "รัน backend/scripts/download_metadata.py",
            ),
            # ไม่ใช่ผลจากสคริปต์ในโปรเจคนี้ — เป็นคลังภายนอกที่ชี้ด้วย SUNSEG_PROTON_DIR
            ("ฟลักซ์โปรตอน (GOES particle)", self.proton.available, "ตั้ง SUNSEG_PROTON_DIR ใน .env"),
            (
                "ฟลักซ์ X-ray ต่อเนื่อง (GOES-15/16)",
                self.xray.available,
                "รัน backend/scripts/download_xray.py",
            ),
            (
                "WCS ของเฟรม (ใช้วางภาพ AIA + พิกัด)",
                self.frame_wcs.available,
                "รัน backend/scripts/download_aia.py --wcs-only",
            ),
            (
                "ภาพ AIA สามชั้นบรรยากาศ",
                self.aia.available,
                "รัน backend/scripts/download_aia.py",
            ),
        ]

        logger.info("-" * 62)
        logger.info("สถานะความพร้อมของแอป")
        for label, ready, hint in checks:
            if ready:
                logger.info("  [ OK ] %s", label)
            else:
                logger.info("  [ -- ] %s  -> %s", label, hint)
        logger.info("-" * 62)

    # ------------------------------------------------------------------ #

    def info(self) -> dict:
        return {
            "forecast": self.forecast.info(),
            "segmentation": self.segmentation.info(),
            "data": {
                "sequence_store": self.sequences.available,
                "n_sequences": 0 if self.sequences.meta is None else len(self.sequences.meta),
                "n_flare_records": 0 if self.flares is None else len(self.flares),
                "proton": self.proton.info(),
                "xray": self.xray.info(),
                "aia": self.aia.info(),
                "frame_wcs": self.frame_wcs.info(),
                "time_range": {
                    "start": self.config.time_range.start.isoformat(),
                    "end": self.config.time_range.end.isoformat(),
                },
                "horizon_hours": self.config.flare.horizon_hours,
                "positive_class": self.config.flare.positive_goes_class,
            },
        }
