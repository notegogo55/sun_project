"""ตัวรวมบริการทั้งหมดที่แอปใช้ — โหลดครั้งเดียวตอน startup

ออกแบบให้ **ทำงานต่อได้แม้ส่วนประกอบบางอย่างยังไม่พร้อม** ผู้ใช้ที่เพิ่งเริ่มโปรเจค
จะยังไม่มีโมเดลที่เทรนแล้ว แอปจึงต้องเปิดได้และบอกอย่างชัดเจนว่าต้องรันสคริปต์ไหน
ต่อ แทนที่จะล่มตั้งแต่ import
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sunseg.config import DataConfig, load_data_config, load_forecast_config, load_tracking_config
from sunseg.data.aia import AiaFrameStore
from sunseg.data.flare_positions import FlarePositionStore
from sunseg.data.frame_wcs import FrameWcsStore
from sunseg.data.proton_flux import ProtonFluxStore
from sunseg.data.xray_flux import XrayFluxStore
from sunseg.inference.class_forecast import ClassForecastService
from sunseg.inference.forecast import ForecastModels, SequenceStore
from sunseg.inference.predictions_store import ForecastPredictions
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

        # โมเดลพยากรณ์ทุกตัวใน configs/forecast.yaml — ตัวที่ยังไม่ได้เทรนอยู่ในรายการด้วย
        # (available=False) เพื่อให้หน้าเว็บบอกได้ว่าต้องรันคำสั่งไหน
        self.forecast_config = load_forecast_config()
        self.forecast = ForecastModels(self.forecast_config, artifacts, device=device)
        self.predictions = ForecastPredictions.from_artifacts(
            {name: service.label for name, service in self.forecast.services.items()},
            artifacts,
            default_name=self.forecast.default_name,
        )
        # โมเดลหลักของโปรเจค (LSTM + V3 แยกระดับ <M/M/X) — ensemble ทุก seed คำนวณทั้ง study dataset ครั้งเดียวที่นี่
        self.class_forecast = ClassForecastService(
            self.forecast_config.class_forecast, artifacts, processed, device=device
        )
        self.segmentation = SegmentationService(
            artifacts / "models" / "unet.pt",
            frames_dir=processed / "frames",
            device=device,
        )
        self.sequences = SequenceStore(processed / "sequences")
        self.flares = self._load_flares(self.config.paths.interim / "flares.parquet")
        self.flare_positions = FlarePositionStore(processed / "flare_positions.parquet")
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
                f"โมเดลหลัก: {self.class_forecast.label} แยกระดับ <M/M/X",
                self.class_forecast.available,
                f"รัน {self.class_forecast.hint}",
            ),
            *[
                (
                    f"โมเดลพยากรณ์ flare: {service.label}"
                    + (" (ปริยาย)" if name == self.forecast.default_name else ""),
                    service.available,
                    f"รัน {service.train_hint}",
                )
                for name, service in self.forecast.services.items()
            ],
            (
                "โมเดล segmentation (U-Net)",
                self.segmentation.available,
                "รัน backend/scripts/segmentation/train.py",
            ),
            (
                "ข้อมูล sequence ย้อนหลัง",
                self.sequences.available,
                "รัน backend/scripts/data/build_sequences.py",
            ),
            (
                "ค่าทำนายราย sample (แผง confusion matrix)",
                self.predictions.available,
                "รัน backend/scripts/forecast/train.py --model <ชื่อ>",
            ),
            (
                "รายการ flare (GOES)",
                self.flares is not None,
                "รัน backend/scripts/data/download_metadata.py",
            ),
            (
                "ตำแหน่ง flare จาก PositionFlare (แผนที่หน้าแรก)",
                self.flare_positions.available,
                "รัน backend/scripts/data/build_flare_positions.py",
            ),
            # ไม่ใช่ผลจากสคริปต์ในโปรเจคนี้ — เป็นคลังภายนอกที่ชี้ด้วย SUNSEG_PROTON_DIR
            ("ฟลักซ์โปรตอน (GOES particle)", self.proton.available, "ตั้ง SUNSEG_PROTON_DIR ใน .env"),
            (
                "ฟลักซ์ X-ray ต่อเนื่อง (GOES-15/16)",
                self.xray.available,
                "รัน backend/scripts/data/download_xray.py",
            ),
            (
                "WCS ของเฟรม (ใช้วางภาพ AIA + พิกัด)",
                self.frame_wcs.available,
                "รัน backend/scripts/data/download_aia.py --wcs-only",
            ),
            (
                "ภาพ AIA สามชั้นบรรยากาศ",
                self.aia.available,
                "รัน backend/scripts/data/download_aia.py",
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
            # "forecast" คือโมเดลปริยาย (รูปแบบเดิม) ส่วน "forecast_models" คือทุกตัวเรียงตาม forecast.yaml
            "forecast": self.forecast.default.info(),
            "forecast_models": {name: s.info() for name, s in self.forecast.services.items()},
            "default_forecast_model": self.forecast.default_name,
            "segmentation": self.segmentation.info(),
            "data": {
                "sequence_store": self.sequences.available,
                "n_sequences": 0 if self.sequences.meta is None else len(self.sequences.meta),
                "splits": self.sequences.split_balance(),
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
