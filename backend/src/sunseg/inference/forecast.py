"""บริการพยากรณ์ flare — ห่อ LSTM checkpoint ให้เรียกใช้ง่ายจาก webapp

โหลดโมเดล **ครั้งเดียวตอนแอปเริ่มทำงาน** แล้วใช้ซ้ำทุก request การโหลด checkpoint
ใหม่ทุกครั้งที่มีคนเรียก API จะทำให้ช้าลงหลายสิบเท่า
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.build_sequences import apply_normalisation
from ..models.lstm import FlareLSTM

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """ผลการพยากรณ์ของ active region หนึ่งดวง ณ เวลาหนึ่ง"""

    harpnum: int
    issue_time: datetime
    probability: float
    predicted_positive: bool
    threshold: float
    #: น้ำหนัก attention รายชั่วโมง (ยาวเท่ากับ sequence) — บอกว่าโมเดลสนใจช่วงไหน
    attention: list[float]
    #: ค่า SHARP parameters ล่าสุด (ค่าดิบ ยังไม่ normalise)
    features: dict[str, float]
    noaa_ar: int | None = None
    lat: float | None = None
    lon: float | None = None
    actual_label: int | None = None

    def to_dict(self) -> dict:
        return {
            "harpnum": self.harpnum,
            "noaa_ar": self.noaa_ar,
            "issue_time": self.issue_time.isoformat(),
            "probability": round(self.probability, 5),
            "predicted_positive": self.predicted_positive,
            "threshold": round(self.threshold, 4),
            "risk_level": self.risk_level,
            "attention": [round(a, 5) for a in self.attention],
            "features": {k: float(v) for k, v in self.features.items()},
            "lat": self.lat,
            "lon": self.lon,
            "actual_label": self.actual_label,
        }

    @property
    def risk_level(self) -> str:
        """จัดระดับความเสี่ยงเทียบกับ threshold ที่เลือกไว้ — ใช้กำหนดสีบนหน้าเว็บ

        อิงกับ threshold ที่ปรับ TSS สูงสุด ไม่ใช่ค่าคงที่ตายตัว เพราะความน่าจะเป็น
        ที่โมเดลให้มาเป็นค่าสัมพัทธ์ (positive จริงมีเพียง ~2% โมเดลจึงแทบไม่เคย
        ให้ค่าเกิน 0.5 แม้กับ AR ที่อันตรายที่สุด)
        """
        ratio = self.probability / max(self.threshold, 1e-6)
        if ratio >= 2.0:
            return "high"
        if ratio >= 1.0:
            return "elevated"
        if ratio >= 0.5:
            return "moderate"
        return "low"


class ForecastService:
    """โหลดและเรียกใช้โมเดล LSTM ที่เทรนไว้แล้ว"""

    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.available = False

        self.model: FlareLSTM | None = None
        self.features: list[str] = []
        self.threshold: float = 0.5
        self.stats: dict[str, np.ndarray] = {}
        self.horizon_hours: int = 24
        self.positive_class: str = "M1.0"
        self.metrics: dict = {}

        self._load()

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            logger.warning(
                "ไม่พบ checkpoint ของ LSTM ที่ %s — ฟีเจอร์พยากรณ์จะปิดใช้งาน "
                "(รัน backend/scripts/train_lstm.py เพื่อสร้าง)",
                self.checkpoint_path,
            )
            return

        try:
            payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = payload["config"]

            self.features = list(config["features"])
            self.threshold = float(config.get("threshold", 0.5))
            self.horizon_hours = int(config.get("horizon_hours", 24))
            self.positive_class = str(config.get("positive_class", "M1.0"))
            self.metrics = payload.get("metrics", {})

            model_cfg = config["model"]
            self.model = FlareLSTM(
                n_features=int(config["n_features"]),
                hidden_size=model_cfg["hidden_size"],
                num_layers=model_cfg["num_layers"],
                dropout=model_cfg["dropout"],
                bidirectional=model_cfg["bidirectional"],
                use_attention=model_cfg["use_attention"],
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device).eval()

            self.stats = {
                "mean": np.asarray(payload["norm_mean"], dtype=np.float32),
                "std": np.asarray(payload["norm_std"], dtype=np.float32),
            }
            self.available = True

            test_tss = self.metrics.get("test", {}).get("tss")
            logger.info(
                "โหลดโมเดลพยากรณ์สำเร็จ: %d features, threshold %.3f%s",
                len(self.features),
                self.threshold,
                f", test TSS {test_tss:+.4f}" if test_tss is not None else "",
            )
        except Exception as exc:  # noqa: BLE001 — checkpoint เสียไม่ควรทำให้ทั้งแอปล่ม
            logger.error("โหลด checkpoint ของ LSTM ไม่สำเร็จ: %s", exc)
            self.available = False

    # ------------------------------------------------------------------ #

    def _require_model(self) -> FlareLSTM:
        if not self.available or self.model is None:
            raise RuntimeError(
                "ยังไม่มีโมเดลพยากรณ์ที่ใช้งานได้ — รัน backend/scripts/train_lstm.py ก่อน"
            )
        return self.model

    @torch.no_grad()
    def predict_batch(self, sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """พยากรณ์หลาย sequence พร้อมกัน

        Parameters
        ----------
        sequences
            ค่าดิบรูปทรง ``(N, L, F)`` — ฟังก์ชันนี้ normalise ให้เอง

        Returns
        -------
        ``(probabilities, attention_weights)``
        """
        model = self._require_model()

        if sequences.ndim != 3:
            raise ValueError(f"คาดหวังรูปทรง (N, L, F) แต่ได้ {sequences.shape}")
        if sequences.shape[-1] != len(self.features):
            raise ValueError(
                f"จำนวน feature ไม่ตรง: โมเดลต้องการ {len(self.features)} "
                f"แต่ได้รับ {sequences.shape[-1]}"
            )

        normalised = apply_normalisation(sequences, self.stats)
        tensor = torch.from_numpy(normalised).to(self.device)

        logits, attention = model(tensor, return_attention=True)
        return (
            torch.sigmoid(logits).cpu().numpy(),
            attention.cpu().numpy(),
        )

    def predict_one(
        self,
        sequence: np.ndarray,
        harpnum: int,
        issue_time: datetime,
        **extra,
    ) -> ForecastResult:
        """พยากรณ์ sequence เดียว รูปทรง ``(L, F)``"""
        probs, attention = self.predict_batch(sequence[np.newaxis, ...])
        probability = float(probs[0])

        # ค่า feature ณ timestep ล่าสุด — คือสภาพปัจจุบันของ AR ที่ผู้ใช้สนใจ
        latest = {name: float(value) for name, value in zip(self.features, sequence[-1], strict=True)}

        return ForecastResult(
            harpnum=harpnum,
            issue_time=issue_time,
            probability=probability,
            predicted_positive=probability >= self.threshold,
            threshold=self.threshold,
            attention=attention[0].tolist(),
            features=latest,
            **extra,
        )

    def info(self) -> dict:
        """ข้อมูลสรุปของโมเดล สำหรับแสดงบนหน้าเว็บ"""
        return {
            "available": self.available,
            "checkpoint": str(self.checkpoint_path),
            "n_features": len(self.features),
            "features": self.features,
            "threshold": round(self.threshold, 4),
            "horizon_hours": self.horizon_hours,
            "positive_class": self.positive_class,
            "metrics": self.metrics,
        }


class SequenceStore:
    """เข้าถึง sequence dataset ที่สร้างไว้ เพื่อให้ webapp เรียกดูย้อนหลังได้

    เก็บ ``X.npy`` ไว้แบบ memory-map — ไฟล์อาจใหญ่ถึงหลาย GB และเราต้องการเพียง
    ไม่กี่แถวต่อ request
    """

    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = Path(processed_dir)
        self.available = False
        self.meta: pd.DataFrame | None = None
        self._x: np.ndarray | None = None

        self._load()

    def _load(self) -> None:
        x_path = self.processed_dir / "X.npy"
        meta_path = self.processed_dir / "meta.parquet"

        if not (x_path.exists() and meta_path.exists()):
            logger.warning(
                "ไม่พบ sequence dataset ที่ %s — การเรียกดูข้อมูลย้อนหลังจะปิดใช้งาน",
                self.processed_dir,
            )
            return

        self._x = np.load(x_path, mmap_mode="r")
        self.meta = pd.read_parquet(meta_path)
        self.meta["issue_time"] = pd.to_datetime(self.meta["issue_time"])
        self.available = True

        logger.info(
            "โหลด sequence store: %d sample, %d HARP, %s ถึง %s",
            len(self.meta),
            self.meta["HARPNUM"].nunique(),
            self.meta["issue_time"].min().date(),
            self.meta["issue_time"].max().date(),
        )

    def sequence_at(self, index: int) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("ยังไม่ได้โหลด sequence dataset")
        return np.asarray(self._x[index], dtype=np.float32)

    def find_rows(
        self,
        harpnum: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        split: str | None = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """ค้นหา sample ตามเงื่อนไข คืน meta พร้อมคอลัมน์ ``row`` ที่ใช้ดึง sequence"""
        if self.meta is None:
            raise RuntimeError("ยังไม่ได้โหลด sequence dataset")

        frame = self.meta.reset_index(names="row")
        if harpnum is not None:
            frame = frame[frame["HARPNUM"] == harpnum]
        if start is not None:
            frame = frame[frame["issue_time"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["issue_time"] <= pd.Timestamp(end)]
        if split is not None:
            frame = frame[frame["split"] == split]

        return frame.sort_values("issue_time").head(limit)

    def available_harps(self, limit: int = 2000) -> list[dict]:
        """รายการ HARP ที่มีข้อมูล พร้อมช่วงเวลาและจำนวนหน้าต่างที่เป็น positive"""
        if self.meta is None:
            return []

        grouped = (
            self.meta.groupby("HARPNUM")
            .agg(
                n_samples=("issue_time", "size"),
                first_time=("issue_time", "min"),
                last_time=("issue_time", "max"),
                n_positive=("label", "sum"),
                noaa_ar=("noaa_ar", "first"),
            )
            .reset_index()
            .sort_values("n_positive", ascending=False)
            .head(limit)
        )

        return [
            {
                "harpnum": int(row.HARPNUM),
                "noaa_ar": None if pd.isna(row.noaa_ar) else int(row.noaa_ar),
                "n_samples": int(row.n_samples),
                "n_positive": int(row.n_positive),
                "first_time": row.first_time.isoformat(),
                "last_time": row.last_time.isoformat(),
            }
            for row in grouped.itertuples()
        ]
