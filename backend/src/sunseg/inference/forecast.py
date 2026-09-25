"""บริการพยากรณ์ flare — ห่อ checkpoint ของโมเดลพยากรณ์ทุกตัวให้เรียกใช้ง่ายจาก webapp

โหลดโมเดล **ครั้งเดียวตอนแอปเริ่มทำงาน** แล้วใช้ซ้ำทุก request การโหลด checkpoint
ใหม่ทุกครั้งที่มีคนเรียก API จะทำให้ช้าลงหลายสิบเท่า

- :class:`ForecastService` — โมเดลหนึ่งตัว (สถาปัตยกรรมใดก็ได้ ประกอบจาก ``kind`` ใน checkpoint)
- :class:`ForecastModels` — ทุกตัวใน ``configs/forecast.yaml`` พร้อมตัวปริยายของหน้าเว็บ
- :class:`SequenceStore` — sequence dataset ที่ทุกโมเดลอ่านร่วมกัน
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import TypeAdapter

from ..artifacts import ForecastArtifacts
from ..config import ArchModelConfig, ForecastModelsConfig
from ..data.build_sequences import apply_normalisation
from ..models.forecast import architecture_label, build_architecture

logger = logging.getLogger(__name__)

_ARCH_ADAPTER: TypeAdapter = TypeAdapter(ArchModelConfig)


def model_config_from_checkpoint(config: dict) -> tuple[str, object]:
    """``(kind, config ของสถาปัตยกรรม)`` จากส่วน ``config`` ของ checkpoint

    checkpoint ของ LSTM ที่เทรนก่อนระบบรองรับหลายโมเดลไม่มี ``kind`` และเก็บ pooling เป็น
    ``use_attention`` (True/False) — แปลงเป็นรูปแบบปัจจุบันที่นี่จุดเดียว เพื่อให้ ``lstm.pt``
    เดิมโหลดได้โดยไม่ต้องเทรนใหม่
    """
    model_cfg = dict(config["model"])
    kind = config.get("kind") or model_cfg.get("kind") or "lstm"
    model_cfg["kind"] = kind
    if "use_attention" in model_cfg:
        model_cfg.setdefault("pooling", "attention" if model_cfg.pop("use_attention") else "last")
    if model_cfg.pop("bidirectional", False):
        raise ValueError("checkpoint เป็น LSTM แบบ bidirectional ซึ่งระบบไม่รองรับแล้ว — เทรนใหม่")
    return kind, _ARCH_ADAPTER.validate_python(model_cfg)


def risk_level(probability: float, threshold: float) -> str:
    """จัดระดับความเสี่ยงเทียบกับ threshold ที่เลือกไว้ — ใช้กำหนดสีบนหน้าเว็บ

    อิงกับ threshold ที่ปรับ TSS สูงสุด ไม่ใช่ค่าคงที่ตายตัว เพราะความน่าจะเป็น
    ที่โมเดลให้มาเป็นค่าสัมพัทธ์ (positive จริงมีเพียง ~2% โมเดลจึงแทบไม่เคย
    ให้ค่าเกิน 0.5 แม้กับ AR ที่อันตรายที่สุด)
    """
    ratio = probability / max(threshold, 1e-6)
    if ratio >= 2.0:
        return "high"
    if ratio >= 1.0:
        return "elevated"
    if ratio >= 0.5:
        return "moderate"
    return "low"


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
        return risk_level(self.probability, self.threshold)


class ForecastService:
    """โหลดและเรียกใช้โมเดลพยากรณ์หนึ่งตัวที่เทรนไว้แล้ว — สถาปัตยกรรมใดก็ได้"""

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        name: str = "lstm",
        label: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.name = name
        self.label = label or name
        self.available = False

        self.model: torch.nn.Module | None = None
        self.kind: str | None = None
        self.pooling: str | None = None
        self.n_parameters: int = 0
        self.seq_len: int | None = None
        self.features: list[str] = []
        self.threshold: float = 0.5
        self.stats: dict[str, np.ndarray] = {}
        self.horizon_hours: int = 24
        self.positive_class: str = "M1.0"
        self.metrics: dict = {}

        self._load()

    @property
    def train_hint(self) -> str:
        return f"python backend/scripts/forecast/train.py --model {self.name}"

    def _load(self) -> None:
        if not self.checkpoint_path.exists():
            logger.warning(
                "ไม่พบ checkpoint ของโมเดล %s ที่ %s — โมเดลนี้จะปิดใช้งาน (รัน %s เพื่อสร้าง)",
                self.label, self.checkpoint_path, self.train_hint,
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
            # checkpoint เดิมของ LSTM ไม่มี seq_len — LSTM รับความยาวเท่าใดก็ได้จึงไม่ต้องใช้
            self.seq_len = int(config["seq_len"]) if config.get("seq_len") else None

            self.kind, model_cfg = model_config_from_checkpoint(config)
            self.pooling = model_cfg.pooling
            self.model = build_architecture(
                self.kind, int(config["n_features"]), self.seq_len or 0, model_cfg
            )
            self.model.load_state_dict(payload["state_dict"])
            self.model.to(self.device).eval()
            self.n_parameters = sum(p.numel() for p in self.model.parameters())

            self.stats = {
                "mean": np.asarray(payload["norm_mean"], dtype=np.float32),
                "std": np.asarray(payload["norm_std"], dtype=np.float32),
            }
            self.available = True

            test_tss = self.metrics.get("test", {}).get("tss")
            logger.info(
                "โหลดโมเดลพยากรณ์ %s สำเร็จ: %d features, threshold %.3f%s",
                self.label,
                len(self.features),
                self.threshold,
                f", test TSS {test_tss:+.4f}" if test_tss is not None else "",
            )
        except Exception as exc:  # noqa: BLE001 — checkpoint เสียไม่ควรทำให้ทั้งแอปล่ม
            logger.error("โหลด checkpoint ของโมเดล %s ไม่สำเร็จ: %s", self.label, exc)
            self.available = False

    # ------------------------------------------------------------------ #

    def _require_model(self) -> torch.nn.Module:
        if not self.available or self.model is None:
            raise RuntimeError(
                f"ยังไม่มีโมเดลพยากรณ์ {self.label} ที่ใช้งานได้ — รัน `{self.train_hint}` ก่อน"
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
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "pooling": self.pooling,
            "available": self.available,
            "checkpoint": str(self.checkpoint_path),
            "n_parameters": self.n_parameters,
            "n_features": len(self.features),
            "features": self.features,
            "threshold": round(self.threshold, 4),
            "horizon_hours": self.horizon_hours,
            "positive_class": self.positive_class,
            "metrics": self.metrics,
        }


class ForecastModels:
    """โมเดลพยากรณ์ทุกตัวใน ``configs/forecast.yaml`` — โหลดครั้งเดียวตอน startup

    โมเดลที่ยังไม่ได้เทรนก็ยังอยู่ในรายการ (``available=False``) เพื่อให้หน้าเว็บแสดงได้ว่า
    ต้องรันคำสั่งไหน แทนที่จะหายไปเงียบ ๆ
    """

    def __init__(self, config: ForecastModelsConfig, artifacts_root: Path, device: str = "cpu") -> None:
        self.default_name = config.default_model
        self.services: dict[str, ForecastService] = {
            name: ForecastService(
                ForecastArtifacts(name, Path(artifacts_root)).checkpoint,
                device=device,
                name=name,
                label=entry.label or architecture_label(entry.model.kind),
            )
            for name, entry in config.models.items()
        }
        #: ผลบน test ของ logistic baseline ที่เทรนคู่กับแต่ละโมเดล (อยู่ใน <name>.json เท่านั้น
        #: ไม่อยู่ใน checkpoint) — บนข้อมูลชุดเดียวกัน baseline ของทุกโมเดลให้ผลเดียวกัน
        self.baseline_test: dict[str, dict | None] = {
            name: _read_baseline(ForecastArtifacts(name, Path(artifacts_root)).metrics)
            for name in self.services
        }

    @property
    def names(self) -> list[str]:
        return list(self.services)

    @property
    def default(self) -> ForecastService:
        return self.services[self.default_name]

    @property
    def any_available(self) -> bool:
        return any(s.available for s in self.services.values())

    def get(self, name: str | None = None) -> ForecastService:
        """โมเดลชื่อ ``name`` (ไม่ระบุคือตัวปริยาย) — ชื่อที่ไม่รู้จัก raise ``KeyError``"""
        key = name or self.default_name
        if key not in self.services:
            raise KeyError(f"ไม่รู้จักโมเดล {key!r} (ที่มี: {', '.join(self.names)})")
        return self.services[key]

    def summaries(self) -> list[dict]:
        """หนึ่งรายการต่อโมเดล เรียงตาม ``forecast.yaml`` — สิ่งที่ปุ่มเลือกโมเดลบนหน้าเว็บต้องรู้"""
        rows = []
        for name, service in self.services.items():
            test = service.metrics.get("test", {}) if service.available else {}
            val = service.metrics.get("val", {}) if service.available else {}
            baseline = self.baseline_test.get(name) if service.available else None
            rows.append({
                "name": name,
                "label": service.label,
                "kind": service.kind,
                "pooling": service.pooling,
                "available": service.available,
                "default": name == self.default_name,
                "threshold": round(service.threshold, 4) if service.available else None,
                "n_parameters": service.n_parameters or None,
                "val_tss": val.get("tss"),
                "test": _metric_summary(test),
                "baseline_test": _metric_summary(baseline) if baseline else None,
                "train_hint": None if service.available else service.train_hint,
            })
        return rows


_SUMMARY_METRICS = ("tss", "auc", "recall", "precision", "hss2", "bss")


def _metric_summary(report: dict) -> dict:
    """ตัวชี้วัดที่หน้าเว็บแสดง + จำนวน sample/positive ของชุดนั้น (จาก confusion counts)"""
    summary = {k: report.get(k) for k in _SUMMARY_METRICS}
    counts = [report.get(k) for k in ("tp", "fp", "tn", "fn")]
    if all(c is not None for c in counts):
        tp, fp, tn, fn = (int(c) for c in counts)
        summary["n"] = tp + fp + tn + fn
        summary["n_positive"] = tp + fn
    return summary


def _read_baseline(metrics_path: Path) -> dict | None:
    """รายงาน test ของ logistic baseline จาก ``<name>.json`` — ไม่มีไฟล์/ไม่มี baseline คืน None"""
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8")).get("baseline_logistic")
    except (FileNotFoundError, json.JSONDecodeError):
        return None


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

    def split_balance(self) -> list[dict]:
        """จำนวน sample/positive และช่วงเวลาของแต่ละ split (train → val → test) — แผง Data split

        อ่านจากคอลัมน์ ``split`` ของ meta ที่ทุกโมเดลใช้ร่วมกัน ตัวเลขจึงตรงกับชุดที่เทรนจริงเสมอ
        แทนที่จะเป็นค่าที่พิมพ์ไว้ในหน้าเว็บแล้วล้าสมัยเมื่อสร้าง dataset ใหม่
        """
        if self.meta is None or "split" not in self.meta.columns:
            return []
        order = {"train": 0, "val": 1, "test": 2}
        grouped = (
            self.meta.groupby("split")
            .agg(n=("label", "size"), n_positive=("label", "sum"),
                 first=("issue_time", "min"), last=("issue_time", "max"))
            .reset_index()
        )
        grouped = grouped[grouped["split"].isin(order)].sort_values("split", key=lambda c: c.map(order))
        return [
            {
                "split": str(row.split),
                "n": int(row.n),
                "n_positive": int(row.n_positive),
                "first": row.first.isoformat(),
                "last": row.last.isoformat(),
            }
            for row in grouped.itertuples()
        ]

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
