"""ที่เก็บค่าทำนายราย sample ที่สคริปต์เทรนบันทึกไว้ — ฐานของแผง confusion matrix

โมเดลแต่ละตัวมีไฟล์ของตัวเองสองไฟล์ (ดู :class:`sunseg.artifacts.ForecastArtifacts`):
``<name>_predictions.parquet`` (ค่าความน่าจะเป็นราย sample ของ val/test) และ ``<name>.json``
(threshold ที่ freeze ไว้) จำเป็นต้องใช้ทั้งคู่เพราะ threshold ของ baseline logistic ไม่ได้ถูก
เก็บไว้ใน checkpoint ของโมเดลหลักเหมือนของโมเดลเอง — มีอยู่แห่งเดียวคือไฟล์ metrics นี้

logistic baseline ถูกเทรนคู่กับโมเดลทุกตัวบนข้อมูลชุดเดียวกัน จึงให้ผลเดียวกันไม่ว่าจะอ่านจาก
ไฟล์ของโมเดลไหน — :class:`ForecastPredictions` อ่านจากโมเดลปริยายก่อนเสมอ
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..artifacts import ForecastArtifacts
from ..metrics import roc_auc, threshold_sweep

logger = logging.getLogger(__name__)

#: ``"model"`` = โมเดลเจ้าของไฟล์ · ``"baseline"`` = logistic baseline ที่เทรนคู่กัน
Source = Literal["model", "baseline"]
SplitName = Literal["val", "test"]
CellName = Literal["tp", "fp", "fn", "tn"]

#: ชื่อพิเศษใน API ที่หมายถึง logistic baseline แทนชื่อโมเดล
BASELINE = "baseline"

#: กริดเดียวกับที่ ``find_best_threshold`` ใช้เลือก threshold ตอนเทรน (ดู
#: ``sunseg.metrics``) — threshold ที่ freeze ไว้จึงตกอยู่บนจุดของกริดนี้พอดี
THRESHOLD_GRID = np.linspace(0.01, 0.99, 200)


class PredictionsStore:
    """ค่าทำนายราย sample ของ val/test ของโมเดลหนึ่งตัว เพื่อคำนวณ confusion matrix แบบ on-the-fly

    การนับ TP/FP/TN/FN ทั้งหมดผ่าน :func:`sunseg.metrics.threshold_sweep` เท่านั้น —
    ไม่มีตรรกะการนับซ้ำอยู่ในคลาสนี้หรือฝั่ง frontend
    """

    def __init__(self, artifacts: ForecastArtifacts, label: str | None = None) -> None:
        self.name = artifacts.name
        self.label = label or artifacts.name
        self.train_hint = f"python backend/scripts/forecast/train.py --model {self.name}"
        self.predictions_path = artifacts.existing_predictions()
        self.metrics_path = artifacts.metrics
        self.prob_columns: dict[str, str] = {"model": artifacts.prob_column, "baseline": "baseline_prob"}
        self.available = False

        self._df: pd.DataFrame | None = None
        self._thresholds: dict[str, float | None] = {}

        self._load()

    def _load(self) -> None:
        if not self.predictions_path.exists() or not self.metrics_path.exists():
            logger.warning(
                "ไม่พบไฟล์ค่าทำนายราย sample ของ %s ที่ %s — แผง confusion matrix ของโมเดลนี้จะปิดใช้งาน "
                "(รัน %s เพื่อสร้าง)",
                self.label, self.predictions_path, self.train_hint,
            )
            return

        try:
            df = pd.read_parquet(self.predictions_path)
            metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))

            required = {"split", "HARPNUM", "noaa_ar", "issue_time", "label", *self.prob_columns.values()}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"{self.predictions_path.name} ขาดคอลัมน์: {sorted(missing)}")

            baseline_metrics = metrics.get("baseline_logistic")
            self._df = df
            self._thresholds = {
                "model": float(metrics[self.name]["val"]["threshold"]),
                "baseline": float(baseline_metrics["threshold"]) if baseline_metrics else None,
            }
            self.available = True

            logger.info(
                "โหลดค่าทำนายราย sample ของ %s สำเร็จ: %d แถว (val %d, test %d)",
                self.label,
                len(df),
                int((df["split"] == "val").sum()),
                int((df["split"] == "test").sum()),
            )
        except Exception as exc:  # noqa: BLE001 — ไฟล์เสียไม่ควรทำให้ทั้งแอปล่ม
            logger.error("โหลดค่าทำนายราย sample ของ %s ไม่สำเร็จ: %s", self.label, exc)
            self.available = False

    # ------------------------------------------------------------------ #

    @property
    def has_baseline(self) -> bool:
        return self.available and self._thresholds.get("baseline") is not None

    def _require_df(self) -> pd.DataFrame:
        if not self.available or self._df is None:
            raise RuntimeError(
                f"ยังไม่มีไฟล์ค่าทำนายราย sample ของ {self.label} — รัน `{self.train_hint}` ก่อน"
            )
        return self._df

    def frozen_threshold(self, source: Source) -> float:
        self._require_df()
        threshold = self._thresholds.get(source)
        if threshold is None:
            raise RuntimeError(
                "ยังไม่มี threshold ของ baseline logistic — เทรนใหม่โดยไม่ใส่ --no-baseline"
            )
        return threshold

    def rows(self, source: Source, split: SplitName) -> pd.DataFrame:
        """แถวของ split ที่ระบุ พร้อมคอลัมน์ ``label`` และ ``prob`` (ของ source ที่เลือก)

        ตัดแถวที่ไม่มีค่าทำนายของ source นั้นทิ้ง — เกิดขึ้นเมื่อเทรนด้วย ``--no-baseline``
        ซึ่งคอลัมน์ ``baseline_prob`` จะว่างทั้งคอลัมน์
        """
        df = self._require_df()
        prob_col = self.prob_columns[source]

        subset = df.loc[df["split"] == split].copy()
        subset["prob"] = subset[prob_col]
        subset = subset.dropna(subset=["prob"])

        if subset.empty and source == "baseline":
            raise RuntimeError(
                "ยังไม่มีค่าทำนายของ baseline logistic — เทรนใหม่โดยไม่ใส่ --no-baseline"
            )
        return subset

    def sweep(self, source: Source, split: SplitName) -> dict:
        """จำนวนนับของทุกจุดในกริด threshold พร้อม threshold ที่ freeze ไว้, AUC, ขนาด split"""
        rows = self.rows(source, split)
        y_true = rows["label"].to_numpy()
        y_prob = rows["prob"].to_numpy()

        counts = threshold_sweep(y_true, y_prob, THRESHOLD_GRID)
        frozen = self.frozen_threshold(source)
        # threshold ที่ freeze ไว้กับ THRESHOLD_GRID มาจาก np.linspace(0.01, 0.99, 200)
        # เดียวกันทั้งคู่ (ดูหมายเหตุบนโมดูล) จึงตกบนจุดกริดพอดีเสมอในทางปฏิบัติ —
        # หา index ด้วย argmin แทนเทียบเท่าตรง ๆ เพื่อกันพลาดจากความคลาดเคลื่อนระดับ
        # floating point ที่อาจเกิดจากการอ่าน/เขียนผ่าน JSON
        frozen_index = int(np.argmin(np.abs(THRESHOLD_GRID - frozen)))

        return {
            "model": self.name if source == "model" else BASELINE,
            "split": split,
            "thresholds": THRESHOLD_GRID.tolist(),
            "tp": counts["tp"].tolist(),
            "fp": counts["fp"].tolist(),
            "tn": counts["tn"].tolist(),
            "fn": counts["fn"].tolist(),
            "frozen_threshold": frozen,
            "frozen_index": frozen_index,
            "auc": roc_auc(y_true, y_prob),
            "n": int(len(rows)),
            "n_positive": int(y_true.sum()),
        }

    def samples_in_cell(
        self,
        source: Source,
        split: SplitName,
        threshold: float,
        cell: CellName,
        limit: int = 200,
    ) -> tuple[pd.DataFrame, int]:
        """แถวของ sample ที่อยู่ในช่องที่ระบุ เรียงตามความน่าจะเป็นมาก -> น้อย

        เรียงมาก -> น้อยเสมอไม่ว่าช่องไหน เพื่อให้แถวบนสุดน่าสนใจที่สุดในทุกกรณี:
        ช่อง miss (FN) จะเจอเคส "เกือบจับได้" ก่อน ส่วนช่อง false alarm (FP) จะเจอ
        เคสที่โมเดลมั่นใจผิดมากที่สุดก่อน

        Returns
        -------
        ``(แถวที่ตัดตาม limit แล้ว, จำนวนเต็มของช่องนี้ก่อนตัด limit)``
        """
        rows = self.rows(source, split)
        predicted_positive = rows["prob"].to_numpy() >= threshold
        actual_positive = rows["label"].to_numpy().astype(bool)

        masks: dict[CellName, np.ndarray] = {
            "tp": actual_positive & predicted_positive,
            "fp": ~actual_positive & predicted_positive,
            "fn": actual_positive & ~predicted_positive,
            "tn": ~actual_positive & ~predicted_positive,
        }

        matched = rows.loc[masks[cell]].sort_values("prob", ascending=False)
        return matched.head(limit), int(len(matched))


class ForecastPredictions:
    """ค่าทำนายของโมเดลทุกตัว — เลือกด้วยชื่อโมเดล หรือ ``"baseline"`` สำหรับ logistic baseline"""

    def __init__(self, stores: dict[str, PredictionsStore], default_name: str) -> None:
        self.stores = stores
        self.default_name = default_name

    @classmethod
    def from_artifacts(
        cls, names: dict[str, str], artifacts_root: Path, default_name: str
    ) -> ForecastPredictions:
        """``names`` คือ ``{ชื่อโมเดล: label}`` ตามลำดับใน ``forecast.yaml``"""
        stores = {
            name: PredictionsStore(ForecastArtifacts(name, Path(artifacts_root)), label=label)
            for name, label in names.items()
        }
        return cls(stores, default_name)

    @property
    def available(self) -> bool:
        return any(s.available for s in self.stores.values())

    def resolve(self, model: str) -> tuple[PredictionsStore, Source]:
        """``(store, source)`` ของชื่อที่หน้าเว็บส่งมา — ชื่อที่ไม่รู้จัก raise ``KeyError``

        ``"baseline"`` อ่านจากไฟล์ของโมเดลปริยายก่อน แล้วค่อยไล่ตัวอื่นที่มี baseline
        """
        if model == BASELINE:
            ordered = [self.default_name, *[n for n in self.stores if n != self.default_name]]
            for name in ordered:
                if self.stores[name].has_baseline:
                    return self.stores[name], "baseline"
            raise RuntimeError(
                "ยังไม่มีค่าทำนายของ baseline logistic — เทรนโมเดลใดก็ได้โดยไม่ใส่ --no-baseline "
                "(เช่น `python backend/scripts/forecast/train.py --model lstm`)"
            )
        if model not in self.stores:
            raise KeyError(f"ไม่รู้จักโมเดล {model!r} (ที่มี: {', '.join([*self.stores, BASELINE])})")
        return self.stores[model], "model"

    def sweep(self, model: str, split: SplitName) -> dict:
        store, source = self.resolve(model)
        return store.sweep(source, split)

    def samples_in_cell(
        self, model: str, split: SplitName, threshold: float, cell: CellName, limit: int = 200
    ) -> tuple[pd.DataFrame, int]:
        store, source = self.resolve(model)
        return store.samples_in_cell(source, split, threshold, cell, limit=limit)
