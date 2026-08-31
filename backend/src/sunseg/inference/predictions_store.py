"""ที่เก็บค่าทำนายราย sample ที่ ``train_lstm.py`` บันทึกไว้ — ฐานของแผง confusion matrix

โหลดจากสองไฟล์: ``predictions.parquet`` (ค่าความน่าจะเป็นราย sample ของ val/test)
และ ``lstm.json`` (threshold ที่ freeze ไว้ของแต่ละโมเดล) จำเป็นต้องใช้ทั้งคู่เพราะ
threshold ของ baseline logistic ไม่ได้ถูกเก็บไว้ใน checkpoint ของ LSTM (``lstm.pt``)
เหมือนของ LSTM เอง — มีอยู่แห่งเดียวคือไฟล์ metrics นี้
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from ..metrics import roc_auc, threshold_sweep

logger = logging.getLogger(__name__)

ModelName = Literal["lstm", "baseline"]
SplitName = Literal["val", "test"]
CellName = Literal["tp", "fp", "fn", "tn"]

#: กริดเดียวกับที่ ``find_best_threshold`` ใช้เลือก threshold ตอนเทรน (ดู
#: ``sunseg.metrics``) — threshold ที่ freeze ไว้จึงตกอยู่บนจุดของกริดนี้พอดี
THRESHOLD_GRID = np.linspace(0.01, 0.99, 200)

_PROB_COLUMN: dict[str, str] = {"lstm": "lstm_prob", "baseline": "baseline_prob"}


class PredictionsStore:
    """เข้าถึงค่าทำนายราย sample ของ val/test เพื่อคำนวณ confusion matrix แบบ on-the-fly

    การนับ TP/FP/TN/FN ทั้งหมดผ่าน :func:`sunseg.metrics.threshold_sweep` เท่านั้น —
    ไม่มีตรรกะการนับซ้ำอยู่ในคลาสนี้หรือฝั่ง frontend
    """

    def __init__(self, predictions_path: Path, metrics_path: Path) -> None:
        self.predictions_path = Path(predictions_path)
        self.metrics_path = Path(metrics_path)
        self.available = False

        self._df: pd.DataFrame | None = None
        self._thresholds: dict[str, float | None] = {}

        self._load()

    def _load(self) -> None:
        if not self.predictions_path.exists() or not self.metrics_path.exists():
            logger.warning(
                "ไม่พบไฟล์ค่าทำนายราย sample ที่ %s — แผง confusion matrix จะปิดใช้งาน "
                "(รัน backend/scripts/train_lstm.py เพื่อสร้าง)",
                self.predictions_path,
            )
            return

        try:
            df = pd.read_parquet(self.predictions_path)
            metrics = json.loads(self.metrics_path.read_text(encoding="utf-8"))

            required = {
                "split", "HARPNUM", "noaa_ar", "issue_time",
                "label", "lstm_prob", "baseline_prob",
            }
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"predictions.parquet ขาดคอลัมน์: {sorted(missing)}")

            baseline_metrics = metrics.get("baseline_logistic")
            self._df = df
            self._thresholds = {
                "lstm": float(metrics["lstm"]["val"]["threshold"]),
                "baseline": float(baseline_metrics["threshold"]) if baseline_metrics else None,
            }
            self.available = True

            logger.info(
                "โหลดค่าทำนายราย sample สำเร็จ: %d แถว (val %d, test %d)",
                len(df),
                int((df["split"] == "val").sum()),
                int((df["split"] == "test").sum()),
            )
        except Exception as exc:  # noqa: BLE001 — ไฟล์เสียไม่ควรทำให้ทั้งแอปล่ม
            logger.error("โหลดค่าทำนายราย sample ไม่สำเร็จ: %s", exc)
            self.available = False

    # ------------------------------------------------------------------ #

    def _require_df(self) -> pd.DataFrame:
        if not self.available or self._df is None:
            raise RuntimeError(
                "ยังไม่มีไฟล์ค่าทำนายราย sample — รัน `python backend/scripts/train_lstm.py` ก่อน"
            )
        return self._df

    def frozen_threshold(self, model: ModelName) -> float:
        self._require_df()
        threshold = self._thresholds.get(model)
        if threshold is None:
            raise RuntimeError(
                "ยังไม่มี threshold ของ baseline logistic — เทรนใหม่โดยไม่ใส่ --no-baseline"
            )
        return threshold

    def rows(self, model: ModelName, split: SplitName) -> pd.DataFrame:
        """แถวของ split ที่ระบุ พร้อมคอลัมน์ ``label`` และ ``prob`` (ของโมเดลที่เลือก)

        ตัดแถวที่ไม่มีค่าทำนายของโมเดลนั้นทิ้ง — เกิดขึ้นเมื่อเทรนด้วย ``--no-baseline``
        ซึ่งคอลัมน์ ``baseline_prob`` จะว่างทั้งคอลัมน์
        """
        df = self._require_df()
        prob_col = _PROB_COLUMN[model]

        subset = df.loc[df["split"] == split].copy()
        subset["prob"] = subset[prob_col]
        subset = subset.dropna(subset=["prob"])

        if subset.empty and model == "baseline":
            raise RuntimeError(
                "ยังไม่มีค่าทำนายของ baseline logistic — เทรนใหม่โดยไม่ใส่ --no-baseline"
            )
        return subset

    def sweep(self, model: ModelName, split: SplitName) -> dict:
        """จำนวนนับของทุกจุดในกริด threshold พร้อม threshold ที่ freeze ไว้, AUC, ขนาด split"""
        rows = self.rows(model, split)
        y_true = rows["label"].to_numpy()
        y_prob = rows["prob"].to_numpy()

        counts = threshold_sweep(y_true, y_prob, THRESHOLD_GRID)
        frozen = self.frozen_threshold(model)
        # threshold ที่ freeze ไว้กับ THRESHOLD_GRID มาจาก np.linspace(0.01, 0.99, 200)
        # เดียวกันทั้งคู่ (ดูหมายเหตุบนโมดูล) จึงตกบนจุดกริดพอดีเสมอในทางปฏิบัติ —
        # หา index ด้วย argmin แทนเทียบเท่าตรง ๆ เพื่อกันพลาดจากความคลาดเคลื่อนระดับ
        # floating point ที่อาจเกิดจากการอ่าน/เขียนผ่าน JSON
        frozen_index = int(np.argmin(np.abs(THRESHOLD_GRID - frozen)))

        return {
            "model": model,
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
        model: ModelName,
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
        rows = self.rows(model, split)
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
