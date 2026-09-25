"""ที่อยู่ไฟล์ผลลัพธ์ของโมเดลพยากรณ์แต่ละตัว — จุดเดียวที่ฝั่งเทรน (เขียน) และฝั่งแอป (อ่าน) ใช้

โมเดลชื่อ ``<name>`` (key ใต้ ``models:`` ของ ``configs/forecast.yaml``) มีไฟล์ของตัวเองเสมอ::

    artifacts/models/<name>.pt                    checkpoint (น้ำหนัก + config + norm stats)
    artifacts/metrics/<name>.json                 ตัวชี้วัด val/test + logistic baseline + history
    artifacts/metrics/<name>_predictions.parquet  ค่าทำนายราย sample ของ val/test (แผง confusion matrix)
    artifacts/<name>_cv/                          ผลของ --cv (ไม่แตะไฟล์สามตัวบน)

**ข้อยกเว้นเดียว**: LSTM ที่เทรนก่อนระบบรองรับหลายโมเดลเขียนค่าทำนายไว้ที่
``artifacts/metrics/predictions.parquet`` — :meth:`ForecastArtifacts.existing_predictions`
ยังอ่านไฟล์นั้นให้ ถ้ายังไม่มีไฟล์ชื่อใหม่ (ชื่อคอลัมน์ ``lstm_prob`` ตรงกับรูปแบบใหม่อยู่แล้ว)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: ชื่อไฟล์ค่าทำนายของ LSTM ก่อนระบบรองรับหลายโมเดล
LEGACY_PREDICTIONS_NAME = "predictions.parquet"


@dataclass(frozen=True)
class ForecastArtifacts:
    name: str
    root: Path

    @property
    def checkpoint(self) -> Path:
        return self.root / "models" / f"{self.name}.pt"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics" / f"{self.name}.json"

    @property
    def predictions(self) -> Path:
        return self.root / "metrics" / f"{self.name}_predictions.parquet"

    @property
    def prob_column(self) -> str:
        """คอลัมน์ความน่าจะเป็นของโมเดลนี้ในไฟล์ค่าทำนาย (คู่กับ ``baseline_prob``)"""
        return f"{self.name}_prob"

    @property
    def cv_dir(self) -> Path:
        return self.root / f"{self.name}_cv"

    def existing_predictions(self) -> Path:
        """ไฟล์ค่าทำนายที่ควรอ่าน — ชื่อใหม่ก่อน แล้วค่อยถอยไปชื่อเดิมของ LSTM"""
        legacy = self.root / "metrics" / LEGACY_PREDICTIONS_NAME
        if not self.predictions.exists() and self.name == "lstm" and legacy.exists():
            return legacy
        return self.predictions
