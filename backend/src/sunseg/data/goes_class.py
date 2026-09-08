"""แปลงระหว่างสัญกรณ์ GOES flare class ("M1.5") กับค่า peak flux (W/m^2).

GOES จัดระดับ flare เป็นสเกล log ฐาน 10 โดยแต่ละตัวอักษรห่างกัน 1 decade::

    A = 1e-8, B = 1e-7, C = 1e-6, M = 1e-5, X = 1e-4  W/m^2

ตัวเลขที่ตามหลังคือตัวคูณ เช่น ``M1.5`` = 1.5e-5 W/m^2 และ ``X2`` = 2e-4 W/m^2
คลาส X ไม่มีเพดาน — ``X28`` = 2.8e-3 W/m^2 เป็นค่าที่ถูกต้อง
"""

from __future__ import annotations

import math
import re

CLASS_BASE_FLUX: dict[str, float] = {
    "A": 1e-8,
    "B": 1e-7,
    "C": 1e-6,
    "M": 1e-5,
    "X": 1e-4,
}

_CLASS_PATTERN = re.compile(r"^\s*([ABCMX])\s*([0-9]*\.?[0-9]*)\s*$", re.IGNORECASE)


def goes_class_to_flux(goes_class: str) -> float:
    """``"M1.0"`` -> ``1e-5``. โยน ValueError ถ้า parse ไม่ได้"""
    if not isinstance(goes_class, str):
        raise ValueError(f"GOES class ต้องเป็น str ไม่ใช่ {type(goes_class).__name__}")

    match = _CLASS_PATTERN.match(goes_class)
    if not match:
        raise ValueError(f"รูปแบบ GOES class ไม่ถูกต้อง: {goes_class!r}")

    letter = match.group(1).upper()
    # "M" เปล่าๆ (ไม่มีตัวเลข) ตีความเป็น M1.0 ตามธรรมเนียมของ catalog
    magnitude = float(match.group(2)) if match.group(2) not in ("", ".") else 1.0
    if magnitude <= 0:
        raise ValueError(f"ตัวคูณของ GOES class ต้องเป็นบวก: {goes_class!r}")

    return CLASS_BASE_FLUX[letter] * magnitude


def flux_to_goes_class(flux: float) -> str:
    """``1.5e-5`` -> ``"M1.5"``. ค่าที่ต่ำกว่าคลาส A คืน ``"A0.0"``"""
    if not math.isfinite(flux) or flux <= 0:
        return "A0.0"
    if flux < CLASS_BASE_FLUX["A"]:
        return "A0.0"

    for letter in ("X", "M", "C", "B", "A"):
        base = CLASS_BASE_FLUX[letter]
        if flux >= base:
            return f"{letter}{flux / base:.1f}"
    return "A0.0"


def is_at_least(goes_class: str, threshold: str) -> bool:
    """flare นี้แรงเท่ากับหรือแรงกว่าเกณฑ์หรือไม่ (เช่น ``is_at_least("X1.0", "M1.0")``)"""
    return goes_class_to_flux(goes_class) >= goes_class_to_flux(threshold)


def safe_goes_class_to_flux(goes_class: object, default: float = float("nan")) -> float:
    """เวอร์ชันที่ไม่โยน exception — ใช้กับข้อมูล catalog ที่มี field ว่าง/เพี้ยนปนมา"""
    try:
        return goes_class_to_flux(goes_class)  # type: ignore[arg-type]
    except (ValueError, TypeError, KeyError):
        return default
