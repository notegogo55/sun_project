"""API คำพยากรณ์ระดับคลาสของโมเดลหลัก (LSTM + V3) — <M / M / X ใน 24 ชม. ถัดไป

แยกจาก ``/api/forecast`` เพราะเป็นคนละ dataset กัน: โมเดลพยากรณ์ใน ``configs/forecast.yaml`` อ่าน sequence
ราย 1 ชม. (18 SHARP) ส่วนโมเดลหลักอ่าน study dataset ราย 12 ชม. (SHARP + intensity + X-ray) และเป็น ensemble
ของทุก seed สองระดับ — ดู ``sunseg.inference.class_forecast``

ค่าทั้งหมดคำนวณไว้แล้วตอน startup · ยังไม่พร้อม → 503 พร้อมคำสั่งที่ต้องรัน (ผ่าน handler ของ RuntimeError
ใน ``main.py``) ยกเว้น ``/summary`` ที่ตอบ 200 เสมอพร้อม ``available``/``hint`` ให้หน้าเว็บแสดงเอง
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from sunseg.inference.class_forecast import LEVELS, MODES

from ..schemas import ClassForecastPoint, ClassForecastSeriesResponse, ClassForecastSummary, parse_iso

router = APIRouter(prefix="/api/class-forecast", tags=["class-forecast"])

Mode = Literal["sensitive", "strict"]
_MODE_QUERY = Query(
    MODES[0], description="จุดทำงาน: sensitive (เตือนไว — threshold ที่ให้ TSS สูงสุด) | strict (ระมัดระวัง)"
)


def _service(request: Request):
    return request.app.state.services.class_forecast


@router.get("/summary", response_model=ClassForecastSummary, summary="โมเดลหลักและผลบน val/test ของทุกจุดทำงาน")
def summary(request: Request):
    service = _service(request)
    info = service.info()
    info["evaluation"] = (
        {mode: {split: service.evaluation(split, mode) for split in ("val", "test")} for mode in MODES}
        if service.available
        else None
    )
    return info


@router.get("", response_model=ClassForecastSeriesResponse, summary="คำพยากรณ์ระดับคลาสตามเวลาของ HARP หนึ่ง")
def series(
    request: Request,
    harpnum: int = Query(..., description="หมายเลข HARP"),
    mode: Mode = _MODE_QUERY,
):
    service = _service(request)
    rows = service.series(harpnum)
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่มีคำพยากรณ์ของโมเดลหลักสำหรับ HARP {harpnum} (study dataset มีเฉพาะช่วงที่ HARP อยู่ในขอบเขตพยากรณ์)",
        )
    points = [_point(row, mode) for row in rows.itertuples(index=False)]
    noaa = rows["noaa_ar"].dropna() if "noaa_ar" in rows else pd.Series(dtype=float)
    return ClassForecastSeriesResponse(
        harpnum=harpnum,
        noaa_ar=None if noaa.empty else int(noaa.iloc[-1]),
        mode=mode,
        cadence_hours=service.cadence_hours,
        horizon_hours=24,
        points=points,
        latest=points[-1],
    )


@router.get("/at", response_model=list[ClassForecastPoint], summary="คำพยากรณ์ระดับคลาสของทุก HARP ณ เวลาหนึ่ง")
def at_time(
    request: Request,
    time: str = Query(..., description="เวลาที่ต้องการ เช่น 2014-10-24T00:00:00"),
    tolerance_hours: float = Query(12.0, ge=0.5, le=48.0),
    mode: Mode = _MODE_QUERY,
):
    """issue_time ที่ใกล้ที่สุดของแต่ละ HARP ภายใน ± tolerance เรียงจากระดับสูงไปต่ำ"""
    service = _service(request)
    rows = service.at(parse_iso(time, "time"), pd.Timedelta(hours=tolerance_hours), mode)
    return [_point(row, mode) for row in rows.itertuples(index=False)]


def _point(row, mode: str) -> ClassForecastPoint:
    return ClassForecastPoint(
        harpnum=int(row.HARPNUM),
        noaa_ar=_optional_int(getattr(row, "noaa_ar", None)),
        issue_time=row.issue_time.isoformat(),
        level=LEVELS[int(getattr(row, f"level_{mode}"))],
        true_level=LEVELS[int(row.true_level)],
        prob_m=round(float(row.prob_M), 5),
        prob_x=round(float(row.prob_X), 5),
        n_alarm_m=int(row.n_alarm_M),
        n_alarm_x=int(row.n_alarm_X),
        split=str(row.split),
        lat=_optional_float(getattr(row, "lat", None)),
        lon=_optional_float(getattr(row, "lon", None)),
    )


def _optional_int(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _optional_float(value) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 4)
