"""API พยากรณ์ flare — ทุก endpoint เลือกโมเดลได้ด้วย ``?model=<ชื่อ>``

ชื่อโมเดลคือ key ใต้ ``models:`` ของ ``configs/forecast.yaml`` (lstm, tcn, transformer, darnn)
ไม่ระบุคือใช้ ``default_model`` ของไฟล์นั้น — client เดิมที่ไม่รู้จักพารามิเตอร์นี้จึงได้ผลเหมือนเดิม
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from sunseg.inference.forecast import ForecastService, risk_level

from ..schemas import (
    ConfusionMatrixSample,
    ConfusionMatrixSampleList,
    ConfusionMatrixSummary,
    ForecastModelSummary,
    ForecastPoint,
    ForecastSeriesResponse,
    HarpSummary,
    parse_iso,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["forecast"])

_MODEL_QUERY = Query(None, description="ชื่อโมเดล (lstm, tcn, transformer, darnn) — ไม่ระบุคือโมเดลปริยาย")
_SEQUENCES_HINT = "ยังไม่มีข้อมูล sequence — รัน `python backend/scripts/data/build_sequences.py` ก่อน"


def _services(request: Request):
    return request.app.state.services


def _model(services, name: str | None) -> ForecastService:
    """โมเดลที่ขอ — ชื่อที่ไม่รู้จักเป็น 422 (ค่าพารามิเตอร์ผิด), ยังไม่ได้เทรนเป็น 503 พร้อมคำสั่งที่ต้องรัน"""
    try:
        service = services.forecast.get(name)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc.args[0])) from exc
    if not service.available:
        raise HTTPException(
            status_code=503,
            detail=f"ยังไม่มีโมเดลพยากรณ์ {service.label} — รัน `{service.train_hint}` ก่อน",
        )
    if not services.sequences.available:
        raise HTTPException(status_code=503, detail=_SEQUENCES_HINT)
    return service


@router.get(
    "/forecast/models",
    response_model=list[ForecastModelSummary],
    summary="โมเดลพยากรณ์ทั้งหมด พร้อมความพร้อมและผลบน test",
)
def list_models(request: Request):
    """เรียงตาม ``configs/forecast.yaml`` — ตัวที่ยังไม่ได้เทรนก็อยู่ในรายการ (``available: false``)
    พร้อมคำสั่งที่ต้องรันใน ``train_hint``"""
    return _services(request).forecast.summaries()


@router.get("/harps", response_model=list[HarpSummary], summary="รายการ active region ที่มีข้อมูล")
def list_harps(
    request: Request,
    limit: int = Query(200, ge=1, le=5000, description="จำนวนสูงสุดที่คืน"),
    only_flaring: bool = Query(False, description="เอาเฉพาะ HARP ที่เคยเกิด flare จริง"),
):
    """เรียงตามจำนวนหน้าต่างเวลาที่เกิด flare จริง — HARP ที่น่าสนใจที่สุดจะอยู่บนสุด"""
    services = _services(request)
    if not services.sequences.available:
        raise HTTPException(status_code=503, detail=_SEQUENCES_HINT)

    harps = services.sequences.available_harps(limit=limit * 4)
    if only_flaring:
        harps = [h for h in harps if h["n_positive"] > 0]
    return harps[:limit]


@router.get(
    "/forecast",
    response_model=ForecastSeriesResponse,
    summary="ชุดค่าพยากรณ์ตามเวลาของ active region หนึ่งดวง",
)
def forecast_series(
    request: Request,
    harpnum: int = Query(..., description="หมายเลข HARP"),
    start: str | None = Query(None, description="เวลาเริ่มต้น เช่น 2014-01-01"),
    end: str | None = Query(None, description="เวลาสิ้นสุด"),
    limit: int = Query(500, ge=1, le=5000),
    model: str | None = _MODEL_QUERY,
):
    """คืนความน่าจะเป็นการเกิด flare ตลอดช่วงเวลาที่ AR ดวงนั้นถูกสังเกต

    เป็น endpoint หลักที่หน้าเว็บใช้วาดกราฟความเสี่ยงตามเวลา
    """
    services = _services(request)
    forecaster = _model(services, model)

    rows = services.sequences.find_rows(
        harpnum=harpnum,
        start=parse_iso(start, "start") if start else None,
        end=parse_iso(end, "end") if end else None,
        limit=limit,
    )
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบข้อมูลของ HARP {harpnum} ในช่วงเวลาที่ระบุ",
        )

    sequences = np.stack([services.sequences.sequence_at(i) for i in rows["row"]])
    probabilities, attention = forecaster.predict_batch(sequences)

    noaa = rows["noaa_ar"].dropna()
    latest_row = rows.iloc[-1]

    latest = ForecastPoint(
        harpnum=harpnum,
        noaa_ar=None if noaa.empty else int(noaa.iloc[-1]),
        issue_time=latest_row["issue_time"].isoformat(),
        probability=float(probabilities[-1]),
        predicted_positive=bool(probabilities[-1] >= forecaster.threshold),
        threshold=forecaster.threshold,
        risk_level=risk_level(float(probabilities[-1]), forecaster.threshold),
        attention=[float(a) for a in attention[-1]],
        features={
            name: float(value)
            for name, value in zip(forecaster.features, sequences[-1, -1], strict=True)
        },
        lat=_optional_float(latest_row.get("lat")),
        lon=_optional_float(latest_row.get("lon")),
        actual_label=_optional_int(latest_row.get("label")),
    )

    return ForecastSeriesResponse(
        model=forecaster.name,
        harpnum=harpnum,
        noaa_ar=None if noaa.empty else int(noaa.iloc[0]),
        n_points=len(rows),
        horizon_hours=forecaster.horizon_hours,
        positive_class=forecaster.positive_class,
        threshold=forecaster.threshold,
        times=[t.isoformat() for t in rows["issue_time"]],
        probabilities=[round(float(p), 5) for p in probabilities],
        actual_labels=[_optional_int(v) for v in rows["label"]],
        latest=latest,
    )


@router.get(
    "/forecast/at",
    response_model=list[ForecastPoint],
    summary="พยากรณ์ทุก active region ณ เวลาที่ระบุ",
)
def forecast_at_time(
    request: Request,
    time: str = Query(..., description="เวลาที่ต้องการ เช่น 2014-01-07T12:00:00"),
    tolerance_hours: float = Query(3.0, ge=0.5, le=48.0),
    model: str | None = _MODEL_QUERY,
):
    """ภาพรวมความเสี่ยงของทุก AR ที่มีข้อมูล ณ ช่วงเวลาหนึ่ง — ใช้คู่กับแผงภาพดวงอาทิตย์"""
    services = _services(request)
    forecaster = _model(services, model)

    target = parse_iso(time, "time")
    window = pd.Timedelta(hours=tolerance_hours)
    rows = services.sequences.find_rows(
        start=target - window, end=target + window, limit=20000
    )
    if rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"ไม่พบข้อมูลใกล้เวลา {time} (ค้นหาภายใน ±{tolerance_hours} ชม.)",
        )

    # เก็บแถวที่ใกล้เวลาเป้าหมายที่สุดของแต่ละ HARP
    rows = rows.assign(_gap=(rows["issue_time"] - target).abs())
    nearest = rows.sort_values("_gap").drop_duplicates("HARPNUM").sort_values("HARPNUM")

    sequences = np.stack([services.sequences.sequence_at(i) for i in nearest["row"]])
    probabilities, attention = forecaster.predict_batch(sequences)

    results = []
    for position, (_, row) in enumerate(nearest.iterrows()):
        probability = float(probabilities[position])
        results.append(
            ForecastPoint(
                harpnum=int(row["HARPNUM"]),
                noaa_ar=_optional_int(row.get("noaa_ar")),
                issue_time=row["issue_time"].isoformat(),
                probability=round(probability, 5),
                predicted_positive=probability >= forecaster.threshold,
                threshold=forecaster.threshold,
                risk_level=risk_level(probability, forecaster.threshold),
                attention=[round(float(a), 5) for a in attention[position]],
                features={
                    name: float(value)
                    for name, value in zip(forecaster.features, sequences[position, -1], strict=True)
                },
                lat=_optional_float(row.get("lat")),
                lon=_optional_float(row.get("lon")),
                actual_label=_optional_int(row.get("label")),
            )
        )

    results.sort(key=lambda r: r.probability, reverse=True)
    return results


def _prediction_source(services, model: str | None) -> str:
    """ชื่อที่แผง confusion matrix ขอ — ไม่ระบุคือโมเดลปริยาย, ชื่อที่ไม่รู้จักเป็น 422"""
    name = model or services.forecast.default_name
    if name != "baseline" and name not in services.forecast.names:
        raise HTTPException(
            status_code=422,
            detail=f"ไม่รู้จักโมเดล {name!r} (ที่มี: {', '.join([*services.forecast.names, 'baseline'])})",
        )
    return name


@router.get(
    "/forecast/confusion-matrix",
    response_model=ConfusionMatrixSummary,
    summary="ตารางจำนวนนับ TP/FP/TN/FN ทุกจุดของกริด threshold — แผง confusion matrix",
)
def confusion_matrix_summary(
    request: Request,
    model: str | None = Query(
        None, description='ชื่อโมเดล หรือ "baseline" สำหรับ logistic baseline — ไม่ระบุคือโมเดลปริยาย'
    ),
    split: Literal["val", "test"] = Query(
        "test", description="val คือชุดที่ใช้เลือก threshold ไม่ใช่ชุดรายงานผล"
    ),
):
    """สไลเดอร์ threshold บนหน้าเว็บอ่านจากผลลัพธ์นี้ทั้งก้อน ไม่ยิง network ซ้ำ

    ``services.predictions.sweep`` โยน ``RuntimeError`` เมื่อยังไม่มีไฟล์ค่าทำนาย
    (หรือยังไม่มีค่าทำนายของ baseline) ซึ่ง handler ส่วนกลางใน ``main.py`` แปลงเป็น
    503 พร้อมข้อความบอกวิธีแก้ให้อัตโนมัติ — ไม่ต้อง try/except ซ้ำที่นี่
    """
    services = _services(request)
    return services.predictions.sweep(_prediction_source(services, model), split)


@router.get(
    "/forecast/confusion-matrix/samples",
    response_model=ConfusionMatrixSampleList,
    summary="รายชื่อ sample ในช่องหนึ่งของ confusion matrix",
)
def confusion_matrix_samples(
    request: Request,
    model: str | None = Query(None, description='ชื่อโมเดล หรือ "baseline" — ไม่ระบุคือโมเดลปริยาย'),
    split: Literal["val", "test"] = Query("test"),
    threshold: float = Query(
        ..., ge=0.0, le=1.0, description="threshold ณ ตำแหน่งสไลเดอร์ตอนคลิก (ไม่จำเป็นต้องเป็นค่าที่ freeze ไว้)"
    ),
    cell: Literal["tp", "fp", "fn", "tn"] = Query(..., description="ช่องที่คลิก"),
    limit: int = Query(200, ge=1, le=2000, description="จำนวนแถวสูงสุดที่คืน"),
):
    """เรียกเฉพาะตอนคลิกช่อง — ข้อมูลราย sample ของ val (สองหมื่นกว่าแถว) จึงไม่ถูก
    ส่งข้ามสายมาทั้งก้อนตั้งแต่ตอนโหลดหน้า (ดู endpoint สรุปด้านบนสำหรับ payload เบา)
    """
    services = _services(request)
    name = _prediction_source(services, model)
    rows, total = services.predictions.samples_in_cell(name, split, threshold, cell, limit=limit)

    samples = [
        ConfusionMatrixSample(
            harpnum=int(row.HARPNUM),
            noaa_ar=_optional_int(row.noaa_ar),
            issue_time=row.issue_time.isoformat(),
            probability=round(float(row.prob), 5),
        )
        for row in rows.itertuples()
    ]

    return ConfusionMatrixSampleList(
        model=name, split=split, threshold=threshold, cell=cell, total=total, samples=samples
    )


def _optional_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _optional_int(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
