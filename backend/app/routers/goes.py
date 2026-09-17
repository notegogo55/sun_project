"""API สำหรับเส้นเวลาของ flare ที่ GOES ตรวจวัดได้"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import FlareEvent, GoesResponse, parse_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["goes"])


@router.get("/goes", response_model=GoesResponse, summary="รายการ flare ในช่วงเวลา")
def goes_events(
    request: Request,
    start: str = Query(..., description="เวลาเริ่มต้น เช่น 2014-01-01"),
    end: str = Query(..., description="เวลาสิ้นสุด เช่น 2014-02-01"),
    min_class: str = Query("C1.0", description="กรองเฉพาะ flare ที่แรงกว่านี้"),
    harpnum: int | None = Query(None, description="กรองเฉพาะ flare ของ HARP ดวงนี้"),
    # ค่าเริ่มต้นต้องครอบคลุมทุก flare ในช่วง time_range ทั้งหมดของโปรเจกต์ (2011-2025)
    # ไม่ใช่แค่จำนวนที่ "ดูน่าจะพอ" — .head(limit) ด้านล่างตัดเอาแค่เหตุการณ์แรกสุด
    # ตามเวลา ถ้า limit ต่ำกว่าจำนวนจริง เหตุการณ์ครึ่งหลังของช่วงจะหายไปเงียบๆ โดย
    # UI ไม่มีทางรู้ (ดู .scratch หรือ git log หัวข้อ xray dense-mode สำหรับที่มาของบั๊กนี้)
    # ที่ min_class=B1.0 (กว้างสุดที่ UI เลือกได้) ทั้งชุดข้อมูลมี ~9,800 เหตุการณ์
    # (วัดจริง 2026-09) ตั้ง default ไว้ที่เพดานสูงสุดที่ endpoint รองรับเพื่อกันชน
    # ในอนาคตเมื่อข้อมูลเพิ่มขึ้นเรื่อยๆ โดยไม่ต้องแก้ไฟล์นี้อีก
    limit: int = Query(20000, ge=1, le=20000),
):
    """คืน flare ที่เกิดในช่วงเวลาที่ระบุ สำหรับวาดกราฟเส้นเวลาบนหน้าเว็บ

    ข้อมูลมาจากรายการ GOES XRS ของ NOAA ที่ดาวน์โหลดไว้แล้วในเครื่อง จึงตอบได้ทันที
    โดยไม่ต้องเรียกบริการภายนอก
    """
    from sunseg.data.goes_class import goes_class_to_flux

    services = request.app.state.services
    if services.flares is None:
        raise HTTPException(
            status_code=503,
            detail="ยังไม่มีรายการ flare — รัน `python backend/scripts/download_metadata.py` ก่อน",
        )

    start_dt = parse_iso(start, "start")
    end_dt = parse_iso(end, "end")
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end ต้องมาหลัง start")

    try:
        threshold = goes_class_to_flux(min_class)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"min_class ไม่ถูกต้อง: {min_class!r} (ตัวอย่าง: C1.0, M1.0)"
        ) from exc

    frame = services.flares
    mask = (
        (frame["peak_time"] >= pd.Timestamp(start_dt))
        & (frame["peak_time"] <= pd.Timestamp(end_dt))
        & (frame["peak_flux"] >= threshold)
    )
    if harpnum is not None:
        mask &= frame["HARPNUM"] == harpnum
    selected = frame[mask]

    if harpnum is None:
        # ไฟล์เก็บเป็นคู่ (flare, HARP) — flare ดวงเดียวที่แมปได้หลาย HARP จะซ้ำ
        selected = selected.drop_duplicates(subset=["peak_time", "goes_class"])

    selected = selected.sort_values("peak_time").head(limit)

    events = [
        FlareEvent(
            peak_time=row.peak_time.isoformat(),
            goes_class=str(row.goes_class),
            peak_flux=float(row.peak_flux),
            noaa_ar=None if pd.isna(row.noaa_ar) else int(row.noaa_ar),
            harpnum=None if pd.isna(getattr(row, "HARPNUM", None)) else int(row.HARPNUM),
            start_time=_optional_time(getattr(row, "start_time", None)),
            end_time=_optional_time(getattr(row, "end_time", None)),
            **_coordinates(getattr(row, "location", None)),
        )
        for row in selected.itertuples()
    ]

    letters = selected["goes_class"].astype("string").str[0].str.upper()
    counts = letters.value_counts()

    return GoesResponse(
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        n_events=len(events),
        events=events,
        class_counts={c: int(counts.get(c, 0)) for c in ["A", "B", "C", "M", "X"]},
        n_located=sum(1 for event in events if event.lat is not None),
    )


def _optional_time(value) -> str | None:
    """เวลาเริ่ม/สิ้นสุดอาจว่างในรายงานต้นทาง — คืน None แทนที่จะพัง"""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _coordinates(location) -> dict[str, float | None]:
    """แปลงตำแหน่งแบบ ``N11W82`` เป็น lat/lon สำหรับแผนที่จานสุริยะ"""
    from sunseg.data.noaa_flares import parse_location

    if location is None or pd.isna(location):
        return {"lat": None, "lon": None}

    parsed = parse_location(str(location))
    if parsed is None:
        return {"lat": None, "lon": None}
    return {"lat": parsed[0], "lon": parsed[1]}
