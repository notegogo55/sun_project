"""API สำหรับฟลักซ์ X-ray ต่อเนื่องรายนาที (GOES-15 ช่วง 2011-2017, GOES-16 ช่วง case study พ.ค. 2024)"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import XrayResponse, parse_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["xray"])


@router.get("/xray", response_model=XrayResponse, summary="ฟลักซ์ X-ray ต่อเนื่องในช่วงเวลา")
def xray_series(
    request: Request,
    start: str = Query(..., description="เวลาเริ่มต้น เช่น 2014-10-18"),
    end: str = Query(..., description="เวลาสิ้นสุด เช่น 2014-10-31"),
):
    """คืนฟลักซ์ช่องยาว (1-8 Å) รายนาทีในช่วงเวลาที่ระบุ สำหรับวาดเส้น X-ray แบบต่อเนื่อง

    ต่างจาก /api/goes ที่คืนแค่ 3 จุดต่อ flare (เริ่ม/peak/จบ) — endpoint นี้คืนฟลักซ์
    ที่วัดได้จริงทุกนาที ทำให้เส้นกราฟขึ้นลงต่อเนื่องเหมือนหน้า SWPC แทนที่จะเป็นหนาม
    """
    services = request.app.state.services
    store = services.xray
    if not store.available:
        raise HTTPException(
            status_code=503,
            detail=f"ไม่พบไฟล์ฟลักซ์ X-ray ที่ {store.root} — รัน backend/scripts/data/download_xray.py ก่อน",
        )

    start_dt = parse_iso(start, "start")
    end_dt = parse_iso(end, "end")
    if end_dt <= start_dt:
        raise HTTPException(status_code=422, detail="end ต้องมาหลัง start")

    series = store.series(start_dt, end_dt)

    return XrayResponse(
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        times=[t.isoformat() for t in series.times],
        flux=series.flux,
        n_samples=len(series.times),
        decimated=series.decimated,
    )
