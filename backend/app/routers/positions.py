"""API ตำแหน่ง flare บนจานสุริยะ — แผนที่ในหน้าแรกของเว็บ"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from ..schemas import FlarePositionsResponse

router = APIRouter(prefix="/api", tags=["positions"])


@router.get(
    "/flare-positions",
    response_model=FlarePositionsResponse,
    summary="flare ของแคตตาล็อกโมเดลพร้อมตำแหน่งจาก PositionFlare",
)
def flare_positions(request: Request):
    """คืน flare ทุกดวงระดับ C+ ที่โมเดลใช้ทำ label พร้อมพิกัดที่จับคู่มาจาก PositionFlare

    ชุด flare และคลาสตรงกับแคตตาล็อกของโมเดลเสมอ (ไม่เพิ่มดวงจาก PositionFlare) ดวงที่หาคู่
    ไม่เจอหรือ PositionFlare ไม่ทราบตำแหน่ง จะมี lat/lon เป็น null — ข้อมูลคงที่ สร้างไว้
    ล่วงหน้าด้วย ``scripts/data/build_flare_positions.py``
    """
    store = request.app.state.services.flare_positions
    if not store.available:
        raise HTTPException(
            status_code=503,
            detail="ยังไม่มีตำแหน่ง flare — รัน `python backend/scripts/data/build_flare_positions.py` ก่อน",
        )
    return store.payload()
