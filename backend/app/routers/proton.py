"""API สำหรับฟลักซ์โปรตอนรอบเวลาที่เกิด flare"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from ..schemas import ProtonChannel, ProtonResponse, parse_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["proton"])


@router.get("/proton", response_model=ProtonResponse, summary="อนุกรมโปรตอนรอบเวลาหนึ่ง")
def proton_window(
    request: Request,
    t: str = Query(..., description="เวลาที่ flare พีค เช่น 2012-03-07T00:24:00"),
    before: int | None = Query(None, ge=1, le=168, description="ชั่วโมงก่อนจุดพีค"),
    after: int | None = Query(None, ge=1, le=336, description="ชั่วโมงหลังจุดพีค"),
    goes_class: str | None = Query(None, description="คลาสของ flare — ใช้เขียนป้ายบนกราฟ"),
):
    """คืนฟลักซ์โปรตอนสี่ช่องพลังงานรอบเวลาที่ระบุ พร้อมตัวเลขสรุปของช่อง >10 MeV

    หน้าเว็บเรียกทีละ flare ตอนผู้ใช้คลิกจุดบนจานสุริยะหรือบนเส้น X-ray — ไม่ได้ส่ง
    ทั้งคลังมาให้ client เพราะอนุกรม 28 ปีมีเกือบ 3 ล้านตัวอย่างต่อช่อง
    """
    from sunseg.data.proton_flux import CHANNELS, REFERENCE_CHANNEL, STEP_MINUTES, s_scale_level

    services = request.app.state.services
    store = services.proton
    if not store.available:
        raise HTTPException(
            status_code=503,
            detail=f"ไม่พบคลัง proton flux ที่ {store.root} — ตั้ง SUNSEG_PROTON_DIR ใน .env "
            "ให้ชี้ไปที่โฟลเดอร์ particle_clean",
        )

    center = parse_iso(t, "t")
    before_hours = before if before is not None else services.config.proton.before_hours
    after_hours = after if after is not None else services.config.proton.after_hours

    window = store.window(center, before_hours, after_hours)
    peak = window.peak_after(window.center, REFERENCE_CHANNEL)
    max_after = None if peak is None else peak[0]

    step_hours = STEP_MINUTES / 60
    origin = window.times[0]
    offsets = [
        round((origin - window.center).total_seconds() / 3600 + i * step_hours, 4)
        for i in range(len(window.times))
    ]

    return ProtonResponse(
        peak_time=window.center.isoformat(),
        goes_class=goes_class,
        before_hours=before_hours,
        after_hours=after_hours,
        step_minutes=STEP_MINUTES,
        offsets_hours=offsets,
        times=[time.isoformat() for time in window.times],
        channels=[
            ProtonChannel(key=key, label=label, values=window.values[key])
            for key, label, _ in CHANNELS
        ],
        sources=window.sources,
        n_samples=len(window.times),
        has_data=window.has_data,
        flux_at_peak=window.value_at(window.center, REFERENCE_CHANNEL),
        max_after=max_after,
        max_after_time=None if peak is None else peak[1].isoformat(),
        s_scale=s_scale_level(max_after),
    )
