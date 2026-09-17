"""FastAPI application สำหรับ sunseg — ให้บริการเฉพาะ API เท่านั้น

frontend/ เป็นโปรเจค Vite + React แยกต่างหากที่รันเป็นคนละ container/process เสมอ
(nginx ตอน deploy จริง, `npm run dev` ตอนพัฒนา) ตัวมันเองเป็นคน proxy คำขอ /api มาที่นี่
ฝั่งนี้จึงไม่ต้องรู้จักหรือเสิร์ฟไฟล์ frontend เลย — ดู frontend/nginx.conf และ
frontend/vite.config.js (dev server proxy) สำหรับการต่อสายทั้งสองฝั่งเข้าด้วยกัน

รันด้วย::

    uvicorn app.main:app --reload

แล้วเปิด http://localhost:8000/docs (เอกสาร API อัตโนมัติ)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# backend/app/main.py -> app -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from sunseg.logging_utils import setup_logging  # noqa: E402

from .routers import forecast, goes, positions, proton, segment, xray  # noqa: E402
from .schemas import HealthResponse, ModelInfoResponse  # noqa: E402
from .services import AppServices  # noqa: E402

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """โหลดโมเดลและข้อมูลครั้งเดียวตอน startup

    การโหลด checkpoint ใหม่ทุก request จะช้ากว่านี้หลายสิบเท่า — จึงต้องโหลดที่นี่
    แล้วเก็บไว้ใน ``app.state`` ให้ทุก endpoint ใช้ร่วมกัน
    """
    setup_logging()
    logger.info("=" * 62)
    logger.info("กำลังเริ่มต้น sunseg web application")
    logger.info("=" * 62)

    app.state.services = AppServices(device="cpu")

    logger.info("พร้อมใช้งานที่ http://localhost:8000  (API docs: /docs)")
    yield
    logger.info("ปิดการทำงาน")


app = FastAPI(
    title="sunseg — Solar Active Region Segmentation & Flare Forecast",
    description=(
        "ระบบติดตามสภาพอวกาศแบบครบวงจร: แบ่งส่วน active region ด้วย U-Net, "
        "ติดตามข้ามเวลาโดยชดเชยการหมุนของดวงอาทิตย์ และพยากรณ์การเกิด solar flare ด้วย LSTM"
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(forecast.router)
app.include_router(segment.router)
app.include_router(goes.router)
app.include_router(proton.router)
app.include_router(xray.router)
app.include_router(positions.router)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health(request: Request):
    """ตรวจว่าส่วนประกอบไหนพร้อมใช้งานบ้าง — หน้าเว็บเรียกตอนโหลดเพื่อปรับ UI"""
    services = request.app.state.services
    return HealthResponse(
        status="ok",
        forecast_model=services.forecast.available,
        segmentation_model=services.segmentation.available,
        sequence_store=services.sequences.available,
        proton_flux=services.proton.available,
        xray_flux=services.xray.available,
        aia_images=services.aia.available,
        n_frames=len(services.segmentation.list_frames()),
        n_aia_frames=services.aia.n_frames,
        flare_positions=services.flare_positions.available,
    )


@app.get("/api/info", response_model=ModelInfoResponse, tags=["system"])
def info(request: Request):
    """รายละเอียดของโมเดลและข้อมูล รวมถึงผลการวัดประสิทธิภาพบน test set"""
    return request.app.state.services.info()


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    """แปลง RuntimeError ให้เป็น 503 พร้อมข้อความที่บอกวิธีแก้

    RuntimeError ในโปรเจคนี้มักหมายถึง "ยังไม่ได้เทรนโมเดล" ซึ่งเป็นสถานะปกติของ
    ผู้ใช้ที่เพิ่งเริ่ม ไม่ใช่ข้อผิดพลาดของระบบ
    """
    logger.warning("%s -> %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})
