"""สคีมาของ request/response — ทำให้ FastAPI สร้างเอกสาร /docs ได้อัตโนมัติ"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    forecast_model: bool = Field(description="โมเดลพยากรณ์ flare พร้อมใช้งานหรือไม่")
    segmentation_model: bool = Field(description="โมเดล segmentation พร้อมใช้งานหรือไม่")
    sequence_store: bool = Field(description="มีข้อมูล sequence ย้อนหลังให้เรียกดูหรือไม่")
    proton_flux: bool = Field(default=False, description="มีคลังฟลักซ์โปรตอนให้เรียกดูหรือไม่")
    xray_flux: bool = Field(default=False, description="มีฟลักซ์ X-ray ต่อเนื่องรายนาทีให้เรียกดูหรือไม่")
    aia_images: bool = Field(default=False, description="มีภาพ AIA สามชั้นบรรยากาศให้เรียกดูหรือไม่")
    n_frames: int = Field(description="จำนวนเฟรมภาพที่ประมวลผลไว้แล้ว")
    n_aia_frames: int = Field(default=0, description="จำนวนเฟรมที่มีภาพ AIA อย่างน้อยหนึ่งช่อง")


class HarpSummary(BaseModel):
    harpnum: int
    noaa_ar: int | None = None
    n_samples: int
    n_positive: int = Field(description="จำนวนหน้าต่างเวลาที่เกิด flare จริงตามมา")
    first_time: str
    last_time: str


class ForecastPoint(BaseModel):
    """ผลพยากรณ์ ณ เวลาหนึ่ง"""

    harpnum: int
    noaa_ar: int | None = None
    issue_time: str
    probability: float = Field(ge=0.0, le=1.0)
    predicted_positive: bool
    threshold: float
    risk_level: str = Field(description="low | moderate | elevated | high")
    attention: list[float] = Field(description="น้ำหนักความสนใจรายชั่วโมงย้อนหลัง")
    features: dict[str, float] = Field(description="ค่า SHARP parameters ล่าสุด (ค่าดิบ)")
    lat: float | None = None
    lon: float | None = None
    actual_label: int | None = Field(default=None, description="ผลจริงที่เกิดขึ้น (มีเฉพาะข้อมูลย้อนหลัง)")


class ForecastSeriesResponse(BaseModel):
    harpnum: int
    noaa_ar: int | None = None
    n_points: int
    horizon_hours: int
    positive_class: str
    threshold: float
    times: list[str]
    probabilities: list[float]
    actual_labels: list[int | None]
    latest: ForecastPoint | None = None


class AiaIntensityOut(BaseModel):
    """ความเข้มแสงของ active region หนึ่งดวงในช่อง AIA หนึ่งช่อง

    หน่วยเป็น DN/s (หารด้วย EXPTIME แล้ว) แต่ **ไม่ได้แก้ instrument degradation**
    จึงเทียบกันได้เฉพาะภายในเฟรมเดียวกันเท่านั้น ห้ามเทียบข้ามปี
    """

    channel: str = Field(description='คีย์ของช่อง เช่น "171"')
    wavelength: int
    mean: float = Field(description="ค่าเฉลี่ยในพื้นที่ mask — ตัวเลขหลักที่ใช้เทียบ AR")
    median: float
    p95: float = Field(
        description="เปอร์เซ็นไทล์ที่ 95 (ใช้แทน max เพราะภาพยังไม่ได้ตัดรังสีคอสมิก)"
    )
    total: float = Field(description="ผลรวมทั้งพื้นที่ — AR ใหญ่แต่จาง อาจแผ่รวมมากกว่า AR เล็กที่สว่าง")
    n_pixels: int


class LayerOut(BaseModel):
    """เลเยอร์ภาพหนึ่งชั้นที่หน้าเว็บเลือกแสดงได้"""

    key: str = Field(description='"mag" หรือคีย์ของช่อง AIA เช่น "304"')
    label: str
    region: str = Field(description="ชั้นบรรยากาศที่ช่องนี้มองเห็น")
    available: bool = Field(description="เฟรมนี้มีข้อมูลของเลเยอร์นี้หรือไม่")


class IntensitySeriesOut(BaseModel):
    """ความเข้มแสงตามเวลาของ active region หนึ่งดวง (หนึ่ง track)"""

    track_id: int
    label: str = Field(description='ชื่อที่แสดงบนกราฟ เช่น "AR 3"')
    n_points: int
    max_area_px: int
    times: list[str] = Field(description="เวลาของแต่ละจุด เรียงจากเก่าไปใหม่")
    b_peak: list[float] = Field(
        default_factory=list,
        description="ค่าสนามแม่เหล็กสูงสุดใน mask (Gauss) เรียงตาม times",
    )
    flux: list[float] = Field(
        default_factory=list,
        description="ฟลักซ์แม่เหล็กรวม (Mx) เรียงตาม times",
    )
    areas: list[int] = Field(
        default_factory=list,
        description="ขนาดพื้นที่ (pixels) เรียงตาม times",
    )
    series: dict[str, list[float | None]] = Field(
        description="คีย์คือช่อง AIA ค่าคือ mean DN/s เรียงตรงกับ times · null = เฟรมนั้นไม่มีข้อมูล"
    )


class IntensitySeriesResponse(BaseModel):
    """ความเข้มแสงของ AR ทุกดวงตามเวลา แยกตามชั้นบรรยากาศ"""

    n_frames: int = Field(description="จำนวนเฟรมที่ประมวลผลจริง")
    n_tracks: int
    frames: list[str] = Field(description="เวลาของเฟรมที่ใช้ เรียงจากเก่าไปใหม่")
    channels: list[LayerOut] = Field(description="ช่อง AIA พร้อมสถานะว่ามีข้อมูลในช่วงนี้หรือไม่")
    tracks: list[IntensitySeriesOut]
    used_ground_truth: bool
    note: str | None = Field(
        default=None,
        description="คำอธิบายข้อจำกัดของช่วงเวลาที่เลือก เช่น เฟรมห่างกันเกินกว่าจะติดตาม AR ได้",
    )
    nearest_frame: str | None = Field(
        default=None,
        description="วันที่ของเฟรมที่ใกล้กับช่วงที่เลือกมากที่สุด (มีเมื่อ n_frames=0 เท่านั้น) รูปแบบ YYYY-MM-DD",
    )


class DetectionOut(BaseModel):
    label: int
    rank: int = Field(
        description="ลำดับตามขนาดพื้นที่ (1 = ใหญ่สุด) — ตรงกับเลขที่เขียนกำกับบนภาพ"
    )
    centroid_x: float
    centroid_y: float
    bbox: list[int]
    area_px: int
    lon: float | None = None
    lat: float | None = None
    total_unsigned_flux: float
    mean_field: float
    max_abs_field: float
    intensities: list[AiaIntensityOut] = Field(
        default_factory=list, description="ความเข้มแสงแยกตามช่อง AIA (ว่างถ้าเฟรมนี้ไม่มีภาพ AIA)"
    )


class SegmentResponse(BaseModel):
    timestamp: str
    image_size: int
    image_png: str = Field(description="ภาพของเลเยอร์ที่เลือก พร้อมขอบ mask (base64 PNG)")
    n_regions: int
    detections: list[DetectionOut]
    predicted_area_fraction: float
    has_ground_truth: bool
    dice_vs_ground_truth: float | None = None
    layer: str = Field(default="mag", description="เลเยอร์ที่ส่งกลับมาจริง (อาจถอยเป็น mag ถ้าที่ขอไม่มีข้อมูล)")
    layers: list[LayerOut] = Field(
        default_factory=list, description="เลเยอร์ทั้งหมดพร้อมสถานะความพร้อมของเฟรมนี้"
    )


class TrackPointOut(BaseModel):
    time: str
    lon: float | None = None
    lat: float | None = None
    centroid_x: float
    centroid_y: float
    area_px: int
    total_unsigned_flux: float


class TrackOut(BaseModel):
    track_id: int
    state: str
    n_points: int
    duration_hours: float
    start_time: str | None = None
    end_time: str | None = None
    max_area_px: int
    points: list[TrackPointOut]


class TrackResponse(BaseModel):
    n_tracks: int
    n_frames_processed: int
    tracks: list[TrackOut]


class FlareEvent(BaseModel):
    peak_time: str
    goes_class: str
    peak_flux: float
    noaa_ar: int | None = None
    harpnum: int | None = None
    start_time: str | None = Field(
        default=None, description="เวลาเริ่มของเหตุการณ์ — ใช้วาดเส้น light curve"
    )
    end_time: str | None = Field(default=None, description="เวลาสิ้นสุดของเหตุการณ์")
    lat: float | None = Field(default=None, description="ละติจูด heliographic ของ flare (เหนือเป็นบวก)")
    lon: float | None = Field(
        default=None, description="ลองจิจูด heliographic ของ flare (ตะวันตกเป็นบวก)"
    )


class GoesResponse(BaseModel):
    start: str
    end: str
    n_events: int
    events: list[FlareEvent]
    class_counts: dict[str, int]
    n_located: int = Field(default=0, description="จำนวน flare ที่มีพิกัด — วาดลงแผนที่จานสุริยะได้")


class ProtonChannel(BaseModel):
    """หนึ่งเส้นในกราฟโปรตอน — ช่องพลังงานเป็นแบบสะสม (>1 ครอบ >10 ครอบ >50 ...)"""

    key: str
    label: str
    values: list[float | None] = Field(
        description="ฟลักซ์ [pfu] เรียงตาม offsets_hours · null = ไม่มีข้อมูล (ห้ามลากเส้นข้าม)"
    )


class ProtonResponse(BaseModel):
    peak_time: str = Field(description="เวลาที่ flare พีค — จุดศูนย์กลางของหน้าต่าง")
    goes_class: str | None = None
    before_hours: int
    after_hours: int
    step_minutes: int
    offsets_hours: list[float] = Field(description="เวลาสัมพัทธ์จากจุดพีค (ลบ = ก่อน flare)")
    times: list[str]
    channels: list[ProtonChannel]
    sources: list[str] = Field(description="ดาวเทียม/ผลิตภัณฑ์ที่ให้ข้อมูลในหน้าต่างนี้")
    n_samples: int
    has_data: bool
    flux_at_peak: float | None = Field(default=None, description=">10 MeV ณ เวลาที่ flare พีค [pfu]")
    max_after: float | None = Field(
        default=None, description=">10 MeV สูงสุดตั้งแต่จุดพีคจนสุดหน้าต่าง [pfu]"
    )
    max_after_time: str | None = None
    s_scale: str | None = Field(default=None, description="ระดับพายุรังสีสุริยะ NOAA (S1-S5) ของค่าสูงสุด")


class XrayResponse(BaseModel):
    start: str
    end: str
    times: list[str]
    flux: list[float | None] = Field(
        description="ฟลักซ์ช่องยาว (1-8 Å) หน่วย W/m2 · null = อ่านค่าไม่ได้ (ไม่ใช่ศูนย์)"
    )
    n_samples: int
    decimated: bool = Field(description="ช่วงยาวเกินจนต้องยุบข้อมูล — ใช้ค่าสูงสุดของแต่ละช่วงย่อยแทนทุกจุด")


class ConfusionMatrixSummary(BaseModel):
    """จำนวนนับ TP/FP/TN/FN ที่ทุกจุดของกริด threshold — ฐานข้อมูลของแผง confusion matrix

    หน้าเว็บมีหน้าที่แค่เปิดตารางที่ index ตามตำแหน่งสไลเดอร์ แล้วคำนวณตัวชี้วัด
    (TSS/recall/precision/HSS2) จากจำนวนนับสี่ค่าด้วยเลขคณิตล้วน — ตรรกะการนับ
    ทั้งหมดอยู่ใน ``sunseg.metrics.threshold_sweep`` เท่านั้น
    """

    model: str = Field(description='"lstm" หรือ "baseline"')
    split: str = Field(description='"val" หรือ "test" — val คือชุดที่ใช้เลือก threshold ไม่ใช่ชุดรายงานผล')
    thresholds: list[float] = Field(description="กริด threshold ที่กวาด (np.linspace(0.01, 0.99, 200))")
    tp: list[int]
    fp: list[int]
    tn: list[int]
    fn: list[int]
    frozen_threshold: float = Field(description="threshold ที่เลือกจาก validation ตอนเทรน (ค่าเริ่มต้นของสไลเดอร์)")
    frozen_index: int = Field(description="ตำแหน่งของ frozen_threshold ใน thresholds")
    auc: float = Field(description="ไม่ขึ้นกับ threshold — ค่าเดียวไม่ว่าสไลเดอร์จะอยู่ตรงไหน")
    n: int = Field(description="จำนวน sample ทั้งหมดของ split นี้")
    n_positive: int = Field(description="จำนวน sample ที่เป็น positive จริง")


class ConfusionMatrixSample(BaseModel):
    """หนึ่งแถวในลิสต์ sample ของช่องที่คลิก — พอสำหรับกระโดดไปดูใน dashboard ต่อ"""

    harpnum: int
    noaa_ar: int | None = None
    issue_time: str
    probability: float = Field(ge=0.0, le=1.0)


class ConfusionMatrixSampleList(BaseModel):
    model: str
    split: str
    threshold: float = Field(description="threshold ที่ใช้แบ่งช่อง ณ ตอนคลิก (ตำแหน่งสไลเดอร์ ไม่ใช่ค่าที่ freeze ไว้เสมอไป)")
    cell: str = Field(description='"tp" | "fp" | "fn" | "tn"')
    total: int = Field(description="จำนวน sample ทั้งหมดในช่องนี้ ก่อนตัดตาม limit")
    samples: list[ConfusionMatrixSample] = Field(description="เรียงตามความน่าจะเป็นมาก -> น้อย")


class ModelInfoResponse(BaseModel):
    forecast: dict
    segmentation: dict
    data: dict


class ErrorResponse(BaseModel):
    detail: str
    hint: str | None = None


def parse_iso(value: str, field: str) -> datetime:
    """แปลงสตริง ISO เป็น datetime พร้อมข้อความ error ที่บอกวิธีแก้"""
    from fastapi import HTTPException

    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{field} ไม่ใช่รูปแบบวันเวลาที่ถูกต้อง: {value!r} "
            "(ตัวอย่างที่ถูก: 2014-01-01 หรือ 2014-01-01T12:00:00)",
        ) from exc
