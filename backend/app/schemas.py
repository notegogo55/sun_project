"""สคีมาของ request/response — ทำให้ FastAPI สร้างเอกสาร /docs ได้อัตโนมัติ"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    forecast_model: bool = Field(description="มีโมเดลพยากรณ์ flare พร้อมใช้งานอย่างน้อยหนึ่งตัวหรือไม่")
    forecast_models: dict[str, bool] = Field(
        default_factory=dict, description="ความพร้อมรายโมเดล เรียงตาม configs/forecast.yaml"
    )
    default_forecast_model: str | None = Field(default=None, description="โมเดลที่หน้าเว็บเปิดมาเจอก่อน")
    segmentation_model: bool = Field(description="โมเดล segmentation พร้อมใช้งานหรือไม่")
    sequence_store: bool = Field(description="มีข้อมูล sequence ย้อนหลังให้เรียกดูหรือไม่")
    proton_flux: bool = Field(default=False, description="มีคลังฟลักซ์โปรตอนให้เรียกดูหรือไม่")
    xray_flux: bool = Field(default=False, description="มีฟลักซ์ X-ray ต่อเนื่องรายนาทีให้เรียกดูหรือไม่")
    aia_images: bool = Field(default=False, description="มีภาพ AIA สามชั้นบรรยากาศให้เรียกดูหรือไม่")
    n_frames: int = Field(description="จำนวนเฟรมภาพที่ประมวลผลไว้แล้ว")
    n_aia_frames: int = Field(default=0, description="จำนวนเฟรมที่มีภาพ AIA อย่างน้อยหนึ่งช่อง")
    flare_positions: bool = Field(
        default=False, description="มีตำแหน่ง flare จาก PositionFlare ที่จับคู่กับแคตตาล็อกโมเดลแล้วหรือไม่"
    )
    class_forecast: bool = Field(
        default=False, description="คำพยากรณ์ระดับคลาส (<M/M/X) ของโมเดลหลัก LSTM + V3 พร้อมใช้งานหรือไม่"
    )


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


class FlarePositionsResponse(BaseModel):
    """flare ทุกดวงระดับ C+ ในแคตตาล็อกของโมเดล พร้อมตำแหน่งที่จับคู่มาจาก PositionFlare

    เป็นแบบ columnar: ทุก list ยาว ``n`` เท่ากัน ตำแหน่งที่ i ของทุก list คือ flare ดวงเดียวกัน
    (ส่งแบบนี้เล็กกว่า list ของ object หลายเท่า — หน้าแรกโหลดทั้งชุดครั้งเดียว)
    """

    epoch: str = Field(description="จุดเริ่มของคอลัมน์ t (UTC ไม่มีโซน)")
    n: int
    n_matched: int = Field(description="จับคู่กับ PositionFlare ได้กี่ดวง")
    n_located: int = Field(description="มีพิกัดวางบนแผนที่ได้กี่ดวง")
    tolerance_min: float = Field(description="เวลาพีคห่างกันได้สูงสุดตอนจับคู่ (นาที)")
    source: str | None = Field(default=None, description="CSV ของ PositionFlare ที่ใช้สร้าง")
    built_at: str | None = None
    t: list[int] = Field(description="เวลาพีค เป็นนาทีนับจาก epoch")
    start: list[int | None] = Field(description="เวลาเริ่ม เทียบกับพีค (นาที ติดลบ)")
    end: list[int | None] = Field(description="เวลาสิ้นสุด เทียบกับพีค (นาที)")
    goes_class: list[str] = Field(description="คลาสตามแคตตาล็อกของโมเดล — ตัวที่ใช้ทำ label")
    peak_flux: list[float]
    noaa_ar: list[int | None]
    harpnum: list[int | None] = Field(description="HARP ตัวแทน (เลขน้อยสุดที่แมปได้)")
    n_harps: list[int]
    match_dt_min: list[float | None] = Field(description="เวลาพีคของ PositionFlare ลบของโมเดล (นาที)")
    pf_goes_class: list[str | None] = Field(description="คลาสใน PositionFlare (สเกล science)")
    lat: list[float | None]
    lon: list[float | None]
    pos_source: list[str | None] = Field(description="ที่มาของตำแหน่ง: SWPC > XRS > XRS-HPC > AR")
    limb: list[bool | None] = Field(description="อยู่ใกล้ขอบจาน — พิกัดคลาดเคลื่อนสูงกว่าปกติ")
    pf_active_region: list[int | None]
    satellite: list[str | None]
    cycle: list[int | None]


class ForecastModelTestMetrics(BaseModel):
    tss: float | None = None
    auc: float | None = None
    recall: float | None = None
    precision: float | None = None
    hss2: float | None = None
    bss: float | None = None
    n: int | None = Field(default=None, description="จำนวน sample ของ test")
    n_positive: int | None = Field(default=None, description="จำนวน positive จริงใน test")


class ForecastModelSummary(BaseModel):
    """โมเดลพยากรณ์หนึ่งตัว — สิ่งที่ปุ่มเลือกโมเดลและตารางเทียบผลบนหน้าเว็บต้องรู้"""

    name: str = Field(description="ชื่อที่ใช้กับ ?model= และ artifacts/models/<name>.pt")
    label: str
    kind: str | None = Field(default=None, description="lstm | tcn | transformer | darnn (null ถ้ายังไม่ได้เทรน)")
    pooling: str | None = Field(
        default=None, description="attention | last | mean — last/mean ไม่มีน้ำหนัก attention ที่เรียนรู้ได้"
    )
    available: bool
    default: bool = Field(description="เป็นโมเดลปริยายของหน้าเว็บหรือไม่")
    threshold: float | None = None
    n_parameters: int | None = None
    val_tss: float | None = None
    test: ForecastModelTestMetrics
    baseline_test: ForecastModelTestMetrics | None = Field(
        default=None, description="ผลบน test ของ logistic baseline ที่เทรนคู่กับโมเดลนี้"
    )
    train_hint: str | None = Field(default=None, description="คำสั่งที่ต้องรันถ้ายังไม่ได้เทรน")


class ForecastSeriesResponse(BaseModel):
    model: str = Field(default="lstm", description="ชื่อโมเดลที่ใช้พยากรณ์")
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


class ClassForecastPoint(BaseModel):
    """คำพยากรณ์ระดับคลาสของโมเดลหลัก ณ issue_time หนึ่ง (ช่วงพยากรณ์ 24 ชม. ถัดไป)"""

    harpnum: int
    noaa_ar: int | None = None
    issue_time: str
    level: str = Field(description='ระดับที่ทำนาย: "<M" (ไม่มีระดับไหนเตือน) | "M" | "X"')
    true_level: str = Field(description="ระดับของ flare ที่เกิดจริงใน 24 ชม. ถัดไป (ข้อมูลย้อนหลัง)")
    prob_m: float = Field(ge=0.0, le=1.0, description="ความน่าจะเป็น ≥M1.0 เฉลี่ยข้าม seed")
    prob_x: float = Field(ge=0.0, le=1.0, description="ความน่าจะเป็น ≥X1.0 เฉลี่ยข้าม seed")
    n_alarm_m: int = Field(description="จำนวน seed ของระดับ M ที่เตือน")
    n_alarm_x: int = Field(description="จำนวน seed ของระดับ X ที่เตือน (threshold ราย seed)")
    split: str = Field(description="train | val | test — ค่าของ train คือข้อมูลที่โมเดลเคยเห็นตอนเทรน")
    lat: float | None = None
    lon: float | None = None


class ClassForecastSeriesResponse(BaseModel):
    harpnum: int
    noaa_ar: int | None = None
    mode: str = Field(description="จุดทำงาน: sensitive (เตือนไว) | strict (ระมัดระวัง)")
    cadence_hours: int | None = Field(default=None, description="ระยะห่างระหว่าง issue_time")
    horizon_hours: int
    points: list[ClassForecastPoint]
    latest: ClassForecastPoint


class ClassLevelMetrics(BaseModel):
    """คำพยากรณ์ระดับคลาสที่ตัดเป็นทวิภาค "≥ ระดับนี้" """

    n_positive: int
    tp: int
    fp: int
    tn: int
    fn: int
    tss: float
    hss2: float
    precision: float | None = None
    recall: float | None = None


class ClassEvaluation(BaseModel):
    levels: list[str]
    n: int
    confusion: list[list[int]] = Field(description="แถว = ระดับจริง, คอลัมน์ = ระดับที่ทำนาย (ลำดับตาม levels)")
    hss_multiclass: float = Field(description="Heidke skill score แบบ 3 คลาส")
    thresholds: dict[str, ClassLevelMetrics] = Field(description="ต่อระดับ M และ X")
    exact_on_events: float | None = Field(
        default=None, description="ในบรรดา sample ที่เกิด ≥M1.0 จริง สัดส่วนที่ทายระดับถูกเป๊ะ"
    )
    n_inconsistent: int = Field(description="ระดับ X เตือนแต่ระดับ M ไม่เตือน (นับเป็น X)")


class ClassForecastSummary(BaseModel):
    """โมเดลหลักของโปรเจค — ข้อมูลของโมเดลพร้อมผลบน val/test ของทุกจุดทำงาน"""

    label: str
    architecture: str
    variant: str
    available: bool
    hint: str | None = Field(default=None, description="คำสั่งที่ต้องรันถ้ายังไม่พร้อม")
    levels: list[str]
    modes: list[str] = Field(description="จุดทำงานทั้งหมด ตัวแรกคือค่าปริยาย")
    n_seeds: dict[str, int] = Field(default_factory=dict)
    strict_threshold: float | None = Field(
        default=None, description="threshold ของความน่าจะเป็นเฉลี่ยระดับ X ในจุดทำงาน strict (เลือกบน val)"
    )
    cadence_hours: int | None = None
    sequence_length: int | None = None
    features: list[str] = Field(default_factory=list)
    positive_classes: dict[str, str]
    evaluation: dict[str, dict[str, ClassEvaluation]] | None = Field(
        default=None, description="[mode][split] — null ถ้ายังไม่พร้อม"
    )


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

    model: str = Field(description='ชื่อโมเดล (เช่น "lstm", "tcn") หรือ "baseline" สำหรับ logistic baseline')
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
    forecast: dict = Field(description="โมเดลพยากรณ์ปริยาย")
    forecast_models: dict[str, dict] = Field(default_factory=dict, description="โมเดลพยากรณ์ทุกตัว")
    default_forecast_model: str | None = None
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
