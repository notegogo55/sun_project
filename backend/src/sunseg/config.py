"""โหลดและ validate ไฟล์ config (YAML + .env) เป็น pydantic model.

ทุกโมดูลในโปรเจคควรเรียก ``load_*_config()`` แทนการอ่าน YAML เองโดยตรง เพื่อให้ path
ถูก resolve เป็น absolute เหมือนกันทั้งระบบ และเพื่อให้ typo ใน config ระเบิดตั้งแต่ตอนโหลด
ไม่ใช่กลางการเทรน

ไฟล์ใน ``backend/configs/``:

- ``data.yaml`` — แหล่งข้อมูล, path, การแบ่ง split (:func:`load_data_config`)
- ``unet.yaml`` — U-Net (:func:`load_unet_config`)
- ``forecast.yaml`` — โมเดลพยากรณ์ทั้งสี่ตัว (:func:`load_forecast_config`)
- ``tracking.yaml`` — detection + tracker (:func:`load_tracking_config`)
- ``study/`` — งานเปรียบเทียบสถาปัตยกรรม × ชุด feature (:func:`load_study_architectures`,
  ``sunseg.data.study_dataset.load_variants``)
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# backend/src/sunseg/config.py -> src/sunseg -> src -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = BACKEND_ROOT / "configs"
FORECAST_CONFIG_FILE = CONFIG_DIR / "forecast.yaml"
STUDY_CONFIG_DIR = CONFIG_DIR / "study"
STUDY_ARCHITECTURES_FILE = STUDY_CONFIG_DIR / "architectures.yaml"
STUDY_VARIANTS_FILE = STUDY_CONFIG_DIR / "variants.yaml"

# data/, artifacts/, .env อยู่ที่ root ของ repo (นอก backend/) — ไม่ใช่โค้ด จึงไม่ย้ายตาม backend
PROJECT_ROOT = BACKEND_ROOT.parent


class _Strict(BaseModel):
    """ปฏิเสธ key ที่ไม่รู้จัก — typo ใน YAML จะ fail ทันทีแทนที่จะถูกละเลยเงียบๆ"""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #


class Paths(_Strict):
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path

    def resolve(self, root: Path) -> Paths:
        """แปลง path ที่เขียนไว้แบบ relative ใน YAML ให้เป็น absolute จาก project root"""
        return Paths(**{k: root / v for k, v in self.model_dump().items()})

    def mkdirs(self) -> None:
        for p in self.model_dump().values():
            Path(p).mkdir(parents=True, exist_ok=True)


class JsocConfig(_Strict):
    sharp_series: str
    sharp_bitmap_series: str
    fulldisk_series: str
    max_retries: int = 4
    retry_backoff_s: float = 5.0
    export_batch_size: int = 200

    @property
    def email(self) -> str:
        """อีเมล JSOC อ่านจาก env เท่านั้น — ไม่เก็บใน YAML ที่ถูก commit"""
        email = os.environ.get("SUNSEG_JSOC_EMAIL", "").strip()
        if not email:
            raise RuntimeError(
                "ไม่พบ SUNSEG_JSOC_EMAIL ใน environment.\n"
                "1) ลงทะเบียนอีเมลกับ JSOC: http://jsoc.stanford.edu/ajax/register_email.html\n"
                "2) คัดลอก .env.example เป็น .env แล้วกรอกอีเมลที่ลงทะเบียนไว้"
            )
        return email


class TimeRange(_Strict):
    start: date
    end: date

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: date, info: Any) -> date:
        start = info.data.get("start")
        if start and v <= start:
            raise ValueError(f"{cls.__name__}: end ({v}) ต้องมาหลัง start ({start})")
        return v


class SharpConfig(_Strict):
    cadence_hours: int
    require_quality_zero: bool
    abs_lon_max_deg: float
    min_npix: int
    features: list[str]
    meta_keys: list[str]

    @field_validator("features")
    @classmethod
    def _no_duplicate_features(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            dupes = {f for f in v if v.count(f) > 1}
            raise ValueError(f"sharp.features มีชื่อซ้ำ: {sorted(dupes)}")
        return v

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def all_keys(self) -> list[str]:
        """keyword ทั้งหมดที่ต้องขอจาก JSOC (meta + features, ไม่ซ้ำ, คงลำดับ)"""
        seen: dict[str, None] = {}
        for k in [*self.meta_keys, *self.features]:
            seen.setdefault(k, None)
        return list(seen)


class FullDiskConfig(_Strict):
    cadence_hours: int
    target_size: int
    bitmap_ar_value: int
    min_ar_area_px: int
    delete_fits_after_process: bool


class FlareConfig(_Strict):
    horizon_hours: int
    positive_goes_class: str
    hek_chunk_days: int


class ProtonConfig(_Strict):
    """คลังฟลักซ์โปรตอน GOES ราย 5 นาที — อยู่นอก repo

    เป็นข้อมูลดิบเกือบ 800 MB ที่โปรเจคอื่นดูแลอยู่ sunseg แค่อ่านอย่างเดียว
    (ดูเหตุผลเต็มใน ``sunseg/data/proton_flux.py``) path จึงต่างกันไปในแต่ละเครื่อง
    และตั้งทับค่าใน YAML ที่ถูก commit ได้ด้วย env ``SUNSEG_PROTON_DIR``
    """

    root: Path
    before_hours: int = 12
    after_hours: int = 60

    @field_validator("before_hours", "after_hours")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("proton.before_hours/after_hours ต้องมากกว่า 0")
        return v

    def resolve(self, root: Path) -> ProtonConfig:
        """env ชนะ YAML เสมอ แล้วค่อยแปลง path แบบ relative ให้เป็น absolute"""
        override = os.environ.get("SUNSEG_PROTON_DIR", "").strip()
        path = Path(override) if override else self.root
        return self.model_copy(update={"root": path if path.is_absolute() else root / path})


class PositionFlareConfig(_Strict):
    """แคตตาล็อก flare ของ PositionFlare — แหล่งตำแหน่ง flare บนจานสุริยะของหน้าแรก
    (ดู ``sunseg/data/flare_positions.py``)

    อยู่นอก repo เหมือน :class:`ProtonConfig` — sunseg อ่านอย่างเดียวตอนรัน
    ``scripts/data/build_flare_positions.py`` เท่านั้น ตัวแอปอ่านผลที่สร้างไว้ใน data/processed
    จึงไม่ต้องเข้าถึง path นี้ตอนรัน ตั้งทับได้ด้วย env ``SUNSEG_POSITION_FLARE_CSV``
    """

    csv: Path
    match_tolerance_min: float = 10.0

    @field_validator("match_tolerance_min")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("position_flare.match_tolerance_min ต้องมากกว่า 0")
        return v

    def resolve(self, root: Path) -> PositionFlareConfig:
        """env ชนะ YAML เสมอ แล้วค่อยแปลง path แบบ relative ให้เป็น absolute"""
        override = os.environ.get("SUNSEG_POSITION_FLARE_CSV", "").strip()
        path = Path(override) if override else self.csv
        return self.model_copy(update={"csv": path if path.is_absolute() else root / path})


class AiaChannel(_Strict):
    """ช่อง AIA หนึ่งช่อง = ชั้นบรรยากาศหนึ่งชั้น"""

    wavelength: int
    label: str
    region: str
    display_vmax: float

    @property
    def key(self) -> str:
        """คีย์ที่ใช้ใน API และชื่อโฟลเดอร์ — ``"171"``, ``"304"``, ``"1600"``"""
        return str(self.wavelength)

    @field_validator("display_vmax")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("aia.channels[].display_vmax ต้องมากกว่า 0")
        return v


class AiaConfig(_Strict):
    """ภาพ AIA ที่ reproject ลงกริดเดียวกับเฟรม HMI แล้ว

    ค่าที่เก็บเป็น DN/s (หารด้วย EXPTIME แล้ว) **ไม่ได้แก้ instrument degradation**
    จึงเทียบกันได้เฉพาะภายในเฟรมเดียวเท่านั้น ห้ามเทียบข้ามปี

    ตั้งทับ ``root`` ด้วย env ``SUNSEG_AIA_DIR`` ได้ เหมือน :class:`ProtonConfig`
    ถ้าไม่มีข้อมูล แอปยังเปิดได้ตามปกติ เพียงแต่เลเยอร์ AIA จะขึ้นว่าไม่มีข้อมูล
    """

    root: Path
    synoptic_base_url: str
    target_size: int
    channels: list[AiaChannel]

    @field_validator("channels")
    @classmethod
    def _non_empty(cls, v: list[AiaChannel]) -> list[AiaChannel]:
        if not v:
            raise ValueError("aia.channels ต้องมีอย่างน้อยหนึ่งช่อง")
        return v

    def channel(self, key: str) -> AiaChannel | None:
        return next((c for c in self.channels if c.key == key), None)

    def resolve(self, root: Path) -> AiaConfig:
        """env ชนะ YAML เสมอ แล้วค่อยแปลง path แบบ relative ให้เป็น absolute"""
        override = os.environ.get("SUNSEG_AIA_DIR", "").strip()
        path = Path(override) if override else self.root
        return self.model_copy(update={"root": path if path.is_absolute() else root / path})


class SequenceConfig(_Strict):
    length: int
    stride_hours: int
    max_missing_frac: float


class CVConfig(_Strict):
    """พารามิเตอร์ rolling/blocked-window cross-validation (ดู ``sunseg.data.splits.rolling_folds``)

    train มีขนาดคงที่ ``train_years`` ปีต่อ fold แล้วเลื่อนไปข้างหน้าทีละ ``step_years`` ปี
    (blocked — ไม่ expand เหมือน single split ปกติ) เพราะข้อมูลมี autocorrelation ภายใน HARP
    และ label มองอนาคต จึงต้องเลื่อนตามเวลาเท่านั้น ห้ามสุ่มแบ่งแบบ k-fold ทั่วไป
    """

    train_years: int
    val_years: int
    test_years: int
    step_years: int
    min_val_positive: int = 40
    min_test_positive: int = 40
    # expanding window: train เริ่มที่จุดตั้งต้นเสมอแล้วยาวขึ้นทีละ step_years ต่อ fold (ยังเลื่อนตามเวลา
    # ไปข้างหน้าเท่านั้น) — ปริยาย False = blocked แบบเดิม · ใช้ในงาน feature-evidence-cv เพื่อให้ทุกปี
    # ที่มี flare ได้เป็น test โดย fold แรก ๆ ไม่ต้องทิ้งข้อมูล train
    expanding: bool = False
    # จุดตั้งต้นของ fold แรก — None = วันแรกที่มีข้อมูล (พฤติกรรมเดิม) · ตั้งเป็น 1 ม.ค. เพื่อให้ fold ตรงปีปฏิทิน
    anchor: date | None = None

    @field_validator("train_years", "val_years", "test_years", "step_years")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("split.cv: train_years/val_years/test_years/step_years ต้องมากกว่า 0")
        return v


class SplitConfig(_Strict):
    train_end: date
    val_end: date
    gap_days: int
    # ขอบเขตเสริมของ fold หนึ่งใน cross-validation — None แปลว่าไม่จำกัด (พฤติกรรมเดิมของ
    # single split: train เอาทุกอย่างก่อน train_end, test เอาทุกอย่างหลัง val_end+gap)
    # ดู sunseg.data.splits.rolling_folds ซึ่งเป็นตัวเติมสองฟิลด์นี้ให้ต่อ fold
    train_start: date | None = None
    test_end: date | None = None
    cv: CVConfig | None = None


class DataConfig(_Strict):
    paths: Paths
    jsoc: JsocConfig
    time_range: TimeRange
    sharp: SharpConfig
    fulldisk: FullDiskConfig
    flare: FlareConfig
    proton: ProtonConfig
    position_flare: PositionFlareConfig
    aia: AiaConfig
    sequence: SequenceConfig
    split: SplitConfig


# --------------------------------------------------------------------------- #
# unet.yaml
# --------------------------------------------------------------------------- #


class UNetModelConfig(_Strict):
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 32
    depth: int = 4
    norm: Literal["batch", "instance", "group"] = "batch"
    bilinear_upsample: bool = True


class UNetDataConfig(_Strict):
    image_size: int
    norm_scale_gauss: float


class UNetTrainConfig(_Strict):
    epochs: int
    batch_size: int
    grad_accum_steps: int = 1
    lr: float
    weight_decay: float
    amp: bool = True
    grad_clip: float = 1.0
    early_stop_patience: int
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    warmup_epochs: int = 0
    num_workers: int = 4
    seed: int = 42


class UNetLossConfig(_Strict):
    bce_weight: float
    dice_weight: float
    pos_weight: float


class AugmentConfig(_Strict):
    hflip: bool
    vflip: bool
    rotate_deg: float
    intensity_jitter: float
    allow_polarity_flip: bool = False


class UNetEvalConfig(_Strict):
    threshold: float
    object_iou_threshold: float


class UNetConfig(_Strict):
    model: UNetModelConfig
    data: UNetDataConfig
    train: UNetTrainConfig
    loss: UNetLossConfig
    augment: AugmentConfig
    eval: UNetEvalConfig


# --------------------------------------------------------------------------- #
# forecast.yaml — โมเดลพยากรณ์ flare สี่สถาปัตยกรรม (LSTM, TCN, Transformer, DA-RNN)
# --------------------------------------------------------------------------- #


class ForecastTrainConfig(_Strict):
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float = 1.0
    early_stop_patience: int
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    warmup_epochs: int = 0
    seed: int = 42


class ForecastLossConfig(_Strict):
    type: Literal["focal", "bce"] = "focal"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


class ForecastEvalConfig(_Strict):
    select_threshold_on: Literal["val", "train"] = "val"
    primary_metric: Literal["tss", "hss2", "auc"] = "tss"


class _ArchModelBase(_Strict):
    """สิ่งที่ทุกสถาปัตยกรรมมีเหมือนกัน

    ``pooling`` เป็น hyperparameter ของ **ทุก** สถาปัตยกรรม ไม่ใช่ของ LSTM ตัวเดียว —
    ถ้า LSTM ตัวเดียวถูกอนุญาตให้ทิ้ง attention ได้ การเปรียบเทียบจะเอนไปทางเดียว
    (ดู docs/adr/0001-tune-per-architecture.md)
    """

    dropout: float
    pooling: Literal["attention", "last", "mean"] = "attention"


class LSTMArchConfig(_ArchModelBase):
    kind: Literal["lstm"]
    hidden_size: int
    num_layers: int = 1


class TCNArchConfig(_ArchModelBase):
    kind: Literal["tcn"]
    channels: int
    kernel_size: int = 3
    # receptive field = 1 + Σ (kernel_size−1)·dilation ต้องไม่น้อยกว่าความยาวหน้าต่าง
    # (1/2/4 ที่ kernel 3 ให้แค่ 15 ซึ่งไม่พอสำหรับหน้าต่าง 24) — FlareTCN ตรวจให้ตอนสร้าง
    dilations: tuple[int, ...] = (1, 2, 4, 8)


class TransformerArchConfig(_ArchModelBase):
    kind: Literal["transformer"]
    d_model: int
    ff_dim: int
    n_heads: int = 2
    n_layers: int = 1


class DARNNArchConfig(_ArchModelBase):
    kind: Literal["darnn"]
    hidden_size: int


ArchModelConfig = Annotated[
    LSTMArchConfig | TCNArchConfig | TransformerArchConfig | DARNNArchConfig,
    Field(discriminator="kind"),
]


class TrainOverride(_Strict):
    """ค่าการเทรนที่โมเดลหนึ่งตัวทับค่าร่วมได้ — ที่ไม่ระบุใช้ค่าร่วมของ ``forecast.yaml``

    จงใจไม่เปิดให้ทับ ``batch_size``/``scheduler``/``warmup_epochs``/``grad_clip``:
    ค่าพวกนี้ต้องเหมือนกันทุกโมเดล ไม่งั้นแยกไม่ออกว่าผลต่างมาจากสถาปัตยกรรม
    หรือมาจากตารางการเทรนที่ไม่เหมือนกัน
    """

    epochs: int | None = None
    lr: float | None = None
    weight_decay: float | None = None
    early_stop_patience: int | None = None


class LossOverride(_Strict):
    focal_alpha: float | None = None
    focal_gamma: float | None = None


class ForecastModelEntry(_Strict):
    """โมเดลหนึ่งตัว — ``model`` ต้องระบุครบ ส่วน ``train``/``loss`` ระบุเฉพาะค่าที่ต่างจากค่าร่วม"""

    model: ArchModelConfig
    train: TrainOverride = TrainOverride()
    loss: LossOverride = LossOverride()
    #: ชื่อที่แสดงในหน้าเว็บ/ตาราง/รูป — ไม่ระบุคือใช้ชื่อตาม kind จาก models.forecast.registry
    label: str | None = None


class ForecastConfig(_Strict):
    """config ของโมเดลหนึ่งตัวที่พร้อมส่งเข้าลูปเทรน — ผลของ :func:`resolve_architecture`

    ``kind`` กำกับว่าจะประกอบสถาปัตยกรรมไหน (ซ้ำกับ ``model.kind`` โดยตั้งใจ เพื่อให้
    ผู้เรียกไม่ต้องเจาะเข้าไปใน ``model``)
    """

    kind: str
    model: ArchModelConfig
    train: ForecastTrainConfig
    loss: ForecastLossConfig
    eval: ForecastEvalConfig


def resolve_architecture(base: Any, entry: ForecastModelEntry) -> ForecastConfig:
    """รวมค่าร่วม (``base.train``/``base.loss``/``base.eval``) เข้ากับส่วนที่โมเดลนี้ทับไว้

    ทำที่จุดเดียวเพื่อให้ค่าที่ไม่ได้ค้นหาเหมือนกันทุกโมเดล **โดยโครงสร้าง** ไม่ใช่โดยวินัย
    ของคนที่แก้ YAML — ใช้ทั้งกับ ``forecast.yaml`` (production) และ
    ``study/architectures.yaml`` (งานเปรียบเทียบ ซึ่งใช้ค่าร่วมของ ``forecast.yaml`` เป็นฐาน)
    """
    train_updates = {k: v for k, v in entry.train.model_dump().items() if v is not None}
    loss_updates = {k: v for k, v in entry.loss.model_dump().items() if v is not None}
    return ForecastConfig(
        kind=entry.model.kind,
        model=entry.model,
        train=base.train.model_copy(update=train_updates),
        loss=base.loss.model_copy(update=loss_updates),
        eval=base.eval,
    )


class ClassLevelSource(_Strict):
    """ที่มาของแบบจำลองระดับหนึ่ง (≥M1.0 หรือ ≥X1.0) ของคำพยากรณ์ระดับคลาส"""

    #: คลาส GOES ต่ำสุดที่ label ของระดับนี้นับเป็น positive — ต้องตรงกับ ``positive_goes_class`` ของ dataset
    positive_class: str
    #: โฟลเดอร์ใต้ ``artifacts/`` ที่ ``study/train.py`` เขียน ``models/<สถาปัตยกรรม>_<แบบ>_seed*.pt`` ไว้
    study_dir: str
    #: โฟลเดอร์ใต้ ``data/processed/`` ที่ ``study/build_dataset.py`` เขียนไว้ (อ่าน label ของระดับนี้)
    dataset: str


class ClassForecastConfig(_Strict):
    """โมเดลหลักของโปรเจค — เซลล์เดียว (สถาปัตยกรรม, แบบ) ที่แยกคำพยากรณ์เป็นระดับคลาส <M / M / X

    แต่ละระดับคือ ensemble ทุก seed ของเซลล์นั้นที่เทรนด้วย label ของระดับนั้น (ดู
    ``sunseg.inference.class_forecast``) · ระดับเรียงจากต่ำไปสูง
    """

    architecture: str = "lstm"
    variant: str = "V3"
    levels: dict[str, ClassLevelSource] = Field(
        default_factory=lambda: {
            "M": ClassLevelSource(positive_class="M1.0", study_dir="model_comparison", dataset="study_sequences"),
            "X": ClassLevelSource(positive_class="X1.0", study_dir="model_comparison_x", dataset="study_sequences_x"),
        }
    )

    @field_validator("levels")
    @classmethod
    def _m_then_x(cls, v: dict[str, ClassLevelSource]) -> dict[str, ClassLevelSource]:
        if list(v) != ["M", "X"]:
            raise ValueError(f"forecast.yaml: class_forecast.levels ต้องเป็น M แล้ว X ตามลำดับ (ได้ {list(v)})")
        return v


class ForecastModelsConfig(_Strict):
    """ทั้งไฟล์ ``forecast.yaml`` — ค่าร่วม + โมเดลทุกตัวที่ระบบรองรับ

    ชื่อโมเดล (key ของ ``models``) คือชื่อที่ใช้ทุกที่: ``--model`` ของสคริปต์เทรน,
    ไฟล์ ``artifacts/models/<ชื่อ>.pt`` และพารามิเตอร์ ``model=`` ของ API
    """

    default_model: str
    train: ForecastTrainConfig
    loss: ForecastLossConfig
    eval: ForecastEvalConfig
    models: dict[str, ForecastModelEntry]
    class_forecast: ClassForecastConfig = Field(default_factory=ClassForecastConfig)

    @field_validator("models")
    @classmethod
    def _non_empty(cls, v: dict[str, ForecastModelEntry]) -> dict[str, ForecastModelEntry]:
        if not v:
            raise ValueError("forecast.yaml: models ต้องมีอย่างน้อยหนึ่งตัว")
        return v

    @field_validator("default_model")
    @classmethod
    def _default_exists(cls, v: str, info: Any) -> str:
        models = info.data.get("models")
        if models is not None and v not in models:
            raise ValueError(f"forecast.yaml: default_model {v!r} ไม่อยู่ใน models ({sorted(models)})")
        return v

    @property
    def names(self) -> list[str]:
        return list(self.models)

    def resolve(self, name: str) -> ForecastConfig:
        """config พร้อมเทรนของโมเดลชื่อ ``name``"""
        if name not in self.models:
            raise ValueError(f"ไม่มีโมเดล {name!r} ใน forecast.yaml (ที่มี: {self.names})")
        return resolve_architecture(self, self.models[name])


# --------------------------------------------------------------------------- #
# study/architectures.yaml — แกน "สถาปัตยกรรม" ของงานเปรียบเทียบ
# --------------------------------------------------------------------------- #


class ArchitectureEntry(ForecastModelEntry):
    """หนึ่งแถวของตารางเปรียบเทียบ — โมเดลหนึ่งตัวพร้อมรายการแบบ (ชุด feature) ที่ต้องรัน"""

    #: แบบที่สถาปัตยกรรมนี้ต้องรัน — ไม่ระบุคือใช้ ``default_variants``
    #: แบบที่เกินจาก ``default_variants`` เป็นแถวเสริม ไม่เข้าตารางหลัก
    variants: list[str] | None = None
    #: True = อยู่ใต้ตาราง ไม่ใช่แถวของตารางหลัก (เช่น LSTM ชุด hyperparameter ของ production)
    supplementary: bool = False


class StudyArchitecturesConfig(_Strict):
    default_variants: list[str]
    architectures: dict[str, ArchitectureEntry]


# --------------------------------------------------------------------------- #
# tracking.yaml
# --------------------------------------------------------------------------- #


class DetectConfig(_Strict):
    min_area_px: int
    morph_close_px: int
    morph_open_px: int


class RotationConfig(_Strict):
    """สัมประสิทธิ์ differential rotation (Snodgrass 1983), หน่วย deg/day"""

    snodgrass_A: float = 14.713
    snodgrass_B: float = -2.396
    snodgrass_C: float = -1.787


class TrackerConfig(_Strict):
    w_dist: float
    w_area: float
    w_iou: float
    max_cost: float
    max_match_dist_deg: float
    n_confirm: int
    max_age_frames: int


class TrackingConfig(_Strict):
    detect: DetectConfig
    rotation: RotationConfig
    tracker: TrackerConfig


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ config: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} ต้องเป็น YAML mapping ที่ระดับบนสุด")
    return data


def load_dotenv(path: Path | None = None) -> None:
    """โหลด .env เข้า os.environ แบบง่าย (ไม่ทับค่าที่ตั้งไว้แล้วใน environment จริง)"""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@lru_cache(maxsize=1)
def load_data_config(path: Path | None = None) -> DataConfig:
    load_dotenv()
    cfg = DataConfig(**_read_yaml(path or CONFIG_DIR / "data.yaml"))
    # path ใน YAML เขียนแบบ relative เพื่อให้อ่านง่าย แต่โค้ดใช้ absolute เสมอ
    cfg.paths = cfg.paths.resolve(PROJECT_ROOT)
    cfg.proton = cfg.proton.resolve(PROJECT_ROOT)
    cfg.position_flare = cfg.position_flare.resolve(PROJECT_ROOT)
    cfg.aia = cfg.aia.resolve(PROJECT_ROOT)
    return cfg


@lru_cache(maxsize=1)
def load_unet_config(path: Path | None = None) -> UNetConfig:
    return UNetConfig(**_read_yaml(path or CONFIG_DIR / "unet.yaml"))


@lru_cache(maxsize=1)
def load_forecast_config(path: Path | None = None) -> ForecastModelsConfig:
    """ทั้งไฟล์ ``forecast.yaml`` — ใช้ ``.resolve(ชื่อ)`` เพื่อได้ config พร้อมเทรนของโมเดลหนึ่งตัว"""
    return ForecastModelsConfig(**_read_yaml(path or FORECAST_CONFIG_FILE))


@lru_cache(maxsize=1)
def load_study_architectures(path: Path | None = None) -> StudyArchitecturesConfig:
    return StudyArchitecturesConfig(**_read_yaml(path or STUDY_ARCHITECTURES_FILE))


@lru_cache(maxsize=1)
def load_tracking_config(path: Path | None = None) -> TrackingConfig:
    return TrackingConfig(**_read_yaml(path or CONFIG_DIR / "tracking.yaml"))
