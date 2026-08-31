"""โหลดและ validate ไฟล์ config (YAML + .env) เป็น pydantic model.

ทุกโมดูลในโปรเจคควรเรียก ``load_data_config()`` / ``load_model_config()`` แทนการ
อ่าน YAML เองโดยตรง เพื่อให้ path ถูก resolve เป็น absolute เหมือนกันทั้งระบบ และ
เพื่อให้ typo ใน config ระเบิดตั้งแต่ตอนโหลด ไม่ใช่กลางการเทรน
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

# backend/src/sunseg/config.py -> src/sunseg -> src -> backend
BACKEND_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = BACKEND_ROOT / "configs"

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


class CaseStudyConfig(TimeRange):
    """หน้าต่างประเมินแยกจากชุดเทรน (ดูเหตุผลใน configs/data.yaml)

    มี cadence เป็นของตัวเองเพราะ case study ต้องการความละเอียดตามเวลาสูงกว่า
    ชุดเทรนมาก — คนละเป้าหมายกัน จึงใช้ค่าเดียวกันไม่ได้
    """

    cadence_hours: int


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


class SplitConfig(_Strict):
    train_end: date
    val_end: date
    gap_days: int


class DataConfig(_Strict):
    paths: Paths
    jsoc: JsocConfig
    time_range: TimeRange
    case_study: CaseStudyConfig
    sharp: SharpConfig
    fulldisk: FullDiskConfig
    flare: FlareConfig
    proton: ProtonConfig
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
# lstm.yaml
# --------------------------------------------------------------------------- #


class LSTMModelConfig(_Strict):
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool = False
    use_attention: bool = True


class LSTMTrainConfig(_Strict):
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    grad_clip: float = 1.0
    early_stop_patience: int
    scheduler: Literal["cosine", "plateau", "none"] = "cosine"
    warmup_epochs: int = 0
    seed: int = 42


class LSTMLossConfig(_Strict):
    type: Literal["focal", "bce"] = "focal"
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0


class LSTMEvalConfig(_Strict):
    select_threshold_on: Literal["val", "train"] = "val"
    primary_metric: Literal["tss", "hss2", "auc"] = "tss"


class LSTMConfig(_Strict):
    model: LSTMModelConfig
    train: LSTMTrainConfig
    loss: LSTMLossConfig
    eval: LSTMEvalConfig


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
    cfg.aia = cfg.aia.resolve(PROJECT_ROOT)
    return cfg


@lru_cache(maxsize=1)
def load_unet_config(path: Path | None = None) -> UNetConfig:
    return UNetConfig(**_read_yaml(path or CONFIG_DIR / "unet.yaml"))


@lru_cache(maxsize=1)
def load_lstm_config(path: Path | None = None) -> LSTMConfig:
    return LSTMConfig(**_read_yaml(path or CONFIG_DIR / "lstm.yaml"))


@lru_cache(maxsize=1)
def load_tracking_config(path: Path | None = None) -> TrackingConfig:
    return TrackingConfig(**_read_yaml(path or CONFIG_DIR / "tracking.yaml"))
