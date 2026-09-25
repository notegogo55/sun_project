"""คำพยากรณ์ระดับคลาสของโมเดลหลัก (LSTM + V3) — รวมแบบจำลอง ≥M1.0 กับ ≥X1.0 เป็น <M / M / X

โมเดลหลักตอบได้ทีละคำถามแบบทวิภาค ("จะเกิด ≥M1.0 ใน 24 ชม. ไหม", "จะเกิด ≥X1.0 ไหม") แต่ละระดับจึงเป็น
ensemble แยกกันของทุก seed ที่ ``study/train.py`` เทรนด้วย label ของระดับนั้น (``configs/forecast.yaml``
หัวข้อ ``class_forecast``) โมดูลนี้รวมสองคำตอบเป็นระดับเดียว:

1. **ในแต่ละระดับ** — seed หนึ่งเตือนเมื่อความน่าจะเป็น ≥ threshold ของ seed นั้นเอง (เลือกบน val แล้ว freeze)
   ensemble เตือนเมื่อ **seed เกินครึ่ง** เตือน (เสมอกันถือว่าไม่เตือน) — กติกาเดียวกับ
   ``scripts/forecast/plot_storm_forecast.py`` · ความน่าจะเป็นที่แสดงคือค่าเฉลี่ยข้าม seed
2. **ข้ามระดับ** — ระดับที่ทำนาย = ระดับสูงสุดที่ ensemble เตือน: X ถ้าระดับ X เตือน, M ถ้าเตือนแค่ระดับ M,
   ไม่งั้น <M · ระดับ X เตือนแต่ระดับ M ไม่เตือนนับเป็น X (≥X1.0 ย่อมเป็น ≥M1.0 ด้วย) และถูกนับแยกไว้ใน
   ``n_inconsistent`` เพราะเป็นสัญญาณว่าสองแบบจำลองขัดกัน

**จุดทำงาน (mode) สองจุด** — ต่างกันแค่ว่าระดับ X เตือนเมื่อไร (ระดับ M เหมือนกันทั้งคู่):

- ``sensitive`` (เตือนไว, ปริยาย) — ระดับ X ใช้เสียงข้างมากของ seed ตามข้อ 1 ซึ่ง threshold ราย seed ถูกเลือกให้
  **TSS** ของ ≥X1.0 สูงสุด · เมื่อ X หายากมาก TSS ดัน threshold ต่ำเพื่อเก็บ recall ผลคือจับ X ได้เกือบหมด แต่
  flare ระดับ M ส่วนใหญ่ถูกเรียกว่า X ไปด้วย
- ``strict`` (ระมัดระวัง) — ระดับ X เตือนเมื่อ **ความน่าจะเป็นเฉลี่ยข้าม seed** ≥ threshold เดียวที่เลือกบน val ให้
  **HSS แบบหลายคลาส** (:func:`multiclass_hss`) ของคำพยากรณ์ 3 ระดับสูงสุด (:func:`select_strict_threshold`) ·
  ทายระดับถูกบ่อยกว่ามาก แต่แทบไม่เตือน X

ไม่มีจุดไหน "ถูก" — X ใน test มีแค่ราวสิบกว่า sample จากไม่กี่ HARP และแบบจำลองระดับ X แยก X ออกจาก M ได้ไม่ดี
หน้าเว็บจึงให้สลับดูได้ทั้งสองจุด

ระดับจริงของ sample มาจาก label สองชุดเดียวกับที่เทรน: X ถ้า label ≥X1.0 เป็น 1, M ถ้าแค่ label ≥M1.0 เป็น 1

**<M ไม่ได้แปลว่า "จะเกิด flare C"** — แปลว่าไม่มีระดับไหนเตือน (อาจไม่มี flare เลย หรือมีแค่ B/C)
ระบบไม่มีแบบจำลอง ≥C1.0 จึงแยก C ออกจากเงียบไม่ได้

- :func:`majority_alarm`, :func:`combine_levels`, :func:`true_levels`, :func:`level_evaluation` — ฟังก์ชันล้วน
  ที่ทั้ง ``scripts/study/class_forecast.py`` (รายงาน) และ API ใช้ร่วมกัน ตัวเลขสองฝั่งจึงตรงกันโดยโครงสร้าง
- :class:`LevelEnsemble` — checkpoint ทุก seed ของระดับหนึ่ง
- :class:`ClassForecastService` — ensemble ทั้งสองระดับรันบน study dataset ทุกแถว **ครั้งเดียวตอน startup**
  (LSTM เล็กมาก 50 ตัว × ~25,000 sequence ใช้เวลาไม่กี่วินาทีบน CPU) แล้วทุก request เป็นแค่การค้นตาราง
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import ClassForecastConfig
from ..data.build_sequences import apply_normalisation
from ..data.study_dataset import select_columns
from ..metrics import confusion_counts, hss2, tss
from ..models.forecast import architecture_label, build_architecture
from .forecast import model_config_from_checkpoint

logger = logging.getLogger(__name__)

#: ระดับจากต่ำไปสูง — ดัชนีในทูเพิลนี้คือค่าจำนวนเต็มที่ทุกฟังก์ชันใช้แทนระดับ
LEVELS: tuple[str, ...] = ("<M", "M", "X")
#: ระดับขั้นต่ำที่แบบจำลองแต่ละตัวถาม (ดัชนีใน :data:`LEVELS`)
LEVEL_INDEX = {"M": 1, "X": 2}
#: จุดทำงาน — ตัวแรกคือค่าปริยาย
MODES: tuple[str, ...] = ("sensitive", "strict")

_SEED_PATTERN = re.compile(r"_seed(\d+)\.pt$")


# --------------------------------------------------------------------------- #
# ฟังก์ชันล้วน
# --------------------------------------------------------------------------- #


def majority_alarm(probs: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """รวม seed ของระดับหนึ่ง — ``probs`` รูป ``(S, N)``, ``thresholds`` รูป ``(S,)``

    คืน ``(ความน่าจะเป็นเฉลี่ย (N,), จำนวน seed ที่เตือน (N,), เตือนหรือไม่ (N,))`` — เตือนเมื่อ seed เกินครึ่ง
    """
    probs = np.asarray(probs, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if probs.ndim != 2 or thresholds.shape != (probs.shape[0],):
        raise ValueError(f"คาดหวัง probs (S, N) กับ thresholds (S,) แต่ได้ {probs.shape} กับ {thresholds.shape}")
    n_alarm = (probs >= thresholds[:, None]).sum(axis=0)
    return probs.mean(axis=0), n_alarm, n_alarm * 2 > probs.shape[0]


def combine_levels(alarm_m: np.ndarray, alarm_x: np.ndarray) -> np.ndarray:
    """ระดับที่ทำนาย (ดัชนีใน :data:`LEVELS`) = ระดับสูงสุดที่เตือน"""
    alarm_m = np.asarray(alarm_m, dtype=bool)
    alarm_x = np.asarray(alarm_x, dtype=bool)
    return np.where(alarm_x, 2, np.where(alarm_m, 1, 0))


def true_levels(label_m: np.ndarray, label_x: np.ndarray) -> np.ndarray:
    """ระดับจริง (ดัชนีใน :data:`LEVELS`) จาก label สองชุด — label ≥X1.0 ที่ไม่เป็น ≥M1.0 คือ dataset ขัดกันเอง"""
    label_m = np.asarray(label_m, dtype=bool)
    label_x = np.asarray(label_x, dtype=bool)
    broken = label_x & ~label_m
    if broken.any():
        raise ValueError(
            f"label ≥X1.0 เป็น 1 แต่ ≥M1.0 เป็น 0 ใน {int(broken.sum())} แถว — dataset สองระดับสร้างจากแคตตาล็อกคนละชุด?"
        )
    return np.where(label_x, 2, np.where(label_m, 1, 0))


def level_confusion(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """ตาราง 3×3 — แถว = ระดับจริง, คอลัมน์ = ระดับที่ทำนาย (ลำดับตาม :data:`LEVELS`)"""
    n = len(LEVELS)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (np.asarray(true, dtype=int), np.asarray(pred, dtype=int)), 1)
    return matrix


def multiclass_hss(confusion: np.ndarray) -> float:
    """Heidke skill score แบบหลายคลาส = (สัดส่วนถูก − สัดส่วนถูกโดยบังเอิญ) / (1 − สัดส่วนถูกโดยบังเอิญ)

    "ถูกโดยบังเอิญ" คือ Σ p(จริง = i)·p(ทำนาย = i) · 0 = ไม่ต่างจากการเดาตามสัดส่วน, 1 = ถูกทุกตัว
    ตารางที่ทุกแถวอยู่ในระดับเดียวกัน (ตัวหารเป็นศูนย์) คืน 0
    """
    cm = np.asarray(confusion, dtype=float)
    n = cm.sum()
    if n == 0:
        return 0.0
    correct = np.trace(cm) / n
    chance = float((cm.sum(axis=1) * cm.sum(axis=0)).sum() / n**2)
    return 0.0 if chance >= 1.0 else float((correct - chance) / (1.0 - chance))


def select_strict_threshold(prob_x: np.ndarray, alarm_m: np.ndarray, true: np.ndarray) -> float:
    """threshold ของความน่าจะเป็นเฉลี่ยระดับ X ที่ทำให้ :func:`multiclass_hss` ของ 3 ระดับสูงสุด — ใช้กับ val เท่านั้น

    ผู้สมัครคือทุกค่าที่ไม่ซ้ำของ ``prob_x`` (เตือนเมื่อ ≥ ค่านั้น) · ค่าเท่ากันเลือกตัวต่ำสุด (เตือนไวกว่า)
    """
    prob_x = np.asarray(prob_x, dtype=float)
    alarm_m = np.asarray(alarm_m, dtype=bool)
    true = np.asarray(true, dtype=int)
    if prob_x.size == 0:
        raise ValueError("ไม่มี sample ให้เลือก threshold")
    best_score, best_threshold = -np.inf, float(prob_x.max())
    for candidate in np.unique(prob_x):
        score = multiclass_hss(level_confusion(true, combine_levels(alarm_m, prob_x >= candidate)))
        if score > best_score:
            best_score, best_threshold = score, float(candidate)
    return best_threshold


def level_evaluation(true: np.ndarray, pred: np.ndarray) -> dict:
    """ตัวชี้วัดของคำพยากรณ์ระดับคลาส

    - ``confusion`` — ตาราง 3×3 (แถวจริง × คอลัมน์ทำนาย)
    - ``thresholds`` — ต่อระดับ M และ X: ตัดคำพยากรณ์ที่ "≥ ระดับนั้น" เป็นทวิภาคแล้ววัดแบบเดียวกับแบบจำลองเดี่ยว
      (``≥M`` นับทั้งช่องที่ทำนาย M และ X ว่าเตือน) — ตัวเลข ≥X ต่างจาก ensemble ระดับ X เดี่ยว ๆ ไม่ได้เลย
      แต่ ≥M ต่างได้ในแถวที่ระดับ X เตือนแต่ระดับ M ไม่เตือน
    - ``exact_on_events`` — ในบรรดา sample ที่เกิด ≥M1.0 จริง สัดส่วนที่ทายระดับถูกเป๊ะ (M เป็น M, X เป็น X)
    """
    true = np.asarray(true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if true.shape != pred.shape:
        raise ValueError(f"ขนาดไม่ตรงกัน: true {true.shape} vs pred {pred.shape}")

    thresholds = {}
    for name, index in LEVEL_INDEX.items():
        y_true, y_pred = true >= index, pred >= index
        c = confusion_counts(y_true, y_pred)
        thresholds[name] = {
            "n_positive": c.tp + c.fn,
            "tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn,
            "tss": tss(y_true, y_pred),
            "hss2": hss2(y_true, y_pred),
            "precision": c.tp / (c.tp + c.fp) if (c.tp + c.fp) else None,
            "recall": c.tp / (c.tp + c.fn) if (c.tp + c.fn) else None,
        }

    events = true >= 1
    confusion = level_confusion(true, pred)
    return {
        "levels": list(LEVELS),
        "n": int(len(true)),
        "confusion": confusion.tolist(),
        "hss_multiclass": multiclass_hss(confusion),
        "thresholds": thresholds,
        "exact_on_events": float((pred[events] == true[events]).mean()) if events.any() else None,
    }


# --------------------------------------------------------------------------- #
# ensemble ของระดับหนึ่ง
# --------------------------------------------------------------------------- #


@dataclass
class _Member:
    seed: int
    model: torch.nn.Module
    features: list[str]
    stats: dict[str, np.ndarray]
    threshold: float


class LevelEnsemble:
    """checkpoint ทุก seed ของเซลล์ (สถาปัตยกรรม, แบบ) ในโฟลเดอร์ ``models/`` ของงานเปรียบเทียบหนึ่งชุด

    อ่านเฉพาะไฟล์ ``<สถาปัตยกรรม>_<แบบ>_seed<n>.pt`` (ไม่รวมรันของ cross-validation ที่มี ``_fold`` ในชื่อ)
    แต่ละ seed ใช้ชื่อ feature, normalisation stats และ threshold ของตัวเองที่เก็บใน checkpoint
    """

    def __init__(self, models_dir: Path, architecture: str, variant: str, device: str = "cpu") -> None:
        self.models_dir = Path(models_dir)
        self.device = torch.device(device)
        self.members: list[_Member] = []
        prefix = f"{architecture}_{variant}_seed"
        paths = sorted(
            (p for p in self.models_dir.glob(f"{prefix}*.pt") if _SEED_PATTERN.search(p.name)),
            key=lambda p: int(_SEED_PATTERN.search(p.name).group(1)),
        )
        for path in paths:
            self.members.append(self._load(path))

    def _load(self, path: Path) -> _Member:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        config = payload["config"]
        kind, model_cfg = model_config_from_checkpoint(config)
        model = build_architecture(kind, int(config["n_features"]), int(config.get("seq_len") or 0), model_cfg)
        model.load_state_dict(payload["state_dict"])
        model.to(self.device).eval()
        return _Member(
            seed=int(config["seed"]),
            model=model,
            features=list(config["features"]),
            stats={
                "mean": np.asarray(payload["norm_mean"], dtype=np.float32),
                "std": np.asarray(payload["norm_std"], dtype=np.float32),
            },
            threshold=float(config["threshold"]),
        )

    @property
    def available(self) -> bool:
        return bool(self.members)

    @property
    def seeds(self) -> list[int]:
        return [m.seed for m in self.members]

    @property
    def thresholds(self) -> np.ndarray:
        return np.array([m.threshold for m in self.members])

    @torch.no_grad()
    def predict(self, x_raw: np.ndarray, features: list[str], batch_size: int = 8192) -> np.ndarray:
        """ความน่าจะเป็นของทุก seed รูป ``(S, N)`` — ``x_raw`` คือค่าดิบ ``(N, L, F)`` ที่มีคอลัมน์ชื่อ ``features``"""
        if not self.members:
            raise RuntimeError(f"ไม่มี checkpoint ใน {self.models_dir}")
        out = np.empty((len(self.members), len(x_raw)), dtype=np.float32)
        for i, member in enumerate(self.members):
            idx = select_columns(features, member.features)
            for start in range(0, len(x_raw), batch_size):
                chunk = apply_normalisation(x_raw[start:start + batch_size][:, :, idx], member.stats)
                logits = member.model(torch.from_numpy(chunk).to(self.device))
                out[i, start:start + len(chunk)] = torch.sigmoid(logits).cpu().numpy()
        return out


# --------------------------------------------------------------------------- #
# service ของหน้าเว็บ
# --------------------------------------------------------------------------- #


class ClassForecastService:
    """คำพยากรณ์ระดับคลาสของทุก sample ใน study dataset — คำนวณครั้งเดียวตอนสร้าง

    ``table`` มีหนึ่งแถวต่อ sample: ``HARPNUM, noaa_ar, issue_time, split, lat, lon``,
    ``prob_M/prob_X`` (เฉลี่ยข้าม seed), ``n_alarm_M/n_alarm_X``, ``true_level`` และ ``level_<mode>`` /
    ``inconsistent_<mode>`` ต่อจุดทำงานใน :data:`MODES` (ระดับเป็นดัชนีใน :data:`LEVELS`)
    — ขาดอะไรก็ตาม (dataset, checkpoint) บริการปิดตัวเอง (``available=False``)
    พร้อม ``hint`` บอกคำสั่งที่ต้องรัน แทนที่จะทำให้แอปล่ม
    """

    def __init__(self, config: ClassForecastConfig, artifacts_root: Path, processed_root: Path,
                 device: str = "cpu") -> None:
        self.config = config
        self.architecture = config.architecture
        self.variant = config.variant
        self.label = f"{architecture_label(config.architecture)} + {config.variant}"
        self.available = False
        self.hint: str | None = None
        self.table: pd.DataFrame | None = None
        self.n_seeds: dict[str, int] = {}
        self.cadence_hours: int | None = None
        self.sequence_length: int | None = None
        self.features: list[str] = []
        #: threshold ของความน่าจะเป็นเฉลี่ยระดับ X ของจุดทำงาน ``strict`` (เลือกบน val)
        self.strict_threshold: float | None = None
        try:
            self._build(Path(artifacts_root), Path(processed_root), device)
        except Exception as exc:  # noqa: BLE001 — ข้อมูลของโมเดลหลักเสียไม่ควรทำให้ทั้งแอปล่ม
            logger.error("เตรียมคำพยากรณ์ระดับคลาสของ %s ไม่สำเร็จ: %s", self.label, exc)
            self.hint = self.hint or str(exc)

    def _build(self, artifacts_root: Path, processed_root: Path, device: str) -> None:
        from ..datasets.study import REQUIRED_FILES

        sources = self.config.levels
        for name, source in sources.items():
            data_dir = processed_root / source.dataset
            if any(not (data_dir / f).exists() for f in REQUIRED_FILES):
                self.hint = (
                    f"python backend/scripts/study/build_dataset.py --positive-class {source.positive_class}"
                    + ("" if name == "M" else f" --out-dir data/processed/{source.dataset}")
                )
                logger.warning("ไม่พบ dataset ระดับ %s ที่ %s — ปิดคำพยากรณ์ระดับคลาส", name, data_dir)
                return

        ensembles = {
            name: LevelEnsemble(artifacts_root / source.study_dir / "models", self.architecture, self.variant, device)
            for name, source in sources.items()
        }
        for name, ensemble in ensembles.items():
            if not ensemble.available:
                self.hint = (
                    f"python backend/scripts/study/train.py --data-dir data/processed/{sources[name].dataset} "
                    f"--out-dir artifacts/{sources[name].study_dir}"
                )
                logger.warning("ไม่พบ checkpoint ของ %s ระดับ %s — ปิดคำพยากรณ์ระดับคลาส", self.label, name)
                return

        base = _load_level_dataset(processed_root / sources["M"].dataset)
        other = _load_level_dataset(processed_root / sources["X"].dataset, x=False)
        keys = ["HARPNUM", "issue_time"]
        if not base.meta[keys].equals(other.meta[keys]):
            raise ValueError(
                f"แถวของ {sources['M'].dataset} กับ {sources['X'].dataset} ไม่ตรงกัน — สร้าง dataset สองระดับใหม่จาก "
                "SHARP/แคตตาล็อกชุดเดียวกัน"
            )

        self.cadence_hours, self.sequence_length = base.cadence_hours, base.x.shape[1]
        self.features = ensembles["M"].members[0].features
        prob, n_alarm, alarm = {}, {}, {}
        for name, ensemble in ensembles.items():
            probs = ensemble.predict(base.x, base.features)
            prob[name], n_alarm[name], alarm[name] = majority_alarm(probs, ensemble.thresholds)
            self.n_seeds[name] = len(ensemble.members)

        true = true_levels(base.meta["label"].to_numpy(), other.meta["label"].to_numpy())
        val = (base.meta["split"] == "val").to_numpy()
        self.strict_threshold = select_strict_threshold(prob["X"][val], alarm["M"][val], true[val])
        alarm_x = {"sensitive": alarm["X"], "strict": prob["X"] >= self.strict_threshold}

        table = base.meta[[c for c in ("HARPNUM", "noaa_ar", "issue_time", "split", "lat", "lon") if c in base.meta]]
        table = table.assign(
            prob_M=prob["M"], n_alarm_M=n_alarm["M"], prob_X=prob["X"], n_alarm_X=n_alarm["X"], true_level=true,
            **{f"level_{mode}": combine_levels(alarm["M"], alarm_x[mode]) for mode in MODES},
            **{f"inconsistent_{mode}": alarm_x[mode] & ~alarm["M"] for mode in MODES},
        )
        table["issue_time"] = pd.to_datetime(table["issue_time"])
        self.table = table.sort_values(["HARPNUM", "issue_time"]).reset_index(drop=True)
        self.available = True
        for mode in MODES:
            ev = self.evaluation("test", mode)
            logger.info(
                "คำพยากรณ์ระดับคลาสของ %s (%s) บน test: TSS ≥M %.3f ≥X %.3f · ทายระดับถูก %.0f%% ของ event",
                self.label, mode, ev["thresholds"]["M"]["tss"], ev["thresholds"]["X"]["tss"],
                100 * (ev["exact_on_events"] or 0),
            )

    # ------------------------------------------------------------------ #

    def _require(self) -> pd.DataFrame:
        if not self.available or self.table is None:
            raise RuntimeError(
                f"ยังไม่มีคำพยากรณ์ระดับคลาสของ {self.label}" + (f" — รัน `{self.hint}` ก่อน" if self.hint else "")
            )
        return self.table

    @staticmethod
    def _check_mode(mode: str) -> str:
        if mode not in MODES:
            raise KeyError(f"ไม่รู้จักจุดทำงาน {mode!r} (ที่มี: {', '.join(MODES)})")
        return mode

    def series(self, harpnum: int) -> pd.DataFrame:
        """ทุก issue_time ของ HARP หนึ่ง เรียงตามเวลา (ว่างได้)"""
        table = self._require()
        return table[table["HARPNUM"] == harpnum]

    def at(self, time: pd.Timestamp, tolerance: pd.Timedelta, mode: str = MODES[0]) -> pd.DataFrame:
        """issue_time ที่ใกล้ ``time`` ที่สุดของแต่ละ HARP ภายใน ± ``tolerance`` เรียงระดับ (ของ ``mode``) สูงไปต่ำ"""
        table = self._require()
        level = f"level_{self._check_mode(mode)}"
        gap = (table["issue_time"] - time).abs()
        near = table[gap <= tolerance].assign(_gap=gap[gap <= tolerance])
        nearest = near.sort_values("_gap").drop_duplicates("HARPNUM")
        return nearest.sort_values([level, "prob_X", "prob_M"], ascending=False).drop(columns="_gap")

    def evaluation(self, split: str, mode: str = MODES[0]) -> dict:
        table = self._require()
        self._check_mode(mode)
        rows = table[table["split"] == split]
        return {
            **level_evaluation(rows["true_level"], rows[f"level_{mode}"]),
            "n_inconsistent": int(rows[f"inconsistent_{mode}"].sum()),
        }

    def info(self) -> dict:
        return {
            "label": self.label,
            "architecture": self.architecture,
            "variant": self.variant,
            "available": self.available,
            "hint": None if self.available else self.hint,
            "levels": list(LEVELS),
            "modes": list(MODES),
            "n_seeds": self.n_seeds,
            "strict_threshold": self.strict_threshold,
            "cadence_hours": self.cadence_hours,
            "sequence_length": self.sequence_length,
            "features": self.features,
            "positive_classes": {name: s.positive_class for name, s in self.config.levels.items()},
        }


@dataclass
class _LevelDataset:
    x: np.ndarray | None
    meta: pd.DataFrame
    features: list[str]
    cadence_hours: int | None


def _load_level_dataset(data_dir: Path, x: bool = True) -> _LevelDataset:
    """meta (+ X ดิบ ถ้าขอ) ของ study dataset หนึ่งชุด — label ของ meta ต้องตรงกับ y.npy"""
    import json

    meta = pd.read_parquet(data_dir / "meta.parquet")
    y = np.load(data_dir / "y.npy")
    if not np.array_equal(meta["label"].to_numpy(), y):
        raise ValueError(f"{data_dir}: meta['label'] ไม่ตรงกับ y.npy")
    npz = np.load(data_dir / "norm_stats.npz", allow_pickle=True)
    report_path = data_dir / "report.json"
    cadence = json.loads(report_path.read_text(encoding="utf-8")).get("cadence_hours") if report_path.exists() else None
    return _LevelDataset(
        x=np.load(data_dir / "X.npy") if x else None,
        meta=meta,
        features=[str(f) for f in npz["features"]],
        cadence_hours=cadence,
    )
