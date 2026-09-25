"""เทรนตาราง (สถาปัตยกรรม × แบบ) × หลาย seed ของงานเปรียบเทียบโมเดล แล้วสรุปผล

ตารางหลักคือสถาปัตยกรรมที่ประกาศใน ``configs/study/architectures.yaml`` คูณกับ
``default_variants`` ของไฟล์นั้น (ปริยาย 4 × 4 = 16 เซลล์) ผลหลักของ ``report.md`` คืออันดับของ
ทุกเซลล์ตาม TSS เฉลี่ยบน test (``sunseg.study.summary.rank_cells``) ส่วนแบบที่เกินมาและรายการที่
ตั้ง ``supplementary: true`` เป็นแถวเสริมใต้ตาราง ไม่ถูกจัดอันดับและไม่เข้าการคำนวณ Δ ของตารางหลัก

    # ต้องมี data/processed/study_sequences/ ก่อน (backend/scripts/study/build_dataset.py)
    python backend/scripts/study/train.py

    # ทดสอบเร็ว — บางแบบ บาง seed เขียนลงที่อื่น
    python backend/scripts/study/train.py --variants V0,V5 --seeds 0 --epochs 3 --out-dir <tmp>

    # รวมผลที่เทรนไว้แล้วเป็นตาราง/รายงานใหม่ โดยไม่เทรนเพิ่ม
    python backend/scripts/study/train.py --summarise-only

กติกาที่ spec บังคับ และสคริปต์นี้รักษาไว้โดยโครงสร้าง:

- **ทุกแบบใช้แถวชุดเดียวกัน** — โหลด ``X`` ก้อนเดียวแล้วหั่นคอลัมน์ตามชื่อ
  (``sunseg.datasets.study.variant_splits``) ไม่มีแบบไหนได้แถวต่างจากแบบอื่น
- **hyperparameter ชุดเดียวต่อสถาปัตยกรรม ใช้เหมือนกันทุกแบบ** — ค่ามาจาก
  ``configs/study/architectures.yaml`` ซึ่งตกทอดสิ่งที่ไม่ได้ระบุมาจากค่าร่วมของ ``configs/forecast.yaml``
  ไม่มีช่องให้ส่งค่ารายแบบ และใช้ลูปเทรนตัวเดียวกับ production (``sunseg.training.forecast``)
  (ค่าเหล่านี้มาจากการค้นหาที่งบเท่ากันทุกสถาปัตยกรรม — ``docs/adr/0001-tune-per-architecture.md``)
- **threshold ของใครของมัน** — เลือกบน val ของแต่ละ (สถาปัตยกรรม, แบบ, seed) แล้ว freeze ก่อนแตะ test
- **seed ชุดเดียวกันทุกเซลล์** — สรุปแบบจับคู่ seed (``sunseg.study.summary``) ซึ่ง error ถ้าคู่ไม่ครบ
- **ลูปเป็น seed-major** — วน seed ชั้นนอกสุด จึงมีตารางครบทุกเซลล์ตั้งแต่ seed แรก หยุดกลางทาง
  แล้วสรุปได้ทันที และความเร็วเครื่องที่แกว่งตกกับทุกเซลล์เท่า ๆ กัน

รันต่อได้: แต่ละ (แบบ, seed) เขียนผลของตัวเองลง ``runs/`` ทันทีที่เสร็จ รอบถัดไปข้ามตัวที่มีแล้ว
และยังตรงกับ config + dataset ปัจจุบัน (``signature`` ในไฟล์) — ผลที่เทรนด้วย config หรือ dataset
อื่น (เช่นรอบทดสอบ ``--epochs 3``) ไม่ถูกนำมารวมเงียบ ๆ ใช้ ``--force`` เพื่อเทรนใหม่

artifacts ทั้งหมดอยู่ใต้ ``artifacts/model_comparison/`` — ไม่แตะ checkpoint/ค่าทำนายของ
โมเดล production (``artifacts/models/``, ``artifacts/metrics/``)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sunseg.config import (  # noqa: E402
    STUDY_ARCHITECTURES_FILE,
    STUDY_VARIANTS_FILE,
    load_data_config,
    load_forecast_config,
    load_study_architectures,
    resolve_architecture,
)
from sunseg.data.cv import FoldPlan, plan_folds  # noqa: E402
from sunseg.data.study_dataset import load_variants, select_columns  # noqa: E402
from sunseg.datasets.sequence import SequenceSplits  # noqa: E402
from sunseg.datasets.study import (  # noqa: E402
    StudyArrays,
    fold_variant_splits,
    load_study_arrays,
    variant_splits,
)
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.metrics import classification_report, confusion_counts  # noqa: E402
from sunseg.models.forecast import architecture_label  # noqa: E402
from sunseg.study.summary import (  # noqa: E402
    delta_by_architecture,
    delta_by_variant,
    end_to_end_hours,
    feature_groups,
    interaction_summary,
    rank_cells,
    seed_table,
    timing_summary,
)
from sunseg.training.forecast import fit_one  # noqa: E402 — ลูปเดียวกับ production โดยตั้งใจ
from sunseg.training.utils import get_device, save_checkpoint  # noqa: E402

logger = logging.getLogger("train_study")

#: anchor ของตาราง = เซลล์ (สถาปัตยกรรมอ้างอิง, แบบอ้างอิง) — ดู CONTEXT.md หัวข้อ anchor
ANCHOR_ARCHITECTURE = "lstm"
ANCHOR_VARIANT = "V0"
DEFAULT_SEEDS = "0,1,2,3,4"
STUDY_DIR_NAME = "model_comparison"
DEFAULT_VARIANTS_FILE = STUDY_VARIANTS_FILE
DEFAULT_ARCHITECTURES_FILE = STUDY_ARCHITECTURES_FILE

#: ขอบเขตการวัดผล: (ชื่อ, split)
SCOPES: tuple[tuple[str, str], ...] = (
    ("val", "val"),
    ("test", "test"),
)
SCOPE_TITLES = {
    # ไม่ฝังปีไว้ในชื่อ: ช่วงของ test ขึ้นกับ split.train_end/val_end ใน data.yaml ซึ่งย้ายได้
    # ป้ายที่ฝัง "2011-2017" ไว้เคยทำให้ตารางบอกช่วงผิด (test จริงคือ 2015-2017) — ช่วงจริง
    # ถูกพิมพ์อยู่แล้วในตาราง "ข้อมูล" ของรายงาน
    "test": "test",
    "val": "validation (ใช้เลือก checkpoint และ threshold — ไม่ใช่ผลที่รายงาน)",
}
METRIC_KEYS = ("tss", "hss2", "bss", "auc", "precision", "recall", "f1", "accuracy", "tp", "fp", "tn", "fn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, help="dataset ของงานศึกษา (ปริยาย: data/processed/study_sequences)")
    p.add_argument(
        "--forecast-config", type=Path,
        help="override configs/forecast.yaml — ค่าร่วม train:/loss:/eval: ของไฟล์นี้เป็นฐานของทุกสถาปัตยกรรม",
    )
    p.add_argument("--out-dir", type=Path, help="ที่เก็บผล (ปริยาย: artifacts/model_comparison)")
    p.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS_FILE)
    p.add_argument("--architectures-file", type=Path, default=DEFAULT_ARCHITECTURES_FILE)
    p.add_argument(
        "--architectures",
        help="เทรนเฉพาะบางสถาปัตยกรรม คั่นด้วยจุลภาค เช่น lstm,tcn (ปริยาย: ทุกตัวในไฟล์)",
    )
    p.add_argument("--variants", help="เทรนเฉพาะบางแบบ คั่นด้วยจุลภาค เช่น V0,V5 (ปริยาย: ตามที่แต่ละสถาปัตยกรรมประกาศ)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help=f"seed ที่ใช้กับทุกเซลล์ (ปริยาย {DEFAULT_SEEDS})")
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch ใน config (ใช้ตอนทดสอบ)")
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    p.add_argument("--force", action="store_true", help="เทรนใหม่แม้มีผลของ (แบบ, seed) นั้นอยู่แล้ว")
    p.add_argument("--summarise-only", action="store_true", help="ไม่เทรน แค่รวมผลที่มีเป็นตาราง/รายงาน")
    p.add_argument(
        "--cv",
        action="store_true",
        help="rolling/blocked-window cross-validation แทน single split เดียว (อ่านพารามิเตอร์ "
        "จาก configs/data.yaml: split.cv) — ต้องใช้ dataset ที่สร้างด้วย "
        "study/build_dataset.py --no-split (--data-dir ปริยายจะเปลี่ยนเป็น study_sequences_cv)",
    )
    p.add_argument(
        "--cv-expanding", action="store_true",
        help="ใช้ expanding window แทน blocked (ทับ split.cv.expanding) — train เริ่มที่จุดตั้งต้นเสมอ",
    )
    p.add_argument(
        "--cv-anchor", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        help="จุดตั้งต้นของ fold แรก YYYY-MM-DD (ทับ split.cv.anchor) เช่น 2011-01-01 ให้ fold ตรงปีปฏิทิน",
    )
    p.add_argument("--cv-train-years", type=int, help="ความยาว train ของ fold แรก (ทับ split.cv.train_years)")
    return p.parse_args()


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    raise TypeError(f"แปลง {type(obj).__name__} เป็น JSON ไม่ได้")


def _fold_bounds_dict(bounds) -> dict:
    """serialize ขอบเขตของ fold หนึ่งเป็น dict ธรรมดา (ไม่ใช่ pydantic model) สำหรับ JSON/รายงาน"""
    return {
        "train_start": bounds.train_start.isoformat() if bounds.train_start else None,
        "train_end": bounds.train_end.isoformat(),
        "val_end": bounds.val_end.isoformat(),
        "test_end": bounds.test_end.isoformat() if bounds.test_end else None,
        "gap_days": bounds.gap_days,
    }


def run_signature(cfg, epochs: int, arrays: StudyArrays, split_col: pd.Series) -> dict:
    """สิ่งที่ต้องเหมือนกันจึงจะเอาผลของ run เก่ามารวมกับ run ใหม่ได้

    ``split_col`` รับมาจากภายนอกแทนที่จะอ่าน ``arrays.meta["split"]`` ตรงๆ เพราะตอน --cv
    คอลัมน์นั้นเป็นค่า placeholder ("train" ทั้งก้อน มาจาก study/build_dataset.py --no-split)
    ส่วน split จริงของแต่ละ fold คำนวณที่ runtime — ถ้า sample ไหนเปลี่ยน split เมื่อ fold
    เปลี่ยน ต้องถือว่าเป็นคนละ run กัน ไม่งั้นผลของ fold เก่าจะถูกนำมารวมกับ fold ใหม่เงียบๆ
    """
    rows = pd.util.hash_pandas_object(
        arrays.meta[["HARPNUM", "issue_time"]].assign(split=split_col.to_numpy()), index=False
    )
    signature = {
        # มี ``kind`` และ hyperparameter ของสถาปัตยกรรมอยู่ในนี้แล้ว — ผลของสถาปัตยกรรมหนึ่ง
        # จึงไม่มีทางถูกนำไปรวมกับอีกตัวโดยเงียบ ๆ แม้ชื่อไฟล์จะถูกเปลี่ยนมือ
        "arch": cfg.model_dump(mode="json"),
        "epochs": int(epochs),
        "dataset": {
            "n_samples": int(len(arrays.x)),
            "features": arrays.features,
            "rows_sha1": hashlib.sha1(rows.to_numpy().tobytes()).hexdigest()[:16],
        },
    }
    return json.loads(json.dumps(signature))


# --------------------------------------------------------------------------- #
# การเทรนหนึ่ง (แบบ, seed) — sunseg.training.forecast.fit_one (ลูปเดียวกับ production)
# --------------------------------------------------------------------------- #


def warm_up(splits: SequenceSplits, cfg, device: torch.device) -> None:
    """เทรนทิ้ง 1 epoch ก่อนตัวจับเวลาของแบบใดเริ่ม แล้วทิ้งผลทั้งหมด

    ต้นทุนเปิดครั้งเดียว (CUDA context, cuDNN, kernel ที่โหลดตอนใช้ครั้งแรก — หลายวินาที) ไม่งั้นจะตกที่
    รันแรกของ session ซึ่งคือ V0 (control) เสมอ แล้วทุกอัตราส่วน "เทียบ V0" เพี้ยนไปด้วย ใช้เส้นทางเทรนจริง
    (``fit_one``) ไม่ใช่โมเดลจำลอง เพื่อให้ครอบคลุมทุกส่วนที่รันจริง — ไม่กระทบผลของรันจริง เพราะ
    ``fit_one`` เรียก ``set_seed`` ใหม่ทุกครั้ง และไม่เขียนไฟล์ใด ๆ
    """
    fit_one(splits, cfg, seed=0, device=device, epochs=1)


def _scope_report(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    n, n_pos = int(len(y)), int(np.sum(y))
    report: dict = {"n": n, "n_pos": n_pos, "threshold": float(threshold)}
    if 0 < n_pos < n:
        full = classification_report(y, prob, threshold)
        return {**report, **{key: full[key] for key in METRIC_KEYS}}

    # มี class เดียว (หรือว่าง): TSS/HSS/AUC ไม่นิยาม — tss() จะคืน -FPR หรือ recall ซึ่งอ่าน
    # ผิดได้ง่ายว่าเป็นผลจริง จึงบังคับเป็น NaN แล้วเก็บแค่ confusion counts
    report.update({key: float("nan") for key in METRIC_KEYS})
    if n:
        counts = confusion_counts(y, np.asarray(prob) >= threshold)
        report.update({"tp": counts.tp, "fp": counts.fp, "tn": counts.tn, "fn": counts.fn})
    return report


def scope_reports(meta: pd.DataFrame, p_val: np.ndarray, p_test: np.ndarray, threshold: float) -> list[dict]:
    """ตัวเลขทุกขอบเขตใน :data:`SCOPES` ด้วย threshold ที่ freeze จาก val แล้ว"""
    probs = {"val": p_val, "test": p_test}
    out = []
    for scope, split in SCOPES:
        rows = meta[meta["split"] == split]
        prob = probs[split]
        if len(rows) != len(prob):
            raise ValueError(f"split={split}: meta มี {len(rows)} แถว แต่มีค่าทำนาย {len(prob)} ค่า")
        out.append({"scope": scope, **_scope_report(rows["label"].to_numpy(), prob, threshold)})
    return out


def long_predictions(
    meta: pd.DataFrame, architecture: str, variant: str, seed: int,
    p_val: np.ndarray, p_test: np.ndarray, threshold: float,
) -> pd.DataFrame:
    """หนึ่งแถวต่อ (สถาปัตยกรรม, แบบ, seed, split, HARPNUM, issue_time) — รูปแบบยาว"""
    columns = [c for c in ("split", "HARPNUM", "noaa_ar", "issue_time", "label") if c in meta.columns]
    frames = []
    for split, prob in (("val", p_val), ("test", p_test)):
        rows = meta.loc[meta["split"] == split, columns].reset_index(drop=True).copy()
        rows.insert(0, "seed", seed)
        rows.insert(0, "variant", variant)
        rows.insert(0, "architecture", architecture)
        rows["prob"] = np.asarray(prob, dtype=np.float64)
        rows["threshold"] = float(threshold)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def run_paths(
    out_dir: Path, architecture: str, variant: str, seed: int, fold: int | None = None
) -> tuple[Path, Path, Path]:
    stem = f"{architecture}_{variant}_seed{seed}"
    if fold is not None:
        stem = f"{architecture}_{variant}_fold{fold}_seed{seed}"
    return out_dir / "runs" / f"{stem}.json", out_dir / "runs" / f"{stem}_pred.parquet", out_dir / "models" / f"{stem}.pt"


def load_run(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# สรุปผล
# --------------------------------------------------------------------------- #


def dataset_balance(meta: pd.DataFrame) -> pd.DataFrame:
    table = (
        meta.groupby("split", sort=False)
        .agg(n=("label", "size"), n_pos=("label", "sum"), first=("issue_time", "min"), last=("issue_time", "max"))
        .reset_index()
    )
    order = {"train": 0, "val": 1, "test": 2}
    return table.sort_values("split", key=lambda s: s.map(order))


def _pm(mean: float, sd: float, signed: bool = True) -> str:
    if not np.isfinite(mean):
        return "n/a"
    head = f"{mean:+.3f}" if signed else f"{mean:.3f}"
    return head if not np.isfinite(sd) else f"{head} ± {sd:.3f}"


#: เมตริกที่ขึ้นตารางผลหลัก เรียงตามความสำคัญ — TSS มาก่อนเพราะเป็นตัวที่ใช้เลือก threshold
#: และเป็นตัวที่ทั้งรายงานอ้างอิง ส่วนที่เหลือเป็นมุมอื่นของผลเดียวกัน
REPORT_METRICS: tuple[str, ...] = ("tss", "hss2", "bss", "auc", "recall", "precision")
METRIC_TITLES = {
    "tss": "TSS", "hss2": "HSS2", "bss": "BSS", "auc": "AUC",
    "recall": "Recall", "precision": "Precision",
}


def metric_cells(runs: pd.DataFrame, metrics: tuple[str, ...] = REPORT_METRICS) -> pd.DataFrame:
    """mean/SD ของทุกเมตริกต่อ ``(scope, metric, architecture, variant)``

    ``skipna=False`` ทุกจุดตามกติกาของงานนี้ — ขอบเขตที่ไม่มี positive ให้ TSS เป็น NaN
    ถ้าเฉลี่ยโดยข้าม NaN จะได้ตัวเลขที่ดูใช้ได้แต่คำนวณจาก seed คนละชุดกับช่องอื่น
    """
    frames = []
    for metric in metrics:
        if metric not in runs.columns:
            continue
        grouped = runs.groupby(["scope", "architecture", "variant"], sort=False)[metric]
        frame = pd.DataFrame({
            "mean": grouped.apply(lambda s: float(s.mean(skipna=False))),
            "sd": grouped.apply(
                lambda s: float(s.std(ddof=1, skipna=False)) if len(s) > 1 else float("nan")
            ),
        }).reset_index()
        frame["metric"] = metric
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["scope", "metric", "architecture", "variant", "mean", "sd"]).set_index(
            ["scope", "metric", "architecture", "variant"]
        )
    # sort_index กัน PerformanceWarning ตอน .loc[(scope, metric)] บน MultiIndex ที่ยังไม่เรียง
    return (
        pd.concat(frames, ignore_index=True)
        .set_index(["scope", "metric", "architecture", "variant"])
        .sort_index()
    )


def _leader(block: pd.DataFrame, architectures: dict[str, str], column: str) -> str | None:
    """สถาปัตยกรรมที่นำในคอลัมน์นี้ **เฉพาะเมื่อนำเกินความผันผวนของ seed** ไม่งั้นคืน None

    ตารางอ้างอิงในวรรณกรรมทำตัวหนาให้ตัวที่ดีที่สุดเสมอ แต่ที่นี่ส่วนต่างระหว่างสถาปัตยกรรม
    (~0.005) เล็กกว่า SD ภายในเซลล์ (~0.015) การทำตัวหนาทุกช่องจึงเท่ากับชี้ไปที่ noise
    แล้วผู้อ่านจะสรุปผิดทันที — ช่องที่ไม่มีตัวหนาเลยคือคำตอบของมันเอง
    """
    values = []
    for name in architectures:
        if (name, column) not in block.index:
            continue
        mean, sd = block.loc[(name, column), "mean"], block.loc[(name, column), "sd"]
        if np.isfinite(mean):
            values.append((float(mean), float(sd) if np.isfinite(sd) else 0.0, name))
    if len(values) < 2:
        return None
    values.sort(key=lambda v: -v[0])
    (best, best_sd, name), (second, second_sd, _) = values[0], values[1]
    return name if best - second >= max(best_sd, second_sd) else None


def _metric_table_lines(
    architectures: dict[str, str],
    columns: list[str],
    cells: pd.DataFrame,
    scope: str,
    variants: dict[str, dict],
    metrics: tuple[str, ...] = REPORT_METRICS,
) -> list[str]:
    """ตารางผลหลัก — กลุ่มละเมตริก ในกลุ่มหนึ่งแถวต่อสถาปัตยกรรม คอลัมน์คือชุด feature

    รูปแบบเดียวกับตารางเปรียบเทียบโมเดลในวรรณกรรมสายนี้: ทุกเมตริกอยู่ในตารางเดียว
    ผู้อ่านจึงตัดสินเซลล์หนึ่งได้โดยไม่ต้องกระโดดข้ามตาราง
    """
    heads = " | ".join(f"{variants[c]['label']}<br>`{c}`" for c in columns)
    lines = [f"| เมตริก | สถาปัตยกรรม | {heads} |", "|---|---|" + "---:|" * len(columns)]

    for metric in metrics:
        try:
            block = cells.loc[(scope, metric)]
        except KeyError:
            continue
        leaders = {c: _leader(block, architectures, c) for c in columns}
        for index, (name, label) in enumerate(architectures.items()):
            texts = []
            for column in columns:
                if (name, column) not in block.index or not np.isfinite(block.loc[(name, column), "mean"]):
                    texts.append("n/a")
                    continue
                mean, sd = block.loc[(name, column), "mean"], block.loc[(name, column), "sd"]
                text = f"{mean:.3f} ({sd:.3f})" if np.isfinite(sd) else f"{mean:.3f}"
                texts.append(f"**{text}**" if leaders.get(column) == name else text)
            head = f"**{METRIC_TITLES.get(metric, metric.upper())}**" if index == 0 else ""
            lines.append(f"| {head} | {label} | " + " | ".join(texts) + " |")
    return lines


def _overview_lines(
    architectures: dict[str, str], columns: list[str], table: pd.DataFrame, scope: str, variants: dict[str, dict]
) -> list[str]:
    """สรุปขอบตารางของ TSS สองบรรทัด + ประโยคบอกระดับ noise

    ตอบคำถามหลักสองข้อโดยไม่ต้องให้ผู้อ่านคำนวณเอง: แกนสถาปัตยกรรมมีผลไหม (บรรทัดแรก)
    และแกนชุด feature มีผลไหม (บรรทัดสอง) ถ้าทั้งสองบรรทัดแบนราบก็คือไม่มีแกนไหนมีผล
    """
    means = _cell_matrix(architectures, columns, table, scope, "mean")
    sds = _cell_matrix(architectures, columns, table, scope, "sd")

    by_arch = " · ".join(f"{architectures[a]} {means.loc[a].mean(skipna=False):.3f}" for a in architectures)
    by_variant = " · ".join(f"{variants[c]['label']} {means[c].mean(skipna=False):.3f}" for c in columns)

    flat = means.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    typical_sd = float(np.nanmean(sds.to_numpy()))
    lines = [
        f"- **เฉลี่ยข้ามชุด feature** (แกนสถาปัตยกรรม): {by_arch}",
        f"- **เฉลี่ยข้ามสถาปัตยกรรม** (แกนชุด feature): {by_variant}",
    ]
    if len(flat) > 1 and np.isfinite(typical_sd) and typical_sd:
        spread = float(flat.max() - flat.min())
        verdict = (
            "**แคบกว่าความผันผวนของ seed ภายในเซลล์เดียว**" if spread <= typical_sd
            else f"กว้างกว่า SD ภายในเซลล์ {spread / typical_sd:.1f} เท่า"
        )
        lines.append(
            f"- ทั้ง {len(flat)} เซลล์อยู่ในช่วง {flat.min():.3f}–{flat.max():.3f} (กว้าง {spread:.3f}) · "
            f"SD ภายในเซลล์โดยเฉลี่ย {typical_sd:.3f} · ช่วงของตาราง{verdict}"
        )
    return lines


def _cell_matrix(
    architectures: dict[str, str], columns: list[str], table: pd.DataFrame, scope: str, column: str
) -> pd.DataFrame:
    """ค่าของทุกเซลล์เป็นตาราง (แถว = สถาปัตยกรรม, คอลัมน์ = แบบ) — เซลล์ที่ไม่มีผลเป็น NaN"""
    data = {
        name: [
            float(table.loc[(scope, name, c), column]) if (scope, name, c) in table.index else float("nan")
            for c in columns
        ]
        for name in architectures
    }
    return pd.DataFrame(data, index=columns).T


def _matrix_lines(
    architectures: dict[str, str],
    columns: list[str],
    table: pd.DataFrame,
    scope: str,
    variants: dict[str, dict],
) -> list[str]:
    """กริดผลหลักพร้อมค่าเฉลี่ยขอบ และประโยคที่บอกระดับ noise ไว้ข้างตาราง

    ค่าเฉลี่ยขอบมีไว้ตอบคำถามหลักสองข้อในแวบเดียว: ขอบขวาคือ "สถาปัตยกรรมนี้ทำได้เท่าไหร่
    โดยเฉลี่ยข้ามชุด feature" และขอบล่างคือ "ชุด feature นี้ให้เท่าไหร่โดยเฉลี่ยข้ามสถาปัตยกรรม"
    ถ้าขอบทั้งสองด้านแบนราบ แปลว่าไม่มีแกนไหนมีผล ซึ่งผู้อ่านควรเห็นได้โดยไม่ต้องคำนวณเอง

    บรรทัดสรุปใต้ตารางเทียบ **ช่วงของทั้งตาราง** กับ **SD ภายในเซลล์** โดยตรง — ถ้าช่วงไม่กว้าง
    กว่า SD ก็คือไม่มีอะไรให้สรุป และนั่นคือผลการทดลอง ไม่ใช่ความล้มเหลวของการทดลอง
    """
    means = _cell_matrix(architectures, columns, table, scope, "mean")
    sds = _cell_matrix(architectures, columns, table, scope, "sd")

    heads = " | ".join(f"{variants[c]['label']}<br>`{c}`" for c in columns)
    lines = [
        f"| สถาปัตยกรรม | {heads} | **เฉลี่ยของแถว** |",
        "|---|" + "---:|" * (len(columns) + 1),
    ]
    for name, label in architectures.items():
        cells = [
            _pm(means.loc[name, c], sds.loc[name, c], signed=False) if np.isfinite(means.loc[name, c]) else "n/a"
            for c in columns
        ]
        row_mean = means.loc[name].mean(skipna=False)
        tail = f"**{row_mean:.3f}**" if np.isfinite(row_mean) else "n/a"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {tail} |")

    col_means = [means[c].mean(skipna=False) for c in columns]
    cells = [f"**{m:.3f}**" if np.isfinite(m) else "n/a" for m in col_means]
    lines.append("| **เฉลี่ยของคอลัมน์** | " + " | ".join(cells) + " | |")

    flat = means.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    typical_sd = float(np.nanmean(sds.to_numpy()))
    if len(flat) > 1:
        spread = float(flat.max() - flat.min())
        verdict = (
            "**แคบกว่าความผันผวนของ seed ภายในเซลล์เดียว**" if spread <= typical_sd
            else f"กว้างกว่า SD ภายในเซลล์ {spread / typical_sd:.1f} เท่า" if np.isfinite(typical_sd) and typical_sd
            else ""
        )
        lines += [
            "",
            f"ทั้ง {len(flat)} เซลล์อยู่ในช่วง {flat.min():.3f}–{flat.max():.3f} (กว้าง {spread:.3f}) · "
            f"SD ภายในเซลล์โดยเฉลี่ย {typical_sd:.3f} · ช่วงของตาราง{verdict}",
        ]
    return lines


def _grid_rows(
    architectures: dict[str, str],
    columns: list[str],
    table: pd.DataFrame,
    scope: str,
    mean_col: str,
    sd_col: str,
    signed: bool = True,
) -> list[str]:
    """หนึ่งแถวต่อสถาปัตยกรรม ช่องละหนึ่งค่า mean ± SD

    ``table`` ต้องมี index เป็น ``(scope, architecture, variant)`` — เซลล์ที่ไม่มีผลแสดง n/a
    แทนที่จะหายไปเงียบ ๆ เพราะช่องว่างในตารางอ่านไม่ออกว่าคือ "ไม่ได้รัน" หรือ "รันแล้ววัดไม่ได้"
    """
    rows = []
    for name, label in architectures.items():
        cells = []
        for column in columns:
            key = (scope, name, column)
            cells.append(
                _pm(table.loc[key, mean_col], table.loc[key, sd_col], signed=signed)
                if key in table.index
                else "n/a"
            )
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return rows


def _ranking_lines(
    ranking: pd.DataFrame, architectures: dict[str, str], variants: dict[str, dict], pair_label: str
) -> list[str]:
    """ผลหลัก — ทุกเซลล์เรียงตาม TSS เฉลี่ยบน test พร้อมระยะห่างจากอันดับ 1 แบบจับคู่ และอันดับบน validation

    คอลัมน์อันดับบน validation มีไว้ตอบว่าผู้ที่ **ไม่ได้เห็น test** จะเลือกเซลล์เดียวกันไหม — การหยิบตัวสูงสุด
    จากหลายเซลล์บน test ชุดเดียวให้ค่าที่เฟ้อขึ้นเสมอ (winner's curse) ถ้าสองอันดับไม่สอดคล้องกัน อันดับบน test
    ส่วนใหญ่คือ noise
    """
    has_val = "val_rank" in ranking.columns
    head = "| อันดับ | สถาปัตยกรรม | ชุด feature | TSS test (mean ± SD) | ห่างจากอันดับ 1 (จับคู่ " + pair_label + ") | แยกจากอันดับ 1 ได้ |"
    rule = "|---:|---|---|---:|---:|:---:|"
    if has_val:
        head += " TSS validation | อันดับบน validation |"
        rule += "---:|---:|"
    lines = [head, rule]
    for row in ranking.itertuples(index=False):
        label = f"{variants.get(row.variant, {'label': row.variant})['label']} `{row.variant}`"
        gap = "—" if row.rank == 1 else _pm(row.gap_mean, row.gap_sd)
        separable = "—" if row.rank == 1 else ("ไม่ได้" if row.tied_with_best else "**ได้**")
        text = (
            f"| {row.rank} | {architectures.get(row.architecture, row.architecture)} | {label} | "
            f"{_pm(row.mean, row.sd, signed=False)} | {gap} | {separable} |"
        )
        if has_val:
            text += f" {_pm(row.val_mean, float('nan'), signed=False)} | {int(row.val_rank)} |"
        lines.append(text)

    best = ranking.iloc[0]
    best_name = (
        f"{architectures.get(best.architecture, best.architecture)} + "
        f"{variants.get(best.variant, {'label': best.variant})['label']} (`{best.variant}`)"
    )
    tied = ranking[(ranking["rank"] > 1) & ranking["tied_with_best"]]
    lines += [
        "",
        f"- **เซลล์ที่ดีที่สุด: {best_name}** — TSS {_pm(best['mean'], best['sd'], signed=False)} บน test",
        f"- **แยกจากอันดับ 1 ไม่ได้ {len(tied)} จาก {len(ranking) - 1} เซลล์** (ระยะห่างไม่เกิน SD ของผลต่างจับคู่ "
        f"{pair_label}) — ที่จำนวน seed เท่านี้ เซลล์เหล่านี้ถือว่าเสมอกับอันดับ 1",
    ]
    if has_val:
        val_best = ranking.loc[ranking["val_rank"] == ranking["val_rank"].min()].iloc[0]
        rho = float(ranking["mean"].corr(ranking["val_mean"], method="spearman"))
        same = val_best.architecture == best.architecture and val_best.variant == best.variant
        lines.append(
            f"- **ถ้าเลือกด้วย validation** (โดยไม่เห็น test) จะได้ "
            f"{architectures.get(val_best.architecture, val_best.architecture)} `{val_best.variant}` "
            f"ซึ่งอยู่อันดับ {int(val_best['rank'])} บน test"
            + (" — ตรงกับอันดับ 1 บน test" if same else f" · อันดับ 1 บน test อยู่อันดับ {int(best.val_rank)} บน validation")
            + f" · Spearman ρ ระหว่างสองอันดับ {rho:+.2f}"
        )
    return lines


def _best_per_architecture_lines(
    ranking: pd.DataFrame, architectures: dict[str, str], variants: dict[str, dict]
) -> list[str]:
    """เซลล์ที่ดีที่สุดของแต่ละสถาปัตยกรรม — ตอบว่าแต่ละสถาปัตยกรรมทำได้ดีสุดเท่าไหร่เมื่อได้เลือกชุด feature เอง"""
    lines = [
        "| สถาปัตยกรรม | ชุด feature ที่ดีที่สุด | TSS test (mean ± SD) | อันดับรวม | เฉลี่ยทุกชุด feature |",
        "|---|---|---:|---:|---:|",
    ]
    for name, label in architectures.items():
        own = ranking[ranking["architecture"] == name]
        if own.empty:
            continue
        top = own.iloc[0]
        spec = variants.get(top.variant, {"label": top.variant})
        lines.append(
            f"| {label} | {spec['label']} `{top.variant}` | {_pm(top['mean'], top['sd'], signed=False)} | "
            f"{int(top['rank'])} | {own['mean'].mean(skipna=False):.3f} |"
        )
    return lines


def _sec(mean: float, sd: float) -> str:
    return "n/a" if not np.isfinite(mean) else f"{mean:.1f}" if not np.isfinite(sd) else f"{mean:.1f} ± {sd:.1f}"


def _timing_lines(timing: pd.DataFrame, labels: dict[str, str], devices: list[str], untimed: int) -> list[str]:
    """หัวข้อ "เวลาที่ใช้" — สองตาราง: (1) ตั้งแต่เริ่มจนได้ผลรายสถาปัตยกรรม (2) แตกเป็นขั้นและต่อ epoch

    สรุปตามสถาปัตยกรรม ไม่ใช่ตามแบบ เพราะบนแกนแบบต้นทุนแทบไม่ต่างกัน (เพิ่ม feature ไม่กี่คอลัมน์)
    ส่วนบนแกนสถาปัตยกรรมต่างกันได้เป็นสิบเท่า ซึ่งเป็นตัวเลขที่ผู้อ่านต้องรู้
    """
    lines = [
        "## เวลาที่ใช้ตั้งแต่เริ่มจนได้ผล (แยกรายสถาปัตยกรรม)",
        "",
        f"เวลาจริงบนนาฬิกาผนัง (wall-clock) · อุปกรณ์: {', '.join(devices) if devices else 'ไม่ทราบ'}",
        "",
        "- **รวมอยู่ในตัวเลข:** เตรียมข้อมูลของแบบนั้น (หั่นคอลัมน์ + normalise ทำครั้งเดียวต่อแบบ) และทุกรัน: "
        "สร้างโมเดล → เทรน → ทำนาย val/test → คำนวณตัวชี้วัด → เขียน predictions + checkpoint",
        "- **ไม่รวม:** การโหลดไฟล์ dataset ก้อนเดียวที่ทุกแบบใช้ร่วมกัน และการดาวน์โหลด/สกัด feature ต้นทางก่อนสร้าง "
        "dataset (SHARP keywords, ภาพ + U-Net สำหรับ intensity, GOES XRS สำหรับ X-ray) — ส่วนหลังคือต้นทุนที่ต่างกัน"
        "ระหว่างแบบจริง ๆ ตารางนี้จึงบอกได้เฉพาะต้นทุนของขั้นเทรน ต้นทุนรวมตั้งแต่ต้นทางอยู่ในหัวข้อ "
        "\"เวลาตั้งแต่ต้นทางจนได้ผล\" ถัดไป (ต้องมี `acquisition_time.json` จาก `measure_acquisition_time.py`)",
        "",
        "### ตั้งแต่เริ่มจนได้ผล",
        "",
        f"| สถาปัตยกรรม | รัน | เตรียมข้อมูล (วินาที) | ต่อ 1 รัน (วินาที, mean ± SD) "
        f"| เทียบ {labels.get(ANCHOR_ARCHITECTURE, ANCHOR_ARCHITECTURE)} | รวมทั้งแถว (นาที) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.architecture, row.architecture)} | {row.n_runs} | {row.prepare_seconds:.2f} | "
            f"{_sec(row.total_mean, row.total_sd)} | {row.total_ratio:.2f}× | {row.start_to_result_seconds / 60:.1f} |"
        )
    lines += [
        "",
        "### ต่อ 1 รันแยกตามขั้น และต่อ epoch",
        "",
        f"| สถาปัตยกรรม | เทรน (วินาที/รัน) | สร้างโมเดล+ประเมิน+เขียนไฟล์ (วินาที/รัน) | epoch ที่รัน (เฉลี่ย) "
        f"| วินาที/epoch | เทียบ {labels.get(ANCHOR_ARCHITECTURE, ANCHOR_ARCHITECTURE)} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in timing.itertuples(index=False):
        lines.append(
            f"| {labels.get(row.architecture, row.architecture)} | {row.train_mean:.1f} | {row.other_mean:.1f} | "
            f"{row.epochs_mean:.1f} | {row.sec_per_epoch:.3f} | {row.sec_per_epoch_ratio:.2f}× |"
        )
    lines += [
        "",
        "- เวลาต่อรันขึ้นกับจำนวน epoch ก่อน early stopping เป็นหลัก ซึ่งต่างกันตาม seed จึงมี SD กว้าง — "
        "ให้ดู **วินาที/epoch** (Σเวลาเทรน ÷ Σepoch) เป็นตัวเทียบต้นทุนต่อ epoch ของแต่ละแบบ",
        "- เทียบเวลาข้ามแบบได้ตรงก็ต่อเมื่อทุกแบบถูกเทรนใน session เดียวกัน — สคริปต์เทรนสลับแบบภายใน seed เดียวกัน "
        "เพื่อให้ความเร็วเครื่องที่แกว่งตามเวลากระทบทุกแบบเท่ากัน ถ้าเทรนบางแบบแยกรอบ (เช่น `--variants`) หรือ resume "
        "ข้ามวัน อย่าเทียบเวลาข้ามแบบ · ต้นทุนเปิด CUDA/cuDNN ครั้งเดียวถูกอุ่นเครื่องไว้ก่อนเริ่มจับเวลา ไม่ตกที่แบบใดแบบหนึ่ง",
    ]
    if untimed:
        lines.append(
            f"- มี {untimed} รันที่ไม่มีข้อมูลเวลา (สร้างก่อนเพิ่มการจับเวลา) จึงไม่นับในตารางนี้ — "
            "ใช้ `--force` เทรนใหม่เพื่อให้ครบ"
        )
    lines.append("")
    return lines


def _hours_text(hours: float) -> str:
    return f"{hours:.1f}" if hours >= 10 else f"{hours:.2f}"


def _acquisition_lines(
    end_to_end: pd.DataFrame, acquisition: dict, variants: dict[str, dict], variant_groups: dict[str, frozenset[str]]
) -> list[str]:
    """หัวข้อ "ตั้งแต่ต้นทางจนได้ผล" — รวมเวลาดาวน์โหลด/สกัด feature ต้นทางเข้ากับเวลาเทรนรายแบบ
    (ตัวเลขต้นทางมาจาก ``measure_acquisition_time.py`` ผ่าน ``acquisition_time.json``)"""
    components = acquisition["components"]
    users = {
        group: ", ".join(v for v in variants if group in variant_groups[v]) or "—"
        for group in ("intensity", "xray")
    }
    users["shared"] = "ทุกแบบ"

    lines = [
        "## เวลาตั้งแต่ต้นทางจนได้ผล (รวมดาวน์โหลด/สกัด feature)",
        "",
        f"เวลาที่ใช้ตอนทำจริง (นาฬิกาผนัง เฉพาะช่วงที่งานยังเดินอยู่) — **ไม่ใช่เวลาที่จะใช้ถ้าทำซ้ำ** ขึ้นกับ JSOC/NOAA "
        f"เครือข่าย และจำนวนโปรเซสที่รันพร้อมกันตอนนั้น · ขั้นต้นทางทำไว้ก่อนมีการจับเวลา จึงประเมินจาก log และเวลา"
        f"แก้ไขไฟล์ผลลัพธ์ ยกเว้นสองขั้นที่วัดสด (วิธีและข้อจำกัด: `sunseg/study/acquisition_time.py`) · "
        f"วัดเมื่อ {acquisition.get('generated_at', 'ไม่ทราบ')}",
        "",
        "### รวมรายแบบ",
        "",
        f"| แบบ | ชุด feature | ต้นทางร่วมทุกแบบ (ชม.) | + intensity (ชม.) | + X-ray (ชม.) | เทรน+ประเมินทั้งแถว (ชม.) "
        f"| รวม (ชม.) | เทียบ {ANCHOR_VARIANT} |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in end_to_end.itertuples(index=False):
        label = variants.get(row.variant, {"label": row.variant})["label"]
        lines.append(
            f"| {row.variant} | {label} | {_hours_text(row.shared_hours)} | {_hours_text(row.intensity_hours)} | "
            f"{_hours_text(row.xray_hours)} | {_hours_text(row.train_hours)} | {_hours_text(row.total_hours)} | "
            f"{row.total_ratio:.1f}× |"
        )
    lines += [
        "",
        "- **ต้นทางร่วมทุกแบบ** = SHARP (แถวข้อมูลและ label ของทุกแบบมาจาก SHARP แม้ V4/V5 ไม่ได้ใช้เป็น feature) + ประกอบ dataset",
        "- **+ intensity** = ภาพ HMI + ภาพ AIA + เทรน U-Net + สกัด intensity · **+ X-ray** = ดาวน์โหลด GOES + แปลงเป็น feature "
        "· แบบที่ไม่ใช้ feature กลุ่มนั้นไม่จ่ายต้นทุนกลุ่มนั้น",
        "- **เทรน+ประเมินทั้งแถว** คือ 'รวมทั้งแบบ' จากหัวข้อก่อนหน้า (ทุก seed)",
        "",
        "### แต่ละขั้นต้นทาง",
        "",
        "| ขั้น | ใช้กับแบบ | เวลาที่คิด (ชม.) | ขอบบน: ทุกอย่างที่ทำไปจริง (ชม.) | ปริมาณ | วิธีได้ตัวเลข |",
        "|---|---|---:|---:|---|---|",
    ]
    for c in components:
        lines.append(
            f"| {c['label']} | {users[c['group']]} | {_hours_text(c['hours'])} | {_hours_text(c['hours_upper'])} | "
            f"{c['volume'] or '—'} | {c['method']} |"
        )
    lines += [
        "",
        "**เวลาที่คิด** = ส่วนที่จำเป็นต่อการทดลองนี้ · **ขอบบน** = รวมรอบที่ซ้ำ งานข้าง ๆ (เช่น เฟรมของ case study, "
        "รอบเทรน U-Net ที่ไม่ได้ใช้) และเฟรมของงานทดลองอื่นที่โหลดปนกัน",
        "",
        *[f"- **{c['label']}** — {c['detail']}" for c in components if c["detail"]],
        "",
    ]
    return lines


def render_report(
    cell_tss: pd.DataFrame,
    delta_arch_tss: pd.DataFrame,
    interaction_tss: pd.DataFrame,
    cells_by_metric: pd.DataFrame,
    runs_table: pd.DataFrame,
    variants: dict[str, dict],
    architectures: dict[str, str],
    matrix_variants: list[str],
    extra_tss: pd.DataFrame,
    seeds: list[int],
    epochs: int,
    balance: pd.DataFrame | None = None,
    fold_summaries: list[dict] | None = None,
    pair_cols: tuple[str, ...] = ("seed",),
    timing: pd.DataFrame | None = None,
    timing_devices: list[str] | None = None,
    untimed_runs: int = 0,
    acquisition: dict | None = None,
    end_to_end: pd.DataFrame | None = None,
    variant_groups: dict[str, frozenset[str]] | None = None,
    ranking: pd.DataFrame | None = None,
) -> str:
    """``balance`` (single split) กับ ``fold_summaries`` (CV) มีให้อย่างใดอย่างหนึ่งเท่านั้น

    ``ranking`` (จาก :func:`sunseg.study.summary.rank_cells` บน test พร้อมคอลัมน์ ``val_mean``/``val_rank``
    ถ้ามี validation) คือผลหลักของรายงาน · ``timing`` (จาก :func:`sunseg.study.summary.timing_summary`)
    ว่างได้ — ผลที่เทรนก่อนมีการจับเวลาไม่มีข้อมูลนี้ หัวข้อเวลาจึงถูกข้ามไปทั้งหัวข้อแทนที่จะแสดงตารางว่าง ·
    ``acquisition`` + ``end_to_end`` (เวลาต้นทาง จาก ``measure_acquisition_time.py``) ว่างได้เช่นกัน
    """
    cv = fold_summaries is not None
    compared = [v for v in matrix_variants if v != ANCHOR_VARIANT]
    rival_archs = [a for a in architectures if a != ANCHOR_ARCHITECTURE]
    pair_label = "/".join(pair_cols)
    anchor_label = architectures.get(ANCHOR_ARCHITECTURE, ANCHOR_ARCHITECTURE)
    n_cells = len(architectures) * len(matrix_variants)
    lines = [
        f"# เปรียบเทียบโมเดลพยากรณ์ flare — {n_cells} เซลล์ "
        f"({len(architectures)} สถาปัตยกรรม × {len(matrix_variants)} ชุด feature)"
        + (" (rolling/blocked-window cross-validation)" if cv else ""),
        "",
        f"สร้างเมื่อ {datetime.now():%Y-%m-%d %H:%M} · seed {', '.join(map(str, seeds))} (ชุดเดียวกันทุกเซลล์)"
        + (f" · {len(fold_summaries)} fold" if cv else "")
        + f" · hyperparameter รายสถาปัตยกรรมจาก `configs/study/architectures.yaml` (epoch สูงสุด {epochs}) · "
        f"threshold เลือกบน validation แยกต่อ (สถาปัตยกรรม, แบบ, {pair_label}) แล้ว freeze ก่อนวัด test",
        "",
        "**คำถามหลัก: เซลล์ไหน (สถาปัตยกรรม + ชุด feature) ให้ TSS เฉลี่ยบน test สูงสุด** — ชุด feature เป็นตัวเลือก"
        "หนึ่งของโมเดลเหมือน hyperparameter ไม่ใช่คำถามแยก · hyperparameter ของทุกสถาปัตยกรรมมาจากการค้นหาด้วยงบเท่ากัน "
        "ซึ่งเป็นชุดเดียวกับที่ระบบใช้ผลิต (`models.<ชื่อ>` ใน `configs/forecast.yaml`) · ค่าตั้งมือเดิมของ LSTM "
        "คือแถวเสริม (ดู `docs/adr/0001-tune-per-architecture.md`)",
        "",
    ]

    if cv:
        lines += [
            "## fold ของ cross-validation",
            "",
            (
                "ทุก fold เป็น expanding window (train เริ่มที่จุดเดียวกันแล้วยาวขึ้นทีละ step) — "
                if fold_summaries and fold_summaries[0]["bounds"].get("train_start") is None
                else "ทุก fold เป็น blocked window (train ขนาดคงที่ เลื่อนไปข้างหน้าทีละ step) ไม่ใช่ expanding — "
            )
            + "fold ที่มี val หรือ test positive น้อยเกินไปถูกตัดทิ้งไปแล้วก่อนเทรน (ดู log)",
            "",
            "| fold | train | val | test | sample (train/val/test) | positive (train/val/test) |",
            "|---|---|---|---|---:|---:|",
        ]
        for fs in fold_summaries:
            b, bal = fs["bounds"], fs["balance"].set_index("split")

            def _col(split: str, column: str, bal=bal) -> int:
                return int(bal.loc[split, column]) if split in bal.index else 0

            lines.append(
                f"| {fs['fold']} | {b['train_start'] or 'ต้นข้อมูล'}..{b['train_end']} | ..{b['val_end']} | ..{b['test_end']} | "
                f"{_col('train', 'n_samples')}/{_col('val', 'n_samples')}/{_col('test', 'n_samples')} | "
                f"{_col('train', 'n_positive')}/{_col('val', 'n_positive')}/{_col('test', 'n_positive')} |"
            )
        lines.append("")
    else:
        lines += [
            "## ข้อมูล",
            "",
            "| split | issue_time | sample | positive |",
            "|---|---|---:|---:|",
        ]
        for row in balance.itertuples(index=False):
            lines.append(
                f"| {row.split} | {pd.Timestamp(row.first):%Y-%m-%d} – {pd.Timestamp(row.last):%Y-%m-%d} | "
                f"{row.n} | {row.n_pos} |"
            )
        lines.append("")

    cell_idx = cell_tss.set_index(["scope", "architecture", "variant"])
    arch_idx = delta_arch_tss.set_index(["scope", "variant", "architecture"])
    inter_idx = interaction_tss.set_index(["scope", "architecture", "variant"])
    extra_idx = extra_tss.set_index(["scope", "architecture", "variant"]) if len(extra_tss) else None
    first_runs = runs_table.drop_duplicates(["scope"]).set_index("scope")

    # ผลหลักมีตารางเดียวและเป็น test เท่านั้น — validation กับ AUC ย้ายไปภาคผนวก
    # เพราะ validation ไม่ใช่ผลที่รายงาน (ใช้เลือก checkpoint กับ threshold) และการวางสองเมตริก
    # คู่กันบังคับให้ผู้อ่านกระโดดไปมาเพื่อตัดสินเซลล์เดียว
    if "test" in cell_idx.index.get_level_values("scope"):
        n, n_pos = int(first_runs.loc["test", "n"]), int(first_runs.loc["test", "n_pos"])
        lines += [
            "## ผลหลัก: อันดับของทุกเซลล์",
            "",
            f"ผลบน **test** · sample {n} · positive {n_pos} · {len(seeds)} seed ต่อเซลล์ · "
            "เรียงตาม TSS เฉลี่ยจากมากไปน้อย",
            "",
        ]
        if ranking is not None and not ranking.empty:
            lines += _ranking_lines(ranking, architectures, variants, pair_label)
            lines += [
                "",
                "### ดีที่สุดของแต่ละสถาปัตยกรรม",
                "",
            ]
            lines += _best_per_architecture_lines(ranking, architectures, variants)
            lines.append("")
        lines += [
            "### ตัวชี้วัดทุกตัว (test)",
            "",
            "ค่าในช่องคือ mean (SD ในวงเล็บ)",
            "",
        ]
        lines += _metric_table_lines(architectures, matrix_variants, cells_by_metric, "test", variants)
        lines += [
            "",
            "**ตัวหนา = สถาปัตยกรรมที่นำในคอลัมน์นั้น เฉพาะเมื่อนำเกิน SD ของทั้งคู่** — "
            "คอลัมน์ที่ไม่มีตัวหนาเลยแปลว่าไม่มีสถาปัตยกรรมไหนแยกออกจากกันได้ที่จำนวน seed เท่านี้ "
            "ซึ่งเป็นผลการทดลอง ไม่ใช่ตารางที่กรอกไม่ครบ",
            "",
            "### ภาพรวมของ TSS",
            "",
        ]
        lines += _overview_lines(architectures, matrix_variants, cell_idx, "test", variants)
        lines.append("")
    if timing is not None:
        lines += _timing_lines(timing, architectures, timing_devices or [], untimed_runs)
    if end_to_end is not None and acquisition is not None and variant_groups is not None:
        lines += _acquisition_lines(end_to_end, acquisition, variants, variant_groups)

    lines += [f"## ตัวเลขดิบราย {pair_label} (TSS)", "", "ให้คำนวณอันดับและ Δ ตรวจเองได้จากตารางนี้", ""]
    table = seed_table(runs_table, "test", pair_cols=pair_cols, unit_cols=("architecture", "variant"))
    if not table.empty:
        if cv:
            headers = [f"f{f}s{s}" for f, s in table.columns]
        else:
            headers = [f"seed {s}" for s in table.columns]
        lines += [f"### {SCOPE_TITLES['test']}", "", "| สถาปัตยกรรม | แบบ | " + " | ".join(headers) + " |",
                  "|---|---|" + "---:|" * len(table.columns)]
        for (name, variant), values in table.iterrows():
            cells = " | ".join("n/a" if not np.isfinite(v) else f"{v:+.3f}" for v in values)
            lines.append(f"| {architectures.get(name, name)} | {variant} | " + cells + " |")
        lines.append("")

    # ภาคผนวก — validation และการวิเคราะห์ที่อ้างอิงแกนใดแกนหนึ่งเป็นฐาน ไม่ใช่คำถามหลัก
    # (อันดับของเซลล์) แต่ต้องมีให้ตรวจได้
    lines += ["## ภาคผนวก", ""]
    if "val" in cell_idx.index.get_level_values("scope"):
        lines += [
            "### TSS บน validation",
            "",
            "ใช้เลือก checkpoint และ threshold ของแต่ละเซลล์ — **ไม่ใช่ผลที่รายงาน** "
            "มีไว้ให้เทียบว่าเซลล์ไหน generalise จาก validation ไป test ได้ดีหรือแย่",
            "",
        ]
        lines += _matrix_lines(architectures, matrix_variants, cell_idx, "val", variants)
        lines.append("")

    if "test" in arch_idx.index.get_level_values("scope") and rival_archs:
        lines += [
            f"### ผลต่างเทียบ {anchor_label} (test)",
            "",
            f"ในแต่ละชุด feature เทียบกับ {anchor_label} ของชุดเดียวกัน จับคู่ {pair_label}",
            "",
            "| สถาปัตยกรรม | " + " | ".join(f"{variants[v]['label']}<br>`{v}`" for v in matrix_variants) + " |",
            "|---|" + "---:|" * len(matrix_variants),
        ]
        for name in rival_archs:
            cells = []
            for variant in matrix_variants:
                key = ("test", variant, name)
                cells.append(
                    _pm(arch_idx.loc[key, "delta_mean"], arch_idx.loc[key, "delta_sd"])
                    if key in arch_idx.index else "n/a"
                )
            lines.append(f"| {architectures[name]} | " + " | ".join(cells) + " |")
        lines.append("")

    # ผลต่างตามชุด feature — เทียบภายในสถาปัตยกรรมเดียวกัน ไม่เกี่ยวกับการเทียบข้ามสถาปัตยกรรม
    if "test" in cell_idx.index.get_level_values("scope") and compared:
        lines += [
            "### ผลของชุด feature ภายในสถาปัตยกรรมเดียวกัน (Δ ตามแบบ, test)",
            "",
            f"ในแต่ละสถาปัตยกรรม เทียบกับ `{ANCHOR_VARIANT}` **ของตัวมันเอง** จับคู่ {pair_label}",
            "",
            "| สถาปัตยกรรม | " + " | ".join(f"{variants[v]['label']}<br>`{v}`" for v in compared) + " |",
            "|---|" + "---:|" * len(compared),
        ]
        lines += _grid_rows(architectures, compared, cell_idx, "test", "delta_mean", "delta_sd")
        lines.append("")

    if "test" in inter_idx.index.get_level_values("scope") and rival_archs and compared:
        lines += [
            "### interaction (test)",
            "",
            f"แต่ละช่องคือ **Δ ตามแบบ ของสถาปัตยกรรมนั้น ลบด้วย Δ ตามแบบ ของ {anchor_label}** "
            f"คำนวณจับคู่ระดับ {pair_label} ทุกขั้น · ค่าใกล้ศูนย์ = ชุด feature นั้นให้ผลเท่ากันทุกสถาปัตยกรรม · "
            f"ค่าบวกอย่างมีนัย = สถาปัตยกรรมนั้นใช้ชุด feature นั้นได้ดีกว่า {anchor_label}",
            "",
            "| สถาปัตยกรรม | " + " | ".join(compared) + " |",
            "|---|" + "---:|" * len(compared),
        ]
        for name in rival_archs:
            cells = []
            for variant in compared:
                key = ("test", name, variant)
                cells.append(_pm(*inter_idx.loc[key, ["interaction_mean", "interaction_sd"]]) if key in inter_idx.index else "n/a")
            lines.append(f"| {architectures[name]} | " + " | ".join(cells) + " |")
        lines.append("")

    if extra_idx is not None:
        lines += [
            "### แถวเสริม (ไม่ใช่เซลล์ของตารางหลัก)",
            "",
            "ไม่ถูกจัดอันดับและไม่ถูกนำไปคิด Δ ของตารางหลัก — มีไว้ตอบคำถามเฉพาะเรื่องที่ระบุในแต่ละแถว",
            "",
            "| สถาปัตยกรรม | แบบ | ชุด feature | TSS (mean ± SD) |",
            "|---|---|---|---:|",
        ]
        for (scope, name, variant), row in extra_idx.iterrows():
            if scope != "test":
                continue
            label = architectures.get(name, name)
            spec = variants.get(variant, {"label": variant})
            lines.append(f"| {label} | {variant} | {spec['label']} | {_pm(row['mean'], row['sd'])} |")
        lines.append("")

    lines += [
        "## อ่านตารางนี้อย่างไร",
        "",
        "การทดลองและข้อกำกับส่วนใหญ่เขียนไว้ใน spec ก่อนเห็นผล · **การจัดอันดับเซลล์เป็นคำถามหลักตั้งแต่ "
        "2026-09-24 หลังเห็นผลแล้ว** (เดิมคำถามหลักคือ feature ที่เพิ่มเข้ามาช่วยหรือไม่) ตัวเลขทุกค่าไม่เปลี่ยน "
        "เปลี่ยนเฉพาะวิธีเรียงและอ่าน",
        "",
        f"- **อันดับ 1 บน test เป็นค่าที่มองโลกแง่ดี** — การหยิบค่าสูงสุดจาก {n_cells} เซลล์ที่มี noise "
        "ให้ค่าที่เฟ้อขึ้นเสมอ (winner's curse) ตัวเลขของอันดับ 1 จึงไม่ใช่สิ่งที่คาดได้บนข้อมูลใหม่ · "
        "เซลล์ที่ \"แยกจากอันดับ 1 ไม่ได้\" ต้องถือว่าเสมอกัน ห้ามเล่าว่าอันดับ 1 ชนะ "
        "· คอลัมน์อันดับบน validation บอกว่าถ้าเลือกโดยไม่เห็น test จะได้ตัวไหน",
        "- **\"แยกได้\" ไม่ใช่การทดสอบนัยสำคัญ** — แค่เทียบระยะห่างกับ SD ของระยะห่างนั้นเอง (จับคู่ "
        f"{pair_label}) ซึ่งเป็นเกณฑ์หยาบที่ผ่อนข้างเสมอ · ไม่มีค่า p โดยตั้งใจ — {'fold×seed' if cv else 'seed'} "
        f"{len(runs_table[list(pair_cols)].drop_duplicates())} ค่าให้กำลังทางสถิติไม่พอให้ค่า p มีความหมาย",
        "- **hyperparameter ของแต่ละสถาปัตยกรรมมาจากการค้นหาแยกกัน** ด้วยงบเท่ากันบน 18 SHARP (V0) เท่านั้น — "
        "เซลล์ของชุด feature อื่นใช้ค่าที่ค้นหามาสำหรับ V0 จึงอาจเสียเปรียบเล็กน้อยในการจัดอันดับ และการค้นหา "
        "40 trial เองมีความไม่แน่นอนราว ±0.02 ซึ่งขนาดเท่ากับช่วงของทั้งตาราง "
        "(ดู `docs/adr/0001-tune-per-architecture.md`)",
        "- **V5 (X-ray) คือ baseline ความคึกคักของดวงอาทิตย์ ไม่ใช่คู่แข่งเท่าเทียม** — ค่า X-ray เท่ากันทุก HARP "
        "ณ เวลาเดียวกัน จึงตอบได้แค่ \"ช่วงนี้ดวงอาทิตย์จะปะทุไหม\" ไม่ใช่ \"ดวงไหนจะปะทุ\" "
        "และได้เปรียบผิดปกติในช่วงที่ทั้งดวงคึกคักตลอดต่อเนื่อง",
        f"- SD คือ sample SD (ddof=1) ข้าม {pair_label} · ขอบเขตที่ไม่มี positive แสดง TSS เป็น n/a "
        "(TSS ไม่นิยามเมื่อมี class เดียว)",
        "",
    ]
    return "\n".join(lines)


def aggregate(
    out_dir: Path,
    variants: dict[str, dict],
    architectures: dict[str, str],
    matrix_architectures: list[str],
    matrix_variants: list[str],
    signatures: dict[tuple[int | None, str], dict],
    arrays: StudyArrays,
    epochs: int,
    fold_plans: list[FoldPlan] | None = None,
) -> int:
    """``signatures`` คือ ``{(fold_index, architecture): signature}`` — โหมด single split ใช้
    ``fold_index = None`` ส่วน CV มีหนึ่งคีย์ต่อ fold ที่ผ่านเกณฑ์ คูณกับทุกสถาปัตยกรรม
    (ดู :func:`sunseg.data.cv.plan_folds`)

    ``matrix_architectures`` × ``matrix_variants`` คือเซลล์ของตารางหลัก ส่วนผลที่อยู่นอกนั้น
    ถูกเก็บไว้เป็นแถวเสริม — ไม่เข้าการคำนวณ Δ/interaction เพราะการจับคู่ต้องการเซลล์ครบทุกช่อง
    """
    cv = fold_plans is not None
    arch_order = list(architectures)
    variant_order = list(variants)
    records, ignored = [], 0
    for path in sorted((out_dir / "runs").glob("*.json")):
        record = load_run(path)
        if record is None or record.get("variant") not in variants:
            ignored += 1
            continue
        architecture = record.get("architecture")
        if architecture not in architectures:
            ignored += 1
            continue
        fold_key = record.get("fold")
        expected_sig = signatures.get((fold_key, architecture))
        if expected_sig is None or record.get("signature") != expected_sig:
            ignored += 1
            continue
        if not run_paths(out_dir, architecture, record["variant"], record["seed"], fold_key)[1].exists():
            ignored += 1
            continue
        records.append(record)
    if ignored:
        logger.warning(
            "ไม่นำผล %d ไฟล์มารวม (เทรนด้วย config/dataset/fold อื่น, ไฟล์ไม่ครบ หรือเป็นแบบที่ไม่มีในไฟล์ variants)",
            ignored,
        )
    if not records:
        logger.error("ยังไม่มีผลที่ตรงกับ config + dataset ปัจจุบันให้สรุป")
        return 1

    records.sort(
        key=lambda r: (
            arch_order.index(r["architecture"]), variant_order.index(r["variant"]), r.get("fold") or 0, r["seed"]
        )
    )
    runs_table = pd.DataFrame(
        [
            {
                "architecture": r["architecture"], "variant": r["variant"],
                "fold": r.get("fold"), "seed": r["seed"], **scope,
            }
            for r in records
            for scope in r["scopes"]
        ]
    )
    runs_table.to_parquet(out_dir / "runs.parquet", index=False)
    predictions = pd.concat(
        [
            pd.read_parquet(run_paths(out_dir, r["architecture"], r["variant"], r["seed"], r.get("fold"))[1])
            for r in records
        ],
        ignore_index=True,
    )
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    logger.info("รวมผล %d run -> runs.parquet, predictions.parquet (%d แถว)", len(records), len(predictions))

    pair_cols = ("fold", "seed") if cv else ("seed",)
    in_matrix = runs_table["architecture"].isin(matrix_architectures) & runs_table["variant"].isin(matrix_variants)
    matrix_runs = runs_table[in_matrix]
    if matrix_runs.empty:
        logger.error("ยังไม่มีผลของเซลล์ในตารางหลักเลย (สถาปัตยกรรม %s × แบบ %s)", matrix_architectures, matrix_variants)
        return 1
    try:
        cell_tss = delta_by_variant(matrix_runs, anchor_variant=ANCHOR_VARIANT, metric="tss", pair_cols=pair_cols)
        cell_auc = delta_by_variant(matrix_runs, anchor_variant=ANCHOR_VARIANT, metric="auc", pair_cols=pair_cols)
        delta_arch_tss = delta_by_architecture(
            matrix_runs, anchor_architecture=ANCHOR_ARCHITECTURE, metric="tss", pair_cols=pair_cols
        )
        interaction_tss = interaction_summary(
            matrix_runs, anchor_architecture=ANCHOR_ARCHITECTURE, anchor_variant=ANCHOR_VARIANT,
            metric="tss", pair_cols=pair_cols,
        )
        # คำถามหลัก: อันดับของเซลล์บน test + อันดับบน validation ไว้ตรวจว่าเลือกโดยไม่เห็น test จะได้ตัวเดียวกันไหม
        scopes_present = set(matrix_runs["scope"])
        ranking = (
            rank_cells(matrix_runs, scope="test", metric="tss", pair_cols=pair_cols)
            if "test" in scopes_present else None
        )
        if ranking is not None and "val" in scopes_present:
            val_rank = rank_cells(matrix_runs, scope="val", metric="tss", pair_cols=pair_cols).set_index(
                ["architecture", "variant"]
            )
            keys = list(zip(ranking["architecture"], ranking["variant"], strict=True))
            ranking["val_mean"] = [float(val_rank.loc[k, "mean"]) for k in keys]
            ranking["val_rank"] = [int(val_rank.loc[k, "rank"]) for k in keys]
    except ValueError as exc:
        logger.error(
            "สรุปแบบจับคู่ %s ไม่ได้: %s\n(runs.parquet/predictions.parquet เขียนแล้ว — เทรนส่วนที่ขาดให้ครบ "
            "แล้วรัน --summarise-only)",
            "/".join(pair_cols), exc,
        )
        return 1

    # mean/SD ของทุกเมตริกสำหรับตารางผลหลัก — ไม่ต้องจับคู่ seed เพราะเป็นค่าที่วัดได้ ไม่ใช่ผลต่าง
    cells_by_metric = metric_cells(matrix_runs)

    # แถวเสริมไม่ต้องการเซลล์ครบ จึงสรุปด้วย mean/SD ตรง ๆ ไม่มี Δ
    extra_runs = runs_table[~in_matrix]
    extra_tss = (
        extra_runs.groupby(["scope", "architecture", "variant"], sort=False)["tss"]
        .agg(n_seeds="size", mean="mean", sd=lambda s: s.std(ddof=1))
        .reset_index()
        if not extra_runs.empty
        else pd.DataFrame(columns=["scope", "architecture", "variant", "n_seeds", "mean", "sd"])
    )

    seeds = sorted({r["seed"] for r in records})
    present = [v for v in variant_order if v in set(runs_table["variant"])]
    shown = {v: variants[v] for v in present}
    shown_archs = {a: architectures[a] for a in arch_order if a in set(runs_table["architecture"])}
    matrix_archs_shown = {a: label for a, label in shown_archs.items() if a in matrix_architectures}
    matrix_variants_shown = [v for v in matrix_variants if v in present]

    # เวลาเป็นข้อมูลเสริม — ผลที่เทรนก่อนมีการจับเวลาไม่มีฟิลด์นี้ ต้องไม่ทำให้สรุปตัวชี้วัดล้ม
    timed = [r for r in records if "total_seconds" in r and "prepare_seconds" in r]
    timing, timing_anchor, timing_devices = None, None, sorted({r.get("device", "ไม่ทราบ") for r in timed})
    if timed:
        timing_cols = (
            "architecture", "variant", "fold", "epochs_run", "train_seconds", "total_seconds", "prepare_seconds",
        )
        timing_df = pd.DataFrame([{k: r.get(k) for k in timing_cols} for r in timed])
        # เฉพาะเซลล์ของตารางหลัก: ถ้าปล่อยให้แถวเสริมปนเข้ามา แต่ละสถาปัตยกรรมจะถูกเฉลี่ยบนชุดแบบ
        # คนละชุด (LSTM มี 6 แบบ ตัวอื่นมี 3) แล้วตัวเลข "เทียบกี่เท่า" จะไม่ได้วัดสถาปัตยกรรมอย่างเดียว
        timing_df = timing_df[
            timing_df["architecture"].isin(matrix_architectures) & timing_df["variant"].isin(matrix_variants)
        ]
        try:
            # ตารางเวลาหลักแยกรายสถาปัตยกรรม — บนแกนนี้ต้นทุนต่างกันได้เป็นสิบเท่า
            timing = timing_summary(timing_df, control=ANCHOR_ARCHITECTURE, unit_cols=("architecture",))
            # ส่วนต้นทุนต้นทางผูกกับชุด feature ไม่ใช่สถาปัตยกรรม จึงคิดบนสถาปัตยกรรมอ้างอิงตัวเดียว
            anchor_rows = timing_df[timing_df["architecture"] == ANCHOR_ARCHITECTURE]
            if not anchor_rows.empty:
                timing_anchor = timing_summary(anchor_rows, control=ANCHOR_VARIANT, unit_cols=("variant",))
        except ValueError as exc:
            logger.warning("สรุปเวลาไม่ได้ (ข้ามหัวข้อเวลา): %s", exc)
    else:
        logger.info("ไม่มีรันไหนบันทึกเวลาไว้ — ข้ามหัวข้อเวลา (ใช้ --force เทรนใหม่เพื่อให้มี)")
    untimed_runs = len(records) - len(timed)

    # เวลาต้นทาง (ดาวน์โหลด/สกัด feature) มาจาก measure_acquisition_time.py — ไม่มีไฟล์ก็ข้ามหัวข้อนั้นไป
    acquisition, end_to_end = None, None
    variant_groups = {v: feature_groups(spec["columns"]) for v, spec in shown.items()}
    acquisition_path = out_dir / "acquisition_time.json"
    if timing_anchor is not None and acquisition_path.exists():
        try:
            acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
            end_to_end = end_to_end_hours(
                timing_anchor, acquisition["components"], variant_groups, control=ANCHOR_VARIANT
            )
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.warning("ใช้ %s ไม่ได้ (ข้ามหัวข้อเวลาต้นทาง): %s", acquisition_path.name, exc)
            acquisition, end_to_end = None, None

    balance = None
    fold_summaries = None
    dataset_summary: list[dict] = []
    if cv:
        signed_folds = {fold for fold, _ in signatures}
        valid_plans = [p for p in fold_plans if p.valid and p.index in signed_folds]
        fold_summaries = [
            {"fold": p.index, "bounds": _fold_bounds_dict(p.bounds), "balance": p.balance} for p in valid_plans
        ]
        dataset_summary = [{"fold": fs["fold"], **fs["bounds"], "balance": fs["balance"].to_dict("records")} for fs in fold_summaries]
    else:
        balance = dataset_balance(arrays.meta)
        dataset_summary = balance.to_dict(orient="records")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "anchor": {"architecture": ANCHOR_ARCHITECTURE, "variant": ANCHOR_VARIANT},
        "cv": cv,
        "seeds": seeds,
        "matrix": {"architectures": list(matrix_archs_shown), "variants": matrix_variants_shown},
        "architectures": shown_archs,
        "variants": {v: {"label": s["label"], "role": s["role"], "n_features": len(s["columns"])} for v, s in shown.items()},
        "dataset": dataset_summary,
        "ranking": None if ranking is None else ranking.to_dict(orient="records"),
        "tss": cell_tss.to_dict(orient="records"),
        "auc": cell_auc.to_dict(orient="records"),
        "delta_by_architecture": delta_arch_tss.to_dict(orient="records"),
        "interaction": interaction_tss.to_dict(orient="records"),
        "supplementary": extra_tss.to_dict(orient="records"),
        "timing": None if timing is None else {
            "devices": timing_devices,
            "untimed_runs": untimed_runs,
            "architectures": timing.to_dict(orient="records"),
        },
        "end_to_end": None if end_to_end is None else {
            "generated_at": acquisition.get("generated_at"),
            "gap_minutes": acquisition.get("gap_minutes"),
            "components": acquisition["components"],
            "variants": end_to_end.to_dict(orient="records"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        render_report(
            cell_tss, delta_arch_tss, interaction_tss, cells_by_metric, runs_table,
            shown, matrix_archs_shown, matrix_variants_shown, extra_tss, seeds, epochs,
            balance=balance, fold_summaries=fold_summaries, pair_cols=pair_cols,
            timing=timing, timing_devices=timing_devices, untimed_runs=untimed_runs,
            acquisition=acquisition, end_to_end=end_to_end, variant_groups=variant_groups,
            ranking=ranking,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 70)
    logger.info("%s", SCOPE_TITLES["test"])
    if ranking is not None:
        logger.info("อันดับเซลล์ (TSS เฉลี่ย):")
        for row in ranking.itertuples(index=False):
            logger.info(
                "  %2d. %-14s %-3s TSS %-16s %s",
                row.rank, architectures.get(row.architecture, row.architecture), row.variant,
                _pm(row.mean, row.sd, signed=False),
                "" if row.rank == 1 else ("เสมออันดับ 1" if row.tied_with_best else "ต่ำกว่าอันดับ 1 เกิน SD"),
            )
    for row in cell_tss[cell_tss["scope"] == "test"].itertuples(index=False):
        logger.info(
            "  %-14s %-3s TSS %-16s Δ ตามแบบ %s",
            architectures.get(row.architecture, row.architecture), row.variant, _pm(row.mean, row.sd),
            "— แบบอ้างอิง" if row.variant == ANCHOR_VARIANT else _pm(row.delta_mean, row.delta_sd),
        )
    for row in interaction_tss[interaction_tss["scope"] == "test"].itertuples(index=False):
        if row.architecture == ANCHOR_ARCHITECTURE or row.variant == ANCHOR_VARIANT:
            continue
        logger.info(
            "  interaction %-14s %-3s %s",
            architectures.get(row.architecture, row.architecture), row.variant,
            _pm(row.interaction_mean, row.interaction_sd),
        )
    if timing is not None:
        logger.info("เวลา (ตั้งแต่เริ่มจนได้ผล) · อุปกรณ์ %s", ", ".join(timing_devices))
        for row in timing.itertuples(index=False):
            logger.info(
                "  %-14s ต่อรัน %s วิ · %.3f วิ/epoch · รวมทั้งแถว %.1f นาที (%d รัน)",
                architectures.get(row.architecture, row.architecture),
                _sec(row.total_mean, row.total_sd), row.sec_per_epoch,
                row.start_to_result_seconds / 60, row.n_runs,
            )
    if end_to_end is not None:
        logger.info("เวลาตั้งแต่ต้นทางจนได้ผล (ชม.): ร่วม + intensity + X-ray + เทรน = รวม")
        for row in end_to_end.itertuples(index=False):
            logger.info(
                "  %-3s %s + %s + %s + %s = %s (%.1f× ของ %s)", row.variant, _hours_text(row.shared_hours),
                _hours_text(row.intensity_hours), _hours_text(row.xray_hours), _hours_text(row.train_hours),
                _hours_text(row.total_hours), row.total_ratio, ANCHOR_VARIANT,
            )
    logger.info("รายงานเต็ม: %s", out_dir / "report.md")
    logger.info("=" * 70)
    return 0


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    cv_overrides = {
        key: value
        for key, value in (
            ("expanding", True if args.cv_expanding else None),
            ("anchor", args.cv_anchor),
            ("train_years", args.cv_train_years),
        )
        if value is not None
    }
    if cv_overrides:
        if not args.cv or data_cfg.split.cv is None:
            print("--cv-expanding/--cv-anchor/--cv-train-years ใช้ได้กับ --cv เท่านั้น", file=sys.stderr)
            return 1
        data_cfg.split.cv = data_cfg.split.cv.model_copy(update=cv_overrides)
    # ค่าร่วม train:/loss:/eval: ของ forecast.yaml คือฐานที่ทุกสถาปัตยกรรมตกทอด (resolve_architecture)
    cfg = load_forecast_config(args.forecast_config) if args.forecast_config else load_forecast_config()
    setup_logging(log_file=data_cfg.paths.artifacts / "logs" / "train_study.log")

    default_dir_name = "study_sequences_cv" if args.cv else "study_sequences"
    data_dir = args.data_dir or data_cfg.paths.processed / default_dir_name
    out_dir = args.out_dir or data_cfg.paths.artifacts / STUDY_DIR_NAME

    variants = load_variants(args.variants_file)
    study = load_study_architectures(args.architectures_file)

    chosen_archs = (
        [a.strip() for a in args.architectures.split(",")] if args.architectures else list(study.architectures)
    )
    if unknown := [a for a in chosen_archs if a not in study.architectures]:
        logger.error(
            "ไม่มีสถาปัตยกรรม %s ใน %s — ที่มี: %s", unknown, args.architectures_file, list(study.architectures)
        )
        return 1

    # แผนของแต่ละสถาปัตยกรรม: config ที่รวมค่าฐานแล้ว + รายชื่อแบบที่ต้องรัน
    plans: dict[str, tuple] = {}
    for name in chosen_archs:
        entry = study.architectures[name]
        wanted = entry.variants or study.default_variants
        if args.variants:
            asked = [v.strip() for v in args.variants.split(",")]
            wanted = [v for v in asked if v in wanted]
        plans[name] = (resolve_architecture(cfg, entry), wanted, entry.label or architecture_label(entry.model.kind))

    selected = list(dict.fromkeys(v for _, wanted, _ in plans.values() for v in wanted))
    if not selected:
        logger.error("ไม่มีแบบเหลือให้เทรนหลังกรองด้วย --variants")
        return 1
    if unknown := [v for v in selected if v not in variants]:
        logger.error("ไม่มีแบบ %s ใน %s — แบบที่มี: %s", unknown, args.variants_file, list(variants))
        return 1

    architectures = {name: label for name, (_, _, label) in plans.items()}
    matrix_architectures = [name for name in plans if not study.architectures[name].supplementary]
    matrix_variants = [v for v in study.default_variants if v in selected]
    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        logger.error("--seeds ต้องเป็นตัวเลขคั่นด้วยจุลภาค เช่น 0,1,2 (ได้ %r)", args.seeds)
        return 1
    if not seeds:
        logger.error("ต้องมีอย่างน้อยหนึ่ง seed")
        return 1

    try:
        arrays = load_study_arrays(data_dir)
        for spec in variants.values():  # ตรวจชื่อคอลัมน์ของทุกแบบก่อนเทรนตัวแรก
            select_columns(arrays.features, spec["columns"])
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    epochs = args.epochs or cfg.train.epochs
    out_dir.mkdir(parents=True, exist_ok=True)

    # โหมด single split แทนด้วย "fold เดียว" ที่ fold_idx=None — ทำให้ทั้งลูปเทรนและ aggregate()
    # ใช้โค้ดเดียวกันสำหรับสองโหมด ไม่ต้องแยกสองชุด logic ที่อาจเพี้ยนไม่ตรงกัน
    if args.cv:
        try:
            fold_plans = plan_folds(data_cfg, arrays.meta)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        active_folds: list[FoldPlan | None] = [p for p in fold_plans if p.valid]
        if not active_folds:
            logger.error("ไม่มี fold ไหนผ่านเกณฑ์ขั้นต่ำเลย ดู log ด้านบนสำหรับเหตุผลของแต่ละ fold")
            return 1
    else:
        fold_plans = None
        active_folds = [None]

    # หนึ่ง signature ต่อ (fold, สถาปัตยกรรม) — hyperparameter ต่างกันรายสถาปัตยกรรม ผลของตัวหนึ่ง
    # จึงต้องไม่ถูกนำไปรวมกับอีกตัวแม้อยู่ fold เดียวกัน
    signatures = {
        ((p.index if p is not None else None), name): run_signature(
            plans[name][0], epochs, arrays, p.split_labels if p is not None else arrays.meta["split"]
        )
        for p in active_folds
        for name in plans
    }

    logger.info("=" * 70)
    logger.info(
        "ตารางเปรียบเทียบ: %d สถาปัตยกรรม (%s) × แบบ %s × seed %s · epoch สูงสุด %d",
        len(plans), ", ".join(plans), selected, seeds, epochs,
    )
    logger.info("ตารางหลัก: %s × %s · แถวเสริม: %d เซลล์",
                matrix_architectures, matrix_variants,
                sum(len(w) for _, w, _ in plans.values()) - len(matrix_architectures) * len(matrix_variants))
    logger.info("dataset: %s (%d sample, %d feature ผู้สมัคร)", data_dir, len(arrays.x), len(arrays.features))
    if args.cv:
        logger.info("cross-validation: %d/%d fold ผ่านเกณฑ์", len(active_folds), len(fold_plans))
    logger.info("ผลลัพธ์: %s", out_dir)
    logger.info("=" * 70)

    if not args.summarise_only:
        device = get_device(prefer_cuda=not args.cpu)
        device_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cells = [(name, variant) for name, (_, wanted, _) in plans.items() for variant in wanted]
        total, index, warmed = len(cells) * len(seeds) * len(active_folds), 0, False
        for plan in active_folds:
            fold_idx = plan.index if plan is not None else None
            split_labels = plan.split_labels if plan is not None else arrays.meta["split"]
            if plan is None:
                # โหมด single split เท่านั้น — CV เช็คเกณฑ์นี้ไปแล้วใน plan_folds()
                split_sizes = split_labels.value_counts()
                train_pos = int(arrays.meta.loc[split_labels == "train", "label"].sum())
                if split_sizes.get("val", 0) == 0 or split_sizes.get("test", 0) == 0 or train_pos == 0:
                    logger.error(
                        "ต้องมี val และ test ไม่ว่าง และ train ต้องมี positive (ได้ %s, train positive %d)",
                        split_sizes.to_dict(), train_pos,
                    )
                    return 1
            fold_tag = f"fold {fold_idx}: " if fold_idx is not None else ""

            # เตรียมข้อมูลของทุกแบบไว้ก่อน แล้วเทรน "ทุกแบบ" ต่อ seed หนึ่งก่อนขยับไป seed ถัดไป
            # (ไม่ใช่เทรนแบบหนึ่งจนครบทุก seed แล้วค่อยแบบถัดไป) — ผลของแต่ละรันไม่เปลี่ยนเพราะ
            # แต่ละรันตั้ง seed ของตัวเอง แต่ความเร็วเครื่องที่แกว่งตามเวลา (เครื่องร้อน งานอื่นแย่ง
            # GPU) จะตกกับทุกแบบเท่า ๆ กัน ไม่สะสมที่แบบท้ายลำดับจนดูเหมือนแบบที่ feature มากกว่าช้ากว่า
            prepared: dict[str, tuple[SequenceSplits, float]] = {}
            for variant in selected:
                prepare_started = time.perf_counter()
                splits = (
                    fold_variant_splits(arrays, variants[variant]["columns"], split_labels)
                    if plan is not None
                    else variant_splits(arrays, variants[variant]["columns"])
                )
                prepared[variant] = (splits, time.perf_counter() - prepare_started)

            if not warmed:
                # warm-up ด้วยสถาปัตยกรรมที่แพงที่สุด เพื่อให้ kernel ของทุกตัวถูกโหลดครบก่อนจับเวลาจริง
                warm_up(prepared[selected[0]][0], plans[chosen_archs[-1]][0], device)
                warmed = True

            for seed in seeds:
                for architecture, variant in cells:
                    arch_cfg, _, arch_label = plans[architecture]
                    signature = signatures[(fold_idx, architecture)]
                    spec = variants[variant]
                    splits, prepare_seconds = prepared[variant]
                    fold_meta = splits.meta
                    index += 1
                    json_path, pred_path, model_path = run_paths(out_dir, architecture, variant, seed, fold_idx)
                    existing = load_run(json_path)
                    if (
                        not args.force and existing is not None
                        and existing.get("signature") == signature and pred_path.exists()
                    ):
                        logger.info(
                            "[%d/%d] %s%s %s seed %d: มีผลแล้ว — ข้าม",
                            index, total, fold_tag, arch_label, variant, seed,
                        )
                        continue

                    run_started = time.perf_counter()
                    result = fit_one(splits, arch_cfg, seed, device, epochs)
                    scopes = scope_reports(fold_meta, result["p_val"], result["p_test"], result["threshold"])

                    # เขียน predictions + model ก่อน แล้วค่อย json — json คือเครื่องหมายว่า run นี้เสร็จ
                    pred_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions = long_predictions(
                        fold_meta, architecture, variant, seed,
                        result["p_val"], result["p_test"], result["threshold"],
                    )
                    if fold_idx is not None:
                        # CV: รวมทุก fold เป็นไฟล์เดียวแล้วต้องยังแยกได้ว่าแถวไหนของ fold ไหน (HARP เดียวกัน
                        # เป็น val ของ fold หนึ่งและ test ของอีก fold ได้) — single split คง schema เดิม
                        predictions.insert(3, "fold", fold_idx)
                    predictions.to_parquet(pred_path, index=False)
                    save_checkpoint(
                        model_path,
                        result["model"],
                        config={
                            "architecture": architecture,
                            "variant": variant,
                            "seed": seed,
                            "fold": fold_idx,
                            "features": spec["columns"],
                            "n_features": len(spec["columns"]),
                            "model": arch_cfg.model.model_dump(),
                            "threshold": result["threshold"],
                        },
                        metrics={scope["scope"]: scope for scope in scopes},
                        extra={"norm_mean": splits.stats["mean"], "norm_std": splits.stats["std"]},
                    )
                    # ทั้งรัน: สร้างโมเดล → เทรน → ทำนาย val/test → คำนวณตัวชี้วัด → เขียน predictions +
                    # checkpoint (ไม่รวมการเขียน json ของรันนี้เอง เพราะ json คือที่เก็บค่านี้)
                    total_seconds = time.perf_counter() - run_started
                    record = {
                        "architecture": architecture,
                        "architecture_label": arch_label,
                        "variant": variant,
                        "seed": seed,
                        "fold": fold_idx,
                        "label": spec["label"],
                        "columns": spec["columns"],
                        "threshold": result["threshold"],
                        "val_score": result["val_score"],
                        "best_epoch": result["best_epoch"],
                        "epochs_run": result["epochs_run"],
                        "train_seconds": result["train_seconds"],
                        # prepare_seconds ทำครั้งเดียวต่อแบบ (ต่อ fold) แล้วทุก seed ใช้ร่วมกัน — บันทึก
                        # ซ้ำในทุกรันให้ไฟล์ครบในตัว ตอนสรุปต้องนับครั้งเดียว (timing_summary)
                        "prepare_seconds": prepare_seconds,
                        "total_seconds": total_seconds,
                        "device": device_label,
                        "scopes": scopes,
                        "history": result["history"],
                        "signature": signature,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    json_path.write_text(
                        json.dumps(record, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
                    )

                    by_scope = {scope["scope"]: scope for scope in scopes}
                    logger.info(
                        "[%d/%d] %s%s %s seed %d: val TSS %+.3f · test TSS %s · "
                        "epoch ที่ดีที่สุด %d/%d (เทรน %.0f วิ · ทั้งรัน %.0f วิ)",
                        index, total, fold_tag, arch_label, variant, seed, result["val_score"],
                        _pm(by_scope["test"]["tss"], float("nan")),
                        result["best_epoch"], result["epochs_run"], result["train_seconds"], total_seconds,
                    )

    return aggregate(
        out_dir, variants, architectures, matrix_architectures, matrix_variants,
        signatures, arrays, epochs, fold_plans=fold_plans,
    )


if __name__ == "__main__":
    raise SystemExit(main())
