"""ค้นหา hyperparameter ของแต่ละสถาปัตยกรรมด้วย Optuna โดยประเมินบน cross-validation

    # ต้องมี data/processed/study_sequences_cv/ ก่อน (study/build_dataset.py --no-split)
    python backend/scripts/study/tune.py --architecture lstm

    # ทั้งสี่ตัวเรียงกันทีละตัว (ไม่ซ้อนกัน) — คำสั่งของ ticket 02
    python backend/scripts/study/tune.py --architecture lstm,tcn,transformer,darnn

    # รอบทดสอบเร็ว
    python backend/scripts/study/tune.py --architecture tcn --n-trials 2 --out-dir <tmp>

ครอบทุกสถาปัตยกรรมในตารางเปรียบเทียบ (แทน ``tune_lstm.py`` เดิมที่ค้นหาได้แค่ LSTM — ขอบเขตของ
LSTM ที่นี่เหมือนเดิมทุกตัว บวกตัวเลือก mean pooling) กติกาที่
``docs/adr/0001-tune-per-architecture.md`` บังคับ และสคริปต์นี้รักษาไว้โดยโครงสร้าง:

- **งบเท่ากันทุกสถาปัตยกรรม** — :data:`N_TRIALS`, :data:`N_SEEDS` และ
  :data:`RISK_AVERSION` เป็นค่าคงที่ระดับโมดูล ไม่ใช่ตัวเลือกรายตัว และทุกตัววัดบน
  fold ชุดเดียวกัน ถ้าตัวหนึ่งได้ค้นหามากกว่าอีกตัว ข้อสรุปว่า "ใครแพ้" จะพิสูจน์ผิดไม่ได้
- **ค้นหาบนชุด 18 SHARP เท่านั้น** — ชุดที่เป็นกลาง ไม่ใช่ชุดที่คาดว่าจะชนะ ไม่งั้นจะเอาเปรียบ
  แบบนั้นตอนกลับไปเทียบกับแบบอื่น
- **objective ใช้ val TSS เท่านั้น ห้ามใช้ test** — ``fit_one`` คืนทั้ง ``p_test`` มาด้วย แต่
  สคริปต์นี้ไม่แตะมันเลย และการค้นหาเกิดบน fold ที่จบที่ test 2024 ส่วนตารางหลักรายงานบน
  single split ที่ test คือปี 2025 การแยกจึงสมบูรณ์ ไม่ใช่แค่ "ไม่ได้ใส่ test ใน objective"
- **ทุกสถาปัตยกรรมค้นหา pooling ของตัวเองได้** — ถ้า LSTM ตัวเดียวถูกอนุญาตให้ทิ้ง attention
  (ซึ่งเป็นสิ่งที่การค้นหารอบแรกเลือกจริง) แต่ตัวอื่นถูกบังคับให้ใช้ การเปรียบเทียบจะเอนไปทางเดียว

ค้นหาต่อจากที่ค้างได้: Optuna เก็บ trial ลง sqlite หนึ่งไฟล์ต่อสถาปัตยกรรม รันคำสั่งเดิมซ้ำ
จะทำ trial ที่ยังขาดให้ครบตามงบ

**หลังค้นหาเสร็จต้องผ่านด่านตรวจการถ่ายทอดก่อน** (ticket 02) — hyperparameter ถูกเลือกบน
train ที่เล็กกว่าของจริง 3-8 เท่า หลักฐานที่มีอยู่ยืนยันแค่การถ่ายทอด fold -> fold เท่านั้น
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import optuna  # noqa: E402
import yaml  # noqa: E402

from sunseg.config import (  # noqa: E402
    STUDY_VARIANTS_FILE,
    ArchitectureEntry,
    DARNNArchConfig,
    LSTMArchConfig,
    TCNArchConfig,
    TransformerArchConfig,
    load_data_config,
    load_forecast_config,
    load_study_architectures,
    resolve_architecture,
)
from sunseg.data.cv import plan_folds  # noqa: E402
from sunseg.data.study_dataset import load_variants, select_columns  # noqa: E402
from sunseg.datasets.study import fold_variant_splits, load_study_arrays  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.models.forecast import ARCHITECTURES, POOLING_KINDS, build_architecture  # noqa: E402
from sunseg.training.forecast import fit_one  # noqa: E402 — ลูปเทรนเดียวกับตารางหลักโดยตั้งใจ
from sunseg.training.utils import get_device  # noqa: E402

logger = logging.getLogger("tune_architectures")

#: งบการค้นหา — เท่ากันทุกสถาปัตยกรรมโดยโครงสร้าง ไม่เปิดเป็นตัวเลือกรายตัว (ADR 0001)
N_TRIALS = 40
N_SEEDS = 3
RISK_AVERSION = 0.5

#: ชุด feature ที่ใช้ค้นหา — ชุดที่เป็นกลาง
SEARCH_VARIANT = "V0"

DEFAULT_VARIANTS_FILE = STUDY_VARIANTS_FILE

#: รูปทรงของ TCN ที่ receptive field คลุมหน้าต่างของ dataset งานนี้ (8 timestep) ทุกตัวเลือก
#:
#: RF = 1 + Σ (kernel−1)·dilation · dilation เป็นกำลังของสอง ตัวเลือกที่ RF ไม่ถึงถูกตัดออก
#: **ตั้งแต่นิยามขอบเขต** ไม่ใช่ตัดทิ้งตอน trial — ไม่งั้นงบที่ใช้จริงของ TCN จะน้อยกว่าตัวอื่น
#:
#: **แก้ไข 2026-09-22 (รอบสอง)** — ชุดแรกออกแบบผิดขนาด: ผมอ่าน ``sequence.length: 24`` จาก
#: configs/data.yaml แล้วสรุปว่าหน้าต่างยาว 24 timestep แต่ ``study/build_dataset.py`` ตั้ง
#: ``STUDY_CADENCE_HOURS = 12`` / ``STUDY_SEQUENCE_LENGTH = 8`` (ตอนนี้อยู่ใน ``sunseg.data.study_dataset``) ทับไว้โดยตั้งใจ (บล็อก
#: ``sequence`` ใน data.yaml เป็นของ pipeline production คนละตัว) ชุดแรกจึงมีแต่รูปทรงที่ RF
#: 31-63 คือใหญ่กว่าหน้าต่างจริง 4-8 เท่า และ TCN ไม่เคยได้ลองรูปทรงที่พอดีกับข้อมูลเลย
#:
#: ชุดนี้ไล่ตั้งแต่พอดีเป๊ะ (RF 8) ไปจนถึงเกินสองเท่า (RF 16) และคง ``k2_d5`` ของรอบแรกไว้
#: เพื่อให้การค้นหารอบใหม่เลือกกลับมาได้ถ้ามันดีจริง — เทียบกันได้ตรง ๆ
TCN_SHAPES: dict[str, tuple[int, tuple[int, ...]]] = {
    "k2_d3": (2, (1, 2, 4)),          # RF 8  — พอดีหน้าต่าง
    "k3_d3": (3, (1, 2, 4)),          # RF 15
    "k2_d4": (2, (1, 2, 4, 8)),       # RF 16
    "k2_d5": (2, (1, 2, 4, 8, 16)),   # RF 32 — ตัวที่ชนะรอบแรก เก็บไว้ให้เทียบ
}


# --------------------------------------------------------------------------- #
# ขอบเขตการค้นหา
# --------------------------------------------------------------------------- #


def _sample_shared(trial: optuna.Trial) -> tuple[float, str, dict, dict]:
    """ค่าที่ทุกสถาปัตยกรรมค้นหาเหมือนกัน — dropout, pooling และตารางการเทรน

    ค่าที่ ``TrainOverride`` ไม่เปิดให้ทับ (batch_size, scheduler, warmup, grad_clip)
    ไม่อยู่ที่นี่โดยตั้งใจ: ต้องเหมือนกันทุกแถวของตาราง ไม่งั้นแยกไม่ออกว่าผลต่างมาจาก
    สถาปัตยกรรมหรือจากตารางการเทรนที่ไม่เหมือนกัน
    """
    dropout = trial.suggest_float("dropout", 0.1, 0.6)
    pooling = trial.suggest_categorical("pooling", list(POOLING_KINDS))
    train = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True),
        "early_stop_patience": trial.suggest_categorical("early_stop_patience", [8, 12, 16]),
    }
    loss = {
        "focal_alpha": trial.suggest_float("focal_alpha", 0.5, 0.9),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 3.0),
    }
    return dropout, pooling, train, loss


def sample_entry(trial: optuna.Trial, kind: str) -> ArchitectureEntry:
    """สุ่ม hyperparameter หนึ่งชุดของสถาปัตยกรรมหนึ่ง แล้วประกอบเป็นรายการ config

    ทุกจุดในขอบเขตต้องสร้างโมเดลที่ประกอบได้จริง — ตัวเลือกที่ทำให้ประกอบไม่ได้
    (receptive field ไม่พอ, d_model หารจำนวน head ไม่ลงตัว) ถูกตัดออกตั้งแต่นิยามขอบเขต
    """
    dropout, pooling, train, loss = _sample_shared(trial)
    common = {"dropout": dropout, "pooling": pooling}

    if kind == "lstm":
        model = LSTMArchConfig(
            kind="lstm",
            hidden_size=trial.suggest_categorical("hidden_size", [16, 24, 32, 48, 64]),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            **common,
        )
    elif kind == "tcn":
        kernel_size, dilations = TCN_SHAPES[trial.suggest_categorical("tcn_shape", list(TCN_SHAPES))]
        model = TCNArchConfig(
            kind="tcn",
            channels=trial.suggest_categorical("channels", [16, 24, 32, 40]),
            kernel_size=kernel_size,
            dilations=dilations,
            **common,
        )
    elif kind == "transformer":
        # d_model ทุกตัวเลือกหารด้วย 4 ลงตัว จึงใช้ได้กับทั้ง 2 และ 4 head
        model = TransformerArchConfig(
            kind="transformer",
            d_model=trial.suggest_categorical("d_model", [16, 24, 32, 48]),
            n_heads=trial.suggest_categorical("n_heads", [2, 4]),
            ff_dim=trial.suggest_categorical("ff_dim", [16, 32, 64]),
            n_layers=trial.suggest_int("n_layers", 1, 2),
            **common,
        )
    elif kind == "darnn":
        model = DARNNArchConfig(
            kind="darnn",
            hidden_size=trial.suggest_categorical("hidden_size", [16, 24, 32, 48]),
            **common,
        )
    else:
        raise ValueError(f"สถาปัตยกรรม {kind!r} ไม่รู้จัก (ที่มี: {sorted(ARCHITECTURES)})")

    return ArchitectureEntry(model=model, train=train, loss=loss)


# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--architecture", required=True,
        help=f"สถาปัตยกรรมที่จะค้นหา คั่นด้วยจุลภาคเพื่อทำหลายตัวเรียงกัน (ที่มี: {','.join(sorted(ARCHITECTURES))})",
    )
    p.add_argument("--data-dir", type=Path, help="pool ของงานศึกษา (ปริยาย: data/processed/study_sequences_cv)")
    p.add_argument("--out-dir", type=Path, help="ที่เก็บผล (ปริยาย: artifacts/architecture_tuning)")
    p.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS_FILE)
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch สูงสุดต่อการเทรนหนึ่งรอบ")
    p.add_argument(
        "--n-trials", type=int, default=N_TRIALS,
        help=f"ทับงบ trial (ปริยาย {N_TRIALS}) — ใช้ตอนทดสอบเท่านั้น งานจริงต้องเท่ากันทุกตัว",
    )
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    return p.parse_args()


def tune_one(
    kind: str, fold_splits: list, fold_indices: list[int], base_cfg, out_dir: Path,
    device, epochs: int, n_trials: int, existing: ArchitectureEntry | None = None,
) -> dict:
    """ค้นหาของสถาปัตยกรรมหนึ่งจนครบงบ แล้วคืนสรุปของ trial ที่ดีที่สุด

    ``existing`` คือรายการเดิมใน ``configs/study/architectures.yaml`` — ``label``,
    ``variants`` และ ``supplementary`` ถูกยกมาใส่ส่วน config ที่คายออกไป เพราะการค้นหา
    ไม่เกี่ยวกับสามค่านั้น ถ้าไม่ยกมา การวางทับทั้งบล็อกจะทำให้แถวเสริมของ LSTM
    (V1/V4/V5) หายไปเงียบ ๆ
    """
    seeds = list(range(N_SEEDS))

    def objective(trial: optuna.Trial) -> float:
        entry = sample_entry(trial, kind)
        cfg = resolve_architecture(base_cfg, entry)
        scores: list[float] = []
        for fold_idx, splits in zip(fold_indices, fold_splits, strict=True):
            for seed in seeds:
                # ใช้เฉพาะ val_score — p_test ที่ fit_one คืนมาไม่ถูกแตะเลย
                result = fit_one(splits, cfg, seed, device, epochs)
                scores.append(result["val_score"])
                trial.set_user_attr(f"fold{fold_idx}_seed{seed}_val_tss", result["val_score"])

        values = np.asarray(scores, dtype=np.float64)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        trial.set_user_attr("val_tss_mean", mean)
        trial.set_user_attr("val_tss_sd", sd)
        return mean - RISK_AVERSION * sd

    study = optuna.create_study(
        study_name=f"tune_{kind}",
        storage=f"sqlite:///{out_dir / f'{kind}.db'}",
        direction="maximize",
        load_if_exists=True,
    )
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    logger.info("[%s] มี trial เสร็จแล้ว %d จากงบ %d — ทำต่ออีก %d", kind, done, n_trials, remaining)

    started = time.time()
    if remaining:
        study.optimize(objective, n_trials=remaining)
    elapsed = time.time() - started

    best = study.best_trial
    entry = sample_entry(optuna.trial.FixedTrial(best.params), kind)
    if existing is not None:
        entry = entry.model_copy(update={
            "label": existing.label,
            "variants": existing.variants,
            "supplementary": existing.supplementary,
        })
    cfg = resolve_architecture(base_cfg, entry)
    n_params = _count_params(cfg, fold_splits[0])

    logger.info("=" * 70)
    logger.info(
        "[%s] trial ที่ดีที่สุด #%d · objective %.4f · val TSS %.4f ± %.4f · %d พารามิเตอร์",
        kind, best.number, best.value, best.user_attrs["val_tss_mean"], best.user_attrs["val_tss_sd"], n_params,
    )
    for key, value in sorted(best.params.items()):
        logger.info("    %-22s %s", key, value)
    logger.info("ใช้เวลาค้นหารอบนี้ %.0f วิ (%.2f ชม.)", elapsed, elapsed / 3600)
    logger.info("=" * 70)

    return {
        "architecture": kind,
        "n_trials": n_trials,
        "n_seeds": N_SEEDS,
        "risk_aversion": RISK_AVERSION,
        "variant": SEARCH_VARIANT,
        "folds": fold_indices,
        "epochs": epochs,
        "best_trial": best.number,
        "objective": best.value,
        "val_tss_mean": best.user_attrs["val_tss_mean"],
        "val_tss_sd": best.user_attrs["val_tss_sd"],
        "n_parameters": n_params,
        "search_seconds": elapsed,
        "params": best.params,
        "entry": entry.model_dump(mode="json", exclude_none=True),
    }


def _count_params(cfg, splits) -> int:
    model = build_architecture(cfg.kind, splits.n_features, splits.seq_len, cfg.model)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    base_cfg = load_forecast_config()  # ค่าร่วม train:/loss:/eval: คือฐานของทุกสถาปัตยกรรม
    setup_logging(log_file=data_cfg.paths.artifacts / "logs" / "tune_architectures.log")

    kinds = [k.strip() for k in args.architecture.split(",") if k.strip()]
    if unknown := [k for k in kinds if k not in ARCHITECTURES]:
        logger.error("สถาปัตยกรรม %s ไม่รู้จัก — ที่มี: %s", unknown, sorted(ARCHITECTURES))
        return 1

    data_dir = args.data_dir or data_cfg.paths.processed / "study_sequences_cv"
    out_dir = args.out_dir or data_cfg.paths.artifacts / "architecture_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variants(args.variants_file)
    if SEARCH_VARIANT not in variants:
        logger.error("ไม่มีแบบ %s ใน %s", SEARCH_VARIANT, args.variants_file)
        return 1
    columns = variants[SEARCH_VARIANT]["columns"]

    try:
        arrays = load_study_arrays(data_dir)
        select_columns(arrays.features, columns)
        fold_plans = plan_folds(data_cfg, arrays.meta)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 1

    valid_folds = [p for p in fold_plans if p.valid]
    if not valid_folds:
        logger.error("ไม่มี fold ไหนผ่านเกณฑ์ขั้นต่ำเลย ดู log ด้านบนสำหรับเหตุผลของแต่ละ fold")
        return 1

    epochs = args.epochs or base_cfg.train.epochs
    device = get_device(prefer_cuda=not args.cpu)
    fold_indices = [p.index for p in valid_folds]
    fold_splits = [fold_variant_splits(arrays, columns, p.split_labels) for p in valid_folds]

    logger.info("=" * 70)
    logger.info("ค้นหา hyperparameter: %s", ", ".join(kinds))
    logger.info(
        "งบเท่ากันทุกตัว: %d trial × %d seed × %d fold = %d ครั้งที่เทรน ต่อสถาปัตยกรรม",
        args.n_trials, N_SEEDS, len(valid_folds), args.n_trials * N_SEEDS * len(valid_folds),
    )
    logger.info(
        "ค้นหาบนแบบ %s (%d feature) · fold %s · epoch สูงสุด %d · risk_aversion %.2f",
        SEARCH_VARIANT, len(columns), fold_indices, epochs, RISK_AVERSION,
    )
    logger.info("objective = mean(val TSS) − %.2f × SD · ไม่มีเส้นทางใดอ่าน test", RISK_AVERSION)
    logger.info("=" * 70)

    study_cfg = load_study_architectures()
    existing_by_kind = {e.model.kind: e for e in study_cfg.architectures.values() if not e.supplementary}

    results = []
    for kind in kinds:  # ทีละตัว ไม่ซ้อนกัน — เวลาที่วัดได้ต้องเทียบกันได้
        results.append(tune_one(
            kind, fold_splits, fold_indices, base_cfg, out_dir, device, epochs, args.n_trials,
            existing=existing_by_kind.get(kind),
        ))
        (out_dir / f"{kind}_best.json").write_text(
            json.dumps(results[-1], indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ส่วนของ config ที่วางลง configs/study/architectures.yaml ได้ทันที
    fragment = {"architectures": {r["architecture"]: r["entry"] for r in results}}
    fragment_path = out_dir / "tuned_architectures.yaml"
    fragment_path.write_text(
        yaml.safe_dump(fragment, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    logger.info("=" * 70)
    logger.info("%-14s %-10s %-18s %10s %12s", "สถาปัตยกรรม", "objective", "val TSS", "พารามิเตอร์", "เวลา (ชม.)")
    for r in results:
        logger.info(
            "%-14s %-10.4f %.4f ± %.4f %10d %12.2f",
            r["architecture"], r["objective"], r["val_tss_mean"], r["val_tss_sd"],
            r["n_parameters"], r["search_seconds"] / 3600,
        )
    logger.info("=" * 70)
    logger.info("ส่วนของ config ที่ได้ (ตรวจก่อนนำไปใช้): %s", fragment_path)
    logger.info(
        "ขั้นถัดไป (ticket 02): วางค่าลง configs/study/architectures.yaml แล้ว**ตรวจการถ่ายทอด**บน "
        "single split ด้วย val 2024 ก่อนเริ่มตารางหลัก — ห้ามแตะ test"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
