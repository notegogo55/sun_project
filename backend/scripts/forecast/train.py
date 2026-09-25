"""เทรนโมเดลพยากรณ์ flare (LSTM / TCN / Transformer / DA-RNN) พร้อมเทียบกับ logistic-regression baseline

    python backend/scripts/forecast/train.py --model lstm
    python backend/scripts/forecast/train.py --model tcn,transformer   # หลายตัว เรียงกันทีละตัว
    python backend/scripts/forecast/train.py --model all               # ทุกตัวใน configs/forecast.yaml

hyperparameter ของแต่ละโมเดลอยู่ที่ ``configs/forecast.yaml`` ผลลัพธ์ของโมเดลชื่อ ``<name>``
(ดู ``sunseg.artifacts``) — ไม่แตะไฟล์ของโมเดลอื่น::

    artifacts/models/<name>.pt                    checkpoint ที่หน้าเว็บโหลด
    artifacts/metrics/<name>.json                 ตัวชี้วัด val/test + baseline + history
    artifacts/metrics/<name>_predictions.parquet  ค่าทำนายราย sample (แผง confusion matrix)

**baseline สำคัญมาก**: logistic regression ใช้ SHARP parameters ณ *เวลาเดียว*
(timestep สุดท้ายของหน้าต่าง) ขณะที่โมเดลลำดับเวลาเห็นประวัติ 24 ชั่วโมง ถ้าโมเดลชนะ
baseline ไม่ชัดเจน แปลว่ามิติเวลาไม่ได้ช่วย และควรใช้โมเดลที่ง่ายกว่าแทน
สคริปต์นี้จึงเทรนทั้งสองตัวและรายงานเทียบกันเสมอ

หลายโมเดลในคำสั่งเดียวรันเรียงกันทีละตัว ไม่รันพร้อมกัน — GPU ตัวเดียว และเวลาที่วัดได้ต้องเทียบกันได้
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sunseg.artifacts import ForecastArtifacts  # noqa: E402
from sunseg.config import ForecastConfig, load_data_config, load_forecast_config  # noqa: E402
from sunseg.data.cv import plan_folds  # noqa: E402
from sunseg.datasets.sequence import (  # noqa: E402
    SequenceSplits,
    fold_sequence_splits,
    load_sequence_pool,
    load_sequence_splits,
)
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.metrics import classification_report, format_report  # noqa: E402
from sunseg.models.forecast import architecture_label  # noqa: E402
from sunseg.training.forecast import fit_forecaster, fit_logistic_baseline  # noqa: E402
from sunseg.training.utils import (  # noqa: E402
    get_device,
    save_checkpoint,
    save_metrics,
    save_predictions,
    set_seed,
)

logger = logging.getLogger("train_forecast")


def parse_args(model_names: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", required=True,
        help=f"ชื่อโมเดลใน configs/forecast.yaml คั่นด้วยจุลภาค หรือ all (ที่มี: {', '.join(model_names)})",
    )
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch ใน config")
    p.add_argument("--no-baseline", action="store_true", help="ข้าม logistic regression baseline")
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    p.add_argument(
        "--artifacts-dir", type=Path,
        help="เขียนผลลงที่อื่นแทน artifacts/ (ใช้ตอนทดสอบ — ไม่ทับ checkpoint ที่หน้าเว็บใช้อยู่)",
    )
    p.add_argument(
        "--cv",
        action="store_true",
        help="rolling/blocked-window cross-validation แทน single split เดียว (อ่านพารามิเตอร์ "
        "จาก configs/data.yaml: split.cv) — ต้องใช้ dataset ที่สร้างด้วย "
        "data/build_sequences.py --no-split --out-dir data/processed/sequences_cv "
        "ไม่เขียนทับ checkpoint ของ production เขียนลง artifacts/<ชื่อ>_cv/ แทน",
    )
    return p.parse_args()


def resolve_names(arg: str, available: list[str]) -> list[str]:
    """แปลง ``--model`` เป็นรายชื่อ — ``all`` คือทุกตัวตามลำดับใน forecast.yaml"""
    if arg.strip() == "all":
        return list(available)
    names = [n.strip() for n in arg.split(",") if n.strip()]
    if unknown := [n for n in names if n not in available]:
        raise ValueError(f"ไม่รู้จักโมเดล {unknown} — ที่มีใน configs/forecast.yaml: {available}")
    return names


def build_predictions_frame(
    meta: pd.DataFrame,
    artifacts: ForecastArtifacts,
    model_val_prob: np.ndarray,
    model_test_prob: np.ndarray,
    baseline_val_prob: np.ndarray | None,
    baseline_test_prob: np.ndarray | None,
) -> pd.DataFrame:
    """รวมค่าทำนายราย sample ของ val และ test เป็นตารางเดียว (ไม่เอา train)

    ลำดับแถวต้องตรงกับที่ ``load_sequence_splits`` ใช้สร้าง ``SequenceDataset`` — คือ
    boolean mask ตามคอลัมน์ ``split`` ของ meta โดยไม่ shuffle (ดู ``sequence.py``)
    เช่นเดียวกับ baseline ที่ index จาก ``splits.val``/``splits.test`` ตรง ๆ
    มิฉะนั้นค่าทำนายจะไปจับคู่ผิด sample โดยไม่มี error ให้เห็น
    """
    frames: list[pd.DataFrame] = []
    for name, model_prob, baseline_prob in (
        ("val", model_val_prob, baseline_val_prob),
        ("test", model_test_prob, baseline_test_prob),
    ):
        rows = meta.loc[meta["split"] == name, ["split", "HARPNUM", "noaa_ar", "issue_time", "label"]]
        rows = rows.reset_index(drop=True)
        if len(rows) != len(model_prob):
            raise ValueError(
                f"จำนวนแถว meta ของ split={name} ({len(rows)}) ไม่ตรงกับจำนวนค่าทำนาย "
                f"({len(model_prob)}) — ตรวจสอบว่า meta กับ SequenceDataset ยังเรียงตรงกันอยู่"
            )
        rows = rows.copy()
        rows[artifacts.prob_column] = np.asarray(model_prob, dtype=np.float64)
        rows["baseline_prob"] = (
            np.asarray(baseline_prob, dtype=np.float64) if baseline_prob is not None else np.nan
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def train_and_evaluate(
    splits: SequenceSplits, cfg: ForecastConfig, label: str, epochs: int, device: torch.device, no_baseline: bool
) -> dict:
    """เทรนหนึ่งรอบเต็ม (โมเดล + threshold + baseline) — ใช้ทั้งกับ single split และแต่ละ fold ของ --cv"""
    fit = fit_forecaster(splits, cfg, device, epochs, verbose=True)

    logger.info("-" * 70)
    logger.info("threshold = %.3f  ให้ val %s = %.4f  (เลือกจาก validation — ห้ามใช้ test)",
                fit["threshold"], cfg.eval.primary_metric, fit["val_score"])
    val_report = classification_report(fit["y_val"], fit["p_val"], fit["threshold"])
    test_report = classification_report(fit["y_test"], fit["p_test"], fit["threshold"])

    logger.info("-" * 70)
    logger.info("ผลของ %s บน validation:\n%s", label, format_report(val_report))
    logger.info("ผลของ %s บน test:\n%s", label, format_report(test_report))

    baseline = None if no_baseline else fit_logistic_baseline(splits)
    return {**fit, "val_report": val_report, "test_report": test_report, "baseline": baseline}


def checkpoint_config(name: str, cfg: ForecastConfig, splits: SequenceSplits, threshold: float, data_cfg) -> dict:
    """ส่วน ``config`` ของ checkpoint — พอให้ ``sunseg.inference.forecast`` ประกอบโมเดลกลับได้เอง"""
    return {
        "name": name,
        "kind": cfg.kind,
        "n_features": splits.n_features,
        "seq_len": splits.seq_len,
        "features": splits.features,
        "model": cfg.model.model_dump(mode="json"),
        "threshold": threshold,
        "horizon_hours": data_cfg.flare.horizon_hours,
        "positive_class": data_cfg.flare.positive_goes_class,
    }


def _print_summary(label: str, test_report: dict, baseline: dict | None) -> None:
    logger.info("  %-28s TSS = %+.4f", f"{label} (มีมิติเวลา)", test_report["tss"])
    if baseline is not None:
        gain = test_report["tss"] - baseline["test_report"]["tss"]
        logger.info("  %-28s TSS = %+.4f", "logistic (เวลาเดียว)", baseline["test_report"]["tss"])
        logger.info("  %-28s      %+.4f", "ส่วนต่าง", gain)
        logger.info("")
        if gain > 0.05:
            logger.info("  [ OK ] มิติเวลาช่วยได้จริงอย่างมีนัยสำคัญ")
        elif gain > 0:
            logger.info("  [WARN] %s ดีกว่าเล็กน้อย — พิจารณาว่าคุ้มกับความซับซ้อนหรือไม่", label)
        else:
            logger.info("  [WARN] %s ไม่ชนะ baseline — ลองเพิ่มความยาว sequence หรือลดขนาดโมเดล", label)
    if test_report["tss"] < 0.3:
        logger.warning("\n  TSS ต่ำกว่า 0.3 — ตรวจสอบคุณภาพ label และปริมาณข้อมูล")


def run_single(name: str, cfg: ForecastConfig, label: str, splits: SequenceSplits, data_cfg, args, device,
               artifacts_root: Path) -> None:
    """เทรนบน single split แล้วเขียนไฟล์ทั้งสามของโมเดลนี้"""
    set_seed(cfg.train.seed)
    epochs = args.epochs or cfg.train.epochs
    result = train_and_evaluate(splits, cfg, label, epochs, device, args.no_baseline)
    baseline = result["baseline"]

    artifacts = ForecastArtifacts(name, artifacts_root)
    save_checkpoint(
        artifacts.checkpoint,
        result["model"],
        config=checkpoint_config(name, cfg, splits, result["threshold"], data_cfg),
        metrics={"val": result["val_report"], "test": result["test_report"]},
        extra={"norm_mean": splits.stats["mean"], "norm_std": splits.stats["std"]},
    )
    save_metrics(
        artifacts.metrics,
        {
            # key เป็นชื่อโมเดล — รูปแบบเดียวกับ lstm.json เดิม ({"lstm": {...}, "baseline_logistic": ...})
            name: {"val": result["val_report"], "test": result["test_report"]},
            "kind": cfg.kind,
            "label": label,
            "n_parameters": result["model"].count_parameters(),
            "train_seconds": result["train_seconds"],
            "best_epoch": result["best_epoch"],
            "baseline_logistic": baseline["test_report"] if baseline else None,
            "history": result["history"],
            "n_train": len(splits.train),
            "n_val": len(splits.val),
            "n_test": len(splits.test),
        },
    )
    save_predictions(
        artifacts.predictions,
        build_predictions_frame(
            splits.meta, artifacts, result["p_val"], result["p_test"],
            baseline["val_prob"] if baseline else None, baseline["test_prob"] if baseline else None,
        ),
    )

    logger.info("=" * 70)
    logger.info("สรุป %s (ตัวเลขบน test set)", label)
    logger.info("=" * 70)
    _print_summary(label, result["test_report"], baseline)
    logger.info("=" * 70)


def run_cv(name: str, cfg: ForecastConfig, label: str, data_cfg, args, device, artifacts_root: Path) -> int:
    """rolling/blocked-window cross-validation — เขียนผลลง artifacts/<ชื่อ>_cv/ เท่านั้น
    ไม่แตะ checkpoint ของ production (ดู docstring หัวไฟล์)
    """
    cv_dir = data_cfg.paths.processed / "sequences_cv"
    try:
        pool = load_sequence_pool(cv_dir)
        fold_plans = plan_folds(data_cfg, pool.meta)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    valid_plans = [p for p in fold_plans if p.valid]
    if not valid_plans:
        logger.error("ไม่มี fold ไหนผ่านเกณฑ์ขั้นต่ำเลย ดู log ด้านบนสำหรับเหตุผลของแต่ละ fold")
        return 1

    epochs = args.epochs or cfg.train.epochs
    out_dir = ForecastArtifacts(name, artifacts_root).cv_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for plan in valid_plans:
        logger.info("=" * 70)
        logger.info("[%s] fold %d: train %s..%s · val ..%s · test ..%s", label,
                    plan.index, plan.bounds.train_start, plan.bounds.train_end,
                    plan.bounds.val_end, plan.bounds.test_end)
        logger.info("=" * 70)
        set_seed(cfg.train.seed)
        splits = fold_sequence_splits(pool, plan.split_labels)
        result = train_and_evaluate(splits, cfg, label, epochs, device, args.no_baseline)
        _print_summary(label, result["test_report"], result["baseline"])

        save_checkpoint(
            out_dir / "models" / f"{name}_fold{plan.index}.pt",
            result["model"],
            config={"fold": plan.index, **checkpoint_config(name, cfg, splits, result["threshold"], data_cfg)},
            metrics={"val": result["val_report"], "test": result["test_report"]},
            extra={"norm_mean": splits.stats["mean"], "norm_std": splits.stats["std"]},
        )
        fold_results.append({"fold": plan.index, **result})

    test_tss = np.array([r["test_report"]["tss"] for r in fold_results], dtype=float)
    baseline_tss = np.array(
        [r["baseline"]["test_report"]["tss"] if r["baseline"] else np.nan for r in fold_results], dtype=float
    )
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "kind": cfg.kind,
        "n_folds": len(fold_results),
        "folds": [
            {
                "fold": r["fold"],
                "val": r["val_report"],
                "test": r["test_report"],
                "baseline_test": r["baseline"]["test_report"] if r["baseline"] else None,
            }
            for r in fold_results
        ],
        "test_tss_mean": float(np.nanmean(test_tss)),
        "test_tss_sd": float(np.nanstd(test_tss, ddof=1)) if len(test_tss) > 1 else float("nan"),
        "baseline_test_tss_mean": float(np.nanmean(baseline_tss)) if not args.no_baseline else None,
    }
    save_metrics(out_dir / "cv_summary.json", summary)

    logger.info("=" * 70)
    logger.info("สรุป cross-validation ของ %s (%d fold, ตัวเลขบน test ของแต่ละ fold)", label, len(fold_results))
    logger.info("  %s TSS mean ± SD = %+.4f ± %.4f", label, summary["test_tss_mean"], summary["test_tss_sd"])
    if not args.no_baseline:
        logger.info("  logistic baseline TSS mean = %+.4f", summary["baseline_test_tss_mean"])
    logger.info("รายละเอียดราย fold: %s", out_dir / "cv_summary.json")
    logger.info("=" * 70)
    return 0


def main() -> int:
    forecast_cfg = load_forecast_config()
    args = parse_args(forecast_cfg.names)
    data_cfg = load_data_config()
    try:
        names = resolve_names(args.model, forecast_cfg.names)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    artifacts_root = args.artifacts_dir or data_cfg.paths.artifacts
    # log แยกไฟล์ต่อโมเดล (train_lstm.log ของเดิมจึงยังต่อท้ายไฟล์เดิม) — หลายตัวใช้ไฟล์ของตัวแรก
    setup_logging(log_file=artifacts_root / "logs" / f"train_{names[0]}.log")
    device = get_device(prefer_cuda=not args.cpu)

    splits = None
    if not args.cv:
        splits = load_sequence_splits(data_cfg.paths.processed / "sequences")
        if len(splits.val) == 0 or len(splits.test) == 0:
            logger.error(
                "val หรือ test ว่างเปล่า — ต้องมีข้อมูลครบทั้งสามชุดจึงจะเทรนได้อย่างถูกต้อง\n"
                "ตรวจสอบเส้นแบ่งใน configs/data.yaml (split.train_end / split.val_end)"
            )
            return 1
        if splits.train.positive_rate == 0:
            logger.error("train ไม่มี positive เลย — ตรวจสอบขั้นตอนสร้าง label")
            return 1

    for name in names:  # ทีละตัว ไม่รันพร้อมกัน
        cfg = forecast_cfg.resolve(name)
        label = forecast_cfg.models[name].label or architecture_label(cfg.kind)
        logger.info("=" * 70)
        logger.info("เทรน %s (%s) พยากรณ์ flare >= %s ใน %d ชม.",
                    label, name, data_cfg.flare.positive_goes_class, data_cfg.flare.horizon_hours)
        logger.info("=" * 70)

        if args.cv:
            if code := run_cv(name, cfg, label, data_cfg, args, device, artifacts_root):
                return code
        else:
            run_single(name, cfg, label, splits, data_cfg, args, device, artifacts_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
