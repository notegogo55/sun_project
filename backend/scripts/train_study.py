"""เทรน 5 แบบ × หลาย seed ของงานเปรียบเทียบชุด feature แล้วสรุปผล (ticket 07 ของ
``.scratch/lstm-feature-ablation/``)

    # ต้องมี data/processed/study_sequences/ ก่อน (backend/scripts/build_study_dataset.py)
    python backend/scripts/train_study.py

    # ทดสอบเร็ว — บางแบบ บาง seed เขียนลงที่อื่น
    python backend/scripts/train_study.py --variants V0,V5 --seeds 0 --epochs 3 --out-dir <tmp>

    # รวมผลที่เทรนไว้แล้วเป็นตาราง/รายงานใหม่ โดยไม่เทรนเพิ่ม
    python backend/scripts/train_study.py --summarise-only

กติกาที่ spec บังคับ และสคริปต์นี้รักษาไว้โดยโครงสร้าง:

- **ทุกแบบใช้แถวชุดเดียวกัน** — โหลด ``X`` ก้อนเดียวแล้วหั่นคอลัมน์ตามชื่อ
  (``sunseg.datasets.study.variant_splits``) ไม่มีแบบไหนได้แถวต่างจากแบบอื่น
- **hyperparameter ชุดเดียว** — ทุกแบบอ่าน ``configs/lstm.yaml`` ชุดเดียวกับ production ไม่มีช่อง
  ให้ส่งค่ารายแบบ และใช้ ``train_one_epoch``/``evaluate`` ตัวเดียวกับ ``train_lstm.py``
- **threshold ของใครของมัน** — เลือกบน val ของแต่ละ (แบบ, seed) แล้ว freeze ก่อนแตะ test
- **seed ชุดเดียวกันทุกแบบ** — สรุปแบบจับคู่ seed (``sunseg.study_summary``) ซึ่ง error ถ้าคู่ไม่ครบ

รันต่อได้: แต่ละ (แบบ, seed) เขียนผลของตัวเองลง ``runs/`` ทันทีที่เสร็จ รอบถัดไปข้ามตัวที่มีแล้ว
และยังตรงกับ config + dataset ปัจจุบัน (``signature`` ในไฟล์) — ผลที่เทรนด้วย config หรือ dataset
อื่น (เช่นรอบทดสอบ ``--epochs 3``) ไม่ถูกนำมารวมเงียบ ๆ ใช้ ``--force`` เพื่อเทรนใหม่

artifacts ทั้งหมดอยู่ใต้ ``artifacts/lstm_feature_ablation/`` — ไม่แตะ ``lstm.pt``/
``predictions.parquet`` ของ production
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from train_lstm import evaluate, train_one_epoch  # noqa: E402 — loop เดียวกับ production โดยตั้งใจ

from sunseg.config import load_data_config, load_lstm_config  # noqa: E402
from sunseg.data.study_dataset import load_variants, select_columns  # noqa: E402
from sunseg.datasets.sequence import SequenceSplits, make_loaders  # noqa: E402
from sunseg.datasets.study import StudyArrays, load_study_arrays, variant_splits  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.losses import build_forecast_loss  # noqa: E402
from sunseg.metrics import (  # noqa: E402
    classification_report,
    confusion_counts,
    find_best_threshold,
)
from sunseg.models.lstm import build_lstm  # noqa: E402
from sunseg.study_summary import paired_summary, seed_table  # noqa: E402
from sunseg.train_utils import (  # noqa: E402
    EarlyStopping,
    build_scheduler,
    get_device,
    save_checkpoint,
    set_seed,
)

logger = logging.getLogger("train_study")

CONTROL = "V0"
DEFAULT_SEEDS = ",".join(str(s) for s in range(35))
STUDY_DIR_NAME = "lstm_feature_ablation"
DEFAULT_VARIANTS_FILE = SCRIPTS_DIR.parent / "configs" / "study_variants.yaml"

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
    p.add_argument("--out-dir", type=Path, help="ที่เก็บผล (ปริยาย: artifacts/lstm_feature_ablation)")
    p.add_argument("--variants-file", type=Path, default=DEFAULT_VARIANTS_FILE)
    p.add_argument("--variants", help="เทรนเฉพาะบางแบบ คั่นด้วยจุลภาค เช่น V0,V5 (ปริยาย: ทุกแบบในไฟล์)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS, help=f"seed ที่ใช้กับทุกแบบ (ปริยาย {DEFAULT_SEEDS})")
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch ใน config (ใช้ตอนทดสอบ)")
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    p.add_argument("--force", action="store_true", help="เทรนใหม่แม้มีผลของ (แบบ, seed) นั้นอยู่แล้ว")
    p.add_argument("--summarise-only", action="store_true", help="ไม่เทรน แค่รวมผลที่มีเป็นตาราง/รายงาน")
    return p.parse_args()


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"แปลง {type(obj).__name__} เป็น JSON ไม่ได้")


def run_signature(cfg, epochs: int, arrays: StudyArrays) -> dict:
    """สิ่งที่ต้องเหมือนกันจึงจะเอาผลของ run เก่ามารวมกับ run ใหม่ได้"""
    rows = pd.util.hash_pandas_object(arrays.meta[["HARPNUM", "issue_time", "split"]], index=False)
    signature = {
        "lstm": cfg.model_dump(mode="json"),
        "epochs": int(epochs),
        "dataset": {
            "n_samples": int(len(arrays.x)),
            "features": arrays.features,
            "rows_sha1": hashlib.sha1(rows.to_numpy().tobytes()).hexdigest()[:16],
        },
    }
    return json.loads(json.dumps(signature))


# --------------------------------------------------------------------------- #
# การเทรนหนึ่ง (แบบ, seed) — ขั้นตอนเดียวกับ main() ของ train_lstm.py
# --------------------------------------------------------------------------- #


def fit_one(splits: SequenceSplits, cfg, seed: int, device: torch.device, epochs: int) -> dict:
    set_seed(seed, deterministic=True)  # ต้องมาก่อนสร้างโมเดลและ loader (ลำดับ shuffle ใช้ RNG เดียวกัน)
    model = build_lstm(splits.n_features, cfg.model).to(device)
    criterion = build_forecast_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.train, epochs)
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")
    train_loader, val_loader, test_loader = make_loaders(splits, cfg.train.batch_size)

    history: list[dict] = []
    started = time.time()
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.train.grad_clip)
        y_val, p_val = evaluate(model, val_loader, device)
        _, val_tss = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)
        if scheduler is not None:
            scheduler.step(val_tss) if cfg.train.scheduler == "plateau" else scheduler.step()
        history.append({"epoch": epoch + 1, "loss": float(loss), "val_tss": float(val_tss)})
        if stopper.step(val_tss, epoch, model):
            break
    stopper.restore(model)
    seconds = time.time() - started

    # threshold เลือกบน val แล้ว freeze — test ถูกแตะหลังจากนี้เท่านั้น
    y_val, p_val = evaluate(model, val_loader, device)
    threshold, val_score = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)
    _, p_test = evaluate(model, test_loader, device)
    return {
        "model": model,
        "threshold": float(threshold),
        "val_score": float(val_score),
        "p_val": np.asarray(p_val, dtype=np.float64),
        "p_test": np.asarray(p_test, dtype=np.float64),
        "best_epoch": int(stopper.best_epoch + 1),
        "epochs_run": len(history),
        "train_seconds": float(seconds),
        "history": history,
    }


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
    meta: pd.DataFrame, variant: str, seed: int, p_val: np.ndarray, p_test: np.ndarray, threshold: float
) -> pd.DataFrame:
    """หนึ่งแถวต่อ (แบบ, seed, split, HARPNUM, issue_time) — รูปแบบยาวตาม spec"""
    columns = [c for c in ("split", "HARPNUM", "noaa_ar", "issue_time", "label") if c in meta.columns]
    frames = []
    for split, prob in (("val", p_val), ("test", p_test)):
        rows = meta.loc[meta["split"] == split, columns].reset_index(drop=True).copy()
        rows.insert(0, "seed", seed)
        rows.insert(0, "variant", variant)
        rows["prob"] = np.asarray(prob, dtype=np.float64)
        rows["threshold"] = float(threshold)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def run_paths(out_dir: Path, variant: str, seed: int) -> tuple[Path, Path, Path]:
    stem = f"{variant}_seed{seed}"
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


def render_report(
    summary_tss: pd.DataFrame,
    summary_auc: pd.DataFrame,
    runs_table: pd.DataFrame,
    variants: dict[str, dict],
    balance: pd.DataFrame,
    seeds: list[int],
    epochs: int,
) -> str:
    compared = [v for v in variants if v != CONTROL]
    lines = [
        f"# ผลเปรียบเทียบชุด feature {len(variants)} แบบ — LSTM พยากรณ์ flare",
        "",
        f"สร้างเมื่อ {datetime.now():%Y-%m-%d %H:%M} · seed {', '.join(map(str, seeds))} (ชุดเดียวกันทุกแบบ) · "
        f"hyperparameter จาก `configs/lstm.yaml` ชุดเดียว (epoch สูงสุด {epochs}) · "
        "threshold เลือกบน validation แยกต่อ (แบบ, seed) แล้ว freeze ก่อนวัด test",
        "",
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

    tss_idx = summary_tss.set_index(["scope", "variant"])
    auc_idx = summary_auc.set_index(["scope", "variant"])
    first_runs = runs_table.drop_duplicates(["scope"]).set_index("scope")
    lines += ["## ผลหลัก", ""]
    for scope in ("test", "val"):
        if scope not in tss_idx.index.get_level_values("scope"):
            continue
        n, n_pos = int(first_runs.loc[scope, "n"]), int(first_runs.loc[scope, "n_pos"])
        lines += [
            f"### {SCOPE_TITLES[scope]}",
            "",
            f"sample {n} · positive {n_pos}",
            "",
            f"| แบบ | ชุด feature | feature | TSS (mean ± SD) | ΔTSS เทียบ {CONTROL} (จับคู่ seed) | AUC (mean ± SD) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for variant, spec in variants.items():
            if (scope, variant) not in tss_idx.index:
                continue
            t, a = tss_idx.loc[(scope, variant)], auc_idx.loc[(scope, variant)]
            delta = "— (control)" if variant == CONTROL else _pm(t["delta_mean"], t["delta_sd"])
            lines.append(
                f"| {variant} | {spec['label']} | {len(spec['columns'])} | {_pm(t['mean'], t['sd'])} | {delta} | "
                f"{_pm(a['mean'], a['sd'], signed=False)} |"
            )
        lines.append("")

    lines += ["## ตัวเลขดิบราย seed (TSS)", "", "ให้คำนวณ Δ ตรวจเองได้จากตารางนี้", ""]
    table = seed_table(runs_table, "test")
    if not table.empty:
        lines += [f"### {SCOPE_TITLES['test']}", "", "| แบบ | " + " | ".join(f"seed {s}" for s in table.columns) + " |",
                  "|---|" + "---:|" * len(table.columns)]
        for variant, values in table.iterrows():
            lines.append(f"| {variant} | " + " | ".join("n/a" if not np.isfinite(v) else f"{v:+.3f}" for v in values) + " |")
        lines.append("")

    lines += [
        "## อ่านตารางนี้อย่างไร",
        "",
        "ข้อกำกับทั้งหมดนี้เขียนไว้ใน spec ก่อนเห็นผล ไม่ได้เขียนตามผลที่ออกมา",
        "",
        f"- **เทียบ {len(compared)} คู่พร้อมกัน** ({', '.join(compared)} เทียบกับ {CONTROL}) ไม่ใช่คู่เดียว — "
        "ต้องรายงานทุกคู่พร้อมกันเสมอ "
        "ห้ามหยิบคู่ที่ Δ มากที่สุดมาเล่าเป็นข้อค้นพบเดี่ยว",
        f"- **ไม่มีค่า p โดยตั้งใจ** — seed {len(seeds)} ค่าให้กำลังทางสถิติไม่พอให้ค่า p มีความหมาย "
        "ให้ดูขนาดของ Δ เทียบกับ SD ของ Δ แทน: Δ ที่เล็กกว่า SD ของตัวเองแยกไม่ออกจากความผันผวนของค่าเริ่มต้น",
        "- **V5 (X-ray) คือ baseline ความคึกคักของดวงอาทิตย์ ไม่ใช่คู่แข่งเท่าเทียม** — ค่า X-ray เท่ากันทุก HARP "
        "ณ เวลาเดียวกัน จึงตอบได้แค่ \"ช่วงนี้ดวงอาทิตย์จะปะทุไหม\" ไม่ใช่ \"ดวงไหนจะปะทุ\" "
        "และได้เปรียบผิดปกติในช่วงที่ทั้งดวงคึกคักตลอดต่อเนื่อง",
        "- SD คือ sample SD (ddof=1) ข้าม seed · ขอบเขตที่ไม่มี positive แสดง TSS เป็น n/a "
        "(TSS ไม่นิยามเมื่อมี class เดียว)",
        "",
    ]
    return "\n".join(lines)


def aggregate(out_dir: Path, variants: dict[str, dict], signature: dict, arrays: StudyArrays, epochs: int) -> int:
    variant_order = list(variants)
    records, ignored = [], 0
    for path in sorted((out_dir / "runs").glob("*.json")):
        record = load_run(path)
        if record is None or record.get("signature") != signature or record.get("variant") not in variants:
            ignored += 1
            continue
        if not run_paths(out_dir, record["variant"], record["seed"])[1].exists():
            ignored += 1
            continue
        records.append(record)
    if ignored:
        logger.warning(
            "ไม่นำผล %d ไฟล์มารวม (เทรนด้วย config/dataset อื่น, ไฟล์ไม่ครบ หรือเป็นแบบที่ไม่มีในไฟล์ variants)",
            ignored,
        )
    if not records:
        logger.error("ยังไม่มีผลที่ตรงกับ config + dataset ปัจจุบันให้สรุป")
        return 1

    records.sort(key=lambda r: (variant_order.index(r["variant"]), r["seed"]))
    runs_table = pd.DataFrame(
        [{"variant": r["variant"], "seed": r["seed"], **scope} for r in records for scope in r["scopes"]]
    )
    runs_table.to_parquet(out_dir / "runs.parquet", index=False)
    predictions = pd.concat(
        [pd.read_parquet(run_paths(out_dir, r["variant"], r["seed"])[1]) for r in records], ignore_index=True
    )
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    logger.info("รวมผล %d run -> runs.parquet, predictions.parquet (%d แถว)", len(records), len(predictions))

    try:
        summary_tss = paired_summary(runs_table, control=CONTROL, metric="tss")
        summary_auc = paired_summary(runs_table, control=CONTROL, metric="auc")
    except ValueError as exc:
        logger.error(
            "สรุปแบบจับคู่ seed ไม่ได้: %s\n(runs.parquet/predictions.parquet เขียนแล้ว — เทรนส่วนที่ขาดให้ครบ "
            "แล้วรัน --summarise-only)",
            exc,
        )
        return 1

    seeds = sorted({r["seed"] for r in records})
    present = [v for v in variant_order if v in set(runs_table["variant"])]
    shown = {v: variants[v] for v in present}
    balance = dataset_balance(arrays.meta)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "control": CONTROL,
        "seeds": seeds,
        "variants": {v: {"label": s["label"], "role": s["role"], "n_features": len(s["columns"])} for v, s in shown.items()},
        "dataset": balance.to_dict(orient="records"),
        "tss": summary_tss.to_dict(orient="records"),
        "auc": summary_auc.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        render_report(summary_tss, summary_auc, runs_table, shown, balance, seeds, epochs), encoding="utf-8"
    )

    logger.info("=" * 70)
    logger.info("%s", SCOPE_TITLES["test"])
    for row in summary_tss[summary_tss["scope"] == "test"].itertuples(index=False):
        logger.info(
            "  %-3s TSS %-16s Δ %s", row.variant, _pm(row.mean, row.sd),
            "control" if row.variant == CONTROL else _pm(row.delta_mean, row.delta_sd),
        )
    logger.info("รายงานเต็ม: %s", out_dir / "report.md")
    logger.info("=" * 70)
    return 0


# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    cfg = load_lstm_config()
    setup_logging(log_file=data_cfg.paths.artifacts / "logs" / "train_study.log")

    data_dir = args.data_dir or data_cfg.paths.processed / "study_sequences"
    out_dir = args.out_dir or data_cfg.paths.artifacts / STUDY_DIR_NAME

    variants = load_variants(args.variants_file)
    selected = [v.strip() for v in args.variants.split(",")] if args.variants else list(variants)
    unknown = [v for v in selected if v not in variants]
    if unknown:
        logger.error("ไม่มีแบบ %s ใน %s — แบบที่มี: %s", unknown, args.variants_file, list(variants))
        return 1
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
    signature = run_signature(cfg, epochs, arrays)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("งานเปรียบเทียบชุด feature: %s × seed %s · epoch สูงสุด %d", selected, seeds, epochs)
    logger.info("dataset: %s (%d sample, %d feature ผู้สมัคร)", data_dir, len(arrays.x), len(arrays.features))
    logger.info("ผลลัพธ์: %s", out_dir)
    logger.info("=" * 70)

    if not args.summarise_only:
        split_sizes = arrays.meta["split"].value_counts()
        train_pos = int(arrays.meta.loc[arrays.meta["split"] == "train", "label"].sum())
        if split_sizes.get("val", 0) == 0 or split_sizes.get("test", 0) == 0 or train_pos == 0:
            logger.error(
                "ต้องมี val และ test ไม่ว่าง และ train ต้องมี positive (ได้ %s, train positive %d)",
                split_sizes.to_dict(), train_pos,
            )
            return 1

        device = get_device(prefer_cuda=not args.cpu)
        total, index = len(selected) * len(seeds), 0
        for variant in selected:
            spec = variants[variant]
            splits = variant_splits(arrays, spec["columns"])
            for seed in seeds:
                index += 1
                json_path, pred_path, model_path = run_paths(out_dir, variant, seed)
                existing = load_run(json_path)
                if (
                    not args.force and existing is not None
                    and existing.get("signature") == signature and pred_path.exists()
                ):
                    logger.info("[%d/%d] %s seed %d: มีผลแล้ว — ข้าม", index, total, variant, seed)
                    continue

                result = fit_one(splits, cfg, seed, device, epochs)
                scopes = scope_reports(arrays.meta, result["p_val"], result["p_test"], result["threshold"])

                # เขียน predictions + model ก่อน แล้วค่อย json — json คือเครื่องหมายว่า run นี้เสร็จ
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                long_predictions(
                    arrays.meta, variant, seed, result["p_val"], result["p_test"], result["threshold"]
                ).to_parquet(pred_path, index=False)
                save_checkpoint(
                    model_path,
                    result["model"],
                    config={
                        "variant": variant,
                        "seed": seed,
                        "features": spec["columns"],
                        "n_features": len(spec["columns"]),
                        "model": cfg.model.model_dump(),
                        "threshold": result["threshold"],
                    },
                    metrics={scope["scope"]: scope for scope in scopes},
                    extra={"norm_mean": splits.stats["mean"], "norm_std": splits.stats["std"]},
                )
                record = {
                    "variant": variant,
                    "seed": seed,
                    "label": spec["label"],
                    "columns": spec["columns"],
                    "threshold": result["threshold"],
                    "val_score": result["val_score"],
                    "best_epoch": result["best_epoch"],
                    "epochs_run": result["epochs_run"],
                    "train_seconds": result["train_seconds"],
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
                    "[%d/%d] %s seed %d: val TSS %+.3f · test TSS %s · "
                    "epoch ที่ดีที่สุด %d/%d (%.0f วิ)",
                    index, total, variant, seed, result["val_score"],
                    _pm(by_scope["test"]["tss"], float("nan")),
                    result["best_epoch"], result["epochs_run"], result["train_seconds"],
                )

    return aggregate(out_dir, variants, signature, arrays, epochs)


if __name__ == "__main__":
    raise SystemExit(main())
