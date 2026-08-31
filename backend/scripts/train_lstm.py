"""เทรน LSTM พยากรณ์ flare พร้อมเทียบกับ logistic-regression baseline

    python backend/scripts/train_lstm.py

**baseline สำคัญมาก**: logistic regression ใช้ SHARP parameters ณ *เวลาเดียว*
(timestep สุดท้ายของหน้าต่าง) ขณะที่ LSTM เห็นประวัติ 24 ชั่วโมง ถ้า LSTM ชนะ
baseline ไม่ชัดเจน แปลว่ามิติเวลาไม่ได้ช่วย และควรใช้โมเดลที่ง่ายกว่าแทน
สคริปต์นี้จึงเทรนทั้งสองตัวและรายงานเทียบกันเสมอ
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from sunseg.config import load_data_config, load_lstm_config  # noqa: E402
from sunseg.datasets.sequence import SequenceSplits, load_sequence_splits, make_loaders  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.losses import build_forecast_loss  # noqa: E402
from sunseg.metrics import classification_report, find_best_threshold, format_report  # noqa: E402
from sunseg.models.lstm import build_lstm  # noqa: E402
from sunseg.train_utils import (  # noqa: E402
    EarlyStopping,
    build_scheduler,
    get_device,
    save_checkpoint,
    save_metrics,
    save_predictions,
    set_seed,
)

logger = logging.getLogger("train_lstm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch ใน config")
    p.add_argument("--no-baseline", action="store_true", help="ข้าม logistic regression baseline")
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """คืน (y_true, y_prob) ของทั้ง loader"""
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(probs)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_baseline(splits: SequenceSplits) -> tuple[dict, np.ndarray, np.ndarray]:
    """logistic regression บน feature ณ timestep สุดท้ายเท่านั้น

    คืน ``(test_report, val_prob, test_prob)`` — ค่าความน่าจะเป็นสองชุดหลังถูกส่งต่อ
    ให้ ``build_predictions_frame`` เพื่อเก็บลงไฟล์ค่าทำนายราย sample ร่วมกับของ LSTM
    """
    from sklearn.linear_model import LogisticRegression

    logger.info("-" * 70)
    logger.info("baseline: logistic regression (ใช้ค่า ณ เวลาเดียว ไม่มีมิติเวลา)")

    # [:, -1, :] = timestep ล่าสุด ซึ่งเป็นข้อมูลที่มีในขณะออกพยากรณ์
    x_train = splits.train.x[:, -1, :].numpy()
    x_val = splits.val.x[:, -1, :].numpy()
    x_test = splits.test.x[:, -1, :].numpy()
    y_train = splits.train.y.numpy()
    y_val = splits.val.y.numpy()
    y_test = splits.test.y.numpy()

    model = LogisticRegression(
        max_iter=2000,
        # ชดเชยความไม่สมดุลของ class ให้เทียบเคียงกับ focal loss ที่ LSTM ใช้
        class_weight="balanced",
        C=1.0,
    )
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, val_tss = find_best_threshold(y_val, val_prob, metric="tss")

    test_prob = model.predict_proba(x_test)[:, 1]
    report = classification_report(y_test, test_prob, threshold)

    logger.info("  val TSS ที่ threshold ดีที่สุด = %.4f (threshold %.3f)", val_tss, threshold)
    logger.info("  ผลบน test:\n%s", format_report(report))
    return report, val_prob, test_prob


def build_predictions_frame(
    meta: pd.DataFrame,
    lstm_val_prob: np.ndarray,
    lstm_test_prob: np.ndarray,
    baseline_val_prob: np.ndarray | None,
    baseline_test_prob: np.ndarray | None,
) -> pd.DataFrame:
    """รวมค่าทำนายราย sample ของ val และ test เป็นตารางเดียว (ไม่เอา train)

    ลำดับแถวต้องตรงกับที่ ``load_sequence_splits`` ใช้สร้าง ``SequenceDataset`` — คือ
    boolean mask ตามคอลัมน์ ``split`` ของ meta โดยไม่ shuffle (ดู ``sequence.py``)
    เช่นเดียวกับ ``run_baseline`` ที่ index จาก ``splits.val``/``splits.test`` ตรง ๆ
    มิฉะนั้นค่าทำนายจะไปจับคู่ผิด sample โดยไม่มี error ให้เห็น
    """
    frames: list[pd.DataFrame] = []
    for name, lstm_prob, baseline_prob in (
        ("val", lstm_val_prob, baseline_val_prob),
        ("test", lstm_test_prob, baseline_test_prob),
    ):
        rows = meta.loc[meta["split"] == name, ["split", "HARPNUM", "noaa_ar", "issue_time", "label"]]
        rows = rows.reset_index(drop=True)
        if len(rows) != len(lstm_prob):
            raise ValueError(
                f"จำนวนแถว meta ของ split={name} ({len(rows)}) ไม่ตรงกับจำนวนค่าทำนาย "
                f"({len(lstm_prob)}) — ตรวจสอบว่า meta กับ SequenceDataset ยังเรียงตรงกันอยู่"
            )
        rows = rows.copy()
        rows["lstm_prob"] = np.asarray(lstm_prob, dtype=np.float64)
        rows["baseline_prob"] = (
            np.asarray(baseline_prob, dtype=np.float64) if baseline_prob is not None else np.nan
        )
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    cfg = load_lstm_config()

    setup_logging(log_file=data_cfg.paths.artifacts / "logs" / "train_lstm.log")
    set_seed(cfg.train.seed)
    device = get_device(prefer_cuda=not args.cpu)

    logger.info("=" * 70)
    logger.info("เทรน LSTM พยากรณ์ flare >= %s ใน %d ชม.",
                data_cfg.flare.positive_goes_class, data_cfg.flare.horizon_hours)
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
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

    epochs = args.epochs or cfg.train.epochs

    # ------------------------------------------------------------------ #
    model = build_lstm(splits.n_features, cfg.model).to(device)
    logger.info(
        "โมเดล: %s features -> LSTM(hidden=%d, layers=%d) -> 1 logit  (%s พารามิเตอร์)",
        splits.n_features,
        cfg.model.hidden_size,
        cfg.model.num_layers,
        f"{model.count_parameters():,}",
    )

    criterion = build_forecast_loss(cfg.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(optimizer, cfg.train, epochs)
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")

    train_loader, val_loader, test_loader = make_loaders(splits, cfg.train.batch_size)

    # ------------------------------------------------------------------ #
    logger.info("-" * 70)
    logger.info("เริ่มเทรน %d epoch (loss = %s)", epochs, cfg.loss.type)
    history: list[dict] = []
    started = time.time()

    for epoch in range(epochs):
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, cfg.train.grad_clip
        )
        y_val, p_val = evaluate(model, val_loader, device)
        # ต้องวัดด้วยเกณฑ์เดียวกับที่ใช้ตัดสินผลสุดท้าย มิฉะนั้นจะเลือก checkpoint ผิด:
        # โมเดลที่จัดอันดับได้ดี (AUC สูง) แต่ให้ค่าความน่าจะเป็นต่ำกว่า 0.5 ทั้งหมด
        # จะได้ TSS@0.5 = 0 ทั้งที่จริงเป็นโมเดลที่ดีเมื่อปรับ threshold แล้ว
        # ซึ่งเป็นสถานการณ์ปกติเมื่อ positive มีเพียง ~2%
        _, val_tss = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)

        if scheduler is not None:
            scheduler.step(val_tss) if cfg.train.scheduler == "plateau" else scheduler.step()

        history.append({"epoch": epoch + 1, "loss": loss, "val_tss": val_tss})
        if (epoch + 1) % 2 == 0 or epoch == 0:
            logger.info(
                "  epoch %3d/%d  loss %.4f  val TSS %+.4f  lr %.2e",
                epoch + 1, epochs, loss, val_tss, optimizer.param_groups[0]["lr"],
            )

        if stopper.step(val_tss, epoch, model):
            logger.info("  หยุดก่อนกำหนดที่ epoch %d (ไม่ดีขึ้น %d epoch ติด)",
                        epoch + 1, stopper.patience)
            break

    stopper.restore(model)
    logger.info("ใช้เวลาเทรน %.1f วินาที", time.time() - started)

    # ------------------------------------------------------------------ #
    logger.info("-" * 70)
    logger.info("เลือก threshold จาก validation (ห้ามใช้ test — จะทำให้ผลสูงเกินจริง)")
    y_val, p_val = evaluate(model, val_loader, device)
    threshold, val_score = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)
    logger.info("  threshold = %.3f  ให้ val %s = %.4f", threshold, cfg.eval.primary_metric, val_score)

    val_report = classification_report(y_val, p_val, threshold)
    y_test, p_test = evaluate(model, test_loader, device)
    test_report = classification_report(y_test, p_test, threshold)

    logger.info("-" * 70)
    logger.info("ผลของ LSTM บน validation:\n%s", format_report(val_report))
    logger.info("ผลของ LSTM บน test:\n%s", format_report(test_report))

    # ------------------------------------------------------------------ #
    baseline_report = None
    baseline_val_prob = None
    baseline_test_prob = None
    if not args.no_baseline:
        baseline_report, baseline_val_prob, baseline_test_prob = run_baseline(splits)

    # ------------------------------------------------------------------ #
    artifacts = data_cfg.paths.artifacts
    save_checkpoint(
        artifacts / "models" / "lstm.pt",
        model,
        config={
            "n_features": splits.n_features,
            "features": splits.features,
            "model": cfg.model.model_dump(),
            "threshold": threshold,
            "horizon_hours": data_cfg.flare.horizon_hours,
            "positive_class": data_cfg.flare.positive_goes_class,
        },
        metrics={"val": val_report, "test": test_report},
        extra={"norm_mean": splits.stats["mean"], "norm_std": splits.stats["std"]},
    )
    save_metrics(
        artifacts / "metrics" / "lstm.json",
        {
            "lstm": {"val": val_report, "test": test_report},
            "baseline_logistic": baseline_report,
            "history": history,
            "n_train": len(splits.train),
            "n_val": len(splits.val),
            "n_test": len(splits.test),
        },
    )
    save_predictions(
        artifacts / "metrics" / "predictions.parquet",
        build_predictions_frame(
            splits.meta, p_val, p_test, baseline_val_prob, baseline_test_prob
        ),
    )

    # ------------------------------------------------------------------ #
    logger.info("=" * 70)
    logger.info("สรุป (ตัวเลขบน test set)")
    logger.info("=" * 70)
    logger.info("  %-28s TSS = %+.4f", "LSTM (มีมิติเวลา)", test_report["tss"])
    if baseline_report is not None:
        gain = test_report["tss"] - baseline_report["tss"]
        logger.info("  %-28s TSS = %+.4f", "logistic (เวลาเดียว)", baseline_report["tss"])
        logger.info("  %-28s      %+.4f", "ส่วนต่าง", gain)
        logger.info("")
        if gain > 0.05:
            logger.info("  [ OK ] มิติเวลาช่วยได้จริงอย่างมีนัยสำคัญ")
        elif gain > 0:
            logger.info("  [WARN] LSTM ดีกว่าเล็กน้อย — พิจารณาว่าคุ้มกับความซับซ้อนหรือไม่")
        else:
            logger.info("  [WARN] LSTM ไม่ชนะ baseline — ลองเพิ่มความยาว sequence หรือลดขนาดโมเดล")

    if test_report["tss"] < 0.3:
        logger.warning("\n  TSS ต่ำกว่า 0.3 — ตรวจสอบคุณภาพ label และปริมาณข้อมูล")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
