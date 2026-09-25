"""ลูปเทรนโมเดลพยากรณ์ — ชุดเดียวที่ทุกสคริปต์ใช้ร่วมกัน

- ``scripts/forecast/train.py`` (production ทีละโมเดล) -> :func:`fit_forecaster` +
  :func:`fit_logistic_baseline`
- ``scripts/study/train.py`` (ตารางเปรียบเทียบ) และ ``scripts/study/tune.py`` (ค้นหา
  hyperparameter) -> :func:`fit_one`

ใช้ลูปเดียวกันทุกที่โดยตั้งใจ: ตัวเลขของงานเปรียบเทียบจะอ้างถึงโมเดล production ได้ก็ต่อเมื่อ
ขั้นตอนการเทรนเหมือนกันทุกประการ ทุกสถาปัตยกรรมผ่านลูปนี้เหมือนกัน — สิ่งที่ต่างกันมีแค่
``cfg.kind`` / ``cfg.model`` ที่ส่งเข้า :func:`sunseg.models.forecast.build_architecture`
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from ..datasets.sequence import SequenceSplits, make_loaders
from ..metrics import classification_report, find_best_threshold, format_report
from ..models.forecast import architecture_label, build_architecture
from .losses import build_forecast_loss
from .utils import EarlyStopping, build_scheduler, set_seed

logger = logging.getLogger(__name__)


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


def fit_forecaster(
    splits: SequenceSplits, cfg, device: torch.device, epochs: int, verbose: bool = False
) -> dict:
    """เทรนโมเดลหนึ่งตัวจนจบ แล้วเลือก threshold บน val — ไม่ตั้ง seed เอง (ผู้เรียกตั้ง)

    ``cfg`` คือ :class:`sunseg.config.ForecastConfig` (``kind`` + ``model`` + ``train`` +
    ``loss`` + ``eval``) ``verbose=True`` พิมพ์ความคืบหน้าทุก 2 epoch แบบสคริปต์ production

    Returns
    -------
    dict ที่มี ``model`` (น้ำหนักของ epoch ที่ดีที่สุด), ``threshold`` (freeze จาก val),
    ``val_score``, ``y_val``/``p_val``, ``y_test``/``p_test``, ``best_epoch``,
    ``epochs_run``, ``train_seconds`` และ ``history``
    """
    model = build_architecture(cfg.kind, splits.n_features, splits.seq_len, cfg.model).to(device)
    if verbose:
        logger.info(
            "โมเดล: %s · %d features × %d timestep -> 1 logit  (%s พารามิเตอร์)",
            architecture_label(cfg.kind), splits.n_features, splits.seq_len,
            f"{model.count_parameters():,}",
        )

    criterion = build_forecast_loss(cfg.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.train, epochs)
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")
    train_loader, val_loader, test_loader = make_loaders(splits, cfg.train.batch_size)

    if verbose:
        logger.info("-" * 70)
        logger.info("เริ่มเทรน %d epoch (loss = %s)", epochs, cfg.loss.type)
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, cfg.train.grad_clip)
        y_val, p_val = evaluate(model, val_loader, device)
        # ต้องวัดด้วยเกณฑ์เดียวกับที่ใช้ตัดสินผลสุดท้าย มิฉะนั้นจะเลือก checkpoint ผิด:
        # โมเดลที่จัดอันดับได้ดี (AUC สูง) แต่ให้ค่าความน่าจะเป็นต่ำกว่า 0.5 ทั้งหมด
        # จะได้ TSS@0.5 = 0 ทั้งที่จริงเป็นโมเดลที่ดีเมื่อปรับ threshold แล้ว
        # ซึ่งเป็นสถานการณ์ปกติเมื่อ positive มีเพียง ~2%
        _, val_tss = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)

        if scheduler is not None:
            scheduler.step(val_tss) if cfg.train.scheduler == "plateau" else scheduler.step()

        history.append({"epoch": epoch + 1, "loss": float(loss), "val_tss": float(val_tss)})
        if verbose and ((epoch + 1) % 2 == 0 or epoch == 0):
            logger.info(
                "  epoch %3d/%d  loss %.4f  val TSS %+.4f  lr %.2e",
                epoch + 1, epochs, loss, val_tss, optimizer.param_groups[0]["lr"],
            )

        if stopper.step(val_tss, epoch, model):
            if verbose:
                logger.info("  หยุดก่อนกำหนดที่ epoch %d (ไม่ดีขึ้น %d epoch ติด)", epoch + 1, stopper.patience)
            break

    stopper.restore(model)
    seconds = time.perf_counter() - started
    if verbose:
        logger.info("ใช้เวลาเทรน %.1f วินาที", seconds)

    # threshold เลือกบน val แล้ว freeze — test ถูกแตะหลังจากนี้เท่านั้น (ห้ามใช้ test เลือก
    # threshold — จะทำให้ผลสูงเกินจริง)
    y_val, p_val = evaluate(model, val_loader, device)
    threshold, val_score = find_best_threshold(y_val, p_val, metric=cfg.eval.primary_metric)
    y_test, p_test = evaluate(model, test_loader, device)
    return {
        "model": model,
        "threshold": float(threshold),
        "val_score": float(val_score),
        "y_val": y_val,
        "p_val": np.asarray(p_val, dtype=np.float64),
        "y_test": y_test,
        "p_test": np.asarray(p_test, dtype=np.float64),
        "best_epoch": int(stopper.best_epoch + 1),
        "epochs_run": len(history),
        "train_seconds": float(seconds),
        "history": history,
    }


def fit_one(splits: SequenceSplits, cfg, seed: int, device: torch.device, epochs: int) -> dict:
    """หนึ่ง run ของงานเปรียบเทียบ/การค้นหา — ตั้ง seed แบบ deterministic ก่อนเทรน

    ``set_seed`` ต้องมาก่อนสร้างโมเดลและ loader (ลำดับ shuffle ใช้ RNG เดียวกัน) ผลของ run
    เดียวกันจึงทำซ้ำได้ ซึ่งการสรุปแบบจับคู่ seed ของ ``sunseg.study.summary`` ต้องพึ่ง
    """
    set_seed(seed, deterministic=True)
    return fit_forecaster(splits, cfg, device, epochs)


def fit_logistic_baseline(splits: SequenceSplits) -> dict:
    """logistic regression บน feature ณ timestep สุดท้ายเท่านั้น — ไม่มีมิติเวลา

    **baseline สำคัญมาก**: ถ้าโมเดลลำดับเวลาชนะตัวนี้ไม่ชัดเจน แปลว่ามิติเวลาไม่ได้ช่วย และ
    ควรใช้โมเดลที่ง่ายกว่าแทน threshold เลือกบน val เหมือนโมเดลหลักทุกประการ

    คืน ``{"threshold", "val_score", "val_prob", "test_prob", "test_report"}``
    """
    from sklearn.linear_model import LogisticRegression

    logger.info("-" * 70)
    logger.info("baseline: logistic regression (ใช้ค่า ณ เวลาเดียว ไม่มีมิติเวลา)")

    # [:, -1, :] = timestep ล่าสุด ซึ่งเป็นข้อมูลที่มีในขณะออกพยากรณ์
    x_train = splits.train.x[:, -1, :].numpy()
    x_val = splits.val.x[:, -1, :].numpy()
    x_test = splits.test.x[:, -1, :].numpy()

    model = LogisticRegression(
        max_iter=2000,
        # ชดเชยความไม่สมดุลของ class ให้เทียบเคียงกับ focal loss ที่โมเดลหลักใช้
        class_weight="balanced",
        C=1.0,
    )
    model.fit(x_train, splits.train.y.numpy())

    val_prob = model.predict_proba(x_val)[:, 1]
    threshold, val_score = find_best_threshold(splits.val.y.numpy(), val_prob, metric="tss")
    test_prob = model.predict_proba(x_test)[:, 1]
    report = classification_report(splits.test.y.numpy(), test_prob, threshold)

    logger.info("  val TSS ที่ threshold ดีที่สุด = %.4f (threshold %.3f)", val_score, threshold)
    logger.info("  ผลบน test:\n%s", format_report(report))
    return {
        "threshold": float(threshold),
        "val_score": float(val_score),
        "val_prob": val_prob,
        "test_prob": test_prob,
        "test_report": report,
    }
