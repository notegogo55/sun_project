"""เทรน U-Net แบ่งส่วน active region จากภาพ magnetogram เต็มดวง

    python backend/scripts/train_unet.py

ปรับมาให้ทำงานบน GPU ที่มี VRAM เพียง 4 GB โดยใช้:

* **AMP (mixed precision)** — ลดหน่วยความจำที่ใช้เก็บ activation ลงราวครึ่งหนึ่ง
* **gradient accumulation** — ได้ผลเทียบเท่า batch ใหญ่โดยไม่ต้องใช้ VRAM เพิ่ม
* **ภาพ 512x512** — ย่อจากต้นฉบับ 4096x4096

ถ้ายังเจอ CUDA out of memory ให้ลด ``model.base_channels`` เป็น 16 หรือ
``data.image_size`` เป็น 384 ใน ``configs/unet.yaml``
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sunseg.config import load_data_config, load_unet_config  # noqa: E402
from sunseg.datasets.segmentation import (  # noqa: E402
    make_segmentation_loaders,
    split_frames_by_time,
)
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.losses import build_segmentation_loss  # noqa: E402
from sunseg.metrics import dice_coefficient, iou_score  # noqa: E402
from sunseg.models.unet import build_unet  # noqa: E402
from sunseg.train_utils import (  # noqa: E402
    EarlyStopping,
    build_scheduler,
    get_device,
    save_checkpoint,
    save_metrics,
    set_seed,
)

logger = logging.getLogger("train_unet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--epochs", type=int, help="ทับจำนวน epoch ใน config")
    p.add_argument("--batch-size", type=int, help="ทับขนาด batch ใน config")
    p.add_argument("--cpu", action="store_true", help="บังคับใช้ CPU")
    p.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="ทดสอบว่าโมเดลเรียนรู้ได้: เทรนซ้ำบน batch เดียว loss ต้องลู่เข้าใกล้ 0",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="เริ่มเทรนใหม่ตั้งแต่ epoch 0 ไม่สนใจ checkpoint ค้างจากรอบก่อน (unet_resume.pt)",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, masks).item()

        predicted = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        truth = masks.cpu().numpy()
        for pred_one, true_one in zip(predicted, truth, strict=True):
            dice_scores.append(dice_coefficient(pred_one, true_one))
            iou_scores.append(iou_score(pred_one, true_one))

    return {
        "loss": total_loss / max(len(loader), 1),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "iou": float(np.mean(iou_scores)) if iou_scores else 0.0,
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg) -> float:
    model.train()
    total_loss, n_steps = 0.0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.train.amp and device.type == "cuda"):
            loss = criterion(model(images), masks) / cfg.train.grad_accum_steps

        scaler.scale(loss).backward()

        # อัปเดตน้ำหนักทุกๆ grad_accum_steps ครั้ง — ได้ผลเท่ากับใช้ batch ใหญ่ขึ้นเท่านั้นเท่า
        if (step + 1) % cfg.train.grad_accum_steps == 0:
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.train.grad_accum_steps
        n_steps += 1

    return total_loss / max(n_steps, 1)


def run_overfit_check(model, loader, criterion, optimizer, device) -> None:
    """เทรนซ้ำบน batch เดียว 200 รอบ — เป็นการตรวจสุขภาพพื้นฐานของโมเดล

    ถ้า loss ไม่ลดลงใกล้ 0 แปลว่ามีบางอย่างผิดในโมเดล/loss/ข้อมูล และการเทรนเต็ม
    รูปแบบจะเสียเวลาเปล่า ควรรันทุกครั้งที่แก้สถาปัตยกรรม
    """
    logger.info("-" * 70)
    logger.info("ทดสอบ overfit บน batch เดียว (loss ควรลดลงใกล้ 0)")

    images, masks = next(iter(loader))
    images, masks = images.to(device), masks.to(device)

    model.train()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == 199:
            with torch.no_grad():
                dice = dice_coefficient(
                    (torch.sigmoid(model(images)) >= 0.5).cpu().numpy(), masks.cpu().numpy()
                )
            logger.info("  รอบที่ %3d  loss %.5f  Dice %.4f", step, loss.item(), dice)

    if loss.item() < 0.1:
        logger.info("  [ OK ] โมเดลเรียนรู้ได้ปกติ")
    else:
        logger.warning("  [WARN] loss ยังสูง — ตรวจสอบสถาปัตยกรรม, loss หรือคุณภาพข้อมูล")


def main() -> int:
    args = parse_args()
    data_cfg = load_data_config()
    cfg = load_unet_config()

    setup_logging(log_file=data_cfg.paths.artifacts / "logs" / "train_unet.log")
    set_seed(cfg.train.seed)
    device = get_device(prefer_cuda=not args.cpu)

    logger.info("=" * 70)
    logger.info("เทรน U-Net แบ่งส่วน active region")
    logger.info("=" * 70)

    # ------------------------------------------------------------------ #
    frames_dir = data_cfg.paths.processed / "frames"
    image_dir = frames_dir / "images"
    if not image_dir.exists() or not any(image_dir.glob("*.npy")):
        logger.error(
            "ไม่พบเฟรมภาพใน %s\n"
            "รัน `python backend/scripts/download_images.py` แล้วตามด้วย `python backend/scripts/build_masks.py`",
            image_dir,
        )
        return 1

    timestamps = sorted(p.stem for p in image_dir.glob("*.npy"))
    logger.info("พบเฟรมทั้งหมด %d เฟรม (%s ถึง %s)", len(timestamps), timestamps[0], timestamps[-1])

    splits = split_frames_by_time(
        timestamps,
        train_end=data_cfg.split.train_end.isoformat(),
        val_end=data_cfg.split.val_end.isoformat(),
        gap_days=data_cfg.split.gap_days,
    )
    if not splits["train"] or not splits["val"]:
        logger.error(
            "train หรือ val ว่างเปล่า — ช่วงเวลาของภาพอาจไม่ครอบคลุมเส้นแบ่งใน configs/data.yaml"
        )
        return 1

    batch_size = args.batch_size or cfg.train.batch_size
    epochs = args.epochs or cfg.train.epochs

    train_loader, val_loader, test_loader = make_segmentation_loaders(
        frames_dir=frames_dir,
        splits=splits,
        batch_size=batch_size,
        norm_scale=cfg.data.norm_scale_gauss,
        augment=cfg.augment,
        num_workers=cfg.train.num_workers,
        seed=cfg.train.seed,
    )

    positive_fraction = train_loader.dataset.positive_fraction()
    logger.info(
        "สัดส่วนพิกเซลที่เป็น AR: %.3f%% — นี่คือเหตุผลที่ต้องใช้ pos_weight=%.0f และ Dice loss",
        100 * positive_fraction,
        cfg.loss.pos_weight,
    )

    # ------------------------------------------------------------------ #
    model = build_unet(cfg.model).to(device)
    if device.type == "cuda":
        # channels_last ช่วยให้ conv บน GPU เข้าถึงหน่วยความจำได้ต่อเนื่องกว่า
        model = model.to(memory_format=torch.channels_last)

    logger.info(
        "โมเดล: U-Net base=%d depth=%d ที่ %dpx (%s พารามิเตอร์)",
        cfg.model.base_channels,
        cfg.model.depth,
        cfg.data.image_size,
        f"{model.count_parameters():,}",
    )

    criterion = build_segmentation_loss(cfg.loss).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    if args.overfit_one_batch:
        run_overfit_check(model, train_loader, criterion, optimizer, device)
        return 0

    scheduler = build_scheduler(optimizer, cfg.train, epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp and device.type == "cuda")
    stopper = EarlyStopping(patience=cfg.train.early_stop_patience, mode="max")

    # ------------------------------------------------------------------ #
    # กู้คืนจาก checkpoint ค้าง ถ้ามี — กัน process ถูก kill กลางทาง (เช่น session
    # ของ agent ที่รันคำสั่งนี้ปิดไป) แล้วต้องเริ่มเทรนหลายชั่วโมงใหม่ทั้งหมด
    resume_path = data_cfg.paths.artifacts / "models" / "unet_resume.pt"
    start_epoch = 0
    history: list[dict] = []
    if resume_path.exists() and not args.fresh:
        state = torch.load(resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        stopper.best_score = state["stopper_best_score"]
        stopper.best_epoch = state["stopper_best_epoch"]
        stopper.counter = state["stopper_counter"]
        stopper.best_state = state["stopper_best_state"]
        history = state["history"]
        start_epoch = state["epoch"] + 1
        logger.info(
            "กู้คืนการเทรนจาก %s: จะเทรนต่อที่ epoch %d (คะแนนดีที่สุดตอนนี้ %.4f ที่ epoch %d)",
            resume_path, start_epoch + 1, stopper.best_score, stopper.best_epoch + 1,
        )

    logger.info("-" * 70)
    logger.info(
        "เริ่มเทรน %d epoch (batch %d x สะสม %d = effective %d)",
        epochs, batch_size, cfg.train.grad_accum_steps, batch_size * cfg.train.grad_accum_steps,
    )
    started = time.time()

    for epoch in range(start_epoch, epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg)
        val = evaluate(model, val_loader, criterion, device, cfg.eval.threshold)

        if scheduler is not None:
            scheduler.step(val["dice"]) if cfg.train.scheduler == "plateau" else scheduler.step()

        history.append({"epoch": epoch + 1, "train_loss": loss, **{f"val_{k}": v for k, v in val.items()}})
        logger.info(
            "  epoch %3d/%d  loss %.4f  val loss %.4f  Dice %.4f  IoU %.4f  lr %.2e",
            epoch + 1, epochs, loss, val["loss"], val["dice"], val["iou"],
            optimizer.param_groups[0]["lr"],
        )

        should_stop = stopper.step(val["dice"], epoch, model)

        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "history": history,
                "stopper_best_score": stopper.best_score,
                "stopper_best_epoch": stopper.best_epoch,
                "stopper_counter": stopper.counter,
                "stopper_best_state": stopper.best_state,
            },
            resume_path,
        )

        if stopper.best_epoch == epoch:
            # เซฟทันทีที่ดีขึ้น กันเทรนพังกลางทาง (เช่น process ถูก kill, out-of-memory)
            # แล้วเสียเวลาเทรนไปหลายชั่วโมงโดยไม่มี checkpoint เหลือเลย
            save_checkpoint(
                data_cfg.paths.artifacts / "models" / "unet.pt",
                model,
                config={
                    "model": cfg.model.model_dump(),
                    "image_size": cfg.data.image_size,
                    "norm_scale_gauss": cfg.data.norm_scale_gauss,
                    "threshold": cfg.eval.threshold,
                },
                metrics={"val": val},
                extra={"epoch": epoch + 1},
            )
        if should_stop:
            logger.info("  หยุดก่อนกำหนดที่ epoch %d", epoch + 1)
            break

    stopper.restore(model)
    logger.info("ใช้เวลาเทรน %.1f นาที", (time.time() - started) / 60)

    # ------------------------------------------------------------------ #
    val_metrics = evaluate(model, val_loader, criterion, device, cfg.eval.threshold)
    test_metrics = evaluate(model, test_loader, criterion, device, cfg.eval.threshold)

    logger.info("-" * 70)
    logger.info("ผลบน validation:  Dice %.4f  IoU %.4f", val_metrics["dice"], val_metrics["iou"])
    logger.info("ผลบน test:        Dice %.4f  IoU %.4f", test_metrics["dice"], test_metrics["iou"])

    artifacts = data_cfg.paths.artifacts
    save_checkpoint(
        artifacts / "models" / "unet.pt",
        model,
        config={
            "model": cfg.model.model_dump(),
            "image_size": cfg.data.image_size,
            "norm_scale_gauss": cfg.data.norm_scale_gauss,
            "threshold": cfg.eval.threshold,
        },
        metrics={"val": val_metrics, "test": test_metrics},
    )
    save_metrics(
        artifacts / "metrics" / "unet.json",
        {
            "val": val_metrics,
            "test": test_metrics,
            "history": history,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
            "positive_pixel_fraction": positive_fraction,
        },
    )

    logger.info("=" * 70)
    if test_metrics["dice"] >= 0.70:
        logger.info("  [ OK ] Dice %.4f ผ่านเกณฑ์ที่ตั้งไว้ (0.70)", test_metrics["dice"])
    else:
        logger.warning(
            "  [WARN] Dice %.4f ต่ำกว่าเกณฑ์ 0.70 — ลองเทรนนานขึ้น เพิ่มข้อมูล "
            "หรือตรวจว่า mask วางตรงตำแหน่งด้วย backend/scripts/plot_masks.py",
            test_metrics["dice"],
        )
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
