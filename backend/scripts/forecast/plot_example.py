"""สร้างรูปตัวอย่าง "ผลพยากรณ์ 24 ชม." ของโมเดลหนึ่งตัว เทียบกับฟลักซ์ GOES X-ray จริง

    python backend/scripts/forecast/plot_example.py                 # โมเดลปริยาย (LSTM)
    python backend/scripts/forecast/plot_example.py --model tcn

อ่านฟลักซ์ X-ray รายนาทีจริง (``sunseg.data.xray_flux.XrayFluxStore``) + ผลทำนายของ
โมเดล production จริง (``artifacts/metrics/<ชื่อ>_predictions.parquet``, threshold ที่
freeze ไว้ใน ``artifacts/metrics/<ชื่อ>.json`` — ดู ``sunseg.artifacts``) มาวาดเป็นรูปเดียว
เพื่อสื่อว่าโมเดล "ฟ้อง" เหตุการณ์ล่วงหน้า 24 ชม. ได้จริงหรือไม่ — ไม่ใช่ภาพร่างประกอบ
เขียนลง ``artifacts/figures/<ชื่อ>_forecast_example.png``

ใช้ HARP 13999 / AR 14274 ช่วง 8-14 พ.ย. 2025 (อยู่ใน test split ที่โมเดลไม่เคยเห็น
ตอนเทรน/เลือก threshold) เพราะมี flare ระดับ X ต่อเนื่องสี่ลูกในสัปดาห์เดียว (X1.7,
X1.2, X5.1, X4.0) เห็นเหตุการณ์ชัดกว่าสัปดาห์อื่นในชุด test ทั้งหมด

กรอบ "ผลพยากรณ์ 24ชม." ในรูปไม่ได้ hardcode วันที่ไว้ตรงๆ: เลือก flare แรงสุด
ในสัปดาห์ (X5.1) แล้วถอยเวลากลับไปพอดี ``flare.horizon_hours`` ชม. จาก peak ของมัน
เพื่อหา issue_time ที่โมเดลออกคำพยากรณ์ล่วงหน้า — ถ้าเปลี่ยนไปใช้ HARP/สัปดาห์อื่น
ในอนาคต ตรรกะนี้ยังใช้ได้โดยไม่ต้องแก้เลขวันที่
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# report_style ต้องมาก่อน pyplot — ตั้ง backend Agg + ธีม seaborn (ดู docstring ของโมดูลนั้น)
from sunseg.report_style import BLUE, THRESHOLD, save_figure, true_runs  # noqa: E402, I001

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from sunseg.artifacts import ForecastArtifacts  # noqa: E402
from sunseg.config import DataConfig, load_data_config, load_forecast_config  # noqa: E402
from sunseg.data.goes_class import CLASS_BASE_FLUX  # noqa: E402
from sunseg.data.xray_flux import XrayFluxStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.models.forecast import architecture_label  # noqa: E402

logger = logging.getLogger("plot_forecast_example")

HARPNUM = 13999
WINDOW_START = pd.Timestamp("2025-11-08 00:00:00")
WINDOW_END = pd.Timestamp("2025-11-14 23:59:00")

FLUX_YLIM = (1e-7, 2e-3)
PROB_YLIM = (0.0, 1.05)


# --------------------------------------------------------------------------- #
# การโหลดข้อมูล
# --------------------------------------------------------------------------- #
def load_flux(cfg: DataConfig) -> pd.DataFrame:
    store = XrayFluxStore(cfg.paths.raw / "xrs")
    if not store.available:
        raise SystemExit(f"ไม่พบไฟล์ XRS ที่ {store.root} — รัน backend/scripts/data/download_xray.py ก่อน")
    series = store.series(WINDOW_START.to_pydatetime(), WINDOW_END.to_pydatetime())
    frame = pd.DataFrame({"time": series.times, "flux": series.flux}).dropna()
    if frame.empty:
        raise SystemExit(f"ไม่มีข้อมูล X-ray ในช่วง {WINDOW_START} – {WINDOW_END}")
    return frame


def load_flares(cfg: DataConfig) -> pd.DataFrame:
    flares = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    flares["peak_time"] = pd.to_datetime(flares["peak_time"])
    mask = (flares["HARPNUM"] == HARPNUM) & flares["peak_time"].between(WINDOW_START, WINDOW_END)
    return flares.loc[mask].sort_values("peak_time")


def load_predictions(cfg: DataConfig, model: str) -> tuple[pd.DataFrame, float]:
    """ค่าทำนายของ HARP ตัวอย่างในช่วงที่เลือก (คอลัมน์ ``prob``) + threshold ที่ freeze ไว้"""
    artifacts = ForecastArtifacts(model, cfg.paths.artifacts)
    path = artifacts.existing_predictions()
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path} — รัน backend/scripts/forecast/train.py --model {model} ก่อน")
    pred = pd.read_parquet(path).rename(columns={artifacts.prob_column: "prob"})
    pred["issue_time"] = pd.to_datetime(pred["issue_time"])
    sub = pred[(pred["HARPNUM"] == HARPNUM) & pred["issue_time"].between(WINDOW_START, WINDOW_END)]
    sub = sub.sort_values("issue_time")
    if sub.empty:
        raise SystemExit(f"ไม่มีผลทำนายของ HARP {HARPNUM} ในช่วง {WINDOW_START} – {WINDOW_END}")

    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    # threshold ของ val/test เป็นค่าเดียวกันเสมอ (freeze บน val แล้วใช้ซ้ำ — ดู configs/forecast.yaml)
    threshold = float(metrics[model]["val"]["threshold"])
    return sub, threshold


def pick_example_window(
    flares: pd.DataFrame, predictions: pd.DataFrame, horizon_hours: int
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Series]:
    """เลือก flare แรงสุดในช่วงเป็นตัวอย่าง แล้วหา issue_time ที่ออกคำพยากรณ์ก่อนหน้าพอดี
    ``horizon_hours`` ชม. — ปัดลงชั่วโมงเพราะ issue_time อยู่บนกริดรายชั่วโมง
    """
    if flares.empty:
        raise SystemExit("ไม่มี flare ในช่วงที่เลือก — เลือกช่วง/HARP ใหม่")
    main_flare = flares.loc[flares["peak_flux"].idxmax()]
    target_issue = main_flare["peak_time"].floor("h") - pd.Timedelta(hours=horizon_hours)

    if predictions.loc[predictions["issue_time"] == target_issue].empty:
        raise SystemExit(
            f"ไม่มี issue_time {target_issue} ในผลทำนายของ HARP {HARPNUM} — เลือกช่วงใหม่"
        )
    return target_issue, target_issue + pd.Timedelta(hours=horizon_hours), main_flare


# --------------------------------------------------------------------------- #
# รูป
# --------------------------------------------------------------------------- #
def _add_forecast_box(ax: plt.Axes, box_start: pd.Timestamp, box_end: pd.Timestamp, y0: float, y1: float) -> None:
    """กรอบคลุมเฉพาะช่วงเวลา + ช่วงค่าที่โมเดลทำนายจริง ไม่ใช่คลุมเต็มความสูงของแผง

    ไม่ใช้ ``axvspan`` เพราะมันคลุมเต็มความสูงเสมอ (ครอบคลุมตั้งแต่ค่าเงียบ/C-class ไปด้วย)
    ซึ่งไม่ตรงกับสิ่งที่โมเดลทำนายจริง — โมเดลทำนายแค่ "จะเกิด flare ≥ M1.0 หรือไม่"
    (ดู ``flare.positive_goes_class`` ใน ``data.yaml``) ความสูงของกรอบจึงต้องเริ่มที่
    เกณฑ์ตัดสินใจ (เส้น M ของฟลักซ์ หรือ threshold ของ probability) ไม่ใช่ขอบล่างสุด
    """
    x0 = mdates.date2num(box_start)
    x1 = mdates.date2num(box_end)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                            facecolor=BLUE, alpha=0.15, edgecolor=BLUE, linewidth=1.4, zorder=3))


def build_figure(
    flux: pd.DataFrame,
    flares: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float,
    box_start: pd.Timestamp,
    box_end: pd.Timestamp,
    main_flare: pd.Series,
    label: str,
) -> plt.Figure:
    fig, (ax_flux, ax_prob) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True, height_ratios=[2.4, 1.0])

    # -- แผงบน: ฟลักซ์ X-ray จริง (log scale) -------------------------------- #
    ax_flux.plot(flux["time"], flux["flux"], color="0.15", linewidth=0.9, zorder=4)
    for letter in ("C", "M", "X"):
        ax_flux.axhline(CLASS_BASE_FLUX[letter], color="0.4", linewidth=0.8, linestyle=":", zorder=2)
    ax_flux.set_yscale("log")
    ax_flux.set_ylim(*FLUX_YLIM)
    ax_flux.set_yticks([1e-7, 1e-6, 1e-5, 1e-4, 1e-3])
    secondary = ax_flux.secondary_yaxis("right")
    secondary.set_yticks([CLASS_BASE_FLUX[c] for c in ("C", "M", "X")], ["C", "M", "X"])
    secondary.set_ylabel("GOES class")
    secondary.tick_params(which="minor", length=0)

    for _, flare in flares.iterrows():
        if flare["goes_class"].startswith("X"):
            ax_flux.annotate(flare["goes_class"], xy=(flare["peak_time"], flare["peak_flux"]),
                             xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)

    _add_forecast_box(ax_flux, box_start, box_end, CLASS_BASE_FLUX["M"], FLUX_YLIM[1])
    box_mid = box_start + (box_end - box_start) / 2
    ax_flux.annotate(
        f"{label} 24-h forecast\n(predicts ≥ M1.0)",
        xy=(box_mid, CLASS_BASE_FLUX["M"] * 1.3), xytext=(box_mid, 1.6e-7),
        ha="center", va="bottom", fontsize=9, zorder=5,
        arrowprops=dict(arrowstyle="-", color=BLUE, linewidth=1.2),
    )
    ax_flux.set_ylabel("GOES X-ray flux (W/m²)")
    ax_flux.grid(axis="x", visible=False)

    # -- แผงล่าง: ความน่าจะเป็นของโมเดล -------------------------------------- #
    flags = predictions["label"].to_numpy() == 1
    times = predictions["issue_time"].to_numpy()
    for start_i, end_i in true_runs(flags):
        ax_prob.axvspan(times[start_i], times[end_i], color="0.88", zorder=1)
    ax_prob.plot(predictions["issue_time"], predictions["prob"], color=BLUE, linewidth=1.6, zorder=4)
    ax_prob.axhline(threshold, color=THRESHOLD, linewidth=1.1, linestyle="--", zorder=3)
    _add_forecast_box(ax_prob, box_start, box_end, threshold, PROB_YLIM[1])

    ax_prob.set(ylim=PROB_YLIM, yticks=[0.0, 0.5, 1.0], ylabel=f"P(M1+), {label}",
                xlabel="Forecast issue time / observation time (UTC)")
    ax_prob.xaxis.set_major_locator(mdates.DayLocator())
    ax_prob.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    ax_flux.set_title(
        f"{label} 24-h forecast vs. observed GOES X-ray flux: HARP {HARPNUM} (AR {int(main_flare['noaa_ar'])}), "
        f"{WINDOW_START:%d %b} – {WINDOW_END:%d %b %Y}",
        loc="left", fontweight="bold",
    )
    handles = [
        Line2D([], [], color="0.15", label="GOES X-ray flux (1-min)"),
        Line2D([], [], color=BLUE, label=f"Hourly {label} probability"),
        Line2D([], [], color=THRESHOLD, linestyle="--", label=f"Decision threshold ({threshold:.2f})"),
        Patch(facecolor="0.88", label="M1+ flare within next 24 h"),
        Patch(facecolor=BLUE, alpha=0.25, edgecolor=BLUE,
              label=f"Forecast window before {main_flare['goes_class']} (≥ decision level)"),
    ]
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    return fig


def save_outputs(
    fig: plt.Figure, out_dir: Path, model: str, flux: pd.DataFrame, predictions: pd.DataFrame, dpi: int
) -> Path:
    name = f"{model}_forecast_example"
    fig_dir, data_dir = out_dir / "figures", out_dir / "figures" / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    png = fig_dir / f"{name}.png"
    save_figure(fig, png, dpi)

    flux.to_csv(data_dir / f"{name}_flux.csv", index=False, encoding="utf-8-sig")
    predictions[["issue_time", "label", "prob"]].rename(columns={"prob": f"{model}_prob"}).to_csv(
        data_dir / f"{name}_prob.csv", index=False, encoding="utf-8-sig"
    )
    return png


def main() -> int:
    forecast_cfg = load_forecast_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=forecast_cfg.default_model, choices=forecast_cfg.names,
                        help=f"โมเดลที่จะวาด (ปริยาย {forecast_cfg.default_model})")
    args = parser.parse_args()
    entry = forecast_cfg.models[args.model]
    label = entry.label or architecture_label(entry.model.kind)

    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "plot_forecast_example.log")

    flux = load_flux(cfg)
    flares = load_flares(cfg)
    predictions, threshold = load_predictions(cfg, args.model)
    box_start, box_end, main_flare = pick_example_window(flares, predictions, cfg.flare.horizon_hours)

    logger.info(
        "HARP %d · %d จุดฟลักซ์ · %d แถวทำนาย · กรอบตัวอย่าง %s – %s (ก่อน %s ที่ %s)",
        HARPNUM, len(flux), len(predictions), box_start, box_end,
        main_flare["goes_class"], main_flare["peak_time"],
    )

    fig = build_figure(flux, flares, predictions, threshold, box_start, box_end, main_flare, label)
    png = save_outputs(fig, cfg.paths.artifacts, args.model, flux, predictions, dpi=200)
    logger.info("เขียน %s", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
