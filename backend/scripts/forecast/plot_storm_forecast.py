"""รูป "คำพยากรณ์ราย 24 ชม." ของบริเวณกัมมันต์หนึ่งดวง เทียบกับฟลักซ์ GOES X-ray จริง

    python backend/scripts/forecast/plot_storm_forecast.py                         # LSTM + V3 (อันดับ 1), HARP 13999
    python backend/scripts/forecast/plot_storm_forecast.py --start 2025-11-01 --end 2025-11-22
    python backend/scripts/forecast/plot_storm_forecast.py --cell tcn:V1
    python backend/scripts/forecast/plot_storm_forecast.py --production lstm       # แบบจำลองที่ระบบใช้งานจริง

ค่าตั้งต้นคือเซลล์อันดับ 1 ของงานเปรียบเทียบ (``artifacts/model_comparison``) ซึ่งเทรน 25 seed และออกคำพยากรณ์ทุก 12 ชม.
ในแต่ละช่องใช้ **ค่าเฉลี่ยความน่าจะเป็นของทุก seed** และถือว่าเตือนเมื่อ **seed ส่วนใหญ่** เตือน (แต่ละ seed ใช้ threshold
ของตัวเอง) · ``--production`` ใช้ค่าทำนายรายชั่วโมงของแบบจำลองในระบบแทน ·
``--start``/``--end`` กำหนดช่วงของแกนเวลา (ปริยาย: ตั้งแต่ HARP ปรากฏจนหายไป)

แผงเดียวตามรูปแบบกราฟฟลักซ์ GOES X-ray ของ NOAA: แกน y คือฟลักซ์ (W m⁻², log) มีเส้นกริดทุกทศนิยมและตัวอักษรคลาส
A/B/C/M/X ด้านขวา คำพยากรณ์จึงวาดบนสเกลเดียวกับสิ่งที่เกิดจริงได้ตรง ๆ · ตั้งใจให้มีองค์ประกอบน้อยที่สุด:

- **เส้น** = ฟลักซ์ GOES XRS ช่องยาว (1–8 Å) ทั้งดวง · จุด = flare ระดับ M/X ที่แคตตาล็อกระบุว่ามาจาก HARP นี้ ใส่ป้าย
  เฉพาะระดับ X (ยอดของเส้นที่ไม่มีจุดมาจากบริเวณอื่นบนดวง)
- **กล่องราย 24 ชม.** (แดงโปร่งแสงคลุมแถบคลาสที่ทำนายทั้งแถบ เช่น X = 10⁻⁴–10⁻³ W m⁻² — ดู ``_forecast_bar``)
  ต่อกันจากคำพยากรณ์แรก แต่ละช่องใช้คำพยากรณ์ที่ออก **ตอนต้นช่องพอดี** เท่านั้น วาดที่ **ระดับคลาส
  สูงสุดที่ทำนาย**: แดงที่ระดับ X = แบบจำลองระดับ X เตือนว่าจะเกิด flare ≥ X1.0 ภายใน 24 ชม. · แดงที่ระดับ M = เตือนแค่
  ≥ M1.0 · เทาที่ระดับ C = ไม่เตือน · **ทึบ = flare จาก HARP นี้ที่เกิดจริงถึงระดับที่ทำนาย, กลวง = ไม่ถึง** (ผลถูก/ผิด
  อ่านได้จากรูปทรง ไม่พึ่งสีอย่างเดียว) · ลายขีด = ไม่มีคำพยากรณ์
  แบบจำลองแต่ละตัวตอบแบบทวิภาค ระดับ X มาจากแบบจำลองแยกที่เทรนด้วย label ≥ X1.0 (``artifacts/model_comparison_x``
  สร้างด้วย ``study/build_dataset.py --positive-class X1.0``) — ถ้าไม่มีโฟลเดอร์นั้น แถบมีแค่ระดับ M/C
- ``--level M`` / ``--level X`` วาดแบบจำลองเดียว (รูปละระดับ): ระดับ M ไม่เตือนวาดที่ C · ระดับ X ไม่เตือนวาดที่ M
  (= "ต่ำกว่า X")
- ``--disk`` (ใช้กับ ``--level M`` หรือ ``X``) **พยากรณ์ทั้งดวงรายวันปฏิทิน**: ช่อง = 00:00–24:00 UT ใช้คำพยากรณ์ที่ออก
  00:00 ของวันนั้นของทุก HARP ที่มีคำพยากรณ์ · เตือน = มี HARP ใดที่ seed ส่วนใหญ่เตือน · เกิดจริง = มี flare ≥ ระดับนั้น
  ที่ใดก็ได้บนดวง (catalog ทั้งหมด) · จุดทึบ = flare จาก HARP ที่มีคำพยากรณ์วันนั้น, จุดกลวง = จาก HARP ที่ไม่มี (พลาด
  เพราะไม่มีข้อมูล ไม่ใช่เพราะแบบจำลอง) · วันที่ไม่มี HARP ใดมีคำพยากรณ์ = ลายขีดเทา
- โหมดแบบจำลองเดียว (``--level M``/``X``): กล่องแดงที่แถบคลาสเป้าหมาย = พยากรณ์ว่าเกิด · **กล่องน้ำเงินที่แถบต่ำกว่า =
  พยากรณ์ว่าไม่เกิด** · ทึบ = พยากรณ์ถูก, ลายขีด = ผิด
- ``--level all`` (ปริยาย) **พยากรณ์ 3 ระดับในรูปเดียว** รวมสองแบบจำลอง: X ถ้าแบบจำลองระดับ X เตือน, M ถ้าเตือนแค่
  แบบจำลองระดับ M, **"ไม่เกิด" (< M) ถ้าไม่เตือนทั้งคู่** — กล่องแดงที่แถบคลาส X/M = พยากรณ์ว่าเกิด, กล่องน้ำเงินที่แถบ
  คลาส C = พยากรณ์ว่าไม่เกิด ≥ M1.0 · ผลจริงของช่อง = คลาสสูงสุดของ flare จาก HARP นี้ในช่อง (ตาม label ≥ X1.0 / ≥ M1.0)
  · **ทึบ = ระดับที่พยากรณ์ตรงกับที่เกิดจริง, ลายขีด = ไม่ตรง** · ไฟล์ .txt/.csv มีตารางไขว้ 3×3
- แถบเทาอ่อน = ช่วงที่บริเวณนั้นอยู่นอกขอบเขตพยากรณ์ (|ลองจิจูด| > ``sharp.abs_lon_max_deg``) หรือยังไม่ปรากฏ

ค่าทำนายมาจาก ``artifacts/model_comparison/predictions.parquet`` หรือ ``artifacts/metrics/<ชื่อ>_predictions.parquet``
(``--production``) ฟลักซ์จาก
``sunseg.data.xray_flux.XrayFluxStore`` · ข้อความในรูปเป็นภาษาอังกฤษ · เขียนรูปลง
``artifacts/figures/<สถาปัตยกรรม>_<แบบ>_storm_forecast_harp<HARP>.png`` (หรือ ``<ชื่อ>_storm_forecast_…`` ของ
``--production``) และตารางช่อง/flare/ฟลักซ์ลง ``figures/data/``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# report_style ต้องมาก่อน pyplot — ตั้ง backend Agg + ธีม seaborn (ดู docstring ของโมดูลนั้น)
from sunseg.report_style import GREY, save_figure  # noqa: E402, I001

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from sunseg.artifacts import ForecastArtifacts  # noqa: E402
from sunseg.config import DataConfig, load_data_config, load_forecast_config  # noqa: E402
from sunseg.data.xray_flux import XrayFluxStore  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402
from sunseg.models.forecast import architecture_label  # noqa: E402

logger = logging.getLogger("plot_storm_forecast")

DEFAULT_HARP = 13999  # AR 14274 พ.ย. 2025 — ชุด test, flare X 5 ลูก (X1.8, X1.7, X1.2, X5.1, X4.0)
FLUX_YLIM = (1e-7, 1e-2)
LIMB_COLOR = "0.93"
ALARM_COLOR = "#d62728"  # แดงของคำเตือน — ตั้งใจใช้แดงจริง ไม่ใช่แดงอมส้มของ PALETTE
QUIET_COLOR = "0.55"     # แถบ "ไม่เตือน" (โหมดแบบจำลองเดียว)
NO_FLARE_COLOR = "#1f77b4"  # กล่อง "พยากรณ์ว่าไม่เกิด" ของโหมด 3 ระดับ — น้ำเงิน ตรงข้ามกับแดงของคำเตือน
CLASS_ORDER = ("<M", "M", "X")
CLASS_BAND = {"<M": "C", "M": "M", "X": "X"}  # แถบคลาสบนแกน y ที่ใช้วาดกล่องของแต่ละคำพยากรณ์
FLUX_COLOR = "0.20"      # เส้นฟลักซ์เป็นสีกลาง — แดงถูกจองไว้ให้คำเตือน
ALARM_ALPHA = 0.25       # กล่องคำเตือนโปร่งแสง คลุมแถบคลาสที่ทำนาย — เส้นฟลักซ์ต้องอ่านได้ผ่านกล่อง
BAR_GAP = pd.Timedelta(hours=2)  # ช่องว่างระหว่างแถบของช่องที่ติดกัน — แทนเส้นแบ่งช่อง
CLASS_BASE = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

LABELS = {"lstm": "LSTM", "tcn": "TCN", "transformer": "Transformer", "darnn": "DA-RNN"}


# --------------------------------------------------------------------------- #
# ข้อมูล
# --------------------------------------------------------------------------- #
def load_predictions(cfg: DataConfig, model: str, harp: int) -> tuple[pd.DataFrame, float]:
    artifacts = ForecastArtifacts(model, cfg.paths.artifacts)
    path = artifacts.existing_predictions()
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path} — รัน backend/scripts/forecast/train.py --model {model} ก่อน")
    pred = pd.read_parquet(path).rename(columns={artifacts.prob_column: "prob"})
    pred["issue_time"] = pd.to_datetime(pred["issue_time"])
    sub = pred[pred["HARPNUM"] == harp].sort_values("issue_time").reset_index(drop=True)
    if sub.empty:
        raise SystemExit(f"ไม่มีผลทำนายของ HARP {harp} ในไฟล์ของ {model}")
    metrics = json.loads(artifacts.metrics.read_text(encoding="utf-8"))
    threshold = float(metrics[model]["val"]["threshold"])  # freeze จาก validation ใช้ค่าเดียวกับ test
    return sub, threshold


def load_study_predictions(cfg: DataConfig, architecture: str, variant: str, harp: int,
                           study_dir: str = "model_comparison") -> tuple[pd.DataFrame, float]:
    """ค่าทำนายของเซลล์หนึ่งในงานเปรียบเทียบ รวมทุก seed เป็นหนึ่งแถวต่อ issue_time

    ``prob`` = ค่าเฉลี่ยข้าม seed · ``n_alarm`` = จำนวน seed ที่ความน่าจะเป็น ≥ threshold ของ seed นั้นเอง · ``n_seed``
    — threshold ที่คืนเป็นค่าเฉลี่ยข้าม seed ใช้แสดงผลเท่านั้น การตัดสินเตือน/ไม่เตือนใช้เสียงข้างมากของ seed
    """
    path = cfg.paths.artifacts / study_dir / "predictions.parquet"
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path} — รัน backend/scripts/study/train.py ก่อน")
    pred = pd.read_parquet(path, filters=[("architecture", "==", architecture), ("variant", "==", variant),
                                          ("HARPNUM", "==", harp)])
    if pred.empty:
        raise SystemExit(f"ไม่มีผลทำนายของ HARP {harp} ในเซลล์ ({architecture}, {variant})")
    pred["issue_time"] = pd.to_datetime(pred["issue_time"])
    pred["alarm"] = pred["prob"] >= pred["threshold"]
    per_time = (
        pred.groupby("issue_time")
        .agg(label=("label", "first"), noaa_ar=("noaa_ar", "first"), split=("split", "first"),
             prob=("prob", "mean"), n_alarm=("alarm", "sum"), n_seed=("seed", "nunique"))
        .reset_index()
    )
    return per_time, float(pred.groupby("seed")["threshold"].first().mean())


def load_track(cfg: DataConfig, harp: int) -> pd.DataFrame:
    """ลองจิจูดของ HARP รายชั่วโมงจาก SHARP keywords — ใช้หาช่วงที่อยู่นอกขอบเขตพยากรณ์"""
    sharp = pd.read_parquet(cfg.paths.interim / "sharp_keywords.parquet", columns=["HARPNUM", "t_rec", "LON_FWT"])
    track = sharp[sharp["HARPNUM"] == harp].sort_values("t_rec").reset_index(drop=True)
    track["t_rec"] = pd.to_datetime(track["t_rec"])
    return track


def load_flares(cfg: DataConfig, harp: int) -> pd.DataFrame:
    flares = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    flares["peak_time"] = pd.to_datetime(flares["peak_time"])
    mask = (flares["HARPNUM"] == harp) & flares["goes_class"].str[0].isin(["M", "X"])
    return flares.loc[mask].sort_values("peak_time").reset_index(drop=True)


def load_flux(cfg: DataConfig, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    store = XrayFluxStore(cfg.paths.raw / "xrs")
    if not store.available:
        raise SystemExit(f"ไม่พบไฟล์ XRS ที่ {store.root} — รัน backend/scripts/data/download_xray.py ก่อน")
    series = store.series(t0.to_pydatetime(), t1.to_pydatetime())
    frame = pd.DataFrame({"time": pd.to_datetime(series.times), "flux": series.flux}).dropna()
    if frame.empty:
        raise SystemExit(f"ไม่มีข้อมูล X-ray ในช่วง {t0} – {t1}")
    # ค่าสูงสุดราย 10 นาที: ลดจำนวนจุดโดยไม่ตัดยอดของ flare
    return frame.set_index("time")["flux"].resample("10min").max().dropna().reset_index()


def limb_spans(track: pd.DataFrame, max_abs_lon: float, t0: pd.Timestamp, t1: pd.Timestamp) -> list[tuple]:
    """ช่วงเวลาใน [t0, t1] ที่ HARP ยังไม่ปรากฏ/หายไปแล้ว หรือ |ลองจิจูด| เกินขอบเขต"""
    inside = track.loc[track["LON_FWT"].abs() <= max_abs_lon, "t_rec"]
    if inside.empty:
        return [(t0, t1)]
    first, last = inside.min(), inside.max()
    spans = []
    if first > t0:
        spans.append((t0, first))
    if last < t1:
        spans.append((last, t1))
    return spans


def daily_windows(pred: pd.DataFrame, threshold: float, horizon_hours: int) -> pd.DataFrame:
    """ช่อง 24 ชม. ต่อกันจากคำพยากรณ์แรก ใช้คำพยากรณ์ที่ออกตอนต้นช่องพอดีเท่านั้น"""
    step = pd.Timedelta(hours=horizon_hours)
    by_time = pred.set_index("issue_time")
    rows = []
    start = pred["issue_time"].min()
    while start <= pred["issue_time"].max():
        row = {"window_start": start, "window_end": start + step}
        if start in by_time.index:
            prob = float(by_time.at[start, "prob"])
            if "n_alarm" in by_time.columns:  # หลาย seed: เตือนเมื่อ seed ส่วนใหญ่เตือน
                n_alarm, n_seed = int(by_time.at[start, "n_alarm"]), int(by_time.at[start, "n_seed"])
                alarm = n_alarm * 2 > n_seed
                row.update(n_alarm=n_alarm, n_seed=n_seed)
            else:
                alarm = prob >= threshold
            row.update(prob=prob, alarm=alarm, event=bool(by_time.at[start, "label"]))
            if "n_alarm_x" in by_time.columns and pd.notna(by_time.at[start, "n_alarm_x"]):
                n_x, seeds_x = int(by_time.at[start, "n_alarm_x"]), int(by_time.at[start, "n_seed_x"])
                row.update(alarm_x=n_x * 2 > seeds_x, event_x=bool(by_time.at[start, "label_x"]),
                           prob_x=float(by_time.at[start, "prob_x"]), n_alarm_x=n_x, n_seed_x=seeds_x)
        else:
            row.update(prob=np.nan, alarm=None, event=None)
        rows.append(row)
        start += step
    return pd.DataFrame(rows)


def outcome(row) -> str:
    """ผลของช่องที่ระดับสูงสุดที่ทำนาย — ช่องที่แบบจำลองระดับ X เตือนตัดสินด้วย label ≥ X1.0"""
    if row["alarm"] is None:
        return "no forecast"
    if row.get("alarm_x") is True:
        return "x hit" if row["event_x"] else "x false alarm"
    if row["alarm"]:
        return "hit" if row["event"] else "false alarm"
    return "miss" if row["event"] else "correct negative"


def classify_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """โหมด 3 ระดับ: คลาสที่พยากรณ์ / คลาสที่เกิดจริง / ตรงกันหรือไม่ ของทุกช่อง (ช่องที่ไม่มีคำพยากรณ์ = NaN)"""
    out = windows.copy()
    has = out["alarm"].notna() & out["alarm_x"].notna()
    out["forecast_class"] = np.where(~has, None, np.where(out["alarm_x"] == True, "X",  # noqa: E712
                                                          np.where(out["alarm"] == True, "M", "<M")))  # noqa: E712
    out["observed_class"] = np.where(~has, None, np.where(out["event_x"] == True, "X",  # noqa: E712
                                                          np.where(out["event"] == True, "M", "<M")))  # noqa: E712
    out["correct"] = np.where(~has, None, out["forecast_class"] == out["observed_class"])
    return out


# --------------------------------------------------------------------------- #
# รูป
# --------------------------------------------------------------------------- #
def _class_box(ax: plt.Axes, w: pd.Series) -> None:
    """โหมด 3 ระดับ: กล่องที่แถบคลาสที่พยากรณ์ — แดง = เกิด (M/X) · น้ำเงิน = ไม่เกิด (แถบ C) · ทึบ = ตรง, ลายขีด = ไม่ตรง"""
    x0 = mdates.date2num(w["window_start"] + BAR_GAP / 2)
    x1 = mdates.date2num(w["window_end"] - BAR_GAP / 2)
    if w["forecast_class"] is None:
        ax.axvspan(w["window_start"] + BAR_GAP / 2, w["window_end"] - BAR_GAP / 2, facecolor="none",
                   edgecolor=GREY, hatch="////", linewidth=0, alpha=0.45, zorder=1)
        return
    base = CLASS_BASE[CLASS_BAND[w["forecast_class"]]]
    color = NO_FLARE_COLOR if w["forecast_class"] == "<M" else ALARM_COLOR
    if w["correct"]:
        style = dict(facecolor=color, alpha=ALARM_ALPHA)
    else:
        style = dict(facecolor="none", edgecolor=color, hatch="///", alpha=ALARM_ALPHA + 0.15)
    ax.add_patch(Rectangle((x0, base), x1 - x0, base * 9, linewidth=0, zorder=1, **style))
def _forecast_bar(ax: plt.Axes, w: pd.Series, top: str = "M", low: str = "C") -> None:
    """ช่องคำพยากรณ์หนึ่งช่อง เป็นกล่องโปร่งแสงคลุม **แถบคลาสที่ทำนายทั้งแถบ** บนแกน y (เช่น X = 10⁻⁴–10⁻³ W m⁻²)

    แดงที่แถบคลาส ``top`` = เตือน · เทาที่แถบคลาส ``low`` = ไม่เตือนแต่เกิด (พลาด) · ไม่เตือนและไม่เกิด = ว่าง ·
    **ทึบโปร่งแสง = เกิดจริงถึงระดับที่เตือน, ลายขีด = ไม่ถึง** (ถูก/ผิดอ่านจากรูปแบบ ไม่พึ่งสีอย่างเดียว) ·
    เส้นฟลักซ์อยู่เหนือกล่อง จึงเห็นได้ทันทีว่ายอดของ flare เข้าไปในกล่องหรือไม่
    """
    x0 = mdates.date2num(w["window_start"] + BAR_GAP / 2)
    x1 = mdates.date2num(w["window_end"] - BAR_GAP / 2)
    kind = w["outcome"]
    alarm_level = "X" if kind.startswith("x ") else top

    def box(level: str, **style) -> None:
        base = CLASS_BASE[level]
        ax.add_patch(Rectangle((x0, base), x1 - x0, base * 9, linewidth=0, zorder=1, **style))

    if kind == "no forecast":  # ไม่มีคำพยากรณ์ = ลายขีดเทาเต็มแกน y — ไม่ได้ทายระดับใดเลย
        ax.axvspan(w["window_start"] + BAR_GAP / 2, w["window_end"] - BAR_GAP / 2, facecolor="none",
                   edgecolor=GREY, hatch="////", linewidth=0, alpha=0.45, zorder=1)
    elif kind in ("hit", "x hit"):
        box(alarm_level, facecolor=ALARM_COLOR, alpha=ALARM_ALPHA)
    elif kind in ("false alarm", "x false alarm"):
        box(alarm_level, facecolor="none", edgecolor=ALARM_COLOR, hatch="///", alpha=ALARM_ALPHA + 0.15)
    elif kind == "correct negative":  # พยากรณ์ว่าไม่เกิด และไม่เกิดจริง
        box(low, facecolor=NO_FLARE_COLOR, alpha=ALARM_ALPHA)
    elif kind == "miss":  # พยากรณ์ว่าไม่เกิด แต่เกิด
        box(low, facecolor="none", edgecolor=NO_FLARE_COLOR, hatch="///", alpha=ALARM_ALPHA + 0.15)


def build_figure(flux: pd.DataFrame, flares: pd.DataFrame, windows: pd.DataFrame, limbs: list[tuple],
                 t0: pd.Timestamp, t1: pd.Timestamp, title: str, top: str = "M", low: str = "C") -> plt.Figure:
    """``top``/``low`` = ระดับของแถบเมื่อเตือน/ไม่เตือน (โหมดแบบจำลองเดียว) — โหมดรวมใช้ค่าปริยาย M/C"""
    y0, y1 = FLUX_YLIM
    days = max(1, (t1 - t0).days)
    # ช่วงยาวต้องกว้างขึ้น ไม่งั้นกล่องรายวันแคบจนอ่านไม่ออก (~0.16 นิ้วต่อวัน, 10-18 นิ้ว)
    fig, ax = plt.subplots(figsize=(min(18.0, max(10.0, 0.16 * days)), 5.0))

    for s, e in limbs:
        ax.axvspan(s, e, color=LIMB_COLOR, linewidth=0, zorder=0)
    three_class = "forecast_class" in windows
    for _, w in windows.iterrows():
        if three_class:
            _class_box(ax, w)
        else:
            _forecast_bar(ax, w, top, low)

    # ฟลักซ์จริง — ขาดเส้นเมื่อข้อมูลหายเกิน 1 ชม.
    gap = (flux["time"].diff() > pd.Timedelta(hours=1)).cumsum()
    for _, part in flux.groupby(gap.values):
        ax.plot(part["time"], part["flux"], color=FLUX_COLOR, linewidth=0.8, zorder=3)  # เหนือแถบ: เส้นต้องอ่านได้ตลอด

    # flare ของ HARP นี้ — จุดทุกลูก ป้ายเฉพาะระดับ X (ป้ายทุกลูกจะรกจนอ่านไม่ออก)
    covered = flares["covered"] if "covered" in flares else pd.Series(True, index=flares.index)
    ax.plot(flares.loc[covered, "peak_time"], flares.loc[covered, "peak_flux"], "o", markersize=4.5, color="black",
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.plot(flares.loc[~covered, "peak_time"], flares.loc[~covered, "peak_flux"], "o", markersize=4.5,
            markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0, zorder=5)
    # ป้ายของ X ที่เกิดห่างกันไม่ถึง 18 ชม. จะทับกัน — ยกป้ายลูกหลังขึ้นเหนือป้ายลูกก่อน
    prev_time, prev_y = None, 0.0
    for _, f in flares[flares["goes_class"].str[0] == "X"].sort_values("peak_time").iterrows():
        y = f["peak_flux"] * 1.5
        if prev_time is not None and f["peak_time"] - prev_time < pd.Timedelta(hours=18):
            y = max(y, prev_y * 2.2)
        ax.text(f["peak_time"], y, f["goes_class"], ha="center", va="bottom", fontsize=9, zorder=6)
        prev_time, prev_y = f["peak_time"], y

    # แกน y แบบกราฟ GOES: ทศนิยมละเส้น + ตัวอักษรคลาสกลางแถบด้านขวา
    ax.set_yscale("log")
    ax.set_ylim(y0, y1)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="y", color="0.82", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    for letter, base in CLASS_BASE.items():
        centre = base * np.sqrt(10)
        if y0 < centre < y1:
            ax.text(1.012, centre, letter, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=10, fontweight="bold", color="0.35")
    ax.set_ylabel("GOES 1–8 Å X-ray flux (W m$^{-2}$)")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Date (UTC)")
    if days <= 31:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    else:  # ช่วงยาว: ทุกวันที่ 1 และ 15 ของเดือน (+ ทุกสัปดาห์ถ้ายาวไม่เกิน ~3 เดือน)
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 8, 15, 22) if days <= 100 else (1, 15)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(title, loc="left", fontsize=11)

    # จุดเริ่มทำนาย — เส้นประแนวตั้ง + ป้ายตรงเส้น วางในแถบฟลักซ์ต่ำสุดซึ่งว่าง ไม่ชนป้ายของ flare/กล่องคำเตือน
    first = windows.loc[windows["outcome"] != "no forecast", "window_start"].iloc[0]
    ax.axvline(first, color="0.1", linestyle="--", linewidth=1.1, zorder=4)
    ax.text(first + pd.Timedelta(hours=3), y0 * 1.6, f"First forecast\n{first:%d %b %H:%M} UT", ha="left",
            va="bottom", fontsize=9, color="0.1", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))

    # ใส่เฉพาะสัญลักษณ์ที่ปรากฏจริงในรูป
    kinds = set(windows["outcome"])
    followed = "M/X" if top == "M" else "X"
    handles = [Line2D([], [], color=FLUX_COLOR, linewidth=1.2, label="X-ray flux (full disk)")]
    if "covered" in flares:
        handles += [Line2D([], [], marker="o", linestyle="none", color="black", markersize=5,
                           label="M/X flare, region forecast that day"),
                    Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor="black",
                           markersize=5, label="M/X flare, region without forecast")]
    else:
        handles.append(Line2D([], [], marker="o", linestyle="none", color="black", markersize=5,
                              label="M/X flare from this region"))
    legend_items = [
        ("x hit", Patch(facecolor=ALARM_COLOR, alpha=ALARM_ALPHA + 0.1, linewidth=0,
                        label="24-h alarm ≥X1.0, X flare followed")),
        ("x false alarm", Patch(facecolor="none", edgecolor=ALARM_COLOR, hatch="///", linewidth=0,
                                alpha=ALARM_ALPHA + 0.25, label="24-h alarm ≥X1.0, no X flare")),
        ("hit", Patch(facecolor=ALARM_COLOR, alpha=ALARM_ALPHA + 0.1, linewidth=0,
                      label=f"Forecast ≥{top}1.0 flare — occurred")),
        ("false alarm", Patch(facecolor="none", edgecolor=ALARM_COLOR, hatch="///", linewidth=0,
                              alpha=ALARM_ALPHA + 0.25, label=f"Forecast ≥{top}1.0 flare — did not occur")),
        ("correct negative", Patch(facecolor=NO_FLARE_COLOR, alpha=ALARM_ALPHA + 0.1, linewidth=0,
                                   label=f"Forecast no ≥{top}1.0 flare — none occurred")),
        ("miss", Patch(facecolor="none", edgecolor=NO_FLARE_COLOR, hatch="///", linewidth=0, alpha=ALARM_ALPHA + 0.25,
                       label=f"Forecast no ≥{top}1.0 flare — {followed} occurred (miss)")),
        ("no forecast", Patch(facecolor="none", edgecolor=GREY, hatch="////", linewidth=0,
                              label="No forecast (data gap)")),
    ]
    if three_class:
        present = {(c, bool(ok)) for c, ok in zip(windows["forecast_class"], windows["correct"], strict=True)
                   if c is not None}
        flare_fc = {ok for c, ok in present if c != "<M"}
        quiet_fc = {ok for c, ok in present if c == "<M"}
        solid = dict(alpha=ALARM_ALPHA + 0.1, linewidth=0)
        hatch = dict(facecolor="none", hatch="///", linewidth=0, alpha=ALARM_ALPHA + 0.25)
        if True in flare_fc:
            handles.append(Patch(facecolor=ALARM_COLOR, label="Forecast M or X flare (box at that class) — correct", **solid))
        if False in flare_fc:
            handles.append(Patch(edgecolor=ALARM_COLOR, label="Forecast M or X flare — wrong class", **hatch))
        if True in quiet_fc:
            handles.append(Patch(facecolor=NO_FLARE_COLOR, label="Forecast no ≥M1.0 flare (box at C) — correct", **solid))
        if False in quiet_fc:
            handles.append(Patch(edgecolor=NO_FLARE_COLOR, label="Forecast no ≥M1.0 flare — M/X occurred (miss)", **hatch))
        if windows["forecast_class"].isna().any():
            handles.append(Patch(facecolor="none", edgecolor=GREY, hatch="////", linewidth=0,
                                 label="No forecast (data gap)"))
    else:
        handles += [patch for kind, patch in legend_items if kind in kinds]
    if limbs:
        handles.append(Patch(facecolor=LIMB_COLOR, label="Outside forecast range (near limb)"))

    # คำอธิบายสัญลักษณ์อยู่มุมขวาบนในกราฟ — แถบ 10⁻³–10⁻² ว่างเกือบเสมอ (flare เกิน X10 หายากมาก)
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="0.8", borderpad=0.5, columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    return fig


def outcome_label(outcome: str, lvl: str) -> str:
    """ผลของช่องเป็นข้อความที่ระบุระดับเสมอ — "no flare" เฉย ๆ ผิดเมื่อมี flare ต่ำกว่าระดับที่เตือนเกิดในช่อง"""
    if outcome.startswith("x "):
        outcome, lvl = outcome[2:], "X"
    return {
        "hit": f"HIT (alarm, >={lvl}1.0 followed)",
        "false alarm": f"FALSE ALARM (alarm, no >={lvl}1.0)",
        "miss": f"MISS (no alarm, >={lvl}1.0 followed)",
        "correct negative": f"CORRECT NEGATIVE (no >={lvl}1.0)",
        "no forecast": "NO FORECAST (data gap)",
    }[outcome]


def text_summary(windows: pd.DataFrame, flares: pd.DataFrame, title: str, level: str, threshold: float,
                 t0: pd.Timestamp, t1: pd.Timestamp, limbs: list[tuple]) -> str:
    """ผลชุดเดียวกับรูปในรูปแบบข้อความ — หนึ่งแถวต่อช่อง 24 ชม. + flare ในช่อง + สรุป + flare ที่อยู่นอกช่วงพยากรณ์

    flare ในช่องนับแบบเดียวกับ label คือ peak อยู่ในช่วง ``(ต้นช่อง, ปลายช่อง]`` และถึงระดับที่เตือน (ระดับ M นับ M/X,
    ระดับ X นับ X) — flare ที่ต่ำกว่าระดับแสดงไว้ในวงเล็บเพื่อให้อ่านเตือนผิดออก
    """
    lvl = "X" if level == "X" else "M"
    rule = "majority of seeds alarm" if "n_seed" in windows else f"probability >= threshold {threshold:.3f}"
    lines = [
        title,
        "=" * len(title),
        "",
        f"Forecast target : flare >= {lvl}1.0 from this active region within the next 24 h",
        f"Alarm rule      : {rule}"
        + (f" (each seed uses its own validation-frozen threshold; mean {threshold:.3f})" if "n_seed" in windows else ""),
        "Windows         : consecutive 24-h windows from the first forecast; each uses the forecast issued at its start",
        f"Plot range      : {t0:%Y-%m-%d} to {(t1 - pd.Timedelta(days=1)):%Y-%m-%d} (UTC)",
        f"First forecast  : {windows['window_start'].iloc[0]:%Y-%m-%d %H:%M} UT",
    ]
    for s0, s1 in limbs:
        lines.append(f"Outside range   : {s0:%Y-%m-%d %H:%M} to {s1:%Y-%m-%d %H:%M} UT (near limb or not yet visible)")
    lines += [
        "",
        f"{'#':>2}  {'window start (UT)':<17}  {'window end (UT)':<17}  {'alarm':<5}  {'seeds':>5}  {'P mean':>6}  "
        f"{'result':<38}  flares from this region in window (peak UT)",
        "-" * 150,
    ]
    for i, w in enumerate(windows.itertuples(index=False), start=1):
        inside = flares[(flares["peak_time"] > w.window_start) & (flares["peak_time"] <= w.window_end)]
        reached = inside["goes_class"].str[0].isin(["M", "X"] if lvl == "M" else ["X"])
        listed = [f"{f.goes_class} {f.peak_time:%m-%d %H:%M}" if ok else f"({f.goes_class} {f.peak_time:%m-%d %H:%M})"
                  for f, ok in zip(inside.itertuples(index=False), reached, strict=True)]
        seeds = (f"{int(w.n_alarm)}/{int(w.n_seed)}" if "n_seed" in windows and pd.notna(getattr(w, "n_alarm", np.nan))
                 else "-")
        prob = "-" if pd.isna(w.prob) else f"{w.prob:.3f}"
        alarm = "-" if w.alarm is None or pd.isna(w.alarm) else ("YES" if w.alarm else "no")
        lines.append(f"{i:>2}  {w.window_start:%Y-%m-%d %H:%M}  {w.window_end:%Y-%m-%d %H:%M}  {alarm:<5}  {seeds:>5}  "
                     f"{prob:>6}  {outcome_label(w.outcome, lvl):<38}  {', '.join(listed) or '-'}")

    counts = windows["outcome"].str.replace("x ", "", regex=False).value_counts()
    lines += [
        "",
        f"Summary: {len(windows)} windows · hits {int(counts.get('hit', 0))} · false alarms "
        f"{int(counts.get('false alarm', 0))} · misses {int(counts.get('miss', 0))} · correct negatives "
        f"{int(counts.get('correct negative', 0))} · no forecast {int(counts.get('no forecast', 0))}",
        "Flares in parentheses are below the forecast level and do not count as events for this target.",
    ]
    start, end = windows["window_start"].iloc[0], windows["window_end"].iloc[-1]
    outside = flares[(flares["peak_time"] <= start) | (flares["peak_time"] > end)]
    outside = outside[(outside["peak_time"] >= t0) & (outside["peak_time"] < t1)]
    if not outside.empty:
        lines += ["", "M/X flares from this region outside the forecast windows (before the first forecast or near limb):"]
        lines += [f"  {f.goes_class:<5} peak {f.peak_time:%Y-%m-%d %H:%M} UT" for f in outside.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def excel_table(windows: pd.DataFrame, flares: pd.DataFrame, level: str) -> pd.DataFrame:
    """ตารางเดียวกับไฟล์ .txt สำหรับเปิดใน Excel — หนึ่งแถวต่อช่อง 24 ชม. คอลัมน์แยกค่าให้กรอง/คำนวณต่อได้"""
    lvl = "X" if level == "X" else "M"
    rows = []
    for i, w in enumerate(windows.itertuples(index=False), start=1):
        inside = flares[(flares["peak_time"] > w.window_start) & (flares["peak_time"] <= w.window_end)]
        reached = inside["goes_class"].str[0].isin(["M", "X"] if lvl == "M" else ["X"])
        fmt = [f"{f.goes_class} {f.peak_time:%Y-%m-%d %H:%M}" for f in inside.itertuples(index=False)]
        alarm = None if w.alarm is None or pd.isna(w.alarm) else bool(w.alarm)
        rows.append({
            "window": i,
            "window_start_UT": f"{w.window_start:%Y-%m-%d %H:%M}",
            "window_end_UT": f"{w.window_end:%Y-%m-%d %H:%M}",
            "forecast_level": f">={lvl}1.0",
            "alarm": "" if alarm is None else ("YES" if alarm else "NO"),
            "seeds_alarm": int(w.n_alarm) if "n_seed" in windows and pd.notna(getattr(w, "n_alarm", np.nan)) else "",
            "seeds_total": int(w.n_seed) if "n_seed" in windows and pd.notna(getattr(w, "n_seed", np.nan)) else "",
            "probability_mean": "" if pd.isna(w.prob) else round(float(w.prob), 3),
            "result": outcome_label(w.outcome, lvl).split(" (")[0],
            "flare_at_forecast_level": "YES" if reached.any() else "NO",
            "flares_at_or_above_level": "; ".join(t for t, ok in zip(fmt, reached, strict=True) if ok),
            "flares_below_level": "; ".join(t for t, ok in zip(fmt, reached, strict=True) if not ok),
        })
    return pd.DataFrame(rows)


def _window_flares(flares: pd.DataFrame, w) -> str:
    inside = flares[(flares["peak_time"] > w.window_start) & (flares["peak_time"] <= w.window_end)]
    return "; ".join(f"{f.goes_class} {f.peak_time:%Y-%m-%d %H:%M}" for f in inside.itertuples(index=False))


def _seeds(w, suffix: str = "") -> str:
    n, total = getattr(w, f"n_alarm{suffix}", np.nan), getattr(w, f"n_seed{suffix}", np.nan)
    return "" if pd.isna(n) else f"{int(n)}/{int(total)}"


def contingency(windows: pd.DataFrame) -> pd.DataFrame:
    """ตารางไขว้ 3×3: แถว = คลาสที่พยากรณ์, คอลัมน์ = คลาสที่เกิดจริง"""
    has = windows["forecast_class"].notna()
    table = pd.crosstab(windows.loc[has, "forecast_class"], windows.loc[has, "observed_class"])
    return table.reindex(index=list(CLASS_ORDER), columns=list(CLASS_ORDER), fill_value=0)


def text_summary_3class(windows: pd.DataFrame, flares: pd.DataFrame, title: str, t0: pd.Timestamp,
                        t1: pd.Timestamp, limbs: list[tuple]) -> str:
    lines = [
        title, "=" * len(title), "",
        "Forecast        : 3 classes per 24-h window — X if the >=X1.0 model alarms, M if only the >=M1.0 model alarms,",
        "                  <M (no >=M1.0 flare) if neither alarms · each model = majority vote of its seeds",
        "Observed        : maximum class of flares from this active region within the window (same rule as the labels)",
        "Windows         : consecutive 24-h windows from the first forecast; each uses the forecast issued at its start",
        f"Plot range      : {t0:%Y-%m-%d} to {(t1 - pd.Timedelta(days=1)):%Y-%m-%d} (UTC)",
        f"First forecast  : {windows['window_start'].iloc[0]:%Y-%m-%d %H:%M} UT",
    ]
    for s0, s1 in limbs:
        lines.append(f"Outside range   : {s0:%Y-%m-%d %H:%M} to {s1:%Y-%m-%d %H:%M} UT (near limb or not yet visible)")
    lines += ["",
              f"{'#':>2}  {'window start (UT)':<17}  {'window end (UT)':<17}  {'forecast':<8}  {'observed':<8}  "
              f"{'result':<7}  {'P(>=M)':>6} {'seeds':>5}  {'P(>=X)':>6} {'seeds':>5}  flares from this region (peak UT)",
              "-" * 150]
    for i, w in enumerate(windows.itertuples(index=False), start=1):
        fc = w.forecast_class or "-"
        ob = w.observed_class or "-"
        result = "-" if w.correct is None else ("OK" if w.correct else "WRONG")
        pm = "-" if pd.isna(w.prob) else f"{w.prob:.3f}"
        px = "-" if pd.isna(getattr(w, "prob_x", np.nan)) else f"{w.prob_x:.3f}"
        lines.append(f"{i:>2}  {w.window_start:%Y-%m-%d %H:%M}  {w.window_end:%Y-%m-%d %H:%M}  {fc:<8}  {ob:<8}  "
                     f"{result:<7}  {pm:>6} {_seeds(w):>5}  {px:>6} {_seeds(w, '_x'):>5}  "
                     f"{_window_flares(flares, w) or '-'}")
    table = contingency(windows)
    n = int(table.to_numpy().sum())
    correct = int(np.trace(table.to_numpy()))
    lines += ["", f"Correct class: {correct} of {n} windows", "",
              "Contingency (rows = forecast, columns = observed):",
              f"{'':>10}" + "".join(f"{c:>8}" for c in CLASS_ORDER)]
    for fc in CLASS_ORDER:
        lines.append(f"{fc:>10}" + "".join(f"{int(table.loc[fc, ob]):>8}" for ob in CLASS_ORDER))
    start, end = windows["window_start"].iloc[0], windows["window_end"].iloc[-1]
    outside = flares[((flares["peak_time"] <= start) | (flares["peak_time"] > end))
                     & (flares["peak_time"] >= t0) & (flares["peak_time"] < t1)]
    if not outside.empty:
        lines += ["", "M/X flares from this region outside the forecast windows (before the first forecast or near limb):"]
        lines += [f"  {f.goes_class:<5} peak {f.peak_time:%Y-%m-%d %H:%M} UT" for f in outside.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def excel_table_3class(windows: pd.DataFrame, flares: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, w in enumerate(windows.itertuples(index=False), start=1):
        rows.append({
            "window": i,
            "window_start_UT": f"{w.window_start:%Y-%m-%d %H:%M}",
            "window_end_UT": f"{w.window_end:%Y-%m-%d %H:%M}",
            "forecast_class": w.forecast_class or "",
            "observed_class": w.observed_class or "",
            "correct": "" if w.correct is None else ("YES" if w.correct else "NO"),
            "p_M_mean": "" if pd.isna(w.prob) else round(float(w.prob), 3),
            "seeds_alarm_M": _seeds(w),
            "p_X_mean": "" if pd.isna(getattr(w, "prob_x", np.nan)) else round(float(w.prob_x), 3),
            "seeds_alarm_X": _seeds(w, "_x"),
            "flares_in_window": _window_flares(flares, w),
        })
    return pd.DataFrame(rows)


def load_study_all(cfg: DataConfig, architecture: str, variant: str, study_dir: str, t0: pd.Timestamp,
                   t1: pd.Timestamp) -> pd.DataFrame:
    """คำพยากรณ์ของทุก HARP ในช่วงเวลา รวม seed เป็นหนึ่งแถวต่อ (HARPNUM, issue_time)"""
    path = cfg.paths.artifacts / study_dir / "predictions.parquet"
    if not path.exists():
        raise SystemExit(f"ไม่พบ {path}")
    pred = pd.read_parquet(path, filters=[("architecture", "==", architecture), ("variant", "==", variant)])
    pred["issue_time"] = pd.to_datetime(pred["issue_time"])
    pred = pred[(pred["issue_time"] >= t0) & (pred["issue_time"] < t1)]
    pred["alarm"] = pred["prob"] >= pred["threshold"]
    return (pred.groupby(["HARPNUM", "issue_time"])
            .agg(prob=("prob", "mean"), n_alarm=("alarm", "sum"), n_seed=("seed", "nunique"))
            .reset_index())


def disk_windows(per_harp: pd.DataFrame, flares_all: pd.DataFrame, level: str, t0: pd.Timestamp,
                 t1: pd.Timestamp) -> pd.DataFrame:
    """ช่องรายวันปฏิทินของการพยากรณ์ทั้งดวง — เตือนเมื่อ HARP ใดที่ seed ส่วนใหญ่เตือน · เกิดจริงเมื่อมี flare ≥ ระดับ
    ที่ใดก็ได้บนดวงใน ``(ต้นช่อง, ปลายช่อง]`` (ขอบเดียวกับ label)"""
    threshold = CLASS_BASE[level]
    rows = []
    for start in pd.date_range(t0, t1 - pd.Timedelta(days=1), freq="D"):
        end = start + pd.Timedelta(days=1)
        at = per_harp[per_harp["issue_time"] == start]
        alarming = at[at["n_alarm"] * 2 > at["n_seed"]]
        inside = flares_all[(flares_all["peak_time"] > start) & (flares_all["peak_time"] <= end)
                            & (flares_all["peak_flux"] >= threshold)]
        row = {"window_start": start, "window_end": end, "n_regions": len(at), "n_regions_alarm": len(alarming),
               "alarm_harps": ",".join(str(int(h)) for h in alarming["HARPNUM"]),
               "event": bool(len(inside))}
        if at.empty:
            row.update(prob=np.nan, alarm=None)
        else:
            row.update(prob=float(at["prob"].max()), alarm=bool(len(alarming)))
        rows.append(row)
    return pd.DataFrame(rows)


def text_summary_disk(windows: pd.DataFrame, flares: pd.DataFrame, title: str, level: str) -> str:
    lines = [
        title, "=" * len(title), "",
        f"Forecast target : flare >= {level}1.0 anywhere on the visible disk within the calendar day (00-24 UT)",
        "Alarm rule      : YES if any active region's forecast issued at 00:00 UT is an alarm (majority of 25 seeds)",
        "Observed        : any >= " + level + "1.0 flare on the disk in the flare catalog during that day",
        "Regions         : number of active regions with a forecast at 00:00 UT (the model needs 4 days of complete",
        "                  SHARP + AIA intensity + X-ray history; regions without it have no forecast)",
        "", f"{'date (UT)':<10}  {'forecast':<8}  {'regions':>7}  {'alarming':>8}  {'max P':>6}  {'observed':<8}  "
        f"{'result':<16}  M/X flares that day (peak UT, * = region had no forecast)", "-" * 140,
    ]
    for w in windows.itertuples(index=False):
        day = flares[(flares["peak_time"] > w.window_start) & (flares["peak_time"] <= w.window_end)]
        listed = ", ".join(f"{f.goes_class} {f.peak_time:%H:%M}{'' if f.covered else '*'}" for f in day.itertuples())
        fc = "-" if w.alarm is None else ("YES" if w.alarm else "NO")
        result = {"hit": "HIT", "false alarm": "FALSE ALARM", "miss": "MISS", "correct negative": "CORRECT NEG",
                  "no forecast": "NO FORECAST"}[w.outcome]
        prob = "-" if pd.isna(w.prob) else f"{w.prob:.3f}"
        lines.append(f"{w.window_start:%Y-%m-%d}  {fc:<8}  {w.n_regions:>7}  {w.n_regions_alarm:>8}  {prob:>6}  "
                     f"{'YES' if w.event else 'NO':<8}  {result:<16}  {listed or '-'}")
    counts = windows["outcome"].value_counts()
    lines += ["", "Summary: " + " · ".join(f"{k} {int(counts.get(k, 0))}" for k in
                                         ("hit", "false alarm", "miss", "correct negative", "no forecast"))]
    return "\n".join(lines) + "\n"


def excel_table_disk(windows: pd.DataFrame, flares: pd.DataFrame, level: str) -> pd.DataFrame:
    rows = []
    for w in windows.itertuples(index=False):
        day = flares[(flares["peak_time"] > w.window_start) & (flares["peak_time"] <= w.window_end)]
        rows.append({
            "date_UT": f"{w.window_start:%Y-%m-%d}",
            "forecast_level": f">={level}1.0",
            "forecast": "" if w.alarm is None else ("YES" if w.alarm else "NO"),
            "regions_with_forecast": w.n_regions,
            "regions_alarming": w.n_regions_alarm,
            "alarming_HARPs": w.alarm_harps,
            "max_probability": "" if pd.isna(w.prob) else round(float(w.prob), 3),
            "observed": "YES" if w.event else "NO",
            "result": w.outcome.upper(),
            "flares_MX_that_day": "; ".join(f"{f.goes_class} {f.peak_time:%H:%M} HARP {int(f.HARPNUM)}"
                                            f"{'' if f.covered else ' (no forecast)'}" for f in day.itertuples()),
        })
    return pd.DataFrame(rows)


def main_disk(args, cfg) -> int:
    architecture, variant = args.cell.split(":")
    level = args.level
    label = f"{LABELS.get(architecture, architecture)} + {variant}"
    if args.start is None or args.end is None:
        raise SystemExit("--disk ต้องระบุ --start และ --end")
    t0, t1 = args.start.floor("D"), args.end.floor("D") + pd.Timedelta(days=1)
    study_dir = args.x_dir if level == "X" else "model_comparison"
    per_harp = load_study_all(cfg, architecture, variant, study_dir, t0, t1)

    flares_all = pd.read_parquet(cfg.paths.interim / "flares.parquet")
    flares_all["peak_time"] = pd.to_datetime(flares_all["peak_time"])
    flares_all = flares_all[(flares_all["peak_time"] >= t0) & (flares_all["peak_time"] < t1)]
    windows = disk_windows(per_harp, flares_all, level, t0, t1)
    windows["outcome"] = windows.apply(outcome, axis=1)

    # จุดของ flare: ทึบถ้า HARP ของมันมีคำพยากรณ์ตอน 00:00 ของวันที่ flare เกิด
    flares = flares_all[flares_all["goes_class"].str[0].isin(["M", "X"])].copy()
    day_start = flares["peak_time"].dt.floor("D")
    day_start = day_start.where(flares["peak_time"] != day_start, day_start - pd.Timedelta(days=1))
    forecast_keys = set(zip(per_harp["HARPNUM"].astype(int), per_harp["issue_time"], strict=True))
    flares["covered"] = [pd.notna(h) and (int(h), d) in forecast_keys
                         for h, d in zip(flares["HARPNUM"], day_start, strict=True)]
    flares = flares.sort_values("peak_time").reset_index(drop=True)

    logger.info("ทั้งดวง %s–%s · %s ≥%s1.0 · ช่อง: %s", t0.date(), (t1 - pd.Timedelta(days=1)).date(), label, level,
                windows["outcome"].value_counts().to_dict())
    title = (f"Full-disk daily forecasts of ≥{level}1.0 flares, {t0:%d %b}–{t1 - pd.Timedelta(days=1):%d %b %Y}: "
             f"{label} (≥{level}1.0 model, majority vote of 25 seeds)")
    top, low = (("X", "M") if level == "X" else ("M", "C"))
    flux = load_flux(cfg, t0, t1)
    fig = build_figure(flux, flares, windows, [], t0, t1, title, top, low)
    fig_dir = cfg.paths.artifacts / "figures"
    name = f"{architecture}_{variant}_{level}_disk_forecast_{t0:%Y%m%d}_{t1 - pd.Timedelta(days=1):%Y%m%d}"
    save_figure(fig, fig_dir / f"{name}.png")
    (fig_dir / f"{name}.txt").write_text(text_summary_disk(windows, flares, title, level), encoding="utf-8")
    excel_table_disk(windows, flares, level).to_csv(fig_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    logger.info("เขียน %s (.png .txt .csv)", fig_dir / name)
    return 0


def main() -> int:
    forecast_cfg = load_forecast_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", default="lstm:V3",
                        help="เซลล์ของงานเปรียบเทียบ <สถาปัตยกรรม>:<แบบ> (ปริยาย lstm:V3 — อันดับ 1)")
    parser.add_argument("--production", choices=forecast_cfg.names,
                        help="ใช้แบบจำลองที่ระบบใช้งานจริงตัวนี้แทนเซลล์ของงานเปรียบเทียบ")
    parser.add_argument("--x-dir", default="model_comparison_x",
                        help="โฟลเดอร์ใต้ artifacts ที่เก็บผลของแบบจำลองระดับ X เซลล์เดียวกัน (ไม่มี = ไม่วาดระดับ X)")
    parser.add_argument("--level", choices=("all", "M", "X"), default="all",
                        help="all = พยากรณ์ 3 ระดับ (ไม่เกิด / M / X) ในรูปเดียว · M หรือ X = แบบจำลองระดับนั้นอย่างเดียว")
    parser.add_argument("--harp", type=int, default=DEFAULT_HARP)
    parser.add_argument("--start", type=pd.Timestamp, help="วันแรกของแกนเวลา (ปริยาย: วันที่ HARP ปรากฏ)")
    parser.add_argument("--end", type=pd.Timestamp, help="วันสุดท้ายของแกนเวลา รวมทั้งวัน (ปริยาย: วันที่ HARP หายไป)")
    parser.add_argument("--disk", action="store_true",
                        help="พยากรณ์ทั้งดวงรายวันปฏิทิน (ใช้กับ --level M หรือ X และต้องมี --start/--end)")
    args = parser.parse_args()

    cfg = load_data_config()
    setup_logging(log_file=cfg.paths.artifacts / "logs" / "plot_storm_forecast.log")
    if args.disk:
        if args.level not in ("M", "X"):
            raise SystemExit("--disk ใช้กับ --level M หรือ --level X")
        return main_disk(args, cfg)

    if args.production:
        entry = forecast_cfg.models[args.production]
        label = entry.label or architecture_label(entry.model.kind)
        pred, threshold = load_predictions(cfg, args.production, args.harp)
        name = f"{args.production}_storm_forecast_harp{args.harp}"
    else:
        architecture, variant = args.cell.split(":")
        label = f"{LABELS.get(architecture, architecture)} + {variant}"
        if args.level == "X":  # แบบจำลองระดับ X อย่างเดียว — ใช้ผลของมันเป็นคำพยากรณ์หลัก
            pred, threshold = load_study_predictions(cfg, architecture, variant, args.harp, args.x_dir)
        else:
            pred, threshold = load_study_predictions(cfg, architecture, variant, args.harp)
        suffix = "" if args.level == "all" else f"_{args.level}"
        name = f"{architecture}_{variant}{suffix}_storm_forecast_harp{args.harp}"
        if args.level == "all" and (cfg.paths.artifacts / args.x_dir / "predictions.parquet").exists():
            pred_x, threshold_x = load_study_predictions(cfg, architecture, variant, args.harp, args.x_dir)
            pred = pred.merge(
                pred_x[["issue_time", "label", "prob", "n_alarm", "n_seed"]].rename(
                    columns={"label": "label_x", "prob": "prob_x", "n_alarm": "n_alarm_x", "n_seed": "n_seed_x"}),
                on="issue_time", how="left")
            logger.info("รวมแบบจำลองระดับ X จาก %s (threshold เฉลี่ย %.3f)", args.x_dir, threshold_x)
        else:
            logger.info("ไม่พบผลของแบบจำลองระดับ X ที่ %s — วาดแค่ระดับ M/C", args.x_dir)
    track = load_track(cfg, args.harp)
    flares = load_flares(cfg, args.harp)
    noaa = int(pred["noaa_ar"].mode().iat[0])
    t0 = args.start.floor("D") if args.start is not None else track["t_rec"].min().floor("D")
    t1 = args.end.floor("D") + pd.Timedelta(days=1) if args.end is not None else track["t_rec"].max().ceil("D")
    limbs = limb_spans(track, cfg.sharp.abs_lon_max_deg, t0, t1)
    flux = load_flux(cfg, t0, t1)
    windows = daily_windows(pred, threshold, cfg.flare.horizon_hours)
    windows["outcome"] = windows.apply(outcome, axis=1)
    three_class = args.level == "all" and "alarm_x" in windows
    if three_class:
        windows = classify_windows(windows)

    logger.info("HARP %d (AR %d) · %s · threshold %.3f · %d flare M/X · ช่อง 24 ชม.: %s", args.harp, noaa,
                label, threshold, len(flares), windows["outcome"].value_counts().to_dict())
    for _, w in windows.iterrows():
        logger.info("  %s – %s  P=%s  %s", w["window_start"], w["window_end"],
                    "n/a" if np.isnan(w["prob"]) else f"{w['prob']:.3f}", w["outcome"])

    if "n_seed" in windows:
        if three_class:
            title = (f"AR {noaa} (HARP {args.harp}): daily 24-h class forecasts (no flare / M / X) of {label} "
                     f"(majority vote of {int(windows['n_seed'].max())} seeds)")
        else:
            models = "" if args.level == "all" or args.production else f"≥{args.level}1.0 model, "
            title = (f"AR {noaa} (HARP {args.harp}): daily 24-h forecasts of {label} "
                     f"({models}majority vote of {int(windows['n_seed'].max())} seeds)")
    else:
        title = f"AR {noaa} (HARP {args.harp}): daily 24-h forecasts of {label}"
    top, low = ("X", "M") if args.level == "X" else ("M", "C")
    fig = build_figure(flux, flares, windows, limbs, t0, t1, title, top, low)
    fig_dir = cfg.paths.artifacts / "figures"
    data_dir = fig_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    save_figure(fig, png)
    txt = fig_dir / f"{name}.txt"
    summary = (text_summary_3class(windows, flares, title, t0, t1, limbs) if three_class
               else text_summary(windows, flares, title, args.level, threshold, t0, t1, limbs))
    txt.write_text(summary, encoding="utf-8")
    logger.info("เขียน %s", txt)
    # UTF-8 มี BOM เพื่อให้ Excel อ่าน ≥ และตัวอักษรอื่นถูก
    csv = fig_dir / f"{name}.csv"
    table = excel_table_3class(windows, flares) if three_class else excel_table(windows, flares, args.level)
    table.to_csv(csv, index=False, encoding="utf-8-sig")
    logger.info("เขียน %s", csv)
    windows.assign(threshold=threshold).to_csv(data_dir / f"{name}_windows.csv", index=False, encoding="utf-8-sig")
    flares[["peak_time", "goes_class", "peak_flux", "noaa_ar", "HARPNUM"]].to_csv(
        data_dir / f"{name}_flares.csv", index=False, encoding="utf-8-sig")
    flux.to_csv(data_dir / f"{name}_xray_flux_10min.csv", index=False, encoding="utf-8-sig")
    logger.info("เขียน %s", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
