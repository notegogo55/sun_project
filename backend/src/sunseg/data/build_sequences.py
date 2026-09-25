"""แปลง SHARP keywords + flare catalog เป็น sequence dataset สำหรับ LSTM

กระบวนการ::

    SHARP keywords (ตารางดิบจาก JSOC)
      -> clean_sharp_frame()      กรอง QUALITY / |LON| / NPIX, บังคับให้เป็นตัวเลข
      -> build_sequences()        จัดลงกริดเวลาสม่ำเสมอ, เลื่อนหน้าต่าง, ติด label
      -> (X, y, meta)             X:(N, L, F) float32, y:(N,) uint8

**นิยามของ label**: sample ที่ "เวลาออกพยากรณ์" ``t`` มี label = 1 เมื่อมี flare
ที่แรงกว่าหรือเท่ากับเกณฑ์ เกิดจาก HARP ดวงนั้น ในช่วงเวลา ``(t, t+horizon]``
สังเกตว่าช่วงเป็นแบบ **เปิดที่ t** — flare ที่กำลังปะทุอยู่ ณ วินาทีที่ออกพยากรณ์
ไม่นับเป็นการทำนาย เพราะเราไม่ได้ทำนาย เราแค่เห็นมัน
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..config import DataConfig
from .goes_class import goes_class_to_flux

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# การทำความสะอาดข้อมูล
# --------------------------------------------------------------------------- #


def _coerce_quality(series: pd.Series) -> pd.Series:
    """QUALITY อาจกลับมาเป็น int หรือสตริงฐานสิบหก ('0x00000000') แล้วแต่เส้นทางที่ query"""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64")

    def parse(value) -> int | None:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        text = str(value).strip()
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(float(text))
        except ValueError:
            return None

    return series.map(parse).astype("Int64")


def clean_sharp_frame(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """กรองและแปลงชนิดข้อมูลของตาราง SHARP keywords ดิบ

    ตัวกรองแต่ละตัวถูก log ไว้ เพราะถ้าจู่ๆ ข้อมูลหายไป 90% เราต้องรู้ว่าหายที่ขั้นไหน
    """
    if df.empty:
        raise ValueError("ตาราง SHARP ว่างเปล่า — ตรวจสอบว่าขั้นตอนดาวน์โหลด metadata สำเร็จหรือไม่")

    out = df.copy()
    n_start = len(out)

    # --- เวลา ---
    if "t_rec" not in out.columns:
        import drms

        out["t_rec"] = drms.to_datetime(out["T_REC"])
    out["t_rec"] = pd.to_datetime(out["t_rec"], errors="coerce")
    out = out[out["t_rec"].notna()]

    out["HARPNUM"] = pd.to_numeric(out["HARPNUM"], errors="coerce").astype("Int64")
    out = out[out["HARPNUM"].notna()]

    # --- บังคับให้คอลัมน์ตัวเลขเป็นตัวเลขจริง (JSOC ส่งกลับมาเป็นสตริงได้) ---
    numeric_cols = [*cfg.sharp.features, "LAT_FWT", "LON_FWT", "NPIX", "CRLN_OBS", "CRLT_OBS"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # --- ตัวกรองคุณภาพ ---
    if cfg.sharp.require_quality_zero and "QUALITY" in out.columns:
        quality = _coerce_quality(out["QUALITY"])
        before = len(out)
        out = out[quality == 0]
        logger.info("กรอง QUALITY == 0 : %d -> %d แถว", before, len(out))

    # SHARP parameters เพี้ยนใกล้ขอบจานเพราะ projection effect (Bobra & Couvidat 2015)
    if "LON_FWT" in out.columns:
        before = len(out)
        out = out[out["LON_FWT"].abs() < cfg.sharp.abs_lon_max_deg]
        logger.info(
            "กรอง |LON_FWT| < %.0f องศา : %d -> %d แถว",
            cfg.sharp.abs_lon_max_deg,
            before,
            len(out),
        )

    if "NPIX" in out.columns:
        before = len(out)
        out = out[out["NPIX"] >= cfg.sharp.min_npix]
        logger.info("กรอง NPIX >= %d : %d -> %d แถว", cfg.sharp.min_npix, before, len(out))

    # --- ต้องมี feature ครบทุกตัว ---
    missing_cols = [c for c in cfg.sharp.features if c not in out.columns]
    if missing_cols:
        raise ValueError(
            f"ตาราง SHARP ขาดคอลัมน์ feature: {missing_cols}\n"
            "ตรวจสอบว่า series ที่ใช้ ({cfg.jsoc.sharp_series}) มี keyword เหล่านี้จริง"
        )

    # รายงานความสมบูรณ์ของแต่ละ feature ก่อนตัดแถว — ถ้า keyword ตัวใดไม่มีอยู่จริงใน
    # series ที่ query มา มันจะเป็น NaN ทั้งคอลัมน์แล้วลาก dropna ให้ล้างข้อมูลทิ้งทั้งชุด
    na_frac = out[cfg.sharp.features].isna().mean().sort_values(ascending=False)
    unusable = na_frac[na_frac > 0.99].index.tolist()
    if unusable:
        raise ValueError(
            "feature ต่อไปนี้ไม่มีข้อมูลเลย (NaN เกิน 99%): "
            f"{unusable}\n"
            f"แปลว่า series '{cfg.jsoc.sharp_series}' ไม่มี keyword เหล่านี้ "
            "หรือสะกดชื่อผิด — ลบออกจาก sharp.features ใน configs/data.yaml\n"
            "สัดส่วน NaN ของทุก feature:\n"
            + na_frac.map(lambda v: f"{v:.1%}").to_string()
        )
    if (na_frac > 0.05).any():
        logger.warning(
            "feature ที่มีค่าขาดหายเกิน 5%%:\n%s",
            na_frac[na_frac > 0.05].map(lambda v: f"{v:.1%}").to_string(),
        )

    before = len(out)
    out = out.dropna(subset=cfg.sharp.features)
    logger.info("ตัดแถวที่มี feature ขาดหาย : %d -> %d แถว", before, len(out))

    out = (
        out.sort_values(["HARPNUM", "t_rec"])
        .drop_duplicates(subset=["HARPNUM", "t_rec"], keep="first")
        .reset_index(drop=True)
    )
    out["HARPNUM"] = out["HARPNUM"].astype("int64")

    logger.info(
        "ทำความสะอาดเสร็จ: %d -> %d แถว (%d HARP, %s ถึง %s)",
        n_start,
        len(out),
        out["HARPNUM"].nunique(),
        out["t_rec"].min().date(),
        out["t_rec"].max().date(),
    )
    if out.empty:
        raise ValueError("ไม่เหลือข้อมูลหลังการกรอง — ตัวกรองใน data.yaml อาจเข้มเกินไป")
    return out


# --------------------------------------------------------------------------- #
# การสร้าง label
# --------------------------------------------------------------------------- #


def build_flare_lookup(
    flares_with_harp: pd.DataFrame, threshold_class: str
) -> dict[int, np.ndarray]:
    """HARPNUM -> array ของเวลา peak (เรียงแล้ว) ของ flare ที่ถึงเกณฑ์ความแรง

    เรียงไว้ล่วงหน้าเพื่อให้ใช้ ``searchsorted`` หา flare ในหน้าต่างเวลาได้ในเวลา O(log n)
    """
    threshold_flux = goes_class_to_flux(threshold_class)
    if flares_with_harp.empty:
        logger.warning("ไม่มี flare event ที่จับคู่กับ HARP ได้ — label จะเป็น 0 ทั้งหมด")
        return {}

    qualifying = flares_with_harp[flares_with_harp["peak_flux"] >= threshold_flux]
    logger.info(
        "flare ที่ถึงเกณฑ์ >= %s : %d จาก %d คู่ (HARP-flare)",
        threshold_class,
        len(qualifying),
        len(flares_with_harp),
    )

    lookup: dict[int, np.ndarray] = {}
    for harpnum, group in qualifying.groupby("HARPNUM"):
        times = np.sort(pd.to_datetime(group["peak_time"]).to_numpy())
        lookup[int(harpnum)] = times

    logger.info("HARP ที่มี flare ถึงเกณฑ์อย่างน้อย 1 ครั้ง: %d ดวง", len(lookup))
    return lookup


def label_times(
    times: np.ndarray, flare_times: np.ndarray | None, horizon_hours: int
) -> np.ndarray:
    """label ของแต่ละเวลาออกพยากรณ์: มี flare ในช่วง ``(t, t+horizon]`` หรือไม่"""
    if flare_times is None or len(flare_times) == 0:
        return np.zeros(len(times), dtype=np.uint8)

    horizon = np.timedelta64(horizon_hours, "h")
    times = times.astype("datetime64[ns]")
    flare_times = flare_times.astype("datetime64[ns]")

    # 'right' -> ตัด flare ที่เกิดพอดีเวลา t ออก (ช่วงเปิดที่ t)
    lo = np.searchsorted(flare_times, times, side="right")
    # 'right' -> รวม flare ที่เกิดพอดี t+horizon (ช่วงปิดที่ปลาย)
    hi = np.searchsorted(flare_times, times + horizon, side="right")
    return (hi > lo).astype(np.uint8)


def persistence_times(
    times: np.ndarray, flare_times: np.ndarray | None, lookback_hours: int
) -> np.ndarray:
    """baseline แบบ persistence: ทายว่าจะเกิด ถ้ามี flare ในช่วง ``(t - lookback, t]``

    เป็นภาพสะท้อนของ :func:`label_times` — ใช้ lookup ชุดเดียวกัน (HARP เดียวกัน เกณฑ์ความแรงเดียวกัน)
    และขอบของช่วงต่อกันพอดี: flare ที่เกิดพอดีเวลา t นับเป็นอดีต (รู้แล้ว ณ เวลาออกพยากรณ์) ไม่ใช่ label
    """
    if flare_times is None or len(flare_times) == 0:
        return np.zeros(len(times), dtype=np.uint8)

    lookback = np.timedelta64(lookback_hours, "h")
    times = times.astype("datetime64[ns]")
    flare_times = flare_times.astype("datetime64[ns]")

    lo = np.searchsorted(flare_times, times - lookback, side="right")  # ตัด flare ที่พอดี t - lookback
    hi = np.searchsorted(flare_times, times, side="right")             # รวม flare ที่พอดี t
    return (hi > lo).astype(np.uint8)


# --------------------------------------------------------------------------- #
# การสร้าง sequence
# --------------------------------------------------------------------------- #


def _regular_grid(group: pd.DataFrame, cadence_hours: int, features: list[str]):
    """จัดข้อมูลของ HARP หนึ่งดวงลงบนกริดเวลาสม่ำเสมอ

    ข้อมูลจริงจาก JSOC มีช่องว่าง (frame เสีย, ถูกกรองออก) การจัดลงกริดก่อนทำให้
    การเลื่อนหน้าต่างเป็นการ slice อาร์เรย์ตรงๆ และทำให้ "24 timesteps" หมายถึง
    "24 ชั่วโมงจริง" เสมอ ไม่ใช่ "24 แถวที่บังเอิญเหลืออยู่"

    Returns
    -------
    (grid_times, values, is_real) โดย values ถูกเติมช่องว่างแล้ว และ is_real บอกว่า
    แต่ละจุดบนกริดมาจากการสังเกตจริงหรือถูกเติม
    """
    freq = pd.Timedelta(hours=cadence_hours)
    t_first, t_last = group["t_rec"].iloc[0], group["t_rec"].iloc[-1]
    grid = pd.date_range(t_first.floor("h"), t_last.ceil("h"), freq=freq)
    if len(grid) == 0:
        return None, None, None

    indexed = group.set_index("t_rec")[features]
    # จับคู่การสังเกตจริงเข้ากับจุดบนกริดที่ใกล้ที่สุด ภายในครึ่งหนึ่งของ cadence
    reindexed = indexed.reindex(grid, method="nearest", tolerance=freq / 2)

    is_real = reindexed.notna().all(axis=1).to_numpy()
    values = reindexed.ffill().bfill().to_numpy(dtype=np.float32)
    return grid, values, is_real


def build_sequences(
    sharp: pd.DataFrame,
    flare_lookup: dict[int, np.ndarray],
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """สร้าง sequence dataset จากตาราง SHARP ที่ทำความสะอาดแล้ว

    Returns
    -------
    X : (N, L, F) float32 — L = ``cfg.sequence.length`` timesteps ย้อนหลัง
    y : (N,) uint8
    meta : DataFrame หนึ่งแถวต่อหนึ่ง sample (HARPNUM, เวลาออกพยากรณ์, ตำแหน่ง, ฯลฯ)
    """
    features = cfg.sharp.features
    seq_len = cfg.sequence.length
    cadence = cfg.sharp.cadence_hours
    horizon = cfg.flare.horizon_hours
    max_missing = cfg.sequence.max_missing_frac

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []

    n_harps_used = 0
    n_too_short = 0

    for harpnum, group in sharp.groupby("HARPNUM", sort=True):
        group = group.sort_values("t_rec")
        grid, values, is_real = _regular_grid(group, cadence, features)

        if grid is None or len(grid) < seq_len:
            n_too_short += 1
            continue

        # หน้าต่าง i ครอบคลุมจุดกริด [i, i+seq_len) และออกพยากรณ์ที่จุดสุดท้าย
        windows = sliding_window_view(values, seq_len, axis=0)  # (N, F, L)
        windows = np.ascontiguousarray(windows.transpose(0, 2, 1))  # (N, L, F)
        real_windows = sliding_window_view(is_real, seq_len)  # (N, L)

        issue_idx = np.arange(seq_len - 1, len(grid))
        issue_times = grid[issue_idx].to_numpy()

        # เก็บเฉพาะหน้าต่างที่ข้อมูลครบพอ และจุดออกพยากรณ์เป็นการสังเกตจริง
        missing_frac = 1.0 - real_windows.mean(axis=1)
        keep = (missing_frac <= max_missing) & real_windows[:, -1]
        if not keep.any():
            continue

        labels = label_times(issue_times[keep], flare_lookup.get(int(harpnum)), horizon)

        x_chunks.append(windows[keep])
        y_chunks.append(labels)

        kept_times = pd.DatetimeIndex(issue_times[keep])
        obs = group.set_index("t_rec").reindex(kept_times, method="nearest")
        meta_rows.append(
            pd.DataFrame(
                {
                    "HARPNUM": int(harpnum),
                    "issue_time": kept_times,
                    "label": labels,
                    "lat": obs["LAT_FWT"].to_numpy() if "LAT_FWT" in obs else np.nan,
                    "lon": obs["LON_FWT"].to_numpy() if "LON_FWT" in obs else np.nan,
                    "noaa_ar": obs["NOAA_AR"].to_numpy() if "NOAA_AR" in obs else np.nan,
                    "missing_frac": missing_frac[keep],
                }
            )
        )
        n_harps_used += 1

    if not x_chunks:
        raise ValueError(
            "ไม่สามารถสร้าง sequence ได้เลย — ลองลด sequence.length หรือผ่อนตัวกรองใน data.yaml"
        )

    x = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    meta = pd.concat(meta_rows, ignore_index=True)

    logger.info(
        "สร้าง sequence สำเร็จ: %s จาก %d HARP (ข้าม %d HARP ที่ข้อมูลสั้นเกินไป)",
        f"{len(x):,} sample รูปทรง {x.shape}",
        n_harps_used,
        n_too_short,
    )
    logger.info(
        "  positive %d ตัว (%.2f%%) — ข้อมูลไม่สมดุลอย่างรุนแรงตามธรรมชาติของปัญหา",
        int(y.sum()),
        100 * y.mean(),
    )
    return x, y, meta


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """บีบ dynamic range โดยรักษาเครื่องหมาย: ``sign(x) * log1p(|x|)``

    SHARP parameters ต่างกันหลาย order of magnitude — USFLUX อยู่ราว 1e22 Mx
    ขณะที่ MEANGAM อยู่ราว 40 องศา ถ้า standardize ตรงๆ จะเกิดสองปัญหา:

    1. ``std`` ของ USFLUX ยกกำลังสองแล้ว **overflow ใน float32** (1e22^2 = 1e44
       ขณะที่เพดานของ float32 คือ ~3.4e38) ทำให้ได้ ``inf`` แล้วค่าที่ normalise
       ออกมากลายเป็น 0 ทั้งคอลัมน์
    2. การกระจายตัวแบบ heavy-tailed ทำให้ค่าสุดโต่งไม่กี่ตัวครอบงำ gradient

    ใช้ ``signed_log1p`` แทน ``log1p`` ธรรมดา เพราะบาง feature (เช่น MEANJZD,
    MEANALP) มีค่าติดลบได้ตามฟิสิกส์
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def compute_normalisation(x: np.ndarray, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """คำนวณ mean/std ต่อ feature หลังแปลง log — **ต้องเรียกด้วยข้อมูล train เท่านั้น**

    การคำนวณทำใน float64 เสมอเพื่อเลี่ยง overflow แล้วค่อยลดเหลือ float32 ตอนเก็บ
    """
    subset = x if mask is None else x[mask]
    if len(subset) == 0:
        raise ValueError("ไม่มีข้อมูลสำหรับคำนวณ normalisation statistics")

    flat = signed_log1p(subset).reshape(-1, subset.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # กัน division by zero สำหรับ feature ที่เป็นค่าคงที่
    std = np.where(std < 1e-8, 1.0, std)

    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(
            "normalisation statistics มีค่า inf/nan — ตรวจสอบว่ามีค่าสุดโต่งผิดปกติในข้อมูล"
        )

    logger.info("คำนวณ normalisation จาก %d sample (%d แถว)", len(subset), len(flat))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_normalisation(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """ต้องใช้คู่กับ :func:`compute_normalisation` เสมอ — ลำดับการแปลงต้องตรงกัน"""
    transformed = signed_log1p(x)
    return ((transformed - stats["mean"]) / stats["std"]).astype(np.float32)
