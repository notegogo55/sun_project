"""รวม SHARP + intensity + X-ray ให้อยู่บนกริดเวลาเดียวกันเป๊ะ — dataset ก้อนเดียวของ
งานเปรียบเทียบ feature 5 แบบ (ticket 06 ของ ``.scratch/lstm-feature-ablation/``)

**ต่างจาก ``build_sequences()`` (production) ตรงไหน**: กริดของแต่ละ HARP ที่นั่นยึด
``t_first`` ของ HARP ตัวเอง (ดู ``_regular_grid``) แต่ละดวงจึง "คนละ phase" กัน — เหมาะกับ
SHARP อย่างเดียวเพราะไม่ต้องจับคู่กับแหล่งข้อมูลอื่น แต่พอต้องรวมกับ intensity/X-ray ที่มา
จากกริดกลาง (เวลาดาวน์โหลดเฟรมเต็มดวงจริง) การจับคู่ต้องเป็น **exact timestamp match**
ไม่ใช่ nearest-merge — โมดูลนี้จึงยึดกริดของทุก HARP ไว้ที่จุดอ้างอิงเดียวกัน
(``grid_start``, มักคือ ``cfg.time_range.start``) แทน

**หลักการที่ทุกอย่างตั้งอยู่บน**: dataset เก็บ**ทุกคอลัมน์ผู้สมัคร**ไว้ในอาร์เรย์เดียว
(``X``) เรียงตามชื่อใน ``feature_names`` — การเลือก "แบบ" ของการทดลอง (V0..V5) คือการ
หยิบบางคอลัมน์ตามชื่อ**หลังจากสร้างเสร็จแล้ว** (:func:`select_columns`) ทุกแบบจึงได้
จำนวนแถวและลำดับ ``(HARPNUM, issue_time)`` เหมือนกันเป๊ะโดยโครงสร้าง ไม่ใช่โดยวินัยของ
คนเขียนสคริปต์ (ดู spec หัวข้อ "แบบของการทดลอง = รายชื่อ column ไม่ใช่ dataset คนละชุด")
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from ..config import DataConfig
from .build_sequences import label_times
from .intensity import STATS as INTENSITY_STATS

logger = logging.getLogger(__name__)

#: กริดเวลาของ dataset งานเปรียบเทียบ — ทับ ``sharp.cadence_hours``/``sequence.length`` ของ
#: ``configs/data.yaml`` (ซึ่งเป็นของ pipeline production: 1 ชม. × 24) โดยตั้งใจ
#: 12 ชม. × 8 = ประวัติ 4 วัน (ดูเหตุผลใน spec) — ใช้ร่วมกันโดย ``scripts/study/build_dataset.py``,
#: ``xray_sequence_gate_check.py`` และ ``measure_acquisition_time.py``
STUDY_CADENCE_HOURS = 12
STUDY_SEQUENCE_LENGTH = 8

#: ความยาวคลื่น AIA ที่ intensity.py สกัดค่าให้ — ต้องตรงกับคอลัมน์ที่ intensity table มีจริง
#: (ต้องเป็น superset ของ ``cfg.aia.channels`` ทุกช่องที่เคยรันผ่าน extract_intensity.py —
#: ช่องไหนไม่อยู่ในนี้จะหายไปจาก feature pool เงียบๆ ถึงจะมีอยู่ใน intensity table จริงก็ตาม)
INTENSITY_CHANNELS: tuple[str, ...] = ("4500", "1600", "304", "171")

#: ชื่อคอลัมน์ intensity ทั้ง 20 ตัว (5 สถิติ x 4 ช่อง) เรียงตามลำดับคงที่ — ตรงกับที่
#: ``extract_frame_intensities()`` สร้างไว้ (ดู ``sunseg.data.intensity``)
INTENSITY_COLUMNS: tuple[str, ...] = tuple(
    f"{channel}_{stat}" for channel in INTENSITY_CHANNELS for stat in INTENSITY_STATS
)

#: ชื่อคอลัมน์ X-ray ที่ ``XrayFluxStore.bin_series()`` คืนมา (ไม่รวม ``t_rec``) — คอลัมน์ใหม่
#: ต่อท้ายเท่านั้น เพื่อให้ index ของ feature เดิมใน X.npy ไม่เลื่อน
XRAY_COLUMNS: tuple[str, ...] = (
    "xray_median", "xray_max", "xray_min", "xray_log10_median", "xray_log10_rel27d",
)

#: ประวัติ flare **ของ HARP นั้นเอง** ต่อ timestep (งาน feature-evidence-cv) — ต่อท้ายสุดเช่นกัน
#: ต่างจาก X-ray ทั้งดวงข้างบนที่เท่ากันทุก HARP ณ เวลาเดียวกัน จึงแยกไม่ได้ว่าดวงไหนจะปะทุ
FLARE_HISTORY_COLUMNS: tuple[str, ...] = ("ar_flare_log10max_12h", "ar_flare_count_c_12h")
#: log10 flux ที่ใช้แทน "ไม่มี flare" — A1.0 ต่ำกว่า flare ที่เล็กที่สุดในแคตตาล็อก (6.8e-8)
FLARE_HISTORY_FLOOR_LOG10 = -8.0
FLARE_HISTORY_C_FLUX = 1e-6


def intensity_columns_for(channels: tuple[str, ...]) -> tuple[str, ...]:
    """ชื่อคอลัมน์ intensity ของช่องที่ระบุ เรียงแบบเดียวกับ :data:`INTENSITY_COLUMNS`"""
    return tuple(f"{channel}_{stat}" for channel in channels for stat in INTENSITY_STATS)


def flare_history_on_grid(
    peak_times: np.ndarray, peak_fluxes: np.ndarray, grid: pd.DatetimeIndex, cadence_hours: int
) -> np.ndarray:
    """ประวัติ flare ของ HARP หนึ่งดวงบนกริด — คืน ``(len(grid), 2)`` ตาม :data:`FLARE_HISTORY_COLUMNS`

    จุดกริด ``t`` สรุป flare ที่ peak ใน ``(t − cadence, t]`` — ช่วงนี้ **ไม่ทับ** กับหน้าต่างของ label
    ``(t, t + horizon]`` ของ :func:`label_times` โดยโครงสร้าง (flare ที่ peak พอดี ``t`` อยู่ฝั่งนี้
    ส่วน label ตัดทิ้ง) · เพราะหน้าต่างยาวเท่า cadence พอดี แต่ละ flare จึงตกลงช่องเดียวเสมอ
    """
    out = np.empty((len(grid), len(FLARE_HISTORY_COLUMNS)), dtype=np.float32)
    out[:, 0] = FLARE_HISTORY_FLOOR_LOG10
    out[:, 1] = 0.0
    if len(peak_times) == 0 or len(grid) == 0:
        return out

    times = np.asarray(peak_times, dtype="datetime64[ns]")
    fluxes = np.asarray(peak_fluxes, dtype=np.float64)
    grid_ns = grid.to_numpy(dtype="datetime64[ns]")
    # จุดกริดแรกที่ >= peak คือปลายขวาของช่อง (t − cadence, t] ที่ flare นั้นตกอยู่
    slot = np.searchsorted(grid_ns, times, side="left")
    inside = (slot < len(grid_ns)) & (times > grid_ns[0] - np.timedelta64(cadence_hours, "h"))
    slot, fluxes = slot[inside], fluxes[inside]

    log_flux = np.log10(np.clip(fluxes, 10**FLARE_HISTORY_FLOOR_LOG10, None)).astype(np.float32)
    np.maximum.at(out[:, 0], slot, log_flux)
    np.add.at(out[:, 1], slot, (fluxes >= FLARE_HISTORY_C_FLUX).astype(np.float32))
    return out


# --------------------------------------------------------------------------- #
# กริดเวลากลาง
# --------------------------------------------------------------------------- #


def anchored_grid(start: datetime, end: datetime, cadence_hours: int) -> pd.DatetimeIndex:
    """กริดเวลาที่ทุกแหล่งข้อมูล (SHARP/intensity/X-ray) ของหน้าต่างเดียวกันต้องใช้ร่วมกัน

    ยึด ``start`` เป๊ะ — เวลาดาวน์โหลดเฟรมจริงจาก ``query_fulldisk_times()`` ก็ยึดจุด
    เดียวกันนี้ (DRMS recordset ``@Nh`` นับ cadence จากจุดเริ่มของ query) กริดนี้จึงตรงกับ
    timestamp ของเฟรมที่ intensity ถูกสกัดมาเป๊ะ ไม่ต้องเผื่อ tolerance ตอน merge
    """
    return pd.date_range(start, end, freq=pd.Timedelta(hours=cadence_hours))


def _reindex_on_grid(
    indexed: pd.DataFrame,
    grid: pd.DatetimeIndex,
    columns: list[str],
    tolerance: pd.Timedelta | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """เหมือนขั้นตอน reindex ใน ``build_sequences._regular_grid`` แต่รับกริดจากภายนอก
    แทนที่จะคำนวณเองจาก min/max ของข้อมูล — ใช้ร่วมกันได้ทั้ง SHARP/intensity/X-ray
    เพราะทำแค่ "จับคู่แถวที่มีอยู่เข้ากับจุดบนกริดที่ใกล้ที่สุด" ไม่สนใจความหมายของคอลัมน์

    ``tolerance`` จำเป็นเพราะเวลาจริงของ observation (T_REC ของ SHARP, T_REC ของเฟรม
    เต็มดวงที่ intensity อิงชื่อไฟล์ตาม) **ไม่ได้ตรงเป๊ะกับกริด nominal ที่ขอ** — DRMS
    คืน record ที่ใกล้ ``@Nh`` ที่สุด ไม่ใช่ record ที่เวลาตรงเป๊ะ (ภาพจริงมี cadence 720s
    ภายใน จึงมี jitter ได้หลักนาที) ``None`` = exact match เท่านั้น (ใช้กับ X-ray ที่กริด
    มาจาก ``pd.date_range`` เดียวกันกับ ``grid`` อยู่แล้วโดยสร้าง)

    Returns
    -------
    ``(values, is_real)`` — ``values`` เติมช่องว่างด้วย ffill/bfill แล้ว (เผื่อโมเดลเห็น
    NaN ไม่ได้) ``is_real`` บอกว่าจุดกริดนั้นมีข้อมูลจริงหรือถูกเติม ใช้กรองว่าจะเก็บ
    sample นั้นไว้หรือไม่ทีหลัง — ไม่ใช่ปล่อยให้ค่าที่เติมมาปนไปกับของจริงเงียบ ๆ
    """
    if indexed.empty:
        empty = np.full((len(grid), len(columns)), np.nan, dtype=np.float32)
        return empty, np.zeros(len(grid), dtype=bool)

    kwargs = {} if tolerance is None else {"method": "nearest", "tolerance": tolerance}
    reindexed = indexed.reindex(grid, **kwargs)
    is_real = reindexed[columns].notna().all(axis=1).to_numpy()
    values = reindexed[columns].ffill().bfill().to_numpy(dtype=np.float32)
    return values, is_real


# --------------------------------------------------------------------------- #
# การสร้าง sequence ของหนึ่งหน้าต่างเวลาที่ต่อเนื่อง (main range หรือ case study)
# --------------------------------------------------------------------------- #


def build_unified_window(
    sharp: pd.DataFrame,
    flare_lookup: dict[int, np.ndarray],
    intensity: pd.DataFrame,
    xray_bins: pd.DataFrame,
    grid_start: datetime,
    grid_end: datetime,
    cfg: DataConfig,
    extra_intensity_channels: tuple[str, ...] = (),
    flare_history: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """สร้าง unified sequence dataset ของหน้าต่างเวลาต่อเนื่องหนึ่งช่วง

    Parameters
    ----------
    extra_intensity_channels
        ช่อง AIA ที่เพิ่มหลัง :data:`INTENSITY_CHANNELS` (เช่น ``("94", "131")``) — คอลัมน์ต่อท้ายหลัง X-ray
        และ **ต้องมีครบทุก timestep ด้วย** เหมือนช่องเดิม (แถวร่วมของทุกแบบจึงเล็กลงตามช่องที่ขาด)
    flare_history
        HARPNUM -> ``(peak_times, peak_fluxes)`` ของ flare ทุกคลาสที่จับคู่กับ HARP นั้น — ส่งมาเมื่อ
        ต้องการคอลัมน์ :data:`FLARE_HISTORY_COLUMNS` (ต่อท้ายสุด) · ``None`` = ไม่มีคอลัมน์เหล่านี้
    sharp
        ผ่าน :func:`sunseg.data.build_sequences.clean_sharp_frame` มาแล้ว
    intensity
        ตารางจาก :func:`sunseg.data.intensity.extract_frame_intensities` ต่อกัน
        (คอลัมน์ ``HARPNUM``, ``issue_time`` เป็นสตริง ``YYYYMMDD_HHMMSS``, บวก
        คอลัมน์ตาม :data:`INTENSITY_COLUMNS`) — ให้เป็น ``DataFrame`` ว่างได้ถ้ายังไม่มี
        (เช่น ระหว่างรอเฟรม 2011-2017) คอลัมน์ intensity ทั้งหมดจะเป็น NaN แทน
    xray_bins
        ผลจาก ``XrayFluxStore.bin_series(grid_start, grid_end, cadence_hours)`` —
        **ต้องเรียกด้วย ``grid_start``/``cadence_hours`` เดียวกับที่ส่งเข้าฟังก์ชันนี้**
        ไม่งั้นกริดจะไม่ตรงกัน

    Returns
    -------
    ``(X, y, meta, feature_names)`` — ``X`` รูปทรง ``(N, L, F)`` ``F`` = จำนวน SHARP
    feature + 15 + 4, ``feature_names`` เรียงลำดับตรงกับแกนสุดท้ายของ ``X``
    """
    features = list(cfg.sharp.features)
    extra_intensity = list(intensity_columns_for(tuple(extra_intensity_channels)))
    all_intensity = [*INTENSITY_COLUMNS, *extra_intensity]
    history_names = list(FLARE_HISTORY_COLUMNS) if flare_history is not None else []
    feature_names = [*features, *INTENSITY_COLUMNS, *XRAY_COLUMNS, *extra_intensity, *history_names]
    n_base_intensity = len(INTENSITY_COLUMNS)

    seq_len = cfg.sequence.length
    cadence_hours = cfg.sharp.cadence_hours
    horizon = cfg.flare.horizon_hours
    max_missing = cfg.sequence.max_missing_frac
    # observation จริง (SHARP T_REC, timestamp ของเฟรมที่ intensity อิงชื่อไฟล์ตาม) มี
    # jitter จากกริด nominal ได้หลักนาที (ดู docstring ของ _reindex_on_grid) — ครึ่งหนึ่ง
    # ของ cadence กันไม่ให้จุดสองจุดบนกริดแย่งกันจับ observation เดียวกัน
    tolerance = pd.Timedelta(hours=cadence_hours) / 2

    grid = anchored_grid(grid_start, grid_end, cadence_hours)
    if len(grid) < seq_len:
        raise ValueError(
            f"กริด {grid_start} ถึง {grid_end} ที่ cadence {cadence_hours} ชม. มีแค่ "
            f"{len(grid)} จุด สั้นกว่า sequence.length={seq_len}"
        )

    xray_indexed = xray_bins.set_index("t_rec")
    if not xray_indexed.index.equals(grid):
        raise ValueError(
            "กริดของ xray_bins ไม่ตรงกับ grid ที่คำนวณในนี้ — ต้องเรียก "
            "XrayFluxStore.bin_series() ด้วย grid_start/cadence_hours เดียวกัน"
        )
    xray_values, xray_is_real = _reindex_on_grid(xray_indexed, grid, list(XRAY_COLUMNS))

    intensity_by_harp: dict[int, pd.DataFrame] = {}
    if not intensity.empty:
        intensity = intensity.copy()
        intensity["_t"] = pd.to_datetime(intensity["issue_time"], format="%Y%m%d_%H%M%S")
        for harpnum, group in intensity.groupby("HARPNUM"):
            intensity_by_harp[int(harpnum)] = group.set_index("_t").sort_index()

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    meta_rows: list[pd.DataFrame] = []
    n_harps_used = 0

    for harpnum, group in sharp.groupby("HARPNUM", sort=True):
        harpnum = int(harpnum)
        group = group.sort_values("t_rec").set_index("t_rec")
        sharp_values, sharp_is_real = _reindex_on_grid(group, grid, features, tolerance)

        intensity_group = intensity_by_harp.get(harpnum, pd.DataFrame(columns=all_intensity))
        if not set(all_intensity) <= set(intensity_group.columns):
            # ตาราง intensity ที่ไม่มีช่องใหม่ (เช่นของรอบเก่า) = ช่องนั้นขาดทุกจุด ไม่ใช่ KeyError
            intensity_group = intensity_group.reindex(columns=all_intensity)
        intensity_values, intensity_is_real = _reindex_on_grid(
            intensity_group, grid, all_intensity, tolerance
        )

        blocks = [
            sharp_values,
            intensity_values[:, :n_base_intensity],
            xray_values,
            intensity_values[:, n_base_intensity:],
        ]
        if flare_history is not None:
            peak_times, peak_fluxes = flare_history.get(harpnum, (np.empty(0), np.empty(0)))
            blocks.append(flare_history_on_grid(peak_times, peak_fluxes, grid, cadence_hours))
        values = np.concatenate(blocks, axis=1)

        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        windows = np.ascontiguousarray(windows.transpose(0, 2, 1))  # (N, L, F)

        sharp_real_w = np.lib.stride_tricks.sliding_window_view(sharp_is_real, seq_len)
        intensity_real_w = np.lib.stride_tricks.sliding_window_view(intensity_is_real, seq_len)
        xray_real_w = np.lib.stride_tricks.sliding_window_view(xray_is_real, seq_len)

        issue_idx = np.arange(seq_len - 1, len(grid))
        issue_times = grid[issue_idx].to_numpy()

        sharp_missing_frac = 1.0 - sharp_real_w.mean(axis=1)
        # เกณฑ์ร่วมที่เข้มงวดที่สุด: SHARP ต้องครบตามเกณฑ์เดิม (missing_frac + จุดออก
        # พยากรณ์เป็นข้อมูลจริง) **และ** intensity/X-ray ต้องครบ**ทุก** timestep — ทุกแบบ
        # (รวม V0 ที่ไม่ใช้ intensity/X-ray เลย) ใช้แถวชุดนี้ร่วมกัน ตามที่ spec กำหนดไว้ที่
        # หัวข้อ "แถวข้อมูล: เข้มงวดที่สุดเป็นเกณฑ์ร่วม" — เพื่อให้ TSS ของทุกแบบเทียบกันได้ตรงๆ
        keep = (
            (sharp_missing_frac <= max_missing)
            & sharp_real_w[:, -1]
            & intensity_real_w.all(axis=1)
            & xray_real_w.all(axis=1)
        )
        if not keep.any():
            continue

        labels = label_times(issue_times[keep], flare_lookup.get(harpnum), horizon)

        x_chunks.append(windows[keep])
        y_chunks.append(labels)

        kept_times = pd.DatetimeIndex(issue_times[keep])
        obs = group.reindex(kept_times, method="nearest")
        meta_rows.append(
            pd.DataFrame(
                {
                    "HARPNUM": harpnum,
                    "issue_time": kept_times,
                    "label": labels,
                    "lat": obs["LAT_FWT"].to_numpy() if "LAT_FWT" in obs else np.nan,
                    "lon": obs["LON_FWT"].to_numpy() if "LON_FWT" in obs else np.nan,
                    "noaa_ar": obs["NOAA_AR"].to_numpy() if "NOAA_AR" in obs else np.nan,
                    "sharp_missing_frac": sharp_missing_frac[keep],
                }
            )
        )
        n_harps_used += 1

    if not x_chunks:
        return (
            np.empty((0, seq_len, len(feature_names)), dtype=np.float32),
            np.empty((0,), dtype=np.uint8),
            pd.DataFrame(
                columns=["HARPNUM", "issue_time", "label", "lat", "lon", "noaa_ar", "sharp_missing_frac"]
            ),
            feature_names,
        )

    x = np.concatenate(x_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    meta = pd.concat(meta_rows, ignore_index=True)

    logger.info(
        "หน้าต่าง %s..%s: %d sample จาก %d HARP (feature %d ตัว)",
        grid_start, grid_end, len(x), n_harps_used, len(feature_names),
    )
    return x, y, meta, feature_names


# --------------------------------------------------------------------------- #
# การเลือก "แบบ" ของการทดลอง — หยิบคอลัมน์ตามชื่อ ไม่ใช่ตามตำแหน่ง
# --------------------------------------------------------------------------- #


def select_columns(feature_names: list[str], wanted: list[str]) -> np.ndarray:
    """คืน index (int) ของคอลัมน์ ``wanted`` ใน ``feature_names`` ตามลำดับที่ขอ

    ขอชื่อที่ไม่มีอยู่ต้องรู้ทันทีว่าชื่อไหนผิด — ไม่ใช่ได้ index ผิดเงียบ ๆ (ทดสอบใน
    ``test_study_dataset.py``, เป็นเกณฑ์ที่ ticket 06 ระบุไว้ตรง ๆ)
    """
    lookup = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise KeyError(
            f"ไม่พบคอลัมน์ต่อไปนี้ใน dataset: {missing}\n"
            f"คอลัมน์ที่มีอยู่จริง: {feature_names}"
        )
    return np.array([lookup[name] for name in wanted], dtype=np.intp)


# --------------------------------------------------------------------------- #
# นิยามแบบของการทดลอง (V0..V5) จากไฟล์ config — ดู configs/study/variants.yaml
# --------------------------------------------------------------------------- #


def _flatten(items: list) -> list[str]:
    """คลี่ list ที่มี list ซ้อนอยู่หนึ่งชั้นให้แบน — จำเป็นเพราะ YAML anchor ที่ใช้แทน
    "รายชื่อ column กลุ่ม X" ในไฟล์ variants (เช่น ``columns: [*sharp, *intensity]``)
    parse ออกมาเป็น list-of-lists เสมอ ไม่มี syntax ของ YAML ที่ flatten sequence ให้เอง
    (``<<`` merge key ใช้ได้แค่กับ mapping ไม่ใช่ sequence)
    """
    out: list[str] = []
    for item in items:
        out.extend(item) if isinstance(item, list) else out.append(item)
    return out


def load_variants(path) -> dict[str, dict]:
    """อ่านนิยาม "แบบ" ของการทดลองจากไฟล์ YAML (ปริยาย ``configs/study/variants.yaml``)

    คืน ``{variant_name: {"label": str, "role": str, "columns": list[str]}}`` —
    ``columns`` คลี่ (flatten) แล้ว พร้อมส่งเข้า :func:`select_columns` ตรงๆ
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    variants: dict[str, dict] = {}
    for name, spec in raw["variants"].items():
        variants[name] = {
            "label": spec.get("label", name),
            "role": spec.get("role", ""),
            "columns": _flatten(spec["columns"]),
        }
    return variants


def load_variant_columns(path, variant_name: str) -> list[str]:
    """ทางลัด: อ่านไฟล์ variants แล้วคืนแค่รายชื่อคอลัมน์ของแบบเดียว"""
    variants = load_variants(path)
    if variant_name not in variants:
        raise KeyError(
            f"ไม่มีแบบชื่อ {variant_name!r} ใน {path} — แบบที่มีอยู่: {sorted(variants)}"
        )
    return variants[variant_name]["columns"]
