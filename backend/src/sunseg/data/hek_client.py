"""ดึงรายการ solar flare จาก HEK (Heliophysics Events Knowledgebase).

เราต้องการ 3 อย่างจากแต่ละ event: **เวลาที่เกิด**, **ความแรง (GOES class)** และ
**NOAA AR ต้นทาง** — ตัวสุดท้ายคือสิ่งที่ทำให้เรา label ได้ว่า HARP ดวงไหนจะปะทุ

หมายเหตุการออกแบบ: เราดึง flare *ทุกคลาส* แล้วค่อยกรองเองในเครื่องด้วย
:func:`~sunseg.data.goes_class.goes_class_to_flux` แทนที่จะให้ HEK กรองให้ เพราะ
HEK เปรียบเทียบ ``fl_goescls`` แบบสตริง ซึ่งพลาดที่ขอบเขต (เช่น ``> "M1.0"`` จะ
ตัด M1.0 ทิ้ง) การกรองเองยังทำให้เราเก็บสถิติของ flare คลาส C ไว้ดูได้ด้วย
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .goes_class import safe_goes_class_to_flux
from .jsoc_client import iter_time_chunks

logger = logging.getLogger(__name__)

# คอลัมน์ที่เราสนใจจาก HEK (ตัวมันคืนมาเป็นร้อยคอลัมน์)
_HEK_COLUMNS = [
    "event_starttime",
    "event_peaktime",
    "event_endtime",
    "fl_goescls",
    "ar_noaanum",
    "frm_name",
    "obs_observatory",
]

# ลำดับความน่าเชื่อถือของแหล่ง detection — SWPC คือรายการอย่างเป็นทางการของ NOAA
_SOURCE_PRIORITY = ["SWPC", "SSW Latest Events"]

#: คลาส ``astropy.time.Time`` โหลดแบบ lazy ด้วยเหตุผลเดียวกับ sunpy — โมดูลนี้ถูก
#: import ตั้งแต่ตอนสตาร์ท แต่ astropy ใช้เวลา import นาน จึงเลี่ยงจนกว่าจะใช้จริง
_TIME_CLS: type | None = None


def _astropy_time_cls() -> type:
    global _TIME_CLS
    if _TIME_CLS is None:
        from astropy.time import Time

        _TIME_CLS = Time
    return _TIME_CLS


def fetch_flare_events(
    start: date,
    end: date,
    chunk_days: int = 60,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """ดึง flare event ทั้งหมดในช่วงเวลา แล้วคืนเป็น DataFrame ที่ทำความสะอาดแล้ว

    Returns
    -------
    DataFrame คอลัมน์: ``start_time``, ``peak_time``, ``end_time``,
    ``goes_class``, ``peak_flux``, ``noaa_ar``, ``frm_name``
    """
    from sunpy.net import attrs as a
    from sunpy.net.hek import HEKClient

    from ..logging_utils import quiet_science_libraries

    # ต้องเรียกหลัง import sunpy เท่านั้น (ดูคำอธิบายในฟังก์ชัน)
    quiet_science_libraries()

    client = HEKClient()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    chunks = list(iter_time_chunks(start, end, chunk_days))

    for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        tag = chunk_start.strftime("%Y%m%d")
        cache_file = cache_dir / f"hek_flares_{tag}.parquet" if cache_dir else None

        if cache_file is not None and cache_file.exists():
            logger.info("[%d/%d] ใช้ cache: %s", idx, len(chunks), cache_file.name)
            frames.append(pd.read_parquet(cache_file))
            continue

        logger.info(
            "[%d/%d] กำลัง query HEK: %s ถึง %s",
            idx,
            len(chunks),
            chunk_start.date(),
            chunk_end.date(),
        )
        try:
            table = client.search(
                a.Time(chunk_start, chunk_end),
                a.hek.EventType("FL"),
            )
        except Exception as exc:  # noqa: BLE001 — HEK ล่ม/timeout ได้บ่อย
            logger.error("HEK query ล้มเหลวสำหรับ %s: %s", chunk_start.date(), exc)
            continue

        df = _hek_table_to_frame(table)
        if df.empty:
            logger.warning("ไม่พบ flare ในช่วง %s ถึง %s", chunk_start.date(), chunk_end.date())
            continue

        if cache_file is not None:
            df.to_parquet(cache_file, index=False)
        frames.append(df)

    if not frames:
        return _empty_flare_frame()

    combined = pd.concat(frames, ignore_index=True)
    return deduplicate_flares(combined)


def _empty_flare_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": pd.Series(dtype="datetime64[ns]"),
            "peak_time": pd.Series(dtype="datetime64[ns]"),
            "end_time": pd.Series(dtype="datetime64[ns]"),
            "goes_class": pd.Series(dtype="object"),
            "peak_flux": pd.Series(dtype="float64"),
            "noaa_ar": pd.Series(dtype="Int64"),
            "frm_name": pd.Series(dtype="object"),
        }
    )


def _hek_table_to_frame(table) -> pd.DataFrame:
    """แปลง HEKTable (astropy Table) เป็น DataFrame ที่สะอาด"""
    if table is None or len(table) == 0:
        return _empty_flare_frame()

    available = [c for c in _HEK_COLUMNS if c in table.colnames]
    missing = set(_HEK_COLUMNS) - set(available)
    if missing:
        logger.debug("HEK ไม่ได้คืนคอลัมน์เหล่านี้: %s", sorted(missing))

    # แปลงทีละคอลัมน์เป็น list of str เพื่อเลี่ยงปัญหา masked/object column ของ astropy
    data = {col: [_scalar(v) for v in table[col]] for col in available}
    df = pd.DataFrame(data)

    for src, dst in (
        ("event_starttime", "start_time"),
        ("event_peaktime", "peak_time"),
        ("event_endtime", "end_time"),
    ):
        df[dst] = pd.to_datetime(df[src], errors="coerce") if src in df else pd.NaT

    df["goes_class"] = df["fl_goescls"].astype("string").str.strip() if "fl_goescls" in df else pd.NA
    df["frm_name"] = df["frm_name"].astype("string") if "frm_name" in df else pd.NA
    df["noaa_ar"] = (
        pd.to_numeric(df["ar_noaanum"], errors="coerce").astype("Int64")
        if "ar_noaanum" in df
        else pd.NA
    )
    df["peak_flux"] = df["goes_class"].map(safe_goes_class_to_flux)

    out = df[
        ["start_time", "peak_time", "end_time", "goes_class", "peak_flux", "noaa_ar", "frm_name"]
    ]

    # ทิ้ง event ที่ไม่มีคลาส GOES ที่ parse ได้ — ใช้ label ไม่ได้อยู่ดี
    before = len(out)
    out = out[out["peak_flux"].notna()].copy()
    if before != len(out):
        logger.debug("ตัด flare ที่ไม่มี GOES class ที่อ่านได้ออก %d รายการ", before - len(out))

    # ถ้าไม่มี peak_time ให้ใช้ start_time แทน (บาง record ขาด peak)
    out["peak_time"] = out["peak_time"].fillna(out["start_time"])
    return out[out["peak_time"].notna()].reset_index(drop=True)


def _scalar(value):
    """ดึงค่า python ธรรมดาออกจาก cell ของ astropy Table

    cell ของ astropy อาจเป็นได้หลายอย่างที่ pandas จัดการไม่ได้โดยตรง:
    ``np.ma.masked`` (ค่าที่หายไป), ``np.str_``/``np.int64`` (numpy scalar),
    หรือ bytes ฟังก์ชันนี้แปลงทั้งหมดให้เป็นชนิดของ Python ธรรมดา

    จุดสำคัญคือ ``np.ma.masked`` ต้องกลายเป็น ``None`` — ถ้าปล่อยผ่านไป
    ``pd.to_numeric`` จะพังด้วย ``TypeError: len() of unsized object``
    """
    if value is None or value is np.ma.masked:
        return None
    if isinstance(value, np.ma.core.MaskedConstant):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        # numpy scalar -> ชนิด Python ที่เทียบเท่า
        return value.item()
    if isinstance(value, _astropy_time_cls()):
        # sunpy ตั้งแต่รุ่นใหม่ (ทดสอบกับ 8.0.0) คืนคอลัมน์เวลาของ HEK เป็น
        # astropy Time ไม่ใช่สตริงแบบเดิมอีกต่อไป
        #
        # นี่เป็นความล้มเหลวที่เงียบเป็นพิเศษ: pd.to_datetime() แปลง Time ไม่ได้
        # และเพราะ _hek_table_to_frame เรียกด้วย errors="coerce" ทั้งคอลัมน์จึง
        # กลายเป็น NaT โดยไม่มี exception แล้วตัวกรอง peak_time.notna() ตอนท้าย
        # ก็ทิ้งทุกแถว ผลลัพธ์คือรายงานว่า "ไม่พบ flare" ทั้งที่ HEK ส่งข้อมูลมาครบ
        return value.iso
    return value


def deduplicate_flares(df: pd.DataFrame) -> pd.DataFrame:
    """ตัด event ซ้ำที่มาจากหลายอัลกอริทึม detection

    HEK รวมรายการจากหลายแหล่ง (SWPC, SSW Latest Events, ...) ซึ่งรายงาน flare
    ดวงเดียวกันซ้ำกัน เราถือว่าเป็น event เดียวกันเมื่อ **peak time ตกในนาทีเดียวกัน
    และคลาส GOES ตรงกัน** แล้วเก็บรายการจากแหล่งที่น่าเชื่อถือที่สุดไว้
    """
    if df.empty:
        return df

    df = df.copy()
    priority = {name: rank for rank, name in enumerate(_SOURCE_PRIORITY)}
    df["_priority"] = df["frm_name"].map(lambda n: priority.get(n, len(_SOURCE_PRIORITY)))
    # record ที่ระบุ NOAA AR ได้มีค่ากับเรามากกว่า จึงให้ชนะ record ที่ไม่ระบุ
    df["_has_ar"] = df["noaa_ar"].notna().astype(int)
    df["_peak_minute"] = df["peak_time"].dt.floor("min")

    before = len(df)
    df = (
        df.sort_values(["_peak_minute", "_has_ar", "_priority"], ascending=[True, False, True])
        .drop_duplicates(subset=["_peak_minute", "goes_class"], keep="first")
        .drop(columns=["_priority", "_has_ar", "_peak_minute"])
        .sort_values("peak_time")
        .reset_index(drop=True)
    )
    logger.info("ตัด flare ซ้ำ: %d -> %d รายการ", before, len(df))
    return df


# หมายเหตุ: ``summarise_flares`` และ ``map_flares_to_harps`` ใช้ร่วมกันได้กับทุกแหล่ง
# จึงอยู่ใน :mod:`sunseg.data.flare_catalog` ไม่ใช่ที่นี่
