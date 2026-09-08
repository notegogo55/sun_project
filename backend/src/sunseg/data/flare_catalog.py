"""ชั้นกลางที่รวมแหล่ง flare catalog หลายแหล่งให้มีหน้าตาเดียวกัน

โปรเจครองรับสองแหล่ง ซึ่งให้ข้อมูลชนิดเดียวกันแต่มีจุดแข็งต่างกัน:

============  ==========================  =========================================
แหล่ง          ช่วงที่ครอบคลุม              ลักษณะ
============  ==========================  =========================================
``ngdc``      1975-2017                   ไฟล์สแตติกรายปี เร็ว เสถียร เป็นทางการ
``hek``       1996-ปัจจุบัน                บริการ query ยืดหยุ่นกว่า แต่ช้าและล่มบ่อย
============  ==========================  =========================================

``auto`` (ค่าเริ่มต้น) เลือก NGDC เมื่อช่วงเวลาที่ขออยู่ในคลังของมันทั้งหมด
มิฉะนั้นจึงใช้ HEK
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

FlareSource = Literal["auto", "ngdc", "hek"]

#: สคีมาที่ทุกแหล่งต้องคืนออกมาให้เหมือนกัน
FLARE_COLUMNS = [
    "start_time",
    "peak_time",
    "end_time",
    "goes_class",
    "peak_flux",
    "noaa_ar",
    "frm_name",
]


def resolve_source(requested: FlareSource, start: date, end: date) -> str:
    """ตัดสินว่าจะใช้แหล่งไหนจริง"""
    from .noaa_flares import NGDC_FIRST_YEAR, NGDC_LAST_YEAR

    if requested != "auto":
        return requested

    if NGDC_FIRST_YEAR <= start.year and end.year <= NGDC_LAST_YEAR:
        logger.info(
            "เลือกแหล่ง NGDC อัตโนมัติ (ช่วง %d-%d อยู่ในคลัง %d-%d และเร็วกว่า HEK มาก)",
            start.year,
            end.year,
            NGDC_FIRST_YEAR,
            NGDC_LAST_YEAR,
        )
        return "ngdc"

    logger.info(
        "เลือกแหล่ง HEK อัตโนมัติ (ช่วง %d-%d เกินคลัง NGDC ที่จบปี %d)",
        start.year,
        end.year,
        NGDC_LAST_YEAR,
    )
    return "hek"


def load_flares(
    source: FlareSource,
    start: date,
    end: date,
    cache_dir: Path,
    hek_chunk_days: int = 60,
) -> pd.DataFrame:
    """ดึงรายการ flare จากแหล่งที่เลือก แล้วคืนในสคีมามาตรฐาน"""
    chosen = resolve_source(source, start, end)

    if chosen == "ngdc":
        from .noaa_flares import fetch_flare_events as fetch_ngdc

        df = fetch_ngdc(start=start, end=end, cache_dir=cache_dir / "ngdc")
    elif chosen == "hek":
        from .hek_client import fetch_flare_events as fetch_hek

        df = fetch_hek(start=start, end=end, chunk_days=hek_chunk_days, cache_dir=cache_dir / "hek")
    else:
        raise ValueError(f"ไม่รู้จักแหล่ง flare catalog: {chosen!r}")

    missing = [c for c in FLARE_COLUMNS if c not in df.columns]
    if missing and not df.empty:
        raise ValueError(f"แหล่ง {chosen!r} คืนข้อมูลขาดคอลัมน์: {missing}")

    return df


def summarise_flares(df: pd.DataFrame) -> str:
    """สรุปจำนวน flare แยกตามคลาส — ใช้ตรวจว่าข้อมูลที่ดึงมาสมเหตุสมผล

    ตัวเลขที่คาดหวังในช่วง solar maximum (2012-2014) คือ flare คลาส C หลักพันต่อปี
    คลาส M หลักร้อย และคลาส X ไม่กี่สิบ ถ้าผิดจากนี้มากแสดงว่า parse ผิด
    """
    if df.empty:
        return "ไม่มี flare event"

    letters = df["goes_class"].astype("string").str[0].str.upper()
    counts = letters.value_counts().reindex(["A", "B", "C", "M", "X"]).fillna(0).astype(int)
    with_ar = int(df["noaa_ar"].notna().sum())

    return "\n".join(
        [
            f"flare ทั้งหมด {len(df):,} รายการ "
            f"({df['peak_time'].min():%Y-%m-%d} ถึง {df['peak_time'].max():%Y-%m-%d})",
            "  " + "  ".join(f"{c}={counts[c]:,}" for c in ["A", "B", "C", "M", "X"]),
            f"  ระบุ NOAA AR ได้ {with_ar:,} รายการ ({with_ar / len(df):.1%})",
        ]
    )


def map_flares_to_harps(
    flares: pd.DataFrame, noaa_to_harp: dict[int, list[int]]
) -> pd.DataFrame:
    """คลี่ flare แต่ละดวงไปยัง HARP ทุกดวงที่ตรงกับ NOAA AR ของมัน

    flare ที่ไม่มี NOAA AR หรือ AR นั้นไม่มี HARP คู่กัน จะถูกตัดทิ้ง — เราต้องรู้ให้ได้
    ว่า flare มาจาก HARP ไหนจึงจะสร้าง label ได้

    การตัดทิ้งนี้เป็นเรื่องปกติและไม่ทำให้ label ผิด: flare ที่ไม่มี AR ส่วนใหญ่เป็น
    คลาสเล็ก (B/C) ที่เกิดนอกกลุ่มจุดดับที่มีเลขทะเบียน
    """
    if flares.empty:
        return flares.assign(HARPNUM=pd.Series(dtype="int64"))

    rows: list[dict] = []
    n_no_ar = 0
    n_no_harp = 0

    for record in flares.to_dict("records"):
        noaa = record.get("noaa_ar")
        if noaa is None or pd.isna(noaa):
            n_no_ar += 1
            continue
        harps = noaa_to_harp.get(int(noaa))
        if not harps:
            n_no_harp += 1
            continue
        for harpnum in harps:
            rows.append({**record, "HARPNUM": harpnum})

    logger.info(
        "จับคู่ flare กับ HARP: ได้ %d คู่ จาก %d event "
        "(ไม่มีเลข AR %d, AR ไม่มี HARP คู่กัน %d)",
        len(rows),
        len(flares),
        n_no_ar,
        n_no_harp,
    )
    if not rows:
        return flares.assign(HARPNUM=pd.Series(dtype="int64"))
    return pd.DataFrame(rows).reset_index(drop=True)
