"""ตารางเชื่อม HARPNUM (SHARP) <-> NOAA active region number.

นี่คือตัวเชื่อมเดียวระหว่างสองโลก: ข้อมูลแม่เหล็ก SHARP ใช้ HARPNUM ส่วน flare
catalog (HEK/SWPC) ใช้ NOAA AR number. ถ้าไม่มีตารางนี้เราจะ label flare ให้ HARP
ไม่ได้เลย

ความสัมพันธ์เป็นแบบ many-to-many: HARP หนึ่งดวงอาจครอบคลุมหลาย NOAA AR (เมื่อ
กลุ่มจุดดับอยู่ใกล้กันจน JSOC รวมเป็น patch เดียว) และ NOAA AR หนึ่งดวงอาจปรากฏ
ในหลาย HARP ได้
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HARP_NOAA_URL = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"


def download_harp_noaa_table(dest: Path, force: bool = False, timeout: int = 60) -> Path:
    """ดาวน์โหลดตาราง HARPNUM->NOAA จาก JSOC (ไฟล์เล็ก ~100 KB)"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        logger.info("ใช้ตาราง HARP-NOAA ที่มีอยู่แล้ว: %s", dest)
        return dest

    logger.info("กำลังดาวน์โหลดตาราง HARP-NOAA จาก %s", HARP_NOAA_URL)
    with urllib.request.urlopen(HARP_NOAA_URL, timeout=timeout) as resp:  # noqa: S310
        content = resp.read().decode("utf-8")

    if "HARPNUM" not in content.upper():
        raise RuntimeError(
            f"เนื้อหาที่ดาวน์โหลดจาก {HARP_NOAA_URL} ไม่มีหัวตาราง HARPNUM — "
            "รูปแบบไฟล์ต้นทางอาจเปลี่ยนไป"
        )
    dest.write_text(content, encoding="utf-8")
    logger.info("บันทึกแล้ว: %s (%d ไบต์)", dest, len(content))
    return dest


def load_harp_noaa_pairs(path: Path) -> pd.DataFrame:
    """อ่านไฟล์ตารางแล้วคลี่เป็นคู่ (HARPNUM, NOAA_AR) หนึ่งแถวต่อหนึ่งคู่

    ไฟล์ต้นทางมีสองคอลัมน์คั่นด้วยช่องว่าง โดยคอลัมน์ที่สองอาจมีหลายเลขคั่นด้วย
    จุลภาค เช่น ``3234    11809,11812``
    """
    rows: list[tuple[int, int]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or line.upper().startswith("HARPNUM"):
            continue

        parts = line.split()
        if len(parts) < 2:
            logger.warning("ข้ามบรรทัดที่ %d ในรูปแบบที่ไม่คาดคิด: %r", lineno, raw)
            continue

        try:
            harpnum = int(parts[0])
        except ValueError:
            logger.warning("ข้ามบรรทัดที่ %d: HARPNUM ไม่ใช่ตัวเลข: %r", lineno, parts[0])
            continue

        for token in parts[1].split(","):
            token = token.strip()
            if not token:
                continue
            try:
                rows.append((harpnum, int(token)))
            except ValueError:
                logger.warning("ข้ามบรรทัดที่ %d: NOAA AR ไม่ใช่ตัวเลข: %r", lineno, token)

    if not rows:
        raise ValueError(f"ไม่พบคู่ HARPNUM-NOAA ที่ใช้งานได้ในไฟล์ {path}")

    df = pd.DataFrame(rows, columns=["HARPNUM", "NOAA_AR"]).drop_duplicates()
    logger.info(
        "โหลดคู่ HARP-NOAA สำเร็จ: %d คู่ (%d HARP, %d NOAA AR)",
        len(df),
        df["HARPNUM"].nunique(),
        df["NOAA_AR"].nunique(),
    )
    return df.reset_index(drop=True)


def build_noaa_to_harp(pairs: pd.DataFrame) -> dict[int, list[int]]:
    """NOAA AR -> รายการ HARPNUM (ใช้ตอน map flare event กลับเข้าหา HARP)"""
    grouped = pairs.groupby("NOAA_AR")["HARPNUM"].apply(lambda s: sorted(set(s)))
    return {int(k): [int(x) for x in v] for k, v in grouped.items()}


def build_harp_to_noaa(pairs: pd.DataFrame) -> dict[int, list[int]]:
    """HARPNUM -> รายการ NOAA AR"""
    grouped = pairs.groupby("HARPNUM")["NOAA_AR"].apply(lambda s: sorted(set(s)))
    return {int(k): [int(x) for x in v] for k, v in grouped.items()}
