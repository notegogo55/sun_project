"""อ่านรายการ flare จากไฟล์ GOES XRS report ของ NOAA/NGDC

เป็นแหล่ง label ที่ดีกว่า HEK สำหรับโปรเจคนี้ เพราะ:

* **เร็วกว่ามาก** — ไฟล์เดียวต่อปี (~200 KB) เทียบกับการ query HEK หลายสิบครั้ง
* **เสถียรกว่า** — เป็นไฟล์สแตติก ไม่ใช่บริการ query ที่ล่มได้
* **เป็นรายการทางการของ NOAA** — ไม่มีปัญหา event ซ้ำจากหลายอัลกอริทึม detection
* ครอบคลุม **1975-2017** ซึ่งพอดีกับช่วงข้อมูลของโปรเจค (2011-2017)

ข้อจำกัด: ไม่มีข้อมูลหลังปี 2017 — ถ้าต้องการช่วงใหม่กว่านั้นต้องใช้ HEK

รูปแบบไฟล์เป็น fixed-width ตัวอย่างจริงจาก ``goes-xrs-report_2014.txt``::

    31777140101  0721 0729 0726 S12W47      C 32    G15  1.0E-03 11940 131228.7
    |    |       |    |    |    |           | |     |    |       |
    |    |       |    |    |    |           | |     |    |       +-- NOAA AR (อาจไม่มี)
    |    |       |    |    |    |           | |     |    +---------- integrated flux
    |    |       |    |    |    |           | |     +--------------- ดาวเทียม
    |    |       |    |    |    |           | +--------------------- ความแรง x10 (32 -> 3.2)
    |    |       |    |    |    |           +----------------------- คลาส GOES
    |    |       |    |    |    +----------------------------------- ตำแหน่ง (อาจไม่มี)
    |    |       |    |    +---------------------------------------- เวลา peak HHMM
    |    |       |    +--------------------------------------------- เวลาสิ้นสุด HHMM
    |    |       +-------------------------------------------------- เวลาเริ่ม HHMM
    |    +---------------------------------------------------------- วันที่ YYMMDD
    +--------------------------------------------------------------- รหัสสถานี

.. warning::
   ลำดับคอลัมน์เวลาคือ **Begin, End, Max** ไม่ใช่ Begin, Max, End ตามที่หลายคน
   (และเอกสารสรุปบางฉบับ) เข้าใจผิด ตรวจสอบได้จากบรรทัดตัวอย่างข้างบน:
   ``0721 0729 0726`` = เริ่ม 07:21, จบ 07:29, peak 07:26 ซึ่งเรียงตามฟิสิกส์
   ถ้าสลับสองคอลัมน์นี้ เวลา peak จะออกมาหลังเวลาจบ (ดู tests/test_flare_catalog.py)
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .goes_class import goes_class_to_flux

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
    "solar-flares/x-rays/goes/xrs"
)

#: ปีแรกและปีสุดท้ายที่คลัง NGDC มีข้อมูล
NGDC_FIRST_YEAR = 1975
NGDC_LAST_YEAR = 2017

#: ตำแหน่งคอลัมน์แบบ fixed-width (0-based, ครึ่งเปิด) ของส่วนหัวบรรทัดที่แน่นอน
_COL_DATE = (5, 11)
_COL_START = (13, 17)
_COL_END = (18, 22)   # หมายเหตุ: End มาก่อน Max ในรูปแบบไฟล์นี้ (ดู warning ด้านบน)
_COL_MAX = (23, 27)
_COL_LOCATION = (28, 34)
_TAIL_START = 34

_SATELLITE_RE = re.compile(r"^G\d+$")
_SCIENTIFIC_RE = re.compile(r"^\d\.\d+E[+-]\d+$", re.IGNORECASE)
_CLASS_TOKEN_RE = re.compile(r"^([ABCMX])(\d*)$", re.IGNORECASE)
_LOCATION_RE = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})$", re.IGNORECASE)


def parse_location(value: str | None) -> tuple[float, float] | None:
    """แปลงตำแหน่ง flare แบบ ``N11W82`` เป็น ``(lat, lon)`` องศา Stonyhurst

    NOAA ระบุตำแหน่งเป็นพิกัด heliographic เทียบกับจุดศูนย์กลางจานที่มองเห็น
    เหนือ/ตะวันตกเป็นบวก ใต้/ตะวันออกเป็นลบ — ระบบเดียวกับ ``LAT_FWT``/``LON_FWT``
    ของ SHARP จึงวางซ้อนกันบนแผนที่จานสุริยะได้โดยไม่ต้องแปลงเพิ่ม

    flare ที่ไม่มีการยืนยันตำแหน่งทางแสงจะไม่มีค่านี้ (ราว 1 ใน 3 ของรายการ)
    จึงคืน ``None`` แทนการโยน exception — ผู้เรียกกรองทิ้งได้ตามปกติ

    >>> parse_location("N11W82")
    (11.0, 82.0)
    >>> parse_location("S26E34")
    (-26.0, -34.0)
    >>> parse_location(None) is None
    True
    """
    if not value:
        return None

    match = _LOCATION_RE.match(value.strip())
    if match is None:
        return None

    ns, lat_text, ew, lon_text = match.groups()
    lat = float(lat_text) * (1 if ns.upper() == "N" else -1)
    lon = float(lon_text) * (1 if ew.upper() == "W" else -1)

    # ตำแหน่งเกินขอบจานเป็นค่าที่อ่านผิด ไม่ใช่ flare ด้านหลังดวงอาทิตย์
    if abs(lat) > 90 or abs(lon) > 90:
        return None
    return lat, lon


def candidate_filenames(year: int) -> list[str]:
    """ชื่อไฟล์ที่เป็นไปได้ของแต่ละปี เรียงตามลำดับความน่าใช้

    NGDC ตั้งชื่อไม่สม่ำเสมอในบางปี — 2015 มีเวอร์ชันที่เติมแถวที่หายไปแล้ว
    (ควรใช้ตัวนี้) ส่วน 2017 เป็นไฟล์ year-to-date เพราะชุดข้อมูลจบที่ปีนั้น
    """
    names = []
    if year == 2015:
        names.append("goes-xrs-report_2015_modifiedreplacedmissingrows.txt")
    if year == 2017:
        names.append("goes-xrs-report_2017-ytd.txt")
    names.append(f"goes-xrs-report_{year}.txt")
    return names


def download_year(year: int, dest_dir: Path, force: bool = False, timeout: int = 90) -> Path | None:
    """ดาวน์โหลดไฟล์รายงานของปีหนึ่ง คืน ``None`` ถ้าไม่มีไฟล์ของปีนั้น"""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename in candidate_filenames(year):
        dest = dest_dir / filename
        if dest.exists() and not force and dest.stat().st_size > 0:
            logger.info("ใช้ไฟล์ที่มีอยู่แล้ว: %s", dest.name)
            return dest

        url = f"{BASE_URL}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "sunseg/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310
                content = resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                logger.debug("ไม่มีไฟล์ %s (404) — ลองชื่อถัดไป", filename)
                continue
            logger.warning("ดาวน์โหลด %s ล้มเหลว: HTTP %s", filename, exc.code)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("ดาวน์โหลด %s ล้มเหลว: %s", filename, exc)
            continue

        if not content.strip():
            logger.warning("ไฟล์ %s ว่างเปล่า", filename)
            continue

        dest.write_bytes(content)
        logger.info("ดาวน์โหลด %s สำเร็จ (%.0f KB)", filename, len(content) / 1024)
        return dest

    logger.warning("ไม่พบไฟล์รายงานของปี %d", year)
    return None


# --------------------------------------------------------------------------- #
# การ parse
# --------------------------------------------------------------------------- #


def _parse_hhmm(text: str) -> tuple[int, int] | None:
    text = text.strip()
    if len(text) != 4 or not text.isdigit():
        return None
    hour, minute = int(text[:2]), int(text[2:])
    # ในไฟล์นี้ '2400' หมายถึงเที่ยงคืนของวันถัดไป
    if hour == 24 and minute == 0:
        return 24, 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _combine(day: date, hhmm: tuple[int, int] | None) -> datetime | None:
    if hhmm is None:
        return None
    hour, minute = hhmm
    if hour == 24:
        return datetime(day.year, day.month, day.day) + timedelta(days=1)
    return datetime(day.year, day.month, day.day, hour, minute)


def _parse_tail(tail: str) -> dict | None:
    """แยกคลาส GOES, ดาวเทียม, integrated flux และ NOAA AR จากส่วนท้ายบรรทัด

    ใช้การจับคู่ตามรูปแบบของ token แทนตำแหน่งคอลัมน์ตายตัว เพราะระยะห่างของ
    คอลัมน์ท้ายบรรทัดไม่คงที่ระหว่างปี
    """
    tokens = tail.split()
    if not tokens:
        return None

    letter: str | None = None
    magnitude: float | None = None
    satellite: str | None = None
    flux: float | None = None
    noaa_ar: int | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if letter is None:
            match = _CLASS_TOKEN_RE.match(token)
            if match:
                letter = match.group(1).upper()
                digits = match.group(2)
                if digits:
                    # ตัวเลขติดกับตัวอักษร เช่น 'C32'
                    magnitude = int(digits) / 10.0
                elif idx + 1 < len(tokens) and tokens[idx + 1].isdigit():
                    # ตัวเลขแยก token เช่น 'C' '32'  (รูปแบบที่พบบ่อยที่สุด)
                    magnitude = int(tokens[idx + 1]) / 10.0
                    idx += 1
                idx += 1
                continue

        if satellite is None and _SATELLITE_RE.match(token):
            satellite = token
            idx += 1
            continue

        if flux is None and _SCIENTIFIC_RE.match(token):
            flux = float(token)
            # NOAA AR (ถ้ามี) อยู่ถัดจาก integrated flux เสมอ และเป็นจำนวนเต็มล้วน
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                if nxt.isdigit() and 4 <= len(nxt) <= 5:
                    noaa_ar = int(nxt)
            idx += 1
            continue

        idx += 1

    if letter is None or magnitude is None or magnitude <= 0:
        return None

    return {
        "goes_class": f"{letter}{magnitude:.1f}",
        "satellite": satellite,
        "integrated_flux": flux,
        "noaa_ar": noaa_ar,
    }


def parse_report_file(path: Path) -> pd.DataFrame:
    """แปลงไฟล์รายงานหนึ่งปีเป็น DataFrame"""
    rows: list[dict] = []
    n_skipped = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if len(raw) < _TAIL_START:
            n_skipped += 1
            continue

        date_text = raw[slice(*_COL_DATE)].strip()
        if len(date_text) != 6 or not date_text.isdigit():
            n_skipped += 1
            continue

        yy, mm, dd = int(date_text[:2]), int(date_text[2:4]), int(date_text[4:])
        # ชุดข้อมูลเริ่มปี 1975 — ปี 2 หลักตั้งแต่ 75 ขึ้นไปคือคริสต์ศตวรรษที่ 20
        year = 1900 + yy if yy >= 75 else 2000 + yy
        try:
            day = date(year, mm, dd)
        except ValueError:
            n_skipped += 1
            continue

        tail = _parse_tail(raw[_TAIL_START:])
        if tail is None:
            n_skipped += 1
            continue

        start_time = _combine(day, _parse_hhmm(raw[slice(*_COL_START)]))
        peak_time = _combine(day, _parse_hhmm(raw[slice(*_COL_MAX)]))
        end_time = _combine(day, _parse_hhmm(raw[slice(*_COL_END)]))

        if peak_time is None:
            peak_time = start_time
        if peak_time is None:
            n_skipped += 1
            continue

        # เหตุการณ์ที่คร่อมเที่ยงคืน: เวลาที่ "ย้อนกลับ" หมายถึงวันถัดไป
        if start_time is not None and peak_time < start_time:
            peak_time += timedelta(days=1)
        if end_time is not None and start_time is not None and end_time < start_time:
            end_time += timedelta(days=1)

        rows.append(
            {
                "start_time": start_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "goes_class": tail["goes_class"],
                "peak_flux": goes_class_to_flux(tail["goes_class"]),
                "noaa_ar": tail["noaa_ar"],
                "location": raw[slice(*_COL_LOCATION)].strip() or None,
                "satellite": tail["satellite"],
                "frm_name": "NOAA/NGDC",
            }
        )

    if n_skipped:
        logger.debug("%s: ข้ามบรรทัดที่ parse ไม่ได้ %d บรรทัด", path.name, n_skipped)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["noaa_ar"] = df["noaa_ar"].astype("Int64")
    logger.info("%s: อ่าน flare ได้ %d รายการ (ข้าม %d บรรทัด)", path.name, len(df), n_skipped)
    return df


def fetch_flare_events(
    start: date,
    end: date,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """ดึงและ parse รายการ flare ของทุกปีที่คาบเกี่ยวกับช่วงเวลาที่ขอ

    Returns
    -------
    DataFrame สคีมาเดียวกับ :func:`sunseg.data.hek_client.fetch_flare_events`
    เพื่อให้โค้ดปลายทางใช้แทนกันได้โดยไม่ต้องแก้
    """
    if start.year < NGDC_FIRST_YEAR or end.year > NGDC_LAST_YEAR:
        logger.warning(
            "คลัง NGDC ครอบคลุมเฉพาะปี %d-%d แต่ช่วงที่ขอคือ %d-%d — "
            "ข้อมูลนอกช่วงนี้จะไม่มี",
            NGDC_FIRST_YEAR,
            NGDC_LAST_YEAR,
            start.year,
            end.year,
        )

    frames: list[pd.DataFrame] = []
    for year in range(start.year, end.year + 1):
        if not (NGDC_FIRST_YEAR <= year <= NGDC_LAST_YEAR):
            continue
        path = download_year(year, cache_dir, force=force)
        if path is None:
            continue
        df = parse_report_file(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.error("ไม่สามารถดึงรายการ flare จาก NGDC ได้เลย")
        return pd.DataFrame(
            columns=[
                "start_time", "peak_time", "end_time", "goes_class",
                "peak_flux", "noaa_ar", "frm_name",
            ]
        )

    combined = pd.concat(frames, ignore_index=True)

    # กรองให้เหลือเฉพาะช่วงที่ขอจริง (ไฟล์เป็นรายปี จึงเกินขอบมาเสมอ)
    mask = (combined["peak_time"] >= pd.Timestamp(start)) & (
        combined["peak_time"] < pd.Timestamp(end)
    )
    result = combined[mask].sort_values("peak_time").reset_index(drop=True)

    logger.info(
        "รวม flare จาก NGDC: %d รายการ ในช่วง %s ถึง %s",
        len(result),
        start,
        end,
    )
    return result
