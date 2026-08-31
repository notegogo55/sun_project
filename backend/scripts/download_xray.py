"""ดาวน์โหลดฟลักซ์ GOES XRS ต่อเนื่องราย 1 นาที (GOES-15, ช่อง 1-8 Å) จาก NOAA NCEI

รายการ flare ที่ download_metadata.py ดึงมามีแค่ 3 จุดต่อเหตุการณ์ (เริ่ม/peak/จบ)
พอสำหรับ label ของโมเดล แต่วาดกราฟจากสามจุดนั้นได้เส้นเป็นหนามแหลม ๆ ไม่ใช่ฟลักซ์จริง
ที่ขึ้นลงต่อเนื่องแบบที่หน้า SWPC แสดง สคริปต์นี้ดึงฟลักซ์จริงรายนาทีมาแทน

ตัวอย่างการใช้งาน::

    # ทดสอบด้วยเดือนเดียวก่อน (แนะนำให้ทำครั้งแรกเสมอ — ไฟล์เต็มช่วงมีเกือบ 2,600 ไฟล์)
    python backend/scripts/download_xray.py --start 2014-10-01 --end 2014-10-31

    # ดึงเต็มช่วงตาม time_range ใน configs/data.yaml (2011-2017 ~150 MB, ใช้เวลานาน)
    python backend/scripts/download_xray.py

รันซ้ำได้ปลอดภัย — ไฟล์ที่มีอยู่แล้วจะถูกข้าม (เว้นแต่ใส่ --overwrite)
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sunseg.config import load_data_config  # noqa: E402
from sunseg.data.xray_flux import SOURCE_SATELLITES  # noqa: E402
from sunseg.logging_utils import setup_logging  # noqa: E402

logger = logging.getLogger("download_xray")

NCEI_ROOT = "https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs"
_PRODUCT = "xrsf-l2-avg1m_science"

# ดวงเดียวที่ใช้จริงตอนนี้คือ g15 (ดู SOURCE_SATELLITES ใน xray_flux.py) — regex
# ครอบคลุมทุกดวงไว้เพื่อให้ขยายทีหลังได้โดยไม่ต้องแก้ตรงนี้
_FILE_RE = re.compile(r'href="(sci_xrsf-l2-avg1m_g\d+_d(\d{8})_v[\d.\-]+\.nc)"')


def _satellite_dir(satellite: str) -> str:
    """ชื่อไดเรกทอรีของ NOAA ("g15" -> "goes15") — ต่างจากชื่อในไฟล์ตรงๆ ("g15")
    ต้องแยกสองชื่อนี้ให้ชัด ผสมกันแล้ว URL จะ 404 แบบเงียบๆ (เจอมาแล้วตอนพัฒนา)"""
    return "goes" + satellite.removeprefix("g")


def _fetch(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "sunseg/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _list_month(satellite: str, year: int, month: int) -> list[str]:
    """ชื่อไฟล์ทั้งหมดที่ NOAA มีจริงในเดือนนั้น — คืน [] เงียบๆ ถ้ายังไม่มีข้อมูล
    (เดือนปัจจุบันที่ยังไม่ reprocess) หรือเน็ตหลุดชั่วคราว ให้ผู้เรียกตัดสินใจเอง"""
    url = f"{NCEI_ROOT}/{_satellite_dir(satellite)}/{_PRODUCT}/{year}/{month:02d}/"
    try:
        html = _fetch(url).decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return []
    return [name for name, _ in _FILE_RE.findall(html)]


def _months_between(start: date, end: date) -> list[tuple[int, int]]:
    months = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def _download_one(
    satellite: str, filename: str, year: int, month: int, dest_root: Path, overwrite: bool
) -> tuple[str, str]:
    out = dest_root / str(year) / filename
    if out.exists() and not overwrite:
        return str(out), "skip"
    url = f"{NCEI_ROOT}/{_satellite_dir(satellite)}/{_PRODUCT}/{year}/{month:02d}/{filename}"
    try:
        data = _fetch(url, timeout=120)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return str(out), f"error: {type(exc).__name__}: {exc}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)  # atomic — โดนตัดกลางทางจะไม่ค้างเป็นไฟล์ดีที่จริงๆ เสีย
    return str(out), "ok"


def download(
    start: date,
    end: date,
    dest_root: Path,
    satellite: str = SOURCE_SATELLITES[0],
    workers: int = 8,
    overwrite: bool = False,
) -> None:
    if end < start:
        raise ValueError(f"end ({end}) ต้องมาหลัง start ({start})")

    jobs: list[tuple[str, int, int]] = []
    for year, month in _months_between(start, end):
        for filename in _list_month(satellite, year, month):
            match = re.search(r"_d(\d{8})_", filename)
            assert match is not None
            day = date(int(match.group(1)[:4]), int(match.group(1)[4:6]), int(match.group(1)[6:8]))
            if start <= day <= end:
                jobs.append((filename, year, month))

    logger.info("พบ %d ไฟล์ในช่วง %s..%s (ดาวเทียม %s)", len(jobs), start, end, satellite)
    if not jobs:
        logger.warning("ไม่พบไฟล์เลย — ตรวจช่วงวันที่ หรือ NOAA อาจยังไม่ reprocess เดือนนี้")
        return

    ok = skip = err = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, satellite, filename, year, month, dest_root, overwrite)
            for filename, year, month in jobs
        ]
        for i, future in enumerate(as_completed(futures), 1):
            out, status = future.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                logger.warning("  %s: %s", out, status)
            if i % 200 == 0 or i == len(jobs):
                logger.info("  %d/%d (โหลดใหม่=%d ข้าม=%d พลาด=%d)", i, len(jobs), ok, skip, err)

    logger.info("เสร็จ: โหลดใหม่=%d ข้าม (มีอยู่แล้ว)=%d พลาด=%d", ok, skip, err)
    if err:
        logger.warning("มีไฟล์พลาด %d ไฟล์ — รันคำสั่งเดิมซ้ำได้ จะโหลดเฉพาะที่ยังขาด", err)


def main() -> int:
    setup_logging()
    config = load_data_config()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=config.time_range.start)
    parser.add_argument("--end", type=date.fromisoformat, default=config.time_range.end)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true", help="โหลดทับไฟล์ที่มีอยู่แล้ว")
    args = parser.parse_args()

    dest_root = config.paths.raw / "xrs"
    logger.info("=" * 62)
    logger.info("ดาวน์โหลดฟลักซ์ GOES XRS ต่อเนื่อง -> %s", dest_root)
    logger.info("=" * 62)
    download(args.start, args.end, dest_root, workers=args.workers, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
