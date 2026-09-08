"""ฟลักซ์รังสีเอกซ์ GOES XRS ต่อเนื่องราย 1 นาที (channel ยาว 1-8 Å)

รายการ flare จาก ``noaa_flares.py`` บอกแค่ "เกิดอะไรตอนไหน" (เริ่ม/peak/จบ 3 จุด)
ซึ่งพอสำหรับ label ของโมเดล แต่วาดกราฟจากสามจุดนั้นจะได้เส้นเป็นหนามแหลม ๆ
ไม่ใช่ฟลักซ์จริงที่ขึ้น-ลงต่อเนื่องแบบที่ SWPC แสดงบนเว็บ — โมดูลนี้อ่านฟลักซ์จริง
รายนาทีจากไฟล์ที่ ``scripts/download_xray.py`` ดาวน์โหลดไว้ล่วงหน้า

**เฉพาะ GOES-15 ช่วง 2011-2017** ซึ่งครอบคลุม ``time_range`` ของโปรเจคพอดี (ตรวจสอบ
ไดเรกทอรีของ NOAA แล้ว: GOES-15 มีข้อมูลต่อเนื่องครบทุกปีในช่วงนี้ ส่วน GOES-13/14 ขาด
บางปี) ออกแบบให้เพิ่มดาวเทียมอื่นได้ทีหลังผ่าน ``SOURCE_SATELLITES`` โดยไม่ต้องแก้
โครงสร้าง แต่ตอนนี้ยังไม่มีความจำเป็นเพราะดวงเดียวก็พอแล้ว

รูปแบบไฟล์ (NOAA NCEI ``xrsf-l2-avg1m_science``) เป็น NetCDF4 ซึ่งจริง ๆ คือ HDF5
จึงอ่านด้วย ``h5py`` ได้ตรง ๆ โดยไม่ต้องติดตั้งแพคเกจ ``netCDF4``/``xarray`` เพิ่ม
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: ดาวเทียมที่อ่าน เรียงตามลำดับความน่าเชื่อถือ (ตอนนี้มีดวงเดียว — ดู docstring บน)
SOURCE_SATELLITES: tuple[str, ...] = ("g15",)

#: NOAA อ้างเวลาเป็นวินาทีนับจากจุดนี้ (ระบุไว้ใน attribute ``units`` ของตัวแปร time)
_EPOCH = datetime(2000, 1, 1, 12, 0, 0)

_FILENAME_RE = re.compile(r"sci_xrsf-l2-avg1m_(?P<sat>g\d+)_d(?P<date>\d{8})_v[\d.\-]+\.nc")

#: คำตอบสูงสุดต่อคำขอ — ช่วงยาวกว่านี้จะถูกยุบด้วยค่าสูงสุดในแต่ละช่วงย่อย (ดู _decimate)
#: ไม่ใช่ค่าเฉลี่ย เพื่อไม่ให้พีคของ flare แคบ ๆ หายไปตอนซูมออกดูช่วงยาว ๆ
MAX_POINTS = 20_000


@dataclass
class XraySeries:
    times: list[datetime]
    flux: list[float | None]  # W/m^2, channel ยาว (1-8 Å) — ตัวกำหนดคลาส GOES
    decimated: bool


class XrayFluxStore:
    """ดัชนีไฟล์ XRS รายวันที่ดาวน์โหลดไว้ + การอ่านช่วงเวลาที่ต้องการ

    ไม่มีไฟล์ = ไม่มีปัญหา: ``available`` เป็น False แล้วหน้าเว็บใช้เส้นที่ประกอบจาก
    รายการ flare แทนแบบเดิม (ดู sunseg.data.noaa_flares) — เหมือนกับส่วนอื่นของแอป
    ที่ต้อง degrade ได้เมื่อยังไม่ได้รันสคริปต์ดาวน์โหลด
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._available = self.root.is_dir() and any(self.root.rglob("*.nc"))
        if self._available:
            logger.info("พบไฟล์ XRS ที่ %s", self.root)
        else:
            logger.warning(
                "ไม่พบไฟล์ XRS ที่ %s — เส้น X-ray จะใช้ข้อมูลจากรายการ flare แทน "
                "(รัน backend/scripts/download_xray.py เพื่อได้เส้นต่อเนื่องจริง)",
                self.root,
            )

    @property
    def available(self) -> bool:
        return self._available

    def info(self) -> dict:
        return {
            "available": self._available,
            "root": str(self.root),
            "satellites": list(SOURCE_SATELLITES),
        }

    # ------------------------------------------------------------------ #

    def _day_path(self, day: date) -> Path | None:
        """หาไฟล์ของวันนั้น — ไม่ผูก version string ตายตัว เผื่อ NOAA reprocess ใหม่"""
        year_dir = self.root / str(day.year)
        if not year_dir.is_dir():
            return None
        for sat in SOURCE_SATELLITES:
            matches = sorted(year_dir.glob(f"sci_xrsf-l2-avg1m_{sat}_d{day:%Y%m%d}_v*.nc"))
            if matches:
                return matches[-1]  # ถ้ามีหลาย version เอาตัวล่าสุด (เรียงชื่อ = เรียง version)
        return None

    def series(self, start: datetime, end: datetime) -> XraySeries:
        """ฟลักซ์ช่อง 1-8 Å ตั้งแต่ ``start`` ถึง ``end`` (รวมขอบทั้งสองด้าน)

        อ่านทีละไฟล์รายวัน (ไฟล์ละ ~1,440 แถว เปิดเร็ว) แล้วต่อกัน — ไม่ต้องมี cache
        ต่างหากเหมือน proton เพราะไฟล์เล็กพอที่จะเปิดสดทุกคำขอได้โดยไม่กระทบ latency
        """
        times: list[np.ndarray] = []
        flux: list[np.ndarray] = []

        day = start.date()
        while day <= end.date():
            path = self._day_path(day)
            if path is not None:
                day_times, day_flux = _read_day(path)
                times.append(day_times)
                flux.append(day_flux)
            day += timedelta(days=1)

        if not times:
            return XraySeries(times=[], flux=[], decimated=False)

        all_times = np.concatenate(times)
        all_flux = np.concatenate(flux)

        mask = (all_times >= np.datetime64(start)) & (all_times <= np.datetime64(end))
        all_times = all_times[mask]
        all_flux = all_flux[mask]

        order = np.argsort(all_times)
        all_times = all_times[order]
        all_flux = all_flux[order]

        decimated = len(all_times) > MAX_POINTS
        if decimated:
            all_times, all_flux = _decimate_keep_peaks(all_times, all_flux, MAX_POINTS)

        return XraySeries(
            times=[t.astype("datetime64[s]").astype(datetime) for t in all_times],
            flux=[None if np.isnan(v) else float(v) for v in all_flux],
            decimated=decimated,
        )


# --------------------------------------------------------------------------- #
# อ่านไฟล์รายวัน
# --------------------------------------------------------------------------- #


def _read_day(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """หนึ่งไฟล์ -> (เวลา, ฟลักซ์ช่องยาว) จุดที่คุณภาพไม่ดีถูกแทนด้วย NaN ไม่ใช่ตัดทิ้ง
    เพื่อให้กริดเวลายังต่อเนื่อง (ช่องว่างในกราฟ ไม่ใช่เวลาที่หายไปเงียบ ๆ)"""
    import h5py

    with h5py.File(path, "r") as handle:
        seconds = handle["time"][:].astype("float64")
        flux = handle["xrsb_flux"][:].astype("float64")
        flag = handle["xrsb_flag"][:]

    times = np.array(_EPOCH, dtype="datetime64[s]") + seconds.astype("timedelta64[s]")
    # flag == 0 คือ "good_data" ล้วน ๆ — บิตอื่นทุกตัว (bad/eclipsed/temperature_recovery)
    # ถือว่าอ่านค่าไม่ได้ ปลอดภัยกว่าเดา
    bad = (flag != 0) | ~np.isfinite(flux) | (flux <= 0) | (flux <= -9998.0)
    flux = np.where(bad, np.nan, flux)
    return times, flux


def _decimate_keep_peaks(
    times: np.ndarray, flux: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """ยุบให้เหลือไม่เกิน ``max_points`` โดยเก็บ **ค่าสูงสุด** ของแต่ละช่วงย่อยไว้

    ใช้ max แทน mean/สุ่มตัวอย่าง เพราะสิ่งที่สำคัญที่สุดบนกราฟนี้คือพีคของ flare —
    ช่วงเวลายาวหลายเดือนที่ต้องยุบหนัก ค่าเฉลี่ยจะกลบพีคแคบ ๆ ที่กินเวลาไม่กี่นาทีจนหายไป
    """
    n = len(times)
    bucket = int(np.ceil(n / max_points))
    n_buckets = int(np.ceil(n / bucket))

    out_times = np.empty(n_buckets, dtype=times.dtype)
    out_flux = np.empty(n_buckets, dtype=flux.dtype)

    for i in range(n_buckets):
        chunk_flux = flux[i * bucket : (i + 1) * bucket]
        chunk_times = times[i * bucket : (i + 1) * bucket]
        if np.all(np.isnan(chunk_flux)):
            out_times[i] = chunk_times[len(chunk_times) // 2]
            out_flux[i] = np.nan
        else:
            peak = int(np.nanargmax(chunk_flux))
            out_times[i] = chunk_times[peak]
            out_flux[i] = chunk_flux[peak]

    return out_times, out_flux
