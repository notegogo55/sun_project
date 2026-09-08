"""ฟลักซ์รังสีเอกซ์ GOES XRS ต่อเนื่องราย 1 นาที (channel ยาว 1-8 Å)

รายการ flare จาก ``noaa_flares.py`` บอกแค่ "เกิดอะไรตอนไหน" (เริ่ม/peak/จบ 3 จุด)
ซึ่งพอสำหรับ label ของโมเดล แต่วาดกราฟจากสามจุดนั้นจะได้เส้นเป็นหนามแหลม ๆ
ไม่ใช่ฟลักซ์จริงที่ขึ้น-ลงต่อเนื่องแบบที่ SWPC แสดงบนเว็บ — โมดูลนี้อ่านฟลักซ์จริง
รายนาทีจากไฟล์ที่ ``scripts/download_xray.py`` ดาวน์โหลดไว้ล่วงหน้า

**GOES-15 (2011-2017) + GOES-16 (หน้าต่าง case study พ.ค. 2024 เป็นต้นไป)** —
GOES-15/14 (คลัง legacy "GOES 1-15" ของ NCEI) หยุดมีข้อมูลหลัง 2020-03-04 ส่วน
GOES-16 (คลัง "GOES-R series" คนละ URL root กันโดยสิ้นเชิง — ดู
``scripts/download_xray.py``) เป็นดาวเทียมที่ยังทำงานอยู่ช่วง พ.ค. 2024 (พายุ Gannon)
ออกแบบให้เพิ่มดาวเทียมอื่นได้ทีหลังผ่าน ``SOURCE_SATELLITES`` โดยไม่ต้องแก้โครงสร้าง

รูปแบบไฟล์ (NOAA NCEI ``xrsf-l2-avg1m_science``) เป็น NetCDF4 ซึ่งจริง ๆ คือ HDF5
จึงอ่านด้วย ``h5py`` ได้ตรง ๆ โดยไม่ต้องติดตั้งแพคเกจ ``netCDF4``/``xarray`` เพิ่ม —
ทั้ง GOES 1-15 และ GOES-R series ใช้ตัวแปรชื่อเดียวกัน (``time``, ``xrsb_flux``,
``xrsb_flag``) และความหมายของ flag เหมือนกัน (0 = good_data) จึงอ่านด้วยฟังก์ชัน
เดียวกันได้ทั้งคู่ ต่างกันแค่ตัวคูณสเกล (ดู ``SATELLITE_SCALE``)

**เรื่องสเกล**: GOES 1-15 กับ GOES-R series คาลิเบรตกันคนละวิธี — เอกสารของ NOAA
ระบุว่าข้อมูล "science quality" ของทั้งสองรุ่นควรอยู่ในหน่วยฟิสิกส์จริงแล้วทั้งคู่
(https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments) แต่ตรวจสอบ
ด้วยข้อมูลจริงในช่วงที่ทั้งสองดวงทำงานทับกัน (GOES-15 กับ GOES-16 ทับกัน 2017 ถึง
2020-03-04) แล้วพบว่า **ไม่ตรงกันเป๊ะ**: g16/g15 median = 1.0805 (n=29,620 จุด
ที่มีข้อมูลดีทั้งคู่ ช่วง 2017-08-15..2017-09-15) ตรวจซ้ำเฉพาะจุดระดับ M-class+
(flux > 1e-5) ได้ 1.0781 (n=1,910) และที่ peak ของ flare X9.3 วันที่ 6 ก.ย. 2017
(เหตุการณ์จริงที่มีชื่อ ตรวจสอบย้อนกลับได้) ได้ 1.0949 — ทั้งสามวิธีวัดสอดคล้องกัน
ในช่วง 1.08-1.09 จึงใช้ **1.0805 (median ทั้งช่วง)** เป็นตัวคูณสเกล ปรับ GOES-16
ให้อยู่บนสเกลเดียวกับ GOES-15 (ตัวอ้างอิง เพราะเป็นดาวเทียมของชุดข้อมูลเทรนหลัก)
สคริปต์ที่ใช้วัด (ดาวน์โหลด + คำนวณอัตราส่วน) ไม่ได้ commit ไว้ในนี้ — วิธีวัดซ้ำได้
คือดาวน์โหลดช่วงทับซ้อนของทั้งสองดวงแล้วเทียบ flux ที่ timestamp เดียวกัน
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: ดาวเทียมที่อ่าน เรียงตามลำดับความน่าเชื่อถือ/priority — ในทางปฏิบัติช่วงเวลาที่
#: แต่ละดวงมีไฟล์ไม่ทับกัน (g15 หยุดข้อมูล 2020-03-04, g16 ใช้ตั้งแต่ case study
#: พ.ค. 2024) จึงไม่มีวันไหนที่ต้องเลือกระหว่างสองดวงจริง ๆ แต่คงลำดับนี้ไว้เผื่อ
#: อนาคตมีไฟล์ซ้อนกัน
SOURCE_SATELLITES: tuple[str, ...] = ("g15", "g16")

#: ตัวคูณสเกลเทียบกับ g15 (ตัวอ้างอิง) — ดู "เรื่องสเกล" ใน docstring บนสำหรับที่มา
#: ของตัวเลขและวิธีวัด ดาวเทียมที่ไม่อยู่ในนี้ถือว่าสเกล = 1.0 (ยังไม่ได้วัด)
SATELLITE_SCALE: dict[str, float] = {
    "g15": 1.0,
    "g16": 1.0 / 1.0805,
}

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

    def _day_path(self, day: date) -> tuple[Path, str] | None:
        """หาไฟล์ของวันนั้น — ไม่ผูก version string ตายตัว เผื่อ NOAA reprocess ใหม่

        คืนดาวเทียมที่เจอมาด้วย เพราะไฟล์ของแต่ละดวงต้องคูณตัวคูณสเกลคนละค่า
        (ดู ``SATELLITE_SCALE``) — เดาจากชื่อไฟล์อย่างเดียวหลังเปิดไฟล์แล้วจะไม่รู้
        """
        year_dir = self.root / str(day.year)
        if not year_dir.is_dir():
            return None
        for sat in SOURCE_SATELLITES:
            matches = sorted(year_dir.glob(f"sci_xrsf-l2-avg1m_{sat}_d{day:%Y%m%d}_v*.nc"))
            if matches:
                return matches[-1], sat  # หลาย version เอาตัวล่าสุด (เรียงชื่อ = เรียง version)
        return None

    def _read_raw(self, start: datetime, end: datetime) -> tuple[np.ndarray, np.ndarray]:
        """อ่านทุกไฟล์รายวันในช่วง ต่อกันโดยไม่ตัดขอบ/ไม่ยุบข้อมูล — ตัวช่วยภายในที่ทั้ง
        ``series()`` (ตัดขอบ + ยุบด้วย _decimate_keep_peaks สำหรับวาดกราฟ) และ
        ``bin_series()`` (ยุบลงกริดด้วยสถิติต่อ bin สำหรับ dataset) ใช้ร่วมกัน — สอง
        ทางนี้ต้องการข้อมูลดิบครบ ไม่ผ่านการยุบแบบ "เก็บพีค" ของ series() ซึ่งจะทำให้
        สถิติต่อ bin ผิดเพราะข้อมูลส่วนใหญ่หายไปเหลือแต่พีค
        """
        times: list[np.ndarray] = []
        flux: list[np.ndarray] = []

        day = start.date()
        while day <= end.date():
            found = self._day_path(day)
            if found is not None:
                path, satellite = found
                day_times, day_flux = _read_day(path, satellite)
                times.append(day_times)
                flux.append(day_flux)
            day += timedelta(days=1)

        if not times:
            return np.array([], dtype="datetime64[s]"), np.array([], dtype="float64")

        all_times = np.concatenate(times)
        all_flux = np.concatenate(flux)
        order = np.argsort(all_times)
        return all_times[order], all_flux[order]

    def series(self, start: datetime, end: datetime) -> XraySeries:
        """ฟลักซ์ช่อง 1-8 Å ตั้งแต่ ``start`` ถึง ``end`` (รวมขอบทั้งสองด้าน)

        อ่านทีละไฟล์รายวัน (ไฟล์ละ ~1,440 แถว เปิดเร็ว) แล้วต่อกัน — ไม่ต้องมี cache
        ต่างหากเหมือน proton เพราะไฟล์เล็กพอที่จะเปิดสดทุกคำขอได้โดยไม่กระทบ latency
        """
        all_times, all_flux = self._read_raw(start, end)
        if len(all_times) == 0:
            return XraySeries(times=[], flux=[], decimated=False)

        mask = (all_times >= np.datetime64(start)) & (all_times <= np.datetime64(end))
        all_times = all_times[mask]
        all_flux = all_flux[mask]

        decimated = len(all_times) > MAX_POINTS
        if decimated:
            all_times, all_flux = _decimate_keep_peaks(all_times, all_flux, MAX_POINTS)

        return XraySeries(
            times=[t.astype("datetime64[s]").astype(datetime) for t in all_times],
            flux=[None if np.isnan(v) else float(v) for v in all_flux],
            decimated=decimated,
        )

    def bin_series(self, start: datetime, end: datetime, cadence_hours: int) -> pd.DataFrame:
        """ยุบฟลักซ์รายนาทีลงกริดเวลาสม่ำเสมอทุก ``cadence_hours`` ชม. — ใช้ทำ feature
        ของ sequence dataset ไม่ใช่วาดกราฟ (นั่นคือหน้าที่ของ ``series()``)

        bin ที่มี timestamp ``t`` ครอบคลุมช่วง ``(t - cadence_hours ชม., t]`` — ค่าที่
        "เห็นได้" ณ เวลาที่ออกพยากรณ์คือฟลักซ์ในอดีตที่ผ่านมาเท่านั้น ไม่ใช่ในอนาคต

        คืนสถิติ 3 ค่าต่อ bin (median, max, min) ตามหลัก "เก็บครบ เลือกใช้ทีหลัง" เดียว
        กับที่ ``sunseg.data.intensity`` ใช้กับความเข้มแสง พร้อมคอลัมน์ ``xray_log10_median``
        ที่แปลง log10 ไว้ให้พร้อมใช้เป็น feature ตรงๆ — ต้องแปลงตรงนี้ ไม่ใช่ปล่อยให้
        ``signed_log1p`` ทั่วไปที่ใช้กับ SHARP จัดการ เพราะฟลักซ์ดิบมีค่าเล็กมาก
        (ระดับ 1e-9 ถึง 1e-3) ซึ่ง log1p(x) ≈ x ในช่วงนี้แทบไม่บีบอัด scale ให้เลย

        bin ที่ไม่มีข้อมูลจริงเลยในช่วงของมัน (ไฟล์ขาด หรือทุกจุดถูกกรองเป็น NaN) ได้ NaN
        ทั้ง 4 คอลัมน์ ไม่ใช่ 0 — 0 จะแปลว่า "ไม่มีรังสีเอกซ์เลย" ซึ่งผิด ความจริงคือ "ไม่รู้"
        """
        cadence = timedelta(hours=cadence_hours)
        grid = pd.date_range(start, end, freq=cadence)

        all_times, all_flux = self._read_raw(start - cadence, end)
        finite = np.isfinite(all_flux)
        times = all_times[finite]  # เรียงแล้วตั้งแต่ _read_raw — searchsorted ใช้ได้ตรงๆ
        flux = all_flux[finite]

        # แทนที่จะ mask อาร์เรย์ทั้งก้อนทีละ bin (O(n_bins * n_points) ซึ่งช้ามากเมื่อช่วง
        # เวลายาวหลายปี — ข้อมูลรายนาทีหลายล้านจุดคูณเข้ากับ bin หลักพัน) ใช้ searchsorted
        # หาขอบเขตของทุก bin ในคราวเดียว (O(n_bins log n_points)) แล้วให้แต่ละ bin แตะ
        # เฉพาะช่วงของตัวเอง งานรวมทั้งหมดจึงเป็น O(n_points + n_bins log n_points)
        edges_hi = grid.values.astype("datetime64[s]")
        edges_lo = edges_hi - np.timedelta64(cadence)
        lo_idx = np.searchsorted(times, edges_lo, side="right")  # times > lo
        hi_idx = np.searchsorted(times, edges_hi, side="right")  # times <= hi

        medians = np.full(len(grid), np.nan)
        maxes = np.full(len(grid), np.nan)
        mins = np.full(len(grid), np.nan)
        for i in range(len(grid)):
            if hi_idx[i] <= lo_idx[i]:
                continue
            vals = flux[lo_idx[i] : hi_idx[i]]
            medians[i] = np.median(vals)
            maxes[i] = np.max(vals)
            mins[i] = np.min(vals)

        positive = np.isfinite(medians) & (medians > 0)
        log10_median = np.full_like(medians, np.nan)
        log10_median[positive] = np.log10(medians[positive])

        return pd.DataFrame(
            {
                "t_rec": grid,
                "xray_median": medians,
                "xray_max": maxes,
                "xray_min": mins,
                "xray_log10_median": log10_median,
            }
        )


# --------------------------------------------------------------------------- #
# อ่านไฟล์รายวัน
# --------------------------------------------------------------------------- #


def _read_day(path: Path, satellite: str) -> tuple[np.ndarray, np.ndarray]:
    """หนึ่งไฟล์ -> (เวลา, ฟลักซ์ช่องยาว) จุดที่คุณภาพไม่ดีถูกแทนด้วย NaN ไม่ใช่ตัดทิ้ง
    เพื่อให้กริดเวลายังต่อเนื่อง (ช่องว่างในกราฟ ไม่ใช่เวลาที่หายไปเงียบ ๆ)

    ``satellite`` กำหนดตัวคูณสเกล (``SATELLITE_SCALE``) — GOES 1-15 กับ GOES-R series
    ใช้ตัวแปรชื่อเดียวกันและความหมาย flag เดียวกัน จึงอ่านโครงสร้างไฟล์แบบเดียวกันได้
    ต่างกันแค่ต้องคูณค่าที่อ่านมาด้วยตัวคูณของดวงนั้นก่อนคืนออกไป
    """
    import h5py

    with h5py.File(path, "r") as handle:
        seconds = handle["time"][:].astype("float64")
        flux = handle["xrsb_flux"][:].astype("float64")
        flag = handle["xrsb_flag"][:]

    times = np.array(_EPOCH, dtype="datetime64[s]") + seconds.astype("timedelta64[s]")
    # flag == 0 คือ "good_data" ล้วน ๆ — บิตอื่นทุกตัว (bad/eclipsed/temperature_recovery
    # ใน GOES 1-15, หรือ eclipse/bad_data/interpolated ใน GOES-R — คำต่างกันแต่ 0 แปลว่า
    # "ดี" เหมือนกันทั้งคู่) ถือว่าอ่านค่าไม่ได้ ปลอดภัยกว่าเดา
    bad = (flag != 0) | ~np.isfinite(flux) | (flux <= 0) | (flux <= -9998.0)
    flux = np.where(bad, np.nan, flux) * SATELLITE_SCALE.get(satellite, 1.0)
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
