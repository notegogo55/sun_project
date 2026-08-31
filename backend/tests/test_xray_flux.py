"""ทดสอบตัวอ่านฟลักซ์ X-ray ต่อเนื่องรายนาที

สร้างไฟล์ HDF5 จำลองที่มีโครงสร้างตัวแปรเดียวกับไฟล์จริงของ NOAA (``time``,
``xrsb_flux``, ``xrsb_flag``) แทนที่จะพึ่งไฟล์จริงที่ต้องดาวน์โหลด — ทำให้ทดสอบได้
โดยไม่ต้องมีอินเทอร์เน็ตหรือรัน ``download_xray.py`` ก่อน
"""

from __future__ import annotations

from datetime import datetime

import h5py
import numpy as np
import pytest

from sunseg.data.xray_flux import XrayFluxStore, _decimate_keep_peaks

_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def _write_day(root, day: datetime, flux: list[float], flag: list[int]) -> None:
    """เขียนไฟล์รายวันจำลอง 1 ไฟล์ในรูปแบบเดียวกับ ``sci_xrsf-l2-avg1m_g15_dYYYYMMDD_v*.nc``"""
    n = len(flux)
    assert len(flag) == n
    seconds = [
        (day.replace(hour=0, minute=0, second=0) - _EPOCH).total_seconds() + i * 60
        for i in range(n)
    ]

    year_dir = root / str(day.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"sci_xrsf-l2-avg1m_g15_d{day:%Y%m%d}_v2-2-1.nc"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("time", data=np.array(seconds, dtype="float64"))
        handle.create_dataset("xrsb_flux", data=np.array(flux, dtype="float32"))
        handle.create_dataset("xrsb_flag", data=np.array(flag, dtype="uint16"))


class TestIndexing:
    def test_missing_directory_degrades_instead_of_raising(self, tmp_path):
        store = XrayFluxStore(tmp_path / "ไม่มีอยู่จริง")
        assert not store.available

    def test_available_once_at_least_one_day_exists(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 3, [0] * 3)
        assert XrayFluxStore(tmp_path).available

    def test_range_with_no_matching_files_returns_empty_not_an_error(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 3, [0] * 3)
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(1999, 1, 1), datetime(1999, 1, 2))

        assert series.times == []
        assert series.flux == []


class TestSeries:
    def test_reads_values_from_the_right_minute(self, tmp_path):
        flux = [float(i) * 1e-7 for i in range(1440)]
        _write_day(tmp_path, datetime(2014, 10, 18), flux, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18, 0, 30), datetime(2014, 10, 18, 0, 32))

        assert series.times == [
            datetime(2014, 10, 18, 0, 30),
            datetime(2014, 10, 18, 0, 31),
            datetime(2014, 10, 18, 0, 32),
        ]
        assert series.flux == pytest.approx([30 * 1e-7, 31 * 1e-7, 32 * 1e-7])

    def test_bad_flag_becomes_none_not_zero(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6, 1e-6, 1e-6], [0, 1, 2])
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18, 0, 0), datetime(2014, 10, 18, 0, 2))

        assert series.flux[0] == pytest.approx(1e-6, rel=1e-6)
        assert series.flux[1:] == [None, None]

    def test_fill_value_and_non_positive_flux_become_none(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6, -9999.0, 0.0, -1.0], [0, 0, 0, 0])
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18, 0, 0), datetime(2014, 10, 18, 0, 3))

        assert series.flux[0] == pytest.approx(1e-6, rel=1e-6)
        assert series.flux[1:] == [None, None, None]

    def test_spans_multiple_days(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 1440, [0] * 1440)
        _write_day(tmp_path, datetime(2014, 10, 19), [2e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18, 23, 58), datetime(2014, 10, 19, 0, 1))

        assert series.times == [
            datetime(2014, 10, 18, 23, 58),
            datetime(2014, 10, 18, 23, 59),
            datetime(2014, 10, 19, 0, 0),
            datetime(2014, 10, 19, 0, 1),
        ]
        assert series.flux == pytest.approx([1e-6, 1e-6, 2e-6, 2e-6])

    def test_missing_day_in_the_middle_leaves_a_gap_not_a_crash(self, tmp_path):
        """วันที่ไฟล์หายไปกลางช่วง (ดาวเทียมเงียบไปวันหนึ่ง) ต้องแค่ไม่มีจุดของวันนั้น
        ไม่ใช่ทำให้ทั้งคำขอพัง"""
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 1440, [0] * 1440)
        _write_day(tmp_path, datetime(2014, 10, 20), [3e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18), datetime(2014, 10, 20, 23, 59))

        assert len(series.times) == 1440 * 2  # วันที่ 19 หายไปทั้งวัน ไม่ใช่แค่ NaN
        assert all(t.day != 19 for t in series.times)

    def test_picks_the_newest_version_when_multiple_exist(self, tmp_path):
        """NOAA reprocess ไฟล์บางวันใหม่แล้วเผยแพร่ version สูงกว่า — ต้องใช้ตัวล่าสุด"""
        year_dir = tmp_path / "2014"
        year_dir.mkdir(parents=True)
        with h5py.File(year_dir / "sci_xrsf-l2-avg1m_g15_d20141018_v2-1-0.nc", "w") as handle:
            handle.create_dataset("time", data=np.array([0.0]))
            handle.create_dataset("xrsb_flux", data=np.array([1e-6], dtype="float32"))
            handle.create_dataset("xrsb_flag", data=np.array([0], dtype="uint16"))
        t0 = (datetime(2014, 10, 18) - _EPOCH).total_seconds()
        with h5py.File(year_dir / "sci_xrsf-l2-avg1m_g15_d20141018_v2-2-1.nc", "w") as handle:
            handle.create_dataset("time", data=np.array([t0]))
            handle.create_dataset("xrsb_flux", data=np.array([9e-6], dtype="float32"))
            handle.create_dataset("xrsb_flag", data=np.array([0], dtype="uint16"))

        store = XrayFluxStore(tmp_path)
        series = store.series(datetime(2014, 10, 18), datetime(2014, 10, 18, 0, 1))

        assert series.flux == pytest.approx([9e-6], rel=1e-6)


class TestDecimation:
    def test_short_series_passes_through_unchanged(self, tmp_path):
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 100, [0] * 100)
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18), datetime(2014, 10, 18, 1, 39))

        assert not series.decimated
        assert len(series.times) == 100

    def test_decimation_preserves_the_peak_inside_each_bucket(self):
        n = 100
        times = np.array(
            [np.datetime64("2014-10-18T00:00") + np.timedelta64(i, "m") for i in range(n)]
        )
        flux = np.zeros(n)
        flux[47] = 5.0  # พีคเดี่ยวซ่อนอยู่กลาง ๆ ช่วงที่ 5 (bucket ที่ 5 จาก max_points=10)

        out_times, out_flux = _decimate_keep_peaks(times, flux, max_points=10)

        assert len(out_times) <= 10
        assert 5.0 in out_flux  # ค่าเฉลี่ยจะทำให้พีคนี้หายไป แต่ max ต้องเก็บไว้ได้

    def test_all_nan_bucket_stays_nan(self):
        times = np.array(
            [np.datetime64("2014-10-18T00:00") + np.timedelta64(i, "m") for i in range(10)]
        )
        flux = np.full(10, np.nan)

        out_times, out_flux = _decimate_keep_peaks(times, flux, max_points=2)

        assert np.all(np.isnan(out_flux))
