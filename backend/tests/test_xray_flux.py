"""ทดสอบตัวอ่านฟลักซ์ X-ray ต่อเนื่องรายนาที

สร้างไฟล์ HDF5 จำลองที่มีโครงสร้างตัวแปรเดียวกับไฟล์จริงของ NOAA (``time``,
``xrsb_flux``, ``xrsb_flag``) แทนที่จะพึ่งไฟล์จริงที่ต้องดาวน์โหลด — ทำให้ทดสอบได้
โดยไม่ต้องมีอินเทอร์เน็ตหรือรัน ``download_xray.py`` ก่อน
"""

from __future__ import annotations

from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import pytest

from sunseg.data.xray_flux import SATELLITE_SCALE, XrayFluxStore, _decimate_keep_peaks, _read_day

_EPOCH = datetime(2000, 1, 1, 12, 0, 0)


def _write_day(
    root, day: datetime, flux: list[float], flag: list[int], satellite: str = "g15"
) -> None:
    """เขียนไฟล์รายวันจำลอง 1 ไฟล์ในรูปแบบเดียวกับ ``sci_xrsf-l2-avg1m_{sat}_dYYYYMMDD_v*.nc``

    GOES 1-15 กับ GOES-R series ใช้ตัวแปรชื่อเดียวกันทุกประการ (ดู docstring ของ
    ``sunseg.data.xray_flux``) ไฟล์จำลองนี้จึงใช้แทนได้ทั้งสองรุ่น ต่างกันแค่ชื่อไฟล์
    """
    n = len(flux)
    assert len(flag) == n
    seconds = [
        (day.replace(hour=0, minute=0, second=0) - _EPOCH).total_seconds() + i * 60
        for i in range(n)
    ]

    year_dir = root / str(day.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    path = year_dir / f"sci_xrsf-l2-avg1m_{satellite}_d{day:%Y%m%d}_v2-2-1.nc"
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


class TestMultiSatellite:
    """g15 (2011-2017) กับ g16 (case study พ.ค. 2024 เป็นต้นไป) คาลิเบรตกันคนละวิธี —
    ดู docstring ของ ``sunseg.data.xray_flux`` สำหรับที่มาของตัวเลข ``SATELLITE_SCALE``"""

    def test_g16_flux_is_scaled_to_g15_reference(self, tmp_path):
        """ค่าดิบในไฟล์ g16 ต้องถูกคูณด้วย SATELLITE_SCALE['g16'] ก่อนคืนออกไป"""
        _write_day(tmp_path, datetime(2024, 5, 10), [1e-6, 2e-6], [0, 0], satellite="g16")
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2024, 5, 10, 0, 0), datetime(2024, 5, 10, 0, 1))

        scale = SATELLITE_SCALE["g16"]
        assert series.flux == pytest.approx([1e-6 * scale, 2e-6 * scale], rel=1e-6)

    def test_g15_flux_is_not_scaled(self, tmp_path):
        """g15 คือดาวเทียมอ้างอิง (scale = 1.0) — ต้องได้ค่าดิบกลับมาตรงๆ"""
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6], [0], satellite="g15")
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2014, 10, 18, 0, 0), datetime(2014, 10, 18, 0, 0))

        assert series.flux == pytest.approx([1e-6], rel=1e-6)

    def test_spans_the_transition_between_satellites(self, tmp_path):
        """ช่วงเวลาที่คาบเกี่ยวสองดวง (g15 หมดข้อมูล, g16 เริ่ม) ต้องต่อกันได้ไร้รอยต่อ
        โดยแต่ละฝั่งถูกปรับสเกลตามดวงของตัวเอง ไม่ใช่ตามดวงเดียวทั้งเส้น"""
        _write_day(tmp_path, datetime(2020, 3, 4), [1e-6] * 3, [0] * 3, satellite="g15")
        _write_day(tmp_path, datetime(2024, 5, 1), [1e-6] * 3, [0] * 3, satellite="g16")
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2020, 3, 4), datetime(2024, 5, 1, 0, 2))

        scale = SATELLITE_SCALE["g16"]
        assert series.flux[:3] == pytest.approx([1e-6] * 3, rel=1e-6)
        assert series.flux[-3:] == pytest.approx([1e-6 * scale] * 3, rel=1e-6)

    def test_g16_flag_semantics_match_g15(self, tmp_path):
        """g16 ใช้คำอธิบาย flag ต่างจาก g15 (eclipse/bad_data/interpolated แทน
        bad_data/eclipsed_by_earth/temperature_recovery) แต่ 0 = ดี เหมือนกันทั้งคู่ —
        flag ที่ไม่ใช่ 0 ต้องกลายเป็น None เหมือนที่ g15 ทำ"""
        _write_day(tmp_path, datetime(2024, 5, 10), [1e-6, 1e-6, 1e-6], [0, 1, 4], satellite="g16")
        store = XrayFluxStore(tmp_path)

        series = store.series(datetime(2024, 5, 10, 0, 0), datetime(2024, 5, 10, 0, 2))

        assert series.flux[0] is not None
        assert series.flux[1:] == [None, None]

    def test_unknown_satellite_defaults_to_unscaled(self, tmp_path):
        """ดาวเทียมที่ยังไม่ได้วัดตัวคูณสเกล (ไม่อยู่ใน SATELLITE_SCALE) ต้องไม่ถูกคูณ
        อะไรเลย แทนที่จะพังหรือใส่ค่าเดามา — ทดสอบ ``_read_day`` ตรงๆ เพราะ SOURCE_SATELLITES
        ปัจจุบันมีแค่ g15/g16 ซึ่งทั้งคู่มีตัวคูณสเกลอยู่แล้ว จึงจำลองผ่าน XrayFluxStore
        (public API) ไม่ได้ — เผื่ออนาคตเพิ่มดาวเทียมใหม่ใน SOURCE_SATELLITES ก่อนวัดสเกล"""
        _write_day(tmp_path, datetime(2015, 1, 1), [1e-6], [0], satellite="g14")
        path = tmp_path / "2015" / "sci_xrsf-l2-avg1m_g14_d20150101_v2-2-1.nc"

        _times, flux = _read_day(path, satellite="g14")

        assert flux == pytest.approx([1e-6], rel=1e-6)


class TestBinSeries:
    """ยุบฟลักซ์รายนาทีลงกริดสำหรับ sequence dataset (ticket 04) — ต่างจาก series()
    ที่ยุบด้วย _decimate_keep_peaks สำหรับวาดกราฟ"""

    def test_hand_computed_window_matches(self, tmp_path):
        """หน้าต่างเดียว ค่าที่ได้ต้องตรงกับสถิติที่คำนวณมือ

        วางจุดทั้งหมดไว้ **ในช่วงเวลา** ของ bin เดียว (ไม่แตะขอบเขตพอดี) เพื่อไม่ให้ปน
        กับกรณีทดสอบขอบเขต (ดู test_bin_window_is_causal_past_only)
        """
        day = datetime(2014, 10, 18)
        flux = [1e-6, 5e-6, 3e-6, 2e-6]
        offsets_min = [60, 120, 180, 240]  # 01:00, 02:00, 03:00, 04:00 — ห่างจากขอบ bin ชัดเจน
        seconds = [(day - _EPOCH).total_seconds() + m * 60 for m in offsets_min]
        year_dir = tmp_path / "2014"
        year_dir.mkdir(parents=True)
        with h5py.File(year_dir / "sci_xrsf-l2-avg1m_g15_d20141018_v2-2-1.nc", "w") as handle:
            handle.create_dataset("time", data=np.array(seconds, dtype="float64"))
            handle.create_dataset("xrsb_flux", data=np.array(flux, dtype="float32"))
            handle.create_dataset("xrsb_flag", data=np.array([0] * 4, dtype="uint16"))

        store = XrayFluxStore(tmp_path)
        df = store.bin_series(datetime(2014, 10, 19), datetime(2014, 10, 19), cadence_hours=24)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["xray_median"] == pytest.approx(float(np.median(flux)), rel=1e-6)
        assert row["xray_max"] == pytest.approx(max(flux), rel=1e-6)
        assert row["xray_min"] == pytest.approx(min(flux), rel=1e-6)
        assert row["xray_log10_median"] == pytest.approx(np.log10(np.median(flux)), rel=1e-6)

    def test_bin_window_is_causal_past_only(self, tmp_path):
        """bin ที่ timestamp t ครอบ (t-cadence, t] — ค่าที่เกิด**หลัง**ขอบบนต้องตกไปอยู่
        bin ถัดไป ไม่ใช่ bin นี้ (จุดออกพยากรณ์ต้องเห็นแต่อดีต ไม่เห็นอนาคต)"""
        # 12:00:00 พอดี = ขอบบนของ bin แรก (00:00, 12:00] -> อยู่ใน bin แรก
        # 12:01:00 = อยู่ bin ถัดไป (12:00, 24:00]
        day = datetime(2014, 10, 18)
        seconds = [
            (day - _EPOCH).total_seconds() + 12 * 3600,  # 12:00:00 -> bin แรก
            (day - _EPOCH).total_seconds() + 12 * 3600 + 60,  # 12:01:00 -> bin สอง
        ]
        year_dir = tmp_path / "2014"
        year_dir.mkdir(parents=True)
        with h5py.File(year_dir / "sci_xrsf-l2-avg1m_g15_d20141018_v2-2-1.nc", "w") as handle:
            handle.create_dataset("time", data=np.array(seconds, dtype="float64"))
            handle.create_dataset("xrsb_flux", data=np.array([1.0, 9.0], dtype="float32"))
            handle.create_dataset("xrsb_flag", data=np.array([0, 0], dtype="uint16"))

        store = XrayFluxStore(tmp_path)
        df = store.bin_series(datetime(2014, 10, 18, 12), datetime(2014, 10, 19), cadence_hours=12)

        first_bin = df[df["t_rec"] == datetime(2014, 10, 18, 12)].iloc[0]
        second_bin = df[df["t_rec"] == datetime(2014, 10, 19, 0)].iloc[0]
        assert first_bin["xray_median"] == pytest.approx(1.0, rel=1e-6)
        assert second_bin["xray_median"] == pytest.approx(9.0, rel=1e-6)

    def test_bad_flagged_points_excluded_from_bin_stats(self, tmp_path):
        """จุดที่ flag ไม่ดีต้องไม่ถูกนับเข้าสถิติของ bin เลย ไม่ใช่นับเป็น 0"""
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6, 100.0, 3e-6], [0, 1, 0])
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 19), datetime(2014, 10, 19), cadence_hours=24)

        assert df.iloc[0]["xray_max"] == pytest.approx(3e-6, rel=1e-6)  # ไม่ใช่ 100.0

    def test_missing_day_gives_nan_not_zero(self, tmp_path):
        """ไม่มีไฟล์เลยในช่วงของ bin — ต้องได้ NaN ทั้ง 4 คอลัมน์ ไม่ใช่ 0"""
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 18), datetime(2014, 10, 18), cadence_hours=24)

        row = df.iloc[0]
        assert np.isnan(row["xray_median"])
        assert np.isnan(row["xray_max"])
        assert np.isnan(row["xray_min"])
        assert np.isnan(row["xray_log10_median"])
        assert np.isnan(row["xray_log10_rel27d"])

    def test_relative_level_is_zero_when_flux_is_constant(self, tmp_path):
        for day in (17, 18, 19):
            _write_day(tmp_path, datetime(2014, 10, day), [3e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 20), datetime(2014, 10, 20), cadence_hours=24)

        assert df.iloc[0]["xray_log10_rel27d"] == pytest.approx(0.0, abs=1e-6)

    def test_relative_level_hand_computed(self, tmp_path):
        """bin (19, 20] มีแต่ 4e-6 · ย้อนหลัง 27 วันมี 1e-6 และ 4e-6 อย่างละ 1440 จุด
        (จุด 19 00:00 ของไฟล์วันที่ 19 ตกอยู่ bin ก่อนหน้า แต่ยังอยู่ในช่วงย้อนหลัง)
        -> median ย้อนหลัง = (1e-6 + 4e-6) / 2 = 2.5e-6 -> ค่าสัมพัทธ์ = log10(4 / 2.5)"""
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6] * 1440, [0] * 1440)
        _write_day(tmp_path, datetime(2014, 10, 19), [4e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 20), datetime(2014, 10, 20), cadence_hours=24)

        assert df.iloc[0]["xray_log10_rel27d"] == pytest.approx(np.log10(4 / 2.5), rel=1e-5)

    def test_relative_level_is_nan_exactly_when_the_bin_is_empty(self, tmp_path):
        """ช่วงย้อนหลังของ bin ที่ว่างยังมีข้อมูล (วันที่ 18) แต่ค่าสัมพัทธ์ต้องเป็น NaN ตาม
        median ของ bin — ไม่งั้น dataset จะเก็บแถวต่างจากเดิมเมื่อเพิ่มคอลัมน์นี้เข้าไป"""
        _write_day(tmp_path, datetime(2014, 10, 18), [4e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 19), datetime(2014, 10, 20), cadence_hours=24)

        assert np.isfinite(df.iloc[0]["xray_log10_rel27d"])
        assert np.isnan(df.iloc[1]["xray_log10_rel27d"])

    def test_partially_missing_range_marks_only_the_empty_bins(self, tmp_path):
        """bin ที่มีข้อมูลจริงต้องได้ค่าจริง bin ที่ไม่มีข้อมูลต้องได้ NaN — ไม่ปนกัน

        เขียนข้อมูลไว้เฉพาะวันที่ 18 (00:00-23:59) แล้วขอ bin ของวันที่ 19 (หน้าต่าง
        (18, 19] — มีข้อมูลเกือบทั้งวัน) กับวันที่ 20 (หน้าต่าง (19, 20] — ไม่มีข้อมูลเลย
        เพราะห่างจากวันที่ 18 ไปสองวันเต็ม) เพื่อเลี่ยงจุดที่ตกขอบ bin พอดี (นาทีแรกของ
        วันที่ 18 คือ 00:00:00 ซึ่งเป็นขอบบนของ bin วันที่ 18 เอง ไม่ใช่ขอบล่างของ
        bin วันที่ 19 — ดู test_hand_computed_window_matches)
        """
        _write_day(tmp_path, datetime(2014, 10, 18), [4e-6] * 1440, [0] * 1440)
        store = XrayFluxStore(tmp_path)

        df = store.bin_series(datetime(2014, 10, 19), datetime(2014, 10, 20), cadence_hours=24)

        assert df.iloc[0]["xray_median"] == pytest.approx(4e-6, rel=1e-6)
        assert np.isnan(df.iloc[1]["xray_median"])

    def test_same_value_for_every_harp_at_the_same_time(self, tmp_path):
        """คุณสมบัติที่ตั้งใจให้เป็น: X-ray เป็นค่าทั้งดวง ไม่ใช่รายดวง — เรียก bin_series
        ซ้ำสองครั้งด้วยช่วงเวลาเดียวกัน (แทนการขอค่าจาก HARP คนละดวง) ต้องได้ค่าเดียวกันเป๊ะ
        เพราะเป็นฟังก์ชันบริสุทธิ์ของเวลา ไม่รับ HARPNUM เป็น input เลย"""
        _write_day(tmp_path, datetime(2014, 10, 18), [1e-6, 5e-6, 3e-6], [0, 0, 0])
        store = XrayFluxStore(tmp_path)

        df_for_harp_a = store.bin_series(datetime(2014, 10, 18), datetime(2014, 10, 18), cadence_hours=24)
        df_for_harp_b = store.bin_series(datetime(2014, 10, 18), datetime(2014, 10, 18), cadence_hours=24)

        pd.testing.assert_frame_equal(df_for_harp_a, df_for_harp_b)


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
