"""ตรวจสอบการแปลงตาราง HEK เป็น DataFrame

เทสต์ชุดนี้เกิดจากบั๊กจริง: sunpy รุ่นใหม่ (ทดสอบกับ 8.0.0) คืนคอลัมน์เวลาของ HEK
เป็น ``astropy.time.Time`` แทนสตริง ทำให้ ``pd.to_datetime(..., errors="coerce")``
เปลี่ยนทั้งคอลัมน์เป็น NaT เงียบๆ แล้วทุกแถวถูกกรองทิ้งตอนท้าย ระบบจึงรายงานว่า
"ไม่พบ flare" สำหรับ พ.ค. 2024 ทั้งที่ HEK ส่งมา 671 event

จึงยืนยันด้วย Time ของจริงจาก astropy ไม่ใช่ mock — บั๊กอยู่ที่ชนิดข้อมูลพอดี
การ mock จะทำให้เทสต์ผ่านโดยไม่จับบั๊ก
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.table import Table
from astropy.time import Time

from sunseg.data.hek_client import _hek_table_to_frame, _scalar


class TestScalar:
    def test_astropy_time_becomes_iso_string(self) -> None:
        result = _scalar(Time("2024-05-14 16:51:00"))
        # ต้องเช็คชนิดด้วย: astropy ยอมให้ Time == "สตริง" เป็น True ได้เอง
        # ถ้าเทียบแต่ค่า เทสต์จะเขียวแม้ _scalar ปล่อย Time ผ่านไปดื้อๆ
        assert isinstance(result, str)
        assert result == "2024-05-14 16:51:00.000"

    def test_masked_becomes_none(self) -> None:
        assert _scalar(np.ma.masked) is None

    def test_bytes_decoded(self) -> None:
        assert _scalar(b"X8.7") == "X8.7"

    def test_numpy_scalar_unwrapped(self) -> None:
        value = _scalar(np.int64(13664))
        assert value == 13664
        assert isinstance(value, int)

    def test_plain_string_untouched(self) -> None:
        assert _scalar("M1.3") == "M1.3"


def _table_with_time_columns() -> Table:
    """เลียนแบบสิ่งที่ HEKClient.search คืนมาจริงบน sunpy 8

    สองแถวแรกมี GOES class สมบูรณ์ ส่วนแถวที่สามเป็น event จาก Flare Detective
    ที่ไม่มีคลาส — ต้องถูกตัดทิ้งเพราะเอาไป label ไม่ได้
    """
    return Table(
        {
            "event_starttime": Time(
                ["2024-05-14 16:51:00", "2024-05-10 06:27:00", "2024-05-10 00:03:08"]
            ),
            "event_peaktime": Time(
                ["2024-05-14 17:02:00", "2024-05-10 06:54:00", "2024-05-10 00:04:00"]
            ),
            "event_endtime": Time(
                ["2024-05-14 17:12:00", "2024-05-10 07:15:00", "2024-05-10 00:05:00"]
            ),
            "fl_goescls": ["X8.7", "X3.9", ""],
            "ar_noaanum": [13664, 13664, 0],
            "frm_name": ["SWPC", "SWPC", "Flare Detective - Trigger Module"],
            "obs_observatory": ["GOES", "GOES", "SDO"],
        }
    )


class TestHekTableToFrame:
    def test_time_columns_survive_conversion(self) -> None:
        """บั๊กเดิมทำให้ผลลัพธ์ว่างเปล่าทั้งที่ตารางมีข้อมูล"""
        out = _hek_table_to_frame(_table_with_time_columns())

        assert len(out) == 2, "แถวที่มี GOES class ต้องรอด (แถวไม่มีคลาสถูกตัดถูกแล้ว)"
        assert out["peak_time"].notna().all(), "peak_time กลายเป็น NaT — บั๊ก astropy Time กลับมา"
        assert out["start_time"].notna().all()
        assert pd.api.types.is_datetime64_any_dtype(out["start_time"])

    def test_values_parsed_correctly(self) -> None:
        out = _hek_table_to_frame(_table_with_time_columns())
        row = out.iloc[0]

        assert row["goes_class"] == "X8.7"
        assert row["peak_flux"] == pytest.approx(8.7e-4)
        assert row["noaa_ar"] == 13664
        assert row["start_time"] == pd.Timestamp("2024-05-14 16:51:00")
        assert row["peak_time"] == pd.Timestamp("2024-05-14 17:02:00")

    def test_empty_table_returns_empty_frame(self) -> None:
        assert _hek_table_to_frame(Table()).empty
        assert _hek_table_to_frame(None).empty
