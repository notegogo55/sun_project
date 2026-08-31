"""ทดสอบตัวอ่านฟลักซ์โปรตอนราย 5 นาที

จุดสำคัญของชุดนี้คือ **การคำนวณ offset ของแถว** ซึ่งผิดแบบเงียบ ๆ ได้ง่ายที่สุด:
ถ้าคำนวณพลาดไปหนึ่งแถว กราฟจะยังวาดออกมาสวยงามและดูสมเหตุสมผลทุกประการ แค่เป็น
ข้อมูลของเวลาอื่น — จึงสร้างคลังจำลองที่รู้คำตอบล่วงหน้าแล้วตรวจค่าตรงจุดที่ระบุ

เขียนไฟล์จำลองเองแทนที่จะพึ่งคลังจริง เพราะคลังจริงอยู่นอก repo และไม่มีในเครื่อง
ที่รัน CI
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from sunseg.data.proton_flux import (
    CHANNEL_KEYS,
    ROW_BYTES,
    STEP,
    ProtonFluxStore,
    s_scale_level,
)

HEADER = """:Data_list: {name}
# ================================================================
# Satellite: {source}
# Coverage: {start} .. {end}  ({rows} rows)
#
# YR MO DA  HHMM    Day     Day         P > 1         P > 5         P >10         P >30         P >50         P>100         E>0.8         E>2.0         E>4.0    QC
#-------------------------------------------------------------------------------
"""


def _row(when: datetime, p1: str, p10: str, p50: str, p100: str) -> str:
    """หนึ่งแถวที่กว้างเท่ากับของจริงเป๊ะ ๆ รวม newline = ROW_BYTES ไบต์"""
    stamp = f"{when:%Y %m %d  %H%M}"
    columns = "".join(
        f"{value:>11s}" for value in (p1, "1.00e+00", p10, "1.00e+00", p50, p100)
    )
    line = f"{stamp}   55927       0{columns}   1.00e+02   1.00e+00         NA     OBS"
    assert len(line) + 1 == ROW_BYTES, f"แถวยาว {len(line) + 1} ไบต์ ไม่ใช่ {ROW_BYTES}"
    return line + "\n"


def _write(root, source: str, start: datetime, rows: list[tuple[str, str, str, str]]) -> None:
    """เขียนไฟล์รายปีจำลองหนึ่งไฟล์ในรูปแบบเดียวกับคลังจริง"""
    year_dir = root / str(start.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    name = f"{source}_{start.year}_5m_clean.txt"

    body = "".join(
        _row(start + i * STEP, *values) for i, values in enumerate(rows)
    )
    header = HEADER.format(
        name=name,
        source=source,
        start=f"{start:%Y-%m-%dT%H:%M:%S}Z",
        end=f"{start + (len(rows) - 1) * STEP:%Y-%m-%dT%H:%M:%S}Z",
        rows=len(rows),
    )
    (year_dir / name).write_text(header + body, encoding="utf-8", newline="")


@pytest.fixture
def store(tmp_path):
    """คลังจำลอง 2 ชั่วโมง: ค่าของ p10 เท่ากับ "นาทีที่ i" เพื่อให้ตรวจ offset ได้ตรง ๆ"""
    start = datetime(2014, 6, 1, 0, 0)
    rows = [
        (f"{1000 + i}.0", f"{i + 1}.0", "0.5", "0.25") for i in range(24)
    ]
    _write(tmp_path, "Primary", start, rows)
    return ProtonFluxStore(tmp_path)


class TestIndexing:
    def test_scans_archive_and_reports_coverage(self, store):
        assert store.available
        assert store.sources == ["Primary"]
        assert store.coverage == (datetime(2014, 6, 1, 0, 0), datetime(2014, 6, 1, 1, 55))

    def test_missing_archive_degrades_instead_of_raising(self, tmp_path):
        """path ผิดต้องไม่ทำให้แอปล่มตอน startup — แค่ปิดแผงโปรตอนไป"""
        store = ProtonFluxStore(tmp_path / "ไม่มีอยู่จริง")
        assert not store.available
        assert store.coverage is None

    def test_window_outside_archive_is_empty_not_an_error(self, store):
        window = store.window(datetime(1999, 1, 1, 0, 0), 1, 1)
        assert not window.has_data
        assert window.sources == []


class TestRowOffsets:
    def test_reads_the_row_that_matches_the_timestamp(self, store):
        """แถวที่ i ต้องได้ค่าที่เขียนไว้ให้แถวที่ i ไม่ใช่แถวข้างเคียง"""
        center = datetime(2014, 6, 1, 1, 0)   # แถวที่ 12
        window = store.window(center, 1, 0)

        assert window.value_at(center, "p10") == 13.0
        assert window.value_at(center - timedelta(minutes=5), "p10") == 12.0
        assert window.value_at(center + timedelta(minutes=5), "p10") is None  # เลยหน้าต่าง

    def test_all_channels_land_on_their_own_column(self, store):
        center = datetime(2014, 6, 1, 0, 0)
        window = store.window(center, 0, 0)

        assert [window.value_at(center, key) for key in CHANNEL_KEYS] == [1000.0, 1.0, 0.5, 0.25]

    def test_center_off_the_five_minute_grid_still_aligns(self, store):
        """เวลาที่ flare พีคเป็นนาทีอะไรก็ได้ กริดต้องถูกปัดลงให้ตรงแถวเสมอ"""
        window = store.window(datetime(2014, 6, 1, 0, 47), 0, 0)

        assert window.times[0] == datetime(2014, 6, 1, 0, 45)
        assert window.values["p10"][0] == 10.0


class TestWindow:
    def test_window_spans_the_requested_hours(self, store):
        window = store.window(datetime(2014, 6, 1, 1, 0), 1, 0)

        assert window.times[0] == datetime(2014, 6, 1, 0, 0)
        assert window.times[-1] == datetime(2014, 6, 1, 1, 0)
        assert len(window.times) == 13

    def test_gaps_stay_none_rather_than_zero(self, tmp_path):
        """0 กับ "ไม่มีข้อมูล" ต้องไม่ปนกัน — บนแกน log ศูนย์วาดไม่ได้ และการลากเส้น
        ข้ามช่วงที่ดาวเทียมไม่ได้วัดคือการแต่งข้อมูลขึ้นมาเอง"""
        start = datetime(2014, 6, 1, 0, 0)
        _write(tmp_path, "Primary", start, [
            ("1.0", "2.0", "3.0", "4.0"),
            ("NaN", "NaN", "NaN", "NaN"),
            ("1.0", "NA", "3.0", "4.0"),
            ("1.0", "0.00e+00", "3.0", "4.0"),   # ฟลักซ์ <= 0 ไม่มีความหมายทางกายภาพ
        ])
        window = ProtonFluxStore(tmp_path).window(start, 0, 1)

        assert window.values["p10"][:4] == [2.0, None, None, None]
        # NA ของช่องหนึ่งต้องไม่ลบค่าที่ช่องอื่นในแถวเดียวกันวัดได้จริง
        assert window.values["p1"][:4] == [1.0, None, 1.0, 1.0]
        # ช่วงที่เลยท้ายไฟล์ไปแล้วก็ว่างเหมือนกัน ไม่ใช่ค่าสุดท้ายค้างไว้
        assert window.values["p10"][4:] == [None] * (len(window.times) - 4)

    def test_lower_priority_source_only_fills_what_is_missing(self, tmp_path):
        """Primary ชนะเสมอ ดาวเทียมสำรองเข้ามาเติมเฉพาะช่วงที่ Primary ไม่มี"""
        start = datetime(2014, 6, 1, 0, 0)
        _write(tmp_path, "Primary", start, [("1.0", "10.0", "1.0", "1.0"),
                                            ("1.0", "NaN", "1.0", "1.0")])
        _write(tmp_path, "Secondary", start, [("1.0", "99.0", "1.0", "1.0"),
                                              ("1.0", "77.0", "1.0", "1.0")])

        window = ProtonFluxStore(tmp_path).window(start, 0, 1)

        assert window.values["p10"][:2] == [10.0, 77.0]
        assert window.sources == ["Primary", "Secondary"]

    def test_window_crosses_a_year_boundary(self, tmp_path):
        """หน้าต่าง 72 ชม.รอบ flare ปลายเดือนธันวาคมต้องต่อไฟล์สองปีเข้าด้วยกัน"""
        _write(tmp_path, "Primary", datetime(2013, 12, 31, 23, 50),
               [("1.0", "5.0", "1.0", "1.0")] * 2)
        _write(tmp_path, "Primary", datetime(2014, 1, 1, 0, 0),
               [("1.0", "6.0", "1.0", "1.0")] * 2)

        window = ProtonFluxStore(tmp_path).window(datetime(2014, 1, 1, 0, 0), 1, 1)

        # กริดเริ่ม 23:00 -> ดัชนี 10..13 คือ 23:50, 23:55, 00:00, 00:05
        assert window.times[10] == datetime(2013, 12, 31, 23, 50)
        assert window.values["p10"][10:14] == [5.0, 5.0, 6.0, 6.0]


class TestSummaryValues:
    def test_peak_after_ignores_everything_before_the_flare(self, store):
        """อนุภาคมาถึงโลกหลัง flare เสมอ — ยอดก่อนหน้านั้นเป็นของเหตุการณ์อื่น"""
        center = datetime(2014, 6, 1, 1, 0)
        window = store.window(center, 1, 0)

        peak = window.peak_after(center, "p10")
        assert peak == (13.0, center)   # หน้าต่างจบที่ center จึงเหลือแค่จุดเดียว

        full = store.window(datetime(2014, 6, 1, 0, 0), 0, 2)
        assert full.peak_after(datetime(2014, 6, 1, 0, 0), "p10")[0] == 24.0

    @pytest.mark.parametrize(
        ("pfu", "expected"),
        [(None, None), (9.9, None), (10.0, "S1"), (136.1, "S2"),
         (6530.0, "S3"), (1e5, "S5"), (1e9, "S5")],
    )
    def test_s_scale_thresholds(self, pfu, expected):
        assert s_scale_level(pfu) == expected
