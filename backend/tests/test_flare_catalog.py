"""ตรวจสอบการแปลงคลาส GOES และการอ่านไฟล์รายงาน flare ของ NOAA

บรรทัดตัวอย่างในไฟล์นี้คัดลอกมาจาก ``goes-xrs-report_2014.txt`` ของจริง และค่าที่
คาดหวังตรวจสอบได้กับเหตุการณ์ที่บันทึกไว้ในประวัติศาสตร์ เช่น flare M9.9 เมื่อวันที่
1 ม.ค. 2014 จาก AR 11936 และ X1.2 เมื่อวันที่ 7 ม.ค. 2014
"""

from __future__ import annotations

import pytest

from sunseg.data.goes_class import (
    flux_to_goes_class,
    goes_class_to_flux,
    is_at_least,
    safe_goes_class_to_flux,
)
from sunseg.data.noaa_flares import (
    _parse_tail,
    candidate_filenames,
    parse_location,
    parse_report_file,
)

# บรรทัดจริงจาก goes-xrs-report_2014.txt
SAMPLE_REPORT = """\
31777140101  0645 0652 0649                                C 21    G15  5.8E-04
31777140101  0721 0729 0726 S12W47                         C 32    G15  1.0E-03 11940 131228.7
31777140101  1840 1903 1852 S14W47                         M 99    G15  8.2E-02 11936 131229.2
31777140107  1804 1858 1832                                X 12    G15  2.5E-01
31777140108  0339 0354 0347 N11W81                         M 36    G15  1.7E-02 11947 140102.0
"""


class TestGoesClassToFlux:
    @pytest.mark.parametrize(
        ("goes_class", "expected"),
        [
            ("A1.0", 1e-8),
            ("B1.0", 1e-7),
            ("C1.0", 1e-6),
            ("M1.0", 1e-5),
            ("X1.0", 1e-4),
            ("M9.9", 9.9e-5),
            ("C3.2", 3.2e-6),
            ("X28.0", 2.8e-3),  # flare ที่แรงที่สุดเท่าที่เคยวัดได้ — คลาส X ไม่มีเพดาน
        ],
    )
    def test_known_values(self, goes_class, expected):
        assert goes_class_to_flux(goes_class) == pytest.approx(expected)

    def test_bare_letter_defaults_to_magnitude_one(self):
        assert goes_class_to_flux("M") == pytest.approx(1e-5)

    def test_case_insensitive(self):
        assert goes_class_to_flux("m1.5") == pytest.approx(goes_class_to_flux("M1.5"))

    def test_ordering_across_class_boundary(self):
        """C9.9 ต้องน้อยกว่า M1.0 — ขอบเขตที่การเทียบสตริงแบบตรงๆ จะพลาด"""
        assert goes_class_to_flux("C9.9") < goes_class_to_flux("M1.0")

    @pytest.mark.parametrize("bad", ["", "Z1.0", "1.0", "M-1", "abc"])
    def test_rejects_invalid_input(self, bad):
        with pytest.raises(ValueError):
            goes_class_to_flux(bad)

    def test_safe_variant_returns_nan_instead_of_raising(self):
        import math

        assert math.isnan(safe_goes_class_to_flux("ไม่ใช่คลาส"))
        assert math.isnan(safe_goes_class_to_flux(None))


class TestFluxToGoesClass:
    @pytest.mark.parametrize(
        ("flux", "expected"),
        [(1e-5, "M1.0"), (1.5e-5, "M1.5"), (1e-4, "X1.0"), (3.2e-6, "C3.2")],
    )
    def test_known_values(self, flux, expected):
        assert flux_to_goes_class(flux) == expected

    def test_round_trip(self):
        for goes_class in ("B5.0", "C3.2", "M1.0", "M9.9", "X2.5"):
            assert flux_to_goes_class(goes_class_to_flux(goes_class)) == goes_class

    def test_below_a_class_and_invalid_values(self):
        assert flux_to_goes_class(1e-12) == "A0.0"
        assert flux_to_goes_class(0.0) == "A0.0"
        assert flux_to_goes_class(float("nan")) == "A0.0"


class TestIsAtLeast:
    def test_threshold_is_inclusive(self):
        """เกณฑ์ต้องรวมค่าที่เท่ากันพอดี — M1.0 ต้องนับเป็น 'ถึงเกณฑ์ M1.0'"""
        assert is_at_least("M1.0", "M1.0")

    def test_comparisons(self):
        assert is_at_least("X1.0", "M1.0")
        assert not is_at_least("C9.9", "M1.0")


class TestParseTail:
    def test_parses_full_line_with_active_region(self):
        result = _parse_tail("       C 32    G15  1.0E-03 11940 131228.7")

        assert result["goes_class"] == "C3.2"
        assert result["satellite"] == "G15"
        assert result["integrated_flux"] == pytest.approx(1.0e-3)
        assert result["noaa_ar"] == 11940

    def test_parses_line_without_active_region(self):
        """flare จำนวนมากในไฟล์ NOAA ไม่มีเลข AR — ต้อง parse ได้และคืน None"""
        result = _parse_tail("       C 21    G15  5.8E-04")

        assert result["goes_class"] == "C2.1"
        assert result["noaa_ar"] is None

    def test_handles_digits_attached_to_letter(self):
        result = _parse_tail("       M12    G15  4.3E-03 11944")
        assert result["goes_class"] == "M1.2"

    def test_returns_none_for_unparseable_tail(self):
        assert _parse_tail("") is None
        assert _parse_tail("   ขยะ   ") is None


class TestParseReportFile:
    @pytest.fixture
    def report(self, tmp_path):
        path = tmp_path / "goes-xrs-report_2014.txt"
        path.write_text(SAMPLE_REPORT, encoding="utf-8")
        return parse_report_file(path)

    def test_reads_every_line(self, report):
        assert len(report) == 5

    def test_parses_the_m99_flare_from_ar_11936(self, report):
        """เหตุการณ์จริง: M9.9 วันที่ 1 ม.ค. 2014 peak 18:52 UT จาก AR 11936

        บรรทัดดิบคือ ``1840 1903 1852`` — ยืนยันว่าลำดับคือ เริ่ม/จบ/peak
        เพราะเวลา peak ที่บันทึกในประวัติศาสตร์ของ flare ดวงนี้คือ 18:52 UT
        """
        row = report[report["goes_class"] == "M9.9"].iloc[0]

        assert row["start_time"].strftime("%H:%M") == "18:40"
        assert row["peak_time"].strftime("%Y-%m-%d %H:%M") == "2014-01-01 18:52"
        assert row["end_time"].strftime("%H:%M") == "19:03"
        assert row["noaa_ar"] == 11936
        assert row["peak_flux"] == pytest.approx(9.9e-5)
        assert row["location"] == "S14W47"

    def test_parses_x12_peak_time(self, report):
        """X1.2 วันที่ 7 ม.ค. 2014 peak 18:32 UT (บรรทัดดิบ ``1804 1858 1832``)"""
        row = report[report["goes_class"] == "X1.2"].iloc[0]

        assert row["start_time"].strftime("%H:%M") == "18:04"
        assert row["peak_time"].strftime("%H:%M") == "18:32"
        assert row["end_time"].strftime("%H:%M") == "18:58"

    def test_parses_the_x_class_flare_without_active_region(self, report):
        """X1.2 วันที่ 7 ม.ค. 2014 — ไฟล์ต้นทางไม่ได้ระบุ AR ไว้"""
        row = report[report["goes_class"].str.startswith("X")].iloc[0]

        assert row["goes_class"] == "X1.2"
        assert row["peak_time"].strftime("%Y-%m-%d") == "2014-01-07"
        assert pd_isna(row["noaa_ar"])

    def test_two_digit_year_maps_to_2014(self, report):
        assert (report["peak_time"].dt.year == 2014).all()

    def test_times_are_ordered_within_each_event(self, report):
        valid = report[report["start_time"].notna() & report["end_time"].notna()]
        assert (valid["start_time"] <= valid["peak_time"]).all()
        assert (valid["peak_time"] <= valid["end_time"]).all()

    def test_flux_matches_class(self, report):
        for _, row in report.iterrows():
            assert row["peak_flux"] == pytest.approx(goes_class_to_flux(row["goes_class"]))

    def test_empty_file_returns_empty_frame(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("", encoding="utf-8")
        assert parse_report_file(path).empty


class TestCandidateFilenames:
    def test_2015_prefers_the_corrected_file(self):
        """ปี 2015 มีเวอร์ชันที่เติมแถวที่หายไปแล้ว ต้องถูกลองก่อน"""
        assert "modifiedreplacedmissingrows" in candidate_filenames(2015)[0]

    def test_2017_uses_year_to_date_file(self):
        assert candidate_filenames(2017)[0] == "goes-xrs-report_2017-ytd.txt"

    def test_ordinary_year_uses_standard_name(self):
        assert candidate_filenames(2014) == ["goes-xrs-report_2014.txt"]


class TestParseLocation:
    """ตำแหน่ง flare แบบ N11W82 — ใช้วาดจุดลงแผนที่จานสุริยะบนหน้าเว็บ"""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("N11W82", (11.0, 82.0)),    # เหนือ+ตะวันตก = บวกทั้งคู่
            ("S26E34", (-26.0, -34.0)),  # ใต้+ตะวันออก = ลบทั้งคู่
            ("N00W00", (0.0, 0.0)),      # กลางจานพอดี
            ("s12w47", (-12.0, 47.0)),   # ไฟล์บางปีใช้ตัวพิมพ์เล็ก
        ],
    )
    def test_parses_heliographic_coordinates(self, value, expected):
        assert parse_location(value) == expected

    @pytest.mark.parametrize("value", [None, "", "   ", "BAD", "N11", "W82", "1182"])
    def test_returns_none_when_unusable(self, value):
        """flare ราว 1 ใน 3 ไม่มีการยืนยันตำแหน่ง — ต้องคืน None ไม่ใช่โยน exception"""
        assert parse_location(value) is None

    @pytest.mark.parametrize("value", ["N99W99", "S05W120"])
    def test_rejects_coordinates_beyond_the_disk(self, value):
        """เกิน ±90° คือค่าที่อ่านผิด ไม่ใช่ flare ด้านหลังดวงอาทิตย์"""
        assert parse_location(value) is None


def pd_isna(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))
