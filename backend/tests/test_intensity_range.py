"""ทดสอบฟังก์ชันบริสุทธิ์ของการสกัดความเข้มแสงช่วงเต็ม (ticket 05)

จุดที่ต้องพิสูจน์: การหา HARP ที่ควรเห็นแบบเร็วให้ผลเท่ากับวิธีกรองตรง ๆ ของด่านเดือนเดียว
· ledger ตัดสินว่าเฟรมไหนเสร็จจากสถานะล่าสุด (เฟรมที่รอ AIA อยู่ต้องถูกลองใหม่) · และตัววัด
drift จับได้ว่า normalisation ลบ degradation ที่คูณทั้งภาพออกไป
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sunseg.data.intensity_range import (
    STATUS_MISSING_AIA,
    STATUS_NO_IDENTITY,
    STATUS_OK,
    completed_stems,
    drift_by_year,
    drift_overlap,
    expected_harps_by_frame,
    iqr_overlap,
    match_rate_by_year,
    missed_pairs,
    overall_match_rate,
)


def _entry(stem, status=STATUS_OK, n_expected=0, n_matched=0, missed_unet=(), missed_no_identity=()):
    return {
        "stem": stem,
        "window": "main",
        "status": status,
        "n_expected": n_expected,
        "n_matched": n_matched,
        "n_extra": 0,
        "n_rows": n_matched,
        "missed_unet": list(missed_unet),
        "missed_no_identity": list(missed_no_identity),
        "missing_channels": "",
    }


class TestExpectedHarpsByFrame:
    def test_matches_brute_force_filter_including_both_edges(self):
        base = pd.Timestamp("2012-03-01 12:00")
        sharp = pd.DataFrame(
            {
                "HARPNUM": [1, 2, 3, 4, 5, 6],
                "t_rec": [
                    base - pd.Timedelta(minutes=30),  # ขอบล่างพอดี -> นับ
                    base + pd.Timedelta(minutes=30),  # ขอบบนพอดี -> นับ
                    base - pd.Timedelta(minutes=31),  # เลยขอบ -> ไม่นับ
                    base + pd.Timedelta(minutes=31),  # เลยขอบ -> ไม่นับ
                    base,
                    base + pd.Timedelta(hours=12),  # ของเฟรมถัดไป
                ],
            }
        )
        stems = ["20120301_120000", "20120302_000000"]

        fast = expected_harps_by_frame(sharp.sample(frac=1, random_state=0), stems)

        for stem in stems:
            moment = pd.Timestamp(pd.to_datetime(stem, format="%Y%m%d_%H%M%S"))
            brute = set(sharp.loc[(sharp["t_rec"] - moment).abs() <= pd.Timedelta(minutes=30), "HARPNUM"])
            assert fast[stem] == brute
        assert fast["20120301_120000"] == {1, 2, 5}
        assert fast["20120302_000000"] == {6}

    def test_same_harp_at_several_cadences_is_counted_once(self):
        base = pd.Timestamp("2012-03-01 00:00")
        sharp = pd.DataFrame(
            {"HARPNUM": [7, 7, 7], "t_rec": [base - pd.Timedelta(minutes=12), base, base + pd.Timedelta(minutes=12)]}
        )
        assert expected_harps_by_frame(sharp, ["20120301_000000"]) == {"20120301_000000": {7}}

    def test_empty_sharp_gives_empty_sets(self):
        empty = pd.DataFrame(columns=["HARPNUM", "t_rec"])
        assert expected_harps_by_frame(empty, ["20120301_000000"]) == {"20120301_000000": set()}


class TestLedger:
    def test_frame_waiting_for_aia_is_retried_until_a_later_run_marks_it_ok(self):
        first_run = pd.DataFrame([_entry("a", STATUS_MISSING_AIA), _entry("b")])
        assert completed_stems(first_run) == {"b"}

        second_run = pd.concat([first_run, pd.DataFrame([_entry("a")])], ignore_index=True)
        assert completed_stems(second_run) == {"a", "b"}

    def test_skipped_statuses_never_count_as_done(self):
        ledger = pd.DataFrame([_entry("a", STATUS_NO_IDENTITY), _entry("b", STATUS_MISSING_AIA)])
        assert completed_stems(ledger) == set()

    def test_empty_ledger_has_no_completed_frames(self):
        assert completed_stems(pd.DataFrame()) == set()


class TestMatchRate:
    def test_per_year_rates_and_reasons_are_hand_computable(self):
        ledger = pd.DataFrame(
            [
                _entry("20120101_000000", n_expected=10, n_matched=9, missed_unet=[5]),
                _entry("20120101_120000", n_expected=10, n_matched=8, missed_no_identity=[6, 7]),
                _entry("20130101_000000", n_expected=4, n_matched=4),
                _entry("20130101_120000", STATUS_NO_IDENTITY, n_expected=99),  # ไม่นับ
            ]
        )

        table = match_rate_by_year(ledger).set_index("year")

        assert table.loc[2012, "n_expected"] == 20
        assert table.loc[2012, "n_matched"] == 17
        assert table.loc[2012, "match_rate"] == pytest.approx(0.85)
        assert table.loc[2012, "n_missed_unet"] == 1
        assert table.loc[2012, "n_missed_no_identity"] == 2
        assert table.loc[2013, "match_rate"] == pytest.approx(1.0)
        assert overall_match_rate(ledger) == (21, 24)

    def test_missed_pairs_are_flattened_with_their_reason(self):
        ledger = pd.DataFrame([_entry("s1", missed_unet=[5, 8], missed_no_identity=[6])])
        pairs = missed_pairs(ledger)
        assert sorted(zip(pairs["HARPNUM"], pairs["reason"], strict=True)) == [
            (5, "unet_missed"), (6, "not_in_identity_map"), (8, "unet_missed"),
        ]


class TestDrift:
    @staticmethod
    def _table_with_degradation():
        """AR ทุกดวงสว่าง 3 เท่าของ quiet Sun เสมอ แต่กล้องเสื่อมจน DN/s ดิบลดลงปีละครึ่ง"""
        rng = np.random.default_rng(0)
        rows = []
        for year_index, year in enumerate(range(2011, 2017)):
            quiet = 100.0 * 0.5**year_index
            for _ in range(40):
                rows.append(
                    {
                        "issue_time": f"{year}0101_000000",
                        "171_median": 3.0 * rng.uniform(0.9, 1.1),
                        "171_quiet_sun": quiet,
                    }
                )
        return pd.DataFrame(rows)

    def test_normalisation_that_removes_a_multiplicative_decline_is_detected(self):
        result = drift_overlap(self._table_with_degradation(), "171")

        assert result["ok"]
        assert result["raw_overlap"] == pytest.approx(0.0)
        assert result["norm_overlap"] > 0.5
        assert result["improved"]
        assert result["early_years"] == [2011, 2012]
        assert result["late_years"] == [2015, 2016]

    def test_raw_values_are_reconstructed_from_normalised_times_quiet_sun(self):
        table = pd.DataFrame(
            {"issue_time": ["20110101_000000", "20170101_000000"], "171_median": [2.0, 2.0], "171_quiet_sun": [50.0, 10.0]}
        )
        by_year = drift_by_year(table, ["171"]).set_index("year")
        assert by_year.loc[2011, "171_raw"] == pytest.approx(100.0)
        assert by_year.loc[2017, "171_raw"] == pytest.approx(20.0)
        assert by_year.loc[2017, "171_normalised"] == pytest.approx(2.0)

    def test_single_year_cannot_be_judged(self):
        table = pd.DataFrame({"issue_time": ["20110101_000000"], "171_median": [2.0], "171_quiet_sun": [5.0]})
        assert drift_overlap(table, "171")["ok"] is False

    def test_iqr_overlap_bounds(self):
        a = pd.Series(np.arange(100.0))
        assert iqr_overlap(a, a) == pytest.approx(1.0)
        assert iqr_overlap(a, a + 1000) == pytest.approx(0.0)
