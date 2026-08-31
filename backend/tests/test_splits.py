"""ตรวจสอบการแบ่ง train/val/test ว่ากัน data leakage ได้จริง

นี่คือ test ที่สำคัญที่สุดชุดหนึ่งของโปรเจค — ถ้า HARP ดวงเดียวกันหลุดไปอยู่ทั้งใน
train และ test ตัวเลข TSS ที่รายงานจะสูงเกินจริงโดยที่เราไม่รู้ตัว
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sunseg.config import SplitConfig
from sunseg.data.splits import (
    apply_splits,
    assign_harp_splits,
    describe_split_balance,
    harp_lifespans,
    verify_no_harp_overlap,
)

CFG = SplitConfig(train_end=date(2015, 6, 30), val_end=date(2016, 6, 30), gap_days=14)


def make_lifespans(entries: list[tuple[int, str, str]]) -> pd.DataFrame:
    """สร้างตารางอายุขัยของ HARP จาก (harpnum, วันเริ่ม, วันจบ)"""
    return pd.DataFrame(
        [
            {
                "HARPNUM": harpnum,
                "t_first": pd.Timestamp(first),
                "t_last": pd.Timestamp(last),
                "n_obs": 100,
            }
            for harpnum, first, last in entries
        ]
    )


class TestAssignHarpSplits:
    def test_assigns_each_period_correctly(self):
        lifespans = make_lifespans(
            [
                (1, "2013-03-01", "2013-03-14"),  # อยู่ในช่วง train
                (2, "2015-08-01", "2015-08-14"),  # อยู่ในช่วง val
                (3, "2017-02-01", "2017-02-14"),  # อยู่ในช่วง test
            ]
        )
        result = assign_harp_splits(lifespans, CFG).set_index("HARPNUM")["split"]

        assert result[1] == "train"
        assert result[2] == "val"
        assert result[3] == "test"

    def test_harp_straddling_a_boundary_is_dropped(self):
        """HARP ที่มีชีวิตคร่อมเส้นแบ่งจะทำให้เกิด leakage — ต้องถูกทิ้งทั้งดวง"""
        lifespans = make_lifespans([(1, "2015-06-25", "2015-07-10")])
        assert assign_harp_splits(lifespans, CFG)["split"].iloc[0] == "drop"

    def test_harp_inside_the_gap_is_dropped(self):
        """HARP ที่อยู่ในช่วง gap ต้องถูกทิ้ง เพื่อรักษาระยะห่างเชิงเวลาระหว่าง split"""
        lifespans = make_lifespans([(1, "2015-07-02", "2015-07-08")])
        assert assign_harp_splits(lifespans, CFG)["split"].iloc[0] == "drop"

    def test_gap_is_at_least_the_forecast_horizon(self):
        """gap ต้องกว้างกว่าหน้าต่างพยากรณ์ 24 ชม. มาก ไม่งั้น label จะมองข้ามเส้นแบ่ง"""
        assert timedelta(days=CFG.gap_days) > timedelta(hours=24)

    def test_no_harp_appears_in_two_splits(self):
        lifespans = make_lifespans(
            [(i, f"2013-{(i % 12) + 1:02d}-01", f"2013-{(i % 12) + 1:02d}-10") for i in range(1, 25)]
        )
        assigned = assign_harp_splits(lifespans, CFG)

        assert not assigned.duplicated("HARPNUM").any()

    def test_empty_input_does_not_crash(self):
        empty = pd.DataFrame(columns=["HARPNUM", "t_first", "t_last", "n_obs"])
        assert "split" in assign_harp_splits(empty, CFG).columns


class TestHarpLifespans:
    def test_computes_first_and_last_observation(self):
        df = pd.DataFrame(
            {
                "HARPNUM": [1, 1, 1, 2, 2],
                "t_rec": pd.to_datetime(
                    [
                        "2014-01-01 00:00",
                        "2014-01-03 00:00",
                        "2014-01-02 00:00",
                        "2014-05-01 00:00",
                        "2014-05-04 00:00",
                    ]
                ),
            }
        )
        result = harp_lifespans(df).set_index("HARPNUM")

        assert result.loc[1, "t_first"] == pd.Timestamp("2014-01-01")
        assert result.loc[1, "t_last"] == pd.Timestamp("2014-01-03")
        assert result.loc[1, "n_obs"] == 3
        assert result.loc[2, "n_obs"] == 2


class TestApplySplitsAndVerification:
    def test_dropped_harps_are_removed_from_samples(self):
        samples = pd.DataFrame({"HARPNUM": [1, 1, 2, 2, 3], "label": [0, 1, 0, 0, 1]})
        assigned = pd.DataFrame({"HARPNUM": [1, 2, 3], "split": ["train", "drop", "test"]})

        result = apply_splits(samples, assigned)
        assert set(result["HARPNUM"]) == {1, 3}
        assert 2 not in result["HARPNUM"].to_numpy()

    def test_harp_not_in_assignment_table_is_dropped(self):
        samples = pd.DataFrame({"HARPNUM": [1, 99], "label": [0, 1]})
        assigned = pd.DataFrame({"HARPNUM": [1], "split": ["train"]})

        result = apply_splits(samples, assigned)
        assert set(result["HARPNUM"]) == {1}

    def test_verify_passes_for_disjoint_splits(self):
        df = pd.DataFrame({"HARPNUM": [1, 1, 2, 3], "split": ["train", "train", "val", "test"]})
        verify_no_harp_overlap(df)  # ต้องไม่โยน exception

    def test_verify_detects_leakage(self):
        """สร้าง leakage โดยตั้งใจ — ตัวตรวจต้องจับได้"""
        df = pd.DataFrame({"HARPNUM": [1, 1, 2], "split": ["train", "test", "val"]})

        with pytest.raises(AssertionError, match="data leakage"):
            verify_no_harp_overlap(df)


class TestDescribeSplitBalance:
    def test_reports_positive_rate_per_split(self):
        df = pd.DataFrame(
            {
                "HARPNUM": [1, 1, 2, 2, 3, 3],
                "split": ["train", "train", "val", "val", "test", "test"],
                "label": [1, 0, 1, 1, 0, 0],
            }
        )
        balance = describe_split_balance(df).set_index("split")

        assert balance.loc["train", "pos_rate"] == pytest.approx(0.5)
        assert balance.loc["val", "pos_rate"] == pytest.approx(1.0)
        assert balance.loc["test", "pos_rate"] == pytest.approx(0.0)

    def test_handles_empty_split(self):
        df = pd.DataFrame({"HARPNUM": [1], "split": ["train"], "label": [1]})
        balance = describe_split_balance(df).set_index("split")

        assert balance.loc["val", "n_samples"] == 0
        assert balance.loc["val", "pos_rate"] == 0.0
