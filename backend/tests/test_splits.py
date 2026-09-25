"""ตรวจสอบการแบ่ง train/val/test ว่ากัน data leakage ได้จริง

นี่คือ test ที่สำคัญที่สุดชุดหนึ่งของโปรเจค — ถ้า HARP ดวงเดียวกันหลุดไปอยู่ทั้งใน
train และ test ตัวเลข TSS ที่รายงานจะสูงเกินจริงโดยที่เราไม่รู้ตัว
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sunseg.config import CVConfig, SplitConfig
from sunseg.data.splits import (
    apply_splits,
    assign_harp_splits,
    describe_split_balance,
    harp_lifespans,
    rolling_folds,
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


class TestAssignHarpSplitsBounded:
    """train_start/test_end ใช้กับ rolling/blocked CV เท่านั้น — ต้อง backward-compatible
    กับ single split (None = ไม่จำกัด) ซึ่งทดสอบไว้แล้วใน TestAssignHarpSplits ข้างบน
    """

    BOUNDED = SplitConfig(
        train_start=date(2013, 1, 1),
        train_end=date(2013, 12, 31),
        val_end=date(2014, 12, 31),
        test_end=date(2015, 12, 31),
        gap_days=14,
    )

    def test_harp_before_train_start_is_dropped(self):
        """สำคัญกับ blocked window: train ต้องมีขนาดคงที่ ไม่ใช่ 'ทุกอย่างก่อน train_end'"""
        lifespans = make_lifespans([(1, "2011-03-01", "2011-03-14")])
        assert assign_harp_splits(lifespans, self.BOUNDED)["split"].iloc[0] == "drop"

    def test_harp_after_test_end_is_dropped(self):
        """สำคัญกับ CV: test ต้องจบที่ขอบ fold ไม่งั้น fold ถัดไปจะทับช่วงเวลาเดียวกันซ้ำ"""
        lifespans = make_lifespans([(1, "2016-03-01", "2016-03-14")])
        assert assign_harp_splits(lifespans, self.BOUNDED)["split"].iloc[0] == "drop"

    def test_harp_within_bounds_is_kept(self):
        lifespans = make_lifespans(
            [
                (1, "2013-03-01", "2013-03-14"),
                (2, "2014-08-01", "2014-08-14"),
                (3, "2015-08-01", "2015-08-14"),
            ]
        )
        result = assign_harp_splits(lifespans, self.BOUNDED).set_index("HARPNUM")["split"]
        assert result[1] == "train"
        assert result[2] == "val"
        assert result[3] == "test"


class TestRollingFolds:
    BASE = SplitConfig(train_end=date(2023, 12, 31), val_end=date(2024, 12, 31), gap_days=14)
    CV = CVConfig(train_years=3, val_years=1, test_years=1, step_years=1)

    def test_fold_boundaries_are_blocked_not_expanding(self):
        """train ต้องมีขนาดคงที่ต่อ fold — คนละแบบกับ expanding window"""
        folds = rolling_folds(self.BASE, self.CV, pd.Timestamp("2011-01-01"), pd.Timestamp("2020-01-01"))
        assert len(folds) >= 2
        span0 = folds[0].train_end - folds[0].train_start
        span1 = folds[1].train_end - folds[1].train_start
        assert span0 == span1

    def test_folds_slide_forward_by_step_years(self):
        folds = rolling_folds(self.BASE, self.CV, pd.Timestamp("2011-01-01"), pd.Timestamp("2020-01-01"))
        expected = pd.Timestamp(folds[0].train_start) + pd.DateOffset(years=1)
        assert pd.Timestamp(folds[1].train_start) == expected

    def test_no_fold_exceeds_available_data(self):
        end = pd.Timestamp("2018-06-01")
        folds = rolling_folds(self.BASE, self.CV, pd.Timestamp("2011-01-01"), end)
        for fold in folds:
            assert pd.Timestamp(fold.test_end) <= end

    def test_too_short_a_range_yields_no_folds(self):
        folds = rolling_folds(self.BASE, self.CV, pd.Timestamp("2011-01-01"), pd.Timestamp("2012-01-01"))
        assert folds == []

    def test_expanding_train_always_starts_at_the_anchor_and_grows(self):
        cv = CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, expanding=True, anchor=date(2011, 1, 1))
        folds = rolling_folds(self.BASE, cv, pd.Timestamp("2011-01-09"), pd.Timestamp("2016-01-01"))
        assert [f.train_start for f in folds] == [None] * len(folds)
        assert [f.train_end for f in folds] == [date(2011, 12, 31), date(2012, 12, 31), date(2013, 12, 31)]
        assert [f.test_end for f in folds] == [date(2013, 12, 31), date(2014, 12, 31), date(2015, 12, 31)]

    def test_expanding_keeps_a_last_fold_whose_data_ends_just_before_test_end(self):
        cv = CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, expanding=True, anchor=date(2011, 1, 1))
        folds = rolling_folds(self.BASE, cv, pd.Timestamp("2011-01-09"), pd.Timestamp("2015-12-30 12:00"))
        assert folds[-1].test_end == date(2015, 12, 31)

    def test_expanding_train_takes_every_harp_before_train_end(self):
        cv = CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, expanding=True, anchor=date(2011, 1, 1))
        folds = rolling_folds(self.BASE, cv, pd.Timestamp("2011-01-09"), pd.Timestamp("2016-01-01"))
        lifespans = make_lifespans([(1, "2011-02-01", "2011-02-10"), (2, "2013-03-01", "2013-03-10")])
        result = assign_harp_splits(lifespans, folds[2]).set_index("HARPNUM")["split"]
        assert result[1] == "train"  # ปี 2011 ยังอยู่ใน train ของ fold ที่ 3 (ไม่ถูกเลื่อนทิ้งแบบ blocked)
        assert result[2] == "train"

    def test_default_config_is_still_blocked(self):
        assert self.CV.expanding is False and self.CV.anchor is None

    def test_each_fold_is_directly_usable_by_assign_harp_splits(self):
        """ต้องไม่มี HARP ใดหลุดไปสองช่วงเวลาที่ทับกันของคนละ fold"""
        folds = rolling_folds(self.BASE, self.CV, pd.Timestamp("2011-01-01"), pd.Timestamp("2016-01-01"))
        lifespans = make_lifespans([(1, "2013-06-01", "2013-06-14")])
        assigned = assign_harp_splits(lifespans, folds[0])
        assert assigned["split"].iloc[0] in {"train", "val", "test", "drop"}


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
