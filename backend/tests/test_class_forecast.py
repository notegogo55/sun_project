"""การรวมแบบจำลอง ≥M1.0 กับ ≥X1.0 ของโมเดลหลักเป็นคำพยากรณ์ระดับคลาส <M / M / X"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.inference.class_forecast import (
    LEVELS,
    combine_levels,
    level_confusion,
    level_evaluation,
    majority_alarm,
    multiclass_hss,
    select_strict_threshold,
    true_levels,
)


class TestMajorityAlarm:
    def test_each_seed_uses_its_own_threshold(self):
        probs = np.array([[0.30, 0.10], [0.30, 0.10], [0.30, 0.10]])
        thresholds = np.array([0.2, 0.4, 0.05])  # seed 0/2 เตือนคอลัมน์แรก, seed 2 เตือนคอลัมน์สอง

        mean, n_alarm, alarm = majority_alarm(probs, thresholds)

        np.testing.assert_allclose(mean, [0.30, 0.10])
        assert n_alarm.tolist() == [2, 1]
        assert alarm.tolist() == [True, False]

    def test_tie_is_not_an_alarm(self):
        """เสียงเท่ากันไม่เตือน — กติกาเดียวกับรูป storm forecast"""
        _, n_alarm, alarm = majority_alarm(np.array([[0.9], [0.1]]), np.array([0.5, 0.5]))
        assert n_alarm.tolist() == [1]
        assert alarm.tolist() == [False]

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            majority_alarm(np.zeros((3, 4)), np.zeros(2))


class TestLevels:
    def test_highest_alarm_wins(self):
        levels = combine_levels([False, True, True, False], [False, False, True, True])
        assert [LEVELS[i] for i in levels] == ["<M", "M", "X", "X"]

    def test_true_level_from_two_labels(self):
        assert true_levels([0, 1, 1], [0, 0, 1]).tolist() == [0, 1, 2]

    def test_x_label_without_m_label_is_an_error(self):
        """≥X1.0 ย่อมเป็น ≥M1.0 — ถ้าไม่ใช่แปลว่า dataset สองระดับมาจากแคตตาล็อกคนละชุด"""
        with pytest.raises(ValueError, match="≥X1.0"):
            true_levels([0], [1])


class TestEvaluation:
    def test_confusion_rows_are_truth(self):
        matrix = level_confusion([0, 1, 2, 2], [0, 2, 2, 1])
        assert matrix.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]

    def test_threshold_metrics_count_higher_levels_as_alarms(self):
        """ทำนาย X นับเป็นการเตือน ≥M ด้วย — sample ระดับ M ที่ถูกเรียกว่า X เป็น hit ของ ≥M"""
        result = level_evaluation(true=[0, 1, 1, 2], pred=[0, 2, 0, 2])

        m = result["thresholds"]["M"]
        assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 1, 1)
        x = result["thresholds"]["X"]
        assert (x["tp"], x["fp"], x["tn"], x["fn"]) == (1, 1, 2, 0)
        # event 3 ตัว (M, M, X) ทายระดับถูกเป๊ะตัวเดียว (X)
        assert result["exact_on_events"] == pytest.approx(1 / 3)

    def test_no_events_gives_no_exact_score(self):
        assert level_evaluation([0, 0], [0, 1])["exact_on_events"] is None

    def test_multiclass_hss_bounds(self):
        assert multiclass_hss(np.diag([5, 3, 2])) == pytest.approx(1.0)
        # ทายตามสัดส่วนโดยไม่รู้อะไรเลย → 0
        assert multiclass_hss(np.array([[4, 4], [4, 4]])) == pytest.approx(0.0)
        assert multiclass_hss(np.zeros((3, 3))) == 0.0


class TestStrictThreshold:
    def test_picks_threshold_that_separates_x_from_m(self):
        # ทุกแถวระดับ M เตือนอยู่แล้ว — X สองตัวมีความน่าจะเป็นสูงกว่า M ทุกตัว
        prob_x = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
        alarm_m = np.ones(5, dtype=bool)
        true = np.array([1, 1, 1, 2, 2])

        threshold = select_strict_threshold(prob_x, alarm_m, true)

        assert threshold == pytest.approx(0.6)
        levels = combine_levels(alarm_m, prob_x >= threshold)
        assert levels.tolist() == true.tolist()

    def test_false_x_alarms_are_penalised(self):
        """0.1 เตือน X ทั้งสองแถวซึ่งไม่มี X จริงเลย · 0.2 เตือนแค่แถวที่สอง จึงได้ skill สูงกว่า"""
        prob_x = np.array([0.1, 0.2])
        threshold = select_strict_threshold(prob_x, np.array([True, False]), np.array([1, 0]))
        assert threshold == pytest.approx(0.2)

    def test_ties_choose_the_lower_threshold(self):
        # ทั้งสอง threshold ได้ HSS = 0 เท่ากัน — เลือกตัวต่ำกว่า (เตือนไวกว่า)
        prob_x = np.array([0.1, 0.2])
        threshold = select_strict_threshold(prob_x, np.array([True, True]), np.array([1, 1]))
        assert threshold == pytest.approx(0.1)
