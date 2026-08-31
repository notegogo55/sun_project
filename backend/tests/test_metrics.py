"""ตรวจสอบตัวชี้วัด โดยเฉพาะ TSS ที่เป็นตัวชี้วัดหลักของโปรเจค

จุดสำคัญที่ต้องพิสูจน์: **TSS ต้องไม่ถูกหลอกโดยข้อมูลที่ไม่สมดุล** ซึ่งเป็นเหตุผล
เดียวที่เราเลือกใช้มันแทน accuracy
"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.metrics import (
    brier_skill_score,
    classification_report,
    confusion_counts,
    dice_coefficient,
    find_best_threshold,
    hss2,
    iou_score,
    precision_recall_f1,
    roc_auc,
    threshold_sweep,
    tss,
)


class TestTSS:
    def test_perfect_prediction_gives_one(self):
        y = np.array([0, 1, 0, 1, 1])
        assert tss(y, y) == pytest.approx(1.0)

    def test_always_positive_gives_zero(self):
        """โมเดลที่ทายบวกตลอด: recall=1 แต่ FPR=1 ด้วย -> TSS=0"""
        y = np.array([0, 0, 0, 1])
        assert tss(y, np.ones_like(y)) == pytest.approx(0.0)

    def test_always_negative_gives_zero(self):
        """นี่คือกับดักที่ accuracy มองไม่เห็น — ทายลบตลอดได้ accuracy 99% แต่ TSS = 0"""
        y = np.array([0] * 99 + [1])
        y_pred = np.zeros_like(y)

        report = classification_report(y, y_pred.astype(float), threshold=0.5)
        assert report["accuracy"] == pytest.approx(0.99)
        assert report["tss"] == pytest.approx(0.0)

    def test_inverted_prediction_gives_minus_one(self):
        y = np.array([0, 1, 0, 1])
        assert tss(y, 1 - y) == pytest.approx(-1.0)

    def test_insensitive_to_class_imbalance(self):
        """คุณสมบัติเด่นของ TSS: เพิ่ม negative ที่ทายถูกเข้าไป ค่าต้องแทบไม่ขยับ"""
        y_balanced = np.array([1, 1, 0, 0])
        pred_balanced = np.array([1, 0, 0, 0])
        base = tss(y_balanced, pred_balanced)

        # เติม true negative อีก 1000 ตัว — สัดส่วน class เปลี่ยนไปมหาศาล
        y_skewed = np.concatenate([y_balanced, np.zeros(1000, dtype=int)])
        pred_skewed = np.concatenate([pred_balanced, np.zeros(1000, dtype=int)])
        assert tss(y_skewed, pred_skewed) == pytest.approx(base, abs=0.01)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="ขนาดไม่ตรงกัน"):
            tss(np.array([0, 1]), np.array([0, 1, 0]))


class TestConfusionCounts:
    def test_counts_are_correct(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        counts = confusion_counts(y_true, y_pred)

        assert (counts.tp, counts.fn, counts.tn, counts.fp) == (2, 1, 1, 1)
        assert counts.tp + counts.fn + counts.tn + counts.fp == len(y_true)


class TestHSS2:
    def test_perfect_prediction_gives_one(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        assert hss2(y, y) == pytest.approx(1.0)

    def test_always_negative_gives_zero(self):
        y = np.array([0] * 50 + [1] * 5)
        assert hss2(y, np.zeros_like(y)) == pytest.approx(0.0)


class TestBrierSkillScore:
    def test_perfect_probabilities_give_one(self):
        y = np.array([0, 1, 0, 1])
        assert brier_skill_score(y, y.astype(float)) == pytest.approx(1.0)

    def test_climatology_forecast_gives_zero(self):
        """ทายค่าเฉลี่ยระยะยาวตลอด = ไม่มี skill เลย"""
        y = np.array([0, 0, 0, 1])
        assert brier_skill_score(y, np.full(4, y.mean())) == pytest.approx(0.0)

    def test_worse_than_climatology_is_negative(self):
        y = np.array([0, 0, 0, 1])
        assert brier_skill_score(y, 1.0 - y) < 0


class TestRocAuc:
    def test_perfect_ranking_gives_one(self):
        y = np.array([0, 0, 1, 1])
        assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)

    def test_random_ranking_gives_half(self):
        y = np.array([0, 1, 0, 1])
        assert roc_auc(y, np.full(4, 0.5)) == pytest.approx(0.5)

    def test_single_class_returns_half(self):
        """ไม่มี positive เลย -> AUC นิยามไม่ได้ ต้องคืน 0.5 แทนการหารด้วยศูนย์"""
        assert roc_auc(np.zeros(5), np.random.rand(5)) == pytest.approx(0.5)


class TestFindBestThreshold:
    def test_finds_separating_threshold(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.1, 0.2, 0.8, 0.9])
        threshold, score = find_best_threshold(y, prob, metric="tss")

        assert 0.2 < threshold <= 0.8
        assert score == pytest.approx(1.0)

    def test_rejects_unknown_metric(self):
        with pytest.raises(ValueError, match="ไม่รู้จักตัวชี้วัด"):
            find_best_threshold(np.array([0, 1]), np.array([0.1, 0.9]), metric="ไม่มีจริง")


class TestThresholdSweep:
    """ฟังก์ชันที่ endpoint สรุปของแผง confusion matrix เรียกใช้โดยตรง

    ข้อกำหนดสำคัญที่สุด: ต้องให้ผลตรงกับการเรียก ``confusion_counts`` ทีละจุด
    เป๊ะ ๆ — spec ห้ามย้ายตรรกะการนับไปไว้ใน JS เพราะฝั่งนั้นไม่มี test คุมเลย
    ฟังก์ชันนี้จึงเป็นจุดเดียวที่พิสูจน์ความถูกต้องของตัวเลขทั้งแผง
    """

    def test_matches_pointwise_confusion_counts(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.2).astype(int)
        prob = rng.random(500)
        thresholds = np.linspace(0.01, 0.99, 50)

        sweep = threshold_sweep(y, prob, thresholds)
        for i, t in enumerate(thresholds):
            expected = confusion_counts(y, prob >= t)
            assert (
                sweep["tp"][i],
                sweep["fp"][i],
                sweep["tn"][i],
                sweep["fn"][i],
            ) == (expected.tp, expected.fp, expected.tn, expected.fn)

    def test_counts_sum_to_n_at_every_point(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        prob = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.5])
        sweep = threshold_sweep(y, prob, np.linspace(0.01, 0.99, 20))

        totals = sweep["tp"] + sweep["fp"] + sweep["tn"] + sweep["fn"]
        assert np.all(totals == len(y))

    def test_lowest_threshold_predicts_all_positive(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.2, 0.4, 0.6, 0.8])
        sweep = threshold_sweep(y, prob, np.array([0.01]))

        assert sweep["fn"][0] == 0
        assert sweep["tn"][0] == 0
        assert sweep["tp"][0] == 2
        assert sweep["fp"][0] == 2

    def test_highest_threshold_predicts_all_negative(self):
        y = np.array([0, 0, 1, 1])
        prob = np.array([0.2, 0.4, 0.6, 0.8])
        sweep = threshold_sweep(y, prob, np.array([0.99]))

        assert sweep["tp"][0] == 0
        assert sweep["fp"][0] == 0
        assert sweep["tn"][0] == 2
        assert sweep["fn"][0] == 2

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="ขนาดไม่ตรงกัน"):
            threshold_sweep(np.array([0, 1]), np.array([0.1, 0.2, 0.3]), np.array([0.5]))


class TestPrecisionRecallF1:
    def test_no_predicted_positives_gives_zero_without_dividing_by_zero(self):
        precision, recall, f1 = precision_recall_f1(np.array([1, 0]), np.array([0, 0]))
        assert (precision, recall, f1) == (0.0, 0.0, 0.0)


class TestSegmentationMetrics:
    def test_identical_masks_give_one(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True
        assert dice_coefficient(mask, mask) == pytest.approx(1.0, abs=1e-5)
        assert iou_score(mask, mask) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_masks_give_zero(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[0:3, 0:3] = True
        b[7:10, 7:10] = True
        assert dice_coefficient(a, b) == pytest.approx(0.0, abs=1e-5)
        assert iou_score(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_half_overlap_values(self):
        """A=4, B=4, ทับกัน 2 -> Dice = 2*2/8 = 0.5, IoU = 2/6 = 0.333"""
        a = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
        b = np.array([0, 0, 1, 1, 1, 1], dtype=bool)
        assert dice_coefficient(a, b) == pytest.approx(0.5, abs=1e-5)
        assert iou_score(a, b) == pytest.approx(1 / 3, abs=1e-5)

    def test_dice_is_always_at_least_iou(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.random((16, 16)) > 0.7
            b = rng.random((16, 16)) > 0.7
            assert dice_coefficient(a, b) >= iou_score(a, b) - 1e-9
