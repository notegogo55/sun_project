"""การวิเคราะห์หลักของงาน feature-evidence-cv (``sunseg.study.evidence``) — ทดสอบด้วยข้อมูลสังเคราะห์
ที่รู้คำตอบล่วงหน้า ก่อนนำไปใช้กับผลจริง"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sunseg.study.evidence import (
    bootstrap_deltas,
    cluster_counts,
    holm,
    pooled_tss,
    tss_from_counts,
    verdict,
)


def _predictions(n_folds=2, n_harps=40, n_rows=5, seeds=(0, 1, 2), better=None, rng_seed=0) -> pd.DataFrame:
    """predictions test สังเคราะห์: HARP ครึ่งหนึ่งปะทุ · V0 ทำนายถูก 70% · ``better`` = แบบที่ถูก 90%"""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for fold in range(n_folds):
        for harp in range(n_harps):
            label = int(harp % 2 == 0)
            for seed in seeds:
                for variant, accuracy in (("V0", 0.7), ("VX", 0.9 if better == "VX" else 0.7), ("VY", 0.7)):
                    for _ in range(n_rows):
                        correct = rng.random() < accuracy
                        prob = 0.9 if (label == 1) == correct else 0.1
                        rows.append(
                            {"variant": variant, "seed": seed, "fold": fold, "HARPNUM": 1000 * fold + harp,
                             "label": label, "prob": prob, "threshold": 0.5}
                        )
    return pd.DataFrame(rows)


class TestCounts:
    def test_tss_of_a_perfect_forecast_is_one(self):
        assert tss_from_counts(np.array([10.0, 0.0, 0.0, 30.0])) == pytest.approx(1.0)

    def test_tss_without_positives_is_nan(self):
        assert np.isnan(tss_from_counts(np.array([0.0, 0.0, 3.0, 30.0])))

    def test_pooled_tss_sums_every_fold_before_computing_tss(self):
        cc = cluster_counts(_predictions(), ["V0", "VX", "VY"])
        assert cc.counts["V0"].shape == (3, 80, 4)
        summed = cc.counts["V0"].sum(axis=1)
        np.testing.assert_allclose(pooled_tss(cc, "V0"), tss_from_counts(summed))

    def test_missing_cluster_for_one_variant_raises(self):
        predictions = _predictions()
        drop = (predictions["variant"] == "VX") & (predictions["HARPNUM"] == 3) & (predictions["seed"] == 1)
        with pytest.raises(ValueError, match="ขาด"):
            cluster_counts(predictions[~drop], ["V0", "VX"])


class TestBootstrap:
    def test_identical_variant_has_zero_delta_and_p_one(self):
        predictions = _predictions()
        twin = predictions[predictions["variant"] == "V0"].assign(variant="VT")
        cc = cluster_counts(pd.concat([predictions, twin]), ["V0", "VT"])
        result = bootstrap_deltas(cc, "V0", n_boot=300)["VT"]
        assert result.delta == 0.0
        assert result.se == 0.0
        assert result.p_two_sided == 1.0

    def test_clearly_better_variant_is_detected(self):
        cc = cluster_counts(_predictions(better="VX"), ["V0", "VX", "VY"])
        results = bootstrap_deltas(cc, "V0", n_boot=1000)
        assert results["VX"].delta > 0.2
        assert results["VX"].ci95[0] > 0
        assert results["VX"].p_two_sided < 0.01
        assert results["VY"].p_two_sided > 0.05  # สุ่มจากความแม่นเท่ากัน

    def test_resampling_stays_inside_each_fold(self):
        """fold เล็กต้องไม่ถูกกลืนหาย: น้ำหนักรวมของแต่ละ fold ต้องเท่าจำนวน cluster ของ fold นั้นทุกรอบ"""
        from sunseg.study.evidence import _stratified_weights

        folds = np.array([0] * 30 + [1] * 5)
        weights = _stratified_weights(folds, 200, np.random.default_rng(1))
        assert (weights[:, folds == 0].sum(axis=1) == 30).all()
        assert (weights[:, folds == 1].sum(axis=1) == 5).all()

    def test_same_seed_gives_same_bootstrap(self):
        cc = cluster_counts(_predictions(), ["V0", "VX"])
        a = bootstrap_deltas(cc, "V0", n_boot=200, seed=7)["VX"]
        b = bootstrap_deltas(cc, "V0", n_boot=200, seed=7)["VX"]
        np.testing.assert_array_equal(a.boot, b.boot)


class TestDecision:
    def test_holm_step_down(self):
        adjusted = holm({"H1": 0.01, "H2": 0.04})
        assert adjusted["H1"] == pytest.approx(0.02)
        assert adjusted["H2"] == pytest.approx(0.04)

    def test_holm_is_monotone(self):
        adjusted = holm({"H1": 0.03, "H2": 0.02})
        assert adjusted["H2"] == pytest.approx(0.04)
        assert adjusted["H1"] == pytest.approx(0.04)  # ต้องไม่ต่ำกว่าตัวก่อนหน้า

    def _result(self, delta, ci90):
        from sunseg.study.evidence import DeltaResult

        return DeltaResult("VX", "V0", delta, 0.01, (ci90[0] - 0.01, ci90[1] + 0.01), ci90, 0.5, np.array([]))

    def test_verdicts_follow_the_spec(self):
        assert verdict(self._result(0.03, (0.01, 0.05)), p_adjusted=0.01) == "ช่วย"
        assert verdict(self._result(-0.03, (-0.05, -0.01)), p_adjusted=0.01) == "แย่ลง"
        assert verdict(self._result(0.002, (-0.01, 0.015)), p_adjusted=0.6) == "ไม่ต่างในทางปฏิบัติ"
        assert verdict(self._result(0.01, (-0.01, 0.03)), p_adjusted=0.3) == "สรุปไม่ได้"
