"""ทดสอบการสรุปผลแบบจับคู่ seed (ticket 07)

เกณฑ์จาก spec: ตารางต้องมีทุกแบบ ทุก seed ครบ และ ΔTSS ต้องตรงกับที่คำนวณมือจากตัวเลขดิบ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sunseg.study_summary import paired_summary, seed_table


def _runs(values: dict[str, list[float]], scope: str = "test") -> pd.DataFrame:
    rows = []
    for variant, per_seed in values.items():
        for seed, value in enumerate(per_seed):
            rows.append({"variant": variant, "seed": seed, "scope": scope, "tss": value})
    return pd.DataFrame(rows)


class TestPairedSummary:
    def test_delta_matches_hand_computation_from_raw_numbers(self):
        runs = _runs({"V0": [0.5, 0.6, 0.7], "V1": [0.6, 0.6, 0.9]})

        summary = paired_summary(runs).set_index("variant")

        # Δ ราย seed = [0.1, 0.0, 0.2] -> mean 0.1, sample SD 0.1
        assert summary.loc["V1", "delta_mean"] == pytest.approx(0.1)
        assert summary.loc["V1", "delta_sd"] == pytest.approx(0.1)
        assert summary.loc["V1", "mean"] == pytest.approx(0.7)
        assert summary.loc["V1", "sd"] == pytest.approx(np.std([0.6, 0.6, 0.9], ddof=1))
        assert summary.loc["V1", "n_seeds"] == 3

    def test_paired_sd_removes_variation_shared_by_the_same_seed(self):
        # V1 = V0 + 0.05 ทุก seed: แต่ละแบบแกว่งมาก แต่ผลต่างคงที่ -> SD แบบจับคู่ = 0
        runs = _runs({"V0": [0.2, 0.8, 0.5], "V1": [0.25, 0.85, 0.55]})
        summary = paired_summary(runs).set_index("variant")
        assert summary.loc["V1", "sd"] > 0.2
        assert summary.loc["V1", "delta_sd"] == pytest.approx(0.0, abs=1e-12)

    def test_control_row_has_zero_delta(self):
        summary = paired_summary(_runs({"V0": [0.5, 0.7], "V1": [0.4, 0.4]})).set_index("variant")
        assert summary.loc["V0", "delta_mean"] == 0.0
        assert summary.loc["V0", "delta_sd"] == 0.0

    def test_variants_keep_their_order_of_appearance(self):
        summary = paired_summary(_runs({"V0": [0.5], "V3": [0.4], "V1": [0.6]}))
        assert list(summary["variant"]) == ["V0", "V3", "V1"]

    def test_missing_seed_raises_and_names_what_is_missing(self):
        runs = _runs({"V0": [0.5, 0.6, 0.7], "V2": [0.6, 0.6]})
        with pytest.raises(ValueError, match=r"\('V2', 2\)"):
            paired_summary(runs)

    def test_missing_control_raises(self):
        with pytest.raises(ValueError, match="V0"):
            paired_summary(_runs({"V1": [0.5]}))

    def test_duplicate_run_raises(self):
        runs = pd.concat([_runs({"V0": [0.5]}), _runs({"V0": [0.6]})], ignore_index=True)
        with pytest.raises(ValueError, match="ซ้ำ"):
            paired_summary(runs)

    def test_scopes_are_summarised_independently(self):
        runs = pd.concat(
            [
                _runs({"V0": [0.5, 0.5], "V1": [0.7, 0.7]}, "test"),
                _runs({"V0": [0.9, 0.9], "V1": [0.8, 0.8]}, "val"),
            ],
            ignore_index=True,
        )
        summary = paired_summary(runs).set_index(["scope", "variant"])
        assert summary.loc[("test", "V1"), "delta_mean"] == pytest.approx(0.2)
        assert summary.loc[("val", "V1"), "delta_mean"] == pytest.approx(-0.1)

    def test_nan_is_not_silently_dropped(self):
        runs = _runs({"V0": [0.5, float("nan")], "V1": [0.6, 0.7]})
        summary = paired_summary(runs).set_index("variant")
        assert np.isnan(summary.loc["V0", "mean"])
        assert np.isnan(summary.loc["V1", "delta_mean"])


def test_seed_table_lays_out_raw_numbers_per_variant_and_seed():
    runs = _runs({"V0": [0.5, 0.6], "V1": [0.7, 0.8]})
    table = seed_table(runs, "test")
    assert list(table.index) == ["V0", "V1"]
    assert table.loc["V1", 1] == pytest.approx(0.8)
