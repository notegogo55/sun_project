"""ทดสอบการสรุปผลแบบจับคู่ seed (ticket 07)

เกณฑ์จาก spec: ตารางต้องมีทุกแบบ ทุก seed ครบ และ ΔTSS ต้องตรงกับที่คำนวณมือจากตัวเลขดิบ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sunseg.study.summary import (
    delta_by_architecture,
    delta_by_variant,
    end_to_end_hours,
    feature_group,
    feature_groups,
    interaction_summary,
    paired_summary,
    rank_cells,
    seed_table,
    timing_summary,
)


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


class TestPairOnFoldAndSeed:
    """cross-validation จับคู่ทั้ง fold และ seed พร้อมกัน — ใช้เครื่องจักรเดียวกับจับคู่ seed"""

    def _cv_runs(self) -> pd.DataFrame:
        # fold 0: V0=[0.4, 0.5]  V1=[0.5, 0.5]  (seed 0,1)
        # fold 1: V0=[0.6, 0.7]  V1=[0.9, 0.9]
        rows = []
        for fold, (v0, v1) in enumerate([([0.4, 0.5], [0.5, 0.5]), ([0.6, 0.7], [0.9, 0.9])]):
            for seed, (a, b) in enumerate(zip(v0, v1, strict=True)):
                rows.append({"variant": "V0", "fold": fold, "seed": seed, "scope": "test", "tss": a})
                rows.append({"variant": "V1", "fold": fold, "seed": seed, "scope": "test", "tss": b})
        return pd.DataFrame(rows)

    def test_delta_is_paired_on_both_fold_and_seed(self):
        runs = self._cv_runs()
        summary = paired_summary(runs, pair_cols=("fold", "seed")).set_index("variant")
        # Δ ต่อ (fold, seed) = [0.1, 0.0, 0.3, 0.2] -> mean 0.15
        assert summary.loc["V1", "delta_mean"] == pytest.approx(0.15)
        assert summary.loc["V1", "n_seeds"] == 4

    def test_missing_fold_seed_combo_raises(self):
        runs = self._cv_runs().drop(index=0)  # ทิ้งแถวหนึ่งของ V0 fold 0 seed 0
        with pytest.raises(ValueError, match=r"\('V0', 0, 0\)"):
            paired_summary(runs, pair_cols=("fold", "seed"))

    def test_seed_table_with_fold_seed_gives_multiindex_columns(self):
        table = seed_table(self._cv_runs(), "test", pair_cols=("fold", "seed"))
        assert (0, 1) in table.columns
        assert table.loc["V1", (1, 0)] == pytest.approx(0.9)


def _timed(variant: str, fold, epochs: int, train: float, total: float, prepare: float) -> dict:
    return {
        "variant": variant, "fold": fold, "epochs_run": epochs,
        "train_seconds": train, "total_seconds": total, "prepare_seconds": prepare,
    }


class TestTimingSummary:
    """เวลาตั้งแต่เริ่มจนได้ผลรายแบบ — ตัวเลขทุกช่องต้องตรงกับที่คำนวณมือจากตัวเลขดิบ"""

    def _runs(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _timed("V0", None, 10, 10.0, 12.0, 1.0),
                _timed("V0", None, 30, 90.0, 94.0, 1.0),
                _timed("V1", None, 20, 50.0, 54.0, 4.0),
                _timed("V1", None, 20, 70.0, 74.0, 4.0),
            ]
        )

    def test_per_variant_numbers_match_hand_computation(self):
        summary = timing_summary(self._runs()).set_index("variant")

        v0 = summary.loc["V0"]
        assert v0["n_runs"] == 2
        assert v0["total_mean"] == pytest.approx(53.0)
        assert v0["total_sd"] == pytest.approx(np.std([12.0, 94.0], ddof=1))
        assert v0["train_mean"] == pytest.approx(50.0)
        assert v0["other_mean"] == pytest.approx(3.0)  # (12-10 + 94-90) / 2
        assert v0["epochs_mean"] == pytest.approx(20.0)

    def test_prepare_is_counted_once_per_variant_not_once_per_run(self):
        summary = timing_summary(self._runs()).set_index("variant")

        # V1 สองรันบันทึก prepare = 4 ซ้ำกัน -> เวลารวม = 4 + (54 + 74) ไม่ใช่ 8 + 128
        assert summary.loc["V1", "prepare_seconds"] == pytest.approx(4.0)
        assert summary.loc["V1", "start_to_result_seconds"] == pytest.approx(132.0)
        assert summary.loc["V0", "start_to_result_seconds"] == pytest.approx(1.0 + 106.0)

    def test_sec_per_epoch_is_pooled_not_a_mean_of_per_run_rates(self):
        summary = timing_summary(self._runs()).set_index("variant")

        # V0: อัตราต่อรัน 1.0 และ 3.0 (เฉลี่ย 2.0) แต่รันหลังมี epoch มากกว่า 3 เท่า -> 100/40 = 2.5
        assert summary.loc["V0", "sec_per_epoch"] == pytest.approx(2.5)
        assert summary.loc["V1", "sec_per_epoch"] == pytest.approx(3.0)

    def test_ratios_are_against_control_and_control_is_one(self):
        summary = timing_summary(self._runs()).set_index("variant")

        assert summary.loc["V0", "total_ratio"] == pytest.approx(1.0)
        assert summary.loc["V0", "sec_per_epoch_ratio"] == pytest.approx(1.0)
        assert summary.loc["V1", "total_ratio"] == pytest.approx(64.0 / 53.0)
        assert summary.loc["V1", "sec_per_epoch_ratio"] == pytest.approx(3.0 / 2.5)

    def test_cross_validation_counts_prepare_once_per_fold(self):
        runs = pd.DataFrame(
            [_timed("V0", fold, 3, 9.0, 10.0, prepare) for fold, prepare in ((0, 2.0), (1, 5.0)) for _ in range(2)]
        )

        summary = timing_summary(runs).set_index("variant")

        # prepare ของแต่ละ fold นับหนึ่งครั้ง: 2 + 5 -> เวลารวม 7 + 4 รัน x 10
        assert summary.loc["V0", "prepare_seconds"] == pytest.approx(7.0)
        assert summary.loc["V0", "start_to_result_seconds"] == pytest.approx(47.0)

    def test_variants_keep_their_order_of_appearance(self):
        runs = pd.DataFrame([_timed(v, None, 5, 5.0, 6.0, 1.0) for v in ("V0", "V3", "V1")])
        assert list(timing_summary(runs)["variant"]) == ["V0", "V3", "V1"]

    def test_missing_control_raises(self):
        runs = pd.DataFrame([_timed("V1", None, 5, 5.0, 6.0, 1.0)])
        with pytest.raises(ValueError, match="V0"):
            timing_summary(runs)

    def test_missing_column_raises_and_names_it(self):
        runs = self._runs().drop(columns="total_seconds")
        with pytest.raises(ValueError, match="total_seconds"):
            timing_summary(runs)


class TestFeatureGroups:
    @pytest.mark.parametrize(
        ("column", "group"),
        [
            ("TOTUSJH", "sharp"),
            ("R_VALUE", "sharp"),  # มี _ แต่ไม่ได้ขึ้นต้นด้วยเลข
            ("4500_p95", "intensity"),
            ("304_median", "intensity"),
            ("xray_log10_median", "xray"),
            ("xray_log10_rel27d", "xray"),
        ],
    )
    def test_group_comes_from_the_column_name(self, column, group):
        assert feature_group(column) == group

    def test_a_variant_reports_every_group_it_draws_from(self):
        assert feature_groups(["TOTUSJH", "4500_p95", "xray_log10_median"]) == {"sharp", "intensity", "xray"}
        assert feature_groups(["xray_log10_median"]) == {"xray"}


class TestEndToEndHours:
    def _timing(self) -> pd.DataFrame:
        # เวลาเทรนทั้งแถว: V0 1 ชม., V1 2 ชม., V2 3 ชม.
        return pd.DataFrame(
            {"variant": ["V0", "V1", "V2"], "start_to_result_seconds": [3600.0, 7200.0, 10800.0]}
        )

    def _components(self) -> list[dict]:
        return [
            {"group": "shared", "hours": 5.0},
            {"group": "shared", "hours": 1.0},
            {"group": "intensity", "hours": 100.0},
            {"group": "xray", "hours": 0.5},
        ]

    def _groups(self) -> dict:
        return {"V0": {"sharp"}, "V1": {"sharp", "intensity"}, "V2": {"sharp", "xray"}}

    def test_each_variant_pays_only_for_the_sources_it_uses(self):
        table = end_to_end_hours(self._timing(), self._components(), self._groups()).set_index("variant")

        assert table.loc["V0", "shared_hours"] == pytest.approx(6.0)  # 5 + 1
        assert table.loc["V0", "intensity_hours"] == 0.0
        assert table.loc["V0", "xray_hours"] == 0.0
        assert table.loc["V0", "total_hours"] == pytest.approx(7.0)  # 6 + เทรน 1
        assert table.loc["V1", "total_hours"] == pytest.approx(6.0 + 100.0 + 2.0)
        assert table.loc["V2", "total_hours"] == pytest.approx(6.0 + 0.5 + 3.0)

    def test_ratio_is_against_control(self):
        table = end_to_end_hours(self._timing(), self._components(), self._groups()).set_index("variant")
        assert table.loc["V0", "total_ratio"] == pytest.approx(1.0)
        assert table.loc["V1", "total_ratio"] == pytest.approx(108.0 / 7.0)

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="ไม่รู้จัก"):
            end_to_end_hours(self._timing(), [{"group": "goes", "hours": 1.0}], self._groups())

    def test_variant_without_group_info_raises_and_names_it(self):
        with pytest.raises(ValueError, match="V2"):
            end_to_end_hours(self._timing(), self._components(), {"V0": {"sharp"}, "V1": {"sharp"}})

    def test_missing_control_raises(self):
        timing = self._timing().iloc[1:]
        with pytest.raises(ValueError, match="V0"):
            end_to_end_hours(timing, self._components(), self._groups())


# --------------------------------------------------------------------------- #
# ตารางสองแกน: สถาปัตยกรรม x แบบ
# --------------------------------------------------------------------------- #


def _cells(values: dict[tuple[str, str], list[float]], scope: str = "test") -> pd.DataFrame:
    """หนึ่งแถวต่อ (สถาปัตยกรรม, แบบ, seed) จาก ``{(arch, variant): [ค่าต่อ seed]}``"""
    rows = []
    for (architecture, variant), per_seed in values.items():
        for seed, value in enumerate(per_seed):
            rows.append(
                {"architecture": architecture, "variant": variant, "seed": seed, "scope": scope, "tss": value}
            )
    return pd.DataFrame(rows)


#: ตัวอย่างที่คำนวณมือได้ — Δ ตามแบบ ของ tcn ใหญ่กว่าของ lstm ทุก seed
_MATRIX = {
    ("lstm", "V0"): [0.70, 0.72],
    ("lstm", "V2"): [0.73, 0.74],
    ("tcn", "V0"): [0.60, 0.66],
    ("tcn", "V2"): [0.68, 0.70],
}


class TestDeltaByVariant:
    def test_each_architecture_is_compared_against_its_own_reference_variant(self):
        table = delta_by_variant(_cells(_MATRIX)).set_index(["architecture", "variant"])

        # lstm: 0.73-0.70 = 0.03, 0.74-0.72 = 0.02 -> เฉลี่ย 0.025
        assert table.loc[("lstm", "V2"), "delta_mean"] == pytest.approx(0.025)
        # tcn: 0.68-0.60 = 0.08, 0.70-0.66 = 0.04 -> เฉลี่ย 0.06 (ไม่ใช่เทียบกับ V0 ของ lstm)
        assert table.loc[("tcn", "V2"), "delta_mean"] == pytest.approx(0.06)

    def test_reference_variant_has_zero_delta_in_every_architecture(self):
        table = delta_by_variant(_cells(_MATRIX)).set_index(["architecture", "variant"])

        for architecture in ("lstm", "tcn"):
            assert table.loc[(architecture, "V0"), "delta_mean"] == pytest.approx(0.0)
            assert table.loc[(architecture, "V0"), "delta_sd"] == pytest.approx(0.0)

    def test_mean_is_the_plain_mean_of_the_cell(self):
        table = delta_by_variant(_cells(_MATRIX)).set_index(["architecture", "variant"])
        assert table.loc[("tcn", "V0"), "mean"] == pytest.approx(0.63)


class TestDeltaByArchitecture:
    def test_each_variant_is_compared_against_the_reference_architecture(self):
        table = delta_by_architecture(_cells(_MATRIX)).set_index(["variant", "architecture"])

        # V2: 0.68-0.73 = -0.05, 0.70-0.74 = -0.04 -> เฉลี่ย -0.045
        assert table.loc[("V2", "tcn"), "delta_mean"] == pytest.approx(-0.045)
        assert table.loc[("V0", "tcn"), "delta_mean"] == pytest.approx(-0.08)

    def test_reference_architecture_has_zero_delta_in_every_variant(self):
        table = delta_by_architecture(_cells(_MATRIX)).set_index(["variant", "architecture"])

        for variant in ("V0", "V2"):
            assert table.loc[(variant, "lstm"), "delta_mean"] == pytest.approx(0.0)

    def test_missing_architecture_for_one_seed_raises_and_names_it(self):
        runs = _cells(_MATRIX)
        broken = runs.drop(runs[(runs.architecture == "tcn") & (runs.variant == "V2") & (runs.seed == 1)].index)

        with pytest.raises(ValueError, match="จับคู่ไม่ได้"):
            delta_by_architecture(broken)


class TestRankCells:
    def test_cells_are_ranked_by_mean_highest_first(self):
        table = rank_cells(_cells(_MATRIX))

        # lstm V2 0.735 > lstm V0 0.71 > tcn V2 0.69 > tcn V0 0.63
        assert list(zip(table["architecture"], table["variant"])) == [
            ("lstm", "V2"), ("lstm", "V0"), ("tcn", "V2"), ("tcn", "V0"),
        ]
        assert list(table["rank"]) == [1, 2, 3, 4]
        assert table.loc[0, "mean"] == pytest.approx(0.735)

    def test_gap_is_paired_against_the_best_cell(self):
        table = rank_cells(_cells(_MATRIX)).set_index(["architecture", "variant"])

        # lstm V0 - lstm V2 ราย seed = [-0.03, -0.02] -> เฉลี่ย -0.025, SD 0.00707
        assert table.loc[("lstm", "V0"), "gap_mean"] == pytest.approx(-0.025)
        assert table.loc[("lstm", "V0"), "gap_sd"] == pytest.approx(np.std([-0.03, -0.02], ddof=1))
        assert table.loc[("lstm", "V2"), "gap_mean"] == 0.0

    def test_tied_only_when_gap_is_within_its_own_sd(self):
        runs = _cells({
            ("lstm", "V0"): [0.70, 0.80, 0.60],   # อันดับ 1 เฉลี่ย 0.70
            ("tcn", "V0"): [0.72, 0.74, 0.61],    # ห่าง [+0.02, -0.06, +0.01] เฉลี่ย -0.01, SD ~0.044
            ("darnn", "V0"): [0.60, 0.70, 0.50],  # ห่างคงที่ -0.10 ทุก seed, SD 0 -> แยกได้ชัด
        })
        table = rank_cells(runs).set_index("architecture")

        assert table.loc["lstm", "rank"] == 1
        assert bool(table.loc["tcn", "tied_with_best"])
        assert not bool(table.loc["darnn", "tied_with_best"])

    def test_other_scopes_are_ignored(self):
        runs = pd.concat([_cells(_MATRIX), _cells({("tcn", "V0"): [0.99, 0.99]}, scope="val")])
        table = rank_cells(runs, scope="val")
        assert list(table["architecture"]) == ["tcn"]

    def test_missing_seed_raises_and_names_it(self):
        runs = _cells(_MATRIX)
        broken = runs.drop(runs[(runs.architecture == "tcn") & (runs.variant == "V2") & (runs.seed == 1)].index)

        with pytest.raises(ValueError, match="จับคู่ไม่ได้"):
            rank_cells(broken)


class TestInteractionSummary:
    def test_reference_architecture_has_zero_interaction(self):
        table = interaction_summary(_cells(_MATRIX)).set_index(["architecture", "variant"])

        assert table.loc[("lstm", "V2"), "interaction_mean"] == pytest.approx(0.0)
        assert table.loc[("lstm", "V2"), "interaction_sd"] == pytest.approx(0.0)

    def test_reference_variant_has_zero_delta_and_zero_interaction(self):
        table = interaction_summary(_cells(_MATRIX)).set_index(["architecture", "variant"])

        assert table.loc[("tcn", "V0"), "delta_mean"] == pytest.approx(0.0)
        assert table.loc[("tcn", "V0"), "interaction_mean"] == pytest.approx(0.0)

    def test_interaction_is_the_difference_of_the_two_deltas(self):
        """tcn ได้ประโยชน์จาก V2 มากกว่า lstm 0.035 โดยเฉลี่ย — นี่คือตัวเลขที่ตารางแกนเดียวให้ไม่ได้"""
        table = interaction_summary(_cells(_MATRIX)).set_index(["architecture", "variant"])

        # ต่อ seed: (0.08-0.03) = 0.05 และ (0.04-0.02) = 0.02
        assert table.loc[("tcn", "V2"), "interaction_mean"] == pytest.approx(0.035)
        assert table.loc[("tcn", "V2"), "interaction_sd"] == pytest.approx(np.std([0.05, 0.02], ddof=1))

    def test_delta_column_matches_delta_by_variant(self):
        interaction = interaction_summary(_cells(_MATRIX)).set_index(["architecture", "variant"])
        by_variant = delta_by_variant(_cells(_MATRIX)).set_index(["architecture", "variant"])

        for key in [("lstm", "V2"), ("tcn", "V2")]:
            assert interaction.loc[key, "delta_mean"] == pytest.approx(by_variant.loc[key, "delta_mean"])
            assert interaction.loc[key, "delta_sd"] == pytest.approx(by_variant.loc[key, "delta_sd"])

    def test_pairing_is_per_seed_not_on_the_means(self):
        """ถ้าคำนวณจากค่าเฉลี่ยจะได้ SD = NaN/0 — ความผันผวนร่วมของ seed ต้องถูกหักออกตรง ๆ"""
        values = {
            ("lstm", "V0"): [0.50, 0.90],   # seed 1 ดีกว่ามากสำหรับทุกเซลล์ (noise ร่วม)
            ("lstm", "V2"): [0.52, 0.92],
            ("tcn", "V0"): [0.40, 0.80],
            ("tcn", "V2"): [0.45, 0.85],
        }
        table = interaction_summary(_cells(values)).set_index(["architecture", "variant"])

        # Δ ต่อ seed คงที่ทั้งคู่ (lstm 0.02, tcn 0.05) -> interaction คงที่ 0.03 และ SD = 0
        assert table.loc[("tcn", "V2"), "interaction_mean"] == pytest.approx(0.03)
        assert table.loc[("tcn", "V2"), "interaction_sd"] == pytest.approx(0.0)

    def test_missing_cell_raises_and_names_it(self):
        runs = _cells(_MATRIX)
        broken = runs.drop(runs[(runs.architecture == "tcn") & (runs.variant == "V0")].index)

        with pytest.raises(ValueError, match="จับคู่ไม่ได้"):
            interaction_summary(broken)

    def test_unknown_anchor_raises(self):
        with pytest.raises(ValueError, match="darnn"):
            interaction_summary(_cells(_MATRIX), anchor_architecture="darnn")


class TestTimingSummaryByArchitecture:
    def _runs(self) -> pd.DataFrame:
        """2 สถาปัตยกรรม x 2 แบบ x 2 seed — prepare เป็นของ (แบบ, fold) ทุกแถวจึงซ้ำค่ากัน"""
        rows = []
        for architecture, per_epoch in (("lstm", 1.0), ("darnn", 8.0)):
            for variant, prepare in (("V0", 10.0), ("V2", 20.0)):
                for seed in (0, 1):
                    rows.append({
                        "architecture": architecture, "variant": variant, "seed": seed, "fold": None,
                        "epochs_run": 10, "train_seconds": 10 * per_epoch,
                        "total_seconds": 10 * per_epoch + 1.0, "prepare_seconds": prepare,
                    })
        return pd.DataFrame(rows)

    def test_prepare_is_not_multiplied_by_the_number_of_seeds_or_variants(self):
        """เวลาที่เตรียมข้อมูลถูกบันทึกซ้ำในทุกรัน — บวกตรง ๆ จะได้ 4 เท่าของความจริง"""
        table = timing_summary(self._runs(), control="lstm", unit_cols=("architecture",))

        # ต่อสถาปัตยกรรมหนึ่ง: prepare ของ V0 (10) + ของ V2 (20) นับอย่างละครั้ง
        assert set(table["prepare_seconds"]) == {30.0}

    def test_ratio_is_against_the_reference_architecture(self):
        table = timing_summary(self._runs(), control="lstm", unit_cols=("architecture",)).set_index("architecture")

        assert table.loc["lstm", "sec_per_epoch_ratio"] == pytest.approx(1.0)
        assert table.loc["darnn", "sec_per_epoch_ratio"] == pytest.approx(8.0)

    def test_per_cell_grouping_keeps_both_axes(self):
        table = timing_summary(self._runs(), control=("lstm", "V0"), unit_cols=("architecture", "variant"))

        assert list(table.columns[:2]) == ["architecture", "variant"]
        assert len(table) == 4
        assert set(table.loc[table.variant == "V0", "prepare_seconds"]) == {10.0}

    def test_control_length_must_match_unit_cols(self):
        with pytest.raises(ValueError, match="unit_cols"):
            timing_summary(self._runs(), control="lstm", unit_cols=("architecture", "variant"))

    def test_missing_reference_raises(self):
        with pytest.raises(ValueError, match="เทียบสัดส่วนไม่ได้"):
            timing_summary(self._runs(), control="tcn", unit_cols=("architecture",))
