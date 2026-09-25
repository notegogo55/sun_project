"""ตรวจว่าตัวเลขในตารางคู่รูปตรงกับตารางสรุปที่รายงานใช้ (ticket 03)

ทดสอบ **ข้อมูลเบื้องหลังรูป ไม่ใช่ตัวภาพ** — สิ่งที่ผิดแล้วเสียหายคือรูปที่เล่าเรื่อง
ไม่ตรงกับตารางในรายงาน ส่วนความสวยงามตรวจด้วยตา

ตารางคู่รูปคือช่องทางเดียวที่ผู้อ่านรายงานฉบับพิมพ์ขาวดำใช้ตรวจตัวเลขในรูปได้
ถ้ามันไม่ตรงกับ report.md ก็ไม่มีใครจับได้
"""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import load_script

from sunseg.study.summary import delta_by_variant, interaction_summary

_figures = load_script("study/plot_figures.py")
ANCHOR_ARCHITECTURE = _figures.ANCHOR_ARCHITECTURE
ANCHOR_VARIANT = _figures.ANCHOR_VARIANT
StudyLabels = _figures.StudyLabels
figure_delta_tss = _figures.figure_delta_tss
figure_interaction = _figures.figure_interaction
matrix_only = _figures.matrix_only
mean_confusion = _figures.mean_confusion
panel_variants = _figures.panel_variants

LABELS = StudyLabels(
    variants={"V0": "18 SHARP", "V2": "18 SHARP + X-ray", "V5": "X-ray"},
    architectures={"lstm": "LSTM", "tcn": "TCN", "lstm_production": "LSTM (production)"},
    matrix_architectures=["lstm", "tcn"],
    matrix_variants=["V0", "V2"],
)

#: ค่าที่คำนวณมือได้ — Δ ตามแบบ ของ tcn (+0.06) ใหญ่กว่าของ lstm (+0.025)
_CELLS = {
    ("lstm", "V0"): [0.70, 0.72],
    ("lstm", "V2"): [0.73, 0.74],
    ("tcn", "V0"): [0.60, 0.66],
    ("tcn", "V2"): [0.68, 0.70],
    # แถวเสริม: มีไม่ครบทุกเซลล์โดยธรรมชาติ ต้องไม่หลุดเข้าไปในรูปของตารางหลัก
    ("lstm", "V5"): [0.08, 0.09],
    ("lstm_production", "V0"): [0.71, 0.71],
}


def _runs() -> pd.DataFrame:
    rows = []
    for (architecture, variant), per_seed in _CELLS.items():
        for seed, value in enumerate(per_seed):
            rows.append({
                "architecture": architecture, "variant": variant, "seed": seed, "scope": "test",
                "tss": value, "auc": 0.9, "n": 100, "n_pos": 10,
                "tp": 8.0, "fp": 5.0, "fn": 2.0, "tn": 85.0,
            })
    return pd.DataFrame(rows)


class TestMatrixOnly:
    def test_supplementary_rows_are_dropped(self):
        """แถวเสริมมีเซลล์ไม่ครบ ถ้าหลุดเข้าไปการจับคู่จะ error หรือแย่กว่านั้นคือเฉลี่ยผิด"""
        kept = matrix_only(_runs(), LABELS)

        assert set(kept["architecture"]) == {"lstm", "tcn"}
        assert set(kept["variant"]) == {"V0", "V2"}


class TestFigureTablesMatchTheReportTables:
    def test_delta_tss_table_matches_the_summary(self):
        runs = _runs()
        _, table = figure_delta_tss(runs, LABELS)
        expected = delta_by_variant(matrix_only(runs, LABELS), anchor_variant=ANCHOR_VARIANT, metric="tss")
        expected = expected[expected["scope"] == "test"].set_index(["architecture", "variant"])

        indexed = table.set_index(["architecture", "variant"])
        for key in [("lstm", "V2"), ("tcn", "V2")]:
            assert indexed.loc[key, "delta_tss_mean"] == pytest.approx(expected.loc[key, "delta_mean"])
            assert indexed.loc[key, "delta_tss_sd"] == pytest.approx(expected.loc[key, "delta_sd"])

    def test_delta_tss_table_leaves_out_the_reference_variant(self):
        """Δ ของแบบอ้างอิงเป็นศูนย์ตามนิยาม ไม่ใช่ผลการวัด — ใส่ในรูปแล้วชวนอ่านผิด"""
        _, table = figure_delta_tss(_runs(), LABELS)

        assert ANCHOR_VARIANT not in set(table["variant"])

    def test_interaction_table_matches_the_summary(self):
        runs = _runs()
        _, table = figure_interaction(runs, LABELS)
        expected = interaction_summary(
            matrix_only(runs, LABELS), anchor_architecture=ANCHOR_ARCHITECTURE,
            anchor_variant=ANCHOR_VARIANT, metric="tss",
        )
        expected = expected[expected["scope"] == "test"].set_index(["architecture", "variant"])

        indexed = table.set_index(["architecture", "variant"])
        assert indexed.loc[("tcn", "V2"), "interaction_mean"] == pytest.approx(
            expected.loc[("tcn", "V2"), "interaction_mean"]
        )
        assert indexed.loc[("tcn", "V2"), "interaction_sd"] == pytest.approx(
            expected.loc[("tcn", "V2"), "interaction_sd"]
        )

    def test_interaction_table_leaves_out_the_reference_architecture(self):
        """interaction ของสถาปัตยกรรมอ้างอิงเทียบกับตัวเองเป็นศูนย์เสมอ"""
        _, table = figure_interaction(_runs(), LABELS)

        assert ANCHOR_ARCHITECTURE not in set(table["architecture"])


class TestConfusionCells:
    def test_panels_cover_the_first_and_last_variant(self):
        assert panel_variants(LABELS) == ["V0", "V2"]

    def test_table_has_every_matrix_cell_in_config_order(self):
        """ทุกสถาปัตยกรรมได้ที่เท่ากัน — ไม่มีการคัดเฉพาะ anchor กับเซลล์ที่ชนะ"""
        table = mean_confusion(_runs(), LABELS)

        assert list(zip(table["architecture"], table["variant"], strict=True)) == [
            ("lstm", "V0"), ("lstm", "V2"), ("tcn", "V0"), ("tcn", "V2"),
        ]

    def test_table_matches_the_mean_over_seeds(self):
        table = mean_confusion(_runs(), LABELS).set_index(["architecture", "variant"])

        assert table.loc[("tcn", "V0"), "tss"] == pytest.approx(0.63)
        assert table.loc[("lstm", "V2"), "tp"] == pytest.approx(8.0)

    def test_ignores_supplementary_rows(self):
        """แถวเสริมต้องไม่หลุดเข้าไปในรูปของตารางหลัก แม้จะได้ TSS สูง"""
        runs = _runs()
        runs.loc[runs.architecture == "lstm_production", "tss"] = 0.99

        table = mean_confusion(runs, LABELS)

        assert "lstm_production" not in set(table["architecture"])
        assert "V5" not in set(table["variant"])
