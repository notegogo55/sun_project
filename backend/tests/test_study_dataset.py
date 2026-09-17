"""ทดสอบการรวม SHARP + intensity + X-ray บนกริดเวลาเดียวกัน (ticket 06)

ใช้ข้อมูลสังเคราะห์ที่คำนวณมือได้แทนไฟล์จริง (แบบเดียวกับ ``test_build_masks.py``/
``test_xray_flux.py``) — จุดที่สำคัญที่สุดคือพิสูจน์ว่าการเลือกคอลัมน์ (แทน "แบบ" ของ
การทดลอง) ไม่ทำให้จำนวนแถวหรือลำดับ ``(HARPNUM, issue_time)`` เปลี่ยน เพราะทุกข้อสรุปของ
งานเปรียบเทียบ 5 แบบตั้งอยู่บนสมมติฐานนี้
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sunseg.data.study_dataset import (
    INTENSITY_COLUMNS,
    XRAY_COLUMNS,
    anchored_grid,
    build_unified_window,
    load_variant_columns,
    load_variants,
    select_columns,
)

VARIANTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "study_variants.yaml"

GRID_START = pd.Timestamp("2020-01-01 00:00")
CADENCE_HOURS = 12
SEQ_LEN = 4
N_GRID_POINTS = 6  # -> 3 หน้าต่างต่อ HARP (issue_idx = 3, 4, 5)


def _cfg(features=("F1",), max_missing_frac=0.3, horizon_hours=24):
    return SimpleNamespace(
        sharp=SimpleNamespace(features=list(features), cadence_hours=CADENCE_HOURS),
        sequence=SimpleNamespace(length=SEQ_LEN, max_missing_frac=max_missing_frac),
        flare=SimpleNamespace(horizon_hours=horizon_hours),
    )


def _grid():
    end = GRID_START + pd.Timedelta(hours=CADENCE_HOURS * (N_GRID_POINTS - 1))
    return anchored_grid(GRID_START, end, CADENCE_HOURS), end


def _make_sharp(harpnums: list[int], grid: pd.DatetimeIndex, feature: str = "F1") -> pd.DataFrame:
    rows = []
    for harpnum in harpnums:
        for i, t in enumerate(grid):
            rows.append({"HARPNUM": harpnum, "t_rec": t, feature: float(i), "LAT_FWT": 0.0, "LON_FWT": 0.0})
    return pd.DataFrame(rows)


def _make_intensity(harpnum: int, grid: pd.DatetimeIndex, present_idx: list[int]) -> pd.DataFrame:
    rows = []
    for i in present_idx:
        row = {"HARPNUM": harpnum, "issue_time": grid[i].strftime("%Y%m%d_%H%M%S")}
        row.update({col: 1.0 for col in INTENSITY_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _make_xray_bins(grid: pd.DatetimeIndex) -> pd.DataFrame:
    data = {"t_rec": grid}
    data.update({col: 1.0 for col in XRAY_COLUMNS})
    return pd.DataFrame(data)


class TestAnchoredGrid:
    def test_spacing_matches_cadence(self):
        grid, end = _grid()
        assert len(grid) == N_GRID_POINTS
        assert grid[0] == GRID_START
        assert grid[-1] == end
        assert (grid[1:] - grid[:-1] == pd.Timedelta(hours=CADENCE_HOURS)).all()


class TestBuildUnifiedWindow:
    def test_keeps_only_timesteps_where_intensity_and_xray_are_complete(self):
        grid, end = _grid()
        # HARP 100: intensity ครบทุกจุดยกเว้นจุดสุดท้าย (index 5)
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=[0, 1, 2, 3, 4])
        xray_bins = _make_xray_bins(grid)

        x, y, meta, feature_names = build_unified_window(
            sharp, {}, intensity, xray_bins, GRID_START, end, _cfg()
        )

        # หน้าต่าง [0,1,2,3] และ [1,2,3,4] ครบ — หน้าต่าง [2,3,4,5] ขาด (index 5 ไม่มี intensity)
        assert x.shape == (2, SEQ_LEN, len(feature_names))
        assert len(feature_names) == 1 + len(INTENSITY_COLUMNS) + len(XRAY_COLUMNS)
        assert meta["HARPNUM"].tolist() == [100, 100]
        assert meta["issue_time"].tolist() == [grid[3], grid[4]]

    def test_harp_with_no_intensity_at_all_contributes_zero_rows(self):
        grid, end = _grid()
        sharp = _make_sharp([200], grid)
        intensity = pd.DataFrame()  # ไม่มี intensity เลยสักแถว
        xray_bins = _make_xray_bins(grid)

        x, y, meta, _ = build_unified_window(sharp, {}, intensity, xray_bins, GRID_START, end, _cfg())

        assert len(x) == 0
        assert len(meta) == 0

    def test_multiple_harps_concatenate_independently(self):
        grid, end = _grid()
        sharp = _make_sharp([100, 300], grid)
        intensity = pd.concat(
            [
                _make_intensity(100, grid, present_idx=[0, 1, 2, 3, 4]),
                _make_intensity(300, grid, present_idx=list(range(N_GRID_POINTS))),  # ครบทุกจุด
            ],
            ignore_index=True,
        )
        xray_bins = _make_xray_bins(grid)

        x, y, meta, _ = build_unified_window(sharp, {}, intensity, xray_bins, GRID_START, end, _cfg())

        # HARP 100: 2 sample (เหมือนเทสต์แรก), HARP 300: 3 sample (ครบทุกหน้าต่าง)
        assert len(x) == 5
        assert sorted(meta["HARPNUM"].tolist()) == [100, 100, 300, 300, 300]


class TestSelectColumns:
    def test_returns_indices_in_requested_order(self):
        names = ["A", "B", "C"]
        idx = select_columns(names, ["C", "A"])
        assert idx.tolist() == [2, 0]

    def test_missing_name_raises_with_the_bad_name_in_the_message(self):
        names = ["A", "B", "C"]
        with pytest.raises(KeyError, match="NOT_A_COLUMN"):
            select_columns(names, ["A", "NOT_A_COLUMN"])

    def test_column_selection_never_changes_row_count_or_order(self):
        """ทดสอบที่สำคัญที่สุดของ ticket 06: การเลือก "แบบ" ต้องไม่แตะจำนวน/ลำดับแถว"""
        grid, end = _grid()
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=list(range(N_GRID_POINTS)))
        xray_bins = _make_xray_bins(grid)

        x, y, meta, feature_names = build_unified_window(
            sharp, {}, intensity, xray_bins, GRID_START, end, _cfg()
        )
        n_rows_full = len(x)

        variants = [
            ["F1"],
            ["F1", *INTENSITY_COLUMNS[:2]],
            [*INTENSITY_COLUMNS, "xray_log10_median"],
            ["xray_log10_median"],
        ]
        for wanted in variants:
            idx = select_columns(feature_names, wanted)
            x_variant = x[:, :, idx]
            assert x_variant.shape[0] == n_rows_full
            assert x_variant.shape[1] == SEQ_LEN
            assert x_variant.shape[2] == len(wanted)


class TestLoadVariants:
    def test_all_variants_present_with_expected_sizes(self):
        variants = load_variants(VARIANTS_PATH)
        assert set(variants) == {"V0", "V1", "V2", "V3", "V4", "V5"}
        assert len(variants["V0"]["columns"]) == 18
        assert len(variants["V1"]["columns"]) == 18 + 3
        assert len(variants["V2"]["columns"]) == 18 + 1
        assert len(variants["V3"]["columns"]) == 18 + 3 + 1
        assert len(variants["V4"]["columns"]) == 3 + 1
        assert len(variants["V5"]["columns"]) == 1

    def test_flattened_columns_are_plain_strings_not_nested_lists(self):
        variants = load_variants(VARIANTS_PATH)
        for spec in variants.values():
            assert all(isinstance(c, str) for c in spec["columns"])

    def test_v5_is_exactly_the_xray_baseline_column(self):
        assert load_variant_columns(VARIANTS_PATH, "V5") == ["xray_log10_median"]

    def test_unknown_variant_name_raises(self):
        with pytest.raises(KeyError, match="V99"):
            load_variant_columns(VARIANTS_PATH, "V99")

    def test_variant_columns_are_selectable_against_the_real_feature_order(self):
        """คอลัมน์ที่ variants.yaml ระบุต้องมีอยู่จริงใน feature_names ที่
        build_study_dataset.py จะสร้าง — เทสต์นี้พังทันทีถ้าพิมพ์ชื่อ keyword ผิดใน YAML
        """
        from sunseg.config import load_data_config

        cfg = load_data_config()
        feature_names = [*cfg.sharp.features, *INTENSITY_COLUMNS, *XRAY_COLUMNS]

        for name, spec in load_variants(VARIANTS_PATH).items():
            select_columns(feature_names, spec["columns"])  # ไม่ควร raise
