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

from sunseg.data.build_sequences import label_times
from sunseg.data.study_dataset import (
    FLARE_HISTORY_COLUMNS,
    FLARE_HISTORY_FLOOR_LOG10,
    INTENSITY_COLUMNS,
    XRAY_COLUMNS,
    anchored_grid,
    build_unified_window,
    flare_history_on_grid,
    intensity_columns_for,
    load_variant_columns,
    load_variants,
    select_columns,
)

VARIANTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "study" / "variants.yaml"

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


def _flares(*items: tuple[pd.Timestamp, float]) -> tuple[np.ndarray, np.ndarray]:
    times = np.array([t.to_datetime64() for t, _ in items], dtype="datetime64[ns]")
    return times, np.array([flux for _, flux in items], dtype=np.float64)


class TestFlareHistoryOnGrid:
    """ประวัติ flare ราย AR (งาน feature-evidence-cv) — จุดกริด t สรุป flare ที่ peak ใน (t − 12 ชม., t]"""

    def test_no_flares_gives_floor_and_zero_count(self):
        grid, _ = _grid()
        out = flare_history_on_grid(np.empty(0), np.empty(0), grid, CADENCE_HOURS)
        assert out.shape == (N_GRID_POINTS, len(FLARE_HISTORY_COLUMNS))
        assert (out[:, 0] == FLARE_HISTORY_FLOOR_LOG10).all()
        assert (out[:, 1] == 0).all()

    def test_each_flare_lands_in_the_slot_that_ends_at_or_after_its_peak(self):
        grid, _ = _grid()
        at_t = grid[2]
        times, fluxes = _flares(
            (at_t, 2e-6),                                  # peak พอดี t -> ช่องของ t
            (at_t - pd.Timedelta(hours=11, minutes=59), 5e-5),  # ยังอยู่ใน (t − 12, t]
            (at_t + pd.Timedelta(minutes=1), 3e-7),        # เลย t ไปแล้ว -> ช่องถัดไป
        )
        out = flare_history_on_grid(times, fluxes, grid, CADENCE_HOURS)
        assert out[2, 0] == pytest.approx(np.log10(5e-5), abs=1e-6)  # ค่าสูงสุดของช่อง
        assert out[2, 1] == 2                                         # สองตัว >= C1.0
        assert out[3, 0] == pytest.approx(np.log10(3e-7), abs=1e-6)
        assert out[3, 1] == 0                                         # B3 ไม่นับเป็น C+
        assert out[1, 0] == FLARE_HISTORY_FLOOR_LOG10

    def test_flares_outside_the_grid_are_ignored(self):
        grid, _ = _grid()
        times, fluxes = _flares(
            (grid[0] - pd.Timedelta(hours=13), 1e-4),
            (grid[-1] + pd.Timedelta(minutes=1), 1e-4),
        )
        out = flare_history_on_grid(times, fluxes, grid, CADENCE_HOURS)
        assert (out[:, 0] == FLARE_HISTORY_FLOOR_LOG10).all()
        assert (out[:, 1] == 0).all()

    def test_history_and_label_never_see_the_same_flare(self):
        """กัน leakage: flare ที่ feature ณ t เห็น ต้องไม่ทำให้ label ณ t เป็น 1 และกลับกัน"""
        grid, _ = _grid()
        t = grid[3]
        for peak in (t, t + pd.Timedelta(minutes=1), t - pd.Timedelta(hours=1)):
            times, fluxes = _flares((peak, 1e-4))
            history_sees = flare_history_on_grid(times, fluxes, grid, CADENCE_HOURS)[3, 1] > 0
            label = label_times(np.array([t.to_datetime64()]), times, horizon_hours=24)[0] == 1
            assert history_sees != label, f"peak {peak}: ทั้งสองฝั่งเห็น flare เดียวกัน"


class TestExtraColumns:
    def test_extra_channels_and_history_are_appended_after_xray(self):
        grid, end = _grid()
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=list(range(N_GRID_POINTS)))
        extra = intensity_columns_for(("94",))
        for column in extra:
            intensity[column] = 7.0
        history = {100: _flares((grid[4], 2e-5))}

        x, _, meta, names = build_unified_window(
            sharp, {}, intensity, _make_xray_bins(grid), GRID_START, end, _cfg(),
            extra_intensity_channels=("94",), flare_history=history,
        )

        expected = ["F1", *INTENSITY_COLUMNS, *XRAY_COLUMNS, *extra, *FLARE_HISTORY_COLUMNS]
        assert names == expected, "คอลัมน์ใหม่ต้องต่อท้าย ไม่งั้น index ของคอลัมน์เดิมเลื่อน"
        assert (x[:, :, names.index("94_p95")] == 7.0).all()
        # sample สุดท้าย (issue = grid[5]) มี timestep grid[2..5] — flare ที่ grid[4] อยู่ตำแหน่ง 2
        last = x[-1, :, names.index("ar_flare_log10max_12h")]
        assert last[2] == pytest.approx(np.log10(2e-5), abs=1e-6)
        assert (np.delete(last, 2) == FLARE_HISTORY_FLOOR_LOG10).all()
        assert meta["issue_time"].iloc[-1] == grid[5]

    def test_missing_extra_channel_drops_the_row_for_everyone(self):
        grid, end = _grid()
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=list(range(N_GRID_POINTS)))
        for column in intensity_columns_for(("94",)):
            intensity[column] = 1.0
        intensity.loc[intensity.index[-1], "94_p95"] = np.nan  # ช่องใหม่ขาดที่จุดสุดท้าย

        x, _, _, _ = build_unified_window(
            sharp, {}, intensity, _make_xray_bins(grid), GRID_START, end, _cfg(),
            extra_intensity_channels=("94",),
        )
        assert len(x) == 2  # หน้าต่างที่มีจุดสุดท้ายหายไป เหมือนตอนช่องเดิมขาด

    def test_old_intensity_table_without_the_new_channel_gives_zero_rows_not_an_error(self):
        grid, end = _grid()
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=list(range(N_GRID_POINTS)))
        x, _, _, _ = build_unified_window(
            sharp, {}, intensity, _make_xray_bins(grid), GRID_START, end, _cfg(),
            extra_intensity_channels=("94",),
        )
        assert len(x) == 0

    def test_defaults_keep_the_old_layout(self):
        grid, end = _grid()
        sharp = _make_sharp([100], grid)
        intensity = _make_intensity(100, grid, present_idx=list(range(N_GRID_POINTS)))
        _, _, _, names = build_unified_window(sharp, {}, intensity, _make_xray_bins(grid), GRID_START, end, _cfg())
        assert names == ["F1", *INTENSITY_COLUMNS, *XRAY_COLUMNS]


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
        study/build_dataset.py จะสร้าง — เทสต์นี้พังทันทีถ้าพิมพ์ชื่อ keyword ผิดใน YAML
        """
        from sunseg.config import load_data_config

        cfg = load_data_config()
        feature_names = [*cfg.sharp.features, *INTENSITY_COLUMNS, *XRAY_COLUMNS]

        for name, spec in load_variants(VARIANTS_PATH).items():
            select_columns(feature_names, spec["columns"])  # ไม่ควร raise
