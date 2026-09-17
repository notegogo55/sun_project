"""การจับคู่ตำแหน่งจาก PositionFlare เข้ากับแคตตาล็อกของโมเดล

สิ่งที่ต้องไม่พังเด็ดขาด: ชุด flare หลังจับคู่ต้องเป็นชุดเดียวกับที่โมเดลใช้ทำ label ทุกดวง
(ไม่เพิ่ม ไม่ตัด ไม่เปลี่ยนคลาส) และ flare ของ PositionFlare หนึ่งดวงห้ามถูกแปะให้สองดวง
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from sunseg.data.flare_positions import (
    FlarePositionStore,
    match_positions,
    meta_path_for,
    model_catalog_events,
)

T0 = pd.Timestamp("2014-10-24 21:41:00")


def _pairs(rows):
    """rows: (peak_offset_min, goes_class, peak_flux, harpnum)"""
    return pd.DataFrame([
        {
            "start_time": T0 + pd.Timedelta(minutes=m - 10),
            "peak_time": T0 + pd.Timedelta(minutes=m),
            "end_time": T0 + pd.Timedelta(minutes=m + 15),
            "goes_class": cls,
            "peak_flux": flux,
            "noaa_ar": 12192,
            "location": None,
            "satellite": "G15",
            "frm_name": "NOAA/NGDC",
            "HARPNUM": harp,
        }
        for m, cls, flux, harp in rows
    ])


def _pf(rows):
    """rows: (peak_offset_min, goes_class, flux, lat, lon)"""
    frame = pd.DataFrame([
        {
            "flare_id": 1000 + i,
            "cycle": 24,
            "peak": T0 + pd.Timedelta(minutes=m),
            "goes_class": cls,
            "xrsb_irrad": flux,
            "lat": lat,
            "lon": lon,
            "pos_source": "SWPC" if lat is not None else None,
            "pos_at_limb": False,
            "active_region": 12192.0,
            "satellite": "GOES-15",
        }
        for i, (m, cls, flux, lat, lon) in enumerate(rows)
    ])
    return frame.sort_values("peak").reset_index(drop=True)


def _match(pairs, pf, **kwargs):
    return match_positions(model_catalog_events(pairs), pf, **kwargs)


class TestModelCatalogEvents:
    def test_pairs_collapse_to_one_event_with_lowest_harp(self):
        events = model_catalog_events(_pairs([(0, "X3.1", 3.1e-4, 4698), (0, "X3.1", 3.1e-4, 4697)]))
        assert len(events) == 1
        assert events.loc[0, "harpnum"] == 4697
        assert events.loc[0, "n_harps"] == 2

    def test_a_and_b_class_are_excluded(self):
        """PositionFlare ไม่มี A/B — ถ้าปล่อยไว้จะกลายเป็น 'หาคู่ไม่เจอ' ปลอม ๆ นับพันดวง"""
        events = model_catalog_events(_pairs([(0, "B5.0", 5e-7, 1), (30, "C2.0", 2e-6, 1)]))
        assert events["goes_class"].tolist() == ["C2.0"]


class TestMatchPositions:
    def test_exact_peak_time_copies_position(self):
        out = _match(_pairs([(0, "M1.0", 1e-5, 1)]), _pf([(0, "M1.4", 1.4e-5, -12.0, 40.0)]))
        row = out.iloc[0]
        assert row["matched"]
        assert (row["lat"], row["lon"]) == (-12.0, 40.0)
        assert row["match_dt_min"] == 0

    def test_keeps_model_class_not_position_flare_class(self):
        """สีและตัวกรองบนแผนที่ต้องตรงกับ label ของโมเดล — คลาส science เก็บแยกไว้ดูเฉย ๆ"""
        out = _match(_pairs([(0, "C1.0", 1e-6, 1)]), _pf([(0, "C1.4", 1.43e-6, 5.0, 5.0)]))
        assert out.loc[0, "goes_class"] == "C1.0"
        assert out.loc[0, "pf_goes_class"] == "C1.4"

    def test_legacy_scaled_boundary_class_still_matches(self):
        """C9.0 ของ NGDC เดิม = M1.3 ใน science — ตัวอักษรต่างแต่เป็นดวงเดียวกัน"""
        out = _match(_pairs([(0, "C9.0", 9e-6, 1)]), _pf([(0, "M1.3", 1.29e-5, 5.0, 5.0)]))
        assert out.loc[0, "matched"]

    def test_within_tolerance_matches_beyond_does_not(self):
        pairs = _pairs([(0, "C2.0", 2e-6, 1), (120, "C3.0", 3e-6, 1)])
        pf = _pf([(4, "C2.0", 2e-6, 1.0, 1.0), (120 + 15, "C3.0", 3e-6, 2.0, 2.0)])
        out = _match(pairs, pf, tolerance_min=10)
        assert out["matched"].tolist() == [True, False]
        assert out.loc[0, "match_dt_min"] == 4

    def test_wildly_different_flux_is_rejected(self):
        """เวลาตรงแต่แรงต่างกันร้อยเท่า = คนละดวงที่บังเอิญพีคพร้อมกัน"""
        out = _match(_pairs([(0, "C1.0", 1e-6, 1)]), _pf([(0, "X1.0", 1e-4, 1.0, 1.0)]))
        assert not out.loc[0, "matched"]
        assert pd.isna(out.loc[0, "lat"])

    def test_one_position_flare_is_never_shared(self):
        pairs = _pairs([(0, "C2.0", 2e-6, 1), (2, "C2.1", 2.1e-6, 1)])
        out = _match(pairs, _pf([(1, "C2.0", 2e-6, 1.0, 1.0)]))
        assert out["matched"].sum() == 1
        assert out["pf_flare_id"].dropna().is_unique

    def test_loser_takes_next_free_candidate(self):
        """สองดวงแย่งคู่เดียวกัน — ดวงที่แพ้ต้องได้ลองคู่ถัดไปที่ยังว่าง ไม่ใช่หลุดไปเฉย ๆ"""
        pairs = _pairs([(0, "C2.0", 2e-6, 1), (2, "C2.0", 2e-6, 2)])
        pf = _pf([(1, "C2.0", 2e-6, 1.0, 1.0), (7, "C2.0", 2e-6, 2.0, 2.0)])
        out = _match(pairs, pf, tolerance_min=10)
        assert out["matched"].all()
        assert out["pf_flare_id"].is_unique

    def test_output_has_exactly_the_model_events(self):
        pairs = _pairs([(0, "C2.0", 2e-6, 1), (60, "M5.0", 5e-5, 1), (500, "X2.0", 2e-4, 3)])
        out = _match(pairs, _pf([(0, "C2.0", 2e-6, 1.0, 1.0)]))
        assert out["goes_class"].tolist() == ["C2.0", "M5.0", "X2.0"]
        assert out["matched"].tolist() == [True, False, False]

    def test_matched_without_known_position_stays_unlocated(self):
        """PositionFlare เองก็ไม่รู้ตำแหน่งบางดวง — จับคู่ได้แต่ต้องไม่ถูกวาดที่ (0, 0)"""
        out = _match(_pairs([(0, "C2.0", 2e-6, 1)]), _pf([(0, "C2.0", 2e-6, None, None)]))
        assert out.loc[0, "matched"]
        assert pd.isna(out.loc[0, "lat"])


class TestStorePayload:
    def test_missing_file_is_not_an_error(self, tmp_path):
        store = FlarePositionStore(tmp_path / "ไม่มี.parquet")
        assert not store.available
        with pytest.raises(RuntimeError, match="build_flare_positions"):
            store.payload()

    def test_columns_align_and_nulls_survive(self, tmp_path):
        out = _match(
            _pairs([(0, "C2.0", 2e-6, 1), (60, "M5.0", 5e-5, 1)]),
            _pf([(0, "C2.0", 2e-6, 12.3, -45.6)]),
        )
        path = tmp_path / "flare_positions.parquet"
        out.to_parquet(path, index=False)
        meta_path_for(path).write_text(json.dumps({"tolerance_min": 7.0}), encoding="utf-8")

        payload = FlarePositionStore(path).payload()
        lists = {k: v for k, v in payload.items() if isinstance(v, list)}
        assert {len(v) for v in lists.values()} == {payload["n"]}
        assert payload["n_matched"] == 1 and payload["n_located"] == 1
        assert payload["lat"] == [12.3, None]
        assert payload["start"] == [-10, -10] and payload["end"] == [15, 15]
        assert payload["tolerance_min"] == 7.0
