"""ทดสอบการหั่น dataset ของงานศึกษาเป็นชุดข้อมูลของแต่ละ "แบบ" (ticket 07)

พิสูจน์สองคุณสมบัติที่ผลเปรียบเทียบทั้งตารางตั้งอยู่บน: (1) ทุกแบบเห็นแถวและ label ชุดเดียวกัน
เรียงเหมือนกัน และ (2) ค่าที่ normalise แล้วของคอลัมน์หนึ่งไม่ขึ้นกับว่าแบบนั้นมีคอลัมน์อื่นอะไร
อยู่ด้วย — ถ้าข้อ 2 ไม่จริง feature เดียวกันจะมีค่าต่างกันในแต่ละแบบ
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sunseg.data.build_sequences import compute_normalisation
from sunseg.datasets.study import StudyArrays, load_study_arrays, variant_splits

FEATURES = ["a", "b", "c"]


def _arrays(n: int = 12, seq_len: int = 3) -> StudyArrays:
    rng = np.random.default_rng(0)
    # สเกลต่างกันหลายสิบ order เหมือน SHARP จริง (USFLUX ~1e22 เทียบ MEANGAM ~40)
    x = (rng.normal(size=(n, seq_len, len(FEATURES))) * np.array([1.0, 1e3, 1e20])).astype(np.float32)
    y = (np.arange(n) % 4 == 0).astype(np.uint8)
    split = np.array(["train"] * 6 + ["val"] * 3 + ["test"] * 3)
    meta = pd.DataFrame(
        {
            "HARPNUM": np.arange(n) + 100,
            "issue_time": pd.date_range("2012-01-01", periods=n, freq="12h"),
            "split": split,
            "label": y,
        }
    )
    stats = compute_normalisation(x, split == "train")
    return StudyArrays(x=x, y=y, meta=meta, stats=stats, features=list(FEATURES))


class TestVariantSplits:
    def test_normalised_column_does_not_depend_on_other_columns(self):
        arrays = _arrays()
        with_others = variant_splits(arrays, ["a", "c"])
        alone = variant_splits(arrays, ["c"])

        for name in ("train", "val", "test"):
            np.testing.assert_array_equal(
                getattr(with_others, name).x[..., 1].numpy(), getattr(alone, name).x[..., 0].numpy()
            )

    def test_every_variant_sees_identical_rows_and_labels(self):
        arrays = _arrays()
        reference = variant_splits(arrays, ["a"])
        for columns in (["b", "c"], ["c", "a", "b"], ["b"]):
            splits = variant_splits(arrays, columns)
            for name in ("train", "val", "test"):
                assert len(getattr(splits, name)) == len(getattr(reference, name))
                np.testing.assert_array_equal(
                    getattr(splits, name).y.numpy(), getattr(reference, name).y.numpy()
                )
            assert splits.meta is arrays.meta

    def test_columns_and_stats_follow_the_requested_order(self):
        arrays = _arrays()
        splits = variant_splits(arrays, ["c", "a"])

        assert splits.features == ["c", "a"]
        assert splits.n_features == 2
        np.testing.assert_array_equal(splits.stats["mean"], arrays.stats["mean"][[2, 0]])
        np.testing.assert_array_equal(splits.stats["std"], arrays.stats["std"][[2, 0]])

    def test_unknown_column_raises_with_its_name(self):
        with pytest.raises(KeyError, match="nope"):
            variant_splits(_arrays(), ["a", "nope"])


class TestLoadStudyArrays:
    @staticmethod
    def _write(tmp_path, arrays: StudyArrays, label_override=None):
        np.save(tmp_path / "X.npy", arrays.x)
        np.save(tmp_path / "y.npy", arrays.y)
        meta = arrays.meta.copy()
        if label_override is not None:
            meta["label"] = label_override
        meta.to_parquet(tmp_path / "meta.parquet", index=False)
        # เก็บแบบเดียวกับ build_study_dataset.py (features เป็น object array)
        np.savez(
            tmp_path / "norm_stats.npz",
            mean=arrays.stats["mean"],
            std=arrays.stats["std"],
            features=np.array(arrays.features, dtype=object),
            transform="signed_log1p",
        )

    def test_roundtrip_of_what_build_study_dataset_writes(self, tmp_path):
        arrays = _arrays()
        self._write(tmp_path, arrays)

        loaded = load_study_arrays(tmp_path)

        assert loaded.features == FEATURES
        np.testing.assert_array_equal(loaded.x, arrays.x)
        assert list(loaded.meta["split"]) == list(arrays.meta["split"])

    def test_missing_files_point_to_the_builder_script(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_study_dataset"):
            load_study_arrays(tmp_path)

    def test_meta_out_of_order_with_y_is_rejected(self, tmp_path):
        arrays = _arrays()
        self._write(tmp_path, arrays, label_override=1 - arrays.y)
        with pytest.raises(ValueError, match="label"):
            load_study_arrays(tmp_path)
