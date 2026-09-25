"""ทดสอบการวางแผน fold ของ rolling/blocked-window cross-validation

เน้นสองเรื่อง: (1) fold ที่ val/test มี positive น้อยเกินไปต้องถูกตัดทิ้งอัตโนมัติ ไม่ใช่
นับรวมอย่างเงียบๆ (2) split_labels ที่คืนมาต้องเรียงตรงกับแถวของ meta ที่ส่งเข้าไปเป๊ะ

หมายเหตุเรื่องวันที่ในข้อมูลจำลอง: วาง HARP แต่ละดวงให้ **จบพอดีที่ปลายหน้าต่างที่ต้องการ**
(ปีนั้นวันที่ 31 ธ.ค.) แทนที่จะเริ่มต้นหน้าต่าง — เพราะ gap 14 วันกินเฉพาะ "ต้น" ของ val/test
เท่านั้น ไม่กระทบปลายหน้าต่าง วิธีนี้เลี่ยงการหลุดไปเป็น "drop" โดยไม่ตั้งใจ และทำให้ HARP
ตัวสุดท้ายพา ``data_end`` ไปถึง ``test_end`` ของ fold สุดท้ายพอดี โดยไม่ต้องมี HARP เสริม
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from sunseg.config import CVConfig, SplitConfig
from sunseg.data.cv import plan_folds

BASE_SPLIT = SplitConfig(train_end=date(2023, 12, 31), val_end=date(2024, 12, 31), gap_days=14)


ANCHOR = pd.Timestamp("2011-01-01")  # ให้ HARP แรกสุดเริ่มตรงนี้ กำหนด data_start แน่นอน


def make_meta(entries: list[tuple[int, str, int]]) -> pd.DataFrame:
    """(harpnum, วันจบ, จำนวน positive) -> 10 sample รายวันต่อ HARP จบที่วันนั้นพอดี

    จบพอดีปลายปี (ไม่ใช่เริ่มต้นปี) โดยตั้งใจ: gap 14 วันกินเฉพาะ "ต้น" ของหน้าต่าง val/test
    เท่านั้น วางไว้ปลายหน้าต่างจึงชัวร์ว่าไม่หลุดไปเป็น "drop" ไม่ว่า data_start จริงจะเป็นวันไหน
    """
    rows = []
    for harpnum, end, n_pos in entries:
        dates = pd.date_range(end=end, periods=10, freq="D")
        for i, d in enumerate(dates):
            rows.append({"HARPNUM": harpnum, "issue_time": d, "label": 1 if i < n_pos else 0})
    return pd.DataFrame(rows)


def anchor_harp(harpnum: int, n_pos: int) -> tuple[int, pd.Timestamp, int]:
    """HARP ที่ตรึง data_start ไว้ที่ ANCHOR พอดี (ใช้แทนตัว 'train' ตัวแรกสุดของทุกเคส)"""
    return harpnum, ANCHOR + pd.Timedelta(days=9), n_pos  # date_range(end=..., periods=10) -> เริ่มที่ ANCHOR


def cfg_with_cv(cv: CVConfig) -> SimpleNamespace:
    return SimpleNamespace(split=BASE_SPLIT.model_copy(update={"cv": cv}))


class TestPlanFolds:
    def test_raises_without_cv_config(self):
        meta = make_meta([(1, "2013-12-31", 2)])
        with pytest.raises(ValueError, match="split.cv"):
            plan_folds(SimpleNamespace(split=BASE_SPLIT), meta)

    def test_raises_on_empty_meta(self):
        cfg = cfg_with_cv(CVConfig(train_years=1, val_years=1, test_years=1, step_years=1))
        with pytest.raises(ValueError, match="HARP"):
            plan_folds(cfg, pd.DataFrame(columns=["HARPNUM", "issue_time", "label"]))

    def test_fold_with_too_few_val_positives_is_marked_invalid(self):
        # data_start = ANCHOR (จาก HARP1) -> fold แรก train=ปีที่ 1, val=ปีที่ 2, test=ปีที่ 3
        meta = make_meta(
            [
                anchor_harp(1, 5),  # train
                (2, "2012-12-31", 1),  # val ของ fold แรก — positive น้อยกว่าขั้นต่ำ
                (3, "2013-12-31", 5),  # test ของ fold แรก
            ]
        )
        cfg = cfg_with_cv(
            CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, min_val_positive=3, min_test_positive=1)
        )
        plans = plan_folds(cfg, meta)
        assert plans, "ควรวางแผน fold ได้อย่างน้อยหนึ่งอัน"
        assert not plans[0].valid
        assert "val" in plans[0].reason

    def test_fold_meeting_thresholds_is_valid(self):
        meta = make_meta([anchor_harp(1, 5), (2, "2012-12-31", 5), (3, "2013-12-31", 5)])
        cfg = cfg_with_cv(
            CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, min_val_positive=3, min_test_positive=3)
        )
        plans = plan_folds(cfg, meta)
        assert plans and plans[0].valid

    def test_split_labels_align_with_input_rows(self):
        meta = make_meta([anchor_harp(1, 5), (2, "2012-12-31", 5), (3, "2013-12-31", 5)])
        cfg = cfg_with_cv(
            CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, min_val_positive=1, min_test_positive=1)
        )
        plans = plan_folds(cfg, meta)
        fold0 = plans[0]
        assert len(fold0.split_labels) == len(meta)
        assert set(fold0.split_labels[meta["HARPNUM"] == 1]) == {"train"}
        assert set(fold0.split_labels[meta["HARPNUM"] == 2]) == {"val"}
        assert set(fold0.split_labels[meta["HARPNUM"] == 3]) == {"test"}

    def test_blocked_window_does_not_grow_train_across_folds(self):
        """ยืนยันว่าเป็น blocked ไม่ใช่ expanding — HARP ของปีแรกสุดต้องหลุดจาก train ของ fold ถัดไป"""
        meta = make_meta(
            [
                anchor_harp(1, 5),  # train fold 0
                (2, "2012-12-31", 5),  # val fold 0 / train fold 1
                (3, "2013-12-31", 5),  # test fold 0 / val fold 1
                (4, "2014-12-31", 5),  # test fold 1 (ดันขอบข้อมูลให้ถึง test_end ของ fold 1 พอดี)
            ]
        )
        cfg = cfg_with_cv(
            CVConfig(train_years=1, val_years=1, test_years=1, step_years=1, min_val_positive=1, min_test_positive=1)
        )
        plans = plan_folds(cfg, meta)
        assert len(plans) >= 2
        fold1_labels = plans[1].split_labels
        assert set(fold1_labels[meta["HARPNUM"] == 1]) != {"train"}
        assert set(fold1_labels[meta["HARPNUM"] == 2]) == {"train"}
