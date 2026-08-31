"""ทดสอบการสกัดความเข้มแสงราย AR และการจับคู่เข้ากับ HARP

จุดสำคัญที่ต้องพิสูจน์: quiet-Sun normalisation ต้องลบผลของการคูณค่าคงที่ทั้งภาพออกได้จริง
(คุณสมบัติที่ทำให้มันแก้ instrument degradation ได้) และ AR ที่ไม่ซ้อนทับ HARP ใดเลยต้อง
ไม่ถูกจับคู่มั่ว
"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.data.intensity import (
    extract_frame_intensities,
    match_detection_to_harp,
    quiet_sun_normalise,
)
from sunseg.tracking.detect import Detection


def make_mask_with_two_ars(size: int = 32) -> np.ndarray:
    """mask ไบนารีที่มี AR สองดวงแยกกันชัดเจน (มุมบนซ้าย, มุมล่างขวา)"""
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[20:26, 20:26] = 1
    return mask


class TestQuietSunNormalise:
    def test_constant_scaling_cancels_out(self):
        """คุณสมบัติหลักที่ทำให้แก้ instrument degradation ได้"""
        rng = np.random.default_rng(0)
        base = rng.uniform(50, 150, size=(16, 16)).astype(np.float64)
        ar_mask = np.zeros((16, 16), dtype=bool)
        ar_mask[4:8, 4:8] = True
        base[ar_mask] = rng.uniform(500, 800, size=int(ar_mask.sum()))

        degraded = base * 0.6   # จำลองกล้องเสื่อม 40%

        normal_a, level_a = quiet_sun_normalise(base, ar_mask)
        normal_b, level_b = quiet_sun_normalise(degraded, ar_mask)

        assert np.allclose(normal_a, normal_b, equal_nan=True)
        assert level_b == pytest.approx(level_a * 0.6)

    def test_no_quiet_pixels_returns_nan(self):
        array = np.full((4, 4), 100.0)
        ar_mask = np.ones((4, 4), dtype=bool)   # mask คลุมทั้งภาพ ไม่เหลือ quiet Sun เลย
        normalised, level = quiet_sun_normalise(array, ar_mask)
        assert np.isnan(level)
        assert np.isnan(normalised).all()

    def test_ignores_non_finite_pixels_when_computing_level(self):
        array = np.array([[np.nan, 100.0], [100.0, 100.0]])
        ar_mask = np.zeros((2, 2), dtype=bool)
        _, level = quiet_sun_normalise(array, ar_mask)
        assert level == pytest.approx(100.0)


class TestMatchDetectionToHarp:
    def test_matches_the_overlapping_harp(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:6, 2:6] = True
        detection = Detection(label=1, centroid_x=4, centroid_y=4, bbox=(2, 2, 6, 6), area_px=16, mask=mask)

        identity = np.zeros((16, 16), dtype=np.int32)
        identity[2:6, 2:6] = 4242

        assert match_detection_to_harp(detection, identity) == 4242

    def test_unmatched_ar_is_not_forced_to_nearest_harp(self):
        """AR ที่ไม่ซ้อนทับกับ HARP ใดเลย ต้องคืน None ไม่ใช่ HARP ที่ใกล้ที่สุด"""
        mask = np.zeros((16, 16), dtype=bool)
        mask[0:2, 0:2] = True   # มุมบนซ้าย

        identity = np.zeros((16, 16), dtype=np.int32)
        identity[14:16, 14:16] = 999   # HARP อยู่มุมตรงข้าม ไม่ซ้อนทับเลย

        assert match_detection_to_harp(detection=Detection(
            label=1, centroid_x=1, centroid_y=1, bbox=(0, 0, 2, 2), area_px=4, mask=mask
        ), identity_map=identity) is None

    def test_plurality_wins_when_ar_straddles_two_harps(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:10, 4:10] = True   # 6x6 = 36 พิกเซล

        identity = np.zeros((16, 16), dtype=np.int32)
        identity[4:10, 4:8] = 100    # 6x4 = 24 พิกเซล อยู่ใน mask ทั้งหมด
        identity[4:10, 8:10] = 200   # 6x2 = 12 พิกเซล อยู่ใน mask ทั้งหมด

        detection = Detection(label=1, centroid_x=6, centroid_y=6, bbox=(4, 4, 10, 10), area_px=36, mask=mask)
        assert match_detection_to_harp(detection, identity) == 100


class TestExtractFrameIntensities:
    def test_missing_channel_does_not_break_other_channels(self):
        mask = make_mask_with_two_ars(32)
        identity = np.zeros((32, 32), dtype=np.int32)
        identity[2:8, 2:8] = 111
        identity[20:26, 20:26] = 222

        magnetogram = np.zeros((32, 32), dtype=np.float32)
        channel_171 = np.full((32, 32), 100.0, dtype=np.float32)
        channel_171[2:8, 2:8] = 900.0

        rows = extract_frame_intensities(
            timestamp="20240501_000000",
            magnetogram=magnetogram,
            predicted_mask=mask,
            identity_map=identity,
            channel_arrays={"171": channel_171, "304": None},
            min_area_px=4,
        )

        assert len(rows) == 2
        by_harp = {row["HARPNUM"]: row for row in rows}
        assert set(by_harp) == {111, 222}

        # ช่องที่มีข้อมูล (171) ต้องได้ค่าจริง ไม่ใช่ NaN
        assert np.isfinite(by_harp[111]["171_mean"])
        assert by_harp[111]["171_n_pixels"] > 0

        # ช่องที่ไม่มีข้อมูล (304) ต้องเป็น NaN ทุกสถิติ ไม่ใช่ทำให้ทั้งฟังก์ชันพัง
        assert np.isnan(by_harp[111]["304_mean"])
        assert np.isnan(by_harp[111]["304_quiet_sun"])

    def test_ar_unmatched_to_any_harp_is_dropped_not_kept_with_sentinel(self):
        mask = make_mask_with_two_ars(32)
        identity = np.zeros((32, 32), dtype=np.int32)
        identity[2:8, 2:8] = 111
        # เจตนาไม่ใส่ HARP ให้ AR ดวงที่สอง (มุมล่างขวา) — ต้องไม่ปรากฏในผลลัพธ์เลย

        magnetogram = np.zeros((32, 32), dtype=np.float32)
        channel_171 = np.full((32, 32), 100.0, dtype=np.float32)

        rows = extract_frame_intensities(
            timestamp="20240501_000000",
            magnetogram=magnetogram,
            predicted_mask=mask,
            identity_map=identity,
            channel_arrays={"171": channel_171},
            min_area_px=4,
        )

        assert len(rows) == 1
        assert rows[0]["HARPNUM"] == 111

    def test_no_detections_gives_empty_list(self):
        empty_mask = np.zeros((16, 16), dtype=np.uint8)
        identity = np.zeros((16, 16), dtype=np.int32)
        rows = extract_frame_intensities(
            timestamp="20240501_000000",
            magnetogram=np.zeros((16, 16)),
            predicted_mask=empty_mask,
            identity_map=identity,
            channel_arrays={"171": np.full((16, 16), 100.0)},
        )
        assert rows == []
