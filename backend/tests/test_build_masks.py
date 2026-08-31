"""ทดสอบการสร้าง ground-truth mask จาก SHARP bitmap

เน้นการแปลงพิกัด ซึ่งผิดแบบเงียบๆ ได้ง่ายที่สุด: ถ้า mask เลื่อนไปจากตำแหน่งจริง
โมเดลจะยังเทรนได้และ loss จะลดลงสวยงาม แต่เรียนรู้สิ่งที่ผิดทั้งหมด
"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.data.build_masks import (
    BITMAP_STRONG_INSIDE,
    BITMAP_STRONG_OUTSIDE,
    BITMAP_WEAK_INSIDE,
    BITMAP_WEAK_OUTSIDE,
    assert_bitmap_encoding,
    binned_wcs,
    bitmap_to_binary,
    build_fulldisk_identity_map,
    build_fulldisk_mask,
    downsample,
    validate_frame,
)


def make_fulldisk(size: int = 64):
    """สร้าง sunpy Map แบบ helioprojective ขนาดเล็กไว้ทดสอบการแปลงพิกัด"""
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    data = np.zeros((size, size), dtype=np.float32)
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, obstime="2014-10-20", observer="earth", frame=frames.Helioprojective
    )
    header = sunpy.map.make_fitswcs_header(
        data, reference, scale=[2.0, 2.0] * u.arcsec / u.pix
    )
    return sunpy.map.Map(data, header)


def make_patch(harpnum: int, tx_arcsec: float, ty_arcsec: float, size: int = 32, value: float = BITMAP_STRONG_INSIDE):
    """สร้าง SHARP patch สังเคราะห์ขนาดเล็ก เต็มไปด้วยค่า bitmap เดียว วางตำแหน่งได้
    ตาม ``(tx_arcsec, ty_arcsec)`` — ใช้จำลอง patch สองดวงที่พื้นที่ซ้อนทับกัน
    """
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    data = np.full((size, size), value, dtype=np.float32)
    reference = SkyCoord(
        tx_arcsec * u.arcsec, ty_arcsec * u.arcsec,
        obstime="2014-10-20", observer="earth", frame=frames.Helioprojective,
    )
    header = sunpy.map.make_fitswcs_header(data, reference, scale=[2.0, 2.0] * u.arcsec / u.pix)
    header["harpnum"] = harpnum
    return sunpy.map.Map(data, header)


class TestBinnedWcs:
    """กริดที่ย่อแล้วต้องชี้ไปยังพิกัดท้องฟ้าเดิม ไม่เลื่อนแม้ครึ่งพิกเซล"""

    @pytest.mark.parametrize("target", [32, 16, 8])
    def test_center_maps_to_same_sky_position(self, target):
        import astropy.units as u

        fulldisk = make_fulldisk(64)
        wcs_small, shape = binned_wcs(fulldisk, target)
        assert shape == (target, target)

        full_center = fulldisk.wcs.pixel_to_world(31.5, 31.5)
        small_center = wcs_small.pixel_to_world((target - 1) / 2, (target - 1) / 2)

        assert abs((full_center.Tx - small_center.Tx).to_value(u.arcsec)) < 1e-6
        assert abs((full_center.Ty - small_center.Ty).to_value(u.arcsec)) < 1e-6

    def test_corner_maps_to_same_sky_position(self):
        """มุมภาพเป็นจุดที่ค่า CRPIX ผิดแล้วจะเห็นชัดที่สุด"""
        import astropy.units as u

        fulldisk = make_fulldisk(64)
        wcs_small, _ = binned_wcs(fulldisk, 8)

        # ขอบนอกสุดของพิกเซลแรก (พิกัด -0.5 ในทั้งสองกริด) ต้องตรงกัน
        full_corner = fulldisk.wcs.pixel_to_world(-0.5, -0.5)
        small_corner = wcs_small.pixel_to_world(-0.5, -0.5)

        assert abs((full_corner.Tx - small_corner.Tx).to_value(u.arcsec)) < 1e-6
        assert abs((full_corner.Ty - small_corner.Ty).to_value(u.arcsec)) < 1e-6

    def test_pixel_scale_grows_by_the_binning_factor(self):
        fulldisk = make_fulldisk(64)
        wcs_small, _ = binned_wcs(fulldisk, 16)
        ratio = np.abs(wcs_small.wcs.cdelt / fulldisk.wcs.wcs.cdelt)
        assert np.allclose(ratio, 4.0)

    def test_rejects_size_that_does_not_divide_evenly(self):
        fulldisk = make_fulldisk(64)
        with pytest.raises(ValueError, match="ไม่ลงตัว"):
            binned_wcs(fulldisk, 24)


class TestBuildFulldiskMask:
    def test_target_size_produces_mask_at_that_resolution(self):
        fulldisk = make_fulldisk(64)
        mask, _ = build_fulldisk_mask(fulldisk, [], target_size=16)
        assert mask.shape == (16, 16)

    def test_without_target_size_keeps_fulldisk_resolution(self):
        fulldisk = make_fulldisk(64)
        mask, _ = build_fulldisk_mask(fulldisk, [], target_size=None)
        assert mask.shape == (64, 64)

    def test_no_patches_gives_empty_mask(self):
        fulldisk = make_fulldisk(64)
        mask, n_used = build_fulldisk_mask(fulldisk, [], target_size=16)
        assert n_used == 0
        assert not mask.any()

    def test_broken_patch_does_not_abort_the_frame(self):
        """patch เดียวพังไม่ควรทำให้ทั้งเฟรมเสีย — ต้อง log แล้วไปต่อ"""

        class BrokenPatch:
            meta = {"harpnum": 999}
            data = np.zeros((4, 4))

        fulldisk = make_fulldisk(64)
        mask, n_used = build_fulldisk_mask(fulldisk, [BrokenPatch()], target_size=16)
        assert n_used == 0
        assert mask.shape == (16, 16)


class TestBuildFulldiskIdentityMap:
    """แผนที่ระบุตัวตนของ HARP — mask ไบนารีเดิม (build_fulldisk_mask) ต้องไม่ถูกแตะ
    เลย ทุก test ในนี้จึงพิสูจน์ผ่านการเทียบผลลัพธ์กับมัน ไม่ใช่แก้มันโดยตรง
    """

    def test_target_size_produces_map_at_that_resolution(self):
        fulldisk = make_fulldisk(64)
        identity, _ = build_fulldisk_identity_map(fulldisk, [], target_size=16)
        assert identity.shape == (16, 16)

    def test_no_patches_gives_empty_map(self):
        fulldisk = make_fulldisk(64)
        identity, n_used = build_fulldisk_identity_map(fulldisk, [], target_size=16)
        assert n_used == 0
        assert not identity.any()

    def test_broken_patch_does_not_abort_the_frame(self):
        """patch เดียวพังไม่ควรทำให้ทั้งเฟรมเสีย — ต้อง log แล้วไปต่อ (เหมือน build_fulldisk_mask)"""

        class BrokenPatch:
            meta = {"harpnum": 999}
            data = np.zeros((4, 4))

        fulldisk = make_fulldisk(64)
        identity, n_used = build_fulldisk_identity_map(fulldisk, [BrokenPatch()], target_size=16)
        assert n_used == 0
        assert identity.shape == (16, 16)

    def test_single_patch_pixels_carry_its_harpnum(self):
        fulldisk = make_fulldisk(64)
        patch = make_patch(harpnum=1234, tx_arcsec=0.0, ty_arcsec=0.0, size=32)

        identity, n_used = build_fulldisk_identity_map(fulldisk, [patch], target_size=16)
        assert n_used == 1
        assert identity.max() == 1234
        assert set(np.unique(identity).tolist()) <= {0, 1234}

    def test_converts_back_to_the_exact_same_binary_mask(self):
        """ข้อกำหนดสำคัญที่สุด: identity_map > 0 ต้องเท่ากับ build_fulldisk_mask() ทุกพิกเซล"""
        fulldisk = make_fulldisk(64)
        patches = [
            make_patch(harpnum=100, tx_arcsec=-15.0, ty_arcsec=5.0, size=32),
            make_patch(harpnum=200, tx_arcsec=15.0, ty_arcsec=-5.0, size=32),
        ]

        reference_mask, reference_n = build_fulldisk_mask(fulldisk, patches, target_size=16)
        identity, identity_n = build_fulldisk_identity_map(fulldisk, patches, target_size=16)

        assert identity_n == reference_n
        assert ((identity > 0).astype(np.uint8) == reference_mask).all()

    def test_overlap_winner_is_independent_of_input_order(self):
        """สองดวงที่ครอบพื้นที่ซ้อนกัน — ผลต้องเหมือนกันไม่ว่าจะส่ง patch เข้าไปลำดับไหน
        (กติกาที่เลือก: เรียงตาม HARPNUM ก่อนเสมอ แล้ว HARPNUM น้อยกว่าชนะ)
        """
        from sunseg.data.build_masks import reproject_patch_to_grid

        fulldisk = make_fulldisk(64)
        low = make_patch(harpnum=100, tx_arcsec=-8.0, ty_arcsec=0.0, size=32)
        high = make_patch(harpnum=200, tx_arcsec=8.0, ty_arcsec=0.0, size=32)

        target_wcs, target_shape = binned_wcs(fulldisk, 16)
        footprint_low = reproject_patch_to_grid(low, target_wcs, target_shape).astype(bool)
        footprint_high = reproject_patch_to_grid(high, target_wcs, target_shape).astype(bool)
        true_overlap = footprint_low & footprint_high
        # การตั้งค่านี้ต้องมีพื้นที่ซ้อนทับจริง ไม่งั้น test นี้ไม่ได้พิสูจน์อะไรเรื่องการชี้ขาด
        assert true_overlap.any()

        identity_a, _ = build_fulldisk_identity_map(fulldisk, [low, high], target_size=16)
        identity_b, _ = build_fulldisk_identity_map(fulldisk, [high, low], target_size=16)

        assert (identity_a == identity_b).all()
        # พิกเซลในบริเวณที่ซ้อนทับจริงต้องถูกตัดสินให้ HARPNUM น้อยกว่า (100) เสมอ
        assert (identity_a[true_overlap] == 100).all()
        # นอกบริเวณซ้อนทับ แต่ละดวงยังครองพื้นที่ของตัวเองตามปกติ
        assert (identity_a[footprint_low & ~true_overlap] == 100).all()
        assert (identity_a[footprint_high & ~true_overlap] == 200).all()


class TestBitmapEncoding:
    def test_threshold_selects_pixels_inside_the_boundary(self):
        bitmap = np.array([[BITMAP_WEAK_OUTSIDE, BITMAP_STRONG_OUTSIDE],
                           [BITMAP_WEAK_INSIDE, BITMAP_STRONG_INSIDE]], dtype=float)
        assert bitmap_to_binary(bitmap).tolist() == [[0, 0], [1, 1]]

    def test_nan_counts_as_outside(self):
        bitmap = np.array([[np.nan, BITMAP_STRONG_INSIDE]])
        assert bitmap_to_binary(bitmap).tolist() == [[0, 1]]

    def test_histogram_reports_every_value(self):
        bitmap = np.array([[1, 1, 33], [34, 2, 33]], dtype=float)
        assert assert_bitmap_encoding(bitmap) == {1: 2, 2: 1, 33: 2, 34: 1}

    def test_warns_on_unknown_encoding(self, caplog):
        assert_bitmap_encoding(np.array([[7.0, 33.0]]))
        assert "ไม่คาดคิด" in caplog.text

    def test_warns_when_no_pixel_is_inside_an_ar(self, caplog):
        assert_bitmap_encoding(np.array([[1.0, 2.0]]))
        assert "mask ที่ได้จะว่างเปล่า" in caplog.text


class TestDownsample:
    def test_mask_pixel_is_set_when_more_than_half_is_ar(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 1          # ครบทั้งพิกเซลปลายทางซ้ายบน
        mask[2, 2] = 1            # 1 ใน 4 ของพิกเซลปลายทางขวาล่าง
        out = downsample(mask, 2, is_mask=True)
        assert out.tolist() == [[1, 0], [0, 0]]

    def test_image_keeps_mean_value(self):
        image = np.full((8, 8), 3.5, dtype=np.float32)
        assert np.allclose(downsample(image, 2, is_mask=False), 3.5)


class TestValidateFrame:
    def test_accepts_a_reasonable_frame(self):
        image = np.ones((8, 8), dtype=np.float32)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[0, 0] = 1
        assert validate_frame(image, mask)[0]

    def test_rejects_shape_mismatch(self):
        ok, reason = validate_frame(np.ones((8, 8)), np.zeros((4, 4), dtype=np.uint8))
        assert not ok and "ขนาดไม่ตรงกัน" in reason

    def test_rejects_mostly_nan_image(self):
        image = np.full((8, 8), np.nan)
        image[:2] = 1.0
        mask = np.ones((8, 8), dtype=np.uint8)
        ok, reason = validate_frame(image, mask)
        assert not ok and "NaN" in reason

    def test_rejects_empty_mask(self):
        ok, reason = validate_frame(np.ones((8, 8)), np.zeros((8, 8), dtype=np.uint8))
        assert not ok and "ว่างเปล่า" in reason

    def test_rejects_mask_covering_most_of_the_image(self):
        ok, reason = validate_frame(np.ones((8, 8)), np.ones((8, 8), dtype=np.uint8))
        assert not ok and "มากผิดปกติ" in reason
