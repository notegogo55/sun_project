"""ทดสอบ pipeline ของภาพ AIA

เน้นการแปลงพิกัดเป็นหลัก ด้วยเหตุผลเดียวกับ ``test_build_masks.py``: ถ้า WCS ผิด ภาพที่
ได้จะยังดูสวยงามและเส้นขอบยังอยู่ในตำแหน่งที่ดูน่าเชื่อ แต่ตัวเลขความเข้มแสงราย AR
กลับถูกสุ่มมาจาก quiet Sun ทั้งหมด — ความล้มเหลวแบบนี้ไม่ส่งสัญญาณอะไรเลย
"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.data.aia import (
    AiaFrameStore,
    exposure_normalise,
    region_intensities,
    reproject_to_frame,
    tai_to_utc,
)
from sunseg.data.build_masks import binned_wcs, binned_wcs_from
from sunseg.data.frame_wcs import FrameWcsStore


def make_aia_like(size: int = 128, blob_xy: tuple[int, int] | None = None):
    """สร้าง Map ที่มีเรขาคณิตแบบ AIA lev1.5 — ไม่หมุน สเกลหยาบ จานอยู่กลางภาพ"""
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    data = np.zeros((size, size), dtype=np.float32)
    if blob_xy is not None:
        yy, xx = np.mgrid[:size, :size]
        data += 100.0 * np.exp(
            -(((xx - blob_xy[0]) ** 2 + (yy - blob_xy[1]) ** 2) / (2 * 3.0**2))
        )

    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec,
        obstime="2014-10-27", observer="earth", frame=frames.Helioprojective,
    )
    header = sunpy.map.make_fitswcs_header(
        data,
        reference,
        reference_pixel=[(size - 1) / 2, (size - 1) / 2] * u.pix,
        scale=[19.2, 19.2] * u.arcsec / u.pix,
        instrument="AIA",
        # ต้องระบุ wavelength ด้วย ไม่งั้น sunpy สร้าง AIAMap แล้วหาตารางสีไม่เจอ
        wavelength=171 * u.angstrom,
    )
    return sunpy.map.Map(data, header)


def make_hmi_like(size: int = 128, rotation_deg: float = 180.0):
    """สร้าง Map ที่มีเรขาคณิตแบบ HMI — **หมุน 180 องศา** และจุดอ้างอิงเยื้องศูนย์

    ทั้งสองอย่างนี้คือค่าจริงของ ``hmi.M_720s`` (``CROTA2 ~ 180.01``,
    ``CRPIX ~ (2037.0, 2049.6)`` ไม่ใช่ 2048.5)
    """
    import astropy.units as u
    import sunpy.map
    from astropy.coordinates import SkyCoord
    from sunpy.coordinates import frames

    data = np.zeros((size, size), dtype=np.float32)
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec,
        obstime="2014-10-27", observer="earth", frame=frames.Helioprojective,
    )
    header = sunpy.map.make_fitswcs_header(
        data,
        reference,
        # เยื้องจากกึ่งกลางไปครึ่งพิกเซล เลียนแบบ CRPIX จริงที่ไม่ลงตัวพอดี
        reference_pixel=[(size - 1) / 2 - 0.4, (size - 1) / 2 + 0.3] * u.pix,
        scale=[16.0, 16.0] * u.arcsec / u.pix,
        rotation_angle=rotation_deg * u.deg,
        instrument="HMI",
    )
    return sunpy.map.Map(data, header)


class TestReprojection:
    """ตัวจับบั๊กหลัก — จุดสว่างต้องลงตำแหน่งเดิมบนท้องฟ้า ไม่ใช่ตำแหน่งเดิมในอาร์เรย์"""

    def test_blob_lands_on_the_same_sky_position(self):
        """reproject ต้องชดเชยทั้งมุมหมุน สเกล และจุดอ้างอิงที่ต่างกันในคราวเดียว"""
        blob_px = (40, 30)
        aia = make_aia_like(128, blob_xy=blob_px)
        hmi = make_hmi_like(128)

        # ตำแหน่งบนท้องฟ้าของจุดสว่าง แล้วแปลงกลับเป็นพิกเซลบนกริดของ HMI
        sky = aia.wcs.pixel_to_world(*blob_px)
        expected_x, expected_y = hmi.wcs.world_to_pixel(sky)

        out = reproject_to_frame(aia, hmi.wcs, (128, 128))
        found_y, found_x = np.unravel_index(np.nanargmax(out), out.shape)

        assert abs(found_x - expected_x) < 1.0
        assert abs(found_y - expected_y) < 1.0

    def test_rotation_is_actually_applied(self):
        """กันการถดถอย: ถ้าละเลย CROTA2 จุดจะไปโผล่ฝั่งตรงข้ามของภาพ

        เป็นบั๊กที่เคยเกิดจริงและไม่มี test ตัวไหนดักไว้เลย
        """
        blob_px = (40, 30)
        aia = make_aia_like(128, blob_xy=blob_px)

        rotated = reproject_to_frame(aia, make_hmi_like(128, 180.0).wcs, (128, 128))
        upright = reproject_to_frame(aia, make_hmi_like(128, 0.0).wcs, (128, 128))

        ry, rx = np.unravel_index(np.nanargmax(rotated), rotated.shape)
        uy, ux = np.unravel_index(np.nanargmax(upright), upright.shape)

        # สองกรณีต้องอยู่คนละฝั่งของจุดกึ่งกลาง ไม่ใช่ตำแหน่งเดียวกัน
        assert (rx - 63.5) * (ux - 63.5) < 0
        assert (ry - 63.5) * (uy - 63.5) < 0

    def test_flux_is_not_wildly_amplified(self):
        """ค่าที่ interpolate ได้ต้องอยู่ในช่วงของภาพต้นทาง ไม่ใช่โตขึ้นเป็นเท่าตัว"""
        aia = make_aia_like(128, blob_xy=(64, 64))
        out = reproject_to_frame(aia, make_hmi_like(128).wcs, (128, 128))
        assert np.nanmax(out) <= np.nanmax(aia.data) * 1.05


class TestBinnedWcsFrom:
    def test_matches_the_map_based_wrapper(self):
        """ตัวที่รับ WCS ตรง ๆ ต้องให้ผลเท่ากับตัวที่รับ Map ทุกประการ"""
        import astropy.units as u

        fulldisk = make_hmi_like(64)
        from_map, shape_a = binned_wcs(fulldisk, 16)
        from_wcs, shape_b = binned_wcs_from(fulldisk.wcs, fulldisk.data.shape, 16)

        assert shape_a == shape_b == (16, 16)
        a = from_map.pixel_to_world(3.5, 4.5)
        b = from_wcs.pixel_to_world(3.5, 4.5)
        assert abs((a.Tx - b.Tx).to_value(u.arcsec)) < 1e-9
        assert abs((a.Ty - b.Ty).to_value(u.arcsec)) < 1e-9

    def test_rejects_size_that_does_not_divide_evenly(self):
        fulldisk = make_hmi_like(64)
        with pytest.raises(ValueError, match="ไม่ลงตัว"):
            binned_wcs_from(fulldisk.wcs, fulldisk.data.shape, 24)


class TestExposureNormalise:
    def test_divides_by_exposure_time(self):
        data = np.full((4, 4), 6.0, dtype=np.float32)
        assert np.allclose(exposure_normalise(data, 2.0), 3.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
    def test_rejects_non_positive_exposure(self, bad):
        with pytest.raises(ValueError, match="EXPTIME"):
            exposure_normalise(np.ones((2, 2)), bad)


class TestRegionIntensities:
    class FakeDetection:
        def __init__(self, mask):
            self.mask = mask

    def test_computes_statistics_inside_the_mask_only(self):
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[0, :4] = True  # ค่า 0,1,2,3

        (stats,) = region_intensities([self.FakeDetection(mask)], image)
        assert stats["n_pixels"] == 4
        assert stats["mean"] == pytest.approx(1.5)
        assert stats["median"] == pytest.approx(1.5)
        assert stats["total"] == pytest.approx(6.0)

    def test_ignores_nan_pixels(self):
        image = np.full((4, 4), np.nan, dtype=np.float32)
        image[0, 0] = 10.0
        mask = np.ones((4, 4), dtype=bool)

        (stats,) = region_intensities([self.FakeDetection(mask)], image)
        assert stats["n_pixels"] == 1
        assert stats["mean"] == pytest.approx(10.0)

    def test_all_nan_region_is_reported_as_empty(self):
        image = np.full((4, 4), np.nan, dtype=np.float32)
        (stats,) = region_intensities([self.FakeDetection(np.ones((4, 4), bool))], image)
        assert stats == {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0, "n_pixels": 0}

    def test_no_detections_gives_no_rows(self):
        assert region_intensities([], np.ones((4, 4), dtype=np.float32)) == []

    def test_shape_mismatch_does_not_raise(self):
        """mask ที่ขนาดไม่ตรงกับภาพต้องถูกข้าม ไม่ใช่ทำให้ทั้งคำขอพัง"""
        detection = self.FakeDetection(np.ones((3, 3), dtype=bool))
        (stats,) = region_intensities([detection], np.ones((8, 8), dtype=np.float32))
        assert stats["n_pixels"] == 0

    def test_detection_without_mask_is_skipped(self):
        detection = self.FakeDetection(None)
        (stats,) = region_intensities([detection], np.ones((8, 8), dtype=np.float32))
        assert stats["n_pixels"] == 0


class TestTaiToUtc:
    def test_subtracts_leap_seconds(self):
        """TAI นำหน้า UTC อยู่ 35 วินาทีในปี 2014 — ถ้าไม่แปลงจะเลือก slot ผิด"""
        from datetime import datetime

        utc = tai_to_utc(datetime(2014, 10, 27, 0, 0, 0))
        assert utc.year == 2014 and utc.month == 10 and utc.day == 26
        assert (datetime(2014, 10, 27) - utc).total_seconds() == pytest.approx(35.0, abs=1.0)

    def test_offset_grew_after_the_2017_leap_second(self):
        from datetime import datetime

        offset = (datetime(2018, 1, 1) - tai_to_utc(datetime(2018, 1, 1))).total_seconds()
        assert offset == pytest.approx(37.0, abs=1.0)


class TestAiaFrameStore:
    def test_missing_directory_is_not_an_error(self, tmp_path):
        """ยังไม่ได้ดาวน์โหลด = แอปต้องเปิดได้ตามปกติ ไม่ใช่ล่ม"""
        store = AiaFrameStore(tmp_path / "ยังไม่มี")
        assert store.available is False
        assert store.channels == []
        assert store.n_frames == 0
        assert store.channels_for("20141027_000000") == []
        assert store.load("20141027_000000", "171") is None
        assert store.info()["available"] is False

    def test_round_trip_save_and_load(self, tmp_path):
        store = AiaFrameStore(tmp_path)
        data = np.arange(16, dtype=np.float32).reshape(4, 4)
        store.save("20141027_000000", "171", data)

        assert np.allclose(store.load("20141027_000000", "171"), data)
        assert AiaFrameStore(tmp_path).available is True

    def test_availability_is_reported_per_frame(self, tmp_path):
        store = AiaFrameStore(tmp_path)
        store.save("20141027_000000", "171", np.ones((4, 4), dtype=np.float32))
        store.save("20141027_000000", "304", np.ones((4, 4), dtype=np.float32))
        store.save("20141020_000000", "171", np.ones((4, 4), dtype=np.float32))

        assert store.channels_for("20141027_000000") == ["171", "304"]
        assert store.channels_for("20141020_000000") == ["171"]
        assert store.channels_for("19990101_000000") == []
        assert store.n_frames == 2

    def test_stores_float32_to_survive_flare_exposures(self, tmp_path):
        """ตอนเกิด flare ระบบ AEC ลด exposure ทำให้ DN/s ทะลุเพดาน float16"""
        store = AiaFrameStore(tmp_path)
        store.save("20141027_000000", "171", np.full((4, 4), 1e5, dtype=np.float32))

        loaded = store.load("20141027_000000", "171")
        assert np.isfinite(loaded).all()
        assert loaded[0, 0] == pytest.approx(1e5)


class TestFrameWcsStore:
    def test_missing_file_is_not_an_error(self, tmp_path):
        store = FrameWcsStore(tmp_path / "ไม่มีไฟล์.parquet")
        assert store.available is False
        assert store.header("20141027_000000") is None
        assert store.wcs("20141027_000000") is None
        assert store.solar_map("20141027_000000", np.zeros((8, 8))) is None
        assert "20141027_000000" not in store

    def test_rebuilds_a_wcs_that_keeps_the_hmi_rotation(self, tmp_path):
        """ค่าที่ประกอบกลับมาต้องคง CROTA2 ~ 180 ไว้ ไม่ใช่ปัดทิ้ง"""
        import pandas as pd

        row = {
            "frame": "20141027_000000",
            "T_REC": "2014.10.27_00:00:00_TAI",
            "DATE__OBS": "2014-10-26T23:59:24Z",
            "CRPIX1": 2037.024292, "CRPIX2": 2049.618164,
            "CDELT1": 0.504359, "CDELT2": 0.504359,
            "CRVAL1": 0.0, "CRVAL2": 0.0,
            "CROTA2": 180.01355,
            "CRLN_OBS": 100.0, "CRLT_OBS": 4.898514,
            "DSUN_OBS": 1.4809e11, "RSUN_OBS": 965.43, "RSUN_REF": 6.957e8,
            "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec", "CUNIT2": "arcsec",
            "QUALITY": 0,
        }
        path = tmp_path / "frame_wcs.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)

        store = FrameWcsStore(path)
        assert store.available is True
        assert "20141027_000000" in store

        wcs = store.wcs("20141027_000000")
        assert wcs is not None
        # ทิศทางต้องถูกพลิก — องค์ประกอบแนวทแยงของเมทริกซ์การหมุนติดลบ
        assert wcs.pixel_scale_matrix[0, 0] < 0

    def test_binned_wcs_scales_the_pixel_size(self, tmp_path):
        import pandas as pd

        row = {
            "frame": "20141027_000000", "T_REC": "x", "DATE__OBS": "2014-10-26T23:59:24Z",
            "CRPIX1": 2048.5, "CRPIX2": 2048.5, "CDELT1": 0.504359, "CDELT2": 0.504359,
            "CRVAL1": 0.0, "CRVAL2": 0.0, "CROTA2": 180.0,
            "CRLN_OBS": 100.0, "CRLT_OBS": 4.9, "DSUN_OBS": 1.4809e11,
            "RSUN_OBS": 965.43, "RSUN_REF": 6.957e8,
            "CTYPE1": "HPLN-TAN", "CTYPE2": "HPLT-TAN",
            "CUNIT1": "arcsec", "CUNIT2": "arcsec", "QUALITY": 0,
        }
        path = tmp_path / "frame_wcs.parquet"
        pd.DataFrame([row]).to_parquet(path, index=False)

        store = FrameWcsStore(path)
        full = store.wcs("20141027_000000", size=4096)
        small = store.binned_wcs("20141027_000000", 512)

        ratio = np.abs(small.wcs.cdelt / full.wcs.cdelt)
        assert np.allclose(ratio, 8.0)
