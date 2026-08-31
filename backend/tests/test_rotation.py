"""ตรวจสอบฟิสิกส์ของการหมุนแบบ differential

การหมุนคือหัวใจของ tracking — ถ้าสูตรผิด การจับคู่ AR จะพังทั้งระบบ จึงตรึงค่าไว้
ด้วยตัวเลขที่ตรวจสอบได้จากดาราศาสตร์
"""

from __future__ import annotations

import numpy as np
import pytest

from sunseg.tracking.rotation import (
    EARTH_ORBITAL_RATE,
    SNODGRASS_A,
    angular_separation,
    rotate_longitude,
    snodgrass_omega,
    wrap_longitude,
)


class TestSnodgrassOmega:
    def test_equator_sidereal_rate_equals_coefficient_a(self):
        """ที่ละติจูด 0 พจน์ sin จะเป็นศูนย์ทั้งหมด เหลือแค่ A"""
        assert snodgrass_omega(0.0, sidereal=True) == pytest.approx(SNODGRASS_A)
        assert snodgrass_omega(0.0, sidereal=True) == pytest.approx(14.713, abs=1e-3)

    def test_synodic_is_sidereal_minus_earth_orbit(self):
        """อัตราที่มองจากโลกช้ากว่าอัตราเทียบดาวฤกษ์ เพราะโลกโคจรตามไปด้วย"""
        sidereal = snodgrass_omega(0.0, sidereal=True)
        synodic = snodgrass_omega(0.0, sidereal=False)
        assert sidereal - synodic == pytest.approx(EARTH_ORBITAL_RATE)

    def test_poles_rotate_slower_than_equator(self):
        """คุณสมบัติสำคัญของ differential rotation — ถ้าข้อนี้ไม่ผ่านแปลว่าเครื่องหมายผิด"""
        equator = snodgrass_omega(0.0, sidereal=True)
        mid = snodgrass_omega(30.0, sidereal=True)
        high = snodgrass_omega(60.0, sidereal=True)
        assert equator > mid > high

    def test_symmetric_between_hemispheres(self):
        """สูตรใช้ sin^2 จึงต้องให้ผลเท่ากันทั้งซีกเหนือและใต้"""
        for latitude in (15.0, 30.0, 45.0, 60.0):
            assert snodgrass_omega(latitude) == pytest.approx(snodgrass_omega(-latitude))

    def test_accepts_array_input(self):
        latitudes = np.array([0.0, 30.0, 60.0])
        result = snodgrass_omega(latitudes)
        assert result.shape == latitudes.shape
        assert np.all(np.diff(result) < 0)

    def test_sidereal_rotation_period_at_equator_is_about_25_days(self):
        """คาบการหมุนที่เส้นศูนย์สูตรราว 24.5-25.4 วัน ตามที่สังเกตได้จริง"""
        period = 360.0 / snodgrass_omega(0.0, sidereal=True)
        assert 24.0 < period < 25.5


class TestRotateLongitude:
    def test_equator_moves_by_synodic_rate_in_one_day(self):
        """ในหนึ่งวัน AR ที่เส้นศูนย์สูตรควรเลื่อนไปเท่ากับอัตรา synodic พอดี"""
        expected = SNODGRASS_A - EARTH_ORBITAL_RATE
        assert rotate_longitude(0.0, 0.0, 1.0) == pytest.approx(expected, abs=1e-6)

    def test_sidereal_flag_gives_full_snodgrass_rate(self):
        assert rotate_longitude(0.0, 0.0, 1.0, sidereal=True) == pytest.approx(
            SNODGRASS_A, abs=1e-6
        )

    def test_zero_elapsed_time_does_not_move(self):
        assert rotate_longitude(42.0, 20.0, 0.0) == pytest.approx(42.0)

    def test_negative_time_rotates_backwards(self):
        """ต้องย้อนเวลาได้ เพื่อให้ทำนายย้อนหลังได้ด้วย"""
        forward = rotate_longitude(0.0, 0.0, 1.0)
        assert rotate_longitude(forward, 0.0, -1.0) == pytest.approx(0.0, abs=1e-6)

    def test_result_is_wrapped_to_pm180(self):
        """หมุน 40 วันต้องไม่ได้ลองจิจูดหลายร้อยองศา"""
        result = rotate_longitude(170.0, 0.0, 40.0)
        assert -180.0 <= result < 180.0

    def test_high_latitude_lags_behind_equator(self):
        """หลัง 10 วัน AR ละติจูดสูงต้องตามหลัง AR ที่เส้นศูนย์สูตรอย่างชัดเจน"""
        equator = rotate_longitude(0.0, 0.0, 10.0)
        high = rotate_longitude(0.0, 60.0, 10.0)
        assert equator - high > 20.0

    def test_twelve_hour_step_moves_about_seven_degrees(self):
        """cadence จริงของโปรเจคคือ 12 ชม. — ระยะเลื่อนขนาดนี้คือเหตุผลที่ต้องชดเชยการหมุน"""
        moved = rotate_longitude(0.0, 15.0, 0.5)
        assert 6.0 < moved < 7.5


class TestWrapLongitude:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0.0, 0.0), (180.0, -180.0), (-180.0, -180.0), (190.0, -170.0), (370.0, 10.0)],
    )
    def test_wraps_into_range(self, value, expected):
        assert wrap_longitude(value) == pytest.approx(expected)


class TestAngularSeparation:
    def test_identical_points_have_zero_separation(self):
        assert angular_separation(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-9)

    def test_along_equator_equals_longitude_difference(self):
        assert angular_separation(0.0, 0.0, 30.0, 0.0) == pytest.approx(30.0)

    def test_along_meridian_equals_latitude_difference(self):
        assert angular_separation(0.0, -10.0, 0.0, 25.0) == pytest.approx(35.0)

    def test_longitude_difference_shrinks_at_high_latitude(self):
        """ที่ละติจูดสูง ระยะทางจริงของลองจิจูดที่ต่างกันเท่ากันจะสั้นลง"""
        at_equator = angular_separation(0.0, 0.0, 20.0, 0.0)
        at_high = angular_separation(0.0, 60.0, 20.0, 60.0)
        assert at_high < at_equator

    def test_symmetric(self):
        forward = angular_separation(5.0, 10.0, 40.0, -15.0)
        backward = angular_separation(40.0, -15.0, 5.0, 10.0)
        assert forward == pytest.approx(backward)
