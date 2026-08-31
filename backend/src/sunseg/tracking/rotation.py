"""การหมุนแบบ differential ของดวงอาทิตย์ — หัวใจของการติดตาม active region

ดวงอาทิตย์ไม่ได้หมุนเป็นวัตถุแข็ง: บริเวณเส้นศูนย์สูตรหมุนครบรอบในราว 25 วัน ขณะที่
บริเวณละติจูดสูงใช้เวลาถึง ~34 วัน กฎเชิงประจักษ์ของ Snodgrass (1983) อธิบายด้วย::

    omega(theta) = A + B*sin^2(theta) + C*sin^4(theta)      [องศา/วัน]

**ทำไมเรื่องนี้สำคัญต่อการ tracking**: ถ้าจับคู่ active region ระหว่างเฟรมโดยดูระยะทาง
ในพิกัดภาพตรงๆ AR จะ "เคลื่อนหนี" ราว 13 องศา/วัน (ประมาณ 6.5 องศา ระหว่างเฟรมที่
ห่างกัน 12 ชม.) ซึ่งมากกว่าระยะห่างระหว่าง AR สองดวงที่อยู่ใกล้กัน ทำให้จับคู่ผิดดวง
เราจึงต้อง **ทำนายตำแหน่งใหม่จากการหมุนก่อน แล้วค่อยวัดระยะจากตำแหน่งที่ทำนายไว้**

หมายเหตุเรื่องกรอบอ้างอิง (จุดที่พลาดกันบ่อย): สัมประสิทธิ์ของ Snodgrass เป็นค่าใน
กรอบ **sidereal** (เทียบดาวฤกษ์คงที่) แต่พิกัด Stonyhurst วัดจากเส้นเมริเดียนกลาง
เมื่อมองจากโลก ซึ่งโลกเองก็โคจรรอบดวงอาทิตย์ไปด้วย ~0.9856 องศา/วัน การติดตามใน
พิกัด Stonyhurst จึงต้องใช้อัตรา **synodic** = sidereal - 0.9856
"""

from __future__ import annotations

import numpy as np

#: สัมประสิทธิ์ Snodgrass (1983) จากการติดตาม magnetic feature — หน่วยองศา/วัน (sidereal)
SNODGRASS_A = 14.713
SNODGRASS_B = -2.396
SNODGRASS_C = -1.787

#: อัตราเชิงมุมเฉลี่ยของการโคจรของโลกรอบดวงอาทิตย์ (องศา/วัน)
#: ใช้แปลงอัตรา sidereal เป็น synodic
EARTH_ORBITAL_RATE = 0.9856

#: อัตราหมุนของระบบพิกัด Carrington (แบบวัตถุแข็ง) — องศา/วัน
CARRINGTON_RATE = 14.1844


def snodgrass_omega(
    latitude_deg: float | np.ndarray,
    a: float = SNODGRASS_A,
    b: float = SNODGRASS_B,
    c: float = SNODGRASS_C,
    sidereal: bool = False,
) -> float | np.ndarray:
    """อัตราการหมุนที่ละติจูดที่กำหนด (องศา/วัน)

    Parameters
    ----------
    latitude_deg
        ละติจูดเฮลิโอกราฟิก เป็นองศา (บวก = ซีกเหนือ)
    sidereal
        ``False`` (ค่าเริ่มต้น) คืนอัตรา **synodic** ซึ่งเป็นอัตราที่ AR เคลื่อนที่จริง
        เมื่อมองจากโลก — ใช้กับพิกัด Stonyhurst
        ``True`` คืนอัตรา sidereal ตามสูตรดั้งเดิมของ Snodgrass

    Examples
    --------
    ที่เส้นศูนย์สูตร อัตรา sidereal เท่ากับสัมประสิทธิ์ A พอดี::

        >>> round(snodgrass_omega(0.0, sidereal=True), 3)
        14.713

    ขั้วหมุนช้ากว่าเส้นศูนย์สูตรอย่างชัดเจน::

        >>> snodgrass_omega(60.0, sidereal=True) < snodgrass_omega(0.0, sidereal=True)
        True
    """
    sin2 = np.sin(np.radians(latitude_deg)) ** 2
    omega = a + b * sin2 + c * sin2**2
    if not sidereal:
        omega = omega - EARTH_ORBITAL_RATE
    return omega


def rotate_longitude(
    longitude_deg: float | np.ndarray,
    latitude_deg: float | np.ndarray,
    delta_days: float,
    sidereal: bool = False,
) -> float | np.ndarray:
    """ทำนายลองจิจูดใหม่หลังผ่านไป ``delta_days`` วัน

    ละติจูดถือว่าคงที่ — active region เคลื่อนที่ตามละติจูดน้อยมากตลอดอายุขัย
    (ราว 1-2 องศาต่อการหมุนหนึ่งรอบ ซึ่งเล็กกว่าความคลาดเคลื่อนของการ detect)

    ค่าที่คืนถูกปรับให้อยู่ในช่วง [-180, 180) เพื่อให้เทียบกับพิกัด Stonyhurst ได้ตรง
    """
    omega = snodgrass_omega(latitude_deg, sidereal=sidereal)
    return wrap_longitude(np.asarray(longitude_deg) + omega * delta_days)


def wrap_longitude(longitude_deg: float | np.ndarray) -> float | np.ndarray:
    """ปรับลองจิจูดให้อยู่ในช่วง [-180, 180)"""
    return (np.asarray(longitude_deg) + 180.0) % 360.0 - 180.0


def angular_separation(
    lon1: float | np.ndarray,
    lat1: float | np.ndarray,
    lon2: float | np.ndarray,
    lat2: float | np.ndarray,
) -> float | np.ndarray:
    """ระยะเชิงมุมบนผิวทรงกลมระหว่างสองจุด (องศา)

    ใช้สูตร haversine ซึ่งเสถียรเชิงตัวเลขสำหรับระยะใกล้ ต่างจากสูตร cosine ตรงๆ
    ที่สูญเสียความแม่นยำเมื่อสองจุดอยู่ชิดกัน (ซึ่งเป็นกรณีปกติในการ tracking)
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lon2) - np.asarray(lon1))

    h = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def pixel_to_stonyhurst(
    x_pix: np.ndarray, y_pix: np.ndarray, solar_map
) -> tuple[np.ndarray, np.ndarray]:
    """แปลงพิกัดพิกเซลเป็นลองจิจูด/ละติจูด Stonyhurst

    Parameters
    ----------
    x_pix, y_pix
        พิกัดพิกเซล (นับจาก 0) ในระบบเดียวกับ ``solar_map.data``
    solar_map
        ``sunpy.map.GenericMap`` ที่มีข้อมูล WCS ครบ

    Returns
    -------
    ``(longitude_deg, latitude_deg)`` โดยจุดที่อยู่นอกจานสุริยะจะได้ ``nan``
    """
    import astropy.units as u
    from sunpy.coordinates import HeliographicStonyhurst

    coords = solar_map.pixel_to_world(np.asarray(x_pix) * u.pix, np.asarray(y_pix) * u.pix)
    heliographic = coords.transform_to(HeliographicStonyhurst(obstime=solar_map.date))

    return (
        np.asarray(heliographic.lon.to_value(u.deg), dtype=np.float64),
        np.asarray(heliographic.lat.to_value(u.deg), dtype=np.float64),
    )


def stonyhurst_to_pixel(
    lon_deg: np.ndarray, lat_deg: np.ndarray, solar_map
) -> tuple[np.ndarray, np.ndarray]:
    """แปลงลองจิจูด/ละติจูด Stonyhurst กลับเป็นพิกัดพิกเซล"""
    import astropy.units as u
    from sunpy.coordinates import HeliographicStonyhurst

    frame = HeliographicStonyhurst(obstime=solar_map.date)
    coords = frame.realize_frame(
        HeliographicStonyhurst(
            lon=np.asarray(lon_deg) * u.deg,
            lat=np.asarray(lat_deg) * u.deg,
            radius=solar_map.rsun_meters,
            obstime=solar_map.date,
        ).data
    )
    x_pix, y_pix = solar_map.world_to_pixel(coords)
    return (
        np.asarray(x_pix.to_value(u.pix), dtype=np.float64),
        np.asarray(y_pix.to_value(u.pix), dtype=np.float64),
    )
