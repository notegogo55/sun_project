"""WCS จริงของภาพเต็มดวงแต่ละเฟรม — ดึงจาก JSOC แล้ว cache เป็น parquet

**ทำไมต้องมีโมดูลนี้**: ``download_images.py`` เก็บเฟรมเป็น ``.npy`` เปล่า ๆ แล้วลบ FITS
ทิ้งเพื่อประหยัดดิสก์ ผลคือ **ไม่มี WCS ติดมากับเฟรมเลย** ซึ่งไม่เป็นไรตอนเทรน U-Net
(โมเดลมองแค่พิกเซล) แต่พอต้องวางภาพจากเครื่องมืออื่น (AIA) ทับลงกริดเดียวกัน หรือแปลง
พิกเซลเป็นพิกัดเฮลิโอกราฟิก เราต้องรู้ WCS ที่ถูกต้องของเฟรมนั้น

**ทำไมไม่สร้าง WCS ประมาณเอา**: เคยทำแบบนั้นแล้วผิด — ค่าจริงของ ``hmi.M_720s`` คือ

* ``CROTA2 ~ 180.01`` (ไม่ใช่ 0) — ภาพ HMI ถูกเก็บในทิศ CCD ดิบซึ่ง **กลับหัว**
  จากทิศที่คนดูคุ้นเคย การละเลยค่านี้ทำให้พิกัดทุกจุดผิดไป 180 องศา
* ``CRPIX ~ (2037.0, 2049.6)`` (ไม่ใช่ 2048.5) — จุดศูนย์กลางจานเยื้องไปราว 12 พิกเซล
* ``CDELT ~ 0.5044 arcsec/px`` ซึ่งต่างจากค่าประมาณจากรัศมี 976 arcsec ราว 1.85%

**ทำไมถูกกว่าการโหลด FITS ใหม่**: การ query *keyword* ของ DRMS ไม่ต้องใช้อีเมลที่
ลงทะเบียน ไม่เข้าคิว export และเร็วราว 1 วินาทีต่อเฟรม — ต่างจากการ export segment
ที่กินเวลาหลักนาทีต่อเฟรมและได้ไฟล์ 30 MB ที่เราจะลบทิ้งอยู่ดี
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: keyword ที่ต้องมีเพื่อประกอบ WCS ขึ้นใหม่ให้ตรงกับ FITS ต้นฉบับทุกประการ
#:
#: .. warning::
#:    **ห้ามใส่ HGLN_OBS** — ``hmi.M_720s`` ไม่มี keyword นี้ และ DRMS จะคืนค่า
#:    ``Invalid KeyLink`` มาทั้งคอลัมน์ ตำแหน่งผู้สังเกตใช้ ``CRLN_OBS``/``CRLT_OBS``
#:    คู่กับ ``DSUN_OBS`` แทนได้ครบอยู่แล้ว
WCS_KEYS: tuple[str, ...] = (
    "T_REC",
    "DATE__OBS",
    "CRPIX1",
    "CRPIX2",
    "CDELT1",
    "CDELT2",
    "CRVAL1",
    "CRVAL2",
    "CROTA2",
    "CRLN_OBS",
    "CRLT_OBS",
    "DSUN_OBS",
    "RSUN_OBS",
    "RSUN_REF",
    "CTYPE1",
    "CTYPE2",
    "CUNIT1",
    "CUNIT2",
    "QUALITY",
)

#: ขนาดภาพเต็มดวงต้นฉบับของ HMI — WCS ที่ดึงมาอ้างอิงกริดนี้ ก่อนถูก bin ลงเป็น 512
FULLDISK_SIZE = 4096

#: keyword เพิ่มจาก WCS_KEYS ที่ต้องใช้ระบุตัวตนของ SHARP patch (ไม่เกี่ยวกับ WCS
#: โดยตรง แต่ ``build_fulldisk_identity_map`` ต้องมี ``harpnum`` ใน ``patch_map.meta``
#: เพื่อบอกว่าแต่ละ patch เป็นของ HARP ไหน)
SHARP_EXTRA_KEYS: tuple[str, ...] = ("HARPNUM", "NOAA_AR", "NOAA_ARS")


def frame_stem(moment: datetime) -> str:
    """ชื่อไฟล์เฟรมมาตรฐานของโปรเจค (ตรงกับ ``save_frame``)"""
    return moment.strftime("%Y%m%d_%H%M%S")


def _build_header(row: pd.Series, naxis1: int, naxis2: int) -> dict:
    """ประกอบ FITS header ขั้นต่ำที่ astropy/sunpy ต้องใช้สร้าง WCS ของภาพขนาด
    ``naxis1`` x ``naxis2`` จาก keyword row เดียว (ภาพเต็มดวงหรือ SHARP patch ก็ได้
    — คอลัมน์ที่ใช้เหมือนกันทุกตัว มีแค่ขนาดภาพที่ต่างกัน)
    """
    header = {
        "naxis": 2,
        "naxis1": naxis1,
        "naxis2": naxis2,
        "ctype1": str(row["CTYPE1"]),
        "ctype2": str(row["CTYPE2"]),
        "cunit1": str(row["CUNIT1"]),
        "cunit2": str(row["CUNIT2"]),
        "crpix1": float(row["CRPIX1"]),
        "crpix2": float(row["CRPIX2"]),
        "crval1": float(row["CRVAL1"]),
        "crval2": float(row["CRVAL2"]),
        "cdelt1": float(row["CDELT1"]),
        "cdelt2": float(row["CDELT2"]),
        "crota2": float(row["CROTA2"]),
        "dsun_obs": float(row["DSUN_OBS"]),
        "rsun_obs": float(row["RSUN_OBS"]),
        "rsun_ref": float(row["RSUN_REF"]),
        "crln_obs": float(row["CRLN_OBS"]),
        "crlt_obs": float(row["CRLT_OBS"]),
        # sunpy ใช้ HGLT_OBS หาตำแหน่งผู้สังเกต ถ้าไม่ระบุจะถอยไปใช้ ephemeris ของโลก
        # ซึ่งคลาดจากตำแหน่งดาวเทียมจริงเล็กน้อย — ระบุตรง ๆ จาก CRLT_OBS ดีกว่า
        "hglt_obs": float(row["CRLT_OBS"]),
        "date-obs": str(row["DATE__OBS"]),
        "telescop": "SDO/HMI",
        "instrume": "HMI",
    }
    # HARPNUM มีเฉพาะ keyword row ของ SHARP (ไม่มีในภาพเต็มดวง) —
    # build_fulldisk_identity_map อ่านค่านี้จาก patch_map.meta ต้องใส่เมื่อมี
    if "HARPNUM" in row.index and pd.notna(row["HARPNUM"]):
        header["harpnum"] = int(row["HARPNUM"])
    return header


def _header_from_row(row: pd.Series, size: int = FULLDISK_SIZE) -> dict:
    """ประกอบ FITS header ขั้นต่ำที่ astropy ต้องใช้สร้าง WCS ของกริด ``size``"""
    return _build_header(row, size, size)


def header_for_shape(row: pd.Series, naxis1: int, naxis2: int) -> dict:
    """เหมือน :func:`_header_from_row` แต่ระบุขนาดภาพตรง ๆ แทนกริดสี่เหลี่ยมจัตุรัส —
    ใช้กับ SHARP patch ที่แต่ละดวงมีขนาดไม่เท่ากัน (ต่างจากภาพเต็มดวงที่ตายตัว 4096)
    """
    return _build_header(row, naxis1, naxis2)


class FrameWcsStore:
    """ดัชนี WCS ของทุกเฟรม อ่านจาก ``data/interim/frame_wcs.parquet``

    ไม่มีไฟล์ = ไม่พัง: ``available`` เป็น False แล้วผู้เรียกถอยไปใช้ WCS ประมาณแทน
    (ซึ่งแม่นน้อยกว่าแต่ยังใช้ติดตาม AR แบบคร่าว ๆ ได้) เหมือนส่วนอื่นของแอปที่ต้อง
    degrade ได้เมื่อยังไม่ได้รันสคริปต์ดาวน์โหลด
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._table: pd.DataFrame | None = None

        if not self.path.exists():
            logger.warning(
                "ไม่พบ WCS ของเฟรมที่ %s — การวางภาพ AIA และพิกัดเฮลิโอกราฟิกจะใช้ค่าประมาณ "
                "(รัน backend/scripts/data/download_aia.py --wcs-only เพื่อดึงค่าจริง)",
                self.path,
            )
            return

        try:
            table = pd.read_parquet(self.path)
            self._table = table.set_index("frame") if "frame" in table.columns else table
            logger.info("โหลด WCS ของเฟรม %d รายการจาก %s", len(self._table), self.path.name)
        except Exception as exc:  # noqa: BLE001 — ไฟล์เสียไม่ควรทำให้แอปเปิดไม่ได้
            logger.error("อ่าน %s ไม่สำเร็จ: %s", self.path, exc)

    @property
    def available(self) -> bool:
        return self._table is not None and not self._table.empty

    def __contains__(self, timestamp: str) -> bool:
        return self.available and timestamp in self._table.index

    def info(self) -> dict:
        return {
            "available": self.available,
            "path": str(self.path),
            "n_frames": 0 if self._table is None else int(len(self._table)),
        }

    # ------------------------------------------------------------------ #

    def header(self, timestamp: str, size: int = FULLDISK_SIZE) -> dict | None:
        """FITS header ของเฟรมนั้นบนกริด ``size`` — ``None`` ถ้าไม่มีข้อมูล"""
        if not self.available or timestamp not in self._table.index:
            return None
        return _header_from_row(self._table.loc[timestamp], size=size)

    def wcs(self, timestamp: str, size: int = FULLDISK_SIZE):
        """``astropy.wcs.WCS`` ของเฟรมนั้นบนกริด ``size`` — ``None`` ถ้าไม่มีข้อมูล"""
        from astropy.wcs import WCS

        header = self.header(timestamp, size=size)
        return None if header is None else WCS(header)

    def binned_wcs(self, timestamp: str, target_size: int):
        """WCS ของกริดที่ย่อแล้ว (เช่น 512) — ``None`` ถ้าไม่มีข้อมูล

        WCS ที่ดึงมาอ้างอิงกริด 4096 ส่วนเฟรมที่เก็บไว้ถูกย่อแล้ว จึงต้อง bin ลงให้
        เข้ากันด้วย :func:`~sunseg.data.build_masks.binned_wcs_from`
        """
        from .build_masks import binned_wcs_from

        wcs_full = self.wcs(timestamp, size=FULLDISK_SIZE)
        if wcs_full is None:
            return None
        wcs_small, _ = binned_wcs_from(
            wcs_full, (FULLDISK_SIZE, FULLDISK_SIZE), int(target_size)
        )
        return wcs_small

    def solar_map(self, timestamp: str, data):
        """สร้าง ``sunpy.map.Map`` จากอาร์เรย์ของเฟรม โดยผูก WCS จริงเข้าไปด้วย"""
        import sunpy.map

        wcs_small = self.binned_wcs(timestamp, int(data.shape[0]))
        if wcs_small is None:
            return None
        return sunpy.map.Map(data, wcs_small.to_header())


def map_from_as_is(path: Path, row: pd.Series):
    """สร้าง ``sunpy.map.Map`` จากไฟล์ FITS แบบ ``as-is`` (header เปล่า ไม่มี WCS)
    โดยต่อ WCS เข้าไปเองจาก ``row`` keyword ที่ query แยกมาต่างหาก (ดู
    :func:`fetch_frame_wcs` / :func:`fetch_sharp_wcs`)

    ใช้แทนการเปิดไฟล์ที่ export ด้วย ``protocol="fits"`` ตรง ๆ — ค่าพิกเซลเหมือนกัน
    ทุกประการ (คนละวิธีดึงข้อมูลเดียวกันจาก JSOC) แต่ ``as-is``/``url_quick`` ไม่ต้อง
    เข้าคิว export (ดู ``JsocClient.export_fast``)
    """
    from astropy.io import fits
    import sunpy.map

    with fits.open(path) as hdul:
        hdu = hdul[1] if len(hdul) > 1 else hdul[0]
        data = hdu.data.astype(float)

    header = _build_header(row, naxis1=data.shape[1], naxis2=data.shape[0])
    return sunpy.map.Map(data, header)


# --------------------------------------------------------------------------- #
# การดึงข้อมูลมาเติมดัชนี (ใช้จาก scripts/data/download_aia.py)
# --------------------------------------------------------------------------- #


def fetch_frame_wcs(client, series: str, timestamps: list[str]) -> pd.DataFrame:
    """ดึง keyword WCS ของเฟรมที่ระบุ (ชื่อแบบ ``20141027_000000``)"""
    from .jsoc_client import to_drms_time

    if not timestamps:
        return pd.DataFrame(columns=["frame", *WCS_KEYS])

    rows: list[pd.DataFrame] = []
    for index, stem in enumerate(timestamps, start=1):
        moment = datetime.strptime(stem, "%Y%m%d_%H%M%S")
        recordset = f"{series}[{to_drms_time(moment)}]"
        try:
            df = client.query_keywords(recordset, list(WCS_KEYS))
        except Exception as exc:  # noqa: BLE001 — เฟรมเดียวพังไม่ควรหยุดทั้งงาน
            logger.warning(
                "[%d/%d] ดึง WCS ของ %s ไม่สำเร็จ: %s", index, len(timestamps), stem, exc
            )
            continue

        if df.empty:
            logger.warning("[%d/%d] ไม่พบ record ของ %s", index, len(timestamps), stem)
            continue

        row = df.head(1).copy()
        row.insert(0, "frame", stem)
        rows.append(row)

        if index % 25 == 0 or index == len(timestamps):
            logger.info("ดึง WCS แล้ว %d/%d เฟรม", index, len(timestamps))

    if not rows:
        return pd.DataFrame(columns=["frame", *WCS_KEYS])
    return pd.concat(rows, ignore_index=True)


def fetch_sharp_wcs(client, series: str, moment: datetime) -> pd.DataFrame:
    """ดึง keyword WCS ของ **ทุก HARP** ณ เวลาเดียว ในคำขอเดียว (bulk keyword query
    ไม่เข้าคิว export เหมือน :func:`fetch_frame_wcs`) — คืน ``DataFrame`` ที่มี
    คอลัมน์ ``HARPNUM`` ให้จับคู่กับไฟล์ ``bitmap`` ที่ export แบบ ``as-is`` มา

    เว้นระยะจาก :func:`fetch_frame_wcs` ที่ดึงทีละเฟรม (ภาพเต็มดวงมีระเบียนเดียวต่อ
    เวลา) เพราะ SHARP หนึ่งเวลามีได้หลาย HARP — DRMS recordset ``[][time]`` (วงเล็บ
    ว่างตัวแรก = ทุก HARPNUM) คืนมาเป็นหลายแถวในคำเดียว
    """
    from .jsoc_client import to_drms_time

    recordset = f"{series}[][{to_drms_time(moment)}]"
    return client.query_keywords(recordset, [*WCS_KEYS, *SHARP_EXTRA_KEYS])


def merge_and_save(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    """รวมกับที่มีอยู่เดิมแล้วบันทึก — ทำให้สคริปต์รันซ้ำได้โดยไม่ดึงของเก่าใหม่"""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, fresh], ignore_index=True)
        combined = combined.drop_duplicates(subset="frame", keep="last")
    else:
        combined = fresh

    combined = combined.sort_values("frame").reset_index(drop=True)
    combined.to_parquet(path, index=False)
    logger.info("บันทึก WCS ของเฟรม %d รายการที่ %s", len(combined), path)
    return combined
