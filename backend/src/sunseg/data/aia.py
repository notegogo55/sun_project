"""ภาพ SDO/AIA สามชั้นบรรยากาศ — ดาวน์โหลด, วางลงกริดของเฟรม HMI, และดึงค่าความเข้มแสง

**ทำไมต้องมี**: magnetogram บอกได้แค่สนามแม่เหล็กที่ผิว (โฟโตสเฟียร์) แต่ active region
แผ่ตัวขึ้นไปทั่วชั้นบรรยากาศ การดูภาพ AIA ที่ตำแหน่งเดียวกันจึงบอกได้ว่าโครงสร้างสนาม
แม่เหล็กนั้นให้ความร้อนกับพลาสมาชั้นบนแค่ไหน

ช่องที่ใช้ (ชั้นละหนึ่ง — ตั้งค่าใน ``configs/data.yaml``):

===========  ==========================  =====================
ช่อง         ชั้นบรรยากาศ                  อุณหภูมิลักษณะเฉพาะ
===========  ==========================  =====================
1600 A       โฟโตสเฟียร์ / transition     ~10,000 K
304 A        โครโมสเฟียร์ (He II)         ~50,000 K
171 A        โคโรนาสงบ (Fe IX)            ~600,000 K
===========  ==========================  =====================

**แหล่งข้อมูล — คลัง synoptic ไม่ใช่คิว export ของ JSOC**

``https://jsoc1.stanford.edu/data/aia/synoptic/YYYY/MM/DD/Hhh00/``

ภาพในคลังนี้เป็น **level 1.5 อยู่แล้ว** (``LVL_NUM=1.5``, ``CROTA2=0``,
``CDELT=2.4 arcsec/px``, 1024x1024 จากการ bin 4x4) แปลว่า ``aiapy.calibrate.register()``
ถูกใช้มาจากต้นทางแล้ว เราจึงไม่ต้องเพิ่ม dependency ``aiapy`` เข้ามาเลย และดึงผ่าน HTTP
ธรรมดาได้โดยไม่ต้องใช้อีเมลที่ลงทะเบียนหรือเข้าคิว export (~1.4 GB เทียบกับ ~40 GB และ
ราว 30 นาทีเทียบกับหลายสิบชั่วโมง)

.. note::
   ไฟล์ถูกบีบอัดแบบ Rice — header จริงอยู่ที่ HDU **1** ไม่ใช่ HDU 0

**หน่วยของค่าที่เก็บ**: DN/s (หารด้วย ``EXPTIME`` แล้ว) จำเป็นเพราะแต่ละช่องใช้เวลาเปิด
หน้ากล้องต่างกัน ถ้าไม่หารตัวเลขข้ามช่องจะไม่มีความหมาย

.. warning::
   **ไม่ได้แก้ instrument degradation** (ความไวของ AIA ลดลงตามปี) ค่าที่ได้จึงเทียบกันได้
   เฉพาะ **ภายในเฟรมเดียวกัน** เท่านั้น ห้ามนำไปเทียบข้ามปี
"""

from __future__ import annotations

import logging
import re
import urllib.request
import warnings
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: ชื่อไฟล์ในคลัง synoptic — ``AIA20141027_0000_0171.fits``
_FILENAME_RE = re.compile(
    r"AIA(?P<date>\d{8})_(?P<time>\d{4})_(?P<wave>\d{4})\.fits", re.IGNORECASE
)

#: สีเส้นขอบ mask บนภาพ AIA (RGB) — ฟ้าสว่าง
#:
#: สีส้มที่ใช้กับ magnetogram มองแทบไม่เห็นบนพื้น 171 (ทอง) และ 304 (แดง) ส่วนฟ้านี้
#: ไม่ปรากฏในตารางสีของช่อง SDO ช่องไหนเลย จึงตัดกับทุกเลเยอร์
MASK_COLOR_AIA = (0, 229, 255)

#: สีเส้นขอบ mask บนภาพ magnetogram (RGB) — ส้ม (ค่าเดิมของโปรเจค)
MASK_COLOR_MAG = (255, 90, 40)


# --------------------------------------------------------------------------- #
# การหาและดาวน์โหลดไฟล์จากคลัง synoptic
# --------------------------------------------------------------------------- #


def tai_to_utc(moment: datetime) -> datetime:
    """แปลงเวลา TAI เป็น UTC

    ชื่อไฟล์เฟรมของโปรเจคมาจาก ``T_REC`` ของ JSOC ซึ่งเป็น **TAI** (``drms.to_datetime``
    ตัด ``_TAI`` ทิ้งโดยไม่แปลงค่า) ส่วนชื่อไฟล์ในคลัง synoptic เป็น **UTC** ถ้าไม่แปลง
    จะพลาดไปราว 35-37 วินาที ซึ่งแม้จะไม่มากแต่ทำให้เลือก slot ผิดช่องได้
    """
    from astropy.time import Time

    return Time(moment, scale="tai").utc.to_datetime()


@lru_cache(maxsize=64)
def _list_hour(
    base_url: str, hour: datetime, timeout_s: float = 60.0
) -> dict[str, tuple[tuple[datetime, str], ...]]:
    """อ่านรายการไฟล์ในไดเรกทอรีของชั่วโมงนั้น จัดกลุ่มตามช่อง

    อ่านรายการจริงแทนการเดาชื่อไฟล์ เพราะคลังมีช่องว่างเป็นระยะ (เช่น 171 มี 54 slot
    ในชั่วโมงที่ช่องอื่นมี 60) การ list ทำให้รองรับช่องว่างได้ฟรี — วิธีเดียวกับ
    ``download_xray.py`` ที่ list รายเดือนแทนการเดาชื่อไฟล์รายวัน

    ผลถูก cache ไว้เพราะหนึ่งไดเรกทอรีมีครบทุกช่องอยู่แล้ว การหาไฟล์ให้ทั้งสามช่องของ
    เฟรมเดียวจึงควรยิงคำขอครั้งเดียว ไม่ใช่สามครั้ง (วัดได้ ~1.3 วินาทีต่อครั้ง)
    คืนค่าเป็น tuple เพราะ ``lru_cache`` ต้องการค่าที่ไม่ถูกแก้ภายหลัง
    """
    url = f"{base_url}/{hour:%Y/%m/%d}/H{hour:%H}00/"
    try:
        html = urllib.request.urlopen(url, timeout=timeout_s).read().decode("utf8", "replace")
    except Exception as exc:  # noqa: BLE001 — ชั่วโมงที่ไม่มีข้อมูลเป็นเรื่องปกติ
        logger.debug("list %s ไม่สำเร็จ: %s", url, exc)
        return {}

    found: dict[str, list[tuple[datetime, str]]] = {}
    for match in _FILENAME_RE.finditer(html):
        stamp = datetime.strptime(match.group("date") + match.group("time"), "%Y%m%d%H%M")
        channel = str(int(match.group("wave")))  # "0171" -> "171"
        found.setdefault(channel, []).append((stamp, url + match.group(0)))

    return {channel: tuple(items) for channel, items in found.items()}


def find_nearest_url(
    base_url: str, moment_utc: datetime, channel: str, timeout_s: float = 60.0
) -> tuple[datetime, str] | None:
    """หา URL ของภาพที่เวลาใกล้ ``moment_utc`` ที่สุดสำหรับช่องที่ระบุ

    ถ้านาทีเป้าหมายอยู่ใกล้ขอบชั่วโมง (< 2 หรือ >= 58) จะ list ชั่วโมงข้างเคียงด้วย
    เพื่อไม่ให้พลาด slot ที่อยู่คนละชั่วโมงแต่ใกล้กว่า
    """
    hours = [moment_utc.replace(minute=0, second=0, microsecond=0)]
    if moment_utc.minute < 2:
        hours.append(hours[0] - timedelta(hours=1))
    elif moment_utc.minute >= 58:
        hours.append(hours[0] + timedelta(hours=1))

    candidates: list[tuple[datetime, str]] = []
    for hour in hours:
        candidates.extend(_list_hour(base_url, hour, timeout_s).get(channel, []))

    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item[0] - moment_utc).total_seconds()))


def download_fits(url: str, out_path: Path, timeout_s: float = 300.0) -> Path:
    """ดาวน์โหลดไฟล์ FITS หนึ่งไฟล์"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        out_path.write_bytes(response.read())
    return out_path


def read_synoptic(path: Path):
    """อ่านไฟล์ synoptic เป็น sunpy Map

    ไฟล์ถูกบีบอัดแบบ Rice — ``sunpy.map.Map`` หา HDU ที่มีข้อมูลเองได้ แต่เราเปิดด้วย
    ``astropy.io.fits`` ตรง ๆ เพื่อให้แน่ใจว่าได้ HDU ที่มีทั้ง data และ header จริง

    .. note::
       ต้องเรียก ``verify("silentfix")`` ก่อนแตะ header — ไฟล์ในคลังนี้มี card ที่ผิด
       รูปแบบอยู่จริง (เช่น ``OSCNMEAN`` ที่เขียนค่า NaN ไว้แบบ parse ไม่ได้) ซึ่งจะทำให้
       astropy โยน ``VerifyError`` ทันทีที่อ่านค่า แม้ WCS ทั้งหมดจะสมบูรณ์ดี
    """
    import sunpy.map
    from astropy.io import fits

    with fits.open(path) as hdul:
        hdul.verify("silentfix")
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None:
            raise RuntimeError(f"ไม่พบ HDU ที่มีข้อมูลใน {path.name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sunpy.map.Map(hdu.data.astype(np.float32), hdu.header)


# --------------------------------------------------------------------------- #
# การประมวลผล
# --------------------------------------------------------------------------- #


def exposure_normalise(data: np.ndarray, exptime: float) -> np.ndarray:
    """แปลง DN เป็น DN/s

    ต้องทำเพราะสามช่องใช้เวลาเปิดหน้ากล้องต่างกัน (วัดจริงได้ 2.0 วินาทีสำหรับ 171 และ
    2.9 วินาทีสำหรับ 1600) ถ้าไม่หาร ค่าที่เทียบข้ามช่องจะสะท้อนเวลาเปิดหน้ากล้องพอ ๆ
    กับความสว่างจริง
    """
    if not np.isfinite(exptime) or exptime <= 0:
        raise ValueError(f"EXPTIME ต้องเป็นค่าบวก ได้รับ {exptime!r}")
    return np.asarray(data, dtype=np.float32) / float(exptime)


def reproject_to_frame(aia_map, target_wcs, target_shape: tuple[int, int]) -> np.ndarray:
    """วางภาพ AIA ลงกริดของเฟรม HMI

    ``reproject_interp`` อ่าน WCS ของทั้งสองภาพแล้วจับคู่พิกัดท้องฟ้าให้เอง จึงจัดการ
    ทั้งมุมหมุน (AIA lev1.5 มี ``CROTA2=0`` ส่วน HMI มี ``CROTA2~180``), สเกลพิกเซลที่
    ต่างกัน (2.4 เทียบกับ 0.504 arcsec/px) และจุดศูนย์กลางจานที่เยื้องกัน ในการ
    interpolate ครั้งเดียว — เหตุผลเดียวกับที่ :func:`~sunseg.data.build_masks.reproject_patch_to_grid`
    เลือก reproject ลงกริดปลายทางโดยตรงแทนการทำสองขั้น
    """
    from reproject import reproject_interp

    with warnings.catch_warnings():
        # พิกเซลนอกขอบเขตเป็นเรื่องปกติ (FOV ของ AIA กว้างกว่ากริดของ HMI)
        warnings.simplefilter("ignore")
        reprojected, _ = reproject_interp(aia_map, target_wcs, shape_out=target_shape)

    return np.asarray(reprojected, dtype=np.float32)


def region_intensities(detections, channel_array: np.ndarray) -> list[dict]:
    """ดึงสถิติความเข้มแสงของแต่ละ AR จากภาพช่องหนึ่ง

    ใช้ ``Detection.mask`` ซึ่งเป็น boolean array **ขนาดเท่าเฟรมทั้งภาพ** จึง index ลง
    ``channel_array`` ได้ตรง ๆ ตราบใดที่ภาพ AIA ถูก reproject ลงกริดเดียวกันแล้ว
    (รูปแบบเดียวกับที่ ``detect_regions`` ใช้ดึงค่าสนามแม่เหล็กจาก magnetogram)

    Returns
    -------
    list ที่เรียงตรงกับ ``detections`` แต่ละตัวเป็น dict ของสถิติ ค่าที่คำนวณไม่ได้
    (ไม่มี mask หรือมีแต่พิกเซล NaN) จะได้ ``n_pixels = 0`` และสถิติเป็น 0.0
    """
    array = np.asarray(channel_array, dtype=np.float32)
    empty = {"mean": 0.0, "median": 0.0, "p95": 0.0, "total": 0.0, "n_pixels": 0}

    results: list[dict] = []
    for detection in detections:
        mask = getattr(detection, "mask", None)
        if mask is None or np.asarray(mask).shape != array.shape:
            results.append(dict(empty))
            continue

        values = array[np.asarray(mask).astype(bool)]
        values = values[np.isfinite(values)]
        if values.size == 0:
            results.append(dict(empty))
            continue

        results.append(
            {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                # p95 ไม่ใช่ max: ภาพ synoptic ไม่ได้ทำ despiking ค่าสูงสุดจึงมักเป็น
                # รังสีคอสมิกที่พุ่งชน CCD ไม่ใช่ความสว่างจริงของ AR
                "p95": float(np.percentile(values, 95)),
                "total": float(values.sum()),
                "n_pixels": int(values.size),
            }
        )

    return results


# --------------------------------------------------------------------------- #
# การเรนเดอร์ภาพ
# --------------------------------------------------------------------------- #


def draw_mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    color: tuple[int, int, int],
    detections=None,
) -> np.ndarray:
    """วาดเส้นขอบ mask (และเลขลำดับ AR) ทับภาพ RGB

    วาดเฉพาะ **เส้นขอบ** ไม่ระบายทึบ เพื่อไม่ให้บังโครงสร้างที่อยู่ข้างใต้ เลขลำดับ
    ทำให้แท่งกราฟ extraction ฝั่งขวาโยงกลับมาที่ AR บนภาพได้ด้วยตา
    """
    import cv2

    if mask is None:
        return rgb

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, color, 1)

    for index, detection in enumerate(detections or [], start=1):
        x = int(round(getattr(detection, "centroid_x", 0)))
        y = int(round(getattr(detection, "centroid_y", 0)))
        cv2.putText(
            rgb,
            str(index),
            (max(x + 6, 0), max(y - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )

    return rgb


def _colormap_for(wavelength: int):
    """ตารางสีมาตรฐานของ SDO สำหรับช่องนั้น"""
    import sunpy.visualization.colormaps  # noqa: F401 — import แล้วสีถูกลงทะเบียนให้ matplotlib
    from matplotlib import colormaps

    return colormaps[f"sdoaia{int(wavelength)}"]


def aia_to_png(
    data: np.ndarray,
    wavelength: int,
    vmax: float,
    mask: np.ndarray | None = None,
    detections=None,
) -> str:
    """เรนเดอร์ภาพ AIA เป็น PNG (เข้ารหัส base64) พร้อมเส้นขอบ mask

    ใช้ ``AsinhStretch`` เพราะช่วงค่าของ AIA กว้างมาก (quiet Sun ราว 100 DN/s ส่วนใจกลาง
    AR เกิน 3,000) การแสดงแบบเชิงเส้นจะเห็นแต่จุดสว่างไม่กี่จุดบนพื้นดำ

    ``vmax`` เป็น **ค่าคงที่ต่อช่อง** จาก config ไม่ใช่ percentile ของภาพนั้น ๆ — ถ้าใช้
    percentile รายภาพ การสลับเฟรมจะเปลี่ยนความสว่างไปเงียบ ๆ จนแยกไม่ออกว่าอะไรคือ
    ปรากฏการณ์จริง อะไรคือผลของการ normalise
    """
    import base64
    import io

    from astropy.visualization import AsinhStretch, ImageNormalize
    from PIL import Image

    array = np.nan_to_num(np.asarray(data, dtype=np.float32), nan=0.0)
    norm = ImageNormalize(vmin=0.0, vmax=float(vmax), stretch=AsinhStretch(0.02), clip=True)
    rgb = (_colormap_for(wavelength)(norm(array))[..., :3] * 255).astype(np.uint8)

    rgb = draw_mask_overlay(rgb, mask, MASK_COLOR_AIA, detections)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# คลังภาพที่ประมวลผลแล้ว
# --------------------------------------------------------------------------- #


class AiaFrameStore:
    """ภาพ AIA ที่ reproject ลงกริดของเฟรมแล้ว เก็บที่ ``<root>/<ช่อง>/<เวลา>.npy``

    ไม่มีไฟล์ = ไม่พัง: ``available`` เป็น False แล้วหน้าเว็บแสดงเฉพาะเลเยอร์ magnetogram
    พร้อมบอกว่าต้องรันสคริปต์ไหน — เหมือนส่วนอื่นของแอปที่ต้อง degrade ได้

    ข้อมูลอาจมีเพียง **บางเฟรม** ด้วย (สคริปต์ดาวน์โหลดรันต่อได้ทีละส่วน) ความพร้อมจึง
    ต้องเช็คเป็นราย ๆ เฟรมผ่าน :meth:`channels_for` ไม่ใช่แค่ระดับคลัง
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._available = self.root.is_dir() and any(self.root.rglob("*.npy"))

        if self._available:
            logger.info("พบภาพ AIA ที่ %s (%d เฟรม)", self.root, self.n_frames)
        else:
            logger.warning(
                "ไม่พบภาพ AIA ที่ %s — หน้าเว็บจะมีเฉพาะเลเยอร์ magnetogram "
                "(รัน backend/scripts/data/download_aia.py เพื่อดึงภาพสามชั้นบรรยากาศ)",
                self.root,
            )

    @property
    def available(self) -> bool:
        return self._available

    @property
    def channels(self) -> list[str]:
        """ช่องที่มีโฟลเดอร์อยู่จริงบนดิสก์"""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    @property
    def n_frames(self) -> int:
        """จำนวนเวลาที่มีภาพอย่างน้อยหนึ่งช่อง"""
        if not self.root.is_dir():
            return 0
        return len({p.stem for p in self.root.rglob("*.npy")})

    def channels_for(self, timestamp: str) -> list[str]:
        """ช่องที่เฟรมนี้มีข้อมูลจริง"""
        if not self.root.is_dir():
            return []
        return sorted(
            channel
            for channel in self.channels
            if (self.root / channel / f"{timestamp}.npy").exists()
        )

    def path_for(self, timestamp: str, channel: str) -> Path:
        return self.root / str(channel) / f"{timestamp}.npy"

    def load(self, timestamp: str, channel: str) -> np.ndarray | None:
        """โหลดภาพช่องหนึ่งของเฟรมหนึ่ง (DN/s) — ``None`` ถ้าไม่มี"""
        path = self.path_for(timestamp, channel)
        if not path.exists():
            return None
        return np.load(path).astype(np.float32)

    def save(self, timestamp: str, channel: str, data: np.ndarray) -> Path:
        """บันทึกภาพที่ประมวลผลแล้ว

        เก็บเป็น **float32 ไม่ใช่ float16** ต่างจากธรรมเนียมของ ``save_frame`` โดยตั้งใจ:
        ตอนเกิด flare ระบบ AEC ของ AIA ลดเวลาเปิดหน้ากล้องเหลือราว 0.1 วินาที ทำให้ค่า
        DN/s พุ่งทะลุเพดานของ float16 (65504) แล้วกลายเป็น ``inf`` แบบเงียบ ๆ
        """
        path = self.path_for(timestamp, channel)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(data, dtype=np.float32))
        return path

    def info(self) -> dict:
        return {
            "available": self._available,
            "root": str(self.root),
            "channels": self.channels,
            "n_frames": self.n_frames,
        }
