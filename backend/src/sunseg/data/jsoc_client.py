"""Wrapper รอบไลบรารี ``drms`` สำหรับดึงข้อมูล SDO/HMI จาก JSOC.

เพิ่มสิ่งที่ drms ดิบไม่มีให้ 3 อย่าง ซึ่งจำเป็นเมื่อต้องดึงข้อมูลหลายปี:

1. **แบ่ง query เป็นช่วงย่อย** — การขอข้อมูลทีเดียวหลายปีทำให้ JSOC timeout
2. **retry พร้อม exponential backoff** — JSOC ล่มหรือ throttle เป็นเรื่องปกติ
3. **cache ระดับ chunk** — ดาวน์โหลดต่อจากที่ค้างไว้ได้ ไม่ต้องเริ่มใหม่ทั้งหมด
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# รูปแบบเวลาที่ DRMS record-set notation ใช้: 2011.01.01_00:00:00_TAI
_DRMS_TIME_FMT = "%Y.%m.%d_%H:%M:%S_TAI"


def to_drms_time(dt: datetime | date) -> str:
    """แปลง datetime เป็นสตริงเวลาแบบที่ DRMS เข้าใจ"""
    if isinstance(dt, datetime):
        return dt.strftime(_DRMS_TIME_FMT)
    return datetime(dt.year, dt.month, dt.day).strftime(_DRMS_TIME_FMT)


def iter_time_chunks(
    start: date, end: date, chunk_days: int
) -> Iterator[tuple[datetime, datetime]]:
    """แบ่งช่วงเวลาเป็นก้อนย่อย ๆ ก้อนละ ``chunk_days`` วัน (ปลายเปิดที่ขวา)"""
    if chunk_days <= 0:
        raise ValueError(f"chunk_days ต้องเป็นบวก ได้รับ {chunk_days}")

    cursor = datetime(start.year, start.month, start.day)
    stop = datetime(end.year, end.month, end.day)
    while cursor < stop:
        chunk_end = min(cursor + timedelta(days=chunk_days), stop)
        yield cursor, chunk_end
        cursor = chunk_end


def _is_pending_conflict(exc: BaseException) -> bool:
    """แยกแยะ error "มีคำขอ export ค้างอยู่" (status=7) ของ JSOC

    ต้องไล่ดูทั้งสาย ``__cause__`` เพราะ :meth:`JsocClient._with_retry` ห่อ
    exception ต้นทางไว้ใน ``RuntimeError`` ข้อความจริงจึงไม่อยู่ในตัวบนสุด
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if "pending export request" in str(current).lower():
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class JsocClient:
    """ดึง keyword และ segment จาก JSOC พร้อม retry

    Parameters
    ----------
    email
        อีเมลที่ลงทะเบียนกับ JSOC — จำเป็นเฉพาะตอน export segment
        (การ query keyword อย่างเดียวไม่ต้องใช้)
    """

    def __init__(
        self,
        email: str | None = None,
        max_retries: int = 4,
        retry_backoff_s: float = 5.0,
    ) -> None:
        import drms  # import ในนี้เพื่อให้ import โมดูลนี้ได้แม้ยังไม่ได้ติดตั้ง drms

        self._drms = drms
        self.email = email
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.client = drms.Client(email=email) if email else drms.Client()

    # ------------------------------------------------------------------ #
    # retry helper
    # ------------------------------------------------------------------ #

    def _with_retry(self, description: str, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — drms โยน exception ได้หลากหลายชนิด
                last_exc = exc
                if attempt == self.max_retries:
                    break
                delay = self.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "%s ล้มเหลว (ครั้งที่ %d/%d): %s — จะลองใหม่ใน %.1f วินาที",
                    description,
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{description} ล้มเหลวหลังพยายาม {self.max_retries} ครั้ง"
        ) from last_exc

    def _submit_export(
        self,
        query: str,
        method: str,
        protocol: str,
        wait_s: float = 60.0,
        max_waits: int = 6,
    ):
        """สั่ง export โดยรอให้คิวของผู้ใช้ว่างก่อนถ้าจำเป็น

        JSOC อนุญาตให้มีคำขอ export ค้างได้ **คำขอเดียวต่อผู้ใช้** ถ้าการเชื่อมต่อ
        หลุดระหว่างรอคำขอก่อนหน้า (เน็ตสะดุด, WinError 10060) คำขอนั้นยังค้างอยู่
        ฝั่ง JSOC ส่วนฝั่งเราทิ้งไปแล้ว ผลคือคำขอถัดไป *ทุกอัน* ถูกปฏิเสธทันที
        ด้วย ``status=7`` และเฟรมที่เหลือทั้งหมดจะล้มเหลวรวดเดียวในไม่กี่วินาที
        (เคยเกิดจริง: 5 เฟรมตายใน 90 วินาที) การรอให้คำขอค้างนั้นเสร็จเองแล้วค่อย
        ส่งใหม่จึงถูกกว่าการปล่อยให้ล้มทั้งชุด

        .. note::
           ข้อความ error บอก request ID ของคำขอที่ค้างมาด้วย และ ``drms`` มี
           ``export_from_id()`` ให้เกาะคำขอเดิมได้ — **ห้ามใช้ที่นี่** เพราะคำขอนั้น
           เป็นของ *เฟรมก่อนหน้า* คนละ record กับที่กำลังขอ ถ้าเกาะไปจะได้ FITS
           ของเวลาอื่นมาเงียบ ๆ แล้วถูกบันทึกภายใต้ tag ของเฟรมนี้
        """
        for attempt in range(1, max_waits + 1):
            try:
                return self._with_retry(
                    f"export {query}",
                    self.client.export,
                    query,
                    method=method,
                    protocol=protocol,
                    email=self.email,
                )
            except Exception as exc:  # noqa: BLE001 — drms โยน exception ได้หลากหลายชนิด
                if not _is_pending_conflict(exc) or attempt == max_waits:
                    raise
                logger.warning(
                    "JSOC ยังมีคำขอ export ค้างอยู่ — รอ %.0f วินาทีแล้วลองใหม่ (%d/%d)",
                    wait_s,
                    attempt,
                    max_waits,
                )
                time.sleep(wait_s)

    # ------------------------------------------------------------------ #
    # keyword queries
    # ------------------------------------------------------------------ #

    def query_keywords(self, recordset: str, keys: list[str]) -> pd.DataFrame:
        """ขอเฉพาะ keyword (metadata) — เร็วและเบา ไม่แตะไฟล์ภาพ"""
        key_str = ",".join(keys)
        df = self._with_retry(
            f"query keyword ของ {recordset}",
            self.client.query,
            recordset,
            key=key_str,
        )
        return pd.DataFrame() if df is None else df

    def query_sharp_series(
        self,
        series: str,
        start: date,
        end: date,
        keys: list[str],
        cadence_hours: int = 1,
        chunk_days: int = 30,
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """ดึง keyword ของ SHARP ทุก HARP ในช่วงเวลาที่กำหนด

        ผลลัพธ์ของแต่ละ chunk ถูก cache เป็น parquet เพื่อให้รันซ้ำแล้วทำต่อจากเดิมได้
        """
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        frames: list[pd.DataFrame] = []
        chunks = list(iter_time_chunks(start, end, chunk_days))

        for idx, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            tag = chunk_start.strftime("%Y%m%d")
            cache_file = cache_dir / f"sharp_keys_{tag}.parquet" if cache_dir else None

            if cache_file is not None and cache_file.exists():
                logger.info("[%d/%d] ใช้ cache: %s", idx, len(chunks), cache_file.name)
                frames.append(pd.read_parquet(cache_file))
                continue

            # `[]` ตัวแรกคือ HARPNUM = ทุกดวง, ตัวที่สองคือช่วง T_REC พร้อม cadence
            recordset = (
                f"{series}[][{to_drms_time(chunk_start)}-"
                f"{to_drms_time(chunk_end)}@{cadence_hours}h]"
            )
            logger.info("[%d/%d] กำลัง query %s", idx, len(chunks), recordset)

            df = self.query_keywords(recordset, keys)
            if df.empty:
                logger.warning("ไม่พบข้อมูลในช่วง %s ถึง %s", chunk_start, chunk_end)
                continue

            if cache_file is not None:
                df.to_parquet(cache_file, index=False)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=keys)
        return pd.concat(frames, ignore_index=True)

    def query_fulldisk_times(
        self,
        series: str,
        start: date,
        end: date,
        cadence_hours: int = 12,
        chunk_days: int = 90,
    ) -> pd.DataFrame:
        """ดึงรายการเวลา (และ QUALITY) ของภาพ full-disk ที่มีอยู่จริง"""
        frames: list[pd.DataFrame] = []
        for chunk_start, chunk_end in iter_time_chunks(start, end, chunk_days):
            recordset = (
                f"{series}[{to_drms_time(chunk_start)}-"
                f"{to_drms_time(chunk_end)}@{cadence_hours}h]"
            )
            logger.info("กำลัง query เวลาของ full-disk: %s", recordset)
            df = self.query_keywords(recordset, ["T_REC", "QUALITY", "CRLN_OBS", "CRLT_OBS"])
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["T_REC", "QUALITY", "CRLN_OBS", "CRLT_OBS"])
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    # segment export (ต้องมีอีเมลที่ลงทะเบียนแล้ว)
    # ------------------------------------------------------------------ #

    def export_segments(
        self,
        recordset: str,
        segments: list[str],
        out_dir: Path,
        method: str = "url",
        protocol: str = "fits",
        timeout_s: int = 900,
        read_timeout_s: int = 300,
    ) -> pd.DataFrame:
        """สั่ง export แล้วดาวน์โหลดไฟล์ FITS ของ segment ที่ระบุ

        **ต้องใช้ ``url``/``fits`` เท่านั้น — ห้ามเปลี่ยนกลับไป ``url_quick``/``as-is``**
        JSOC เก็บ keyword ไว้ในฐานข้อมูล DRMS ไม่ได้เก็บใน FITS header ไฟล์ segment
        ดิบบนดิสก์ที่ ``as-is`` ส่งกลับมาจึงมี header เปล่า (ไม่มี WCS เลยสักตัว) และ
        ``sunpy.map.Map`` จะล้มด้วย "Image coordinate units for axis 1 not present in
        metadata" ส่วน ``protocol="fits"`` สั่งให้ JSOC สร้างไฟล์ใหม่พร้อมเขียน keyword
        ลง header ให้ (~110 ตัว รวม CTYPE/CUNIT/CRVAL/CDELT/CRPIX/CROTA2/RSUN_OBS/DSUN_OBS)
        ซึ่งจำเป็นทั้งกับภาพเต็มดวงและ SHARP patch เพราะ mask สร้างจากการแปลงพิกัดผ่าน WCS

        แลกมาด้วยการเข้าคิว export ของ JSOC (~20-60 วินาทีต่อคำขอ แทนที่จะได้ทันที)

        Parameters
        ----------
        timeout_s
            เวลารอสูงสุดให้คำขอ export เสร็จคิว
        read_timeout_s
            timeout ของการอ่าน socket ตอนดาวน์โหลดไฟล์ ค่าปริยายของ ``drms`` คือ 60
            วินาที ซึ่งสั้นเกินไปสำหรับภาพเต็มดวง (~30-60 MB): ถ้าฝั่ง JSOC หยุดส่ง
            ข้อมูลนานกว่านั้นระหว่างทาง urllib จะโยน "The read operation timed out"
            ทั้งที่การเชื่อมต่อยังดีอยู่ ทำให้เฟรมนั้นถูกข้ามทิ้งทั้งที่แค่เซิร์ฟเวอร์ช้า

        Returns
        -------
        DataFrame ที่มีคอลัมน์ ``record`` และ ``download`` (path ในเครื่อง)
        """
        if not self.email:
            raise RuntimeError(
                "การ export ต้องใช้อีเมลที่ลงทะเบียนกับ JSOC — "
                "ตั้งค่า SUNSEG_JSOC_EMAIL ใน .env"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        query = f"{recordset}{{{','.join(segments)}}}"

        logger.info("กำลังขอ export: %s", query)
        request = self._submit_export(query, method, protocol)

        # การรอสถานะก็ต้อง retry ด้วย: ถ้าเน็ตสะดุดตอน poll แล้วเราปล่อยเฟรมนี้ทิ้ง
        # คำขอจะยังค้างอยู่ฝั่ง JSOC และไปบล็อกทุกเฟรมถัดไป (ดู _submit_export)
        if hasattr(request, "wait"):
            self._with_retry(f"wait {query}", request.wait, timeout=timeout_s)

        if getattr(request, "status", 0) != 0:
            raise RuntimeError(
                f"JSOC ปฏิเสธคำขอ export ({query}): status={request.status}"
            )

        result = self._with_retry(
            f"download {query}", request.download, str(out_dir), timeout=read_timeout_s
        )
        logger.info("ดาวน์โหลดสำเร็จ %d ไฟล์ไปที่ %s", len(result), out_dir)
        return result

    def parse_trec(self, series: pd.Series) -> pd.Series:
        """แปลงคอลัมน์ T_REC (สตริง TAI) เป็น pandas datetime"""
        return self._drms.to_datetime(series)
