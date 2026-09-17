"""Wrapper รอบไลบรารี ``drms`` สำหรับดึงข้อมูล SDO/HMI จาก JSOC.

เพิ่มสิ่งที่ drms ดิบไม่มีให้ 3 อย่าง ซึ่งจำเป็นเมื่อต้องดึงข้อมูลหลายปี:

1. **แบ่ง query เป็นช่วงย่อย** — การขอข้อมูลทีเดียวหลายปีทำให้ JSOC timeout
2. **retry พร้อม exponential backoff** — JSOC ล่มหรือ throttle เป็นเรื่องปกติ
3. **cache ระดับ chunk** — ดาวน์โหลดต่อจากที่ค้างไว้ได้ ไม่ต้องเริ่มใหม่ทั้งหมด
"""

from __future__ import annotations

import logging
import re
import socket
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# drms/urllib ไม่ตั้ง timeout ให้ socket โดย default (ไม่มีขีดจำกัดเลย) — พบจริงว่า
# connection ที่ client ใช้ซ้ำมานาน (thousands of request ต่อ process เดียวตลอดงาน
# ดาวน์โหลดหลายวัน) บางครั้งค้างเงียบได้หลายนาทีโดยไม่มี error ให้ _with_retry จับ เลย
# ไม่ retry เลยสักครั้ง (แยกไปเปิด connection ใหม่ทดสอบพร้อมกันตอนที่ค้างอยู่ ได้ผลตอบ
# กลับใน 1-2 วิ ยืนยันว่าฝั่ง JSOC ไม่ได้ช้า connection เดิมของ process นี้ต่างหากที่ค้าง)
# ตั้ง default timeout ระดับ process ไว้กันไม่ให้ค้างไม่มีที่สิ้นสุด — หมดเวลาแล้วจะโยน
# ``socket.timeout`` ซึ่ง ``_with_retry`` จับได้และลองใหม่ (เปิด connection ใหม่ ซึ่งพิสูจน์
# แล้วว่าเร็วปกติ) ตั้งไว้สูง (90 วิ) เพราะคำขอ export ปกติก็ใช้เวลาได้ถึงหลักสิบวิอยู่แล้ว
socket.setdefaulttimeout(90.0)

# หมายเหตุ: ``socket.setdefaulttimeout`` ครอบเฉพาะ operation หลัง socket ถูกสร้างแล้ว
# (connect/send/recv) — ``socket.getaddrinfo`` (DNS resolve) ที่ ``urllib`` เรียกก่อนสร้าง
# socket ไม่ถูกจำกัดเวลาด้วยค่านี้เลย พบจริงว่า process ค้างเงียบ "ไม่มี log อะไรเลย" นาน
# กว่า 90 วิ (ไม่มี WARNING จาก _with_retry ให้เห็นด้วยซ้ำ) แม้เพิ่ง start ใหม่ ยังไม่ทันมี
# connection เก่าให้ reuse — เข้าเงื่อนไข DNS hang ระดับ OS มากกว่า socket hang ธรรมดา จึง
# ต้องมี hard timeout ระดับ thread ครอบอีกชั้น (ดู ``_HARD_CALL_TIMEOUT_S`` /
# ``JsocClient._call_with_hard_timeout``) เพราะ getaddrinfo ที่ค้างจริงจะค้างไม่มีกำหนด
_HARD_CALL_TIMEOUT_S = 110.0

# คำขอที่โอนข้อมูลจริง (``request.download``) ต้องให้เวลามากกว่า metadata/keyword query
# ธรรมดา — พบจริงว่าตอนเน็ตแย่ (packet loss สูงไปเซิร์ฟเวอร์ที่ Stanford, RTT ~260ms) ไฟล์
# magnetogram ~14MB ใช้เวลาโอนจริงได้หลายนาที (เคยเจอ 6-7 นาที) ถ้าตั้ง timeout สั้นแบบ
# keyword query (110 วิ) จะ abandon thread ที่กำลังโอนข้อมูลอยู่จริงทั้งที่ใกล้เสร็จ — เสียของ
# เปล่า (thread ที่ถูกทิ้งจะโอนต่อจนเสร็จเองในพื้นหลัง แต่ผลลัพธ์มาช้าเกินจะใช้ เพราะฝั่งที่
# เรียกยกเลิกไปแล้ว) จึงต้องให้เวลานานกว่านี้มากสำหรับ call ที่โอนข้อมูลจริงโดยเฉพาะ
_HARD_DOWNLOAD_TIMEOUT_S = 600.0

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


_PENDING_ID_RE = re.compile(r"pending export requests?\s*\((\S+?)\)", re.IGNORECASE)


def _is_pending_conflict(exc: BaseException) -> bool:
    """แยกแยะ error "มีคำขอ export ค้างอยู่" (status=7) ของ JSOC

    ต้องไล่ดูทั้งสาย ``__cause__`` เพราะ :meth:`JsocClient._with_retry` ห่อ
    exception ต้นทางไว้ใน ``RuntimeError`` ข้อความจริงจึงไม่อยู่ในตัวบนสุด
    """
    return _pending_conflict_id(exc) is not None or any(
        "pending export request" in str(cur).lower() for cur in _exc_chain(exc)
    )


def _exc_chain(exc: BaseException) -> list[BaseException]:
    seen: set[int] = set()
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _pending_conflict_id(exc: BaseException) -> str | None:
    """ดึง request ID ของคำขอที่ค้าง (เช่น ``JSOC_20260908_003076``) จากข้อความ error"""
    for cur in _exc_chain(exc):
        match = _PENDING_ID_RE.search(str(cur))
        if match:
            return match.group(1)
    return None


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

    @staticmethod
    def _call_with_hard_timeout(fn, args, kwargs, timeout_s: float = _HARD_CALL_TIMEOUT_S):
        """เรียก ``fn`` ใน thread แยก แล้วบังคับ timeout จริง — กัน DNS/connection
        ค้างไม่มีกำหนดที่ ``socket.setdefaulttimeout`` เอาไม่อยู่ (ดูหมายเหตุบนสุดของ
        ไฟล์) thread ที่ค้างจะถูกทิ้งไว้ (Python ฆ่า thread ไม่ได้) แต่ฝั่งเรียกจะได้
        ``TimeoutError`` กลับไปให้ ``_with_retry`` ลองใหม่แทนที่จะค้างทั้ง process"""
        # ห้ามใช้ context manager (`with ThreadPoolExecutor(...) as pool`) — ตอน
        # exit มันเรียก shutdown(wait=True) ซึ่งจะรอ thread ที่ค้างอยู่จนจบ เท่ากับ
        # ค้างต่อเหมือนเดิม ต้อง shutdown(wait=False) เองตอน timeout เพื่อปล่อยมือ
        # จาก thread ที่ยังค้างอยู่จริง ๆ (ฆ่าไม่ได้ แต่ไม่ต้องรอมันด้วย)
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fn, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_s)
        except FutureTimeoutError:
            pool.shutdown(wait=False)
            raise TimeoutError(
                f"ไม่ตอบสนองเกิน {timeout_s:.0f} วินาที (อาจเป็น DNS/connection ค้าง)"
            ) from None
        else:
            pool.shutdown(wait=False)
            return result

    def _with_retry(
        self, description: str, fn, *args, hard_timeout_s: float = _HARD_CALL_TIMEOUT_S, **kwargs
    ):
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_with_hard_timeout(fn, args, kwargs, timeout_s=hard_timeout_s)
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

    def _retry_on_pending(
        self,
        description: str,
        fn,
        wait_s: float = 90.0,
        max_waits: int = 10,
        own_id: str | None = None,
    ):
        """เรียก ``fn`` (ไม่รับพารามิเตอร์ — ผูกด้วย lambda/closure) ซ้ำถ้า JSOC ตอบ
        "pending export request" (``status=7``)

        JSOC อนุญาตให้มีคำขอ export ค้างได้ **คำขอเดียวต่อผู้ใช้** ถ้าการเชื่อมต่อ
        หลุดระหว่างรอคำขอก่อนหน้า (เน็ตสะดุด, WinError 10060) คำขอนั้นยังค้างอยู่
        ฝั่ง JSOC ส่วนฝั่งเราทิ้งไปแล้ว ผลคือคำขอถัดไป *ทุกอัน* ถูกปฏิเสธทันที
        ด้วย ``status=7`` และเฟรมที่เหลือทั้งหมดจะล้มเหลวรวดเดียวในไม่กี่วินาที
        (เคยเกิดจริง: 5 เฟรมตายใน 90 วินาที) การรอให้คำขอค้างนั้นเสร็จเองแล้วค่อย
        ลองใหม่จึงถูกกว่าการปล่อยให้ล้มทั้งชุด — ใช้ backoff ยาวกว่า
        :meth:`_with_retry` มาก (นาทีแทนวินาที) เพราะต้องรอให้คำขอเดิมเสร็จเองจริง ๆ
        ไม่ใช่แค่เน็ตสะดุดชั่วคราว ต้องครอบทั้งขั้นส่ง (``client.export``) และขั้นรอ
        สถานะ (``request.wait``) เพราะ pending conflict โผล่ได้ทั้งสองจุด

        .. note::
           ข้อความ error บอก request ID ของคำขอที่ค้างมาด้วย เคยลองรอเฉย ๆ
           (sleep แล้วลองใหม่) แล้วไม่พอ — วัดจากของจริง: คำขอค้างใช้เวลาประมวลผล
           เสร็จฝั่ง JSOC เร็วมาก (แค่ยังไม่มีใคร poll ไปรับทราบ) แต่ตราบใดที่ไม่มี
           ใคร poll มันค้างอยู่ในสถานะ "pending" ของผู้ใช้ตลอดไปและบล็อกคำขอถัดไป
           *ทุกอัน* ทันที — และคำขอถัดไปเองก็จะกลายเป็นคำขอค้างตัวใหม่อีกถ้าเราไม่รอ
           มันจนจบ (self-perpetuating) จึงต้อง**เกาะ id ที่ error บอกมาด้วย
           ``export_from_id()`` แล้ว ``.wait()`` เพื่อ poll สถานะให้จบ**ก่อนลองใหม่
           — ใช้แค่ poll สถานะเฉย ๆ (ไม่ดาวน์โหลด/ไม่แตะไฟล์ของมัน) จึงปลอดภัย
           แม้ id นั้นจะเป็นของคำขออื่น (เฟรมก่อนหน้า/คนละ record) เพราะเราไม่เคยเอา
           ข้อมูลของมันมาใช้ที่นี่เลย

        .. important::
           พบจริงว่าตอนเรียก ``request.wait()`` (ไม่ใช่ตอน ``client.export()``)
           แล้วเจอ pending conflict ที่ id **ตรงกับ id ของ ``request`` ตัวเองเป๊ะ ๆ**
           — object เดิมที่ ``.wait()`` ค้างอยู่บางทีไม่เห็นว่าตัวเองเสร็จแล้ว (เป็น
           race/cache bug ฝั่ง ``drms``/JSOC) เรียก ``.wait()`` ซ้ำบน object เดิมกี่
           ครั้งก็ยังฟ้องอ้างถึงตัวเองไม่เลิก ทั้งที่ ``export_from_id()`` แบบสด ๆ
           ยืนยัน status=0 (เสร็จจริง) ไปแล้ว ผู้เรียกที่รู้ id ของตัวเองอยู่แล้ว
           (เช่น ``export_segments``) จึงควรส่ง ``own_id`` มาด้วย — ถ้า drain แล้วเจอ
           id ตรงกับ ``own_id`` และ status=0 จะถือว่าสำเร็จทันที ไม่ไปเรียก ``fn()``
           ซ้ำบน object เดิมอีก
        """
        for attempt in range(1, max_waits + 1):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 — drms โยน exception ได้หลากหลายชนิด
                if not _is_pending_conflict(exc) or attempt == max_waits:
                    raise
                blocking_id = _pending_conflict_id(exc)
                if blocking_id:
                    logger.warning(
                        "%s: JSOC มีคำขอค้างอยู่ (%s) — เกาะ poll สถานะให้จบก่อนลองใหม่ (%d/%d)",
                        description,
                        blocking_id,
                        attempt,
                        max_waits,
                    )
                    drained = self._drain_pending(blocking_id, timeout_s=wait_s)
                    if blocking_id == own_id and getattr(drained, "status", None) == 0:
                        return None
                else:
                    logger.warning(
                        "%s: JSOC ยังมีคำขอ export ค้างอยู่ (ไม่มี id ในข้อความ error) — "
                        "รอ %.0f วินาทีแล้วลองใหม่ (%d/%d)",
                        description,
                        wait_s,
                        attempt,
                        max_waits,
                    )
                    time.sleep(wait_s)

    def _drain_pending(self, request_id: str, timeout_s: float):
        """poll สถานะของคำขอ export ที่ค้างจน ``.wait()`` จบ (สำเร็จหรือ error ก็ตาม)
        เพื่อให้ JSOC ปลดล็อกคิวของผู้ใช้ แล้วคืน request object ที่เพิ่ง poll สด ๆ นี้
        (``None`` ถ้า poll ไม่สำเร็จเลย) — ปกติใช้แค่เช็ค ``.status`` เพื่อปลดล็อก
        เท่านั้น ไม่ดาวน์โหลดไฟล์ของมัน **ยกเว้น** ผู้เรียกยืนยันแล้วว่า id ตรงกับ
        คำขอของตัวเอง (ดู note ใน :meth:`_retry_on_pending`) ซึ่งตอนนั้นการใช้ object
        นี้แทนตัวเดิมที่ค้างอยู่ก็ปลอดภัย เพราะมันคือคำขอเดียวกัน แค่ instance ใหม่ที่
        เห็นสถานะล่าสุดถูกต้อง
        """
        try:
            request = self.client.export_from_id(request_id)
            if hasattr(request, "wait"):
                request.wait(timeout=timeout_s)
            return request
        except Exception as exc:  # noqa: BLE001 — แค่พยายาม poll ให้จบ ไม่ใช่ critical path
            logger.debug("เกาะ poll คำขอค้าง %s ไม่สำเร็จ (จะลองใหม่รอบถัดไป): %s", request_id, exc)
            return None

    def _submit_export(
        self,
        query: str,
        method: str,
        protocol: str,
        wait_s: float = 90.0,
        max_waits: int = 10,
    ):
        """สั่ง export โดยรอให้คิวของผู้ใช้ว่างก่อนถ้าจำเป็น (ดู :meth:`_retry_on_pending`)"""
        return self._retry_on_pending(
            f"export {query}",
            lambda: self._with_retry(
                f"export {query}",
                self.client.export,
                query,
                method=method,
                protocol=protocol,
                email=self.email,
            ),
            wait_s=wait_s,
            max_waits=max_waits,
        )

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

    def export_fast(
        self,
        recordset: str,
        segments: list[str],
        out_dir: Path,
        method: str = "url_quick",
        protocol: str = "as-is",
    ) -> pd.DataFrame:
        """เหมือน :meth:`export_segments` แต่ใช้ ``url_quick``/``as-is`` — **ไม่เข้าคิว
        export ของ JSOC เลย** (วัดจริง: ~2-20 วิ/คำขอ แทนที่จะเป็น 20-60 วิ) เพราะ
        ``request.id`` เป็น ``None`` เสมอ — ไม่มี object ให้ track สถานะ จึงไม่มี
        "pending export request" ให้ชนกับคำขออื่นแบบที่ :meth:`export_segments` ต้อง
        กัน (ดู :meth:`_retry_on_pending`)

        แลกมาด้วยไฟล์ที่ header **ไม่มี WCS เลย** (มีแค่ NAXIS/BITPIX) เพราะ JSOC ส่ง
        segment ดิบบนดิสก์มาตรง ๆ ไม่ได้เขียน keyword จาก DRMS DB ลงไปให้เหมือน
        ``protocol="fits"`` — ค่าพิกเซลเหมือนกันทุกประการ (คนละวิธีดึงข้อมูลเดียวกัน)
        ผู้เรียกต้องต่อ WCS เองจาก keyword query แยกต่างหาก (:func:`frame_wcs.fetch_frame_wcs`
        / :func:`frame_wcs.fetch_sharp_wcs` + :func:`frame_wcs.map_from_as_is`) ก่อน
        สร้าง ``sunpy.map.Map`` — **ห้ามเรียก** ``sunpy.map.Map(fits_path)`` ตรง ๆ กับ
        ไฟล์ที่ได้จากเมธอดนี้ จะพังด้วย "Image coordinate units for axis 1 not
        present in metadata" แบบเดียวกับที่ ``export_segments`` เขียนเตือนไว้

        Returns
        -------
        DataFrame ที่มีคอลัมน์ ``record`` และ ``download`` (path ในเครื่อง) เหมือน
        :meth:`export_segments`
        """
        if not self.email:
            raise RuntimeError(
                "การ export ต้องใช้อีเมลที่ลงทะเบียนกับ JSOC — "
                "ตั้งค่า SUNSEG_JSOC_EMAIL ใน .env"
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        query = f"{recordset}{{{','.join(segments)}}}"

        request = self._with_retry(
            f"export (fast) {query}",
            self.client.export,
            query,
            method=method,
            protocol=protocol,
            email=self.email,
        )
        if getattr(request, "status", 0) != 0:
            raise RuntimeError(
                f"JSOC ปฏิเสธคำขอ export (fast) ({query}): status={request.status}"
            )

        result = self._with_retry(
            f"download (fast) {query}", request.download, str(out_dir),
            hard_timeout_s=_HARD_DOWNLOAD_TIMEOUT_S,
        )
        logger.info("ดาวน์โหลดสำเร็จ (fast) %d ไฟล์ไปที่ %s", len(result), out_dir)
        return result

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

        ใช้ ``url``/``fits`` — JSOC เก็บ keyword ไว้ในฐานข้อมูล DRMS ไม่ได้เก็บใน FITS
        header ไฟล์ segment ดิบบนดิสก์ ถ้าเปลี่ยนไปใช้ ``url_quick``/``as-is`` ตรง ๆ
        โดยไม่ต่อ WCS เอง ไฟล์จะมี header เปล่า (ไม่มี WCS เลยสักตัว) และ
        ``sunpy.map.Map`` จะล้มด้วย "Image coordinate units for axis 1 not present in
        metadata" ส่วน ``protocol="fits"`` สั่งให้ JSOC สร้างไฟล์ใหม่พร้อมเขียน keyword
        ลง header ให้ (~110 ตัว รวม CTYPE/CUNIT/CRVAL/CDELT/CRPIX/CROTA2/RSUN_OBS/DSUN_OBS)
        ซึ่งจำเป็นทั้งกับภาพเต็มดวงและ SHARP patch เพราะ mask สร้างจากการแปลงพิกัดผ่าน WCS

        แลกมาด้วยการเข้าคิว export ของ JSOC (~20-60 วินาทีต่อคำขอ แทนที่จะได้ทันที) —
        ถ้าต้องดาวน์โหลดจำนวนมาก (หลักพันเฟรมขึ้นไป) ใช้ :meth:`export_fast` แทน:
        ได้ไฟล์ ``as-is`` แบบไม่เข้าคิวเลย (ค่าพิกเซลเหมือนกันทุกประการ) แล้วต่อ WCS เอง
        จาก keyword query ด้วย ``frame_wcs.fetch_frame_wcs``/``fetch_sharp_wcs`` +
        ``frame_wcs.map_from_as_is`` — ดู ``download_images.py`` เป็นตัวอย่าง

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

        # การรอสถานะก็ต้อง retry แบบเดียวกับตอนส่ง (ทั้ง network hiccup และ pending
        # conflict — ข้อความ "pending export request" โผล่ตอน wait() ได้เหมือนกัน
        # ไม่ใช่แค่ตอน export()) ถ้าไม่ retry ให้ครบ คำขอจะยังค้างอยู่ฝั่ง JSOC และไป
        # บล็อกทุกเฟรมถัดไป — แต่ต่างจาก _submit_export ตรงที่พบว่า object ``request``
        # ตัวนี้เอง บางทีเรียก ``.wait()`` ซ้ำกี่ครั้งก็ยังฟ้อง pending conflict อ้างถึง
        # id ของตัวเองไม่เลิก (race/cache bug ฝั่ง drms) ทั้งที่ export_from_id() แบบ
        # สด ๆ ยืนยันว่า status=0 (เสร็จจริง) แล้ว จึงต้องเช็คเป็นพิเศษ: ถ้า drain
        # แล้วเจอ id ตรงกับ request ของเราเองและ status=0 ให้เชื่อผลจาก drain แทน
        # ไม่ไปเชื่อ ``request.status`` ที่อาจยังไม่อัปเดต (ดู _drain_pending)
        own_id = getattr(request, "id", None)
        if hasattr(request, "wait"):
            max_waits = 10
            for attempt in range(1, max_waits + 1):
                try:
                    self._with_retry(f"wait {query}", request.wait, timeout=timeout_s)
                    break
                except Exception as exc:  # noqa: BLE001 — drms โยน exception ได้หลากหลายชนิด
                    if not _is_pending_conflict(exc) or attempt == max_waits:
                        raise
                    blocking_id = _pending_conflict_id(exc)
                    logger.warning(
                        "wait %s: JSOC มีคำขอค้างอยู่ (%s) — เกาะ poll สถานะให้จบก่อนลองใหม่ (%d/%d)",
                        query,
                        blocking_id or "?",
                        attempt,
                        max_waits,
                    )
                    if blocking_id:
                        drained = self._drain_pending(blocking_id, timeout_s=90.0)
                        if blocking_id == own_id and getattr(drained, "status", None) == 0:
                            # object เดิมค้าง ใช้ instance ใหม่ที่เพิ่ง poll เห็นสถานะ
                            # ถูกต้องแทน (คำขอเดียวกัน แค่คนละ instance — ดาวน์โหลดได้
                            # ปกติ) ดู note ใน _retry_on_pending
                            request = drained
                            break
                    else:
                        time.sleep(90.0)

        if getattr(request, "status", 0) != 0:
            raise RuntimeError(
                f"JSOC ปฏิเสธคำขอ export ({query}): status={request.status}"
            )

        result = self._with_retry(
            f"download {query}", request.download, str(out_dir), timeout=read_timeout_s,
            hard_timeout_s=max(_HARD_DOWNLOAD_TIMEOUT_S, read_timeout_s + 60.0),
        )
        logger.info("ดาวน์โหลดสำเร็จ %d ไฟล์ไปที่ %s", len(result), out_dir)
        return result

    def parse_trec(self, series: pd.Series) -> pd.Series:
        """แปลงคอลัมน์ T_REC (สตริง TAI) เป็น pandas datetime"""
        return self._drms.to_datetime(series)
