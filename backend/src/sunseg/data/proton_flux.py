"""ฟลักซ์โปรตอนราย 5 นาทีจากคลัง GOES particle

X-ray บอกว่า flare "ปะทุแรงแค่ไหน" แต่ไม่ได้บอกว่ามีอะไรเดินทางมาถึงโลกหรือเปล่า
flare บางดวงตามมาด้วยพายุอนุภาคพลังงานสูง (SEP) ซึ่งเป็นภัยจริงต่อดาวเทียม ระบบ
นำร่อง และนักบินอวกาศ ส่วนอีกหลายดวงที่แรงพอ ๆ กันกลับเงียบสนิท ต้องดูอนุกรม
โปรตอนรอบเวลานั้นเท่านั้นถึงจะแยกออก

**คลังต้นทางอยู่นอก repo** (ค่าปริยาย ``D:/PositionFlare/...``) เพราะเป็นข้อมูลดิบ
เกือบ 800 MB ที่โปรเจคอื่นดูแลอยู่แล้ว โมดูลนี้ "อ่านอย่างเดียว" ไม่ทำสำเนาและไม่
แก้ไขอะไรในนั้น ตั้งทับ path ได้ด้วย env ``SUNSEG_PROTON_DIR``

ทำไมอ่านด้วย ``seek`` แทน pandas: ไฟล์รายปีไฟล์ละ ~30 MB แต่กราฟหนึ่งใบใช้แค่
72 ชั่วโมง = 865 แถว ทุกไฟล์ในคลังเป็น fixed-width 140 ไบต์ต่อแถวและเป็นกริด
5 นาทีที่ต่อเนื่องไม่ขาด (ตรวจครบทั้ง 64 ไฟล์แล้ว — ดู ``_scan``) จึงคำนวณ byte
offset ของแถวได้ตรง ๆ อ่านราว 120 KB ต่อคำขอแทนที่จะ parse 30 MB ทุกครั้ง
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# รูปแบบไฟล์
# --------------------------------------------------------------------------- #

#: ทุกแถวข้อมูลยาวเท่ากันเป๊ะรวม newline — เป็นสิ่งที่ทำให้ seek ตรงแถวได้
ROW_BYTES = 140
STEP_MINUTES = 5
STEP = timedelta(minutes=STEP_MINUTES)

#: ช่องที่เอาไปพล็อต + ตำแหน่งคอลัมน์หลัง ``split()``
#: หนึ่งแถวคือ  YR MO DA HHMM MJD SOD | P>1 P>5 P>10 P>30 P>50 P>100 | E×3 | QC
#: เลือกแค่สี่ช่องนี้เพราะเป็นช่วงพลังงานที่บอกเรื่องคนละอย่าง: >1 MeV คือพื้นหลัง
#: ที่ขยับตลอด ส่วน >100 MeV คือตัวที่ทะลุเกราะและทำอันตรายได้จริง
CHANNELS: tuple[tuple[str, str, int], ...] = (
    ("p1", "> 1 MeV", 6),
    ("p10", "> 10 MeV", 8),
    ("p50", "> 50 MeV", 10),
    ("p100", "> 100 MeV", 11),
)
CHANNEL_KEYS = tuple(key for key, _, _ in CHANNELS)

#: ช่องอ้างอิงของ NOAA S-scale และของสรุปตัวเลขทั้งหมดในหน้าเว็บ
REFERENCE_CHANNEL = "p10"

#: ลำดับความน่าเชื่อถือของแหล่งข้อมูล — แหล่งที่มาก่อนชนะเสมอ แหล่งถัดไปเข้ามา
#: เติมเฉพาะช่วงที่ยังว่างเท่านั้น
#:
#: Primary/Secondary เป็นผลิตภัณฑ์ทางการของ SWPC จึงมาก่อน GOES16/18 ที่คลังต้นทาง
#: คำนวณเองจาก SGPS L2 ส่วน G08-G12 เป็นดาวเทียมยุคก่อนปี 2010 ใช้เติมส่วนหัวของ
#: ช่วงเวลาที่ยังไม่มี Primary
SOURCE_PRIORITY = ("Primary", "Secondary", "GOES16", "GOES18", "G11", "G10", "G12", "G08", "G09")

#: ระดับ NOAA S-scale วัดที่ช่อง >10 MeV หน่วย pfu (protons/cm²-s-sr)
S_SCALE: tuple[tuple[float, str], ...] = (
    (10.0, "S1"), (100.0, "S2"), (1000.0, "S3"), (10_000.0, "S4"), (100_000.0, "S5")
)

#: "NA"  = เครื่องวัดของดาวเทียมดวงนี้ไม่เคยรายงานช่องนี้เลย
#: "NaN" = ควรมีข้อมูลแต่ช่วงที่ขาดยาวเกินกว่าจะเติมได้อย่างปลอดภัย
#: ทั้งสองแบบมีค่าเท่ากันสำหรับการวาดกราฟ: เว้นช่องว่างไว้ ห้ามลากเส้นข้าม
_MISSING = frozenset({"NA", "NaN", "nan", "-1.00e+05"})

_COVERAGE_RE = re.compile(rb"# Coverage: (\S+) \.\. \S+\s+\((\d+) rows\)")
_FILENAME_RE = re.compile(r"^(?P<source>.+?)_(?P<year>\d{4})_5m_clean\.txt$")

#: header ของไฟล์ยาวไม่กี่กิโลไบต์ — อ่านเท่านี้พอหาบรรทัด Coverage กับจุดเริ่มข้อมูลได้
_HEADER_PROBE_BYTES = 8192


# --------------------------------------------------------------------------- #
# ดัชนีไฟล์
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _YearFile:
    """ไฟล์รายปีหนึ่งไฟล์ พร้อมทุกอย่างที่ต้องใช้คำนวณ offset ของแถว"""

    path: Path
    source: str
    start: datetime          # เวลาของแถวแรก (UTC, naive)
    n_rows: int
    data_offset: int         # ไบต์แรกของแถวแรก (หลัง header)

    @property
    def end(self) -> datetime:
        return self.start + (self.n_rows - 1) * STEP

    def row_at(self, when: datetime) -> int | None:
        """ดัชนีแถวของเวลานั้น — None ถ้าอยู่นอกช่วงของไฟล์นี้"""
        if when < self.start or when > self.end:
            return None
        offset = (when - self.start).total_seconds()
        index, remainder = divmod(offset, STEP_MINUTES * 60)
        # กริดเวลาถูกจัดให้ตรง 5 นาทีมาก่อนเรียกแล้ว เศษที่ไม่ลงตัวแปลว่ามีบั๊ก
        # ในการจัดกริด ไม่ใช่ข้อมูลผิด — คืน None ดีกว่าอ่านแถวเพี้ยน
        return int(index) if remainder == 0 else None


def _parse_header(path: Path) -> _YearFile | None:
    """อ่าน header เพื่อหาเวลาแถวแรกและตำแหน่งไบต์ที่ข้อมูลเริ่ม

    จำนวนแถวคำนวณจากขนาดไฟล์ ไม่ใช่จากตัวเลขใน header — ถ้าไฟล์ถูกเขียนค้างไว้
    ตัวเลขใน header จะหลอกให้ seek เลยท้ายไฟล์ แต่ขนาดไฟล์โกหกไม่ได้
    """
    match = _FILENAME_RE.match(path.name)
    if match is None:
        return None

    with path.open("rb") as handle:
        head = handle.read(_HEADER_PROBE_BYTES)

    separator = head.find(b"#---")
    coverage = _COVERAGE_RE.search(head)
    if separator < 0 or coverage is None:
        logger.warning("ข้ามไฟล์ที่ header ไม่ตรงรูปแบบที่คาด: %s", path)
        return None

    data_offset = head.find(b"\n", separator) + 1
    n_rows = (path.stat().st_size - data_offset) // ROW_BYTES
    if n_rows <= 0:
        return None

    start = datetime.fromisoformat(coverage.group(1).decode().replace("Z", "+00:00"))
    return _YearFile(
        path=path,
        source=match.group("source"),
        start=start.astimezone(UTC).replace(tzinfo=None),
        n_rows=int(n_rows),
        data_offset=data_offset,
    )


def _scan(root: Path) -> list[_YearFile]:
    """ไล่ดู ``<root>/<ปี>/<แหล่ง>_<ปี>_5m_clean.txt`` แล้วเรียงตามลำดับความน่าเชื่อถือ

    เรียงครั้งเดียวตอน startup เพื่อให้ ``window()`` วนไฟล์ตามลำดับได้เลยโดยไม่ต้อง
    sort ซ้ำทุกคำขอ แหล่งที่ไม่รู้จักไปต่อท้ายแทนที่จะถูกทิ้ง — คลังต้นทางเพิ่ม
    ดาวเทียมดวงใหม่ได้โดยที่นี่ไม่ต้องแก้
    """
    found = [parsed for path in sorted(root.glob("*/*_5m_clean.txt"))
             if (parsed := _parse_header(path)) is not None]

    def rank(entry: _YearFile) -> tuple[int, str, datetime]:
        try:
            priority = SOURCE_PRIORITY.index(entry.source)
        except ValueError:
            priority = len(SOURCE_PRIORITY)
        return (priority, entry.source, entry.start)

    return sorted(found, key=rank)


# --------------------------------------------------------------------------- #
# ผลลัพธ์
# --------------------------------------------------------------------------- #


@dataclass
class ProtonWindow:
    """อนุกรมโปรตอนรอบเวลาหนึ่ง — ``None`` แปลว่าไม่มีข้อมูล ไม่ใช่ศูนย์"""

    center: datetime
    times: list[datetime]
    values: dict[str, list[float | None]]
    sources: list[str]

    @property
    def has_data(self) -> bool:
        return any(v is not None for series in self.values.values() for v in series)

    def value_at(self, when: datetime, channel: str = REFERENCE_CHANNEL) -> float | None:
        """ค่าของช่องนั้น ณ ตัวอย่างที่ใกล้เวลาที่ระบุที่สุด"""
        series = self.values.get(channel)
        if not series or not self.times:
            return None
        index = round((when - self.times[0]).total_seconds() / (STEP_MINUTES * 60))
        return series[index] if 0 <= index < len(series) else None

    def peak_after(
        self, when: datetime, channel: str = REFERENCE_CHANNEL
    ) -> tuple[float, datetime] | None:
        """ยอดสูงสุดของช่องนั้นตั้งแต่ ``when`` ไปจนสุดหน้าต่าง

        ค่า ณ เวลาที่ flare พีคอย่างเดียวไม่พอ: อนุภาคเดินทางมาถึงโลกช้ากว่าแสงเป็น
        ชั่วโมง ตอน flare พีคจึงมักยังเป็นระดับพื้นหลังอยู่ ต้องดูยอดในหน้าต่างหลัง
        จากนั้นถึงจะรู้ว่าตามมาด้วยพายุรังสีหรือไม่
        """
        series = self.values.get(channel)
        if not series:
            return None
        best: tuple[float, datetime] | None = None
        for time, value in zip(self.times, series, strict=True):
            if time < when or value is None:
                continue
            if best is None or value > best[0]:
                best = (value, time)
        return best


def s_scale_level(pfu: float | None) -> str | None:
    """ระดับพายุรังสีสุริยะตามมาตรา NOAA (S1-S5) — None ถ้าต่ำกว่าเกณฑ์ S1"""
    if pfu is None:
        return None
    level = None
    for threshold, label in S_SCALE:
        if pfu >= threshold:
            level = label
    return level


# --------------------------------------------------------------------------- #
# ตัวอ่าน
# --------------------------------------------------------------------------- #


class ProtonFluxStore:
    """ดัชนีคลัง particle 5 นาที + การอ่านหน้าต่างเวลารอบ flare หนึ่งดวง

    ออกแบบให้ "ไม่มีคลัง = ไม่พัง" เหมือนส่วนอื่นของแอป: ถ้า path ไม่มีอยู่จริง
    ``available`` เป็น False แล้วหน้าเว็บซ่อนแผงนี้ไป ไม่ใช่ระเบิดตอน startup
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._files: list[_YearFile] = []

        if not self.root.exists():
            logger.warning("ไม่พบคลัง proton flux ที่ %s — แผงโปรตอนจะถูกซ่อน", self.root)
            return

        self._files = _scan(self.root)
        if not self._files:
            logger.warning("คลัง proton flux ที่ %s ไม่มีไฟล์ *_5m_clean.txt", self.root)
            return

        first, last = self.coverage  # type: ignore[misc]
        logger.info(
            "โหลดดัชนี proton flux: %d ไฟล์ %d แหล่ง  %s .. %s",
            len(self._files), len(self.sources), first.date(), last.date(),
        )

    # ------------------------------------------------------------------ #

    @property
    def available(self) -> bool:
        return bool(self._files)

    @property
    def sources(self) -> list[str]:
        """ชื่อแหล่งข้อมูลที่มีในคลัง เรียงตามลำดับความน่าเชื่อถือ"""
        seen: dict[str, None] = {}
        for entry in self._files:
            seen.setdefault(entry.source, None)
        return list(seen)

    @property
    def coverage(self) -> tuple[datetime, datetime] | None:
        if not self._files:
            return None
        return (min(f.start for f in self._files), max(f.end for f in self._files))

    def info(self) -> dict:
        span = self.coverage
        return {
            "available": self.available,
            "root": str(self.root),
            "n_files": len(self._files),
            "sources": self.sources,
            "channels": [{"key": key, "label": label} for key, label, _ in CHANNELS],
            "step_minutes": STEP_MINUTES,
            "start": None if span is None else span[0].isoformat(),
            "end": None if span is None else span[1].isoformat(),
        }

    # ------------------------------------------------------------------ #

    def window(self, center: datetime, before_hours: int, after_hours: int) -> ProtonWindow:
        """อนุกรมทุกช่องในช่วง ``[center - before, center + after]``

        ไล่เติมจากแหล่งที่น่าเชื่อถือที่สุดก่อน แหล่งถัดไปแตะเฉพาะช่องที่ยังว่าง —
        ผลคือเส้นเดียวที่ต่อเนื่องที่สุดเท่าที่คลังมี โดยที่ค่าจาก Primary ไม่เคยถูก
        ค่าจากดาวเทียมสำรองทับ
        """
        center = _as_naive_utc(center)
        grid_start = _floor_to_step(center - timedelta(hours=before_hours))
        grid_end = center + timedelta(hours=after_hours)
        n_samples = int((grid_end - grid_start).total_seconds() // (STEP_MINUTES * 60)) + 1

        times = [grid_start + i * STEP for i in range(n_samples)]
        values: dict[str, list[float | None]] = {
            key: [None] * n_samples for key in CHANNEL_KEYS
        }
        pending = n_samples * len(CHANNEL_KEYS)
        used: list[str] = []

        for entry in self._files:
            if pending == 0:
                break
            if entry.end < grid_start or entry.start > times[-1]:
                continue
            filled = self._fill_from(entry, times, values)
            if filled:
                pending -= filled
                if entry.source not in used:
                    used.append(entry.source)

        return ProtonWindow(center=center, times=times, values=values, sources=used)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _fill_from(
        entry: _YearFile,
        times: list[datetime],
        values: dict[str, list[float | None]],
    ) -> int:
        """เติมค่าจากไฟล์เดียวลงช่องที่ยังว่าง — คืนจำนวนช่องที่เติมได้

        อ่านทีเดียวเป็นบล็อกเดียวแล้วหั่นเอง แทนที่จะ seek ทีละแถว: ช่วงที่ต้องการ
        อยู่ติดกันเสมอ การอ่าน 120 KB ครั้งเดียวจึงเร็วกว่า 865 ครั้งของ syscall
        """
        first = next((i for i, t in enumerate(times) if entry.row_at(t) is not None), None)
        if first is None:
            return 0
        last = next(
            i for i in range(len(times) - 1, first - 1, -1) if entry.row_at(times[i]) is not None
        )

        start_row = entry.row_at(times[first])
        assert start_row is not None
        n_rows = last - first + 1

        with entry.path.open("rb") as handle:
            handle.seek(entry.data_offset + start_row * ROW_BYTES)
            block = handle.read(n_rows * ROW_BYTES)

        filled = 0
        for offset in range(n_rows):
            slot = first + offset
            fields = block[offset * ROW_BYTES:(offset + 1) * ROW_BYTES].split()
            if len(fields) < CHANNELS[-1][2] + 1:
                continue
            for key, _, column in CHANNELS:
                if values[key][slot] is not None:
                    continue
                parsed = _parse_flux(fields[column])
                if parsed is not None:
                    values[key][slot] = parsed
                    filled += 1
        return filled


# --------------------------------------------------------------------------- #
# ตัวช่วย
# --------------------------------------------------------------------------- #


def _parse_flux(raw: bytes) -> float | None:
    """``b"4.16e+00"`` -> 4.16 · ค่าที่หายไปหรือไม่เป็นบวก -> None

    ฟลักซ์ที่ <= 0 ไม่มีความหมายทางกายภาพและวาดบนแกน log ไม่ได้ จึงนับเป็น
    "ไม่มีข้อมูล" เหมือนกัน
    """
    text = raw.decode("ascii", "replace")
    if text in _MISSING:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if value > 0 else None


def _floor_to_step(when: datetime) -> datetime:
    """ปัดเวลาลงให้ตรงกริด 5 นาที เพื่อให้ทุกช่องตรงแถวในไฟล์พอดี"""
    return when.replace(
        minute=when.minute - when.minute % STEP_MINUTES, second=0, microsecond=0
    )


def _as_naive_utc(when: datetime) -> datetime:
    """ทั้งโปรเจคใช้ datetime แบบ naive ที่หมายถึง UTC — แปลงค่าที่ติด tzinfo มาให้ตรงกัน"""
    if when.tzinfo is None:
        return when
    return when.astimezone(UTC).replace(tzinfo=None)
