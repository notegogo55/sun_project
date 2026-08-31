"""ตั้งค่า logging และแก้ปัญหา encoding ของ console บน Windows

console เริ่มต้นของ Windows ใช้ code page แบบ legacy (cp874/cp1252) ซึ่ง encode
ภาษาไทยไม่ได้ ทำให้สคริปต์ตายด้วย ``UnicodeEncodeError`` ตั้งแต่บรรทัด print แรก
ทุก entrypoint จึงต้องเรียก :func:`setup_logging` ก่อนพิมพ์อะไรออกจอ
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def force_utf8_stdio() -> None:
    """บังคับให้ stdout/stderr ใช้ UTF-8 (idempotent — เรียกซ้ำได้)"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        if (getattr(stream, "encoding", "") or "").lower().replace("-", "") == "utf8":
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # stream ถูก redirect ไปยังที่ที่ reconfigure ไม่ได้ — ปล่อยผ่าน
            pass


def setup_logging(
    level: int | str = logging.INFO,
    log_file: Path | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """ตั้งค่า root logger พร้อมรูปแบบที่อ่านง่าย

    Parameters
    ----------
    level
        ระดับ log ของโค้ดเรา
    log_file
        ถ้าระบุ จะเขียน log ลงไฟล์ด้วย (UTF-8) นอกเหนือจากแสดงบนจอ
    quiet_libraries
        ลดเสียงรบกวนจาก drms/sunpy/matplotlib ที่ log ละเอียดเกินจำเป็น
    """
    force_utf8_stdio()

    root = logging.getLogger()
    root.setLevel(level)
    # ล้าง handler เดิม เพื่อไม่ให้ข้อความซ้ำเมื่อเรียกฟังก์ชันนี้หลายครั้ง
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-7s  %(name)-28s  %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

    if quiet_libraries:
        # หมายเหตุ: ห้ามใส่ "astropy"/"sunpy" ที่นี่ — ดู quiet_science_libraries()
        for name in ("drms", "parfive", "matplotlib", "urllib3", "asyncio", "PIL"):
            logging.getLogger(name).setLevel(logging.WARNING)
        quiet_science_libraries()

    return root


def quiet_science_libraries(level: int = logging.WARNING) -> None:
    """ลดเสียงรบกวนจาก astropy/sunpy ผ่าน API ของตัวไลบรารีเอง

    ทั้งสองตัวติดตั้ง logger class ของตัวเองด้วย ``logging.setLoggerClass()`` แล้ว
    คาดหวังว่า ``logging.getLogger("astropy")`` จะคืน instance ของคลาสนั้น ถ้าเรา
    ไปเรียก ``getLogger("astropy")`` ก่อนที่ไลบรารีจะถูก import มันจะสร้าง
    ``logging.Logger`` ธรรมดาค้างไว้ใน cache แล้วตัวไลบรารีจะพังตอน init ด้วย
    ``AttributeError: 'Logger' object has no attribute '_set_defaults'``

    จึงต้องแตะ logger ของมัน **หลัง** import แล้วเท่านั้น และเรียกผ่าน ``mod.log``
    ที่ไลบรารีเตรียมไว้ให้ ฟังก์ชันนี้เรียกซ้ำได้ปลอดภัย และไม่ทำอะไรเลยถ้ายังไม่ได้
    import ไลบรารีเหล่านั้น
    """
    for mod_name in ("astropy", "sunpy"):
        module = sys.modules.get(mod_name)
        log = getattr(module, "log", None)
        if log is not None:
            try:
                log.setLevel(level)
            except (AttributeError, ValueError):
                pass
