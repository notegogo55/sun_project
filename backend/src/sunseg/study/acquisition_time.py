"""ประเมินเวลาที่ใช้ดาวน์โหลด/สกัด feature ต้นทางของงานเปรียบเทียบ feature

ขั้นเหล่านี้ทำไว้ก่อนมีการจับเวลา จึงไม่มีตัวเลขที่บันทึกตรง ๆ ต้องประเมินจากสองแหล่งที่ไม่ขึ้นต่อกัน
แล้วเทียบกันเพื่อดูว่าขนาดตรงกันไหม:

- **log** (:func:`log_sessions`) — แต่ละบรรทัดมีแต่ ``HH:MM:SS`` ไม่มีวันที่ และ log ถูกต่อท้ายหลาย session
  รวมทั้งบางขั้นเคยรันหลายโปรเซสพร้อมกันเขียนไฟล์เดียว บรรทัดจึงสลับลำดับได้
- **เวลาแก้ไขไฟล์ผลลัพธ์** (:func:`busy_seconds`) — ไม่ขึ้นกับลำดับใน log และใช้ได้แม้ขั้นนั้นไม่มี log
  เลย (เช่น ดาวน์โหลด X-ray รอบแรก) แต่ไม่นับเวลาที่งานใช้โดยไม่มีไฟล์ออก จึงเป็นค่าต่ำสุดของเวลาทำงาน

ทั้งสองแบบนับเฉพาะ **เวลาที่งานยังเดินอยู่**: ช่วงห่างระหว่างเหตุการณ์ (บรรทัด log หรือไฟล์ที่ทยอยเกิด)
ที่ยาวกว่าเกณฑ์ถือว่าเครื่องพัก/ไม่ได้รัน ไม่นับ — ไม่งั้นงานที่ทำข้ามหลายวันจะได้เวลาเป็นสัปดาห์
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_LOG_LINE = re.compile(r"^﻿?(\d\d):(\d\d):(\d\d)\s+[A-Z]+\s+\S+\s+(.*)$")
_DAY = 86400


@dataclass
class Session:
    """หนึ่งรอบการรันสคริปต์ใน log (เริ่มที่บรรทัดชื่อ)"""

    title: str
    active_seconds: float = 0.0
    idle_gaps: int = 0
    completed: bool = False


def log_sessions(
    lines: Iterable[str], title: str, idle_minutes: float = 30.0, done: str | None = None
) -> list[Session]:
    """แยก log เป็น session แล้วรวมเวลาที่ทำงานอยู่จริงของแต่ละ session

    Parameters
    ----------
    title
        regex ของบรรทัดที่เริ่ม session ใหม่ (บรรทัดก่อนหน้าบรรทัดแรกที่ตรงถูกข้าม)
    idle_minutes
        ช่วงห่างระหว่างสองบรรทัดที่ยาวกว่านี้ไม่นับเป็นเวลาทำงาน (นับจำนวนไว้ใน ``idle_gaps``)
    done
        regex ของบรรทัดที่บอกว่า session นั้นเสร็จสมบูรณ์ (ตั้ง ``completed``) — ไม่ระบุ = ไม่ตรวจ

    ช่วงห่างคิดแบบมีเครื่องหมายในช่วง ±12 ชม. เพื่อให้ข้ามเที่ยงคืนต่อเนื่อง (23:59:50 → 00:00:10 = 20 วินาที)
    และบรรทัดที่เวลาถอยหลังเล็กน้อย (หลายโปรเซสเขียนสลับกัน) ไม่ถูกนับเป็น ~24 ชม. หรือนับซ้ำ
    """
    title_re = re.compile(title)
    done_re = re.compile(done) if done else None
    limit = idle_minutes * 60

    sessions: list[Session] = []
    current: Session | None = None
    previous: int | None = None
    for raw in lines:
        match = _LOG_LINE.match(raw.rstrip("\n"))
        if match is None:
            continue
        hours, minutes, seconds, message = match.groups()
        now = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

        if title_re.search(message):
            current = Session(title=message.strip())
            sessions.append(current)
            previous = now  # ช่วงระหว่าง session ไม่ใช่เวลาทำงานของ session ใด
        if current is None:
            previous = now
            continue

        if previous is not None:
            gap = (now - previous + _DAY // 2) % _DAY - _DAY // 2
            if gap >= 0:
                if gap <= limit:
                    current.active_seconds += gap
                else:
                    current.idle_gaps += 1
                previous = now
            # gap < 0: บรรทัดที่มาสลับ ไม่ขยับ previous เพื่อไม่ให้ช่วงเดียวกันถูกนับซ้ำ
        else:
            previous = now
        if done_re is not None and done_re.search(message):
            current.completed = True
    return sessions


def busy_seconds(mtimes: Iterable[float], max_gap_minutes: float = 30.0) -> float:
    """เวลาที่งานยังเดินอยู่ จากเวลาแก้ไขของไฟล์ผลลัพธ์ที่ทยอยเกิด

    เรียงเวลาไฟล์ทั้งหมด แล้วรวมเฉพาะช่วงห่างระหว่างไฟล์ที่อยู่ติดกันที่ไม่เกิน ``max_gap_minutes``
    """
    ordered = sorted(mtimes)
    limit = max_gap_minutes * 60
    return sum(b - a for a, b in zip(ordered, ordered[1:], strict=False) if b - a <= limit)
