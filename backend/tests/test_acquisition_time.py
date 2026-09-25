"""ทดสอบการประเมินเวลาดาวน์โหลด/สกัด feature ต้นทางจาก log และเวลาแก้ไขไฟล์

ตัวเลขที่ได้ต้องตรงกับที่คำนวณมือ และสามกรณีที่ log จริงของโปรเจกต์เคยทำให้ตัวเลขพัง (ข้ามเที่ยงคืน,
เครื่องพักกลางงาน, หลายโปรเซสเขียน log สลับกัน) ต้องถูกจัดการไว้
"""

from __future__ import annotations

import pytest

from sunseg.study.acquisition_time import busy_seconds, log_sessions


def _log(*rows: tuple[str, str]) -> list[str]:
    return [f"{clock}  INFO     script                        {message}\n" for clock, message in rows]


class TestLogSessions:
    def test_active_time_of_each_session_matches_hand_computation(self):
        lines = _log(
            ("10:00:00", "START one"),
            ("10:00:10", "step"),
            ("10:00:40", "step"),
            ("10:05:00", "START two"),
            ("10:05:30", "step"),
        )

        one, two = log_sessions(lines, "START")

        assert one.active_seconds == pytest.approx(40)  # 10 + 30
        assert two.active_seconds == pytest.approx(30)  # ช่วง 10:00:40 -> 10:05:00 ไม่ใช่เวลาทำงานของใคร

    def test_midnight_crossing_is_a_continuation_not_a_day_long_gap(self):
        lines = _log(("23:59:50", "START"), ("00:00:10", "step"))
        (session,) = log_sessions(lines, "START")
        assert session.active_seconds == pytest.approx(20)
        assert session.idle_gaps == 0

    def test_gap_longer_than_threshold_is_excluded_and_counted(self):
        lines = _log(("10:00:00", "START"), ("10:10:00", "step"), ("10:10:30", "step"))

        (session,) = log_sessions(lines, "START", idle_minutes=5)

        assert session.active_seconds == pytest.approx(30)  # ช่วง 10 นาทีไม่นับ
        assert session.idle_gaps == 1

    def test_out_of_order_lines_from_concurrent_writers_are_not_double_counted(self):
        # สองโปรเซสเขียนสลับกัน: 10:00:05 ตามด้วย 10:00:03 แล้ว 10:00:08 — เวลาจริงที่ผ่านไปคือ 8 วินาที
        lines = _log(("10:00:00", "START"), ("10:00:05", "a"), ("10:00:03", "b"), ("10:00:08", "c"))

        (session,) = log_sessions(lines, "START")

        assert session.active_seconds == pytest.approx(8)

    def test_lines_before_the_first_title_are_ignored(self):
        lines = _log(("09:00:00", "noise"), ("09:30:00", "noise"), ("10:00:00", "START"), ("10:00:20", "step"))
        (session,) = log_sessions(lines, "START")
        assert session.active_seconds == pytest.approx(20)

    def test_completed_flag_follows_the_done_pattern(self):
        lines = _log(
            ("10:00:00", "START a"), ("10:00:10", "DONE"),
            ("10:01:00", "START b"), ("10:01:10", "step"),
        )

        first, second = log_sessions(lines, "START", done="DONE")

        assert first.completed is True
        assert second.completed is False

    def test_lines_without_a_timestamp_and_a_bom_are_tolerated(self):
        lines = ["﻿10:00:00  INFO     s     START\n", "Traceback (most recent call last):\n", *_log(("10:00:10", "x"))]
        (session,) = log_sessions(lines, "START")
        assert session.active_seconds == pytest.approx(10)


class TestBusySeconds:
    def test_only_gaps_within_the_threshold_count(self):
        # ไฟล์ที่ 0, 60, 120 วินาที แล้วเงียบไปนาน แล้วมาอีกที่ 10000, 10060
        assert busy_seconds([0, 60, 120, 10000, 10060], max_gap_minutes=30) == pytest.approx(180)

    def test_input_order_does_not_matter(self):
        assert busy_seconds([120, 0, 60], max_gap_minutes=30) == pytest.approx(120)

    def test_fewer_than_two_files_is_zero(self):
        assert busy_seconds([]) == 0
        assert busy_seconds([5.0]) == 0
