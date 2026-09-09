"""ทดสอบ JsocClient — เน้นจุดที่พังเงียบและตรวจจับยาก"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sunseg.data.jsoc_client import (
    JsocClient,
    _is_pending_conflict,
    _pending_conflict_id,
    iter_time_chunks,
    to_drms_time,
)


def _pending_error(request_id: str, email: str = "tester@example.com") -> RuntimeError:
    """เลียนแบบข้อความ error จริงของ drms ตอน JSOC ตอบ status=7 (มีคำขอค้าง)"""
    return RuntimeError(
        f"User {email} has 1 pending export requests ({request_id}); please wait "
        "until at least one request has completed before submitting a new one. [status=7]"
    )


class FakeExportRequest:
    """เลียนแบบ drms ExportRequest ที่ export สำเร็จ"""

    status = 0

    def __init__(self, request_id: str | None = None) -> None:
        self.waited = False
        self.id = request_id

    def wait(self, timeout: int | None = None) -> None:  # noqa: ARG002
        self.waited = True

    # ลายเซ็นต้องตรงกับ drms.ExportRequest.download ตัวจริง ซึ่งรับ timeout เป็น
    # keyword-only (drms >= 0.7) — export_segments ส่งค่านี้เข้ามาเสมอ
    def download(self, directory: str, *, timeout: int | None = None) -> pd.DataFrame:  # noqa: ARG002
        return pd.DataFrame({"record": ["r"], "download": [str(Path(directory) / "a.fits")]})


class FakeDrmsClient:
    """บันทึกอาร์กิวเมนต์ที่ถูกส่งเข้า export() ไว้ให้ตรวจ"""

    def __init__(self) -> None:
        self.export_calls: list[dict] = []

    def export(self, query, **kwargs):
        self.export_calls.append({"query": query, **kwargs})
        return FakeExportRequest()


@pytest.fixture
def client(monkeypatch) -> JsocClient:
    """JsocClient ที่ผูกกับ drms ปลอม — ไม่แตะเครือข่ายจริง"""
    fake_module = SimpleNamespace(Client=lambda **_kwargs: FakeDrmsClient())
    monkeypatch.setitem(sys.modules, "drms", fake_module)
    return JsocClient(email="tester@example.com")


class TestExportSegments:
    def test_requests_fits_protocol_not_as_is(self, client, tmp_path):
        """protocol ต้องเป็น 'fits' เท่านั้น

        ``as-is`` ส่งไฟล์ segment ดิบบนดิสก์ของ JSOC กลับมา ซึ่ง **มี header เปล่า**
        เพราะ JSOC เก็บ keyword ไว้ในฐานข้อมูล DRMS ไม่ใช่ใน FITS ผลคือ
        ``sunpy.map.Map`` ล้มด้วย "Image coordinate units for axis 1 not present
        in metadata" และสร้าง mask ไม่ได้เลยทั้ง pipeline
        """
        client.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)

        call = client.client.export_calls[0]
        assert call["protocol"] == "fits"
        assert call["method"] == "url"
        assert call["protocol"] != "as-is"

    def test_passes_email_and_builds_segment_query(self, client, tmp_path):
        client.export_segments("hmi.sharp_720s[][t]", ["bitmap", "magnetogram"], tmp_path)

        call = client.client.export_calls[0]
        assert call["email"] == "tester@example.com"
        assert call["query"] == "hmi.sharp_720s[][t]{bitmap,magnetogram}"

    def test_creates_output_directory(self, client, tmp_path):
        target = tmp_path / "nested" / "frame"
        client.export_segments("hmi.M_720s[t]", ["magnetogram"], target)
        assert target.is_dir()

    def test_refuses_without_email(self, monkeypatch, tmp_path):
        fake_module = SimpleNamespace(Client=lambda **_kwargs: FakeDrmsClient())
        monkeypatch.setitem(sys.modules, "drms", fake_module)
        anonymous = JsocClient()

        with pytest.raises(RuntimeError, match="SUNSEG_JSOC_EMAIL"):
            anonymous.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)

    def test_raises_when_jsoc_rejects_request(self, client, tmp_path, monkeypatch):
        class Rejected(FakeExportRequest):
            status = 4

        monkeypatch.setattr(client.client, "export", lambda *a, **k: Rejected())

        with pytest.raises(RuntimeError, match="ปฏิเสธ"):
            client.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)


class TestRetry:
    def test_retries_then_succeeds(self, client, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)
        attempts = {"n": 0}

        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise ConnectionError("JSOC ไม่ตอบ")
            return "ok"

        assert client._with_retry("ทดสอบ", flaky) == "ok"
        assert attempts["n"] == 3

    def test_gives_up_after_max_retries(self, client, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _s: None)

        def always_fails():
            raise ConnectionError("JSOC ล่ม")

        with pytest.raises(RuntimeError, match="ล้มเหลวหลังพยายาม 4 ครั้ง"):
            client._with_retry("ทดสอบ", always_fails)


class TestPendingConflictParsing:
    """เจอบั๊กจริง: เน็ตสะดุดตอนรอคำขอ export ก่อนหน้า ฝั่งเราทิ้งไปแล้วแต่ JSOC ยัง
    ค้างไว้ ทำให้คำขอถัดไปทุกอันถูกปฏิเสธด้วย status=7 ทันที (เคยเกิดจริง: 5 เฟรม
    ตายใน 90 วินาที) — ทดสอบตัวแยกวิเคราะห์ error ที่ใช้ตัดสินใจว่าจะ drain แล้วลองใหม่"""

    def test_extracts_request_id_from_pending_error(self):
        exc = _pending_error("JSOC_20260908_002892")
        assert _pending_conflict_id(exc) == "JSOC_20260908_002892"

    def test_returns_none_when_message_has_no_id(self):
        assert _pending_conflict_id(RuntimeError("มีคำขอ export ค้างอยู่")) is None

    def test_returns_none_for_unrelated_error(self):
        assert _pending_conflict_id(ConnectionError("JSOC ไม่ตอบ")) is None

    def test_is_pending_conflict_true_for_direct_error(self):
        assert _is_pending_conflict(_pending_error("JSOC_X"))

    def test_is_pending_conflict_walks_the_cause_chain(self):
        """_with_retry ห่อ exception ต้นทางไว้ใน RuntimeError ใหม่ (raise ... from
        last_exc) — ต้องไล่ __cause__ ถึงจะเจอข้อความจริง ไม่ใช่แค่ดูตัวบนสุด"""
        inner = _pending_error("JSOC_Y")
        wrapped = RuntimeError("wait hmi.M_720s[t] ล้มเหลวหลังพยายาม 4 ครั้ง")
        wrapped.__cause__ = inner

        assert _is_pending_conflict(wrapped)
        assert _pending_conflict_id(wrapped) == "JSOC_Y"

    def test_is_pending_conflict_false_for_unrelated_error(self):
        assert not _is_pending_conflict(ConnectionError("The read operation timed out"))


class TestExportRecoversFromPendingConflict:
    """ทดสอบผ่าน export_segments() ทั้งเส้น (ไม่ mock _drain_pending) เพื่อยืนยัน
    พฤติกรรมที่ผู้ใช้เห็นจริง: เฟรมสำเร็จในที่สุดโดยไม่ raise และไม่ค้าง"""

    def test_recovers_when_export_hits_someone_elses_pending_request(
        self, client, tmp_path, monkeypatch
    ):
        """คำขอค้างเป็นของ *เฟรมก่อนหน้า* (คนละ id กับที่กำลังขอ) — drain แล้วลองส่ง
        คำขอใหม่อีกครั้งต้องสำเร็จ ไม่ใช่เกาะเอาผลของคำขอเก่ามาใช้ (จะได้ FITS ผิดเวลา)

        ``export()`` เองก็อยู่ใต้ ``_with_retry`` อยู่แล้ว (retry เน็ตสะดุดทั่วไป
        ``max_retries`` ครั้งก่อนห่อแล้วโยนขึ้นมา) ต้องให้มันพัง**ครบทุกครั้ง**ในรอบแรก
        ไม่งั้น _with_retry ชั้นในจะกลืน error นี้ไปเงียบ ๆ โดยที่ path การ drain ของ
        _retry_on_pending (สิ่งที่เทสต์นี้ต้องการพิสูจน์) ไม่ถูกใช้งานจริงเลย"""
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls = {"export": 0}

        def export(query, **kwargs):  # noqa: ARG001
            calls["export"] += 1
            if calls["export"] <= client.max_retries:
                raise _pending_error("JSOC_PREVIOUS_FRAME")
            return FakeExportRequest(request_id="JSOC_THIS_FRAME")

        monkeypatch.setattr(client.client, "export", export)
        monkeypatch.setattr(
            client.client, "export_from_id", lambda rid: FakeExportRequest(request_id=rid),
            raising=False,
        )

        result = client.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)

        # max_retries ครั้งพังหมดในรอบแรก (_with_retry ชั้นในยอมแพ้แล้วห่อโยนขึ้นมา)
        # แล้ว _retry_on_pending drain ก่อนลองรอบสอง ซึ่งสำเร็จตั้งแต่ครั้งแรก
        assert calls["export"] == client.max_retries + 1
        assert len(result) == 1

    def test_recovers_from_self_referential_pending_bug_during_wait(
        self, client, tmp_path, monkeypatch
    ):
        """บั๊กที่วัดได้จริง: .wait() ของ request ตัวเองฟ้อง pending conflict ที่ id
        ตรงกับตัวเองไม่เลิก ทั้งที่ export_from_id() แบบสด ๆ ยืนยันว่า status=0 แล้ว
        (race/cache ฝั่ง drms) ต้องไม่ raise และไม่วน .wait() ซ้ำไม่รู้จบบน object เดิม"""
        monkeypatch.setattr("time.sleep", lambda _s: None)

        class StuckOnItself(FakeExportRequest):
            def __init__(self):
                super().__init__(request_id="JSOC_SELF_001")
                self.wait_calls = 0

            def wait(self, timeout=None):  # noqa: ARG002
                self.wait_calls += 1
                raise _pending_error(self.id)

        stuck = StuckOnItself()
        monkeypatch.setattr(client.client, "export", lambda *a, **k: stuck)  # noqa: ARG005
        monkeypatch.setattr(
            client.client,
            "export_from_id",
            lambda rid: FakeExportRequest(request_id=rid),  # status=0 เสมอ — เสร็จจริงแล้ว
            raising=False,
        )

        result = client.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)

        # _with_retry เรียก .wait() 4 ครั้งก่อนยอมแพ้แล้วห่อ exception — ต้องไม่เกิน
        # นั้น (ไม่ใช่วนเรียกซ้ำนอกเหนือกลไก retry ปกติจนกว่าจะครบ max_waits ทั้งหมด)
        assert stuck.wait_calls == client.max_retries
        assert len(result) == 1

    def test_falls_back_to_plain_wait_when_error_has_no_request_id(
        self, client, tmp_path, monkeypatch
    ):
        """ข้อความ error บางทีไม่มี id แนบมา (รูปแบบข้อความต่างไป) — ต้องยัง retry ได้
        ด้วยการรอเฉย ๆ ไม่ใช่พังเพราะ parse id ไม่ได้

        ยังต้องมีวลี "pending export request" อยู่ (ภาษาอังกฤษ ตรงกับที่ drms/JSOC
        ใช้จริง) ไม่งั้น _is_pending_conflict จะไม่รู้จักว่านี่คือ pending conflict
        เลยตั้งแต่แรก — กรณีนี้จำลอง "มี id แต่ regex จับไม่ได้" ไม่ใช่ "ไม่ใช่ pending
        conflict"""
        monkeypatch.setattr("time.sleep", lambda _s: None)
        calls = {"export": 0}

        def export(query, **kwargs):  # noqa: ARG001
            calls["export"] += 1
            if calls["export"] <= client.max_retries:
                raise RuntimeError(
                    "User t@example.com has 1 pending export requests; please wait "
                    "until at least one request has completed. [status=7]"
                )  # ไม่มี id ในวงเล็บ
            return FakeExportRequest(request_id="JSOC_OK")

        monkeypatch.setattr(client.client, "export", export)

        result = client.export_segments("hmi.M_720s[t]", ["magnetogram"], tmp_path)

        assert calls["export"] == client.max_retries + 1
        assert len(result) == 1


class TestTimeHelpers:
    def test_to_drms_time_uses_tai_format(self):
        assert to_drms_time(datetime(2014, 10, 20, 12, 0, 0)) == "2014.10.20_12:00:00_TAI"

    def test_to_drms_time_accepts_date(self):
        assert to_drms_time(date(2014, 10, 20)) == "2014.10.20_00:00:00_TAI"

    def test_chunks_cover_range_without_gap_or_overlap(self):
        chunks = list(iter_time_chunks(date(2011, 1, 1), date(2011, 1, 10), chunk_days=4))
        assert chunks[0][0] == datetime(2011, 1, 1)
        assert chunks[-1][1] == datetime(2011, 1, 10)
        for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:], strict=False):
            assert prev_end == next_start

    def test_rejects_non_positive_chunk_days(self):
        with pytest.raises(ValueError, match="chunk_days"):
            list(iter_time_chunks(date(2011, 1, 1), date(2011, 2, 1), chunk_days=0))
