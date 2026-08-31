"""ทดสอบ JsocClient — เน้นจุดที่พังเงียบและตรวจจับยาก"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from sunseg.data.jsoc_client import JsocClient, iter_time_chunks, to_drms_time


class FakeExportRequest:
    """เลียนแบบ drms ExportRequest ที่ export สำเร็จ"""

    status = 0

    def __init__(self) -> None:
        self.waited = False

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
