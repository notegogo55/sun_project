"""ตรวจว่า webapp เริ่มทำงานได้และ endpoint ตอบสนองถูกต้อง

จุดสำคัญของ test ชุดนี้: แอปต้อง **เปิดได้แม้ยังไม่มีโมเดลที่เทรนแล้ว** ผู้ใช้ที่
เพิ่ง clone โปรเจคมาต้องเปิดหน้าเว็บดูได้และเห็นว่าต้องรันสคริปต์ไหนต่อ ไม่ใช่เจอ
stack trace ตั้งแต่ import
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from sunseg.config import load_data_config


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestSystemEndpoints:
    def test_health_reports_component_readiness(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

        payload = response.json()
        assert payload["status"] == "ok"
        # ค่าจะเป็น True หรือ False ก็ได้ขึ้นกับว่าเทรนโมเดลแล้วหรือยัง แต่ต้องมีคีย์ครบ
        for key in ("forecast_model", "segmentation_model", "sequence_store"):
            assert isinstance(payload[key], bool)
        assert isinstance(payload["n_frames"], int)

    def test_info_returns_all_sections(self, client):
        payload = client.get("/api/info").json()

        assert set(payload) == {"forecast", "segmentation", "data"}
        assert payload["data"]["horizon_hours"] == 24
        assert payload["data"]["positive_class"] == "M1.0"

    def test_openapi_schema_is_generated(self, client):
        """/docs พึ่งสคีมานี้ — ถ้าสคีมาพัง เอกสาร API จะว่างเปล่า"""
        schema = client.get("/openapi.json").json()

        assert "/api/forecast" in schema["paths"]
        assert "/api/segment" in schema["paths"]
        assert "/api/goes" in schema["paths"]
        assert "/api/track" in schema["paths"]

    def test_index_page_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "sunseg" in response.text


class TestGracefulDegradation:
    """endpoint ที่ต้องพึ่งโมเดลต้องคืน 503 พร้อมคำแนะนำ ไม่ใช่ 500"""

    def test_forecast_without_model_returns_actionable_error(self, client):
        health = client.get("/api/health").json()
        response = client.get("/api/forecast?harpnum=1")

        if health["forecast_model"] and health["sequence_store"]:
            # มีโมเดลแล้ว: HARP 1 อาจไม่มีข้อมูลจริง จึงยอมรับ 200 หรือ 404
            assert response.status_code in (200, 404)
        else:
            assert response.status_code == 503
            assert "backend/scripts/" in response.json()["detail"]

    def test_segment_without_frames_returns_actionable_error(self, client):
        health = client.get("/api/health").json()
        response = client.get("/api/segment?timestamp=20140101_000000")

        if health["n_frames"] == 0:
            assert response.status_code == 503
            assert "backend/scripts/" in response.json()["detail"]
        else:
            assert response.status_code in (200, 404, 503)


class TestValidation:
    def test_rejects_malformed_date(self, client):
        response = client.get("/api/goes?start=ไม่ใช่วันที่&end=2014-02-01")
        assert response.status_code in (422, 503)

    def test_rejects_reversed_time_range(self, client):
        response = client.get("/api/goes?start=2014-02-01&end=2014-01-01")
        assert response.status_code in (422, 503)

    def test_forecast_requires_harpnum(self, client):
        assert client.get("/api/forecast").status_code == 422


class TestLayers:
    """เลเยอร์ภาพ AIA — ต้องใช้งานได้แม้ยังไม่ได้ดาวน์โหลดภาพเลย"""

    def test_health_exposes_aia_flags(self, client):
        health = client.get("/api/health").json()
        assert isinstance(health["aia_images"], bool)
        assert isinstance(health["n_aia_frames"], int)

    def test_unknown_layer_is_rejected(self, client):
        """คีย์ที่สะกดผิดคือความผิดพลาดของโปรแกรม ไม่ใช่สถานะของข้อมูล"""
        frames = client.get("/api/frames?limit=1").json()
        if not isinstance(frames, list) or not frames:
            pytest.skip("ยังไม่มีเฟรมภาพ")

        response = client.get(f"/api/segment?timestamp={frames[0]}&layer=ไม่มีจริง")
        assert response.status_code == 422
        assert "ไม่รู้จักเลเยอร์" in response.json()["detail"]

    def test_magnetogram_layer_is_always_available(self, client):
        frames = client.get("/api/frames?limit=1").json()
        if not isinstance(frames, list) or not frames:
            pytest.skip("ยังไม่มีเฟรมภาพ")

        response = client.get(
            f"/api/segment?timestamp={frames[0]}&use_ground_truth=true"
        )
        if response.status_code != 200:
            pytest.skip("เฟรมแรกไม่มี mask จริงกำกับ")

        data = response.json()
        mag = next(layer for layer in data["layers"] if layer["key"] == "mag")
        assert mag["available"] is True
        assert data["layer"] == "mag"

    def test_layers_list_every_atmospheric_channel(self, client):
        """ทั้งสามชั้นต้องปรากฏเสมอ แม้ยังไม่มีข้อมูล — เป็นสาระเชิงความรู้ของหน้านี้"""
        frames = client.get("/api/frames?limit=1").json()
        if not isinstance(frames, list) or not frames:
            pytest.skip("ยังไม่มีเฟรมภาพ")

        response = client.get(
            f"/api/segment?timestamp={frames[0]}&use_ground_truth=true"
        )
        if response.status_code != 200:
            pytest.skip("เฟรมแรกไม่มี mask จริงกำกับ")

        keys = [layer["key"] for layer in response.json()["layers"]]
        assert keys[0] == "mag"
        assert {"171", "304", "1600"}.issubset(set(keys))

    def test_missing_layer_falls_back_instead_of_failing(self, client):
        """เฟรมที่ยังไม่ได้ดาวน์โหลดเป็นสถานะปกติ ต้องได้ 200 พร้อมภาพ magnetogram"""
        frames = client.get("/api/frames?limit=2000").json()
        if not isinstance(frames, list) or not frames:
            pytest.skip("ยังไม่มีเฟรมภาพ")

        for stem in frames:
            response = client.get(
                f"/api/segment?timestamp={stem}&use_ground_truth=true&layer=171"
            )
            if response.status_code != 200:
                continue

            data = response.json()
            aia = next(layer for layer in data["layers"] if layer["key"] == "171")
            if aia["available"]:
                # เฟรมนี้มีข้อมูลจริง — ต้องได้เลเยอร์ที่ขอ พร้อมตัวเลขความเข้มแสง
                assert data["layer"] == "171"
                if data["detections"]:
                    channels = {
                        i["channel"] for i in data["detections"][0]["intensities"]
                    }
                    assert "171" in channels
            else:
                assert data["layer"] == "mag"
            return

        pytest.skip("ไม่มีเฟรมที่มี mask จริงกำกับ")

    def test_detections_are_ranked_by_area(self, client):
        frames = client.get("/api/frames?limit=2000").json()
        if not isinstance(frames, list) or not frames:
            pytest.skip("ยังไม่มีเฟรมภาพ")

        for stem in frames:
            response = client.get(
                f"/api/segment?timestamp={stem}&use_ground_truth=true"
            )
            if response.status_code != 200:
                continue
            detections = response.json()["detections"]
            if len(detections) < 2:
                continue

            assert [d["rank"] for d in detections] == list(range(1, len(detections) + 1))
            areas = [d["area_px"] for d in detections]
            assert areas == sorted(areas, reverse=True)
            return

        pytest.skip("ไม่พบเฟรมที่มี AR ตั้งแต่สองดวงขึ้นไป")


class TestConfusionMatrix:
    """แผง confusion matrix ในหน้าเว็บ — เรื่องนี้เชื่อถือได้ก็ต่อเมื่อจำนวนนับที่หน้าเว็บ
    เห็นตรงกับ artifacts/metrics/lstm.json เป๊ะ มิฉะนั้นจะมีตัวเลขสองชุดขัดกันในงานเดียว
    """

    def test_missing_predictions_file_returns_actionable_error(self, client):
        response = client.get("/api/forecast/confusion-matrix")
        if response.status_code == 200:
            pytest.skip("มีไฟล์ค่าทำนายอยู่แล้วในเครื่องนี้ — ทดสอบเคส 503 ไม่ได้โดยไม่ลบไฟล์จริง")

        assert response.status_code == 503
        assert "backend/scripts/" in response.json()["detail"]

    def test_sweep_covers_full_grid_and_contains_frozen_threshold(self, client):
        response = client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        if response.status_code != 200:
            pytest.skip("ยังไม่มีไฟล์ค่าทำนายราย sample")

        data = response.json()
        assert len(data["thresholds"]) == 200
        for key in ("tp", "fp", "tn", "fn"):
            assert len(data[key]) == 200

        # threshold ที่ freeze ไว้ต้องตกบนจุดของกริดพอดี ไม่ใช่แค่ใกล้เคียง
        assert data["thresholds"][data["frozen_index"]] == pytest.approx(
            data["frozen_threshold"], abs=1e-9
        )

    def test_counts_at_frozen_threshold_match_metrics_file(self, client):
        """test ที่สำคัญที่สุดของ ticket นี้ — เทียบจำนวนนับกับไฟล์ที่ train_lstm.py บันทึกไว้
        ตรงๆ ทั้งสองโมเดลและทั้งสอง split
        """
        metrics_path = load_data_config().paths.artifacts / "metrics" / "lstm.json"
        if not metrics_path.exists():
            pytest.skip("ยังไม่มีไฟล์ตัวชี้วัด — รัน train_lstm.py ก่อน")
        reference = json.loads(metrics_path.read_text(encoding="utf-8"))

        cases = [("lstm", "val"), ("lstm", "test")]
        if reference.get("baseline_logistic"):
            # baseline ในไฟล์ metrics มีแค่รายงานของ test (val ใช้เลือก threshold เท่านั้น)
            cases.append(("baseline", "test"))

        for model, split in cases:
            response = client.get(f"/api/forecast/confusion-matrix?model={model}&split={split}")
            if response.status_code != 200:
                pytest.skip(f"ยังไม่มีค่าทำนายของ {model}/{split}")
            data = response.json()

            expected = reference["lstm"][split] if model == "lstm" else reference["baseline_logistic"]
            idx = data["frozen_index"]
            counts = {
                "tp": data["tp"][idx], "fp": data["fp"][idx],
                "tn": data["tn"][idx], "fn": data["fn"][idx],
            }
            assert counts == {k: expected[k] for k in ("tp", "fp", "tn", "fn")}, (
                f"{model}/{split}: จำนวนนับจาก endpoint ไม่ตรงกับ lstm.json"
            )

    def test_switching_model_changes_frozen_threshold(self, client):
        lstm = client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        baseline = client.get("/api/forecast/confusion-matrix?model=baseline&split=test")
        if lstm.status_code != 200 or baseline.status_code != 200:
            pytest.skip("ยังไม่มีค่าทำนายครบทั้งสองโมเดล")

        assert lstm.json()["frozen_threshold"] != pytest.approx(
            baseline.json()["frozen_threshold"]
        )

    def test_switching_split_changes_sample_size(self, client):
        val = client.get("/api/forecast/confusion-matrix?model=lstm&split=val")
        test = client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        if val.status_code != 200 or test.status_code != 200:
            pytest.skip("ยังไม่มีค่าทำนายของ LSTM")

        assert val.json()["n"] != test.json()["n"]
        assert val.json()["n"] > 0
        assert test.json()["n"] > 0

    def test_unknown_model_is_rejected(self, client):
        response = client.get("/api/forecast/confusion-matrix?model=ไม่มีจริง")
        assert response.status_code == 422

    def test_samples_endpoint_returns_only_matching_cell(self, client):
        summary = client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        if summary.status_code != 200:
            pytest.skip("ยังไม่มีไฟล์ค่าทำนายราย sample")
        threshold = summary.json()["frozen_threshold"]

        response = client.get(
            "/api/forecast/confusion-matrix/samples"
            f"?model=lstm&split=test&threshold={threshold}&cell=fn&limit=500"
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["cell"] == "fn"

        # fn = ทำนาย negative (prob < threshold) แต่ label จริงคือ positive
        for sample in payload["samples"]:
            assert sample["probability"] < threshold
            assert 0.0 <= sample["probability"] <= 1.0
            assert sample["harpnum"] > 0

    def test_sample_totals_match_sweep_counts_at_frozen_threshold(self, client):
        """test สำคัญของ ticket นี้ — จำนวนที่ endpoint ลิสต์รายงานว่ามีทั้งหมด ต้อง
        ตรงกับจำนวนนับในตาราง sweep ที่ threshold เดียวกันเป๊ะ ทุกช่อง
        """
        summary_resp = client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        if summary_resp.status_code != 200:
            pytest.skip("ยังไม่มีไฟล์ค่าทำนายราย sample")
        summary = summary_resp.json()
        idx = summary["frozen_index"]
        threshold = summary["frozen_threshold"]

        for cell in ("tp", "fp", "fn", "tn"):
            response = client.get(
                "/api/forecast/confusion-matrix/samples"
                f"?model=lstm&split=test&threshold={threshold}&cell={cell}&limit=1"
            )
            assert response.status_code == 200
            assert response.json()["total"] == summary[cell][idx], f"cell={cell}"

    def test_samples_are_capped_and_sorted_by_probability_desc(self, client):
        response = client.get(
            "/api/forecast/confusion-matrix/samples"
            "?model=lstm&split=test&threshold=0.01&cell=tn&limit=10"
        )
        if response.status_code != 200:
            pytest.skip("ยังไม่มีไฟล์ค่าทำนายราย sample")
        payload = response.json()

        assert len(payload["samples"]) <= 10
        probs = [s["probability"] for s in payload["samples"]]
        assert probs == sorted(probs, reverse=True)

    def test_empty_cell_returns_empty_list_not_error(self, client):
        """threshold=0.0 ทำให้ทุก sample ถูกทำนายเป็น positive หมด — ช่อง fn/tn ต้องว่างเปล่า"""
        response = client.get(
            "/api/forecast/confusion-matrix/samples"
            "?model=lstm&split=test&threshold=0.0&cell=fn&limit=10"
        )
        if response.status_code != 200:
            pytest.skip("ยังไม่มีไฟล์ค่าทำนายราย sample")
        payload = response.json()

        assert payload["total"] == 0
        assert payload["samples"] == []

    def test_playing_with_slider_does_not_leak_into_live_forecast_threshold(self, client):
        """threshold ของแผงนี้ต้องไม่ไปเปลี่ยน threshold ที่ /api/forecast ใช้จริง"""
        health = client.get("/api/health").json()
        if not (health["forecast_model"] and health["sequence_store"]):
            pytest.skip("ยังไม่มีโมเดลพยากรณ์")

        harps = client.get("/api/harps?limit=1").json()
        if not harps:
            pytest.skip("ยังไม่มี active region ให้ทดสอบ")
        harpnum = harps[0]["harpnum"]

        before = client.get(f"/api/forecast?harpnum={harpnum}&limit=5").json()["threshold"]
        client.get("/api/forecast/confusion-matrix?model=lstm&split=test")
        client.get("/api/forecast/confusion-matrix?model=baseline&split=val")
        after = client.get(f"/api/forecast?harpnum={harpnum}&limit=5").json()["threshold"]

        assert before == after


class TestIntensitySeries:
    """ความเข้มแสงราย AR ตามเวลา — กราฟฝั่งขวาของ dashboard"""

    def test_rejects_reversed_range(self, client):
        r = client.get("/api/intensity-series?start=2014-10-31&end=2014-10-01")
        assert r.status_code in (422, 503)

    def test_empty_range_explains_itself_instead_of_erroring(self, client):
        """ช่วงที่แคบกว่าระยะห่างระหว่างเฟรมเป็นสถานะปกติ ไม่ใช่ error

        เกิดตั้งแต่เปิดหน้าเว็บครั้งแรก เพราะช่วงเวลาถูกตั้งตามอายุของ AR ที่เลือก
        อัตโนมัติ ซึ่งมักสั้นกว่า cadence 7 วันของชุดข้อมูลหลัก
        """
        r = client.get("/api/intensity-series?start=1990-01-01&end=1990-01-02")
        if r.status_code == 503:
            pytest.skip("ยังไม่มีเฟรมภาพเลย")

        assert r.status_code == 200
        data = r.json()
        assert data["n_frames"] == 0
        assert data["tracks"] == []
        assert data["note"], "ต้องบอกผู้ใช้ว่าทำไมกราฟถึงว่าง"

    def test_series_align_with_times(self, client):
        """ทุกช่องต้องมีจำนวนค่าเท่ากับจำนวนเวลา ไม่งั้น frontend จะวาดเหลื่อมกัน"""
        r = client.get(
            "/api/intensity-series?start=2014-01-01&end=2014-12-31&use_ground_truth=true"
        )
        if r.status_code != 200:
            pytest.skip("ยังไม่มีเฟรมในช่วงนี้")

        data = r.json()
        assert data["n_frames"] >= 1
        assert {c["key"] for c in data["channels"]} == {"171", "304", "1600"}

        for track in data["tracks"]:
            assert track["n_points"] == len(track["times"])
            for values in track["series"].values():
                assert len(values) == len(track["times"])

    def test_tracks_are_ordered_by_size(self, client):
        r = client.get(
            "/api/intensity-series?start=2014-01-01&end=2014-12-31&use_ground_truth=true&top=4"
        )
        if r.status_code != 200:
            pytest.skip("ยังไม่มีเฟรมในช่วงนี้")

        tracks = r.json()["tracks"]
        assert len(tracks) <= 4
        assert [t["label"] for t in tracks] == [f"AR {i}" for i in range(1, len(tracks) + 1)]
        areas = [t["max_area_px"] for t in tracks]
        assert areas == sorted(areas, reverse=True)

    def test_warns_when_frames_are_far_apart(self, client):
        """ชุดหลักเก็บทุก 7 วัน — ต้องบอกผู้ใช้ว่าการจับคู่ AR เชื่อถือได้แค่ไหน"""
        r = client.get(
            "/api/intensity-series?start=2014-01-01&end=2014-12-31&use_ground_truth=true"
        )
        if r.status_code != 200 or r.json()["n_frames"] < 2:
            pytest.skip("ยังไม่มีเฟรมพอในช่วงนี้")
        assert r.json()["note"], "ช่วง cadence 7 วันควรมีคำเตือนกำกับ"
