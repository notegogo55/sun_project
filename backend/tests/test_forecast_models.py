"""โมเดลพยากรณ์หลายตัวใน production — config, checkpoint, การโหลด และค่าทำนายรายโมเดล

จุดที่พังแล้วเสียหายที่สุดคือ **สัญญาระหว่างฝั่งเขียนกับฝั่งอ่าน**: สคริปต์เทรนเขียน checkpoint
แบบหนึ่ง แอปอ่านอีกแบบ แล้วหน้าเว็บขึ้นว่า "ยังไม่มีโมเดล" ทั้งที่เทรนเสร็จแล้ว ไฟล์นี้จึงทดสอบด้วย
``checkpoint_config`` ตัวจริงของ ``scripts/forecast/train.py`` ไม่ใช่ dict ที่เขียนมือ
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from conftest import load_script

from sunseg.artifacts import ForecastArtifacts
from sunseg.config import load_forecast_config, load_study_architectures, resolve_architecture
from sunseg.data.build_sequences import apply_normalisation
from sunseg.inference.forecast import ForecastModels, ForecastService, model_config_from_checkpoint
from sunseg.inference.predictions_store import ForecastPredictions
from sunseg.models.forecast import ARCHITECTURES, build_architecture
from sunseg.training.utils import save_checkpoint

#: หน้าต่างของ dataset production (``sequence.length`` ใน data.yaml) — 24 timestep x 18 SHARP
SEQ_LEN = 24
N_FEATURES = 18

train_script = load_script("forecast/train.py")


@pytest.fixture(scope="module")
def forecast_cfg():
    return load_forecast_config()


def _write_checkpoint(root, name, cfg, n_features=N_FEATURES, seq_len=SEQ_LEN, threshold=0.3):
    """เขียน checkpoint ด้วยฟังก์ชันเดียวกับที่สคริปต์เทรนใช้"""
    torch.manual_seed(0)
    model = build_architecture(cfg.kind, n_features, seq_len, cfg.model)
    splits = SimpleNamespace(n_features=n_features, seq_len=seq_len, features=[f"F{i}" for i in range(n_features)])
    data_cfg = SimpleNamespace(flare=SimpleNamespace(horizon_hours=24, positive_goes_class="M1.0"))
    save_checkpoint(
        ForecastArtifacts(name, root).checkpoint,
        model,
        config=train_script.checkpoint_config(name, cfg, splits, threshold, data_cfg),
        metrics={"val": {"tss": 0.5}, "test": {"tss": 0.4, "auc": 0.9}},
        extra={"norm_mean": np.zeros(n_features, np.float32), "norm_std": np.ones(n_features, np.float32)},
    )
    return model


class TestForecastConfig:
    def test_has_one_model_per_architecture_and_lstm_is_default(self, forecast_cfg):
        kinds = sorted(forecast_cfg.models[n].model.kind for n in forecast_cfg.names)
        assert kinds == sorted(ARCHITECTURES)
        assert forecast_cfg.default_model == "lstm"

    @pytest.mark.parametrize("name", ["lstm", "tcn", "transformer", "darnn"])
    def test_every_model_builds_on_the_production_window(self, forecast_cfg, name):
        """หน้าต่าง production ยาว 24 — TCN ต้องมี receptive field คลุม (FlareTCN raise ถ้าไม่พอ)"""
        cfg = forecast_cfg.resolve(name)
        model = build_architecture(cfg.kind, N_FEATURES, SEQ_LEN, cfg.model)
        logits, attention = model(torch.randn(3, SEQ_LEN, N_FEATURES), return_attention=True)
        assert logits.shape == (3,)
        assert attention.shape == (3, SEQ_LEN)

    def test_every_model_matches_its_tuned_study_entry(self, forecast_cfg):
        """ระบบใช้ค่าจากการค้นหาทั้งสี่ตัว (ตั้งแต่ 2026-09-23) — ต้องตรงกับแถวของงานเปรียบเทียบทุกค่า
        ไม่งั้นตัวเลขของงานเปรียบเทียบจะอ้างถึงแบบจำลองในระบบไม่ได้"""
        study = load_study_architectures()
        for name in ("lstm", "tcn", "transformer", "darnn"):
            system = forecast_cfg.resolve(name)
            tuned = resolve_architecture(forecast_cfg, study.architectures[name])
            assert system.model_dump() == tuned.model_dump(), name

    def test_untuned_lstm_row_keeps_the_original_hyperparameters(self, forecast_cfg):
        """ค่าตั้งมือเดิมของ LSTM (configs/lstm.yaml) อยู่ต่อในงานเปรียบเทียบเป็นแถวเสริม lstm_production
        ห้ามเปลี่ยน — signature ของ run ทั้ง 100 ตัวใน artifacts/model_comparison ผูกกับค่าเหล่านี้"""
        cfg = resolve_architecture(forecast_cfg, load_study_architectures().architectures["lstm_production"])
        assert (cfg.model.hidden_size, cfg.model.num_layers, cfg.model.dropout) == (32, 1, 0.4)
        assert cfg.model.pooling == "attention"
        assert (cfg.train.lr, cfg.train.weight_decay, cfg.train.epochs, cfg.train.batch_size) == (5e-4, 1e-3, 60, 256)
        assert (cfg.loss.focal_alpha, cfg.loss.focal_gamma) == (0.75, 2.0)

    def test_overrides_only_touch_what_they_name(self, forecast_cfg):
        tcn = forecast_cfg.resolve("tcn")
        assert tcn.train.lr != forecast_cfg.train.lr
        # ค่าที่ไม่เปิดให้ทับต้องเหมือนค่าร่วมทุกตัว
        assert tcn.train.batch_size == forecast_cfg.train.batch_size
        assert tcn.train.scheduler == forecast_cfg.train.scheduler

    def test_unknown_name_is_rejected(self, forecast_cfg):
        with pytest.raises(ValueError, match="ไม่มีโมเดล"):
            forecast_cfg.resolve("gru")


class TestCheckpointRoundTrip:
    @pytest.mark.parametrize("name", ["lstm", "tcn", "transformer", "darnn"])
    def test_service_reproduces_the_trained_model(self, tmp_path, forecast_cfg, name):
        cfg = forecast_cfg.resolve(name)
        model = _write_checkpoint(tmp_path, name, cfg)

        service = ForecastService(ForecastArtifacts(name, tmp_path).checkpoint, name=name)
        assert service.available
        assert service.kind == cfg.kind
        assert service.seq_len == SEQ_LEN

        x = np.random.default_rng(0).normal(size=(5, SEQ_LEN, N_FEATURES)).astype(np.float32)
        probs, attention = service.predict_batch(x)
        model.eval()
        with torch.no_grad():
            # predict_batch รับค่าดิบแล้ว normalise เอง (signed_log1p + mean/std จาก checkpoint)
            normalised = apply_normalisation(x, service.stats)
            expected = torch.sigmoid(model(torch.from_numpy(normalised))).numpy()
        np.testing.assert_allclose(probs, expected, rtol=1e-5, atol=1e-6)
        assert attention.shape == (5, SEQ_LEN)

    def test_legacy_lstm_checkpoint_without_kind_still_loads(self, tmp_path):
        """lstm.pt ที่เทรนก่อนรองรับหลายโมเดล: ไม่มี kind/seq_len และใช้ use_attention"""
        from sunseg.models.forecast import FlareLSTM

        model = FlareLSTM(n_features=N_FEATURES, hidden_size=32, num_layers=1, dropout=0.4, pooling="attention")
        path = tmp_path / "models" / "lstm.pt"
        save_checkpoint(
            path, model,
            config={
                "n_features": N_FEATURES,
                "features": [f"F{i}" for i in range(N_FEATURES)],
                "model": {"hidden_size": 32, "num_layers": 1, "dropout": 0.4,
                          "bidirectional": False, "use_attention": True},
                "threshold": 0.2, "horizon_hours": 24, "positive_class": "M1.0",
            },
            metrics={"test": {"tss": 0.69}},
            extra={"norm_mean": np.zeros(N_FEATURES), "norm_std": np.ones(N_FEATURES)},
        )
        service = ForecastService(path, name="lstm")
        assert service.available and service.kind == "lstm"
        assert service.model.pooling == "attention"

    @pytest.mark.parametrize(("use_attention", "pooling"), [(True, "attention"), (False, "last")])
    def test_use_attention_maps_to_pooling(self, use_attention, pooling):
        kind, cfg = model_config_from_checkpoint(
            {"model": {"hidden_size": 8, "num_layers": 1, "dropout": 0.1, "use_attention": use_attention}}
        )
        assert kind == "lstm" and cfg.pooling == pooling


class TestForecastModels:
    def test_untrained_models_stay_listed_with_a_hint(self, tmp_path, forecast_cfg):
        _write_checkpoint(tmp_path, "tcn", forecast_cfg.resolve("tcn"))
        models = ForecastModels(forecast_cfg, tmp_path)

        assert models.names == forecast_cfg.names
        assert models.any_available
        assert not models.default.available  # lstm ยังไม่ได้เทรนใน tmp_path
        rows = {r["name"]: r for r in models.summaries()}
        assert rows["tcn"]["available"] and rows["tcn"]["train_hint"] is None
        assert rows["tcn"]["test"]["tss"] == 0.4
        assert not rows["darnn"]["available"]
        assert "--model darnn" in rows["darnn"]["train_hint"]

    def test_get_without_name_is_the_default_and_unknown_raises(self, tmp_path, forecast_cfg):
        models = ForecastModels(forecast_cfg, tmp_path)
        assert models.get().name == forecast_cfg.default_model
        with pytest.raises(KeyError):
            models.get("gru")


def _write_predictions(root, name, prob, baseline_prob, legacy=False):
    artifacts = ForecastArtifacts(name, root)
    frame = pd.DataFrame({
        "split": ["val", "val", "test", "test"],
        "HARPNUM": [1, 2, 3, 4],
        "noaa_ar": [11, 12, 13, 14],
        "issue_time": pd.to_datetime(["2020-01-01"] * 4),
        "label": [0, 1, 0, 1],
        artifacts.prob_column: prob,
        "baseline_prob": baseline_prob,
    })
    path = root / "metrics" / "predictions.parquet" if legacy else artifacts.predictions
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    artifacts.metrics.write_text(json.dumps({
        name: {"val": {"threshold": 0.5}, "test": {}},
        "baseline_logistic": {"threshold": 0.4},
    }), encoding="utf-8")


class TestPredictions:
    def test_legacy_lstm_predictions_file_is_still_read(self, tmp_path):
        _write_predictions(tmp_path, "lstm", [0.1, 0.9, 0.2, 0.8], [0.3] * 4, legacy=True)
        assert ForecastArtifacts("lstm", tmp_path).existing_predictions().name == "predictions.parquet"
        preds = ForecastPredictions.from_artifacts({"lstm": "LSTM"}, tmp_path, "lstm")
        assert preds.sweep("lstm", "test")["n"] == 2

    def test_each_model_reads_its_own_file_and_baseline_prefers_default(self, tmp_path):
        _write_predictions(tmp_path, "lstm", [0.1, 0.9, 0.2, 0.8], [0.3, 0.3, 0.3, 0.3])
        _write_predictions(tmp_path, "tcn", [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7])
        preds = ForecastPredictions.from_artifacts({"lstm": "LSTM", "tcn": "TCN"}, tmp_path, "lstm")

        tcn = preds.sweep("tcn", "test")
        assert tcn["model"] == "tcn" and tcn["frozen_threshold"] == 0.5
        baseline = preds.sweep("baseline", "test")
        assert baseline["model"] == "baseline" and baseline["frozen_threshold"] == 0.4
        rows, _ = preds.samples_in_cell("baseline", "test", 0.5, "fn")
        # baseline ต้องมาจากไฟล์ของโมเดลปริยาย (0.3 ทุกแถว) ไม่ใช่ของ tcn (0.7)
        assert set(rows["prob"]) == {0.3}

    def test_untrained_model_raises_runtime_error_with_hint(self, tmp_path):
        preds = ForecastPredictions.from_artifacts({"lstm": "LSTM", "darnn": "DA-RNN"}, tmp_path, "lstm")
        with pytest.raises(RuntimeError, match="--model darnn"):
            preds.sweep("darnn", "test")
        with pytest.raises(KeyError):
            preds.sweep("gru", "test")


class TestTrainScript:
    def test_all_expands_to_every_model_in_config_order(self, forecast_cfg):
        assert train_script.resolve_names("all", forecast_cfg.names) == forecast_cfg.names

    def test_comma_list_and_unknown_name(self, forecast_cfg):
        assert train_script.resolve_names("tcn, darnn", forecast_cfg.names) == ["tcn", "darnn"]
        with pytest.raises(ValueError, match="gru"):
            train_script.resolve_names("lstm,gru", forecast_cfg.names)
