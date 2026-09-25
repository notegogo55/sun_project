"""ตรวจทะเบียนสถาปัตยกรรมและไฟล์ config ของงานเปรียบเทียบ

จับข้อผิดพลาดที่แพงที่สุดของงานนี้: ตารางใช้เวลารันสิบชั่วโมง ความผิดพลาดอย่าง
ชื่อแบบสะกดผิด, receptive field ไม่คลุมหน้าต่าง หรือสถาปัตยกรรมที่ประกอบไม่ได้
ต้องระเบิดที่นี่ ไม่ใช่ชั่วโมงที่หกของการรัน
"""

from __future__ import annotations

import pytest
import torch

from sunseg.config import (
    ArchitectureEntry,
    LSTMArchConfig,
    load_forecast_config,
    load_study_architectures,
    resolve_architecture,
)
from sunseg.data.study_dataset import load_variants
from sunseg.models.forecast import POOLING_KINDS
from sunseg.models.forecast.registry import ARCHITECTURES, architecture_label, build_architecture
from sunseg.models.forecast.tcn import FlareTCN, receptive_field

#: ความยาวลำดับจริงของ dataset งานนี้ — 8 timestep ที่ cadence 12 ชม. ตามค่าคงที่ใน
#: study/build_dataset.py ไม่ใช่ ``sequence.length`` ใน data.yaml ซึ่งเป็นของ pipeline production
SEQ_LEN = 8
#: จำนวน feature ของสามแบบในตาราง: 18 SHARP / +X-ray / +X-ray +intensity
FEATURE_COUNTS = (18, 19, 22)


@pytest.fixture(scope="module")
def study():
    return load_study_architectures()


@pytest.fixture(scope="module")
def base():
    return load_forecast_config()


def _configs(study, base):
    return {name: resolve_architecture(base, entry) for name, entry in study.architectures.items()}


class TestEveryArchitectureBuilds:
    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    @pytest.mark.parametrize("n_features", FEATURE_COUNTS)
    def test_forward_gives_one_logit_per_sample(self, kind, n_features, study, base):
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, n_features, SEQ_LEN, cfg.model)

        assert model(torch.randn(4, SEQ_LEN, n_features)).shape == (4,)

    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_attention_weights_sum_to_one(self, kind, study, base):
        """ทุกสถาปัตยกรรมต้องคืนน้ำหนักรายชั่วโมงที่ตีความได้ ไม่ใช่ศูนย์ล้วน —
        รายงานและหน้าเว็บวาดจากค่านี้โดยไม่ต้องรู้ว่าข้างในเป็น pooling แบบไหน"""
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, 18, SEQ_LEN, cfg.model)

        _, attention = model(torch.randn(4, SEQ_LEN, 18), return_attention=True)

        assert attention.shape == (4, SEQ_LEN)
        assert torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-5)

    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_gradients_reach_every_parameter(self, kind, study, base):
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, 18, SEQ_LEN, cfg.model)

        model(torch.randn(4, SEQ_LEN, 18)).sum().backward()

        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"{kind}: พารามิเตอร์เหล่านี้ไม่ได้รับ gradient: {missing}"

    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    @pytest.mark.parametrize("pooling", POOLING_KINDS)
    def test_every_pooling_works_on_every_architecture(self, kind, pooling, study, base):
        """pooling เป็น hyperparameter ของทุกตัว — ถ้าตัวใดรองรับไม่ครบ การค้นหาจะ
        เจอขอบเขตไม่เท่ากันโดยที่ไม่มีใครเห็น (ดู ADR 0001)"""
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, 18, SEQ_LEN, cfg.model.model_copy(update={"pooling": pooling}))

        assert model(torch.randn(2, SEQ_LEN, 18)).shape == (2,)

    def test_unknown_architecture_is_rejected(self):
        with pytest.raises(ValueError, match="ไม่รู้จัก"):
            build_architecture("gru", 18, SEQ_LEN, None)


class TestParameterBudget:
    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_stays_near_the_reference_size(self, kind, study, base):
        """ทุกตัวต้องอยู่ใกล้ขนาดของ LSTM ที่ production ใช้ (~7.8k ที่ 18 feature)

        หลังการค้นหา ขนาดต่างกันได้ตามธรรมชาติและต่างกันจริง (ADR 0001) — ที่ 18 feature
        ได้ lstm 7,265 · tcn 6,409 · transformer 5,097 · darnn 2,809 · production 7,810
        เกณฑ์ที่นี่จึงเป็นแถบกว้าง ๆ ไว้จับกรณีที่มีคนแก้ config แล้วขนาดหลุดไปคนละอันดับ
        ไม่ใช่การบังคับให้เท่ากัน
        """
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, 18, SEQ_LEN, cfg.model)

        assert 1_000 <= model.count_parameters() <= 20_000


class TestTCNReceptiveField:
    def test_formula(self):
        assert receptive_field(3, (1, 2, 4)) == 15
        assert receptive_field(3, (1, 2, 4, 8)) == 31

    def test_rejects_field_shorter_than_the_window(self):
        """1/2/4 ที่ kernel 3 ให้ RF 15 ซึ่งมองไม่ถึงต้นหน้าต่าง 24 ชม. — ถ้าปล่อยผ่าน
        TCN จะแพ้ในตารางด้วยเหตุผลที่ไม่เกี่ยวกับสถาปัตยกรรม"""
        with pytest.raises(ValueError, match="receptive field"):
            FlareTCN(n_features=18, seq_len=24, kernel_size=3, dilations=(1, 2, 4))

    def test_is_causal(self):
        """เปลี่ยนค่าที่ timestep สุดท้ายต้องไม่กระทบผลลัพธ์ของ timestep ก่อนหน้า"""
        model = FlareTCN(n_features=4, seq_len=8, channels=6, dilations=(1, 2, 4), pooling="last").eval()
        x = torch.randn(1, 8, 4)
        changed = x.clone()
        changed[0, -1] += 10.0

        with torch.no_grad():
            a = model.blocks(x.transpose(1, 2))
            b = model.blocks(changed.transpose(1, 2))

        assert torch.allclose(a[:, :, :-1], b[:, :, :-1], atol=1e-5)
        assert not torch.allclose(a[:, :, -1], b[:, :, -1], atol=1e-5)


class TestFixedLengthArchitectures:
    @pytest.mark.parametrize("kind", ["transformer", "darnn"])
    def test_rejects_a_different_sequence_length(self, kind, study, base):
        """ทั้งสองตัวมีพารามิเตอร์ที่ผูกกับตำแหน่ง — ลำดับยาวไม่ตรงต้อง raise
        ไม่ใช่เงียบ ๆ ใช้ตำแหน่งผิด"""
        entry = next(e for e in study.architectures.values() if e.model.kind == kind)
        cfg = resolve_architecture(base, entry)
        model = build_architecture(kind, 18, SEQ_LEN, cfg.model)

        with pytest.raises(ValueError, match="ความยาวลำดับไม่ตรง"):
            model(torch.randn(2, SEQ_LEN + 1, 18))


class TestDARNNFeatureAttention:
    def test_weights_sum_to_one_per_timestep(self, study, base):
        """เหตุผลเดียวที่ DA-RNN อยู่ในตาราง: มันบอกได้ว่า feature ตัวไหนถูกใช้
        ซึ่งแยกคำถาม 'X-ray ไม่มีข้อมูล' ออกจาก 'โมเดลไม่ได้ใช้ X-ray'"""
        cfg = resolve_architecture(base, study.architectures["darnn"])
        model = build_architecture("darnn", 22, SEQ_LEN, cfg.model)

        weights = model.feature_attention(torch.randn(3, SEQ_LEN, 22))

        assert weights.shape == (3, SEQ_LEN, 22)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(3, SEQ_LEN), atol=1e-5)


class TestResolveArchitecture:
    def test_override_replaces_only_what_it_names(self, base):
        entry = ArchitectureEntry(
            model=LSTMArchConfig(kind="lstm", hidden_size=16, dropout=0.2),
            train={"lr": 0.003},
        )

        cfg = resolve_architecture(base, entry)

        assert cfg.train.lr == 0.003
        assert cfg.train.batch_size == base.train.batch_size
        assert cfg.train.weight_decay == base.train.weight_decay
        assert cfg.loss.focal_alpha == base.loss.focal_alpha
        assert cfg.eval == base.eval

    def test_base_config_is_not_mutated(self, base):
        before = base.train.lr
        resolve_architecture(base, ArchitectureEntry(
            model=LSTMArchConfig(kind="lstm", hidden_size=16, dropout=0.2), train={"lr": 0.003}
        ))
        assert base.train.lr == before


class TestStudyArchitecturesFile:
    def test_every_architecture_is_known_to_the_registry(self, study):
        unknown = [name for name, e in study.architectures.items() if e.model.kind not in ARCHITECTURES]
        assert not unknown, f"kind ที่ทะเบียนไม่รู้จัก: {unknown}"

    def test_every_variant_named_exists(self, study):
        """ชื่อแบบสะกดผิดต้องเจอที่นี่ ไม่ใช่หลังรันไปแล้วสองชั่วโมง"""
        variants = load_variants(_variants_file())
        named = set(study.default_variants)
        for entry in study.architectures.values():
            named |= set(entry.variants or ())

        missing = sorted(named - set(variants))
        assert not missing, f"แบบที่อ้างถึงแต่ไม่มีใน study/variants.yaml: {missing}"

    def test_default_variants_are_the_four_columns_in_id_order(self, study):
        """เรียงตามเลข ID (V0, V1, V2, V3) ให้ตรงกับรายงานทุกตารางและทุกรูป — ลำดับนี้คือสิ่งที่ไปปรากฏ
        เป็นคอลัมน์ของตาราง ส่วนความหมาย 'สองคอลัมน์กลางเพิ่มคนละหนึ่งกลุ่ม' ตรวจแยกในเทสต์ถัดไป"""
        assert study.default_variants == ["V0", "V1", "V2", "V3"]

    def test_the_two_middle_columns_add_one_feature_group_each(self, study):
        """คอลัมน์กลางสองอันต้องเทียบกับ baseline เดียวกัน คืออันละหนึ่งกลุ่ม feature
        ไม่ใช่สะสมกัน — ไม่งั้นตารางตอบคำถาม 'แต่ละกลุ่มช่วยอะไรแยกกัน' ไม่ได้"""
        variants = load_variants(_variants_file())
        base = set(variants["V0"]["columns"])

        for name in study.default_variants[1:3]:
            added = set(variants[name]["columns"]) - base
            groups = {"xray" if c.startswith("xray_") else "intensity" for c in added}
            assert len(groups) == 1, f"{name} เพิ่ม feature มากกว่าหนึ่งกลุ่ม: {groups}"

    def test_main_table_has_one_row_per_architecture_kind(self, study):
        kinds = [e.model.kind for e in study.architectures.values() if not e.supplementary]
        assert sorted(kinds) == sorted(ARCHITECTURES), (
            "ตารางหลักต้องมีครบทุกสถาปัตยกรรมและไม่ซ้ำ — แถวที่ซ้ำ kind ต้องตั้ง supplementary: true"
        )

    def test_labels_are_distinct(self, study):
        labels = [e.label or architecture_label(e.model.kind) for e in study.architectures.values()]
        assert len(labels) == len(set(labels)), f"ชื่อซ้ำกันในตาราง: {labels}"


def _variants_file():
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "configs" / "study" / "variants.yaml"


class TestEvidenceStudyConfig:
    """งาน feature-evidence-cv คัดลอก LSTM มาไว้ไฟล์แยก (spec หัวข้อ 5: ห้ามค้นหาใหม่) — ต้องไม่เพี้ยนจากไฟล์หลัก"""

    @pytest.mark.parametrize("name", ["architectures_evidence.yaml", "architectures_v3_aia.yaml"])
    def test_lstm_hyperparameters_match_the_main_study_file(self, study, name):
        from sunseg.config import STUDY_CONFIG_DIR

        other = load_study_architectures(STUDY_CONFIG_DIR / name)
        main, copy = study.architectures["lstm"], other.architectures["lstm"]
        assert copy.model == main.model
        assert copy.train == main.train
        assert copy.loss == main.loss

    def test_v3_in_the_aia_comparison_is_the_main_v3(self):
        from sunseg.config import STUDY_CONFIG_DIR

        main = load_variants(STUDY_CONFIG_DIR / "variants.yaml")["V3"]["columns"]
        aia = load_variants(STUDY_CONFIG_DIR / "variants_v3_aia.yaml")
        assert aia["V3"]["columns"] == main
        assert aia["V3A"]["columns"] == [*main, "94_p95", "131_p95"]

    def test_every_evidence_variant_is_defined(self):
        from sunseg.config import STUDY_CONFIG_DIR

        evidence = load_study_architectures(STUDY_CONFIG_DIR / "architectures_evidence.yaml")
        variants = load_variants(STUDY_CONFIG_DIR / "variants_evidence.yaml")
        assert set(evidence.architectures["lstm"].variants) <= set(variants)
        assert set(evidence.default_variants) <= set(variants)
