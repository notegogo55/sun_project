"""ตรวจขอบเขตการค้นหา hyperparameter รายสถาปัตยกรรม (ticket 01)

ทดสอบ **โดยไม่รัน Optuna** — ป้อนค่าที่ตายตัวเข้าไปทุกมุมของขอบเขตแล้วดูว่าประกอบโมเดลได้จริง
เหตุผลคือความผิดพลาดที่แพงที่สุดของงานนี้คือความผิดพลาดที่เจอตอนชั่วโมงที่สองของการค้นหา:
ถ้ามีมุมใดของขอบเขตที่ประกอบไม่ได้ trial นั้นจะพังกลางทางแล้วงบของสถาปัตยกรรมนั้นจะไม่เท่าตัวอื่น
"""

from __future__ import annotations

import itertools
import sys

import optuna
import pytest
from conftest import load_script

from sunseg.config import load_forecast_config, resolve_architecture
from sunseg.models.forecast import POOLING_KINDS
from sunseg.models.forecast.registry import ARCHITECTURES, build_architecture
from sunseg.models.forecast.tcn import receptive_field

_tune = load_script("study/tune.py")
N_SEEDS = _tune.N_SEEDS
N_TRIALS = _tune.N_TRIALS
RISK_AVERSION = _tune.RISK_AVERSION
SEARCH_VARIANT = _tune.SEARCH_VARIANT
TCN_SHAPES = _tune.TCN_SHAPES
parse_args = _tune.parse_args
sample_entry = _tune.sample_entry

#: ความยาวลำดับจริงของ dataset งานนี้ — 8 timestep ที่ cadence 12 ชม. ตามค่าคงที่ใน
#: study/build_dataset.py ไม่ใช่ ``sequence.length`` ใน data.yaml ซึ่งเป็นของ pipeline production
SEQ_LEN = 8

#: ค่าต่อเนื่องถูกตรึงไว้ — สิ่งที่ทดสอบคือมุมของตัวเลือกแบบหมวดหมู่ ซึ่งเป็นที่ที่ประกอบไม่ได้
_CONTINUOUS = {
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "early_stop_patience": 12,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
}

#: ตัวเลือกแบบหมวดหมู่ของแต่ละสถาปัตยกรรม — ต้องตรงกับที่ ``sample_entry`` เรียกใช้
_CATEGORICAL: dict[str, dict[str, list]] = {
    "lstm": {"hidden_size": [16, 24, 32, 48, 64], "num_layers": [1, 2]},
    "tcn": {"tcn_shape": list(TCN_SHAPES), "channels": [16, 24, 32, 40]},
    "transformer": {
        "d_model": [16, 24, 32, 48],
        "n_heads": [2, 4],
        "ff_dim": [16, 32, 64],
        "n_layers": [1, 2],
    },
    "darnn": {"hidden_size": [16, 24, 32, 48]},
}


def _points(kind: str):
    """ทุกจุดของขอบเขตแบบหมวดหมู่ คูณกับทุกวิธี pooling"""
    grid = _CATEGORICAL[kind]
    names = list(grid)
    for values in itertools.product(*(grid[n] for n in names)):
        for pooling in POOLING_KINDS:
            yield {**_CONTINUOUS, "pooling": pooling, **dict(zip(names, values, strict=True))}


@pytest.fixture(scope="module")
def base():
    return load_forecast_config()


class TestEveryPointInTheSearchSpaceBuilds:
    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_builds_and_runs_forward(self, kind, base):
        """ทุกมุมของขอบเขตต้องประกอบเป็นโมเดลที่รับ input จริงได้

        มุมที่ประกอบไม่ได้ = trial ที่พังกลางการค้นหา = งบของสถาปัตยกรรมนั้นไม่เท่าตัวอื่น
        ซึ่งทำลายข้อตกลงหลักของ ADR 0001
        """
        import torch

        for params in _points(kind):
            entry = sample_entry(optuna.trial.FixedTrial(params), kind)
            cfg = resolve_architecture(base, entry)
            model = build_architecture(cfg.kind, 18, SEQ_LEN, cfg.model)

            assert model(torch.randn(2, SEQ_LEN, 18)).shape == (2,), f"{kind} {params}"

    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_every_pooling_is_reachable(self, kind, base):
        """ทุกสถาปัตยกรรมต้องเลือก pooling ได้ครบทุกแบบ — ถ้า LSTM ตัวเดียวทิ้ง attention ได้
        แต่ตัวอื่นทำไม่ได้ การเปรียบเทียบจะเอนไปทางเดียว (ADR 0001)"""
        reached = set()
        for pooling in POOLING_KINDS:
            entry = sample_entry(optuna.trial.FixedTrial({**_CONTINUOUS, "pooling": pooling,
                                                          **{k: v[0] for k, v in _CATEGORICAL[kind].items()}}), kind)
            reached.add(entry.model.pooling)

        assert reached == set(POOLING_KINDS)

    def test_unknown_architecture_is_rejected(self):
        with pytest.raises(ValueError, match="ไม่รู้จัก"):
            sample_entry(optuna.trial.FixedTrial(_CONTINUOUS | {"pooling": "attention"}), "gru")


class TestTCNShapesAreValidByConstruction:
    """receptive field ที่ไม่พอถูกตัดออกตั้งแต่นิยามขอบเขต ไม่ใช่ตัดทิ้งตอน trial —
    ไม่งั้น TCN จะใช้งบจริงน้อยกว่าตัวอื่น"""

    @pytest.mark.parametrize("name", list(TCN_SHAPES))
    def test_receptive_field_covers_the_window(self, name):
        kernel_size, dilations = TCN_SHAPES[name]
        assert receptive_field(kernel_size, dilations) >= SEQ_LEN

    def test_a_shape_that_does_not_cover_is_absent(self):
        """kernel 2 กับ dilation 1/2 ให้ RF 4 ซึ่งไม่พอสำหรับหน้าต่าง 8 — ต้องไม่มีในขอบเขต"""
        assert (2, (1, 2)) not in TCN_SHAPES.values()

    def test_the_space_offers_a_shape_that_fits_the_window_closely(self):
        """ขอบเขตต้องมีตัวเลือกที่พอดีกับหน้าต่าง ไม่ใช่ใหญ่เกินทุกตัว

        รอบแรกออกแบบผิด: ทุกรูปทรงมี RF 31-63 บนหน้าต่างยาว 8 ชั้น dilation ท้าย ๆ จึงเห็น
        แต่ padding และ TCN ไม่เคยได้ลองรูปทรงที่เหมาะกับข้อมูลเลย
        """
        fields = [receptive_field(k, d) for k, d in TCN_SHAPES.values()]

        assert min(fields) <= SEQ_LEN * 2, f"RF ที่เล็กที่สุดคือ {min(fields)} ซึ่งเกินหน้าต่าง {SEQ_LEN} ไปมาก"


class TestLockedKnobsStayOutOfTheSearch:
    @pytest.mark.parametrize("kind", sorted(ARCHITECTURES))
    def test_only_permitted_training_knobs_are_overridden(self, kind, base):
        """batch_size / scheduler / warmup / grad_clip ต้องเหมือนกันทุกแถวของตาราง
        ไม่งั้นแยกไม่ออกว่าผลต่างมาจากสถาปัตยกรรมหรือจากตารางการเทรน"""
        params = {**_CONTINUOUS, "pooling": "attention", **{k: v[0] for k, v in _CATEGORICAL[kind].items()}}
        cfg = resolve_architecture(base, sample_entry(optuna.trial.FixedTrial(params), kind))

        assert cfg.train.batch_size == base.train.batch_size
        assert cfg.train.scheduler == base.train.scheduler
        assert cfg.train.warmup_epochs == base.train.warmup_epochs
        assert cfg.train.grad_clip == base.train.grad_clip
        assert cfg.eval == base.eval


class TestBudgetIsEqualByConstruction:
    def test_budget_is_a_single_module_level_value(self):
        """งบเป็นค่าเดียวของทั้งโมดูล ไม่ใช่ตารางรายสถาปัตยกรรม — ถ้าเป็น dict เมื่อไหร่
        แปลว่ามีคนเปิดช่องให้ตัวใดตัวหนึ่งได้ค้นหามากกว่าเพื่อน"""
        assert isinstance(N_TRIALS, int) and N_TRIALS > 0
        assert isinstance(N_SEEDS, int) and N_SEEDS > 0
        assert isinstance(RISK_AVERSION, float)

    def test_seeds_cannot_be_set_per_architecture_from_the_command_line(self):
        """--n-seeds ต้องไม่มี ไม่งั้นสั่งให้ตัวหนึ่งได้ seed มากกว่าอีกตัวได้โดยไม่มีใครเห็น"""
        sys.argv = ["study/tune.py", "--architecture", "lstm"]
        args = parse_args()

        assert not hasattr(args, "n_seeds")
        assert args.architecture == "lstm"

    def test_search_runs_on_the_neutral_feature_set(self):
        """ค้นหาบนชุดที่เป็นกลาง ไม่ใช่ชุดที่คาดว่าจะชนะ"""
        assert SEARCH_VARIANT == "V0"
