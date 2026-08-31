"""ตรวจสอบสถาปัตยกรรมโมเดลและ loss

จับข้อผิดพลาดที่พบบ่อยและเสียเวลามากถ้าไปเจอตอนเทรน: รูปทรงไม่ตรง, gradient
ไม่ไหล, loss ไม่ลงโทษ positive อย่างที่ควร
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sunseg.losses import BceDiceLoss, DiceLoss, FocalLoss, TverskyLoss
from sunseg.models.lstm import FlareLSTM
from sunseg.models.unet import UNet, normalise_magnetogram


class TestFlareLSTM:
    @pytest.fixture
    def model(self):
        return FlareLSTM(n_features=18, hidden_size=32, num_layers=1)

    def test_output_shape_is_one_logit_per_sample(self, model):
        logits = model(torch.randn(8, 24, 18))
        assert logits.shape == (8,)

    def test_returns_attention_weights_that_sum_to_one(self, model):
        """attention เป็นผลลัพธ์ของ softmax จึงต้องรวมกันได้ 1 ในทุก sample"""
        _, attention = model(torch.randn(4, 24, 18), return_attention=True)

        assert attention.shape == (4, 24)
        assert torch.allclose(attention.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_rejects_wrong_feature_count(self, model):
        with pytest.raises(ValueError, match="จำนวน feature ไม่ตรง"):
            model(torch.randn(4, 24, 7))

    def test_rejects_wrong_dimensionality(self, model):
        with pytest.raises(ValueError, match="3 มิติ"):
            model(torch.randn(4, 18))

    def test_accepts_any_sequence_length(self, model):
        """LSTM ต้องรับความยาวลำดับเท่าใดก็ได้ — ทำให้ทดลองเปลี่ยนหน้าต่างเวลาได้ง่าย"""
        for length in (6, 12, 24, 48):
            assert model(torch.randn(2, length, 18)).shape == (2,)

    def test_gradients_reach_every_parameter(self, model):
        """พารามิเตอร์ที่ไม่ได้รับ gradient แปลว่ามีชิ้นส่วนที่หลุดจากกราฟการคำนวณ"""
        loss = model(torch.randn(4, 24, 18)).sum()
        loss.backward()

        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"พารามิเตอร์เหล่านี้ไม่ได้รับ gradient: {missing}"

    def test_predict_proba_is_in_unit_range(self, model):
        probs = model.predict_proba(torch.randn(16, 24, 18))
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_forget_gate_bias_starts_at_one(self, model):
        """ตั้ง forget gate bias = 1 เพื่อให้จำระยะยาวได้ตั้งแต่ต้น (Jozefowicz 2015)"""
        hidden = model.hidden_size
        bias = model.lstm.bias_ih_l0.data
        assert torch.allclose(bias[hidden : 2 * hidden], torch.ones(hidden))

    def test_small_model_has_few_parameters(self):
        """ขนาดโมเดลต้องเล็กเพราะ effective sample size จริงมีเพียง ~114 HARP"""
        model = FlareLSTM(n_features=18, hidden_size=32, num_layers=1)
        assert model.count_parameters() < 20_000


class TestUNet:
    @pytest.fixture
    def model(self):
        # base เล็กและตื้นเพื่อให้ test เร็ว — สถาปัตยกรรมเหมือนของจริงทุกประการ
        return UNet(base_channels=4, depth=2)

    def test_output_has_same_spatial_size_as_input(self, model):
        assert model(torch.randn(2, 1, 64, 64)).shape == (2, 1, 64, 64)

    @pytest.mark.parametrize("size", [32, 64, 96])
    def test_handles_various_input_sizes(self, model, size):
        assert model(torch.randn(1, 1, size, size)).shape == (1, 1, size, size)

    def test_handles_size_not_divisible_by_two_to_the_depth(self, model):
        """การ pad ในชั้น Up ต้องรับมือกับขนาดที่หารไม่ลงตัวได้"""
        assert model(torch.randn(1, 1, 50, 50)).shape == (1, 1, 50, 50)

    def test_rejects_wrong_dimensionality(self, model):
        with pytest.raises(ValueError, match="4 มิติ"):
            model(torch.randn(1, 64, 64))

    def test_output_bias_starts_negative(self, model):
        """เริ่มด้วยการทายว่า 'ไม่ใช่ AR' เพราะ AR กินพื้นที่เพียง ~2% ของพิกเซล"""
        assert model.out_conv.bias.item() < -2.0

    def test_initial_predictions_are_mostly_negative(self, model):
        """ผลจากการตั้ง bias ข้างต้น — ป้องกัน loss ระเบิดใน epoch แรก"""
        probs = torch.sigmoid(model(torch.randn(2, 1, 64, 64)))
        assert probs.mean().item() < 0.2

    def test_gradients_reach_every_parameter(self, model):
        model(torch.randn(1, 1, 64, 64)).sum().backward()

        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"พารามิเตอร์เหล่านี้ไม่ได้รับ gradient: {missing}"

    def test_predict_mask_returns_binary(self, model):
        mask = model.predict_mask(torch.randn(1, 1, 64, 64))
        assert set(torch.unique(mask).tolist()) <= {0, 1}


class TestNormaliseMagnetogram:
    def test_output_stays_within_unit_range(self):
        """tanh อิ่มตัวที่ ±1 พอดีใน float32 เมื่อสนามแรงมาก — ขอบเขตจึงเป็นแบบรวมปลาย"""
        data = np.array([-5000.0, -300.0, 0.0, 300.0, 5000.0], dtype=np.float32)
        result = normalise_magnetogram(data)
        assert np.all(np.abs(result) <= 1.0)

    def test_preserves_magnetic_polarity(self):
        """ขั้วแม่เหล็กเป็นข้อมูลทางฟิสิกส์ — เครื่องหมายต้องไม่เปลี่ยน"""
        data = np.array([-1000.0, -10.0, 10.0, 1000.0], dtype=np.float32)
        result = normalise_magnetogram(data)
        assert np.array_equal(np.sign(result), np.sign(data))

    def test_zero_maps_to_zero(self):
        assert normalise_magnetogram(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_is_strictly_increasing_before_saturation(self):
        """ในช่วงที่ยังไม่อิ่มตัว (|B| < ~4 เท่าของ scale) ต้องเพิ่มขึ้นอย่างเข้มงวด

        เกินช่วงนี้ไป tanh จะอิ่มตัวจนค่าที่ติดกันเท่ากันใน float32 ซึ่งเป็น
        พฤติกรรมที่ต้องการ: เราไม่ต้องการแยกแยะ 2500 G กับ 3000 G เพราะทั้งคู่คือ
        "ใจกลางจุดดับ" เหมือนกัน
        """
        data = np.linspace(-1000, 1000, 100, dtype=np.float32)
        assert np.all(np.diff(normalise_magnetogram(data)) > 0)

    def test_saturates_for_very_strong_fields(self):
        data = np.array([2000.0, 3000.0], dtype=np.float32)
        result = normalise_magnetogram(data)
        assert np.all(result > 0.99)

    def test_resolves_weak_field_far_better_than_linear_scaling(self):
        """เหตุผลที่ใช้ tanh: quiet Sun (~10 G) ต้องไม่ถูกบีบจนหายไป

        การหารด้วยค่าสูงสุด 3000 G ให้ 10/3000 = 0.0033 (แทบมองไม่เห็น)
        ส่วน tanh(10/300) = 0.0333 ซึ่งมากกว่าราวสิบเท่า
        """
        weak = float(normalise_magnetogram(np.array([10.0]))[0])
        linear = 10.0 / 3000.0
        assert weak / linear > 9.0

    def test_works_with_torch_tensors(self):
        result = normalise_magnetogram(torch.tensor([-1000.0, 0.0, 1000.0]))
        assert isinstance(result, torch.Tensor)


class TestLosses:
    def test_focal_loss_penalises_confident_mistakes_most(self):
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        targets = torch.ones(4)

        confident_wrong = criterion(torch.full((4,), -5.0), targets)
        confident_right = criterion(torch.full((4,), 5.0), targets)
        assert confident_wrong > confident_right * 100

    def test_focal_loss_downweights_easy_examples(self):
        """หัวใจของ focal loss: gamma สูงขึ้นต้องกด loss ของตัวอย่างที่ทายถูกอยู่แล้ว"""
        logits = torch.full((8,), 3.0)
        targets = torch.ones(8)

        gamma_zero = FocalLoss(gamma=0.0)(logits, targets)
        gamma_two = FocalLoss(gamma=2.0)(logits, targets)
        assert gamma_two < gamma_zero

    def test_focal_loss_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            FocalLoss(alpha=1.5)

    def test_dice_loss_is_near_zero_for_perfect_prediction(self):
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:10, 4:10] = 1.0
        # logit ขนาดใหญ่ -> sigmoid เข้าใกล้ 0/1
        logits = (targets * 2 - 1) * 20.0

        assert DiceLoss()(logits, targets).item() < 0.01

    def test_dice_loss_is_near_one_for_inverted_prediction(self):
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:10, 4:10] = 1.0
        logits = (targets * 2 - 1) * -20.0

        assert DiceLoss()(logits, targets).item() > 0.9

    def test_tversky_with_equal_weights_matches_dice(self):
        """Tversky ที่ alpha=beta=0.5 เท่ากับ Dice ทางคณิตศาสตร์

        ``tp/(tp+0.5fp+0.5fn)`` เท่ากับ ``2tp/(2tp+fp+fn)`` เมื่อคูณทั้งเศษและส่วนด้วย 2
        ต้องใช้ ``smooth`` เล็กมากในการทดสอบ เพราะทั้งสองคลาสใส่พจน์นี้คนละตำแหน่ง
        (Dice ใส่ในสูตรที่คูณ 2 แล้ว) ซึ่งทำให้ต่างกันเล็กน้อยเมื่อ smooth ใหญ่
        """
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:10, 4:10] = 1.0
        logits = torch.randn(2, 1, 16, 16)

        tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=1e-6)(logits, targets)
        dice = DiceLoss(smooth=1e-6)(logits, targets)
        assert tversky.item() == pytest.approx(dice.item(), abs=1e-4)

    def test_tversky_beta_penalises_false_negatives_harder(self):
        """ตั้ง beta > alpha เพื่อให้ 'พลาด AR ทั้งดวง' แย่กว่า 'ระบายเกินขอบ'"""
        targets = torch.zeros(1, 1, 16, 16)
        targets[:, :, 4:12, 4:12] = 1.0

        miss_everything = torch.full((1, 1, 16, 16), -20.0)   # false negative ล้วน
        predict_everything = torch.full((1, 1, 16, 16), 20.0)  # false positive ล้วน

        criterion = TverskyLoss(alpha=0.3, beta=0.7)
        assert criterion(miss_everything, targets) > criterion(predict_everything, targets)

    def test_bce_dice_combines_both_terms(self):
        targets = torch.zeros(2, 1, 16, 16)
        targets[:, :, 4:10, 4:10] = 1.0
        logits = torch.randn(2, 1, 16, 16)

        combined = BceDiceLoss(bce_weight=0.5, dice_weight=0.5, pos_weight=20.0)(logits, targets)
        assert torch.isfinite(combined) and combined.item() > 0

    def test_bce_dice_pos_weight_increases_cost_of_missing_positives(self):
        targets = torch.zeros(1, 1, 16, 16)
        targets[:, :, 4:10, 4:10] = 1.0
        missing = torch.full((1, 1, 16, 16), -10.0)

        low = BceDiceLoss(pos_weight=1.0)(missing, targets)
        high = BceDiceLoss(pos_weight=20.0)(missing, targets)
        assert high > low
