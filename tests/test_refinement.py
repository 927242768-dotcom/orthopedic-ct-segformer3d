import torch

from src.modeling.refinement import (
    UncertaintyRefinementNet3D,
    blend_refinement_logits,
    canonical_binary_logits_from_prediction_entropy,
)
from src.modeling.uncertainty import predictive_entropy


def test_canonical_binary_logits_recover_prediction_probability_and_entropy() -> None:
    torch.manual_seed(11)
    logits = torch.randn((1, 2, 7, 8, 9), dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1)
    entropy = predictive_entropy(logits)

    restored = canonical_binary_logits_from_prediction_entropy(prediction, entropy)

    assert torch.equal(torch.argmax(restored, dim=1), prediction)
    # float32 entropy 在接近最大熵（p≈0.5）处反演条件数较差；保存 entropy 后
    # 无法逐位恢复原概率，但 prediction 与 entropy 必须保持等价。
    assert torch.allclose(torch.softmax(restored, dim=1), probabilities, atol=1e-4, rtol=1e-4)
    assert torch.allclose(predictive_entropy(restored), entropy, atol=2e-6, rtol=2e-6)


def test_canonical_binary_logits_preserve_saved_class_at_max_entropy_tie() -> None:
    prediction = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
    entropy = torch.ones_like(prediction, dtype=torch.float32)

    restored = canonical_binary_logits_from_prediction_entropy(prediction, entropy)
    restored_prediction = torch.argmax(restored, dim=1)

    assert torch.equal(restored_prediction, prediction.unsqueeze(0).long())
    assert torch.allclose(
        predictive_entropy(restored)[:, 0], prediction.new_ones((1, *prediction.shape), dtype=torch.float32),
        atol=2e-6,
        rtol=2e-6,
    )


def test_blend_refinement_changes_only_roi() -> None:
    coarse = torch.zeros((1, 2, 8, 8, 8), dtype=torch.float32)
    residual = torch.ones_like(coarse)
    roi = torch.zeros((1, 1, 8, 8, 8), dtype=torch.bool)
    roi[:, :, 2:6, 2:6, 2:6] = True

    refined = blend_refinement_logits(coarse, residual, roi)

    expanded_roi = roi.expand_as(refined)
    assert torch.all(refined[expanded_roi] == 1.0)
    assert torch.all(refined[~expanded_roi] == 0.0)


def test_refinement_network_forward_backward_and_roi_invariance() -> None:
    torch.manual_seed(3)
    image = torch.rand((1, 2, 12, 12, 12), dtype=torch.float32)
    coarse = torch.randn((1, 2, 12, 12, 12), dtype=torch.float32)
    roi = torch.zeros((1, 1, 12, 12, 12), dtype=torch.bool)
    roi[:, :, 3:9, 3:9, 3:9] = True

    model = UncertaintyRefinementNet3D(
        image_channels=2,
        num_classes=2,
        hidden_channels=8,
        residual_blocks=1,
    )
    refined = model(image, coarse, roi)

    assert refined.shape == coarse.shape
    outside = ~roi.expand_as(refined)
    assert torch.equal(refined[outside], coarse[outside])

    target = torch.randint(0, 2, (1, 12, 12, 12), dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(refined, target)
    loss.backward()

    assert model.head.weight.grad is not None
    assert torch.isfinite(model.head.weight.grad).all()
