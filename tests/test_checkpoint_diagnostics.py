from __future__ import annotations

import pytest
import torch

from src.modeling.diagnostics import (
    batchnorm_batch_stats_mode,
    batchnorm_running_diagnostics,
    find_final_segmentation_head,
    foreground_centered_patch,
    head_gradient_diagnostics,
    head_parameter_diagnostics,
    logits_probability_diagnostics,
    region_loss_diagnostics,
    sampled_tensor_stats,
)
from src.modeling.joint_loss import RegionDiceCELoss3D
from src.modeling.train import _model_predictor, resize_logits_to_target


class _DummyBinarySeg(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segformer_decoder = torch.nn.Module()
        self.segformer_decoder.linear_pred = torch.nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segformer_decoder.linear_pred(x)


def _config() -> dict:
    return {
        "model": {"num_classes": 2},
        "loss": {
            "type": "region_dice_ce",
            "dice_weight": 1.0,
            "ce_weight": 1.0,
            "include_background": False,
        },
    }


def test_sampled_tensor_stats_contains_required_quantiles() -> None:
    values = torch.arange(100, dtype=torch.float32)
    stats = sampled_tensor_stats(values, max_samples=100, seed=7)
    assert stats["total_count"] == 100
    assert stats["sampled_count"] == 100
    for key in ("q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"):
        assert key in stats
    assert stats["q50"] == pytest.approx(49.5)


def test_region_loss_diagnostics_matches_region_dice_ce() -> None:
    logits = torch.tensor(
        [
            [
                [[[1.0, -0.5], [0.2, 0.0]]],
                [[[-0.2, 0.8], [0.3, 1.2]]],
            ]
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.long)
    criterion = RegionDiceCELoss3D(dice_weight=1.0, ce_weight=1.0, include_background=False)
    expected = float(criterion(logits, target).item())
    diagnostics = region_loss_diagnostics(logits, target)

    assert diagnostics["weighted_total_loss"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert diagnostics["ce_loss"] == pytest.approx(
        diagnostics["foreground_ce_weighted_contribution"]
        + diagnostics["background_ce_weighted_contribution"],
        rel=1e-8,
    )
    assert diagnostics["foreground_voxel_count"] == 2
    assert diagnostics["background_voxel_count"] == 2


def test_logits_probability_diagnostics_separates_gt_foreground_background() -> None:
    logits = torch.zeros((1, 2, 1, 1, 4), dtype=torch.float32)
    logits[:, 1, ..., :2] = 2.0
    logits[:, 0, ..., 2:] = 2.0
    target = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.long)

    payload = logits_probability_diagnostics(logits, target, max_samples=100)
    fg_mean = payload["foreground_probability_on_gt_foreground"]["mean"]
    bg_mean = payload["foreground_probability_on_gt_background"]["mean"]
    assert fg_mean > 0.8
    assert bg_mean < 0.2
    assert payload["prediction_foreground_fraction"] == pytest.approx(0.5)
    assert payload["target_foreground_fraction"] == pytest.approx(0.5)


def test_find_and_summarize_final_segmentation_head() -> None:
    model = _DummyBinarySeg()
    name, head = find_final_segmentation_head(model, num_classes=2)
    assert name == "segformer_decoder.linear_pred"
    assert head is model.segformer_decoder.linear_pred

    payload = head_parameter_diagnostics(model, num_classes=2)
    assert payload["name"] == name
    assert payload["weight_shape"] == [2, 1, 1, 1, 1]
    assert len(payload["bias"]) == 2
    assert "foreground_minus_background_bias" in payload


def test_batchnorm_batch_stats_mode_does_not_mutate_running_stats() -> None:
    model = torch.nn.Sequential(torch.nn.BatchNorm3d(2))
    model.eval()
    batchnorm = model[0]
    before_mean = batchnorm.running_mean.detach().clone()
    before_var = batchnorm.running_var.detach().clone()

    payload = batchnorm_running_diagnostics(model)
    assert payload["batchnorm3d_count"] == 1

    with batchnorm_batch_stats_mode(model):
        assert batchnorm.training is True
        assert batchnorm.track_running_stats is False
        model(torch.randn((1, 2, 4, 4, 4)))

    assert batchnorm.training is False
    assert batchnorm.track_running_stats is True
    assert torch.equal(batchnorm.running_mean, before_mean)
    assert torch.equal(batchnorm.running_var, before_var)


def test_foreground_centered_patch_keeps_foreground() -> None:
    image = torch.zeros((1, 1, 10, 10, 10), dtype=torch.float32)
    target = torch.zeros((1, 10, 10, 10), dtype=torch.long)
    target[:, 7:9, 7:9, 7:9] = 1

    patch_image, patch_target = foreground_centered_patch(image, target, (6, 6, 6))
    assert patch_image.shape == (1, 1, 6, 6, 6)
    assert patch_target.shape == (1, 6, 6, 6)
    assert bool((patch_target > 0).any())


def test_head_gradient_diagnostics_does_not_update_parameters() -> None:
    torch.manual_seed(0)
    model = _DummyBinarySeg()
    image = torch.randn((1, 1, 6, 6, 6), dtype=torch.float32)
    target = torch.zeros((1, 6, 6, 6), dtype=torch.long)
    target[:, 2:4, 2:4, 2:4] = 1
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    payload = head_gradient_diagnostics(model, image, target, _config(), roi_size_dhw=(6, 6, 6))

    assert payload["weight_gradient_norm"] is not None
    assert payload["weight_gradient_norm"] > 0
    assert payload["bias_gradient_norm"] is not None
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach(), before[name])


def test_full_volume_predictor_uses_same_resize_helper_as_training() -> None:
    class _HalfResolutionModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pooled = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
            return torch.cat([pooled, -pooled], dim=1)

    model = _HalfResolutionModel()
    image = torch.randn((1, 1, 8, 10, 12), dtype=torch.float32)
    direct = resize_logits_to_target(model(image), tuple(image.shape[-3:]))
    predicted = _model_predictor(model)(image)
    assert torch.equal(predicted, direct)
