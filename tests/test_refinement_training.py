import torch

from src.modeling.refinement import UncertaintyRefinementNet3D
from src.modeling.refinement_training import (
    refinement_error_metrics,
    refinement_training_step,
    roi_refinement_cross_entropy,
)


def _sample() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    image = torch.rand((1, 2, 12, 12, 12), dtype=torch.float32)
    coarse = torch.zeros((1, 2, 12, 12, 12), dtype=torch.float32)
    target = torch.zeros((1, 12, 12, 12), dtype=torch.long)
    target[:, 4:8, 4:8, 4:8] = 1
    roi = torch.zeros((1, 1, 12, 12, 12), dtype=torch.bool)
    roi[:, :, 3:9, 3:9, 3:9] = True
    return image, coarse, target, roi


def test_roi_refinement_ce_ignores_outside_roi() -> None:
    _, logits, target, roi = _sample()
    baseline = roi_refinement_cross_entropy(logits, target, roi)

    modified = logits.clone()
    # ROI 外故意制造极差 logits，ROI 内保持不变；loss 应完全一致。
    outside = ~roi[:, 0]
    modified[:, 1][outside] = 100.0
    changed = roi_refinement_cross_entropy(modified, target, roi)

    assert torch.allclose(baseline, changed)


def test_refinement_metrics_detect_no_outside_roi_change() -> None:
    _, coarse, target, roi = _sample()
    refined = coarse.clone()
    refined[:, 1, 4:8, 4:8, 4:8] = 10.0

    metrics = refinement_error_metrics(coarse, refined, target, roi)

    assert metrics.refined_roi_error_rate < metrics.coarse_roi_error_rate
    assert metrics.roi_error_rate_delta < 0.0
    assert metrics.outside_roi_changed_fraction == 0.0


def test_refinement_training_step_updates_head_with_finite_roi_loss() -> None:
    image, coarse, target, roi = _sample()
    model = UncertaintyRefinementNet3D(
        image_channels=2,
        num_classes=2,
        hidden_channels=8,
        residual_blocks=1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.head.weight.detach().clone()

    metrics = refinement_training_step(
        model,
        optimizer,
        image=image,
        coarse_logits=coarse,
        target=target,
        roi_mask=roi,
        max_grad_norm=5.0,
    )

    assert metrics.loss > 0.0
    assert 0.0 < metrics.roi_fraction < 1.0
    assert torch.isfinite(model.head.weight).all()
    assert not torch.equal(before, model.head.weight.detach())
    # blend 设计保证 ROI 外 prediction 不应被 refinement 改写。
    assert metrics.outside_roi_changed_fraction == 0.0
