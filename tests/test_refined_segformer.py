import torch
from torch import nn

from src.modeling.refined_segformer import SegFormer3DInputRefinement


class _DummyCoarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.nn.functional.interpolate(
            x,
            scale_factor=0.25,
            mode="trilinear",
            align_corners=False,
        )
        return self.proj(low)


def test_refined_wrapper_restores_full_resolution_and_backpropagates() -> None:
    model = SegFormer3DInputRefinement(
        _DummyCoarse(),
        in_channels=1,
        num_classes=2,
        refinement_channels=8,
        refinement_downsample=2,
        groupnorm_groups=4,
    )
    x = torch.randn(1, 1, 16, 16, 16)
    logits = model(x)
    assert logits.shape == (1, 2, 16, 16, 16)
    logits.mean().backward()
    assert any(parameter.grad is not None for parameter in model.refinement.parameters())
    assert any(parameter.grad is not None for parameter in model.base_model.parameters())


def test_frozen_base_starts_as_exact_coarse_residual_zero() -> None:
    coarse = _DummyCoarse()
    model = SegFormer3DInputRefinement(
        coarse,
        in_channels=1,
        num_classes=2,
        refinement_channels=8,
        refinement_downsample=2,
        groupnorm_groups=4,
        freeze_base_model=True,
    )
    model.train()
    assert model.base_model.training is False
    assert model.refinement.training is True
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())

    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = torch.nn.functional.interpolate(
            coarse(x), size=(16, 16, 16), mode="trilinear", align_corners=False
        )
        actual = model(x)
    assert torch.equal(actual, expected)


def test_gated_direct_starts_with_same_hard_prediction_and_trains_direct_branch() -> None:
    coarse = _DummyCoarse()
    model = SegFormer3DInputRefinement(
        coarse,
        in_channels=1,
        num_classes=2,
        refinement_channels=8,
        refinement_downsample=1,
        groupnorm_groups=4,
        freeze_base_model=True,
        refinement_arch="tiny_unet3d",
        fusion_mode="gated_direct",
        fusion_alpha_init=0.5,
    )
    model.train()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = torch.nn.functional.interpolate(
            coarse(x), size=(16, 16, 16), mode="trilinear", align_corners=False
        )
        initial = model(x)
    assert torch.equal(initial.argmax(dim=1), expected.argmax(dim=1))

    logits = model(x)
    logits.square().mean().backward()
    assert model.fusion_logit is not None
    assert model.fusion_logit.grad is not None
    assert any(parameter.grad is not None for parameter in model.refinement.parameters())
    assert all(parameter.grad is None for parameter in model.base_model.parameters())


def test_frozen_tiny_unet_refinement_also_starts_as_exact_coarse() -> None:
    coarse = _DummyCoarse()
    model = SegFormer3DInputRefinement(
        coarse,
        in_channels=1,
        num_classes=2,
        refinement_channels=8,
        refinement_downsample=1,
        groupnorm_groups=4,
        freeze_base_model=True,
        refinement_arch="tiny_unet3d",
    )
    model.train()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = torch.nn.functional.interpolate(
            coarse(x), size=(16, 16, 16), mode="trilinear", align_corners=False
        )
        actual = model(x)
    assert torch.equal(actual, expected)
