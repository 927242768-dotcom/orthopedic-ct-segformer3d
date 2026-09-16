import torch
from torch import nn

from src.modeling.refined_segformer import SegFormer3DInputRefinement


class _DummyCoarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Conv3d(1, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = torch.nn.functional.interpolate(x, scale_factor=0.25, mode="trilinear", align_corners=False)
        return self.proj(low)


def test_direct_plus_coarse_preserves_initial_hard_prediction_and_full_branch_gradient() -> None:
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
        fusion_mode="direct_plus_coarse",
        coarse_scale_init=0.25,
    )
    model.train()
    x = torch.randn(1, 1, 16, 16, 16)
    with torch.no_grad():
        expected = torch.nn.functional.interpolate(coarse(x), size=(16, 16, 16), mode="trilinear", align_corners=False)
        initial = model(x)
    assert torch.equal(initial.argmax(dim=1), expected.argmax(dim=1))

    logits = model(x)
    logits.square().mean().backward()
    assert model.coarse_scale is not None
    assert model.coarse_scale.grad is not None
    assert any(parameter.grad is not None for parameter in model.refinement.parameters())
    assert all(parameter.grad is None for parameter in model.base_model.parameters())
