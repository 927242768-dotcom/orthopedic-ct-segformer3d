import torch

from src.modeling.refinement import UncertaintyRefinementNet3D, blend_refinement_logits


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
