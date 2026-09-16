import pytest

torch = pytest.importorskip("torch")

from src.modeling.joint_loss import (  # noqa: E402
    JointLossWeights,
    JointOrthopedicSegLoss,
    RegionDiceCELoss3D,
    RegionDiceWeightedCELoss3D,
    RegionTverskyCELoss3D,
    SoftClDiceLoss3D,
)


def test_region_loss_binary_backward() -> None:
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.7).float()
    loss = RegionDiceCELoss3D()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_weighted_ce_binary_backward() -> None:
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.7).float()
    loss = RegionDiceWeightedCELoss3D(background_weight=1.25)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_weighted_ce_penalizes_background_false_positive_more() -> None:
    target = torch.zeros(1, 1, 1, 1, 4)
    logits = torch.tensor([[[[[5.0, 5.0, -5.0, -5.0]]]]])
    base = RegionDiceWeightedCELoss3D(
        dice_weight=0.0,
        ce_weight=1.0,
        background_weight=1.0,
        foreground_weight=1.0,
    )(logits, target)
    weighted = RegionDiceWeightedCELoss3D(
        dice_weight=0.0,
        ce_weight=1.0,
        background_weight=1.25,
        foreground_weight=1.0,
    )(logits, target)
    assert float(weighted) > float(base)


def test_tversky_loss_binary_backward() -> None:
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.7).float()
    loss = RegionTverskyCELoss3D(alpha=0.65, beta=0.35)(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_fp_sensitive_tversky_increases_loss_when_alpha_increases() -> None:
    target = torch.zeros(1, 1, 1, 1, 4)
    target[..., 0] = 1.0
    logits = torch.tensor([[[[[5.0, 5.0, -5.0, -5.0]]]]])
    balanced = RegionTverskyCELoss3D(
        tversky_weight=1.0,
        ce_weight=0.0,
        alpha=0.5,
        beta=0.5,
    )(logits, target)
    fp_sensitive = RegionTverskyCELoss3D(
        tversky_weight=1.0,
        ce_weight=0.0,
        alpha=0.65,
        beta=0.35,
    )(logits, target)
    assert float(fp_sensitive) > float(balanced)


def test_soft_cldice_identical_prediction_is_low() -> None:
    target = torch.zeros(1, 1, 9, 9, 9)
    target[:, :, 2:7, 4, 4] = 1.0
    logits = torch.where(target > 0.5, torch.tensor(10.0), torch.tensor(-10.0))
    loss = SoftClDiceLoss3D(iterations=5)(logits, target)
    assert float(loss) < 0.05


def test_joint_loss_returns_components_and_backward() -> None:
    logits = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(1, 1, 8, 8, 8) > 0.8).float()
    criterion = JointOrthopedicSegLoss(
        weights=JointLossWeights(region=1.0, boundary=0.1, topology=0.1),
        topology_iterations=3,
    )
    loss, parts = criterion(logits, target, return_components=True)
    assert set(parts) == {"region", "boundary", "topology", "total"}
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
