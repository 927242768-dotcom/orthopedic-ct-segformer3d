import pytest

torch = pytest.importorskip("torch")

from src.modeling.joint_loss import (  # noqa: E402
    JointLossWeights,
    JointOrthopedicSegLoss,
    RegionDiceCELoss3D,
    SoftClDiceLoss3D,
)


def test_region_loss_binary_backward() -> None:
    logits = torch.randn(2, 1, 8, 8, 8, requires_grad=True)
    target = (torch.rand(2, 1, 8, 8, 8) > 0.7).float()
    loss = RegionDiceCELoss3D()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


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
