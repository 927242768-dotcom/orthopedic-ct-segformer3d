import torch
import torch.nn.functional as F
from torch import nn

from src.modeling.segformer3d_hr_decoder import SegFormer3DHighResolutionDecoder


class _DummyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv3d(1, 4, kernel_size=1)
        self.c2 = nn.Conv3d(4, 8, kernel_size=1)
        self.c3 = nn.Conv3d(8, 12, kernel_size=1)
        self.c4 = nn.Conv3d(12, 16, kernel_size=1)

    def forward(self, x: torch.Tensor):
        c1 = self.c1(F.avg_pool3d(x, kernel_size=4, stride=4))
        c2 = self.c2(F.avg_pool3d(c1, kernel_size=2, stride=2))
        c3 = self.c3(F.avg_pool3d(c2, kernel_size=2, stride=2))
        c4 = self.c4(F.avg_pool3d(c3, kernel_size=2, stride=2))
        return [c1, c2, c3, c4]


def test_hr_decoder_restores_full_resolution_and_backpropagates() -> None:
    encoder = _DummyEncoder()
    model = SegFormer3DHighResolutionDecoder(
        encoder,
        in_channels=1,
        num_classes=2,
        embed_dims=[4, 8, 12, 16],
        decoder_channels=[16, 16, 12, 8, 8, 4],
        shallow_channels=4,
        groupnorm_groups=4,
    )
    x = torch.randn(1, 1, 32, 32, 32)
    logits = model(x)
    assert logits.shape == (1, 2, 32, 32, 32)
    logits.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.fuse_full.parameters())


def test_hr_decoder_can_freeze_encoder_without_freezing_decoder() -> None:
    model = SegFormer3DHighResolutionDecoder(
        _DummyEncoder(),
        in_channels=1,
        num_classes=2,
        embed_dims=[4, 8, 12, 16],
        decoder_channels=[16, 16, 12, 8, 8, 4],
        shallow_channels=4,
        groupnorm_groups=4,
        freeze_encoder=True,
    )
    model.train()
    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())

    x = torch.randn(1, 1, 32, 32, 32)
    model(x).mean().backward()
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.fuse_full.parameters())
