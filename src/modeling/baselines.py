"""用于隔离数据/损失链路问题的轻量 3D segmentation baseline。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int, requested: int = 8) -> int:
    for value in range(min(channels, requested), 0, -1):
        if channels % value == 0:
            return value
    return 1


class _ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = _groups(out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet3D(nn.Module):
    """两级轻量 3D U-Net，只用于 sanity/baseline，不作为最终架构结论。"""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 16) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        self.enc1 = _ConvBlock3D(in_channels, c1)
        self.enc2 = _ConvBlock3D(c1, c2)
        self.bottleneck = _ConvBlock3D(c2, c3)
        self.dec2 = _ConvBlock3D(c3 + c2, c2)
        self.dec1 = _ConvBlock3D(c2 + c1, c1)
        self.head = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool3d(e1, kernel_size=2, stride=2))
        b = self.bottleneck(F.max_pool3d(e2, kernel_size=2, stride=2))

        d2 = F.interpolate(b, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


def build_tiny_unet3d(config: dict) -> TinyUNet3D:
    model_cfg = config.get("model", {})
    return TinyUNet3D(
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        base_channels=int(model_cfg.get("base_channels", 16)),
    )
