"""不确定性驱动的局部分割精修网络。

设计目标：
- 主 SegFormer3D 先产生 coarse logits；
- predictive entropy 选择高不确定 ROI；
- 小型 3D CNN 只学习 ROI 内的残差修正；
- ROI 外严格保留 coarse logits，避免二次网络无意义改写整幅体数据。

该模块目前是工程候选，需要真实 baseline 后通过消融验证收益、额外显存和时间成本。
"""

from __future__ import annotations

import torch
from torch import nn

from src.modeling.uncertainty import predictive_entropy, probability_from_logits


def blend_refinement_logits(
    coarse_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    """只在 ROI 内叠加残差，ROI 外保持 coarse logits 完全不变。"""
    if coarse_logits.shape != residual_logits.shape:
        raise ValueError("coarse_logits 与 residual_logits shape 必须一致")
    if roi_mask.ndim != 5 or roi_mask.shape[1] != 1:
        raise ValueError("roi_mask 必须为 (B,1,D,H,W)")
    if roi_mask.shape[0] != coarse_logits.shape[0] or roi_mask.shape[-3:] != coarse_logits.shape[-3:]:
        raise ValueError("roi_mask 与 logits 的 batch/spatial shape 不一致")
    return coarse_logits + residual_logits * roi_mask.to(dtype=coarse_logits.dtype)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels, affine=True),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class UncertaintyRefinementNet3D(nn.Module):
    """使用 CT + coarse probability + entropy 预测局部 residual logits。"""

    def __init__(
        self,
        *,
        image_channels: int,
        num_classes: int,
        hidden_channels: int = 24,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()
        if image_channels <= 0:
            raise ValueError("image_channels 必须 > 0")
        if num_classes < 2:
            raise ValueError("当前 refinement 设计要求 num_classes >= 2")
        if hidden_channels <= 0 or residual_blocks < 0:
            raise ValueError("hidden_channels 必须 > 0，residual_blocks 不能为负数")

        self.image_channels = int(image_channels)
        self.num_classes = int(num_classes)
        in_channels = self.image_channels + self.num_classes + 1

        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.GELU(),
        ]
        layers.extend(ResidualBlock3D(hidden_channels) for _ in range(residual_blocks))
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv3d(hidden_channels, self.num_classes, kernel_size=1)

        # 初始时 residual 接近 0，使精修模块从 coarse prediction 附近开始学习。
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 5:
            raise ValueError("image 必须为 (B,C,D,H,W)")
        if coarse_logits.ndim != 5:
            raise ValueError("coarse_logits 必须为 (B,C,D,H,W)")
        if image.shape[1] != self.image_channels:
            raise ValueError(
                f"image channel 不匹配: expected={self.image_channels}, got={image.shape[1]}"
            )
        if coarse_logits.shape[1] != self.num_classes:
            raise ValueError(
                f"coarse logits class 不匹配: expected={self.num_classes}, got={coarse_logits.shape[1]}"
            )
        if image.shape[0] != coarse_logits.shape[0] or image.shape[-3:] != coarse_logits.shape[-3:]:
            raise ValueError("image 与 coarse_logits 的 batch/spatial shape 必须一致")

        probabilities = probability_from_logits(coarse_logits)
        uncertainty = predictive_entropy(coarse_logits)
        features = torch.cat([image, probabilities, uncertainty], dim=1)
        residual = self.head(self.features(features))
        return blend_refinement_logits(coarse_logits, residual, roi_mask)
