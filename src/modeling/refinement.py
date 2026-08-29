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


def canonical_binary_logits_from_prediction_entropy(
    prediction: torch.Tensor,
    normalized_entropy: torch.Tensor,
    *,
    eps: float = 1e-8,
    iterations: int = 40,
) -> torch.Tensor:
    """由二分类 prediction + 归一化熵恢复一组等价 canonical logits。

    对二分类 softmax，已知 argmax 类别以及归一化 binary entropy 后，较大类别的
    confidence 在 ``[0.5, 1)`` 上唯一确定。这里用单调二分恢复 confidence，再令
    ``logits = log(probability)``。这不会恢复原网络 logits 的公共平移项，但 softmax、
    predictive entropy 与最终 argmax 均保持等价，因此可复用已保存的 prediction +
    entropy，而无需为 refinement 重跑 full-volume coarse inference。
    """
    if iterations <= 0:
        raise ValueError("iterations 必须 > 0")
    if not (0.0 < eps < 0.5):
        raise ValueError("eps 必须位于 (0,0.5)")

    pred = prediction
    entropy = normalized_entropy
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
    if entropy.ndim == 5 and entropy.shape[1] == 1:
        entropy = entropy[:, 0]
    elif entropy.ndim == 3:
        entropy = entropy.unsqueeze(0)
    if pred.ndim == 5 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 4 or entropy.ndim != 4:
        raise ValueError("prediction/normalized_entropy 必须为 (D,H,W) 或 (B,D,H,W)")
    if pred.shape != entropy.shape:
        raise ValueError("prediction 与 normalized_entropy shape 必须一致")
    if torch.any((pred != 0) & (pred != 1)):
        raise ValueError("当前 canonical 恢复只支持二分类 prediction=0/1")
    if not torch.isfinite(entropy).all():
        raise ValueError("normalized_entropy 包含 NaN/Inf")
    if torch.any(entropy < -1e-6) or torch.any(entropy > 1.0 + 1e-6):
        raise ValueError("normalized_entropy 必须位于 [0,1]")

    entropy = entropy.to(dtype=torch.float32).clamp(0.0, 1.0)
    low = torch.full_like(entropy, 0.5)
    high = torch.full_like(entropy, 1.0 - eps)
    log2 = torch.log(torch.tensor(2.0, dtype=entropy.dtype, device=entropy.device))
    for _ in range(iterations):
        confidence = (low + high) * 0.5
        binary_entropy = -(
            confidence * torch.log(confidence.clamp_min(eps))
            + (1.0 - confidence) * torch.log((1.0 - confidence).clamp_min(eps))
        ) / log2
        # binary entropy 在 [0.5,1) 上单调递减。
        move_right = binary_entropy > entropy
        low = torch.where(move_right, confidence, low)
        high = torch.where(move_right, high, confidence)

    confidence = (low + high) * 0.5
    # float32 保存的最大熵附近会把反演结果舍入为恰好 0.5；此时两个
    # canonical logits 完全相同，torch.argmax 会固定选择 class 0，导致原本
    # saved prediction=1 的体素被错误翻转。把较大类别的 confidence 至少推进
    # 到 0.5 之后的下一个可表示浮点数，既严格保留 saved prediction 类别，
    # 又只引入 float32 ULP 级别的熵误差。
    half = torch.full_like(confidence, 0.5)
    above_half = torch.nextafter(half, torch.ones_like(half))
    confidence = torch.maximum(confidence, above_half)
    p1 = torch.where(pred.long() == 1, confidence, 1.0 - confidence)
    probabilities = torch.stack([1.0 - p1, p1], dim=1).clamp_min(eps)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    return torch.log(probabilities)


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

    def residual_logits(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
    ) -> torch.Tensor:
        """预测 residual logits；不在这里应用 ROI mask，便于 tiled validation 复用计算。"""
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
        return self.head(self.features(features))

    def forward(
        self,
        image: torch.Tensor,
        coarse_logits: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual_logits(image, coarse_logits)
        return blend_refinement_logits(coarse_logits, residual, roi_mask)
