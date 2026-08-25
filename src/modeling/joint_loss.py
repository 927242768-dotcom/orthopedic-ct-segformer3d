"""骨科 CT 分割联合损失：区域 + 边界 + 拓扑。

设计目标：
- Region：Dice + CE/BCE，保证整体区域重叠；
- Boundary：基于 GT signed distance field 的边界约束；
- Topology：soft-clDice 形式的可微拓扑约束候选。

重要说明：
1. soft-clDice 最初更适合管状/网络结构，本项目必须通过骨结构消融验证其适用性。
2. Boundary loss 中 GT SDF 通过 SciPy 在 CPU 计算，适合作为首版正确性基线；
   后续若成为训练瓶颈，应在预处理阶段缓存 SDF 或实现 GPU 近似。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _to_one_hot(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """把 target 转成与 logits 同形状的 one-hot/float tensor。"""
    if logits.ndim != 5:
        raise ValueError(f"logits 应为 (B,C,D,H,W)，实际: {tuple(logits.shape)}")

    b, c, d, h, w = logits.shape

    if c == 1:
        if target.ndim == 4:
            target = target.unsqueeze(1)
        if target.shape != logits.shape:
            raise ValueError(
                f"二分类 target 应为 (B,D,H,W) 或 (B,1,D,H,W)，实际: {tuple(target.shape)}"
            )
        return target.to(dtype=logits.dtype)

    if target.ndim == 5 and target.shape[1] == c:
        return target.to(dtype=logits.dtype)

    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]

    if target.ndim != 4:
        raise ValueError(
            f"多分类 target 应为整数标签 (B,D,H,W)/(B,1,D,H,W) 或 one-hot，实际: {tuple(target.shape)}"
        )

    if tuple(target.shape) != (b, d, h, w):
        raise ValueError("target 空间尺寸与 logits 不一致")

    one_hot = F.one_hot(target.long(), num_classes=c).permute(0, 4, 1, 2, 3)
    return one_hot.to(dtype=logits.dtype)


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)


class RegionDiceCELoss3D(nn.Module):
    """Dice + CE/BCE 区域损失。

    多分类默认不在 Dice 中计算背景类，以减少巨大背景对指标的主导。
    """

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.include_background = include_background
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probabilities_from_logits(logits)
        target_oh = _to_one_hot(logits, target)

        if logits.shape[1] == 1:
            ce = F.binary_cross_entropy_with_logits(logits, target_oh)
            probs_dice = probs
            target_dice = target_oh
        else:
            if target.ndim == 5 and target.shape[1] == logits.shape[1]:
                target_index = target.argmax(dim=1)
            elif target.ndim == 5 and target.shape[1] == 1:
                target_index = target[:, 0].long()
            else:
                target_index = target.long()
            ce = F.cross_entropy(logits, target_index)

            if self.include_background:
                probs_dice = probs
                target_dice = target_oh
            else:
                probs_dice = probs[:, 1:]
                target_dice = target_oh[:, 1:]

        reduce_dims = (0, 2, 3, 4)
        intersection = torch.sum(probs_dice * target_dice, dim=reduce_dims)
        denominator = torch.sum(probs_dice, dim=reduce_dims) + torch.sum(
            target_dice, dim=reduce_dims
        )
        dice_per_class = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice_per_class.mean()
        return self.dice_weight * dice_loss + self.ce_weight * ce


def _binary_signed_distance(mask: np.ndarray) -> np.ndarray:
    """生成归一化 signed distance map：内部为负，外部为正。"""
    mask = mask.astype(bool)
    if not mask.any():
        # 空目标：只返回正区域距离没有明确边界意义，首版置零避免制造异常梯度。
        return np.zeros(mask.shape, dtype=np.float32)
    if mask.all():
        return np.zeros(mask.shape, dtype=np.float32)

    outside = distance_transform_edt(~mask)
    inside = distance_transform_edt(mask)
    sdf = outside - inside
    scale = float(np.max(np.abs(sdf)))
    if scale > 0:
        sdf = sdf / scale
    return sdf.astype(np.float32)


class BoundaryLoss3D(nn.Module):
    """基于 GT signed distance field 的边界损失。

    对多分类默认忽略背景类。
    """

    def __init__(self, *, include_background: bool = False) -> None:
        super().__init__()
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probabilities_from_logits(logits)
        target_oh = _to_one_hot(logits, target)

        if logits.shape[1] > 1 and not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        target_np = target_oh.detach().cpu().numpy() > 0.5
        sdf = np.zeros_like(target_np, dtype=np.float32)
        for bi in range(target_np.shape[0]):
            for ci in range(target_np.shape[1]):
                sdf[bi, ci] = _binary_signed_distance(target_np[bi, ci])

        sdf_t = torch.from_numpy(sdf).to(device=logits.device, dtype=logits.dtype)
        return torch.mean(probs * sdf_t)


def _soft_erode_3d(img: torch.Tensor) -> torch.Tensor:
    return -F.max_pool3d(-img, kernel_size=3, stride=1, padding=1)


def _soft_dilate_3d(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(img, kernel_size=3, stride=1, padding=1)


def _soft_open_3d(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate_3d(_soft_erode_3d(img))


def soft_skeletonize_3d(img: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """soft morphological skeletonization 的 3D 版本。"""
    if iterations < 1:
        raise ValueError("iterations 必须 >= 1")

    img = torch.clamp(img, 0.0, 1.0)
    opened = _soft_open_3d(img)
    skeleton = F.relu(img - opened)

    for _ in range(iterations - 1):
        img = _soft_erode_3d(img)
        opened = _soft_open_3d(img)
        delta = F.relu(img - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)

    return skeleton


class SoftClDiceLoss3D(nn.Module):
    """soft-clDice 拓扑损失候选。"""

    def __init__(
        self,
        *,
        iterations: int = 10,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.iterations = int(iterations)
        self.include_background = include_background
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = probabilities_from_logits(logits)
        target_oh = _to_one_hot(logits, target)

        if logits.shape[1] > 1 and not self.include_background:
            probs = probs[:, 1:]
            target_oh = target_oh[:, 1:]

        skel_pred = soft_skeletonize_3d(probs, self.iterations)
        skel_true = soft_skeletonize_3d(target_oh, self.iterations)

        reduce_dims = (2, 3, 4)
        tprec = (
            torch.sum(skel_pred * target_oh, dim=reduce_dims) + self.smooth
        ) / (torch.sum(skel_pred, dim=reduce_dims) + self.smooth)
        tsens = (
            torch.sum(skel_true * probs, dim=reduce_dims) + self.smooth
        ) / (torch.sum(skel_true, dim=reduce_dims) + self.smooth)

        cldice = (2.0 * tprec * tsens + self.smooth) / (
            tprec + tsens + self.smooth
        )
        return 1.0 - cldice.mean()


@dataclass(frozen=True)
class JointLossWeights:
    region: float = 1.0
    boundary: float = 0.1
    topology: float = 0.1


class JointOrthopedicSegLoss(nn.Module):
    """区域 + 边界 + 拓扑联合损失。"""

    def __init__(
        self,
        *,
        weights: JointLossWeights = JointLossWeights(),
        topology_iterations: int = 10,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        if min(weights.region, weights.boundary, weights.topology) < 0:
            raise ValueError("loss 权重不能为负数")
        self.weights = weights
        self.region = RegionDiceCELoss3D(include_background=include_background)
        self.boundary = BoundaryLoss3D(include_background=include_background)
        self.topology = SoftClDiceLoss3D(
            iterations=topology_iterations,
            include_background=include_background,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        region = self.region(logits, target)
        boundary = self.boundary(logits, target)
        topology = self.topology(logits, target)

        total = (
            self.weights.region * region
            + self.weights.boundary * boundary
            + self.weights.topology * topology
        )

        if not return_components:
            return total

        components = {
            "region": region.detach(),
            "boundary": boundary.detach(),
            "topology": topology.detach(),
            "total": total.detach(),
        }
        return total, components


def build_joint_loss(config: dict[str, Any]) -> JointOrthopedicSegLoss:
    """从配置字典构建联合损失，便于 YAML 驱动实验。"""
    loss_cfg = config.get("loss", config)
    weights_cfg = loss_cfg.get("weights", {})
    weights = JointLossWeights(
        region=float(weights_cfg.get("region", 1.0)),
        boundary=float(weights_cfg.get("boundary", 0.1)),
        topology=float(weights_cfg.get("topology", 0.1)),
    )
    return JointOrthopedicSegLoss(
        weights=weights,
        topology_iterations=int(loss_cfg.get("topology_iterations", 10)),
        include_background=bool(loss_cfg.get("include_background", False)),
    )
