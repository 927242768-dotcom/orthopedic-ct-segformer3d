"""不确定性 ROI 二阶段精修的训练/诊断基础。

原则：
- coarse SegFormer3D 默认冻结；
- refinement loss 只在 ROI 内归一化，避免 ROI 很小时被全图平均稀释；
- ROI 外 logits 由 ``blend_refinement_logits`` 保持不变；
- 每次训练/评估报告 ROI 与全图错误率变化，防止“局部变好、全局变坏”被忽略。

该模块只提供工程训练闭环；是否真正改善 DSC/HD95/ASSD 必须等真实 baseline checkpoint 后消融。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from src.modeling.refinement import UncertaintyRefinementNet3D


@dataclass(frozen=True)
class RefinementStepMetrics:
    loss: float
    roi_fraction: float
    coarse_roi_error_rate: float
    refined_roi_error_rate: float
    roi_error_rate_delta: float
    coarse_global_error_rate: float
    refined_global_error_rate: float
    global_error_rate_delta: float
    outside_roi_changed_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _prediction_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits 必须为 (B,C,D,H,W)")
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) >= 0.5).long()
    return torch.argmax(logits, dim=1)


def validate_refinement_inputs(
    logits: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits 必须为 (B,C,D,H,W)")
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError("target 必须为 (B,D,H,W) 或 (B,1,D,H,W)")
    if roi_mask.ndim != 5 or roi_mask.shape[1] != 1:
        raise ValueError("roi_mask 必须为 (B,1,D,H,W)")
    if logits.shape[0] != target.shape[0] or logits.shape[-3:] != target.shape[-3:]:
        raise ValueError("logits 与 target 的 batch/spatial shape 不一致")
    if roi_mask.shape[0] != target.shape[0] or roi_mask.shape[-3:] != target.shape[-3:]:
        raise ValueError("roi_mask 与 target 的 batch/spatial shape 不一致")
    if int(roi_mask.sum()) <= 0:
        raise ValueError("roi_mask 为空，无法训练 refinement")
    return target.long()


def roi_refinement_cross_entropy(
    refined_logits: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    """只在 ROI 内计算并按 ROI 体素数归一化的 CE。"""
    target = validate_refinement_inputs(refined_logits, target, roi_mask)
    if refined_logits.shape[1] < 2:
        raise ValueError("当前 refinement CE 要求 num_classes >= 2")
    per_voxel = F.cross_entropy(refined_logits, target, reduction="none")
    roi = roi_mask[:, 0].to(dtype=per_voxel.dtype)
    return (per_voxel * roi).sum() / roi.sum().clamp_min(1.0)


def refinement_error_metrics(
    coarse_logits: torch.Tensor,
    refined_logits: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
    *,
    loss: torch.Tensor | None = None,
) -> RefinementStepMetrics:
    target = validate_refinement_inputs(refined_logits, target, roi_mask)
    if coarse_logits.shape != refined_logits.shape:
        raise ValueError("coarse/refined logits shape 不一致")

    coarse_pred = _prediction_from_logits(coarse_logits)
    refined_pred = _prediction_from_logits(refined_logits)
    roi = roi_mask[:, 0].bool()
    outside = ~roi
    coarse_errors = coarse_pred != target
    refined_errors = refined_pred != target

    roi_count = int(roi.sum())
    total_count = int(roi.numel())
    outside_count = int(outside.sum())
    coarse_roi = float(coarse_errors[roi].float().mean().item())
    refined_roi = float(refined_errors[roi].float().mean().item())
    coarse_global = float(coarse_errors.float().mean().item())
    refined_global = float(refined_errors.float().mean().item())
    outside_changed = (
        0.0
        if outside_count == 0
        else float((coarse_pred[outside] != refined_pred[outside]).float().mean().item())
    )
    return RefinementStepMetrics(
        loss=0.0 if loss is None else float(loss.detach().cpu()),
        roi_fraction=float(roi_count / total_count),
        coarse_roi_error_rate=coarse_roi,
        refined_roi_error_rate=refined_roi,
        roi_error_rate_delta=float(refined_roi - coarse_roi),
        coarse_global_error_rate=coarse_global,
        refined_global_error_rate=refined_global,
        global_error_rate_delta=float(refined_global - coarse_global),
        outside_roi_changed_fraction=outside_changed,
    )


def refinement_training_step(
    model: UncertaintyRefinementNet3D,
    optimizer: torch.optim.Optimizer,
    *,
    image: torch.Tensor,
    coarse_logits: torch.Tensor,
    target: torch.Tensor,
    roi_mask: torch.Tensor,
    detach_coarse: bool = True,
    max_grad_norm: float | None = None,
) -> RefinementStepMetrics:
    """执行单步 refinement 更新并返回可解释诊断指标。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    coarse_input = coarse_logits.detach() if detach_coarse else coarse_logits
    refined_logits = model(image, coarse_input, roi_mask)
    loss = roi_refinement_cross_entropy(refined_logits, target, roi_mask)
    if not torch.isfinite(loss):
        raise RuntimeError("refinement loss 为 NaN/Inf")
    loss.backward()
    if max_grad_norm is not None:
        value = float(max_grad_norm)
        if value <= 0:
            raise ValueError("max_grad_norm 必须 > 0")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=value)
    optimizer.step()
    return refinement_error_metrics(
        coarse_input.detach(),
        refined_logits.detach(),
        target,
        roi_mask,
        loss=loss,
    )
