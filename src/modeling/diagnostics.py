"""SegFormer3D 训练稳定性诊断工具。

本模块只读取 logits/参数/梯度并生成诊断统计，不执行 optimizer.step，
不会改变训练数学行为。主要用于定位 foreground explosion / background collapse。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F

from src.modeling.train import build_criterion, resize_logits_to_target


STAT_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
PROBABILITY_HISTOGRAM_EDGES = np.linspace(0.0, 1.0, 11, dtype=np.float64)


def sampled_tensor_stats(
    values: torch.Tensor,
    *,
    max_samples: int = 500_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """对大 tensor 做可复现抽样后统计分位数，避免全量复制到 NumPy。"""
    flat = values.detach().reshape(-1)
    total = int(flat.numel())
    if total == 0:
        raise ValueError("values 不能为空")
    if max_samples <= 0:
        raise ValueError("max_samples 必须 > 0")

    sampled_count = min(total, int(max_samples))
    if sampled_count < total:
        rng = np.random.default_rng(seed)
        indices_np = rng.integers(0, total, size=sampled_count, dtype=np.int64)
        indices = torch.from_numpy(indices_np).to(flat.device)
        sample = flat.index_select(0, indices).float().cpu().numpy()
    else:
        sample = flat.float().cpu().numpy()

    result: dict[str, float | int] = {
        "total_count": total,
        "sampled_count": sampled_count,
        "mean": float(np.mean(sample)),
        "std": float(np.std(sample)),
        "min": float(np.min(sample)),
        "max": float(np.max(sample)),
    }
    for quantile in STAT_QUANTILES:
        result[f"q{int(round(quantile * 100)):02d}"] = float(np.quantile(sample, quantile))
    return result


def _probability_histogram(
    values: torch.Tensor,
    *,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        return {
            "edges": PROBABILITY_HISTOGRAM_EDGES.tolist(),
            "counts": [0] * 10,
            "fractions": [0.0] * 10,
            "sampled_count": 0,
        }
    total = int(flat.numel())
    sampled_count = min(total, int(max_samples))
    if sampled_count < total:
        rng = np.random.default_rng(seed)
        indices_np = rng.integers(0, total, size=sampled_count, dtype=np.int64)
        indices = torch.from_numpy(indices_np).to(flat.device)
        sample = flat.index_select(0, indices).float().cpu().numpy()
    else:
        sample = flat.float().cpu().numpy()
    counts, edges = np.histogram(sample, bins=PROBABILITY_HISTOGRAM_EDGES)
    fractions = counts.astype(np.float64) / max(int(counts.sum()), 1)
    return {
        "edges": edges.astype(float).tolist(),
        "counts": counts.astype(int).tolist(),
        "fractions": fractions.astype(float).tolist(),
        "sampled_count": sampled_count,
    }


def foreground_probability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """返回二分类前景概率，支持 1-channel sigmoid 或 2-class softmax 等价形式。"""
    if logits.ndim != 5:
        raise ValueError("logits 必须为 (B,C,D,H,W)")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    if logits.shape[1] == 2:
        # softmax([bg, fg]) 的 fg 概率等价于 sigmoid(fg_logit-bg_logit)，少建一个 2-channel tensor。
        return torch.sigmoid(logits[:, 1] - logits[:, 0])
    raise ValueError("当前稳定性 diagnostics 只支持 binary 1/2-channel logits")


def logits_probability_diagnostics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    max_samples: int = 500_000,
    seed: int = 42,
) -> dict[str, Any]:
    """统计 bg/fg logits 与 GT 前景/背景上的前景概率分布。"""
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4:
        raise ValueError("target 必须为 (B,D,H,W) 或 (B,1,D,H,W)")
    if tuple(logits.shape[-3:]) != tuple(target.shape[-3:]):
        raise ValueError("logits 与 target 空间尺寸必须一致")

    payload: dict[str, Any] = {}
    if logits.shape[1] == 2:
        payload["background_logit"] = sampled_tensor_stats(
            logits[:, 0], max_samples=max_samples, seed=seed
        )
        payload["foreground_logit"] = sampled_tensor_stats(
            logits[:, 1], max_samples=max_samples, seed=seed + 1
        )
        payload["foreground_minus_background_logit"] = sampled_tensor_stats(
            logits[:, 1] - logits[:, 0], max_samples=max_samples, seed=seed + 2
        )
    elif logits.shape[1] == 1:
        payload["foreground_logit"] = sampled_tensor_stats(
            logits[:, 0], max_samples=max_samples, seed=seed
        )
    else:
        raise ValueError("当前稳定性 diagnostics 只支持 binary 1/2-channel logits")

    probability = foreground_probability_from_logits(logits)
    target_foreground = target > 0
    gt_foreground_probability = probability[target_foreground]
    gt_background_probability = probability[~target_foreground]
    if gt_foreground_probability.numel() == 0 or gt_background_probability.numel() == 0:
        raise ValueError("diagnostics 需要同时存在 GT foreground/background voxel")

    payload["foreground_probability_on_gt_foreground"] = sampled_tensor_stats(
        gt_foreground_probability,
        max_samples=max_samples,
        seed=seed + 3,
    )
    payload["foreground_probability_on_gt_background"] = sampled_tensor_stats(
        gt_background_probability,
        max_samples=max_samples,
        seed=seed + 4,
    )
    payload["foreground_probability_histogram_on_gt_foreground"] = _probability_histogram(
        gt_foreground_probability,
        max_samples=max_samples,
        seed=seed + 5,
    )
    payload["foreground_probability_histogram_on_gt_background"] = _probability_histogram(
        gt_background_probability,
        max_samples=max_samples,
        seed=seed + 6,
    )
    payload["prediction_foreground_fraction"] = float((probability >= 0.5).float().mean().item())
    payload["target_foreground_fraction"] = float(target_foreground.float().mean().item())
    return payload


def region_loss_diagnostics(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    smooth: float = 1e-5,
    chunk_size: int = 1_000_000,
) -> dict[str, float | int]:
    """对 binary Region Dice+CE 分解，不改变原 loss 公式。

    CE 分为：
    - foreground/background voxel 内部平均 CE；
    - 以全体 voxel 为分母的 weighted contribution，两者相加等于 total CE。
    """
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    probability = foreground_probability_from_logits(logits)
    if tuple(probability.shape) != tuple(target.shape):
        raise ValueError("probability 与 target 尺寸必须一致")

    target_foreground = target > 0
    target_float = target_foreground.to(dtype=probability.dtype)
    intersection = torch.sum(probability * target_float, dtype=torch.float64)
    denominator = torch.sum(probability, dtype=torch.float64) + torch.sum(
        target_float, dtype=torch.float64
    )
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - float(dice_score.item())

    p_flat = probability.detach().reshape(-1)
    t_flat = target_foreground.detach().reshape(-1)
    foreground_ce_sum = 0.0
    background_ce_sum = 0.0
    foreground_count = 0
    background_count = 0
    eps = 1e-7
    for start in range(0, int(p_flat.numel()), int(chunk_size)):
        p = p_flat[start : start + chunk_size].clamp(eps, 1.0 - eps)
        t = t_flat[start : start + chunk_size]
        if bool(t.any()):
            fg_p = p[t]
            foreground_ce_sum += float((-torch.log(fg_p)).double().sum().item())
            foreground_count += int(fg_p.numel())
        bg_mask = ~t
        if bool(bg_mask.any()):
            bg_p = p[bg_mask]
            background_ce_sum += float((-torch.log1p(-bg_p)).double().sum().item())
            background_count += int(bg_p.numel())

    total_count = foreground_count + background_count
    if total_count == 0:
        raise ValueError("target 不能为空")
    ce_loss = (foreground_ce_sum + background_ce_sum) / total_count
    foreground_weighted = foreground_ce_sum / total_count
    background_weighted = background_ce_sum / total_count
    total_loss = float(dice_weight) * dice_loss + float(ce_weight) * ce_loss
    return {
        "dice_loss": float(dice_loss),
        "ce_loss": float(ce_loss),
        "weighted_total_loss": float(total_loss),
        "foreground_voxel_count": foreground_count,
        "background_voxel_count": background_count,
        "foreground_ce_mean": (
            float(foreground_ce_sum / foreground_count) if foreground_count else 0.0
        ),
        "background_ce_mean": (
            float(background_ce_sum / background_count) if background_count else 0.0
        ),
        "foreground_ce_weighted_contribution": float(foreground_weighted),
        "background_ce_weighted_contribution": float(background_weighted),
    }


def find_final_segmentation_head(
    model: torch.nn.Module,
    *,
    num_classes: int,
) -> tuple[str, torch.nn.Conv3d]:
    """定位真正输出 num_classes logits 的最后 Conv3d。"""
    decoder = getattr(model, "segformer_decoder", None)
    preferred = getattr(decoder, "linear_pred", None)
    if isinstance(preferred, torch.nn.Conv3d) and preferred.out_channels == num_classes:
        return "segformer_decoder.linear_pred", preferred

    candidates: list[tuple[str, torch.nn.Conv3d]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and module.out_channels == num_classes:
            candidates.append((name, module))
    if not candidates:
        raise ValueError(f"未找到 out_channels={num_classes} 的最终 segmentation Conv3d")
    return candidates[-1]


def head_parameter_diagnostics(
    model: torch.nn.Module,
    *,
    num_classes: int,
) -> dict[str, Any]:
    name, head = find_final_segmentation_head(model, num_classes=num_classes)
    weight = head.weight.detach().float()
    payload: dict[str, Any] = {
        "name": name,
        "weight_shape": list(weight.shape),
        "weight_norm": float(torch.linalg.vector_norm(weight).item()),
        "weight_norm_per_output_class": [
            float(torch.linalg.vector_norm(weight[index]).item())
            for index in range(weight.shape[0])
        ],
    }
    if head.bias is not None:
        bias = head.bias.detach().float()
        payload["bias"] = [float(value) for value in bias.cpu().tolist()]
        payload["bias_norm"] = float(torch.linalg.vector_norm(bias).item())
        if bias.numel() == 2:
            payload["foreground_minus_background_bias"] = float((bias[1] - bias[0]).item())
    else:
        payload["bias"] = None
        payload["bias_norm"] = None
    return payload


def batchnorm_running_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    """记录所有 BatchNorm3d 的 running statistics，定位小批量/patch 分布漂移。"""
    modules: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.BatchNorm3d):
            continue
        running_mean = module.running_mean
        running_var = module.running_var
        modules[name] = {
            "num_features": int(module.num_features),
            "eps": float(module.eps),
            "momentum": float(module.momentum) if module.momentum is not None else None,
            "track_running_stats": bool(module.track_running_stats),
            "num_batches_tracked": (
                int(module.num_batches_tracked.item())
                if module.num_batches_tracked is not None
                else None
            ),
            "running_mean": (
                sampled_tensor_stats(running_mean, max_samples=max(int(module.num_features), 1))
                if running_mean is not None
                else None
            ),
            "running_var": (
                sampled_tensor_stats(running_var, max_samples=max(int(module.num_features), 1))
                if running_var is not None
                else None
            ),
            "affine_weight_norm": (
                float(torch.linalg.vector_norm(module.weight.detach()).item())
                if module.affine and module.weight is not None
                else None
            ),
            "affine_bias_norm": (
                float(torch.linalg.vector_norm(module.bias.detach()).item())
                if module.affine and module.bias is not None
                else None
            ),
        }
    return {
        "batchnorm3d_count": len(modules),
        "modules": modules,
    }


@contextmanager
def batchnorm_batch_stats_mode(model: torch.nn.Module) -> Iterator[None]:
    """临时让 BN 在 inference 使用当前 window 的 batch statistics，且不更新 running stats。"""
    states: list[tuple[torch.nn.BatchNorm3d, bool, bool]] = []
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            states.append((module, module.training, bool(module.track_running_stats)))
            module.train(True)
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, training, track_running_stats in states:
            module.track_running_stats = track_running_stats
            module.train(training)


def foreground_centered_patch(
    image: torch.Tensor,
    target: torch.Tensor,
    roi_size_dhw: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """从单病例整卷中提取确定性的 GT 前景中心 patch，用于无 optimizer-step 梯度诊断。"""
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("image 必须为单病例 (1,C,D,H,W)")
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if target.ndim != 4 or target.shape[0] != 1:
        raise ValueError("target 必须为单病例 (1,D,H,W)")

    roi = tuple(int(v) for v in roi_size_dhw)
    if any(value <= 0 for value in roi):
        raise ValueError("roi_size_dhw 必须全部 > 0")

    spatial = tuple(int(v) for v in image.shape[-3:])
    pad_d = max(0, roi[0] - spatial[0])
    pad_h = max(0, roi[1] - spatial[1])
    pad_w = max(0, roi[2] - spatial[2])
    if pad_d or pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))
        target = F.pad(target.unsqueeze(1).float(), (0, pad_w, 0, pad_h, 0, pad_d))[:, 0].long()
        spatial = tuple(int(v) for v in image.shape[-3:])

    coords = torch.nonzero(target[0] > 0, as_tuple=False)
    if coords.numel() > 0:
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values
        center = ((mins + maxs) // 2).tolist()
    else:
        center = [value // 2 for value in spatial]

    starts: list[int] = []
    for dim, size, center_value in zip(spatial, roi, center):
        starts.append(max(0, min(int(dim - size), int(center_value) - size // 2)))
    d0, h0, w0 = starts
    rd, rh, rw = roi
    return (
        image[..., d0 : d0 + rd, h0 : h0 + rh, w0 : w0 + rw],
        target[..., d0 : d0 + rd, h0 : h0 + rh, w0 : w0 + rw],
    )


def head_gradient_diagnostics(
    model: torch.nn.Module,
    image: torch.Tensor,
    target: torch.Tensor,
    config: dict[str, Any],
    *,
    roi_size_dhw: tuple[int, int, int],
) -> dict[str, Any]:
    """在固定前景中心 patch 上只 backward 一次，记录最终 head 梯度，不更新参数。"""
    num_classes = int(config["model"]["num_classes"])
    name, head = find_final_segmentation_head(model, num_classes=num_classes)
    patch_image, patch_target = foreground_centered_patch(image, target, roi_size_dhw)
    criterion = build_criterion(config).to(patch_image.device)

    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(patch_image)
    logits = resize_logits_to_target(logits, tuple(patch_target.shape[-3:]))
    loss = criterion(logits, patch_target)
    loss.backward()

    weight_grad = head.weight.grad
    bias_grad = head.bias.grad if head.bias is not None else None
    payload: dict[str, Any] = {
        "head_name": name,
        "patch_shape": list(patch_image.shape),
        "patch_target_foreground_fraction": float((patch_target > 0).float().mean().item()),
        "loss": float(loss.detach().cpu().item()),
        "weight_parameter_norm": float(torch.linalg.vector_norm(head.weight.detach()).item()),
        "weight_gradient_norm": (
            float(torch.linalg.vector_norm(weight_grad.detach()).item())
            if weight_grad is not None
            else None
        ),
        "weight_gradient_norm_per_output_class": (
            [
                float(torch.linalg.vector_norm(weight_grad[index].detach()).item())
                for index in range(weight_grad.shape[0])
            ]
            if weight_grad is not None
            else None
        ),
        "bias_parameter_norm": (
            float(torch.linalg.vector_norm(head.bias.detach()).item()) if head.bias is not None else None
        ),
        "bias_gradient_norm": (
            float(torch.linalg.vector_norm(bias_grad.detach()).item()) if bias_grad is not None else None
        ),
        "bias_gradient": (
            [float(value) for value in bias_grad.detach().cpu().tolist()]
            if bias_grad is not None
            else None
        ),
    }
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return payload
