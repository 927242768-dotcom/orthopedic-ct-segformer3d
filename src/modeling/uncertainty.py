"""分割不确定性估计与 ROI 选择。

首版使用预测熵作为低成本基线，不声称等同于完整 Bayesian uncertainty。
后续将通过误差相关性、质量控制能力和精修收益验证其实际价值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class UncertaintyROIConfig:
    top_percent: float = 10.0
    dilation_iterations: int = 2
    min_voxels: int = 64


@dataclass(frozen=True)
class UncertaintyErrorMetrics:
    total_voxels: int
    sampled_voxels: int
    sampling_fraction: float
    error_rate: float
    mean_uncertainty_error: float | None
    mean_uncertainty_correct: float | None
    error_auroc: float | None
    error_auprc: float | None
    top_percent: float
    top_uncertainty_error_recall: float
    top_uncertainty_error_rate: float
    top_uncertainty_fraction: float

    def to_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentationCalibrationMetrics:
    total_voxels: int
    sampled_voxels: int
    sampling_fraction: float
    n_bins: int
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    negative_log_likelihood: float
    mean_confidence: float
    accuracy: float
    confidence_gap: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def probability_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError("logits 必须为 (B,C,D,H,W)")
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits)
        return torch.cat([1.0 - p1, p1], dim=1)
    return torch.softmax(logits, dim=1)


def predictive_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """归一化预测熵，输出形状 (B,1,D,H,W)，范围近似 [0,1]。"""
    probs = probability_from_logits(logits)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
    max_entropy = torch.log(torch.tensor(float(probs.shape[1]), device=logits.device, dtype=logits.dtype))
    return entropy / max_entropy.clamp_min(eps)


def segmentation_calibration_metrics(
    logits: torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    n_bins: int = 15,
    max_samples: int = 500_000,
    seed: int = 42,
    eps: float = 1e-8,
) -> SegmentationCalibrationMetrics:
    """计算体素级置信度校准指标。

    采用 top-1 confidence 的 ECE/MCE，并同时报告 multiclass Brier score、NLL、
    平均置信度与体素准确率。对大体积使用固定随机种子的体素采样，避免评估阶段
    因完整概率张量占用额外大量内存。该函数用于模型校准分析，不等同于分割重叠指标。
    """
    if logits.ndim != 5:
        raise ValueError("logits 必须为 (B,C,D,H,W)")
    if n_bins <= 0:
        raise ValueError("n_bins 必须 > 0")
    if max_samples <= 0:
        raise ValueError("max_samples 必须 > 0")

    if isinstance(target, torch.Tensor):
        target_tensor = target.detach()
    else:
        target_tensor = torch.as_tensor(np.asarray(target))
    if target_tensor.ndim == 5 and target_tensor.shape[1] == 1:
        target_tensor = target_tensor[:, 0]
    if target_tensor.ndim == 3 and logits.shape[0] == 1:
        target_tensor = target_tensor.unsqueeze(0)
    if target_tensor.ndim != 4:
        raise ValueError("target 必须为 (B,D,H,W) 或可压缩到该形状")
    if tuple(target_tensor.shape) != (logits.shape[0], *logits.shape[2:]):
        raise ValueError(
            "logits/target 空间 shape 必须一致，"
            f"当前为 {tuple(logits.shape)}/{tuple(target_tensor.shape)}"
        )

    class_count = 2 if logits.shape[1] == 1 else int(logits.shape[1])
    target_flat = target_tensor.reshape(-1).long()
    if target_flat.numel() == 0:
        raise ValueError("输入为空")
    target_min = int(target_flat.min().item())
    target_max = int(target_flat.max().item())
    if target_min < 0 or target_max >= class_count:
        raise ValueError(
            f"target 类别必须位于 [0,{class_count - 1}]，当前范围 {target_min}..{target_max}"
        )

    total = int(target_flat.numel())
    sample_count = min(total, int(max_samples))
    if sample_count < total:
        rng = np.random.default_rng(int(seed))
        sample_indices_np = rng.choice(total, size=sample_count, replace=False)
        sample_indices = torch.as_tensor(sample_indices_np, device=logits.device, dtype=torch.long)
    else:
        sample_indices = torch.arange(total, device=logits.device, dtype=torch.long)

    sampled_target = target_flat.to(logits.device)[sample_indices]
    flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(total, logits.shape[1])
    sampled_logits = flat_logits[sample_indices]
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(sampled_logits[:, 0])
        probs = torch.stack([1.0 - p1, p1], dim=1)
    else:
        probs = torch.softmax(sampled_logits, dim=1)

    confidence, prediction = probs.max(dim=1)
    correct = prediction.eq(sampled_target)
    true_probability = probs.gather(1, sampled_target[:, None]).squeeze(1)
    one_hot = F.one_hot(sampled_target, num_classes=class_count).to(dtype=probs.dtype)

    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    nll = -torch.log(true_probability.clamp_min(eps)).mean()
    confidence_np = confidence.detach().cpu().numpy().astype(np.float64, copy=False)
    correct_np = correct.detach().cpu().numpy().astype(np.float64, copy=False)

    edges = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    ece = 0.0
    mce = 0.0
    for bin_index in range(n_bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index == 0:
            in_bin = (confidence_np >= lower) & (confidence_np <= upper)
        else:
            in_bin = (confidence_np > lower) & (confidence_np <= upper)
        bin_count = int(in_bin.sum())
        if bin_count == 0:
            continue
        bin_accuracy = float(correct_np[in_bin].mean())
        bin_confidence = float(confidence_np[in_bin].mean())
        gap = abs(bin_accuracy - bin_confidence)
        ece += gap * (bin_count / sample_count)
        mce = max(mce, gap)

    mean_confidence = float(confidence_np.mean())
    accuracy = float(correct_np.mean())
    return SegmentationCalibrationMetrics(
        total_voxels=total,
        sampled_voxels=sample_count,
        sampling_fraction=float(sample_count / total),
        n_bins=int(n_bins),
        expected_calibration_error=float(ece),
        maximum_calibration_error=float(mce),
        brier_score=float(brier.item()),
        negative_log_likelihood=float(nll.item()),
        mean_confidence=mean_confidence,
        accuracy=accuracy,
        confidence_gap=float(mean_confidence - accuracy),
    )


def _binary_dilate(mask: torch.Tensor, iterations: int) -> torch.Tensor:
    if iterations < 0:
        raise ValueError("dilation_iterations 不能为负数")
    out = mask.float()
    for _ in range(iterations):
        out = F.max_pool3d(out, kernel_size=3, stride=1, padding=1)
    return out > 0.5


def select_uncertain_voxels(
    uncertainty: torch.Tensor,
    config: UncertaintyROIConfig = UncertaintyROIConfig(),
) -> torch.Tensor:
    """按每病例 Top-percent 选取高不确定体素并做轻量膨胀。

    该函数只生成 ROI mask；具体精修网络在 baseline 完成后接入。
    """
    if uncertainty.ndim != 5 or uncertainty.shape[1] != 1:
        raise ValueError("uncertainty 必须为 (B,1,D,H,W)")
    if not (0.0 < config.top_percent <= 100.0):
        raise ValueError("top_percent 必须位于 (0,100]")

    b = uncertainty.shape[0]
    flat = uncertainty.flatten(1)
    masks = []
    quantile = 1.0 - config.top_percent / 100.0
    for idx in range(b):
        threshold = torch.quantile(flat[idx], quantile)
        mask = uncertainty[idx : idx + 1] >= threshold
        if int(mask.sum()) < config.min_voxels:
            k = min(config.min_voxels, flat.shape[1])
            topk_idx = torch.topk(flat[idx], k=k, largest=True).indices
            fallback = torch.zeros_like(flat[idx], dtype=torch.bool)
            fallback[topk_idx] = True
            mask = fallback.view_as(uncertainty[idx : idx + 1])
        mask = _binary_dilate(mask, config.dilation_iterations)
        masks.append(mask)
    return torch.cat(masks, dim=0)


def uncertainty_error_overlap(
    uncertainty_mask: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """简单评估高不确定区域是否覆盖真实错误。

    返回：
    - error_recall: 所有错误体素中有多少落在 uncertainty ROI；
    - roi_error_rate: ROI 内有多少是真实错误；
    - roi_fraction: ROI 占总体素比例。
    """
    if uncertainty_mask.ndim != 5 or uncertainty_mask.shape[1] != 1:
        raise ValueError("uncertainty_mask 必须为 (B,1,D,H,W)")
    if prediction.ndim == 5 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 5 and target.shape[1] == 1:
        target = target[:, 0]
    if prediction.ndim != 4 or target.ndim != 4:
        raise ValueError("prediction/target 必须为 (B,D,H,W)")

    roi = uncertainty_mask[:, 0].bool()
    errors = prediction.long() != target.long()
    roi_errors = torch.logical_and(roi, errors)

    error_count = int(errors.sum())
    roi_count = int(roi.sum())
    total = errors.numel()
    return {
        "error_recall": 1.0 if error_count == 0 else float(roi_errors.sum().item() / error_count),
        "roi_error_rate": 0.0 if roi_count == 0 else float(roi_errors.sum().item() / roi_count),
        "roi_fraction": float(roi_count / total),
    }


def uncertainty_error_metrics(
    uncertainty: np.ndarray | torch.Tensor,
    prediction: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    *,
    top_percent: float = 10.0,
    max_samples: int = 500_000,
    seed: int = 42,
) -> UncertaintyErrorMetrics:
    """量化不确定性对真实分割错误的排序/定位能力。

    ``error_auroc/error_auprc`` 将“预测错误体素”视为正类、uncertainty 视为分数。
    对超大 3D 体积会进行固定随机种子的体素下采样，避免排序指标占用过多内存。
    Top-percent 指标仍在同一采样集合上计算，因此不同模型比较时必须固定参数/seed。

    当所有采样体素都正确或都错误时 AUROC/AUPRC 没有正常二分类定义，返回 ``None``，
    而不是伪造 0/1 分数。
    """
    if not (0.0 < float(top_percent) <= 100.0):
        raise ValueError("top_percent 必须位于 (0,100]")
    if max_samples <= 0:
        raise ValueError("max_samples 必须 > 0")

    def to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    uncertainty_np = to_numpy(uncertainty).astype(np.float64, copy=False)
    prediction_np = to_numpy(prediction)
    target_np = to_numpy(target)

    # 兼容 (B,1,D,H,W)/(1,D,H,W)/(D,H,W)，单病例评估时去掉 singleton 维度。
    uncertainty_np = np.squeeze(uncertainty_np)
    prediction_np = np.squeeze(prediction_np)
    target_np = np.squeeze(target_np)
    if uncertainty_np.shape != prediction_np.shape or prediction_np.shape != target_np.shape:
        raise ValueError(
            "uncertainty/prediction/target shape 必须一致，"
            f"当前为 {uncertainty_np.shape}/{prediction_np.shape}/{target_np.shape}"
        )
    if uncertainty_np.ndim < 1:
        raise ValueError("输入不能是标量")
    if not np.isfinite(uncertainty_np).all():
        raise ValueError("uncertainty 包含 NaN/Inf")

    scores_full = uncertainty_np.reshape(-1)
    errors_full = (prediction_np.reshape(-1) != target_np.reshape(-1)).astype(np.uint8)
    total = int(scores_full.size)
    if total == 0:
        raise ValueError("输入为空")

    if total > max_samples:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(total, size=int(max_samples), replace=False)
        scores = scores_full[indices]
        errors = errors_full[indices]
    else:
        scores = scores_full
        errors = errors_full

    sample_count = int(scores.size)
    error_count = int(errors.sum())
    correct_count = sample_count - error_count
    error_scores = scores[errors == 1]
    correct_scores = scores[errors == 0]

    auroc: float | None = None
    auprc: float | None = None
    if error_count > 0 and correct_count > 0:
        auroc = float(roc_auc_score(errors, scores))
        auprc = float(average_precision_score(errors, scores))

    top_fraction = float(top_percent) / 100.0
    top_k = max(1, min(sample_count, int(np.ceil(sample_count * top_fraction))))
    top_indices = np.argpartition(scores, sample_count - top_k)[sample_count - top_k :]
    top_errors = int(errors[top_indices].sum())

    return UncertaintyErrorMetrics(
        total_voxels=total,
        sampled_voxels=sample_count,
        sampling_fraction=float(sample_count / total),
        error_rate=float(error_count / sample_count),
        mean_uncertainty_error=(
            None if error_count == 0 else float(error_scores.mean())
        ),
        mean_uncertainty_correct=(
            None if correct_count == 0 else float(correct_scores.mean())
        ),
        error_auroc=auroc,
        error_auprc=auprc,
        top_percent=float(top_percent),
        top_uncertainty_error_recall=(
            1.0 if error_count == 0 else float(top_errors / error_count)
        ),
        top_uncertainty_error_rate=float(top_errors / top_k),
        top_uncertainty_fraction=float(top_k / sample_count),
    )
