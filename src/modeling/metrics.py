"""3D 骨科分割评价指标。

包含区域指标和表面指标，按类别逐病例计算，便于最终导出
`metrics_per_case.csv` 和论文统计。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import (
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
    label as connected_components,
)


@dataclass
class BinarySegMetrics:
    dice: float
    iou: float
    precision: float
    recall: float
    hd95_mm: float
    assd_mm: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class StructuralSegMetrics:
    pred_components: int
    target_components: int
    component_count_error: int
    false_merge_count: int
    false_break_count: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _safe_ratio(num: float, den: float, *, both_empty_value: float = 1.0) -> float:
    if den == 0:
        return both_empty_value
    return float(num / den)


def binary_overlap_metrics(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError("pred 与 target shape 不一致")

    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    pred_sum = tp + fp
    target_sum = tp + fn

    if pred_sum == 0 and target_sum == 0:
        return 1.0, 1.0, 1.0, 1.0

    dice = _safe_ratio(2.0 * tp, pred_sum + target_sum, both_empty_value=1.0)
    iou = _safe_ratio(tp, tp + fp + fn, both_empty_value=1.0)
    precision = _safe_ratio(tp, tp + fp, both_empty_value=1.0 if target_sum == 0 else 0.0)
    recall = _safe_ratio(tp, tp + fn, both_empty_value=1.0 if pred_sum == 0 else 0.0)
    return dice, iou, precision, recall


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)


def symmetric_surface_distances_mm(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_dhw_mm: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """返回 pred→target 与 target→pred 表面距离。

    若两者都为空，返回两个空数组；若仅一方为空，返回 [inf]，明确标记失败。
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError("pred 与 target shape 不一致")
    if len(spacing_dhw_mm) != pred.ndim:
        raise ValueError("spacing 维度与 mask 不一致")

    pred_surface = _surface(pred)
    target_surface = _surface(target)

    if not pred_surface.any() and not target_surface.any():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if not pred_surface.any() or not target_surface.any():
        inf = np.asarray([np.inf], dtype=np.float64)
        return inf, inf.copy()

    # distance_transform_edt 输入为非零区域到最近 0 的距离，因此对 ~surface 计算。
    dt_to_target = distance_transform_edt(~target_surface, sampling=tuple(spacing_dhw_mm))
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=tuple(spacing_dhw_mm))
    pred_to_target = dt_to_target[pred_surface].astype(np.float64)
    target_to_pred = dt_to_pred[target_surface].astype(np.float64)
    return pred_to_target, target_to_pred


def surface_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_dhw_mm: Sequence[float],
) -> tuple[float, float]:
    a, b = symmetric_surface_distances_mm(pred, target, spacing_dhw_mm)
    if a.size == 0 and b.size == 0:
        return 0.0, 0.0
    all_distances = np.concatenate([a, b])
    if np.isinf(all_distances).any():
        return float("inf"), float("inf")
    hd95 = float(np.percentile(all_distances, 95))
    assd = float((a.mean() + b.mean()) / 2.0)
    return hd95, assd


def compute_binary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_dhw_mm: Sequence[float] = (1.0, 1.0, 1.0),
) -> BinarySegMetrics:
    dice, iou, precision, recall = binary_overlap_metrics(pred, target)
    hd95, assd = surface_metrics(pred, target, spacing_dhw_mm)
    return BinarySegMetrics(
        dice=dice,
        iou=iou,
        precision=precision,
        recall=recall,
        hd95_mm=hd95,
        assd_mm=assd,
    )


def compute_structural_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    connectivity: int = 3,
) -> StructuralSegMetrics:
    """统计 3D 连通性错误，辅助评估 topology loss。

    false_merge_count：一个预测连通分量同时覆盖多个真值分量时，多合并的分量数；
    false_break_count：一个真值连通分量被多个预测分量覆盖时，多断裂的分量数。
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError("pred 与 target shape 不一致")
    if pred.ndim != 3:
        raise ValueError("结构指标当前只支持 3D mask")
    if connectivity not in {1, 2, 3}:
        raise ValueError("connectivity 必须为 1/2/3")

    structure = generate_binary_structure(3, connectivity)
    pred_labels, pred_count = connected_components(pred, structure=structure)
    target_labels, target_count = connected_components(target, structure=structure)

    false_merge_count = 0
    for pred_id in range(1, int(pred_count) + 1):
        overlapping = np.unique(target_labels[pred_labels == pred_id])
        overlapping = overlapping[overlapping > 0]
        false_merge_count += max(0, int(len(overlapping)) - 1)

    false_break_count = 0
    for target_id in range(1, int(target_count) + 1):
        overlapping = np.unique(pred_labels[target_labels == target_id])
        overlapping = overlapping[overlapping > 0]
        false_break_count += max(0, int(len(overlapping)) - 1)

    return StructuralSegMetrics(
        pred_components=int(pred_count),
        target_components=int(target_count),
        component_count_error=abs(int(pred_count) - int(target_count)),
        false_merge_count=int(false_merge_count),
        false_break_count=int(false_break_count),
    )


def compute_multiclass_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    class_ids: Iterable[int],
    spacing_dhw_mm: Sequence[float] = (1.0, 1.0, 1.0),
) -> dict[int, BinarySegMetrics]:
    pred = np.asarray(pred)
    target = np.asarray(target)
    if pred.shape != target.shape:
        raise ValueError("pred 与 target shape 不一致")
    return {
        int(class_id): compute_binary_metrics(
            pred == int(class_id),
            target == int(class_id),
            spacing_dhw_mm,
        )
        for class_id in class_ids
    }
