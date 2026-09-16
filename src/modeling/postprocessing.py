"""分割预测后处理。

所有后处理默认关闭，只有配置显式启用时才生效，避免改变历史实验结果。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import generate_binary_structure, label as connected_components


def remove_small_components(
    prediction: np.ndarray,
    *,
    min_component_voxels: int = 0,
    connectivity: int = 3,
) -> np.ndarray:
    """按类别删除小于阈值的 3D 连通分量。

    ``prediction`` 为整数标签 ``(D,H,W)``。背景必须为 0；每个前景类别独立做
    连通域分析，因此二分类和多分类都可用。阈值 ``<=1`` 时严格 no-op。
    """
    array = np.asarray(prediction)
    if array.ndim != 3:
        raise ValueError(f"prediction 必须是 3D，实际 shape={array.shape}")
    if min_component_voxels < 0:
        raise ValueError("min_component_voxels 必须 >= 0")
    if connectivity not in {1, 2, 3}:
        raise ValueError("connectivity 必须为 1/2/3")
    if min_component_voxels <= 1:
        return array.copy()

    result = array.copy()
    structure = generate_binary_structure(3, connectivity)
    for class_id in np.unique(array):
        class_id_int = int(class_id)
        if class_id_int <= 0:
            continue
        mask = array == class_id
        labels, count = connected_components(mask, structure=structure)
        if int(count) == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=int(count) + 1)
        remove_ids = np.flatnonzero(sizes < int(min_component_voxels))
        remove_ids = remove_ids[remove_ids > 0]
        if remove_ids.size:
            result[np.isin(labels, remove_ids)] = 0
    return result


def postprocess_prediction(
    prediction: np.ndarray,
    config: dict | None,
) -> np.ndarray:
    """根据 ``inference.postprocessing`` 配置处理预测标签。"""
    cfg = dict(config or {})
    min_voxels = int(cfg.get("min_component_voxels", 0) or 0)
    connectivity = int(cfg.get("connectivity", 3) or 3)
    return remove_small_components(
        prediction,
        min_component_voxels=min_voxels,
        connectivity=connectivity,
    )
