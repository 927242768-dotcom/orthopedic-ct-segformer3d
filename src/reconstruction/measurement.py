"""物理空间几何测量基础。

所有输入/输出都以病人/影像物理坐标为准，而不是屏幕像素或重采样数组下标。
该模块只提供几何计算，不赋予任何临床诊断含义。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _point3(value: Sequence[float], name: str) -> np.ndarray:
    if len(value) != 3:
        raise ValueError(f"{name} 必须包含 3 个坐标")
    point = np.asarray([float(v) for v in value], dtype=np.float64)
    if not np.isfinite(point).all():
        raise ValueError(f"{name} 包含 NaN/Inf")
    return point


def distance_mm(point_a_xyz_mm: Sequence[float], point_b_xyz_mm: Sequence[float]) -> float:
    """返回两个物理 XYZ 点之间的欧氏距离，单位 mm。"""
    a = _point3(point_a_xyz_mm, "point_a")
    b = _point3(point_b_xyz_mm, "point_b")
    return float(np.linalg.norm(b - a))


def angle_degrees(
    point_a_xyz_mm: Sequence[float],
    vertex_b_xyz_mm: Sequence[float],
    point_c_xyz_mm: Sequence[float],
) -> float:
    """返回 ∠ABC，单位 degree，范围 [0, 180]。"""
    a = _point3(point_a_xyz_mm, "point_a")
    b = _point3(vertex_b_xyz_mm, "vertex_b")
    c = _point3(point_c_xyz_mm, "point_c")
    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        raise ValueError("角度测量要求 A、B、C 三点中 B 与 A/C 不能重合")
    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def index_xyz_to_physical_mm(
    index_xyz: Sequence[float],
    *,
    spacing_xyz_mm: Sequence[float],
    origin_xyz_mm: Sequence[float],
    direction: Sequence[float],
) -> tuple[float, float, float]:
    """把连续 voxel index XYZ 转换为 SimpleITK/DICOM 物理 XYZ 毫米坐标。"""
    index = _point3(index_xyz, "index_xyz")
    spacing = _point3(spacing_xyz_mm, "spacing_xyz_mm")
    origin = _point3(origin_xyz_mm, "origin_xyz_mm")
    if np.any(spacing <= 0):
        raise ValueError("spacing_xyz_mm 必须全部 > 0")
    if len(direction) != 9:
        raise ValueError("direction 必须包含 3×3=9 个值")
    matrix = np.asarray([float(v) for v in direction], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("direction 包含 NaN/Inf")
    physical = origin + matrix @ (index * spacing)
    return tuple(float(v) for v in physical)
