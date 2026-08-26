"""分割 mask 到物理空间三角网格的基础实现。

输入约定：
- mask 使用医学影像常见的 numpy (Z,Y,X) 轴顺序；
- spacing/origin/direction 使用 SimpleITK 的 (X,Y,Z) 物理空间定义；
- marching cubes 在 voxel array 坐标中提取表面后，显式转换到物理 XYZ。

当前仅提供可靠 baseline，不包含平滑、孔洞修复或强制拓扑修正；这些操作必须在真实骨结构上验证后再启用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area


@dataclass(frozen=True)
class SurfaceMesh:
    vertices_xyz_mm: np.ndarray
    faces: np.ndarray
    normals_xyz: np.ndarray
    values: np.ndarray
    surface_area_mm2: float

    @property
    def vertex_count(self) -> int:
        return int(self.vertices_xyz_mm.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.faces.shape[0])


def _direction_matrix(direction: Sequence[float] | None) -> np.ndarray:
    if direction is None:
        return np.eye(3, dtype=np.float64)
    values = np.asarray(direction, dtype=np.float64)
    if values.size != 9:
        raise ValueError("direction 必须包含 9 个元素")
    matrix = values.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("direction 包含 NaN/Inf")
    return matrix


def mask_to_surface_mesh(
    mask_zyx: np.ndarray,
    *,
    spacing_xyz_mm: Sequence[float] = (1.0, 1.0, 1.0),
    origin_xyz_mm: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | None = None,
    level: float = 0.5,
) -> SurfaceMesh:
    """使用 marching cubes 将二值/概率 mask 转换到真实物理 XYZ 网格。"""
    mask = np.asarray(mask_zyx)
    if mask.ndim != 3:
        raise ValueError("mask_zyx 必须为 3D")
    if min(mask.shape) < 2:
        raise ValueError("mask 每个维度至少需要 2 个 voxel")
    if not np.isfinite(mask).all():
        raise ValueError("mask 包含 NaN/Inf")

    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=np.float64)
    origin_xyz = np.asarray(origin_xyz_mm, dtype=np.float64)
    if spacing_xyz.shape != (3,) or np.any(spacing_xyz <= 0) or not np.isfinite(spacing_xyz).all():
        raise ValueError("spacing_xyz_mm 必须为 3 个有限正数")
    if origin_xyz.shape != (3,) or not np.isfinite(origin_xyz).all():
        raise ValueError("origin_xyz_mm 必须为 3 个有限数")

    minimum = float(mask.min())
    maximum = float(mask.max())
    if not (minimum <= level <= maximum) or minimum == maximum:
        raise ValueError(
            f"mask 没有可提取的等值面: min={minimum}, max={maximum}, level={level}"
        )

    # skimage 输入轴是 Z,Y,X，因此 spacing 同样必须按 Z,Y,X 传入。
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    verts_zyx, faces, normals_zyx, values = marching_cubes(
        mask.astype(np.float32, copy=False),
        level=level,
        spacing=spacing_zyx,
        allow_degenerate=False,
    )

    # 转换到局部物理 XYZ，再应用方向矩阵和 origin。
    verts_local_xyz = verts_zyx[:, ::-1].astype(np.float64, copy=False)
    direction_matrix = _direction_matrix(direction)
    vertices_xyz = verts_local_xyz @ direction_matrix.T + origin_xyz

    normals_local_xyz = normals_zyx[:, ::-1].astype(np.float64, copy=False)
    normals_xyz = normals_local_xyz @ direction_matrix.T
    normal_norm = np.linalg.norm(normals_xyz, axis=1, keepdims=True)
    normals_xyz = normals_xyz / np.maximum(normal_norm, 1e-12)

    faces = faces.astype(np.int32, copy=False)
    area = float(mesh_surface_area(vertices_xyz, faces))
    return SurfaceMesh(
        vertices_xyz_mm=vertices_xyz.astype(np.float32),
        faces=faces,
        normals_xyz=normals_xyz.astype(np.float32),
        values=values.astype(np.float32, copy=False),
        surface_area_mm2=area,
    )


def _vertex_normals_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals.astype(np.float32)


def vertex_normal_variation_scores(mesh: SurfaceMesh) -> np.ndarray:
    """用相邻顶点法向差异构造轻量曲率/尖锐特征代理分数。"""
    if mesh.vertex_count < 1:
        return np.empty(0, dtype=np.float32)
    faces = mesh.faces.astype(np.int64, copy=False)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
        return np.zeros(mesh.vertex_count, dtype=np.float32)

    normals = np.asarray(mesh.normals_xyz, dtype=np.float64)
    if normals.shape != (mesh.vertex_count, 3) or not np.isfinite(normals).all():
        normals = _vertex_normals_from_faces(
            mesh.vertices_xyz_mm.astype(np.float64, copy=False),
            faces,
        ).astype(np.float64, copy=False)
    normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_lengths, 1e-12)

    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    dots = np.sum(normals[edges[:, 0]] * normals[edges[:, 1]], axis=1)
    variation = 1.0 - np.clip(dots, -1.0, 1.0)

    scores = np.zeros(mesh.vertex_count, dtype=np.float64)
    np.maximum.at(scores, edges[:, 0], variation)
    np.maximum.at(scores, edges[:, 1], variation)
    return scores.astype(np.float32)


def simplify_mesh_vertex_clustering(
    mesh: SurfaceMesh,
    *,
    cluster_size_mm: float,
    feature_preservation_strength: float = 0.0,
) -> SurfaceMesh:
    """按物理空间规则网格做确定性的 vertex-clustering 简化。

    ``feature_preservation_strength>0`` 时，以相邻顶点法向变化作为轻量曲率/关键边缘代理，
    在每个聚类中提高高特征顶点对代表点位置的权重。该策略只是高保真重建候选，
    不强制拓扑修复，也不保证保持所有细小骨性突起；正式参数仍需真实 prediction 消融。
    """
    cluster_size = float(cluster_size_mm)
    if not np.isfinite(cluster_size) or cluster_size <= 0:
        raise ValueError("cluster_size_mm 必须为有限正数")
    feature_strength = float(feature_preservation_strength)
    if not np.isfinite(feature_strength) or feature_strength < 0:
        raise ValueError("feature_preservation_strength 必须为有限非负数")
    if mesh.vertex_count < 4 or mesh.face_count < 1:
        raise ValueError("mesh 过小，无法简化")

    vertices = mesh.vertices_xyz_mm.astype(np.float64, copy=False)
    minimum = vertices.min(axis=0)
    cluster_keys = np.floor((vertices - minimum) / cluster_size).astype(np.int64)
    _, inverse = np.unique(cluster_keys, axis=0, return_inverse=True)
    cluster_count = int(inverse.max()) + 1

    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    vertex_weights = np.ones(mesh.vertex_count, dtype=np.float64)
    if feature_strength > 0:
        feature_scores = vertex_normal_variation_scores(mesh).astype(np.float64, copy=False)
        max_score = float(feature_scores.max()) if feature_scores.size else 0.0
        if max_score > 1e-12:
            vertex_weights += feature_strength * (feature_scores / max_score)
    weighted_counts = np.bincount(
        inverse,
        weights=vertex_weights,
        minlength=cluster_count,
    ).astype(np.float64)

    new_vertices = np.zeros((cluster_count, 3), dtype=np.float64)
    for axis in range(3):
        new_vertices[:, axis] = np.bincount(
            inverse,
            weights=vertices[:, axis] * vertex_weights,
            minlength=cluster_count,
        ) / np.maximum(weighted_counts, 1e-12)

    source_values = mesh.values.astype(np.float64, copy=False)
    if source_values.shape[0] == mesh.vertex_count:
        new_values = np.bincount(
            inverse,
            weights=source_values,
            minlength=cluster_count,
        ) / np.maximum(counts, 1.0)
    else:
        new_values = np.zeros(cluster_count, dtype=np.float64)

    remapped = inverse[mesh.faces.astype(np.int64, copy=False)]
    nondegenerate = (
        (remapped[:, 0] != remapped[:, 1])
        & (remapped[:, 1] != remapped[:, 2])
        & (remapped[:, 0] != remapped[:, 2])
    )
    remapped = remapped[nondegenerate]
    if remapped.size == 0:
        raise ValueError("cluster_size_mm 过大，所有三角面均退化")

    # 同一个无向三角形可能因聚类产生重复；保留第一次出现的原始 winding。
    canonical = np.sort(remapped, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    faces = remapped[np.sort(unique_indices)].astype(np.int32, copy=False)

    tri = new_vertices[faces]
    double_area = np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    faces = faces[double_area > 1e-10]
    if faces.size == 0:
        raise ValueError("简化后没有有效三角面")

    used = np.unique(faces.reshape(-1))
    compact_index = np.full(cluster_count, -1, dtype=np.int64)
    compact_index[used] = np.arange(len(used), dtype=np.int64)
    compact_vertices = new_vertices[used]
    compact_values = new_values[used]
    compact_faces = compact_index[faces].astype(np.int32, copy=False)
    normals = _vertex_normals_from_faces(compact_vertices, compact_faces)
    area = float(mesh_surface_area(compact_vertices, compact_faces))
    return SurfaceMesh(
        vertices_xyz_mm=compact_vertices.astype(np.float32),
        faces=compact_faces,
        normals_xyz=normals,
        values=compact_values.astype(np.float32),
        surface_area_mm2=area,
    )


def write_ascii_ply(mesh: SurfaceMesh, output_path: str | Path) -> Path:
    """写出轻量 ASCII PLY，便于后续 Web/Three.js 或离线查看器接入。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {mesh.vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {mesh.face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines = header
    lines.extend(
        f"{float(x):.6f} {float(y):.6f} {float(z):.6f}"
        for x, y, z in mesh.vertices_xyz_mm
    )
    lines.extend(
        f"3 {int(a)} {int(b)} {int(c)}"
        for a, b, c in mesh.faces
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
