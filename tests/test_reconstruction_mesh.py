from pathlib import Path

import numpy as np

from src.reconstruction.mesh import (
    mask_to_surface_mesh,
    simplify_mesh_vertex_clustering,
    vertex_normal_variation_scores,
    write_ascii_ply,
)


def test_mask_to_surface_mesh_uses_physical_spacing_and_origin(tmp_path: Path) -> None:
    mask = np.zeros((16, 18, 20), dtype=np.uint8)
    mask[4:12, 5:14, 6:15] = 1

    mesh = mask_to_surface_mesh(
        mask,
        spacing_xyz_mm=(0.7, 0.8, 1.5),
        origin_xyz_mm=(10.0, 20.0, 30.0),
    )

    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert mesh.surface_area_mm2 > 0.0
    assert mesh.vertices_xyz_mm.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert float(mesh.vertices_xyz_mm[:, 0].min()) > 10.0
    assert float(mesh.vertices_xyz_mm[:, 1].min()) > 20.0
    assert float(mesh.vertices_xyz_mm[:, 2].min()) > 30.0

    ply = write_ascii_ply(mesh, tmp_path / "surface.ply")
    text = ply.read_text(encoding="utf-8")
    assert text.startswith("ply\nformat ascii 1.0")
    assert f"element vertex {mesh.vertex_count}" in text
    assert f"element face {mesh.face_count}" in text


def test_vertex_clustering_reduces_mesh_and_preserves_valid_faces() -> None:
    mask = np.zeros((30, 32, 34), dtype=np.uint8)
    mask[5:25, 6:26, 7:27] = 1
    full = mask_to_surface_mesh(mask)

    simplified = simplify_mesh_vertex_clustering(full, cluster_size_mm=2.0)

    assert simplified.vertex_count < full.vertex_count
    assert simplified.face_count < full.face_count
    assert simplified.surface_area_mm2 > 0.0
    assert np.all(simplified.faces >= 0)
    assert int(simplified.faces.max()) < simplified.vertex_count
    assert np.isfinite(simplified.vertices_xyz_mm).all()
    assert np.isfinite(simplified.normals_xyz).all()
    # 工程简化不应把一个规则立方体的面积改到完全失真。
    relative_area_change = abs(
        simplified.surface_area_mm2 - full.surface_area_mm2
    ) / full.surface_area_mm2
    assert relative_area_change < 0.15


def test_feature_weighted_clustering_preserves_high_normal_variation_vertices() -> None:
    mask = np.zeros((30, 32, 34), dtype=np.uint8)
    mask[5:25, 6:26, 7:27] = 1
    full = mask_to_surface_mesh(mask)
    scores = vertex_normal_variation_scores(full)

    assert scores.shape == (full.vertex_count,)
    assert np.isfinite(scores).all()
    assert float(scores.max()) > 0.0

    baseline = simplify_mesh_vertex_clustering(full, cluster_size_mm=2.0)
    protected = simplify_mesh_vertex_clustering(
        full,
        cluster_size_mm=2.0,
        feature_preservation_strength=8.0,
    )
    threshold = float(np.percentile(scores, 90))
    feature_vertices = full.vertices_xyz_mm[scores >= threshold]

    def mean_nearest_distance(source: np.ndarray, target: np.ndarray) -> float:
        deltas = source[:, None, :] - target[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        return float(distances.min(axis=1).mean())

    baseline_distance = mean_nearest_distance(
        feature_vertices,
        baseline.vertices_xyz_mm,
    )
    protected_distance = mean_nearest_distance(
        feature_vertices,
        protected.vertices_xyz_mm,
    )
    assert protected_distance <= baseline_distance
    assert protected.vertex_count == baseline.vertex_count
    assert protected.face_count == baseline.face_count


def test_mask_to_surface_mesh_rejects_empty_mask() -> None:
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    try:
        mask_to_surface_mesh(mask)
    except ValueError as exc:
        assert "没有可提取的等值面" in str(exc)
    else:
        raise AssertionError("空 mask 应拒绝生成 mesh")
