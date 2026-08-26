"""从标准化 NIfTI label/prediction 导出物理空间 PLY 网格。

用途：
- 在没有 Web 3D 渲染之前先固定可追溯的 mask→mesh 工程入口；
- 支持二值前景或指定类别；
- 输出 PLY 与 JSON 摘要。

该工具不做诊断，也不自动平滑/修补拓扑；高保真后处理必须通过真实数据消融验证。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from src.reconstruction.mesh import (
    mask_to_surface_mesh,
    simplify_mesh_vertex_clustering,
    write_ascii_ply,
)
from src.sitk_compat import sitk_io_path


def export_nifti_mask_mesh(
    input_path: str | Path,
    output_ply: str | Path,
    *,
    class_id: int | None = None,
    summary_path: str | Path | None = None,
    simplify_cluster_mm: float | None = None,
    feature_preservation_strength: float = 0.0,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_ply = Path(output_ply)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    image = sitk.ReadImage(sitk_io_path(input_path))
    if image.GetDimension() != 3:
        raise ValueError(f"只支持 3D NIfTI，当前维度={image.GetDimension()}")

    array_zyx = sitk.GetArrayFromImage(image)
    if not np.isfinite(array_zyx).all():
        raise ValueError("输入 mask 包含 NaN/Inf")

    unique_values = sorted(int(v) for v in np.unique(np.rint(array_zyx)))
    if class_id is None:
        mask = array_zyx > 0
        selection = "foreground_gt_0"
    else:
        if int(class_id) not in unique_values:
            raise ValueError(f"class_id={class_id} 不存在，当前标签={unique_values}")
        mask = array_zyx == int(class_id)
        selection = f"class_{int(class_id)}"

    foreground_voxels = int(mask.sum())
    if foreground_voxels == 0:
        raise ValueError("选定 mask 没有前景体素")

    full_mesh = mask_to_surface_mesh(
        mask.astype(np.uint8),
        spacing_xyz_mm=image.GetSpacing(),
        origin_xyz_mm=image.GetOrigin(),
        direction=image.GetDirection(),
        level=0.5,
    )
    mesh = full_mesh
    simplification: dict[str, object] | None = None
    if simplify_cluster_mm is not None:
        mesh = simplify_mesh_vertex_clustering(
            full_mesh,
            cluster_size_mm=float(simplify_cluster_mm),
            feature_preservation_strength=float(feature_preservation_strength),
        )
        full_vertices = full_mesh.vertices_xyz_mm.astype(np.float64, copy=False)
        simplified_vertices = mesh.vertices_xyz_mm.astype(np.float64, copy=False)
        simplified_tree = cKDTree(simplified_vertices)
        full_tree = cKDTree(full_vertices)
        full_to_simplified = simplified_tree.query(full_vertices, k=1, workers=-1)[0]
        simplified_to_full = full_tree.query(simplified_vertices, k=1, workers=-1)[0]
        all_distances = np.concatenate([full_to_simplified, simplified_to_full])
        simplification = {
            "method": (
                "vertex_clustering_feature_weighted"
                if float(feature_preservation_strength) > 0
                else "vertex_clustering"
            ),
            "cluster_size_mm": float(simplify_cluster_mm),
            "feature_preservation_strength": float(feature_preservation_strength),
            "original_vertex_count": full_mesh.vertex_count,
            "original_face_count": full_mesh.face_count,
            "original_surface_area_mm2": float(full_mesh.surface_area_mm2),
            "vertex_reduction_fraction": float(1.0 - mesh.vertex_count / full_mesh.vertex_count),
            "face_reduction_fraction": float(1.0 - mesh.face_count / full_mesh.face_count),
            "surface_area_relative_change": float(
                (mesh.surface_area_mm2 - full_mesh.surface_area_mm2)
                / max(full_mesh.surface_area_mm2, 1e-12)
            ),
            "vertex_assd_mm": float(
                (full_to_simplified.mean() + simplified_to_full.mean()) / 2.0
            ),
            "vertex_hd95_mm": float(np.percentile(all_distances, 95)),
            "vertex_hdmax_mm": float(all_distances.max()),
            "note": (
                "Vertex-nearest engineering approximation between full and simplified surfaces; "
                "not a clinical surface-distance metric."
            ),
        }
    write_ascii_ply(mesh, output_ply)

    vertices = mesh.vertices_xyz_mm.astype(np.float64, copy=False)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    summary: dict[str, object] = {
        "input": str(input_path),
        "output_ply": str(output_ply),
        "selection": selection,
        "source_label_values": unique_values,
        "foreground_voxels": foreground_voxels,
        "spacing_xyz_mm": [float(v) for v in image.GetSpacing()],
        "origin_xyz_mm": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "surface_area_mm2": float(mesh.surface_area_mm2),
        "simplification": simplification,
        "bounds_xyz_mm": {
            "min": [float(v) for v in bounds_min],
            "max": [float(v) for v in bounds_max],
        },
        "note": "Engineering geometry output only; not a clinical measurement or model-performance result.",
    }

    if summary_path is None:
        summary_path = output_ply.with_suffix(".json")
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 3D NIfTI mask 导出物理空间 PLY")
    parser.add_argument("input", type=Path, help="label/prediction NIfTI")
    parser.add_argument("output", type=Path, help="输出 PLY")
    parser.add_argument("--class-id", type=int, default=None, help="指定标签类别；默认导出所有 >0 前景")
    parser.add_argument("--summary", type=Path, default=None, help="可选 JSON 摘要路径")
    parser.add_argument(
        "--simplify-cluster-mm",
        type=float,
        default=None,
        help="可选物理空间 vertex-clustering 网格大小（mm）；不提供则保留全分辨率网格",
    )
    parser.add_argument(
        "--feature-preservation-strength",
        type=float,
        default=0.0,
        help="可选曲率/关键边缘代理权重；0 保持原 vertex-clustering 行为",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()
    summary = export_nifti_mask_mesh(
        args.input,
        args.output,
        class_id=args.class_id,
        summary_path=args.summary,
        simplify_cluster_mm=args.simplify_cluster_mm,
        feature_preservation_strength=args.feature_preservation_strength,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
