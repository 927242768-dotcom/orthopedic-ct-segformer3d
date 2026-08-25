"""基于物理距离 signed-distance field 的表面平滑工程 baseline。

目的：
- 不直接在三角网格上随意拉点，而是在真实 mm spacing 下构造 SDF；
- 使用以 mm 为单位的 Gaussian 平滑，再提取 0 等值面；
- 显式比较原始 Marching-Cubes 表面与 SDF 表面误差；
- 检查阈值后的连通域数量是否变化，防止“更光滑”却把骨结构错误粘连/断开。

这不是学习型高保真重建，也不是临床表面精度；只作为可解释工程 baseline。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

from src.reconstruction.mesh import SurfaceMesh, mask_to_surface_mesh, write_ascii_ply
from src.sitk_compat import sitk_io_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SDFSurfaceMetrics:
    sigma_mm: float
    original_components: int
    smoothed_components: int
    component_count_preserved: bool
    original_vertex_count: int
    sdf_vertex_count: int
    original_face_count: int
    sdf_face_count: int
    original_surface_area_mm2: float
    sdf_surface_area_mm2: float
    surface_area_relative_change: float
    vertex_assd_mm: float
    vertex_hd95_mm: float
    vertex_hdmax_mm: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def signed_distance_field_mm(
    mask_zyx: np.ndarray,
    *,
    spacing_xyz_mm: Sequence[float],
) -> np.ndarray:
    mask = np.asarray(mask_zyx, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("mask_zyx 必须为 3D")
    if not np.any(mask) or np.all(mask):
        raise ValueError("mask 必须同时包含前景和背景")
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if spacing_xyz.shape != (3,) or np.any(spacing_xyz <= 0):
        raise ValueError("spacing_xyz_mm 必须是 3 个正数")
    spacing_zyx = tuple(float(v) for v in spacing_xyz[::-1])
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing_zyx)
    inside = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
    return (outside - inside).astype(np.float32)


def smooth_sdf_mm(
    sdf_zyx: np.ndarray,
    *,
    spacing_xyz_mm: Sequence[float],
    sigma_mm: float,
) -> np.ndarray:
    sdf = np.asarray(sdf_zyx, dtype=np.float32)
    sigma = float(sigma_mm)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma_mm 必须为有限非负数")
    if sigma == 0:
        return sdf.copy()
    spacing_xyz = np.asarray(spacing_xyz_mm, dtype=np.float64)
    if spacing_xyz.shape != (3,) or np.any(spacing_xyz <= 0):
        raise ValueError("spacing_xyz_mm 必须是 3 个正数")
    spacing_zyx = spacing_xyz[::-1]
    sigma_zyx = tuple(float(sigma / spacing) for spacing in spacing_zyx)
    return ndimage.gaussian_filter(sdf, sigma=sigma_zyx, mode="nearest").astype(np.float32)


def _component_count(mask: np.ndarray) -> int:
    _, count = ndimage.label(np.asarray(mask, dtype=bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    return int(count)


def _mesh_distance_metrics(reference: SurfaceMesh, candidate: SurfaceMesh) -> tuple[float, float, float]:
    ref = reference.vertices_xyz_mm.astype(np.float64, copy=False)
    cand = candidate.vertices_xyz_mm.astype(np.float64, copy=False)
    ref_tree = cKDTree(ref)
    cand_tree = cKDTree(cand)
    ref_to_cand = cand_tree.query(ref, k=1, workers=-1)[0]
    cand_to_ref = ref_tree.query(cand, k=1, workers=-1)[0]
    combined = np.concatenate([ref_to_cand, cand_to_ref])
    assd = float((ref_to_cand.mean() + cand_to_ref.mean()) / 2.0)
    return assd, float(np.percentile(combined, 95)), float(combined.max())


def sdf_surface_from_mask(
    mask_zyx: np.ndarray,
    *,
    spacing_xyz_mm: Sequence[float],
    origin_xyz_mm: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] | None = None,
    sigma_mm: float = 0.4,
) -> tuple[SurfaceMesh, SDFSurfaceMetrics]:
    mask = np.asarray(mask_zyx, dtype=bool)
    sdf = signed_distance_field_mm(mask, spacing_xyz_mm=spacing_xyz_mm)
    smoothed = smooth_sdf_mm(sdf, spacing_xyz_mm=spacing_xyz_mm, sigma_mm=sigma_mm)

    original_mesh = mask_to_surface_mesh(
        mask.astype(np.uint8),
        spacing_xyz_mm=spacing_xyz_mm,
        origin_xyz_mm=origin_xyz_mm,
        direction=direction,
        level=0.5,
    )
    sdf_mesh = mask_to_surface_mesh(
        smoothed,
        spacing_xyz_mm=spacing_xyz_mm,
        origin_xyz_mm=origin_xyz_mm,
        direction=direction,
        level=0.0,
    )
    original_components = _component_count(mask)
    smoothed_components = _component_count(smoothed <= 0.0)
    assd, hd95, hdmax = _mesh_distance_metrics(original_mesh, sdf_mesh)
    original_area = float(original_mesh.surface_area_mm2)
    sdf_area = float(sdf_mesh.surface_area_mm2)
    area_change = 0.0 if original_area <= 1e-12 else float((sdf_area - original_area) / original_area)
    metrics = SDFSurfaceMetrics(
        sigma_mm=float(sigma_mm),
        original_components=original_components,
        smoothed_components=smoothed_components,
        component_count_preserved=original_components == smoothed_components,
        original_vertex_count=original_mesh.vertex_count,
        sdf_vertex_count=sdf_mesh.vertex_count,
        original_face_count=original_mesh.face_count,
        sdf_face_count=sdf_mesh.face_count,
        original_surface_area_mm2=original_area,
        sdf_surface_area_mm2=sdf_area,
        surface_area_relative_change=area_change,
        vertex_assd_mm=assd,
        vertex_hd95_mm=hd95,
        vertex_hdmax_mm=hdmax,
    )
    return sdf_mesh, metrics


def export_nifti_sdf_surface(
    input_path: str | Path,
    output_ply: str | Path,
    *,
    class_id: int | None = None,
    sigma_mm: float = 0.4,
    summary_path: str | Path | None = None,
    require_component_preservation: bool = True,
) -> dict[str, object]:
    source = _resolve(input_path)
    output = _resolve(output_ply)
    if not source.exists():
        raise FileNotFoundError(source)
    image = sitk.ReadImage(sitk_io_path(source))
    if image.GetDimension() != 3:
        raise ValueError("只支持 3D NIfTI")
    array = sitk.GetArrayFromImage(image)
    values = sorted(int(v) for v in np.unique(np.rint(array)))
    if class_id is None:
        mask = array > 0
        selection = "foreground_gt_0"
    else:
        if int(class_id) not in values:
            raise ValueError(f"class_id={class_id} 不存在，当前标签={values}")
        mask = array == int(class_id)
        selection = f"class_{int(class_id)}"
    if not np.any(mask):
        raise ValueError("选定 mask 没有前景")

    mesh, metrics = sdf_surface_from_mask(
        mask,
        spacing_xyz_mm=image.GetSpacing(),
        origin_xyz_mm=image.GetOrigin(),
        direction=image.GetDirection(),
        sigma_mm=sigma_mm,
    )
    if require_component_preservation and not metrics.component_count_preserved:
        raise ValueError(
            "SDF 平滑改变了前景连通域数量，默认拒绝导出: "
            f"{metrics.original_components} -> {metrics.smoothed_components}; "
            "如确有研究目的，必须显式允许 component change"
        )
    write_ascii_ply(mesh, output)
    vertices = mesh.vertices_xyz_mm.astype(np.float64, copy=False)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    summary: dict[str, object] = {
        "input": str(source),
        "output_ply": str(output),
        "selection": selection,
        "surface_method": "sdf_smoothed_zero_level",
        "source_label_values": values,
        "spacing_xyz_mm": [float(v) for v in image.GetSpacing()],
        "sigma_mm": float(sigma_mm),
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "surface_area_mm2": float(mesh.surface_area_mm2),
        "bounds_xyz_mm": {
            "min": [float(v) for v in bounds_min],
            "max": [float(v) for v in bounds_max],
        },
        "metrics": metrics.to_dict(),
        "note": (
            "SDF smoothing engineering baseline only. Component-count preservation and vertex-nearest "
            "surface distances are engineering checks, not clinical or model-performance metrics."
        ),
    }
    summary_output = output.with_suffix(".json") if summary_path is None else _resolve(summary_path)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Export physical-space SDF-smoothed surface from NIfTI mask")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--sigma-mm", type=float, default=0.4)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument(
        "--allow-component-change",
        action="store_true",
        help="默认连通域数量变化会拒绝导出；仅研究性参数扫描时显式允许",
    )
    args = parser.parse_args()
    result = export_nifti_sdf_surface(
        args.input,
        args.output,
        class_id=args.class_id,
        sigma_mm=args.sigma_mm,
        summary_path=args.summary,
        require_component_preservation=not args.allow_component_change,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
