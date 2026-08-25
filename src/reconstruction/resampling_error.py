"""真实 label 在重采样前后的物理表面几何误差评估。

该工具衡量预处理/最近邻重采样本身带来的表面离散化变化，不衡量模型预测精度。
输出的 HD95/ASSD 是基于 Marching Cubes 顶点最近邻的工程近似，用于比较不同层厚来源，
不能直接当成临床测量误差结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from src.reconstruction.mesh import mask_to_surface_mesh
from src.sitk_compat import sitk_io_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ResamplingSurfaceMetrics:
    raw_vertex_count: int
    processed_vertex_count: int
    raw_surface_area_mm2: float
    processed_surface_area_mm2: float
    surface_area_relative_change: float
    raw_to_processed_mean_mm: float
    processed_to_raw_mean_mm: float
    assd_vertices_mm: float
    hd95_vertices_mm: float
    hdmax_vertices_mm: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _read_mask_mesh(path: Path, class_id: int | None = None):
    image = sitk.ReadImage(sitk_io_path(path))
    if image.GetDimension() != 3:
        raise ValueError(f"只支持 3D label: {path}")
    array = sitk.GetArrayFromImage(image)
    if class_id is None:
        mask = array > 0
    else:
        mask = array == int(class_id)
    if not np.any(mask):
        raise ValueError(f"{path.name}: 选定标签没有前景")
    mesh = mask_to_surface_mesh(
        mask.astype(np.uint8),
        spacing_xyz_mm=image.GetSpacing(),
        origin_xyz_mm=image.GetOrigin(),
        direction=image.GetDirection(),
        level=0.5,
    )
    return image, mesh


def compare_resampled_label_surfaces(
    raw_label_path: str | Path,
    processed_label_path: str | Path,
    *,
    class_id: int | None = None,
) -> ResamplingSurfaceMetrics:
    raw_path = _resolve(raw_label_path)
    processed_path = _resolve(processed_label_path)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not processed_path.exists():
        raise FileNotFoundError(processed_path)

    _, raw_mesh = _read_mask_mesh(raw_path, class_id=class_id)
    _, processed_mesh = _read_mask_mesh(processed_path, class_id=class_id)

    raw_vertices = raw_mesh.vertices_xyz_mm.astype(np.float64, copy=False)
    processed_vertices = processed_mesh.vertices_xyz_mm.astype(np.float64, copy=False)
    processed_tree = cKDTree(processed_vertices)
    raw_tree = cKDTree(raw_vertices)
    raw_to_processed = processed_tree.query(raw_vertices, k=1, workers=-1)[0]
    processed_to_raw = raw_tree.query(processed_vertices, k=1, workers=-1)[0]

    all_distances = np.concatenate([raw_to_processed, processed_to_raw])
    raw_area = float(raw_mesh.surface_area_mm2)
    processed_area = float(processed_mesh.surface_area_mm2)
    area_change = (
        0.0 if raw_area <= 1e-12 else float((processed_area - raw_area) / raw_area)
    )
    return ResamplingSurfaceMetrics(
        raw_vertex_count=raw_mesh.vertex_count,
        processed_vertex_count=processed_mesh.vertex_count,
        raw_surface_area_mm2=raw_area,
        processed_surface_area_mm2=processed_area,
        surface_area_relative_change=area_change,
        raw_to_processed_mean_mm=float(raw_to_processed.mean()),
        processed_to_raw_mean_mm=float(processed_to_raw.mean()),
        assd_vertices_mm=float(
            (raw_to_processed.mean() + processed_to_raw.mean()) / 2.0
        ),
        hd95_vertices_mm=float(np.percentile(all_distances, 95)),
        hdmax_vertices_mm=float(all_distances.max()),
    )


def _finite_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None, "count": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "count": int(finite.size),
    }


def evaluate_processed_manifest(
    processed_root: str | Path,
    *,
    class_id: int | None = None,
    limit: int | None = None,
    output_csv: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    root = _resolve(processed_root)
    manifest_path = root / "ctspine1k_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("ctspine1k_manifest.json 必须为 list")
    entries = payload[:limit] if limit is not None else payload
    if not entries:
        raise ValueError("没有可评估病例")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in entries:
        case_id = str(item["case_id"])
        raw_label = _resolve(str(item["label_path"]))
        processed_label = root / case_id / "label.nii.gz"
        try:
            raw_image = sitk.ReadImage(sitk_io_path(raw_label))
            metrics = compare_resampled_label_surfaces(
                raw_label,
                processed_label,
                class_id=class_id,
            )
            row = {
                "case_id": case_id,
                "source_split": str(item.get("source_split", "unknown")),
                "class_id": "foreground" if class_id is None else int(class_id),
                "raw_spacing_x_mm": float(raw_image.GetSpacing()[0]),
                "raw_spacing_y_mm": float(raw_image.GetSpacing()[1]),
                "raw_spacing_z_mm": float(raw_image.GetSpacing()[2]),
                **metrics.to_dict(),
            }
            rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if output_csv is None:
        suffix = "foreground" if class_id is None else f"class_{class_id}"
        csv_path = root / f"resampling_geometry_error_{suffix}.csv"
    else:
        csv_path = _resolve(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "source_split",
        "class_id",
        "raw_spacing_x_mm",
        "raw_spacing_y_mm",
        "raw_spacing_z_mm",
        "raw_vertex_count",
        "processed_vertex_count",
        "raw_surface_area_mm2",
        "processed_surface_area_mm2",
        "surface_area_relative_change",
        "raw_to_processed_mean_mm",
        "processed_to_raw_mean_mm",
        "assd_vertices_mm",
        "hd95_vertices_mm",
        "hdmax_vertices_mm",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = [
        "surface_area_relative_change",
        "raw_to_processed_mean_mm",
        "processed_to_raw_mean_mm",
        "assd_vertices_mm",
        "hd95_vertices_mm",
        "hdmax_vertices_mm",
    ]
    summary_metrics = {
        key: _finite_stats([float(row[key]) for row in rows]) for key in metric_keys
    }
    by_z_spacing: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{float(row['raw_spacing_z_mm']):.3f}"
        group = by_z_spacing.setdefault(key, {"case_ids": [], "rows": []})
        group["case_ids"].append(row["case_id"])
        group["rows"].append(row)
    grouped_summary: dict[str, Any] = {}
    for key, group in by_z_spacing.items():
        grouped_summary[key] = {
            "case_ids": group["case_ids"],
            "case_count": len(group["rows"]),
            "assd_vertices_mm": _finite_stats(
                [float(row["assd_vertices_mm"]) for row in group["rows"]]
            ),
            "hd95_vertices_mm": _finite_stats(
                [float(row["hd95_vertices_mm"]) for row in group["rows"]]
            ),
            "surface_area_relative_change": _finite_stats(
                [float(row["surface_area_relative_change"]) for row in group["rows"]]
            ),
        }

    summary: dict[str, Any] = {
        "evaluated_at": datetime.now().isoformat(),
        "processed_root": str(root),
        "manifest": str(manifest_path),
        "class_id": class_id,
        "case_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "metrics": summary_metrics,
        "by_raw_z_spacing_mm": grouped_summary,
        "csv": str(csv_path),
        "note": (
            "Engineering comparison of raw-vs-resampled label surfaces using nearest-neighbor "
            "Marching-Cubes vertices. This quantifies preprocessing discretization, not model accuracy "
            "or clinical measurement error."
        ),
    }
    if output_json is None:
        json_path = csv_path.with_suffix(".json")
    else:
        json_path = _resolve(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估真实 label 重采样前后物理表面几何差异")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    summary = evaluate_processed_manifest(
        args.processed_root,
        class_id=args.class_id,
        limit=args.limit,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
