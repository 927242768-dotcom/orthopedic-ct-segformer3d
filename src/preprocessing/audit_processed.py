"""处理后骨科 CT 数据完整性审计。

目标：在正式训练前对每个病例执行轻量、可追溯的结构检查，避免把预处理版本混用、
几何错位、空标签或异常 spacing 悄悄带入实验。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.sitk_compat import sitk_io_path


def _geometry_signature(image: sitk.Image) -> dict[str, object]:
    return {
        "size_xyz": [int(v) for v in image.GetSize()],
        "spacing_xyz_mm": [float(v) for v in image.GetSpacing()],
        "origin_xyz_mm": [float(v) for v in image.GetOrigin()],
        "direction": [float(v) for v in image.GetDirection()],
    }


def _geometry_matches(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol)
    )


def audit_case(case_dir: str | Path) -> dict[str, object]:
    case_dir = Path(case_dir)
    required = {
        "image": case_dir / "image_normalized.nii.gz",
        "label": case_dir / "label.nii.gz",
        "metadata": case_dir / "metadata.json",
        "qc": case_dir / "qc.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return {
            "case_id": case_dir.name,
            "status": "fail",
            "errors": [f"missing:{name}" for name in missing],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    qc_payload = json.loads(required["qc"].read_text(encoding="utf-8"))

    image = sitk.ReadImage(sitk_io_path(required["image"]))
    label = sitk.ReadImage(sitk_io_path(required["label"]))
    if not _geometry_matches(image, label):
        errors.append("image_label_geometry_mismatch")

    bone_path = case_dir / "image_bone_window.nii.gz"
    bone_present = bone_path.exists()
    if bone_present:
        bone = sitk.ReadImage(sitk_io_path(bone_path))
        if not _geometry_matches(image, bone):
            errors.append("image_bone_geometry_mismatch")

    spacing = tuple(float(v) for v in image.GetSpacing())
    if any(not math.isfinite(v) or v <= 0 for v in spacing):
        errors.append("invalid_spacing")
    if max(spacing) - min(spacing) > 1e-5:
        warnings.append("anisotropic_processed_spacing")

    label_array = sitk.GetArrayFromImage(label)
    label_values = [int(v) for v in np.unique(label_array)]
    foreground_voxels = int(np.count_nonzero(label_array))
    voxel_count = int(label_array.size)
    foreground_fraction = foreground_voxels / max(voxel_count, 1)
    if foreground_voxels == 0:
        errors.append("empty_foreground_label")

    pipeline_version = str(metadata.get("pipeline_version", "unknown"))
    normalization = metadata.get("processed", {}).get("normalization", {})
    normalization_method = str(normalization.get("method", "missing"))
    if pipeline_version.startswith("0.3") and normalization_method != "clip_then_case_zscore":
        errors.append("missing_v03_normalization_metadata")
    if pipeline_version != "0.3.0":
        warnings.append(f"pipeline_version={pipeline_version}")

    qc_status = str(qc_payload.get("status", metadata.get("qc", {}).get("status", "unknown")))
    if qc_status == "fail":
        errors.append("qc_status_fail")
    elif qc_status not in {"pass", "warning"}:
        warnings.append(f"qc_status={qc_status}")

    status = "fail" if errors else ("warning" if warnings else "pass")
    return {
        "case_id": case_dir.name,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "pipeline_version": pipeline_version,
        "source_type": metadata.get("source_type"),
        "qc_status": qc_status,
        "bone_window_present": bone_present,
        "geometry": _geometry_signature(image),
        "label_values": label_values,
        "foreground_voxels": foreground_voxels,
        "foreground_fraction": foreground_fraction,
        "normalization_method": normalization_method,
        "clipped_mean_hu": normalization.get("clipped_mean_hu"),
        "clipped_std_hu": normalization.get("clipped_std_hu"),
    }


def audit_processed_root(processed_root: str | Path) -> dict[str, object]:
    processed_root = Path(processed_root)
    if not processed_root.exists() or not processed_root.is_dir():
        raise FileNotFoundError(processed_root)

    case_dirs = sorted(
        path for path in processed_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if not case_dirs:
        raise RuntimeError(f"没有发现处理后病例目录: {processed_root}")

    cases = [audit_case(case_dir) for case_dir in case_dirs]
    status_counts = Counter(str(case["status"]) for case in cases)
    pipeline_versions = Counter(str(case.get("pipeline_version", "missing")) for case in cases)
    return {
        "audited_at": datetime.now().isoformat(),
        "processed_root": str(processed_root),
        "case_count": len(cases),
        "status_counts": dict(status_counts),
        "pipeline_versions": dict(pipeline_versions),
        "all_pass": status_counts.get("fail", 0) == 0,
        "cases": cases,
    }


def write_audit_outputs(summary: dict[str, object], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "processed_data_audit.json"
    csv_path = output_dir / "processed_data_audit.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = summary["cases"]
    fieldnames = [
        "case_id",
        "status",
        "pipeline_version",
        "qc_status",
        "label_values",
        "foreground_fraction",
        "normalization_method",
        "errors",
        "warnings",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in rows:
            writer.writerow(
                {
                    "case_id": case.get("case_id"),
                    "status": case.get("status"),
                    "pipeline_version": case.get("pipeline_version"),
                    "qc_status": case.get("qc_status"),
                    "label_values": json.dumps(case.get("label_values", []), ensure_ascii=False),
                    "foreground_fraction": case.get("foreground_fraction"),
                    "normalization_method": case.get("normalization_method"),
                    "errors": ";".join(case.get("errors", [])),
                    "warnings": ";".join(case.get("warnings", [])),
                }
            )
    return json_path, csv_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Audit processed orthopedic CT cases before training")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = audit_processed_root(args.processed_root)
    if args.output_dir is not None:
        write_audit_outputs(summary, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
