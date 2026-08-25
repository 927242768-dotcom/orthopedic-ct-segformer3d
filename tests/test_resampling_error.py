import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.reconstruction.resampling_error import (
    compare_resampled_label_surfaces,
    evaluate_processed_manifest,
)


def _write_label(path: Path, *, origin=(0.0, 0.0, 0.0)) -> None:
    array = np.zeros((16, 18, 20), dtype=np.int16)
    array[4:12, 5:14, 6:16] = 7
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin(tuple(float(v) for v in origin))
    sitk.WriteImage(image, str(path))


def test_surface_comparison_is_zero_for_identical_physical_label(tmp_path: Path) -> None:
    label = tmp_path / "label.nii.gz"
    _write_label(label)

    metrics = compare_resampled_label_surfaces(label, label)

    assert metrics.assd_vertices_mm == 0.0
    assert metrics.hd95_vertices_mm == 0.0
    assert metrics.hdmax_vertices_mm == 0.0
    assert metrics.surface_area_relative_change == 0.0


def test_surface_comparison_detects_physical_origin_shift(tmp_path: Path) -> None:
    raw = tmp_path / "raw.nii.gz"
    shifted = tmp_path / "shifted.nii.gz"
    _write_label(raw)
    _write_label(shifted, origin=(4.0, 0.0, 0.0))

    metrics = compare_resampled_label_surfaces(raw, shifted)

    assert metrics.assd_vertices_mm > 0.0
    assert metrics.hd95_vertices_mm > 0.0


def test_manifest_evaluation_writes_traceable_csv_and_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw_label.nii.gz"
    _write_label(raw)
    processed_root = tmp_path / "processed"
    case_dir = processed_root / "case_001"
    case_dir.mkdir(parents=True)
    processed = case_dir / "label.nii.gz"
    _write_label(processed)
    (processed_root / "ctspine1k_manifest.json").write_text(
        json.dumps(
            [
                {
                    "case_id": "case_001",
                    "source_split": "trainset",
                    "label_path": str(raw),
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = evaluate_processed_manifest(processed_root)

    assert summary["case_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["metrics"]["assd_vertices_mm"]["mean"] == 0.0
    assert (processed_root / "resampling_geometry_error_foreground.csv").exists()
    assert (processed_root / "resampling_geometry_error_foreground.json").exists()
