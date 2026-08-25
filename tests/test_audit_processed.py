from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.audit_processed import audit_processed_root, write_audit_outputs
from src.preprocessing.nifti_pipeline import process_nifti_case


def test_processed_audit_passes_valid_v03_case_and_writes_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    image_array = np.linspace(-1000.0, 1600.0, 8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    label_array = np.zeros((8, 9, 10), dtype=np.int16)
    label_array[2:7, 3:8, 4:9] = 4

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)
    image_path = raw / "ct.nii.gz"
    label_path = raw / "seg.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))

    processed_root = tmp_path / "processed"
    process_nifti_case(
        image_path,
        processed_root / "case_001",
        label_path=label_path,
        bone_window=(500.0, 2000.0),
    )

    summary = audit_processed_root(processed_root)
    assert summary["all_pass"] is True
    assert summary["status_counts"] == {"pass": 1}
    assert summary["pipeline_versions"] == {"0.3.0": 1}
    assert summary["cases"][0]["label_values"] == [0, 4]

    json_path, csv_path = write_audit_outputs(summary, tmp_path / "audit")
    assert json_path.exists()
    assert csv_path.exists()
