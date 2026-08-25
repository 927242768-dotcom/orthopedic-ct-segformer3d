from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.nifti_pipeline import process_nifti_case
from src.preprocessing.qc_visualization import generate_case_qc, generate_qc_batch


def _make_processed_case(tmp_path: Path, case_id: str = "case_001") -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    image_array = np.linspace(-1000.0, 1800.0, 12 * 14 * 16, dtype=np.float32).reshape(
        12, 14, 16
    )
    label_array = np.zeros_like(image_array, dtype=np.int16)
    label_array[3:10, 4:12, 5:14] = 7

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)

    image_path = raw_dir / f"{case_id}.nii.gz"
    label_path = raw_dir / f"{case_id}_seg.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))

    case_dir = tmp_path / "processed" / case_id
    process_nifti_case(
        image_path,
        case_dir,
        label_path=label_path,
        bone_window=(500.0, 2000.0),
    )
    return case_dir


def test_generate_case_qc_creates_contact_sheet(tmp_path: Path) -> None:
    case_dir = _make_processed_case(tmp_path)

    result = generate_case_qc(case_dir)

    output = Path(result["qc_image"])
    assert output.exists()
    assert output.stat().st_size > 1000
    assert result["label_values"] == [0, 7]
    assert result["has_bone_window"] is True
    assert set(result["review_indices"]) == {"axial", "coronal", "sagittal"}


def test_generate_qc_batch_creates_manual_review_template(tmp_path: Path) -> None:
    _make_processed_case(tmp_path, "case_001")
    _make_processed_case(tmp_path, "case_002")
    root = tmp_path / "processed"

    summary = generate_qc_batch(root, limit=2)

    assert summary["generated_count"] == 2
    assert summary["failure_count"] == 0
    assert (root / "manual_qc_review.csv").exists()
    assert (root / "qc_visualization_summary.json").exists()
    assert (root / "case_001" / "qc_contact_sheet.png").exists()
    assert (root / "case_002" / "qc_contact_sheet.png").exists()
