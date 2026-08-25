from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from src.preprocessing.nifti_pipeline import (
    compare_image_label_geometry,
    process_nifti_case,
)


def _write_synthetic_case(tmp_path: Path) -> tuple[Path, Path]:
    image_array = np.linspace(-1200.0, 1800.0, 8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    label_array = np.zeros((8, 10, 12), dtype=np.int16)
    label_array[2:6, 3:8, 4:10] = 3

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((0.8, 0.9, 2.0))
    image.SetOrigin((10.0, 20.0, -30.0))

    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)

    image_path = tmp_path / "ct.nii.gz"
    label_path = tmp_path / "seg.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))
    return image_path, label_path


def test_process_nifti_case_outputs_training_layout(tmp_path: Path) -> None:
    image_path, label_path = _write_synthetic_case(tmp_path)
    output_dir = tmp_path / "processed" / "case_001"

    result = process_nifti_case(
        image_path,
        output_dir,
        label_path=label_path,
        target_spacing_xyz=(1.0, 1.0, 1.0),
        hu_clip=(-1000.0, 2000.0),
        bone_window=(500.0, 2000.0),
    )

    assert result["source_type"] == "nifti"
    assert result["pipeline_version"] == "0.3.0"
    assert result["qc"]["status"] == "pass"
    normalization = result["processed"]["normalization"]
    assert normalization["method"] == "clip_then_case_zscore"
    assert normalization["clipped_std_hu"] > 0.0
    assert result["label"]["label_values_before"] == [0, 3]
    assert set(result["label"]["label_values_after"]).issubset({0, 3})

    expected_files = {
        "image_normalized.nii.gz",
        "image_bone_window.nii.gz",
        "label.nii.gz",
        "metadata.json",
        "qc.json",
    }
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})

    image_out = sitk.ReadImage(str(output_dir / "image_normalized.nii.gz"))
    label_out = sitk.ReadImage(str(output_dir / "label.nii.gz"))
    assert compare_image_label_geometry(image_out, label_out)["aligned"] is True
    assert np.isclose(image_out.GetSpacing(), (1.0, 1.0, 1.0)).all()


def test_process_nifti_case_rejects_misaligned_label(tmp_path: Path) -> None:
    image_path, label_path = _write_synthetic_case(tmp_path)
    label = sitk.ReadImage(str(label_path))
    label.SetOrigin((11.0, 20.0, -30.0))
    sitk.WriteImage(label, str(label_path))

    with pytest.raises(ValueError, match="物理空间不一致"):
        process_nifti_case(
            image_path,
            tmp_path / "bad_case",
            label_path=label_path,
        )
