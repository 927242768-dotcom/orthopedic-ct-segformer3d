from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.prepare_verse import (
    build_official_split,
    discover_verse_cases,
    prepare_verse_dataset,
)


def _touch_pair(root: Path, split_dir: str, subject: str) -> None:
    raw = root / split_dir / "rawdata" / subject
    derivatives = root / split_dir / "derivatives" / subject
    raw.mkdir(parents=True, exist_ok=True)
    derivatives.mkdir(parents=True, exist_ok=True)
    (raw / f"{subject}_dir-orient_ct.nii.gz").touch()
    (derivatives / f"{subject}_dir-orient_seg-vert_msk.nii.gz").touch()


def test_discover_verse_cases_pairs_image_and_mask(tmp_path: Path) -> None:
    _touch_pair(tmp_path, "01_training", "sub-verse001")
    _touch_pair(tmp_path, "02_validation", "sub-verse101")
    _touch_pair(tmp_path, "03_test", "sub-verse201")

    cases = discover_verse_cases(tmp_path)
    assert len(cases) == 3
    assert [case.source_split for case in cases] == ["train", "validation", "test"]
    assert {case.patient_group for case in cases} == {
        "sub-verse001",
        "sub-verse101",
        "sub-verse201",
    }

    split = build_official_split(cases)
    assert len(split["train"]) == 1
    assert len(split["validation"]) == 1
    assert len(split["test"]) == 1
    assert split["meta"]["unknown_split_cases"] == []


def test_prepare_verse_can_generate_qc_contact_sheet(tmp_path: Path) -> None:
    raw = tmp_path / "01_training" / "rawdata" / "sub-verse001"
    derivatives = tmp_path / "01_training" / "derivatives" / "sub-verse001"
    raw.mkdir(parents=True)
    derivatives.mkdir(parents=True)

    image_array = np.linspace(-1000.0, 1800.0, 8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    label_array = np.zeros((8, 10, 12), dtype=np.int16)
    label_array[2:7, 3:9, 4:11] = 4
    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)

    image_path = raw / "sub-verse001_dir-orient_ct.nii.gz"
    label_path = derivatives / "sub-verse001_dir-orient_seg-vert_msk.nii.gz"
    sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_path))

    output = tmp_path / "processed"
    summary = prepare_verse_dataset(tmp_path, output, limit=1, generate_qc=True)

    assert summary["processed_count"] == 1
    assert summary["qc_generated_count"] == 1
    case_dir = output / "sub-verse001_dir-orient"
    assert (case_dir / "qc_contact_sheet.png").exists()
