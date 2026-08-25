from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.preprocessing.prepare_ctspine1k import (
    discover_ctspine1k_cases,
    parse_official_split,
    prepare_ctspine1k_dataset,
)


def _write_case(root: Path, case_name: str, label_value: int = 5) -> tuple[Path, Path]:
    volume_dir = root / "raw_data" / "volumes" / "MSD-T10"
    label_dir = root / "raw_data" / "labels" / "MSD-T10"
    volume_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    array = np.linspace(-1200.0, 1600.0, 8 * 10 * 12, dtype=np.float32).reshape(8, 10, 12)
    label_array = np.zeros((8, 10, 12), dtype=np.int16)
    label_array[2:7, 3:9, 4:11] = label_value

    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 0.9, 1.8))
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)

    volume_path = volume_dir / f"{case_name}.nii.gz"
    label_path = label_dir / f"{case_name}_seg.nii.gz"
    sitk.WriteImage(image, str(volume_path))
    sitk.WriteImage(label, str(label_path))
    return volume_path, label_path


def test_parse_official_split_and_discover_cases(tmp_path: Path) -> None:
    source = tmp_path / "ctspine1k"
    _write_case(source, "liver_0")
    _write_case(source, "liver_169")
    split_file = source / "data_split.txt"
    split_file.write_text(
        "ignored-prefix:\n123\n\ntrainset:\nliver_0.nii.gz\n\n"
        "test_public:\nliver_169.nii.gz\n\ntest_private:\nother.nii.gz\n",
        encoding="utf-8",
    )

    mapping = parse_official_split(split_file)
    cases = discover_ctspine1k_cases(source, split_file=split_file)

    assert mapping["liver_0.nii.gz"] == "trainset"
    assert mapping["liver_169.nii.gz"] == "test_public"
    assert [case.source_split for case in cases] == ["trainset", "test_public"]
    assert all(case.sub_dataset == "MSD-T10" for case in cases)


def test_discover_small_sample_layout_matches_volume_and_label(tmp_path: Path) -> None:
    source = tmp_path / "ctspine1k"
    volume_dir = source / "MSD-T10" / "volumes"
    label_dir = source / "MSD-T10" / "labels"
    volume_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    array = np.zeros((6, 7, 8), dtype=np.float32)
    label_array = np.zeros_like(array, dtype=np.int16)
    label_array[1:5, 2:6, 2:7] = 1
    image = sitk.GetImageFromArray(array)
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image)
    sitk.WriteImage(image, str(volume_dir / "liver_169.nii.gz"))
    sitk.WriteImage(label, str(label_dir / "liver_169_seg.nii.gz"))

    cases = discover_ctspine1k_cases(source)
    assert len(cases) == 1
    assert cases[0].case_id == "ctspine1k-msd-t10-liver_169"
    assert cases[0].sub_dataset == "MSD-T10"


def test_prepare_ctspine1k_outputs_standardized_case_and_qc(tmp_path: Path) -> None:
    source = tmp_path / "ctspine1k"
    _write_case(source, "liver_169", label_value=9)
    split_file = source / "data_split.txt"
    split_file.write_text(
        "trainset:\nliver_169.nii.gz\n\ntest_public:\nfoo.nii.gz\n\ntest_private:\nbar.nii.gz\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed"

    summary = prepare_ctspine1k_dataset(
        source,
        output,
        split_file=split_file,
        limit=1,
        generate_qc=True,
    )

    assert summary["processed_count"] == 1
    assert summary["qc_generated_count"] == 1
    assert summary["failure_count"] == 0
    assert summary["source_split_counts"] == {"trainset": 1}

    case_dir = output / "ctspine1k-msd-t10-liver_169"
    assert (case_dir / "image_normalized.nii.gz").exists()
    assert (case_dir / "image_bone_window.nii.gz").exists()
    assert (case_dir / "label.nii.gz").exists()
    assert (case_dir / "qc_contact_sheet.png").exists()
    assert (output / "ctspine1k_manifest.json").exists()
    assert (output / "batch_qc_summary.json").exists()
