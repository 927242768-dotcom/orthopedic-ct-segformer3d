import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from src.modeling.preflight import run_preflight


def _write_case(root: Path, case_id: str, label_value: int = 7) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    image = np.zeros((12, 10, 8), dtype=np.float32)
    label = np.zeros((12, 10, 8), dtype=np.int16)
    label[2:10, 2:8, 2:6] = label_value
    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "image_normalized.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "label.nii.gz"))
    metadata = {
        "pipeline_version": "0.3.0",
        "label": {"label_values_after": [0, label_value]},
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_config(tmp_path: Path, processed: Path, split_file: Path, *, label_mode: str = "binary", num_classes: int = 2) -> Path:
    config = {
        "data": {
            "processed_root": str(processed),
            "split_file": str(split_file),
            "input_channels": ["ct_normalized"],
            "label_mode": label_mode,
            "target_spacing_xyz_mm": [1.0, 1.0, 1.0],
            "num_classes": num_classes,
        },
        "model": {"in_channels": 1, "num_classes": num_classes},
    }
    path = tmp_path / f"config_{label_mode}_{num_classes}.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_split(path: Path, *, train: list[str], validation: list[str], test: list[str], formal_experiment: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "_meta": {"formal_experiment": formal_experiment},
                "train": train,
                "validation": validation,
                "test": test,
            }
        ),
        encoding="utf-8",
    )


def test_engineering_preflight_accepts_binary_raw_multiclass_labels(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for case_id in ("case_a", "case_b", "case_c"):
        _write_case(processed, case_id, label_value=7)
    split = tmp_path / "split.json"
    _write_split(split, train=["case_a"], validation=["case_b"], test=["case_c"])
    config = _write_config(tmp_path, processed, split, label_mode="binary", num_classes=2)

    report = run_preflight(config, mode="engineering")

    assert report.ready is True
    assert report.checked_case_count == 3
    assert report.label_values_union == [0, 7]
    assert not [issue for issue in report.issues if issue.severity == "error"]


def test_formal_preflight_blocks_engineering_split_human_qc_and_missing_gpu(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for case_id in ("case_a", "case_b"):
        _write_case(processed, case_id)
    split = tmp_path / "split.json"
    _write_split(split, train=["case_a"], validation=["case_b"], test=[])
    config = _write_config(tmp_path, processed, split)

    report = run_preflight(config, mode="formal")
    codes = {issue.code for issue in report.issues if issue.severity == "error"}

    assert report.ready is False
    assert "engineering_split_for_formal_run" in codes
    assert "human_qc_missing" in codes
    assert "gpu_unavailable" in codes


def test_preflight_rejects_private_test_in_training(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for case_id in ("case_train", "case_val", "case_private"):
        _write_case(processed, case_id)
    manifest = [
        {"case_id": "case_train", "source_split": "trainset"},
        {"case_id": "case_val", "source_split": "trainset"},
        {"case_id": "case_private", "source_split": "test_private"},
    ]
    (processed / "ctspine1k_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    split = tmp_path / "split.json"
    _write_split(
        split,
        train=["case_private"],
        validation=["case_val"],
        test=["case_train"],
    )
    config = _write_config(tmp_path, processed, split)

    report = run_preflight(config, mode="engineering")

    assert report.ready is False
    assert any(
        issue.code == "private_test_leakage" and issue.case_id == "case_private"
        for issue in report.issues
    )


def test_preflight_rejects_multiclass_label_out_of_model_range(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for case_id in ("case_a", "case_b"):
        _write_case(processed, case_id, label_value=7)
    split = tmp_path / "split.json"
    _write_split(split, train=["case_a"], validation=["case_b"], test=[])
    config = _write_config(tmp_path, processed, split, label_mode="multiclass", num_classes=2)

    report = run_preflight(config, mode="engineering")

    assert report.ready is False
    assert any(issue.code == "multiclass_out_of_range" for issue in report.issues)


def test_formal_preflight_accepts_signed_human_qc_when_gpu_not_required(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for case_id in ("case_a", "case_b"):
        _write_case(processed, case_id)
    split = tmp_path / "split.json"
    _write_split(split, train=["case_a"], validation=["case_b"], test=[], formal_experiment=True)
    config = _write_config(tmp_path, processed, split)

    with (processed / "manual_qc_review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "case_id",
            "orientation_ok",
            "spacing_ok",
            "label_alignment_ok",
            "bone_window_ok",
            "review_status",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case_id in ("case_a", "case_b"):
            writer.writerow(
                {
                    "case_id": case_id,
                    "orientation_ok": "yes",
                    "spacing_ok": "yes",
                    "label_alignment_ok": "yes",
                    "bone_window_ok": "yes",
                    "review_status": "pass",
                }
            )

    report = run_preflight(config, mode="formal", require_gpu=False)

    assert report.ready is True
