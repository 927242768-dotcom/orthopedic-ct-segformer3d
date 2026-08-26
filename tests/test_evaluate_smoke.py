import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml

from src.modeling.evaluate import _multiclass_case_rows, evaluate_checkpoint
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d


def _config(processed_root: Path, split_file: Path) -> dict:
    return {
        "seed": 42,
        "data": {
            "processed_root": str(processed_root),
            "split_file": str(split_file),
            "input_channels": ["ct_normalized"],
            "label_mode": "binary",
            "roi_size_dhw": [36, 36, 36],
        },
        "model": {
            "in_channels": 1,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [8, 16, 32, 64],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [2, 2, 2, 2],
            "num_heads": [1, 1, 2, 4],
            "depths": [1, 1, 1, 1],
            "decoder_head_embedding_dim": 32,
            "num_classes": 2,
            "decoder_dropout": 0.0,
        },
        "inference": {
            "roi_size_dhw": [36, 36, 36],
            "sw_batch_size": 1,
            "overlap": 0.0,
            "calibration": {
                "enabled": True,
                "n_bins": 10,
                "metric_max_samples": 10_000,
            },
        },
        "logging": {
            "save_predictions": True,
            "save_uncertainty": True,
        },
    }


def _write_case(root: Path) -> None:
    case_dir = root / "case_eval"
    case_dir.mkdir(parents=True)
    image = np.zeros((36, 36, 36), dtype=np.float32)
    image[7:29, 7:29, 7:29] = 0.8
    label = np.zeros((36, 36, 36), dtype=np.int16)
    label[12:24, 12:24, 12:24] = 1
    affine = np.diag([0.8, 0.9, 1.2, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "image_normalized.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "label.nii.gz"))


def test_evaluate_checkpoint_writes_traceable_outputs(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_case(processed)
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["case_eval"], "validation": ["case_eval"], "test": ["case_eval"]}),
        encoding="utf-8",
    )

    config = _config(processed, split_file)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    model = build_orthopedic_segformer3d(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    output = evaluate_checkpoint(
        config_path,
        checkpoint_path,
        split="test",
        output_dir=tmp_path / "evaluation",
        case_id="case_eval",
    )

    metrics_csv = output / "metrics_per_case.csv"
    summary_json = output / "summary.json"
    prediction = output / "predictions" / "case_eval" / "prediction.nii.gz"
    uncertainty = output / "uncertainty" / "case_eval" / "predictive_entropy.nii.gz"

    assert metrics_csv.exists()
    assert summary_json.exists()
    assert prediction.exists()
    assert uncertainty.exists()

    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["case_id"] == "case_eval"
    assert 0.0 <= float(rows[0]["dice"]) <= 1.0
    assert 0.0 <= float(rows[0]["prediction_foreground_fraction"]) <= 1.0
    assert 0.0 <= float(rows[0]["target_foreground_fraction"]) <= 1.0
    assert float(rows[0]["prediction_to_target_foreground_ratio"]) >= 0.0
    assert 0.0 <= float(rows[0]["uncertainty_error_rate"]) <= 1.0
    assert int(rows[0]["uncertainty_sampled_voxels"]) > 0
    if rows[0]["uncertainty_error_auroc"]:
        assert 0.0 <= float(rows[0]["uncertainty_error_auroc"]) <= 1.0
    if rows[0]["uncertainty_error_auprc"]:
        assert 0.0 <= float(rows[0]["uncertainty_error_auprc"]) <= 1.0
    assert 0.0 <= float(rows[0]["calibration_expected_calibration_error"]) <= 1.0
    assert 0.0 <= float(rows[0]["calibration_maximum_calibration_error"]) <= 1.0
    assert float(rows[0]["calibration_brier_score"]) >= 0.0
    assert float(rows[0]["calibration_negative_log_likelihood"]) >= 0.0
    assert int(rows[0]["calibration_sampled_voxels"]) > 0

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["split"] == "test"
    assert summary["case_filter"] == "case_eval"
    assert summary["metrics"]["case_count"] == 1
    assert "prediction_foreground_fraction" in summary["metrics"]
    assert "target_foreground_fraction" in summary["metrics"]
    assert "prediction_to_target_foreground_ratio" in summary["metrics"]
    assert "uncertainty_error_rate" in summary["metrics"]
    assert "calibration_expected_calibration_error" in summary["metrics"]
    assert "calibration_brier_score" in summary["metrics"]


def test_evaluate_checkpoint_rejects_case_outside_requested_split(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    _write_case(processed)
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps({"train": ["case_eval"], "validation": ["case_eval"], "test": ["case_eval"]}),
        encoding="utf-8",
    )
    config = _config(processed, split_file)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    model = build_orthopedic_segformer3d(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    with pytest.raises(ValueError, match="不属于 split"):
        evaluate_checkpoint(
            config_path,
            checkpoint_path,
            split="validation",
            output_dir=tmp_path / "evaluation_invalid",
            case_id="case_other",
        )


def test_multiclass_macro_excludes_classes_absent_from_both_prediction_and_target() -> None:
    target = np.zeros((8, 8, 8), dtype=np.int16)
    pred = np.zeros_like(target)
    target[1:4, 1:4, 1:4] = 2
    pred[1:4, 1:4, 1:4] = 2
    # class 1/3/4 都不存在，不能因为“双方都空”给宏平均额外贡献 1 分。
    macro, rows = _multiclass_case_rows(
        "case_multi",
        pred,
        target,
        (1.0, 1.0, 1.0),
        num_classes=5,
    )

    assert macro["dice"] == 1.0
    assert [row["class_id"] for row in rows] == [2]


def test_multiclass_macro_includes_false_positive_class_and_penalizes_it() -> None:
    target = np.zeros((8, 8, 8), dtype=np.int16)
    pred = np.zeros_like(target)
    target[1:4, 1:4, 1:4] = 2
    pred[1:4, 1:4, 1:4] = 2
    pred[5:7, 5:7, 5:7] = 3

    macro, rows = _multiclass_case_rows(
        "case_multi",
        pred,
        target,
        (1.0, 1.0, 1.0),
        num_classes=5,
    )

    assert [row["class_id"] for row in rows] == [2, 3]
    assert macro["dice"] < 1.0
    class3 = next(row for row in rows if row["class_id"] == 3)
    assert class3["target_present"] is False
    assert class3["pred_present"] is True
    assert class3["dice"] == 0.0
