import json
from pathlib import Path

import nibabel as nib
import numpy as np

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.generate_residual_fp_guidance import (
    _select_evenly_spaced,
    select_residual_fp_centers,
)


def test_select_residual_fp_centers_prefers_high_confidence_and_spacing() -> None:
    residual = np.zeros((20, 20, 20), dtype=bool)
    probability = np.zeros_like(residual, dtype=np.float32)
    residual[2, 2, 2] = True
    probability[2, 2, 2] = 0.99
    residual[3, 2, 2] = True
    probability[3, 2, 2] = 0.98
    residual[15, 15, 15] = True
    probability[15, 15, 15] = 0.90

    centers = select_residual_fp_centers(
        residual,
        probability,
        max_centers=2,
        min_center_distance_voxels=5.0,
    )

    assert centers == [(2, 2, 2), (15, 15, 15)]


def test_select_residual_fp_centers_returns_empty_when_no_residual_fp() -> None:
    residual = np.zeros((8, 8, 8), dtype=bool)
    probability = np.zeros_like(residual, dtype=np.float32)
    assert select_residual_fp_centers(
        residual,
        probability,
        max_centers=2,
        min_center_distance_voxels=4.0,
    ) == []


def test_evenly_spaced_train_subset_matches_training_policy() -> None:
    case_ids = [f"case-{index:03d}" for index in range(100)]
    selected = _select_evenly_spaced(case_ids, 8)
    expected_indices = np.linspace(0, len(case_ids) - 1, num=8, dtype=int)
    assert selected == [case_ids[int(index)] for index in expected_indices]


def test_empty_residual_guidance_falls_back_to_normal_background_sampling(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    case_dir = processed / "case-a"
    case_dir.mkdir(parents=True)
    affine = np.eye(4)
    image = np.zeros((24, 24, 24), dtype=np.uint16)
    label = np.zeros((24, 24, 24), dtype=np.uint8)
    label[8:16, 8:16, 8:16] = 1
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "image_normalized_u16.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "label.nii.gz"))

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["case-a"]}), encoding="utf-8")
    guidance_case = tmp_path / "guidance" / "case-a"
    guidance_case.mkdir(parents=True)
    empty_guidance = np.zeros((24, 24, 24), dtype=np.uint8)
    nib.save(nib.Nifti1Image(empty_guidance, affine), str(guidance_case / "hard_centers.nii.gz"))

    dataset = ProcessedOrthopedicCTDataset(
        processed,
        split,
        "train",
        input_channels=["ct_normalized_u16"],
        roi_size_dhw=[8, 8, 8],
        training=True,
        foreground_probability=0.5,
        patches_per_case=2,
        foreground_sampling_mode="fixed_per_case",
        augmentation={
            "enabled": True,
            "geometric": {"random_flip": False, "transform_probability": 0.0},
            "intensity": {"probability": 0.0},
            "hard_sampling": {
                "enabled": True,
                "strategy": "residual_false_positive",
                "guidance_root": str(tmp_path / "guidance"),
                "preferred_probability": 0.25,
            },
        },
        seed=42,
    )

    # patch_slot=1 是 fixed_per_case 的 background 分支；空 residual guidance 应正常回退。
    item = dataset[1]
    assert tuple(item["image"].shape) == (1, 8, 8, 8)
    assert tuple(item["label"].shape) == (8, 8, 8)
