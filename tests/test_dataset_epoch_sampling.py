import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src.modeling.dataset import ProcessedOrthopedicCTDataset


def _write_sampling_case(root: Path, case_id: str) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)

    shape_xyz = (48, 50, 52)
    image = np.arange(np.prod(shape_xyz), dtype=np.float32).reshape(shape_xyz)
    image /= float(image.max())
    label = np.zeros(shape_xyz, dtype=np.int16)

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "image_normalized.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "label.nii.gz"))


def _fixed_no_aug_config() -> dict:
    return {
        "enabled": True,
        "geometric": {
            "random_flip": False,
            "random_rotate_deg": 0.0,
            "random_scale_range": [1.0, 1.0],
            "transform_probability": 0.0,
        },
        "intensity": {
            "probability": 0.0,
            "gamma_range": [1.0, 1.0],
            "gaussian_noise_std_range": [0.0, 0.0],
            "hu_shift_range": [0.0, 0.0],
        },
        "hard_sampling": {"enabled": False},
    }


def _build_dataset(tmp_path: Path) -> ProcessedOrthopedicCTDataset:
    processed_root = tmp_path / "processed"
    _write_sampling_case(processed_root, "case_train")
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps(
            {
                "train": ["case_train"],
                "validation": ["case_train"],
                "test": ["case_train"],
            }
        ),
        encoding="utf-8",
    )
    return ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "train",
        roi_size_dhw=(36, 36, 36),
        training=True,
        foreground_probability=0.0,
        label_mode="binary",
        augmentation=_fixed_no_aug_config(),
        seed=123,
    )


def test_epoch_sampling_is_reproducible_within_epoch_and_changes_across_epochs(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset(tmp_path)

    dataset.set_epoch(3)
    epoch3_first = dataset[0]["image"].clone()
    epoch3_second = dataset[0]["image"].clone()
    assert torch.equal(epoch3_first, epoch3_second)

    dataset.set_epoch(4)
    epoch4 = dataset[0]["image"]
    assert not torch.equal(epoch3_first, epoch4)


def test_multiple_patches_per_case_expand_epoch_and_use_distinct_random_streams(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset(tmp_path)
    dataset.patches_per_case = 3
    dataset.set_epoch(2)

    assert len(dataset) == 3
    samples = [dataset[index] for index in range(3)]
    assert [sample["case_id"] for sample in samples] == ["case_train"] * 3
    images = [sample["image"] for sample in samples]
    assert any(not torch.equal(images[0], image) for image in images[1:])


def test_set_epoch_rejects_negative_values(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="epoch 不能为负数"):
        dataset.set_epoch(-1)


def test_patches_per_case_rejects_non_positive_values(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="patches_per_case 必须 >= 1"):
        ProcessedOrthopedicCTDataset(
            dataset.processed_root,
            dataset.split_file,
            "train",
            roi_size_dhw=(36, 36, 36),
            training=True,
            foreground_probability=0.0,
            patches_per_case=0,
            label_mode="binary",
            augmentation=_fixed_no_aug_config(),
            seed=123,
        )
