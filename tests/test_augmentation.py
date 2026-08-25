import random

import numpy as np

from src.modeling.dataset import (
    _augment_geometry,
    _augment_intensity,
    _boundary_mask,
)


def _sample_volume() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((2, 24, 24, 24), dtype=np.float32)
    image[0] = -1.2  # ct_normalized: z-score 域允许负值
    image[0, 5:19, 6:18, 7:17] = 0.7
    image[1, 5:19, 6:18, 7:17] = 0.9
    label = np.zeros((24, 24, 24), dtype=np.int16)
    label[7:17, 8:16, 9:15] = 1
    return image, label


def test_boundary_proxy_contains_only_foreground_edge() -> None:
    _, label = _sample_volume()
    boundary = _boundary_mask(label)

    assert boundary.shape == label.shape
    assert np.any(boundary)
    assert np.all(label[boundary] == 1)
    assert int(boundary.sum()) < int((label > 0).sum())


def test_geometry_augmentation_preserves_shape_and_label_classes() -> None:
    image, label = _sample_volume()
    augmented_image, augmented_label = _augment_geometry(
        image,
        label,
        rng=random.Random(7),
        rotate_deg=10.0,
        scale_range=(0.9, 1.1),
        probability=1.0,
    )

    assert augmented_image.shape == image.shape
    assert augmented_label.shape == label.shape
    assert set(np.unique(augmented_label)).issubset({0, 1})
    assert np.isfinite(augmented_image).all()


def test_intensity_augmentation_is_bounded_and_channel_aware() -> None:
    image, _ = _sample_volume()
    augmented = _augment_intensity(
        image,
        rng=random.Random(11),
        input_channels=("ct_normalized", "bone_window"),
        gamma_range=(0.9, 1.1),
        gaussian_noise_std_range=(0.01, 0.02),
        hu_shift_range=(50.0, 50.0),
        hu_clip=(-1000.0, 2000.0),
        bone_window_width=2000.0,
        ct_zscore_mean_hu=-250.0,
        ct_zscore_std_hu=550.0,
        probability=1.0,
    )

    assert augmented.shape == image.shape
    assert np.isfinite(augmented).all()
    assert float(augmented[0].min()) < 0.0  # z-score 负值不能被错误裁掉
    assert float(augmented[1].min()) >= 0.0
    assert float(augmented[1].max()) <= 1.0
    assert not np.allclose(augmented, image)


def test_ct_physical_intensity_augmentation_requires_normalization_metadata() -> None:
    image, _ = _sample_volume()
    try:
        _augment_intensity(
            image,
            rng=random.Random(5),
            input_channels=("ct_normalized", "bone_window"),
            gamma_range=(0.9, 1.1),
            hu_shift_range=(20.0, 20.0),
            probability=1.0,
        )
    except ValueError as exc:
        assert "preprocessing 0.3+ metadata" in str(exc)
    else:
        raise AssertionError("缺少 z-score HU 参数时不应静默执行物理强度增强")
