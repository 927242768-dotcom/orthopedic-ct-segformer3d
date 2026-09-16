import numpy as np
import pytest

from src.modeling.postprocessing import postprocess_prediction, remove_small_components


def test_remove_small_components_noop_when_disabled() -> None:
    pred = np.zeros((8, 8, 8), dtype=np.int16)
    pred[1, 1, 1] = 1
    result = remove_small_components(pred, min_component_voxels=0)
    assert np.array_equal(result, pred)
    assert result is not pred


def test_remove_small_components_removes_small_and_keeps_large() -> None:
    pred = np.zeros((12, 12, 12), dtype=np.int16)
    pred[1:4, 1:4, 1:4] = 1  # 27 voxels
    pred[6:11, 6:11, 6:11] = 1  # 125 voxels

    result = remove_small_components(pred, min_component_voxels=64, connectivity=3)

    assert not result[1:4, 1:4, 1:4].any()
    assert np.all(result[6:11, 6:11, 6:11] == 1)


def test_remove_small_components_filters_each_class_independently() -> None:
    pred = np.zeros((12, 12, 12), dtype=np.int16)
    pred[1:5, 1:5, 1:5] = 1  # 64 voxels，保留
    pred[7:9, 7:9, 7:9] = 2  # 8 voxels，删除
    pred[6:11, 1:6, 6:11] = 2  # 125 voxels，保留

    result = postprocess_prediction(
        pred,
        {"min_component_voxels": 32, "connectivity": 3},
    )

    assert np.all(result[1:5, 1:5, 1:5] == 1)
    assert not result[7:9, 7:9, 7:9].any()
    assert np.all(result[6:11, 1:6, 6:11] == 2)


def test_remove_small_components_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="3D"):
        remove_small_components(np.zeros((8, 8), dtype=np.int16), min_component_voxels=8)
    with pytest.raises(ValueError, match=">= 0"):
        remove_small_components(np.zeros((8, 8, 8), dtype=np.int16), min_component_voxels=-1)
    with pytest.raises(ValueError, match="connectivity"):
        remove_small_components(
            np.zeros((8, 8, 8), dtype=np.int16),
            min_component_voxels=8,
            connectivity=4,
        )
