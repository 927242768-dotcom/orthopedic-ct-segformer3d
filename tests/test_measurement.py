import math

import numpy as np
import pytest

from src.reconstruction.measurement import (
    angle_degrees,
    distance_mm,
    index_xyz_to_physical_mm,
)


def test_distance_and_angle_use_physical_mm() -> None:
    assert math.isclose(distance_mm((0, 0, 0), (3, 4, 0)), 5.0)
    assert math.isclose(angle_degrees((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0)


def test_angle_rejects_degenerate_arm() -> None:
    with pytest.raises(ValueError, match="不能重合"):
        angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0))


def test_index_to_physical_respects_spacing_origin_and_direction() -> None:
    # X/Y 方向翻转，Z 保持正向。
    physical = index_xyz_to_physical_mm(
        (2.0, 3.0, 4.0),
        spacing_xyz_mm=(0.5, 2.0, 1.5),
        origin_xyz_mm=(10.0, 20.0, 30.0),
        direction=(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert np.allclose(physical, (9.0, 14.0, 36.0))
