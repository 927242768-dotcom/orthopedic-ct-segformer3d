import random

import numpy as np

from src.modeling.dataset import _random_crop_3d
from src.modeling.generate_hard_mining_guidance import (
    generate_candidate_centers,
    select_top_centers,
)


def _label() -> np.ndarray:
    label = np.zeros((24, 24, 24), dtype=np.int64)
    label[8:16, 8:16, 8:16] = 1
    return label


def test_candidate_centers_are_deterministic_and_keep_both_branches() -> None:
    label = _label()
    first = generate_candidate_centers(
        label,
        candidate_per_branch=4,
        seed=42,
        case_id="case-a",
    )
    second = generate_candidate_centers(
        label,
        candidate_per_branch=4,
        seed=42,
        case_id="case-a",
    )

    assert first == second
    assert [item["branch"] for item in first].count("foreground") == 4
    assert [item["branch"] for item in first].count("background") == 4
    for item in first:
        center = tuple(item["center_dhw"])
        if item["branch"] == "foreground":
            assert label[center] > 0
        else:
            assert label[center] == 0


def test_select_top_centers_ranks_each_branch_independently() -> None:
    records = [
        {"branch": "foreground", "center_dhw": [1, 1, 1], "loss_score": 1.0, "uncertainty_score": 0.1},
        {"branch": "foreground", "center_dhw": [2, 2, 2], "loss_score": 3.0, "uncertainty_score": 0.4},
        {"branch": "background", "center_dhw": [3, 3, 3], "loss_score": 2.0, "uncertainty_score": 0.2},
        {"branch": "background", "center_dhw": [4, 4, 4], "loss_score": 4.0, "uncertainty_score": 0.3},
    ]

    selected = select_top_centers(records, strategy="high_loss", top_percent=50.0)
    assert {tuple(item["center_dhw"]) for item in selected} == {(2, 2, 2), (4, 4, 4)}


def test_model_guidance_respects_original_foreground_branch() -> None:
    image = np.zeros((1, 24, 24, 24), dtype=np.float32)
    label = _label()
    preferred = np.zeros_like(label, dtype=bool)
    preferred[12, 12, 12] = True  # foreground hard center
    preferred[2, 2, 2] = True  # background hard center

    _, foreground_patch = _random_crop_3d(
        image,
        label,
        (8, 8, 8),
        foreground_probability=1.0,
        rng=random.Random(7),
        preferred_mask=preferred,
        preferred_probability=1.0,
        preferred_respects_foreground_branch=True,
    )
    _, background_patch = _random_crop_3d(
        image,
        label,
        (8, 8, 8),
        foreground_probability=0.0,
        rng=random.Random(7),
        preferred_mask=preferred,
        preferred_probability=1.0,
        preferred_respects_foreground_branch=True,
    )

    assert int(foreground_patch.sum()) > 0
    assert int(background_patch.sum()) == 0
