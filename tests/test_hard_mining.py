import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.modeling.dataset import _random_crop_3d
from src.modeling.generate_hard_mining_guidance import (
    generate_candidate_centers,
    generate_guidance,
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


def test_preferred_sampling_miss_keeps_control_fallback_crop_identical() -> None:
    image = np.arange(24 * 24 * 24, dtype=np.float32).reshape(1, 24, 24, 24)
    label = _label()
    preferred = np.zeros_like(label, dtype=bool)
    preferred[2, 2, 2] = True

    control_image, control_label = _random_crop_3d(
        image,
        label,
        (8, 8, 8),
        foreground_probability=0.0,
        rng=random.Random(7),
        force_foreground=False,
    )
    hard_image, hard_label = _random_crop_3d(
        image,
        label,
        (8, 8, 8),
        foreground_probability=0.0,
        rng=random.Random(7),
        force_foreground=False,
        preferred_mask=preferred,
        preferred_probability=0.25,
        preferred_respects_foreground_branch=True,
    )

    # fixed_per_case 背景 slot 会 force_foreground=False；此时
    # random.Random(7).random() > 0.25，因此本次不会命中 preferred center。
    # 此时 hard_sampling 必须严格退化为与 control 完全相同的随机背景 crop。
    assert np.array_equal(hard_image, control_image)
    assert np.array_equal(hard_label, control_label)


def test_false_positive_selection_only_uses_background_and_moves_to_fp_center() -> None:
    records = [
        {
            "branch": "background",
            "center_dhw": [2, 2, 2],
            "false_positive_center_dhw": [3, 4, 5],
            "false_positive_score": 0.10,
            "false_positive_confidence": 0.80,
        },
        {
            "branch": "background",
            "center_dhw": [6, 6, 6],
            "false_positive_center_dhw": [7, 8, 9],
            "false_positive_score": 0.30,
            "false_positive_confidence": 0.75,
        },
    ]

    selected = select_top_centers(records, strategy="false_positive", top_percent=50.0)

    assert len(selected) == 1
    assert selected[0]["branch"] == "background"
    assert selected[0]["candidate_center_dhw"] == [6, 6, 6]
    assert selected[0]["center_dhw"] == [7, 8, 9]


def test_false_positive_candidate_generation_can_skip_foreground_branch() -> None:
    records = generate_candidate_centers(
        _label(),
        candidate_per_branch=5,
        seed=42,
        case_id="case-a",
        branches=("background",),
    )

    assert len(records) == 5
    assert all(item["branch"] == "background" for item in records)
    assert all(_label()[tuple(item["center_dhw"])] == 0 for item in records)


@pytest.mark.parametrize("forbidden_case", ["case-val", "case-test"])
def test_guidance_rejects_validation_and_test_cases_before_inference(
    tmp_path: Path,
    forbidden_case: str,
) -> None:
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "train": ["case-train"],
                "validation": ["case-val"],
                "test": ["case-test"],
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "data": {
                    "processed_root": str(tmp_path / "processed"),
                    "split_file": str(split_path),
                    "input_channels": ["ct_normalized_u16"],
                    "roi_size_dhw": [8, 8, 8],
                    "label_mode": "binary",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不属于 train split"):
        generate_guidance(
            config_path=config_path,
            checkpoint_path=tmp_path / "unused.pt",
            output_root=tmp_path / "guidance",
            strategies=["false_positive"],
            candidate_per_branch=2,
            top_percent=50.0,
            case_id=forbidden_case,
        )
