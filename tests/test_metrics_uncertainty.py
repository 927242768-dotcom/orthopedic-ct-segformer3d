import numpy as np
import torch

from src.modeling.metrics import compute_binary_metrics, compute_structural_metrics
from src.modeling.uncertainty import (
    UncertaintyROIConfig,
    predictive_entropy,
    segmentation_calibration_metrics,
    select_uncertain_voxels,
    uncertainty_error_metrics,
    uncertainty_error_overlap,
)


def test_identical_binary_masks_have_perfect_overlap_and_zero_surface_error() -> None:
    mask = np.zeros((12, 10, 8), dtype=bool)
    mask[2:10, 2:8, 2:6] = True
    metrics = compute_binary_metrics(mask, mask, spacing_dhw_mm=(1.0, 1.0, 1.0))
    assert metrics.dice == 1.0
    assert metrics.iou == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.hd95_mm == 0.0
    assert metrics.assd_mm == 0.0


def test_structural_metrics_detect_false_merge_and_false_break() -> None:
    target = np.zeros((12, 12, 12), dtype=bool)
    target[2:5, 2:5, 2:5] = True
    target[7:10, 7:10, 7:10] = True

    merged = target.copy()
    for i in range(4, 8):
        merged[i, i, i] = True
    merge_metrics = compute_structural_metrics(merged, target, connectivity=3)
    assert merge_metrics.target_components == 2
    assert merge_metrics.pred_components == 1
    assert merge_metrics.false_merge_count == 1

    one_target = np.zeros((12, 12, 12), dtype=bool)
    one_target[2:10, 4:8, 4:8] = True
    broken = one_target.copy()
    broken[5:7] = False
    break_metrics = compute_structural_metrics(broken, one_target, connectivity=3)
    assert break_metrics.target_components == 1
    assert break_metrics.pred_components == 2
    assert break_metrics.false_break_count == 1


def test_entropy_is_highest_near_ambiguous_binary_probability() -> None:
    logits = torch.tensor([[[[[0.0, 10.0, -10.0]]]]])
    uncertainty = predictive_entropy(logits)
    assert uncertainty.shape == (1, 1, 1, 1, 3)
    assert 0.99 <= float(uncertainty[0, 0, 0, 0, 0]) <= 1.0
    assert float(uncertainty[0, 0, 0, 0, 1]) < 0.01
    assert float(uncertainty[0, 0, 0, 0, 2]) < 0.01


def test_uncertainty_error_metrics_reward_correct_error_ranking() -> None:
    target = np.zeros((4, 4, 4), dtype=np.int16)
    pred = target.copy()
    pred[0, 0, :4] = 1
    uncertainty = np.full(target.shape, 0.05, dtype=np.float32)
    uncertainty[0, 0, :4] = 0.95

    metrics = uncertainty_error_metrics(
        uncertainty,
        pred,
        target,
        top_percent=10.0,
        max_samples=10_000,
    )

    assert metrics.error_rate == 4 / 64
    assert metrics.error_auroc == 1.0
    assert metrics.error_auprc == 1.0
    assert metrics.mean_uncertainty_error is not None
    assert metrics.mean_uncertainty_correct is not None
    assert metrics.mean_uncertainty_error > metrics.mean_uncertainty_correct
    assert metrics.top_uncertainty_error_recall == 1.0


def test_uncertainty_error_metrics_returns_none_auc_without_both_classes() -> None:
    target = np.zeros((4, 4, 4), dtype=np.int16)
    pred = target.copy()
    uncertainty = np.linspace(0.0, 1.0, target.size, dtype=np.float32).reshape(target.shape)

    metrics = uncertainty_error_metrics(uncertainty, pred, target)

    assert metrics.error_rate == 0.0
    assert metrics.error_auroc is None
    assert metrics.error_auprc is None
    assert metrics.mean_uncertainty_error is None
    assert metrics.mean_uncertainty_correct is not None


def test_uncertainty_error_metrics_sampling_is_deterministic() -> None:
    target = np.zeros((50, 50, 10), dtype=np.int16)
    pred = target.copy()
    pred.reshape(-1)[::11] = 1
    uncertainty = np.linspace(0.0, 1.0, target.size, dtype=np.float32).reshape(target.shape)

    first = uncertainty_error_metrics(
        uncertainty,
        pred,
        target,
        max_samples=1000,
        seed=123,
    )
    second = uncertainty_error_metrics(
        uncertainty,
        pred,
        target,
        max_samples=1000,
        seed=123,
    )

    assert first.to_dict() == second.to_dict()
    assert first.sampled_voxels == 1000
    assert first.total_voxels == target.size


def test_calibration_metrics_reward_confident_correct_predictions() -> None:
    target = torch.tensor([[[[0, 1, 0, 1]]]], dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [[[10.0, -10.0, 10.0, -10.0]]],
                [[[-10.0, 10.0, -10.0, 10.0]]],
            ]
        ],
        dtype=torch.float32,
    )

    metrics = segmentation_calibration_metrics(logits, target, n_bins=10)

    assert metrics.accuracy == 1.0
    assert metrics.mean_confidence > 0.999
    assert metrics.expected_calibration_error < 0.001
    assert metrics.maximum_calibration_error < 0.001
    assert metrics.brier_score < 1e-6
    assert metrics.negative_log_likelihood < 0.001
    assert abs(metrics.confidence_gap) < 0.001


def test_calibration_metrics_penalize_overconfident_errors() -> None:
    target = torch.zeros((1, 1, 1, 4), dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [[[-10.0, -10.0, -10.0, -10.0]]],
                [[[10.0, 10.0, 10.0, 10.0]]],
            ]
        ],
        dtype=torch.float32,
    )

    metrics = segmentation_calibration_metrics(logits, target, n_bins=10)

    assert metrics.accuracy == 0.0
    assert metrics.mean_confidence > 0.999
    assert metrics.expected_calibration_error > 0.999
    assert metrics.maximum_calibration_error > 0.999
    assert metrics.brier_score > 1.99
    assert metrics.negative_log_likelihood > 10.0
    assert metrics.confidence_gap > 0.999


def test_calibration_metrics_sampling_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((1, 3, 12, 12, 12), generator=generator)
    target = torch.randint(0, 3, (1, 12, 12, 12), generator=generator)

    first = segmentation_calibration_metrics(
        logits,
        target,
        max_samples=400,
        seed=123,
    )
    second = segmentation_calibration_metrics(
        logits,
        target,
        max_samples=400,
        seed=123,
    )

    assert first.to_dict() == second.to_dict()
    assert first.sampled_voxels == 400
    assert first.total_voxels == 12**3


def test_uncertainty_roi_and_error_overlap_are_well_formed() -> None:
    uncertainty = torch.zeros((1, 1, 6, 6, 6), dtype=torch.float32)
    uncertainty[:, :, 2:4, 2:4, 2:4] = 1.0
    roi = select_uncertain_voxels(
        uncertainty,
        UncertaintyROIConfig(top_percent=5.0, dilation_iterations=0, min_voxels=4),
    )
    assert roi.dtype == torch.bool
    assert roi.shape == uncertainty.shape
    assert int(roi.sum()) >= 4

    pred = torch.zeros((1, 6, 6, 6), dtype=torch.long)
    target = pred.clone()
    target[:, 2:4, 2:4, 2:4] = 1
    stats = uncertainty_error_overlap(roi, pred, target)
    assert 0.0 <= stats["error_recall"] <= 1.0
    assert 0.0 <= stats["roi_error_rate"] <= 1.0
    assert 0.0 < stats["roi_fraction"] <= 1.0
