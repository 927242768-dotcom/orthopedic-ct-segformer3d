"""使用冻结 v13 coarse checkpoint 训练并验证 uncertainty ROI refinement。

科研纪律：
- refinement network 只在 train split patch 上训练；
- liver_7/liver_8 仅用于 validation 参数比较；
- validation 复用已保存 prediction + predictive entropy，不重复 full-volume coarse inference；
- 独立 test split 不在本脚本中读取、推理或评价。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from scipy import ndimage
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.metrics import compute_binary_metrics, compute_structural_metrics
from src.modeling.refinement import (
    UncertaintyRefinementNet3D,
    canonical_binary_logits_from_prediction_entropy,
)
from src.modeling.refinement_training import refinement_training_step
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import resize_logits_to_target, seed_everything
from src.modeling.uncertainty import (
    UncertaintyROIConfig,
    predictive_entropy,
    select_uncertain_voxels,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 顶层必须是对象: {path}")
    return payload


def _load_dhw_nifti(path: Path, dtype: np.dtype | type) -> np.ndarray:
    image = nib.load(str(path))
    array_xyz = np.asarray(image.dataobj).astype(dtype, copy=False)
    if array_xyz.ndim != 3:
        raise ValueError(f"只支持 3D NIfTI: {path}, shape={array_xyz.shape}")
    return np.transpose(array_xyz, (2, 1, 0))


def _save_dhw_nifti(array_dhw: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference = nib.load(str(reference_path))
    array_xyz = np.transpose(np.asarray(array_dhw), (2, 1, 0))
    header = reference.header.copy()
    header.set_data_dtype(array_xyz.dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array_xyz, reference.affine, header), str(output_path))


def _spacing_dhw(reference_path: Path) -> tuple[float, float, float]:
    zooms = nib.load(str(reference_path)).header.get_zooms()[:3]
    return float(zooms[2]), float(zooms[1]), float(zooms[0])


def _load_coarse_inference_seconds(evaluation_dir: Path) -> float:
    path = evaluation_dir / "metrics_per_case.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"期望单病例 metrics_per_case.csv，得到 {len(rows)} 行: {path}")
    return float(rows[0]["inference_seconds"])


def _build_refinement_model(
    experiment_config: dict[str, Any], device: torch.device
) -> UncertaintyRefinementNet3D:
    refinement_cfg = experiment_config["refinement"]
    return UncertaintyRefinementNet3D(
        image_channels=int(refinement_cfg["image_channels"]),
        num_classes=int(refinement_cfg["num_classes"]),
        hidden_channels=int(refinement_cfg.get("hidden_channels", 24)),
        residual_blocks=int(refinement_cfg.get("residual_blocks", 2)),
    ).to(device)


def _load_coarse_model(
    config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> torch.nn.Module:
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _load_refinement_checkpoint(
    experiment_config: dict[str, Any], checkpoint_path: Path, device: torch.device
) -> UncertaintyRefinementNet3D:
    model = _build_refinement_model(experiment_config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _build_train_dataset(coarse_config: dict[str, Any]) -> ProcessedOrthopedicCTDataset:
    data_cfg = coarse_config["data"]
    train_cfg = coarse_config["training"]
    bone_window_cfg = data_cfg.get("bone_window", {})
    return ProcessedOrthopedicCTDataset(
        _resolve(data_cfg["processed_root"]),
        _resolve(data_cfg["split_file"]),
        "train",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [64, 64, 64]),
        training=True,
        foreground_probability=float(data_cfg.get("foreground_probability", 0.25)),
        patches_per_case=int(train_cfg.get("patches_per_case", 4)),
        foreground_sampling_mode=str(data_cfg.get("foreground_sampling_mode", "bernoulli")),
        label_mode=str(data_cfg.get("label_mode", "binary")),
        augmentation=coarse_config.get("augmentation", {}),
        hu_clip=data_cfg.get("hu_clip", [-1000.0, 2000.0]),
        bone_window_width=float(bone_window_cfg.get("width", 2000.0)),
        seed=int(coarse_config.get("seed", 42)),
    )


def _train_refinement(
    *,
    experiment_config: dict[str, Any],
    coarse_config: dict[str, Any],
    coarse_model: torch.nn.Module,
    run_dir: Path,
    device: torch.device,
) -> tuple[UncertaintyRefinementNet3D, list[dict[str, float]]]:
    training_cfg = experiment_config["training"]
    model = _build_refinement_model(experiment_config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    dataset = _build_train_dataset(coarse_config)
    generator = torch.Generator().manual_seed(int(experiment_config.get("seed", 42)))
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
        drop_last=True,
        generator=generator,
    )
    roi_config = UncertaintyROIConfig(
        top_percent=float(training_cfg.get("train_roi_top_percent", 10.0)),
        dilation_iterations=int(training_cfg.get("train_roi_dilation_iterations", 1)),
        min_voxels=int(training_cfg.get("train_roi_min_voxels", 64)),
    )

    history: list[dict[str, float]] = []
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(training_cfg.get("epochs", 2))
    for epoch in range(1, epochs + 1):
        dataset.set_epoch(epoch)
        sums: dict[str, float] = {}
        steps = 0
        started = time.perf_counter()
        for batch in loader:
            image = batch["image"].to(device)
            target = batch["label"].to(device)
            with torch.no_grad():
                coarse_logits = coarse_model(image)
                coarse_logits = resize_logits_to_target(coarse_logits, tuple(target.shape[-3:]))
                uncertainty = predictive_entropy(coarse_logits)
                roi_mask = select_uncertain_voxels(uncertainty, roi_config)
            metrics = refinement_training_step(
                model,
                optimizer,
                image=image,
                coarse_logits=coarse_logits,
                target=target,
                roi_mask=roi_mask,
                detach_coarse=True,
                max_grad_norm=5.0,
            ).to_dict()
            steps += 1
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value)

        row = {"epoch": float(epoch), "steps": float(steps)}
        for key, value in sums.items():
            row[key] = value / max(steps, 1)
        row["epoch_seconds"] = float(time.perf_counter() - started)
        history.append(row)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "experiment_config": experiment_config,
            },
            checkpoint_dir / "last.pt",
        )

    history_path = run_dir / "training_history.csv"
    if history:
        with history_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    return model, history


def _read_training_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _tile_slices(shape_dhw: tuple[int, int, int], tile_dhw: tuple[int, int, int]):
    d, h, w = shape_dhw
    td, th, tw = tile_dhw
    for sd in range(0, d, td):
        for sh in range(0, h, th):
            for sw in range(0, w, tw):
                yield (
                    slice(sd, min(sd + td, d)),
                    slice(sh, min(sh + th, h)),
                    slice(sw, min(sw + tw, w)),
                )


def _candidate_key(top_percent: float, dilation: int) -> str:
    top_token = str(float(top_percent)).replace(".", "p")
    return f"top{top_token}_dil{int(dilation)}"


def _roi_tile_from_entropy(
    entropy: np.ndarray,
    spatial_slices: tuple[slice, slice, slice],
    *,
    threshold: float,
    dilation_iterations: int,
) -> np.ndarray:
    """仅对当前 tile + halo 生成 uncertainty ROI，避免保存全卷 9 份 mask。"""
    if dilation_iterations < 0:
        raise ValueError("dilation_iterations 不能为负数")
    starts = [int(value.start or 0) for value in spatial_slices]
    stops = [int(value.stop) for value in spatial_slices]
    expanded_starts = [
        max(0, start - dilation_iterations) for start in starts
    ]
    expanded_stops = [
        min(size, stop + dilation_iterations)
        for stop, size in zip(stops, entropy.shape)
    ]
    expanded = entropy[
        expanded_starts[0] : expanded_stops[0],
        expanded_starts[1] : expanded_stops[1],
        expanded_starts[2] : expanded_stops[2],
    ] >= float(threshold)
    if dilation_iterations > 0:
        expanded = ndimage.binary_dilation(
            expanded,
            structure=np.ones((3, 3, 3), dtype=bool),
            iterations=dilation_iterations,
            border_value=0,
        )
    local_starts = [start - expanded_start for start, expanded_start in zip(starts, expanded_starts)]
    local_stops = [
        local_start + (stop - start)
        for local_start, start, stop in zip(local_starts, starts, stops)
    ]
    return np.asarray(
        expanded[
            local_starts[0] : local_stops[0],
            local_starts[1] : local_stops[1],
            local_starts[2] : local_stops[2],
        ],
        dtype=bool,
    )


def _evaluate_case_grid(
    *,
    case_id: str,
    processed_root: Path,
    evaluation_dir: Path,
    refinement_model: UncertaintyRefinementNet3D,
    validation_cfg: dict[str, Any],
    tile_dhw: tuple[int, int, int],
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    case_dir = processed_root / case_id
    label_path = case_dir / "label.nii.gz"
    image = _load_dhw_nifti(case_dir / "image_normalized.nii.gz", np.float32)
    target = (_load_dhw_nifti(label_path, np.int16) > 0).astype(np.uint8)
    coarse_pred = (
        _load_dhw_nifti(
            evaluation_dir / "predictions" / case_id / "prediction.nii.gz", np.int16
        )
        > 0
    ).astype(np.uint8)
    entropy = _load_dhw_nifti(
        evaluation_dir / "uncertainty" / case_id / "predictive_entropy.nii.gz", np.float32
    )
    if image.shape != target.shape or target.shape != coarse_pred.shape or coarse_pred.shape != entropy.shape:
        raise ValueError(
            f"{case_id}: image/target/prediction/entropy shape 不一致: "
            f"{image.shape}/{target.shape}/{coarse_pred.shape}/{entropy.shape}"
        )

    top_grid = [float(v) for v in validation_cfg.get("top_percent_grid", [10.0])]
    dilation_grid = [int(v) for v in validation_cfg.get("dilation_iterations_grid", [1])]
    if not top_grid or not dilation_grid:
        raise ValueError("validation ROI grid 不能为空")
    if any(not (0.0 < value <= 100.0) for value in top_grid):
        raise ValueError("top_percent_grid 必须位于 (0,100]")
    if any(value < 0 for value in dilation_grid):
        raise ValueError("dilation_iterations_grid 不能包含负数")

    quantiles = np.asarray([1.0 - value / 100.0 for value in top_grid], dtype=np.float64)
    threshold_values = np.quantile(entropy, quantiles)
    thresholds = {top: float(threshold) for top, threshold in zip(top_grid, threshold_values)}

    candidate_states: dict[str, dict[str, Any]] = {}
    for top_percent in top_grid:
        for dilation in dilation_grid:
            key = _candidate_key(top_percent, dilation)
            candidate_states[key] = {
                "top_percent": top_percent,
                "dilation_iterations": dilation,
                "threshold": thresholds[top_percent],
                "prediction": coarse_pred.copy(),
                "roi_count": 0,
                "coarse_roi_error_count": 0,
                "refined_roi_error_count": 0,
                "outside_changed_count": 0,
                "roi_selection_seconds": 0.0,
                "canonical_reconstruction_seconds": 0.0,
                "refinement_network_seconds": 0.0,
                "processed_tiles": 0,
            }

    compare_full = bool(validation_cfg.get("compare_full_volume_second_pass", True))
    full_prediction = coarse_pred.copy() if compare_full else None
    full_reconstruction_seconds = 0.0
    full_network_seconds = 0.0
    full_tiles = 0
    reconstruction_mismatch = 0
    reconstruction_entropy_max_abs_error = 0.0

    refinement_model.eval()
    with torch.no_grad():
        for spatial_slices in _tile_slices(image.shape, tile_dhw):
            d_slice, h_slice, w_slice = spatial_slices
            tile_masks: dict[str, np.ndarray] = {}
            active_keys: list[str] = []
            for key, state in candidate_states.items():
                roi_started = time.perf_counter()
                roi_tile = _roi_tile_from_entropy(
                    entropy,
                    spatial_slices,
                    threshold=float(state["threshold"]),
                    dilation_iterations=int(state["dilation_iterations"]),
                )
                state["roi_selection_seconds"] += float(time.perf_counter() - roi_started)
                tile_masks[key] = roi_tile
                if bool(roi_tile.any()):
                    active_keys.append(key)

            if not active_keys and not compare_full:
                continue

            tile_prediction = coarse_pred[d_slice, h_slice, w_slice]
            tile_entropy = entropy[d_slice, h_slice, w_slice]
            reconstruction_started = time.perf_counter()
            tile_coarse = canonical_binary_logits_from_prediction_entropy(
                torch.from_numpy(tile_prediction), torch.from_numpy(tile_entropy)
            )
            reconstruction_seconds = float(time.perf_counter() - reconstruction_started)
            reconstructed_pred = torch.argmax(tile_coarse, dim=1)[0].numpy().astype(np.uint8)
            reconstruction_mismatch += int(np.count_nonzero(reconstructed_pred != tile_prediction))
            reconstructed_entropy = predictive_entropy(tile_coarse)[0, 0].numpy()
            reconstruction_entropy_max_abs_error = max(
                reconstruction_entropy_max_abs_error,
                float(
                    np.max(
                        np.abs(
                            reconstructed_entropy.astype(np.float64)
                            - tile_entropy.astype(np.float64)
                        )
                    )
                ),
            )

            tile_image = torch.from_numpy(image[d_slice, h_slice, w_slice])[None, None].to(device)
            tile_coarse_device = tile_coarse.to(device)
            network_started = time.perf_counter()
            residual = refinement_model.residual_logits(tile_image, tile_coarse_device).cpu()
            network_seconds = float(time.perf_counter() - network_started)
            tile_coarse_cpu = tile_coarse.cpu()
            tile_target = target[d_slice, h_slice, w_slice]
            coarse_tile_errors = tile_prediction != tile_target

            for key in active_keys:
                state = candidate_states[key]
                roi_tile = tile_masks[key]
                refined_logits = tile_coarse_cpu + residual * torch.from_numpy(roi_tile)[None, None].to(
                    dtype=tile_coarse_cpu.dtype
                )
                refined_pred = torch.argmax(refined_logits, dim=1)[0].numpy().astype(np.uint8)
                output_tile = state["prediction"][d_slice, h_slice, w_slice]
                output_tile[roi_tile] = refined_pred[roi_tile]
                changed = output_tile != tile_prediction
                outside_changed = np.logical_and(changed, ~roi_tile)
                if bool(outside_changed.any()):
                    raise RuntimeError(f"{case_id}/{key}: ROI 外 prediction 被修改")
                state["roi_count"] += int(roi_tile.sum())
                state["coarse_roi_error_count"] += int(coarse_tile_errors[roi_tile].sum())
                state["refined_roi_error_count"] += int((refined_pred[roi_tile] != tile_target[roi_tile]).sum())
                state["outside_changed_count"] += int(outside_changed.sum())
                state["canonical_reconstruction_seconds"] += reconstruction_seconds
                state["refinement_network_seconds"] += network_seconds
                state["processed_tiles"] += 1

            if compare_full and full_prediction is not None:
                refined_full = torch.argmax(tile_coarse_cpu + residual, dim=1)[0]
                full_prediction[d_slice, h_slice, w_slice] = refined_full.numpy().astype(np.uint8)
                full_reconstruction_seconds += reconstruction_seconds
                full_network_seconds += network_seconds
                full_tiles += 1

    if reconstruction_mismatch != 0:
        raise RuntimeError(
            f"{case_id}: canonical logits reconstruction 改变 coarse prediction，"
            f"mismatch_voxels={reconstruction_mismatch}"
        )

    spacing = _spacing_dhw(label_path)
    coarse_inference_seconds = _load_coarse_inference_seconds(evaluation_dir)
    coarse_errors = coarse_pred != target
    coarse_global_error = float(coarse_errors.mean())
    total_voxels = int(coarse_pred.size)
    rows: list[dict[str, Any]] = []

    def build_row(
        *,
        mode: str,
        pred: np.ndarray,
        roi_count: int,
        coarse_roi_error_count: int,
        refined_roi_error_count: int,
        outside_changed_count: int,
        roi_selection_seconds: float,
        canonical_reconstruction_seconds: float,
        refinement_network_seconds: float,
        processed_tiles: int,
        top_percent: float | None,
        dilation: int | None,
    ) -> dict[str, Any]:
        overlap = compute_binary_metrics(pred > 0, target > 0, spacing).to_dict()
        structural = compute_structural_metrics(pred > 0, target > 0).to_dict()
        pred_fg = float(np.mean(pred > 0))
        target_fg = float(np.mean(target > 0))
        ratio = pred_fg / target_fg if target_fg > 0 else None
        refined_errors = pred != target
        refined_global_error = float(refined_errors.mean())
        coarse_roi_error = (
            0.0 if roi_count == 0 else float(coarse_roi_error_count / roi_count)
        )
        refined_roi_error = (
            0.0 if roi_count == 0 else float(refined_roi_error_count / roi_count)
        )
        outside_count = max(0, total_voxels - roi_count)
        outside_changed_fraction = (
            0.0 if outside_count == 0 else float(outside_changed_count / outside_count)
        )
        refinement_seconds = float(
            roi_selection_seconds + canonical_reconstruction_seconds + refinement_network_seconds
        )
        return {
            "case_id": case_id,
            "mode": mode,
            "top_percent": top_percent,
            "dilation_iterations": dilation,
            **overlap,
            **structural,
            "prediction_foreground_fraction": pred_fg,
            "target_foreground_fraction": target_fg,
            "prediction_to_target_foreground_ratio": ratio,
            "roi_fraction": float(roi_count / total_voxels),
            "coarse_roi_error_rate": coarse_roi_error,
            "refined_roi_error_rate": refined_roi_error,
            "roi_error_rate_delta": float(refined_roi_error - coarse_roi_error),
            "coarse_global_error_rate": coarse_global_error,
            "refined_global_error_rate": refined_global_error,
            "global_error_rate_delta": float(refined_global_error - coarse_global_error),
            "outside_roi_changed_fraction": outside_changed_fraction,
            "coarse_inference_seconds": coarse_inference_seconds,
            "roi_selection_seconds": float(roi_selection_seconds),
            "canonical_reconstruction_seconds": float(canonical_reconstruction_seconds),
            "refinement_network_seconds": float(refinement_network_seconds),
            "refinement_seconds": refinement_seconds,
            "total_pipeline_seconds": float(coarse_inference_seconds + refinement_seconds),
            "processed_tiles": int(processed_tiles),
        }

    rows.append(
        build_row(
            mode="coarse",
            pred=coarse_pred,
            roi_count=0,
            coarse_roi_error_count=0,
            refined_roi_error_count=0,
            outside_changed_count=0,
            roi_selection_seconds=0.0,
            canonical_reconstruction_seconds=0.0,
            refinement_network_seconds=0.0,
            processed_tiles=0,
            top_percent=None,
            dilation=None,
        )
    )

    save_predictions = bool(validation_cfg.get("save_predictions", True))
    for key, state in candidate_states.items():
        pred = state["prediction"]
        rows.append(
            build_row(
                mode="roi_only",
                pred=pred,
                roi_count=int(state["roi_count"]),
                coarse_roi_error_count=int(state["coarse_roi_error_count"]),
                refined_roi_error_count=int(state["refined_roi_error_count"]),
                outside_changed_count=int(state["outside_changed_count"]),
                roi_selection_seconds=float(state["roi_selection_seconds"]),
                canonical_reconstruction_seconds=float(state["canonical_reconstruction_seconds"]),
                refinement_network_seconds=float(state["refinement_network_seconds"]),
                processed_tiles=int(state["processed_tiles"]),
                top_percent=float(state["top_percent"]),
                dilation=int(state["dilation_iterations"]),
            )
        )
        if save_predictions:
            _save_dhw_nifti(
                pred.astype(np.uint8),
                label_path,
                output_dir / "predictions" / key / case_id / "prediction.nii.gz",
            )

    if compare_full and full_prediction is not None:
        full_errors = full_prediction != target
        rows.append(
            build_row(
                mode="full_volume_second_pass",
                pred=full_prediction,
                roi_count=total_voxels,
                coarse_roi_error_count=int(coarse_errors.sum()),
                refined_roi_error_count=int(full_errors.sum()),
                outside_changed_count=0,
                roi_selection_seconds=0.0,
                canonical_reconstruction_seconds=full_reconstruction_seconds,
                refinement_network_seconds=full_network_seconds,
                processed_tiles=full_tiles,
                top_percent=None,
                dilation=None,
            )
        )
        if save_predictions:
            _save_dhw_nifti(
                full_prediction.astype(np.uint8),
                label_path,
                output_dir / "predictions" / "full_volume_second_pass" / case_id / "prediction.nii.gz",
            )

    diagnostics = {
        "case_id": case_id,
        "canonical_reconstruction_prediction_mismatch_voxels": reconstruction_mismatch,
        "canonical_reconstruction_entropy_max_abs_error": reconstruction_entropy_max_abs_error,
        "shape_dhw": list(image.shape),
        "spacing_dhw_mm": list(spacing),
        "uncertainty_thresholds": {
            str(top): threshold for top, threshold in thresholds.items()
        },
    }
    return rows, diagnostics


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float | None, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["mode"], row["top_percent"], row["dilation_iterations"])
        groups.setdefault(key, []).append(row)

    numeric_fields = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm",
        "prediction_to_target_foreground_ratio",
        "component_count_error",
        "false_merge_count",
        "false_break_count",
        "roi_fraction",
        "coarse_roi_error_rate",
        "refined_roi_error_rate",
        "roi_error_rate_delta",
        "coarse_global_error_rate",
        "refined_global_error_rate",
        "global_error_rate_delta",
        "outside_roi_changed_fraction",
        "coarse_inference_seconds",
        "roi_selection_seconds",
        "canonical_reconstruction_seconds",
        "refinement_network_seconds",
        "refinement_seconds",
        "total_pipeline_seconds",
        "processed_tiles",
    ]
    aggregated: list[dict[str, Any]] = []
    for (mode, top_percent, dilation), items in groups.items():
        row: dict[str, Any] = {
            "mode": mode,
            "top_percent": top_percent,
            "dilation_iterations": dilation,
            "case_count": len(items),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in items if item.get(field) is not None]
            row[f"mean_{field}"] = None if not values else float(np.mean(values))
            row[f"std_{field}"] = None if not values else float(np.std(values))
        aggregated.append(row)
    return aggregated


def run(
    config_path: Path,
    output_dir: Path | None = None,
    refinement_checkpoint: Path | None = None,
) -> Path:
    experiment_config = _load_yaml(config_path)
    seed = int(experiment_config.get("seed", 42))
    seed_everything(seed)
    coarse_cfg = experiment_config["coarse"]
    coarse_config_path = _resolve(coarse_cfg["config"])
    coarse_checkpoint_path = _resolve(coarse_cfg["checkpoint"])
    coarse_config = _load_yaml(coarse_config_path)

    split_file = _resolve(coarse_config["data"]["split_file"])
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    validation_cases = [str(value) for value in split_payload["validation"]]
    test_cases = {str(value) for value in split_payload.get("test", [])}
    artifacts = {
        str(key): _resolve(value) for key, value in coarse_cfg["validation_artifacts"].items()
    }
    if set(artifacts) != set(validation_cases):
        raise ValueError(
            "validation_artifacts 必须且只能覆盖 validation split，"
            f"expected={validation_cases}, got={sorted(artifacts)}"
        )
    if set(artifacts) & test_cases:
        raise RuntimeError("refinement validation 禁止访问 test split")

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = str(experiment_config.get("experiment_name", "refinement_validation"))
        run_dir = PROJECT_ROOT / "experiments" / f"{stamp}_{name}"
    else:
        run_dir = _resolve(output_dir)

    resume_validation = refinement_checkpoint is not None
    if run_dir.exists() and not resume_validation:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=resume_validation)
    config_copy = run_dir / "config.yaml"
    source_config_text = config_path.read_text(encoding="utf-8")
    if config_copy.exists() and config_copy.read_text(encoding="utf-8") != source_config_text:
        raise ValueError("已有 refinement run 的 config.yaml 与当前 config 不一致")
    if not config_copy.exists():
        config_copy.write_text(source_config_text, encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if refinement_checkpoint is None:
        coarse_model = _load_coarse_model(coarse_config, coarse_checkpoint_path, device)
        refinement_model, training_history = _train_refinement(
            experiment_config=experiment_config,
            coarse_config=coarse_config,
            coarse_model=coarse_model,
            run_dir=run_dir,
            device=device,
        )
        del coarse_model
        refinement_checkpoint_used = run_dir / "checkpoint" / "last.pt"
    else:
        refinement_checkpoint_used = _resolve(refinement_checkpoint)
        refinement_model = _load_refinement_checkpoint(
            experiment_config, refinement_checkpoint_used, device
        )
        training_history = _read_training_history(run_dir / "training_history.csv")

    processed_root = _resolve(coarse_config["data"]["processed_root"])
    tile_dhw = tuple(int(v) for v in experiment_config["refinement"]["tile_size_dhw"])
    all_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case_id in validation_cases:
        case_rows, case_diagnostics = _evaluate_case_grid(
            case_id=case_id,
            processed_root=processed_root,
            evaluation_dir=artifacts[case_id],
            refinement_model=refinement_model,
            validation_cfg=experiment_config["validation"],
            tile_dhw=tile_dhw,
            device=device,
            output_dir=run_dir,
        )
        all_rows.extend(case_rows)
        diagnostics.append(case_diagnostics)

    per_case_path = run_dir / "metrics_per_case.csv"
    with per_case_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate = _aggregate_rows(all_rows)
    aggregate_path = run_dir / "summary_by_candidate.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    summary = {
        "finished_at": datetime.now().isoformat(),
        "device": str(device),
        "coarse_config": str(coarse_config_path),
        "coarse_checkpoint": str(coarse_checkpoint_path),
        "refinement_checkpoint": str(refinement_checkpoint_used),
        "train_split_case_count": len(split_payload["train"]),
        "validation_cases": validation_cases,
        "test_accessed": False,
        "validation_reused_saved_prediction_entropy": True,
        "resumed_validation_from_existing_refinement_checkpoint": resume_validation,
        "training_history": training_history,
        "canonical_reconstruction": diagnostics,
        "candidate_count": len(aggregate),
        "note": (
            "Refinement network trained on train split only. Validation grid uses liver_7/liver_8 only. "
            "No independent test access; final selection must be made from validation evidence before lock."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v13 uncertainty ROI refinement validation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--refinement-checkpoint",
        type=Path,
        default=None,
        help="复用已完成 refinement 训练的 checkpoint，仅继续 validation，不重新训练",
    )
    args = parser.parse_args()
    run_dir = run(
        _resolve(args.config),
        args.output_dir,
        None if args.refinement_checkpoint is None else _resolve(args.refinement_checkpoint),
    )
    print(f"Refinement validation completed: {run_dir}")


if __name__ == "__main__":
    main()
