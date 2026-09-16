"""生成与正式推理口径一致的 residual false-positive hard-negative guidance。

仅允许 train split。流程固定为：
full-volume running-BN inference -> ROI/overlap from config -> configured postprocessing
-> residual FP = postprocessed prediction foreground AND GT background
-> 选少量空间分离中心并保存稀疏 hard_centers.nii.gz。

这比局部 72^3 patch FP mining 更贴近最终稳定推理输出。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from scipy.ndimage import generate_binary_structure, label as connected_components

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.postprocessing import postprocess_prediction
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import (
    _model_predictor,
    _resolve_project_path,
    logits_to_prediction,
    seed_everything,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _save_center_mask(
    centers_dhw: list[tuple[int, int, int]],
    *,
    shape_dhw: tuple[int, int, int],
    reference_label: Path,
    output_path: Path,
) -> None:
    mask = np.zeros(shape_dhw, dtype=np.uint8)
    for center in centers_dhw:
        mask[center] = 1
    reference = nib.load(str(reference_label))
    mask_xyz = np.transpose(mask, (2, 1, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(mask_xyz, affine=reference.affine, header=reference.header),
        str(output_path),
    )


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"logits 必须为 [B,C,D,H,W]，实际 {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    probabilities = torch.softmax(logits, dim=1)
    return 1.0 - probabilities[:, 0]


def select_residual_fp_centers(
    residual_mask: np.ndarray,
    probability: np.ndarray,
    *,
    max_centers: int,
    min_center_distance_voxels: float,
) -> list[tuple[int, int, int]]:
    """从 residual FP 中选高置信、空间分离的中心。"""
    residual = np.asarray(residual_mask, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    if residual.shape != probability.shape:
        raise ValueError("residual_mask 与 probability shape 不一致")
    if residual.ndim != 3:
        raise ValueError("residual_mask 必须为 3D")
    if max_centers < 1:
        raise ValueError("max_centers 必须 >= 1")
    if min_center_distance_voxels < 0:
        raise ValueError("min_center_distance_voxels 必须 >= 0")

    coords = np.argwhere(residual)
    if len(coords) == 0:
        return []
    scores = probability[residual]
    order = np.argsort(scores)[::-1]
    selected: list[tuple[int, int, int]] = []
    min_sq = float(min_center_distance_voxels) ** 2
    for index in order:
        coord = tuple(int(v) for v in coords[int(index)])
        if selected and min_sq > 0:
            if any(
                sum((coord[axis] - prev[axis]) ** 2 for axis in range(3)) < min_sq
                for prev in selected
            ):
                continue
        selected.append(coord)
        if len(selected) >= max_centers:
            break
    return selected


def _select_evenly_spaced(case_ids: list[str], max_cases: int | None) -> list[str]:
    if max_cases is None or max_cases >= len(case_ids):
        return list(case_ids)
    max_cases = int(max_cases)
    if max_cases <= 0:
        raise ValueError("max_cases 必须 > 0")
    indices = np.linspace(0, len(case_ids) - 1, num=max_cases, dtype=int)
    return [case_ids[int(index)] for index in indices]


def generate_residual_fp_guidance(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    max_cases: int | None = None,
    case_id: str | None = None,
    max_centers_per_case: int = 2,
    min_center_distance_voxels: float = 48.0,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    output_root = _resolve_project_path(output_root)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if case_id is not None and max_cases is not None:
        raise ValueError("case_id 与 max_cases 不能同时使用")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]
    if str(infer_cfg.get("batchnorm_mode", "running")).lower() != "running":
        raise ValueError("residual FP guidance 必须使用正式 running BN 推理口径")
    post_cfg = dict(infer_cfg.get("postprocessing", {}) or {})
    min_component_voxels = int(post_cfg.get("min_component_voxels", 0) or 0)
    connectivity = int(post_cfg.get("connectivity", 3) or 3)
    if min_component_voxels <= 1:
        raise ValueError("v4-B residual FP guidance 要求启用正式连通域后处理")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    processed_root = _resolve_project_path(data_cfg["processed_root"])
    split_file = _resolve_project_path(data_cfg["split_file"])
    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "train",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [72, 72, 72]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=seed,
    )
    if case_id is not None:
        if case_id not in dataset.case_ids:
            raise ValueError(f"case_id={case_id!r} 不属于 train split")
        dataset.case_ids = [case_id]
    else:
        dataset.case_ids = _select_evenly_spaced(dataset.case_ids, max_cases)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()

    roi_size = tuple(int(v) for v in infer_cfg.get("roi_size_dhw", [96, 96, 96]))
    sw_batch_size = int(infer_cfg.get("sw_batch_size", 1))
    overlap = float(infer_cfg.get("overlap", 0.25))
    status_path = output_root / "guidance_status.json"
    started = time.perf_counter()
    case_summaries: list[dict[str, Any]] = []

    with torch.no_grad():
        for index in range(len(dataset)):
            current_case = str(dataset.case_ids[index])
            case_dir = output_root / current_case
            existing_scores = case_dir / "scores.json"
            existing_mask = case_dir / "hard_centers.nii.gz"
            if resume and existing_scores.exists() and existing_mask.exists():
                try:
                    case_payload = json.loads(existing_scores.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    case_payload = {}
                if str(case_payload.get("case_id") or "") == current_case:
                    case_summaries.append(case_payload)
                    completed = index + 1
                    elapsed = time.perf_counter() - started
                    _write_json_atomic(
                        status_path,
                        {
                            "updated_at": datetime.now().isoformat(),
                            "phase": "running" if completed < len(dataset) else "completed",
                            "split": "train",
                            "completed_case_count": completed,
                            "expected_case_count": len(dataset),
                            "current_case_id": current_case,
                            "elapsed_seconds": elapsed,
                            "eta_seconds": None,
                            "checkpoint": str(checkpoint_path),
                            "config": str(config_path),
                        },
                    )
                    print(f"[GUIDANCE] {completed}/{len(dataset)} | {current_case} | resume skip", flush=True)
                    continue

            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            target = item["label"].numpy()
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=_model_predictor(model),
                overlap=overlap,
                mode="gaussian",
            )
            pred_raw = logits_to_prediction(logits)[0].detach().cpu().numpy()
            pred_post = postprocess_prediction(pred_raw, post_cfg)
            probability = foreground_probability(logits)[0].detach().cpu().numpy()
            residual = np.logical_and(pred_post > 0, target == 0)

            centers = select_residual_fp_centers(
                residual,
                probability,
                max_centers=max_centers_per_case,
                min_center_distance_voxels=min_center_distance_voxels,
            )
            pred_labels, _ = connected_components(
                pred_post > 0,
                structure=generate_binary_structure(3, connectivity),
            )
            selected_records: list[dict[str, Any]] = []
            for center in centers:
                component_id = int(pred_labels[center])
                component_size = int(np.count_nonzero(pred_labels == component_id)) if component_id else 0
                selected_records.append(
                    {
                        "center_dhw": list(center),
                        "foreground_probability": float(probability[center]),
                        "prediction_component_size": component_size,
                    }
                )

            label_path = processed_root / current_case / "label.nii.gz"
            _save_center_mask(
                centers,
                shape_dhw=tuple(int(v) for v in target.shape),
                reference_label=label_path,
                output_path=case_dir / "hard_centers.nii.gz",
            )
            case_payload = {
                "generated_at": datetime.now().isoformat(),
                "case_id": current_case,
                "split": "train",
                "selected_count": len(centers),
                "residual_fp_voxel_count": int(residual.sum()),
                "postprocessed_foreground_voxel_count": int(np.count_nonzero(pred_post > 0)),
                "selected": selected_records,
            }
            _write_json_atomic(case_dir / "scores.json", case_payload)
            case_summaries.append(case_payload)

            completed = index + 1
            elapsed = time.perf_counter() - started
            eta = elapsed / completed * (len(dataset) - completed) if completed else None
            _write_json_atomic(
                status_path,
                {
                    "updated_at": datetime.now().isoformat(),
                    "phase": "running" if completed < len(dataset) else "completed",
                    "split": "train",
                    "completed_case_count": completed,
                    "expected_case_count": len(dataset),
                    "current_case_id": current_case,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "checkpoint": str(checkpoint_path),
                    "config": str(config_path),
                },
            )
            print(
                f"[GUIDANCE] {completed}/{len(dataset)} | {current_case} | "
                f"residual_fp={int(residual.sum())} | centers={len(centers)}",
                flush=True,
            )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "split": "train",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "case_count": len(case_summaries),
        "max_cases": max_cases,
        "inference": {
            "batchnorm_mode": "running",
            "roi_size_dhw": list(roi_size),
            "overlap": overlap,
            "postprocessing": post_cfg,
        },
        "max_centers_per_case": int(max_centers_per_case),
        "min_center_distance_voxels": float(min_center_distance_voxels),
        "cases_with_guidance": int(sum(int(case["selected_count"]) > 0 for case in case_summaries)),
        "selected_center_count": int(sum(int(case["selected_count"]) for case in case_summaries)),
        "cases": case_summaries,
        "note": "Train-only residual FP guidance aligned to formal running-BN ROI96 + configured postprocessing. Validation/test are never used.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_root / "summary.json", summary)
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="生成 v4-B 正式推理残余假阳性 guidance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--max-centers-per-case", type=int, default=2)
    parser.add_argument("--min-center-distance-voxels", type=float, default=48.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = generate_residual_fp_guidance(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        max_cases=args.max_cases,
        case_id=args.case_id,
        max_centers_per_case=args.max_centers_per_case,
        min_center_distance_voxels=args.min_center_distance_voxels,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
