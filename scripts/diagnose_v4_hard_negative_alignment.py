"""诊断 v4 hard-negative guidance 与正式 ROI96 sliding-window 推理错误是否对齐。

仅处理 train split 中已经生成 guidance 的病例；不会访问 validation/test。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from scipy.ndimage import generate_binary_structure, label as connected_components

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.generate_hard_mining_guidance import _foreground_probability
from src.modeling.postprocessing import postprocess_prediction
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import _model_predictor, _resolve_project_path


def diagnose(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    guidance_root: str | Path,
    max_cases: int = 8,
    local_radius: int = 6,
) -> dict:
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    guidance_root = _resolve_project_path(guidance_root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]

    score_paths = sorted(guidance_root.glob("*/scores.json"))
    if not score_paths:
        raise FileNotFoundError(f"guidance_root 下没有 scores.json: {guidance_root}")
    max_cases = min(int(max_cases), len(score_paths))
    if max_cases <= 0:
        raise ValueError("max_cases 必须 > 0")
    indices = np.linspace(0, len(score_paths) - 1, num=max_cases, dtype=int)
    selected_score_paths = [score_paths[int(i)] for i in indices]
    case_ids = [
        str(json.loads(path.read_text(encoding="utf-8"))["case_id"])
        for path in selected_score_paths
    ]

    split_file = _resolve_project_path(data_cfg["split_file"])
    split_payload = json.loads(split_file.read_text(encoding="utf-8"))
    train_ids = set(map(str, split_payload["train"]))
    leaked = sorted(set(case_ids) - train_ids)
    if leaked:
        raise RuntimeError(f"发现非 train guidance 病例，拒绝继续: {leaked}")

    dataset = ProcessedOrthopedicCTDataset(
        _resolve_project_path(data_cfg["processed_root"]),
        split_file,
        "train",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [72, 72, 72]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=int(config.get("seed", 42)),
    )
    case_to_index = {case_id: index for index, case_id in enumerate(dataset.case_ids)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()

    roi = tuple(int(v) for v in infer_cfg.get("roi_size_dhw", [96, 96, 96]))
    overlap = float(infer_cfg.get("overlap", 0.25))
    sw_batch_size = int(infer_cfg.get("sw_batch_size", 1))
    post_cfg = dict(infer_cfg.get("postprocessing", {}) or {})
    structure = generate_binary_structure(3, int(post_cfg.get("connectivity", 3) or 3))

    rows: list[dict] = []
    with torch.no_grad():
        for score_path, case_id in zip(selected_score_paths, case_ids):
            item = dataset[case_to_index[case_id]]
            image = item["image"].unsqueeze(0).to(device)
            target = item["label"].numpy() > 0
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=_model_predictor(model),
                overlap=overlap,
                mode="gaussian",
            )
            probability = _foreground_probability(logits)[0].cpu().numpy()
            raw_prediction = np.argmax(logits.cpu().numpy()[0], axis=0).astype(np.uint8)
            raw_fg = raw_prediction > 0
            component_labels, component_count = connected_components(raw_fg, structure=structure)
            sizes = np.bincount(component_labels.ravel(), minlength=int(component_count) + 1)
            post_prediction = postprocess_prediction(raw_prediction, post_cfg) > 0

            payload = json.loads(score_path.read_text(encoding="utf-8"))
            for selected in payload["selected"]:
                center = tuple(int(v) for v in selected["center_dhw"])
                component_id = int(component_labels[center])
                component_size = int(sizes[component_id]) if component_id > 0 else 0
                slices = tuple(
                    slice(max(0, center[axis] - local_radius), min(target.shape[axis], center[axis] + local_radius + 1))
                    for axis in range(3)
                )
                local_bg = np.logical_not(target[slices])
                local_prob = probability[slices]
                local_bg_probs = local_prob[local_bg]
                local_max = float(local_bg_probs.max()) if local_bg_probs.size else float("nan")
                rows.append(
                    {
                        "case_id": case_id,
                        "center_dhw": list(center),
                        "gt_background": bool(not target[center]),
                        "patch_level_confidence": float(selected["false_positive_confidence"]),
                        "patch_level_fp_voxels": int(selected["false_positive_voxel_count"]),
                        "full_volume_probability": float(probability[center]),
                        "full_volume_raw_fp": bool(raw_fg[center] and not target[center]),
                        "full_volume_component_size": component_size,
                        "full_volume_survives_postprocessing": bool(post_prediction[center] and not target[center]),
                        "local_bg_max_probability_r": local_max,
                        "local_raw_fp_within_r": bool(np.any(np.logical_and(raw_fg[slices], local_bg))),
                    }
                )

    full_probs = np.asarray([row["full_volume_probability"] for row in rows], dtype=np.float64)
    patch_probs = np.asarray([row["patch_level_confidence"] for row in rows], dtype=np.float64)
    component_sizes = np.asarray([row["full_volume_component_size"] for row in rows], dtype=np.int64)
    min_component = int(post_cfg.get("min_component_voxels", 0) or 0)
    summary = {
        "split": "train",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "guidance_root": str(guidance_root),
        "case_count": len(case_ids),
        "selected_center_count": len(rows),
        "inference_roi_size_dhw": list(roi),
        "inference_overlap": overlap,
        "postprocessing_min_component_voxels": min_component,
        "exact_raw_fp_count": sum(bool(row["full_volume_raw_fp"]) for row in rows),
        "exact_raw_fp_fraction": float(np.mean([bool(row["full_volume_raw_fp"]) for row in rows])),
        "survives_postprocessing_count": sum(bool(row["full_volume_survives_postprocessing"]) for row in rows),
        "survives_postprocessing_fraction": float(
            np.mean([bool(row["full_volume_survives_postprocessing"]) for row in rows])
        ),
        "local_raw_fp_within_radius_count": sum(bool(row["local_raw_fp_within_r"]) for row in rows),
        "local_raw_fp_within_radius_fraction": float(
            np.mean([bool(row["local_raw_fp_within_r"]) for row in rows])
        ),
        "patch_level_confidence_mean": float(patch_probs.mean()),
        "patch_level_confidence_median": float(np.median(patch_probs)),
        "full_volume_probability_mean": float(full_probs.mean()),
        "full_volume_probability_median": float(np.median(full_probs)),
        "full_volume_probability_ge_0_5_count": int(np.count_nonzero(full_probs >= 0.5)),
        "full_volume_component_size_median": float(np.median(component_sizes)),
        "full_volume_component_size_max": int(component_sizes.max(initial=0)),
        "rows": rows,
    }
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--guidance-root", required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--local-radius", type=int, default=6)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    result = diagnose(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        guidance_root=args.guidance_root,
        max_cases=args.max_cases,
        local_radius=args.local_radius,
    )
    if args.summary_only:
        result = {key: value for key, value in result.items() if key != "rows"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
