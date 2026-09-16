from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from scipy.ndimage import generate_binary_structure, label as connected_components
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.diagnostics import batchnorm_batch_stats_mode
from src.modeling.metrics import binary_overlap_metrics
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import (
    _model_predictor,
    _resolve_project_path,
    logits_to_prediction,
    select_validation_case_subset,
)


def _mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _write_json_atomic(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _filtered_from_labels(labels: np.ndarray, sizes: np.ndarray, threshold: int) -> tuple[np.ndarray, int]:
    if threshold <= 1:
        kept = np.arange(1, len(sizes), dtype=np.int32)
    else:
        kept = np.flatnonzero(sizes >= int(threshold))
        kept = kept[kept > 0]
    lookup = np.zeros(len(sizes), dtype=bool)
    lookup[kept] = True
    return lookup[labels], int(len(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only batch-BN component threshold sweep")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[0, 1024, 2048, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 30720, 32768, 34816],
    )
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]

    dataset = ProcessedOrthopedicCTDataset(
        processed_root=_resolve_project_path(data_cfg["processed_root"]),
        split_file=_resolve_project_path(data_cfg["split_file"]),
        split="validation",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [128, 128, 128]),
        training=False,
        foreground_probability=float(data_cfg.get("foreground_probability", 0.5)),
        label_mode=str(data_cfg.get("label_mode", "binary")),
        hu_clip=data_cfg.get("hu_clip", [-1000, 2000]),
        seed=int(config.get("seed", 42)),
    )
    dataset.case_ids = select_validation_case_subset(dataset.case_ids, int(args.max_cases))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()

    roi = tuple(int(v) for v in infer_cfg.get("roi_size_dhw", data_cfg.get("roi_size_dhw", [128] * 3)))
    sw_batch_size = int(infer_cfg.get("sw_batch_size", 1))
    overlap = float(infer_cfg.get("overlap", 0.5))
    thresholds = sorted(set(int(v) for v in args.thresholds))
    if any(v < 0 for v in thresholds):
        raise ValueError("thresholds 必须 >= 0")

    structure = generate_binary_structure(3, 3)
    accum: dict[int, dict[str, list[float]]] = {
        t: {k: [] for k in ("dice", "iou", "precision", "recall", "fg_ratio", "component_error")}
        for t in thresholds
    }
    zero_dice_count = {t: 0 for t in thresholds}
    empty_prediction_count = {t: 0 for t in thresholds}
    case_rows: list[dict] = []

    started = time.perf_counter()
    with batchnorm_batch_stats_mode(model), torch.no_grad():
        for index, batch in enumerate(loader, start=1):
            case_id = str(batch["case_id"][0])
            image = batch["image"].to(device)
            target = batch["label"][0].cpu().numpy().astype(np.int16) > 0
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=_model_predictor(model),
                overlap=overlap,
                mode="gaussian",
            )
            pred = logits_to_prediction(logits)[0].cpu().numpy().astype(np.int16) > 0
            pred_labels, pred_count = connected_components(pred, structure=structure)
            pred_sizes = np.bincount(pred_labels.ravel(), minlength=int(pred_count) + 1)
            _, target_count = connected_components(target, structure=structure)
            target_fg = float(target.mean())

            per_case: dict[str, object] = {"case_id": case_id, "thresholds": {}}
            for threshold in thresholds:
                filtered, kept_count = _filtered_from_labels(pred_labels, pred_sizes, threshold)
                dice, iou, precision, recall = binary_overlap_metrics(filtered, target)
                pred_fg = float(filtered.mean())
                fg_ratio = pred_fg / target_fg if target_fg > 0 else float("nan")
                component_error = float(abs(int(kept_count) - int(target_count)))
                values = {
                    "dice": float(dice),
                    "iou": float(iou),
                    "precision": float(precision),
                    "recall": float(recall),
                    "fg_ratio": float(fg_ratio),
                    "component_error": component_error,
                }
                for key, value in values.items():
                    accum[threshold][key].append(value)
                if dice == 0.0:
                    zero_dice_count[threshold] += 1
                if not np.any(filtered):
                    empty_prediction_count[threshold] += 1
                per_case["thresholds"][str(threshold)] = values
            case_rows.append(per_case)
            _write_json_atomic(
                status_path,
                {
                    "updated_at": datetime.now().isoformat(),
                    "phase": "running",
                    "bn_mode": "batch",
                    "completed_case_count": index,
                    "expected_case_count": len(dataset.case_ids),
                    "last_case_id": case_id,
                },
            )

    rows: list[dict] = []
    for threshold in thresholds:
        row = {"min_component_voxels": threshold}
        for key, values in accum[threshold].items():
            row[f"mean_{key}"] = _mean(values)
        row["zero_dice_case_count"] = int(zero_dice_count[threshold])
        row["empty_prediction_case_count"] = int(empty_prediction_count[threshold])
        rows.append(row)

    best = max(rows, key=lambda item: float(item["mean_dice"]))
    output = {
        "created_at": datetime.now().isoformat(),
        "scope": "validation_only",
        "test_used": False,
        "bn_mode": "batch",
        "case_count": len(dataset.case_ids),
        "case_ids": list(dataset.case_ids),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "thresholds_voxels": thresholds,
        "rows": rows,
        "best_min_component_voxels_by_mean_dice": int(best["min_component_voxels"]),
        "best_row": best,
        "cases": case_rows,
        "note": "仅用于 validation 上判断 batch-BN 推理是否稳定；不得写成最终 test 性能。",
    }
    _write_json_atomic(output_dir / "summary.json", output)
    _write_json_atomic(
        status_path,
        {
            "updated_at": datetime.now().isoformat(),
            "phase": "completed",
            "bn_mode": "batch",
            "completed_case_count": len(dataset.case_ids),
            "expected_case_count": len(dataset.case_ids),
        },
    )


if __name__ == "__main__":
    main()
