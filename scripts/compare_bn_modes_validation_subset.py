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
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.diagnostics import batchnorm_batch_stats_mode
from src.modeling.metrics import binary_overlap_metrics, compute_structural_metrics
from src.modeling.postprocessing import postprocess_prediction
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


def _write_status(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only BN running/batch stats comparison")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--min-component-voxels", type=int, default=34816)
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    output_dir = _resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]
    model_cfg = config["model"]

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
    min_voxels = int(args.min_component_voxels)

    all_results: dict[str, dict] = {}
    for bn_mode in ("running", "batch"):
        mode_rows: list[dict] = []
        context = batchnorm_batch_stats_mode(model) if bn_mode == "batch" else nullcontext()
        started = time.perf_counter()
        with context, torch.no_grad():
            for index, batch in enumerate(loader, start=1):
                case_id = str(batch["case_id"][0])
                image = batch["image"].to(device)
                target = batch["label"][0].cpu().numpy().astype(np.int16)
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi,
                    sw_batch_size=sw_batch_size,
                    predictor=_model_predictor(model),
                    overlap=overlap,
                    mode="gaussian",
                )
                pred_raw = logits_to_prediction(logits)[0].cpu().numpy().astype(np.int16)
                pred_filtered = postprocess_prediction(
                    pred_raw,
                    {"min_component_voxels": min_voxels, "connectivity": 3},
                )

                row = {"case_id": case_id}
                for label, pred in (("raw", pred_raw), ("filtered", pred_filtered)):
                    dice, iou, precision, recall = binary_overlap_metrics(pred > 0, target > 0)
                    target_fg = float(np.mean(target > 0))
                    pred_fg = float(np.mean(pred > 0))
                    structural = compute_structural_metrics(pred > 0, target > 0)
                    row.update(
                        {
                            f"{label}_dice": float(dice),
                            f"{label}_iou": float(iou),
                            f"{label}_precision": float(precision),
                            f"{label}_recall": float(recall),
                            f"{label}_fg_ratio": pred_fg / target_fg if target_fg > 0 else float("nan"),
                            f"{label}_component_error": float(structural.component_count_error),
                        }
                    )
                mode_rows.append(row)
                _write_status(
                    status_path,
                    {
                        "updated_at": datetime.now().isoformat(),
                        "bn_mode": bn_mode,
                        "completed_case_count": index,
                        "expected_case_count": len(dataset.case_ids),
                        "last_case_id": case_id,
                    },
                )

        summary: dict[str, float | int] = {
            "case_count": len(mode_rows),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        for label in ("raw", "filtered"):
            for metric in ("dice", "iou", "precision", "recall", "fg_ratio", "component_error"):
                summary[f"mean_{label}_{metric}"] = _mean(
                    [float(row[f"{label}_{metric}"]) for row in mode_rows]
                )
        all_results[bn_mode] = {"summary": summary, "rows": mode_rows}

    output = {
        "created_at": datetime.now().isoformat(),
        "scope": "validation_only",
        "test_used": False,
        "case_count": len(dataset.case_ids),
        "case_ids": list(dataset.case_ids),
        "min_component_voxels": min_voxels,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "results": all_results,
        "note": "仅用于 validation 模型选择；不得写成最终 test 性能。",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_status(
        status_path,
        {
            "updated_at": datetime.now().isoformat(),
            "phase": "completed",
            "completed_case_count": len(dataset.case_ids),
            "expected_case_count": len(dataset.case_ids),
        },
    )


if __name__ == "__main__":
    main()
