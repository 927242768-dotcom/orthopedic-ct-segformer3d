from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label as connected_components

from src.modeling.metrics import compute_binary_metrics, compute_structural_metrics


def _safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    array = np.asarray(image.dataobj) > 0
    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
    return array, spacing


def _overlap(pred: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    pred_sum = tp + fp
    target_sum = tp + fn
    dice = 1.0 if pred_sum + target_sum == 0 else 2.0 * tp / (pred_sum + target_sum)
    iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    precision = (1.0 if target_sum == 0 else 0.0) if pred_sum == 0 else tp / pred_sum
    recall = (1.0 if pred_sum == 0 else 0.0) if target_sum == 0 else tp / target_sum
    return float(dice), float(iou), float(precision), float(recall)


def _filtered_mask(labels: np.ndarray, sizes: np.ndarray, min_voxels: int) -> tuple[np.ndarray, int]:
    if min_voxels <= 1:
        kept = np.arange(1, len(sizes), dtype=np.int32)
    else:
        kept = np.flatnonzero(sizes >= int(min_voxels))
        kept = kept[kept > 0]
    lookup = np.zeros(len(sizes), dtype=bool)
    lookup[kept] = True
    return lookup[labels], int(len(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only connected-component filter sweep")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 32, 64, 128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    processed_root = args.processed_root.resolve()
    metrics_csv = evaluation_dir / "metrics_per_case.csv"
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        case_ids = [str(row["case_id"]) for row in csv.DictReader(f)]
    if len(case_ids) != 197:
        raise RuntimeError(f"validation 必须完整 197 例，当前={len(case_ids)}")

    thresholds = sorted(set(int(v) for v in args.thresholds))
    structure = generate_binary_structure(3, 3)
    accum: dict[int, dict[str, list[float]]] = {
        t: {k: [] for k in ("dice", "iou", "precision", "recall", "fg_ratio", "component_error")}
        for t in thresholds
    }

    for index, case_id in enumerate(case_ids, start=1):
        pred_path = evaluation_dir / "predictions" / case_id / "prediction.nii.gz"
        label_path = processed_root / case_id / "label.nii.gz"
        pred, _ = _load_mask(pred_path)
        target, _ = _load_mask(label_path)
        if pred.shape != target.shape:
            raise RuntimeError(f"{case_id}: prediction/label shape 不一致 {pred.shape} vs {target.shape}")

        pred_labels, pred_count = connected_components(pred, structure=structure)
        pred_sizes = np.bincount(pred_labels.ravel(), minlength=int(pred_count) + 1)
        _, target_count = connected_components(target, structure=structure)
        target_sum = float(target.sum())

        for threshold in thresholds:
            filtered, kept_count = _filtered_mask(pred_labels, pred_sizes, threshold)
            dice, iou, precision, recall = _overlap(filtered, target)
            pred_sum = float(filtered.sum())
            fg_ratio = pred_sum / target_sum if target_sum > 0 else float("nan")
            accum[threshold]["dice"].append(dice)
            accum[threshold]["iou"].append(iou)
            accum[threshold]["precision"].append(precision)
            accum[threshold]["recall"].append(recall)
            accum[threshold]["fg_ratio"].append(fg_ratio)
            accum[threshold]["component_error"].append(abs(int(kept_count) - int(target_count)))

        if index % 25 == 0 or index == len(case_ids):
            print(f"[SWEEP] {index}/{len(case_ids)}", flush=True)

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        item = {"min_component_voxels": threshold}
        for key, values in accum[threshold].items():
            item[f"mean_{key}"] = _safe_mean(values)
        rows.append(item)

    best = max(rows, key=lambda item: float(item["mean_dice"]))
    best_threshold = int(best["min_component_voxels"])

    # 对验证集选择出的最佳阈值再完整计算表面/结构指标。
    full_rows: list[dict[str, float | int | str]] = []
    for index, case_id in enumerate(case_ids, start=1):
        pred_path = evaluation_dir / "predictions" / case_id / "prediction.nii.gz"
        label_path = processed_root / case_id / "label.nii.gz"
        pred, spacing = _load_mask(pred_path)
        target, _ = _load_mask(label_path)
        pred_labels, pred_count = connected_components(pred, structure=structure)
        pred_sizes = np.bincount(pred_labels.ravel(), minlength=int(pred_count) + 1)
        filtered, _ = _filtered_mask(pred_labels, pred_sizes, best_threshold)
        region = compute_binary_metrics(filtered, target, spacing).to_dict()
        structural = compute_structural_metrics(filtered, target).to_dict()
        pred_fg = float(filtered.mean())
        target_fg = float(target.mean())
        full_rows.append({
            "case_id": case_id,
            **region,
            **structural,
            "prediction_foreground_fraction": pred_fg,
            "target_foreground_fraction": target_fg,
            "prediction_to_target_foreground_ratio": pred_fg / target_fg if target_fg > 0 else float("nan"),
        })
        if index % 25 == 0 or index == len(case_ids):
            print(f"[FULL] {index}/{len(case_ids)}", flush=True)

    summary_metrics: dict[str, dict[str, float]] = {}
    for key in (
        "dice", "iou", "precision", "recall", "hd95_mm", "assd_mm",
        "prediction_foreground_fraction", "target_foreground_fraction",
        "prediction_to_target_foreground_ratio", "component_count_error",
        "false_merge_count", "false_break_count",
    ):
        values = [float(row[key]) for row in full_rows]
        finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
        summary_metrics[key] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
        }

    output_dir = evaluation_dir / "postprocessing_component_filter"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "threshold_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "best_threshold_metrics_per_case.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(full_rows[0].keys()))
        writer.writeheader()
        writer.writerows(full_rows)

    summary = {
        "created_at": datetime.now().isoformat(),
        "purpose": "validation_only_postprocessing_selection",
        "test_used": False,
        "case_count": len(case_ids),
        "thresholds_voxels": thresholds,
        "best_min_component_voxels_by_mean_dice": best_threshold,
        "best_sweep_row": best,
        "best_threshold_full_metrics": summary_metrics,
        "note": "该阈值在 validation 上选择，只能作为模型/后处理选择依据；最终冻结后再对 test 评估一次。",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
