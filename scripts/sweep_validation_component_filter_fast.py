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


def _safe_mean(values: list[float]) -> float:
    finite = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def _load_bool(path: Path) -> np.ndarray:
    image = nib.load(str(path))
    return np.asarray(image.dataobj) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="快速 validation-only 连通域过滤阈值扫描")
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 32, 64, 128, 256, 512, 1024, 2048])
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir.resolve()
    processed_root = args.processed_root.resolve()
    with (evaluation_dir / "metrics_per_case.csv").open("r", encoding="utf-8", newline="") as f:
        case_ids = [str(row["case_id"]) for row in csv.DictReader(f)]
    if len(case_ids) != 197:
        raise RuntimeError(f"validation 必须完整 197 例，当前={len(case_ids)}")

    thresholds = sorted(set(int(v) for v in args.thresholds))
    structure = generate_binary_structure(3, 3)
    accum = {
        t: {key: [] for key in ("dice", "iou", "precision", "recall", "fg_ratio", "component_error")}
        for t in thresholds
    }

    status_path = evaluation_dir / "postprocessing_component_filter_fast_status.json"
    for index, case_id in enumerate(case_ids, start=1):
        pred = _load_bool(evaluation_dir / "predictions" / case_id / "prediction.nii.gz")
        target = _load_bool(processed_root / case_id / "label.nii.gz")
        if pred.shape != target.shape:
            raise RuntimeError(f"{case_id}: prediction/label shape 不一致 {pred.shape} vs {target.shape}")

        pred_labels, pred_count = connected_components(pred, structure=structure)
        pred_sizes = np.bincount(pred_labels.ravel(), minlength=int(pred_count) + 1).astype(np.int64)
        # 每个预测连通分量中与真值重叠的体素数。这样无需为每个阈值重新构造整幅 3D mask。
        overlap_sizes = np.bincount(
            pred_labels[target].ravel(), minlength=int(pred_count) + 1
        ).astype(np.int64)
        _, target_count = connected_components(target, structure=structure)
        target_sum = int(target.sum())

        component_ids = np.arange(1, int(pred_count) + 1, dtype=np.int64)
        component_sizes = pred_sizes[1:]
        component_overlap = overlap_sizes[1:]

        for threshold in thresholds:
            if threshold <= 1:
                keep = np.ones_like(component_ids, dtype=bool)
            else:
                keep = component_sizes >= int(threshold)
            pred_sum = int(component_sizes[keep].sum())
            tp = int(component_overlap[keep].sum())
            fp = pred_sum - tp
            fn = target_sum - tp

            dice = 1.0 if pred_sum + target_sum == 0 else (2.0 * tp) / (pred_sum + target_sum)
            iou = 1.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
            precision = (1.0 if target_sum == 0 else 0.0) if pred_sum == 0 else tp / pred_sum
            recall = (1.0 if pred_sum == 0 else 0.0) if target_sum == 0 else tp / target_sum
            fg_ratio = pred_sum / target_sum if target_sum > 0 else float("nan")
            kept_count = int(keep.sum())

            accum[threshold]["dice"].append(float(dice))
            accum[threshold]["iou"].append(float(iou))
            accum[threshold]["precision"].append(float(precision))
            accum[threshold]["recall"].append(float(recall))
            accum[threshold]["fg_ratio"].append(float(fg_ratio))
            accum[threshold]["component_error"].append(float(abs(kept_count - int(target_count))))

        if index % 10 == 0 or index == len(case_ids):
            status_path.write_text(
                json.dumps(
                    {
                        "updated_at": datetime.now().isoformat(),
                        "completed_case_count": index,
                        "expected_case_count": len(case_ids),
                        "last_case_id": case_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        row: dict[str, float | int] = {"min_component_voxels": threshold}
        for key, values in accum[threshold].items():
            row[f"mean_{key}"] = _safe_mean(values)
        rows.append(row)

    best = max(rows, key=lambda row: float(row["mean_dice"]))
    output = {
        "created_at": datetime.now().isoformat(),
        "purpose": "validation_only_fast_component_filter_sweep",
        "test_used": False,
        "case_count": len(case_ids),
        "thresholds_voxels": thresholds,
        "rows": rows,
        "best_min_component_voxels_by_mean_dice": int(best["min_component_voxels"]),
        "best_row": best,
        "note": "仅用于 validation 后处理选择；最终配置冻结后才允许对 test 评估一次。",
    }
    output_path = evaluation_dir / "postprocessing_component_filter_fast_summary.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    status_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now().isoformat(),
                "completed_case_count": len(case_ids),
                "expected_case_count": len(case_ids),
                "phase": "completed",
                "summary": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
