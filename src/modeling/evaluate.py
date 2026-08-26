"""SegFormer3D 独立测试集评估入口。

用途：
- 从已训练 checkpoint 恢复模型；
- 对 validation/test 做 sliding-window inference；
- 逐病例导出 Dice/IoU/Precision/Recall/HD95/ASSD 与推理时间；
- 可选保存 prediction NIfTI 与 predictive-entropy NIfTI；
- 生成可追溯 summary.json。

该脚本不会把 validation 结果伪装成 test，也不会在没有 checkpoint 时生成结果。
"""

from __future__ import annotations

import argparse
import csv
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
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_structural_metrics,
)
from src.modeling.preflight import run_preflight
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d, upstream_provenance
from src.modeling.train import PROJECT_ROOT, _model_predictor, _resolve_project_path, logits_to_prediction
from src.modeling.uncertainty import (
    predictive_entropy,
    segmentation_calibration_metrics,
    uncertainty_error_metrics,
)


def _save_dhw_nifti(array_dhw: np.ndarray, reference_path: Path, output_path: Path) -> None:
    reference = nib.load(str(reference_path))
    array_xyz = np.transpose(np.asarray(array_dhw), (2, 1, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(array_xyz, affine=reference.affine, header=reference.header),
        str(output_path),
    )


def _spacing_dhw_from_label(label_path: Path) -> tuple[float, float, float]:
    image = nib.load(str(label_path))
    spacing_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
    return spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]


def _finite_stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    inf_count = sum(1 for v in values if math.isinf(v))
    if finite.size == 0:
        return {"mean": None, "std": None, "finite_count": 0, "inf_count": inf_count}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "finite_count": int(finite.size),
        "inf_count": int(inf_count),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm",
        "component_count_error",
        "false_merge_count",
        "false_break_count",
        "inference_seconds",
    ]
    summary: dict[str, Any] = {"case_count": len(rows)}
    for key in metrics:
        summary[key] = _finite_stats([float(row[key]) for row in rows])

    optional_metric_keys = [
        "uncertainty_error_rate",
        "uncertainty_mean_uncertainty_error",
        "uncertainty_mean_uncertainty_correct",
        "uncertainty_error_auroc",
        "uncertainty_error_auprc",
        "uncertainty_top_uncertainty_error_recall",
        "uncertainty_top_uncertainty_error_rate",
        "uncertainty_top_uncertainty_fraction",
        "calibration_expected_calibration_error",
        "calibration_maximum_calibration_error",
        "calibration_brier_score",
        "calibration_negative_log_likelihood",
        "calibration_mean_confidence",
        "calibration_accuracy",
        "calibration_confidence_gap",
    ]
    for key in optional_metric_keys:
        values = [
            float(row[key])
            for row in rows
            if key in row and row[key] is not None
        ]
        if values:
            summary[key] = _finite_stats(values)
    return summary


def _binary_metrics_row(
    pred: np.ndarray,
    target: np.ndarray,
    spacing_dhw: tuple[float, float, float],
) -> dict[str, float]:
    return compute_binary_metrics(pred > 0, target > 0, spacing_dhw).to_dict()


def _active_multiclass_ids(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
) -> list[int]:
    """只返回本病例真值或预测中实际出现的前景类别。

    若把 ``1..num_classes-1`` 全部纳入宏平均，则真值和预测都缺失的类别会得到
    Dice=1，从而在椎体覆盖范围不同的 CT 中系统性虚高 macro Dice。
    """
    present = {
        int(value)
        for value in np.union1d(np.unique(pred), np.unique(target))
        if int(value) > 0
    }
    invalid = sorted(value for value in present if value >= num_classes)
    if invalid:
        raise ValueError(
            f"prediction/target 存在超出 num_classes={num_classes} 的类别: {invalid}"
        )
    return sorted(present)


def _multiclass_case_rows(
    case_id: str,
    pred: np.ndarray,
    target: np.ndarray,
    spacing_dhw: tuple[float, float, float],
    num_classes: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    class_ids = _active_multiclass_ids(pred, target, num_classes)
    if not class_ids:
        raise ValueError("multiclass 评估至少需要一个实际出现的前景类别")
    per_class = compute_multiclass_metrics(
        pred,
        target,
        class_ids=class_ids,
        spacing_dhw_mm=spacing_dhw,
    )
    keys = ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm")
    macro: dict[str, float] = {}
    for key in keys:
        values = [float(getattr(metric, key)) for metric in per_class.values()]
        macro[key] = float(np.mean(values))

    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        metric = per_class[class_id]
        rows.append(
            {
                "case_id": case_id,
                "class_id": int(class_id),
                "target_present": bool(np.any(target == class_id)),
                "pred_present": bool(np.any(pred == class_id)),
                **metric.to_dict(),
            }
        )
    return macro, rows


def _aggregate_per_class_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["class_id"]), []).append(row)
    result: dict[str, Any] = {}
    metric_keys = ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm")
    for class_id, class_rows in sorted(grouped.items()):
        payload: dict[str, Any] = {
            "case_count": len(class_rows),
            "target_present_count": sum(bool(row["target_present"]) for row in class_rows),
            "pred_present_count": sum(bool(row["pred_present"]) for row in class_rows),
        }
        for key in metric_keys:
            payload[key] = _finite_stats([float(row[key]) for row in class_rows])
        result[str(class_id)] = payload
    return result


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    split: str = "test",
    output_dir: str | Path | None = None,
    case_id: str | None = None,
) -> Path:
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    if split not in {"validation", "test"}:
        raise ValueError("独立评估 split 只能是 validation 或 test")
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]
    model_cfg = config["model"]
    logging_cfg = config.get("logging", {})

    processed_root = _resolve_project_path(data_cfg["processed_root"])
    split_file = _resolve_project_path(data_cfg["split_file"])
    if not processed_root.exists() or not split_file.exists():
        raise FileNotFoundError("processed_root 或 split_file 不存在")

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "experiments" / f"evaluation_{stamp}_{split}"
    else:
        output_path = _resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        split,
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [128, 128, 128]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=int(config.get("seed", 42)),
    )
    if case_id is not None:
        case_id = str(case_id)
        if case_id not in dataset.case_ids:
            raise ValueError(f"case_id={case_id!r} 不属于 split={split!r}")
        dataset.case_ids = [case_id]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    roi = tuple(int(v) for v in infer_cfg.get("roi_size_dhw", data_cfg.get("roi_size_dhw", [128] * 3)))
    sw_batch_size = int(infer_cfg.get("sw_batch_size", 1))
    overlap = float(infer_cfg.get("overlap", 0.5))
    num_classes = int(model_cfg["num_classes"])
    label_mode = str(data_cfg.get("label_mode", "binary"))
    save_predictions = bool(logging_cfg.get("save_predictions", True))
    save_uncertainty = bool(logging_cfg.get("save_uncertainty", False))
    uncertainty_cfg = infer_cfg.get("uncertainty", {})
    evaluate_uncertainty = bool(uncertainty_cfg.get("enabled", False) or save_uncertainty)
    uncertainty_top_percent = float(uncertainty_cfg.get("top_percent", 10.0))
    uncertainty_max_samples = int(uncertainty_cfg.get("metric_max_samples", 500_000))
    calibration_cfg = infer_cfg.get("calibration", {})
    evaluate_calibration = bool(calibration_cfg.get("enabled", False))
    calibration_bins = int(calibration_cfg.get("n_bins", 15))
    calibration_max_samples = int(calibration_cfg.get("metric_max_samples", 500_000))

    rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            case_id = str(batch["case_id"][0])
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            start = time.perf_counter()
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=_model_predictor(model),
                overlap=overlap,
                mode="gaussian",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - start

            pred = logits_to_prediction(logits)[0].cpu().numpy().astype(np.int16)
            target = label[0].cpu().numpy().astype(np.int16)
            case_dir = processed_root / case_id
            label_path = case_dir / "label.nii.gz"
            spacing_dhw = _spacing_dhw_from_label(label_path)

            if label_mode == "binary":
                metric_payload = _binary_metrics_row(pred, target, spacing_dhw)
                metric_payload.update(
                    compute_structural_metrics(pred > 0, target > 0).to_dict()
                )
            else:
                metric_payload, case_class_rows = _multiclass_case_rows(
                    case_id,
                    pred,
                    target,
                    spacing_dhw,
                    num_classes,
                )
                per_class_rows.extend(case_class_rows)
                structural = compute_structural_metrics(pred > 0, target > 0).to_dict()
                metric_payload.update(structural)

            row: dict[str, Any] = {
                "case_id": case_id,
                **metric_payload,
                "inference_seconds": float(inference_seconds),
            }
            entropy: np.ndarray | None = None
            if evaluate_uncertainty:
                entropy = predictive_entropy(logits)[0, 0].cpu().numpy().astype(np.float32)
                uncertainty_metrics = uncertainty_error_metrics(
                    entropy,
                    pred,
                    target,
                    top_percent=uncertainty_top_percent,
                    max_samples=uncertainty_max_samples,
                    seed=int(config.get("seed", 42)),
                )
                for key, value in uncertainty_metrics.to_dict().items():
                    row[f"uncertainty_{key}"] = value
            if evaluate_calibration:
                calibration_metrics = segmentation_calibration_metrics(
                    logits,
                    label,
                    n_bins=calibration_bins,
                    max_samples=calibration_max_samples,
                    seed=int(config.get("seed", 42)),
                )
                for key, value in calibration_metrics.to_dict().items():
                    row[f"calibration_{key}"] = value
            rows.append(row)

            if save_predictions:
                _save_dhw_nifti(
                    pred.astype(np.int16),
                    label_path,
                    output_path / "predictions" / case_id / "prediction.nii.gz",
                )
            if save_uncertainty:
                if entropy is None:
                    entropy = predictive_entropy(logits)[0, 0].cpu().numpy().astype(np.float32)
                _save_dhw_nifti(
                    entropy,
                    label_path,
                    output_path / "uncertainty" / case_id / "predictive_entropy.nii.gz",
                )

    csv_path = output_path / "metrics_per_case.csv"
    fieldnames = [
        "case_id",
        "dice",
        "iou",
        "precision",
        "recall",
        "hd95_mm",
        "assd_mm",
        "pred_components",
        "target_components",
        "component_count_error",
        "false_merge_count",
        "false_break_count",
        "inference_seconds",
        "uncertainty_total_voxels",
        "uncertainty_sampled_voxels",
        "uncertainty_sampling_fraction",
        "uncertainty_error_rate",
        "uncertainty_mean_uncertainty_error",
        "uncertainty_mean_uncertainty_correct",
        "uncertainty_error_auroc",
        "uncertainty_error_auprc",
        "uncertainty_top_percent",
        "uncertainty_top_uncertainty_error_recall",
        "uncertainty_top_uncertainty_error_rate",
        "uncertainty_top_uncertainty_fraction",
        "calibration_total_voxels",
        "calibration_sampled_voxels",
        "calibration_sampling_fraction",
        "calibration_n_bins",
        "calibration_expected_calibration_error",
        "calibration_maximum_calibration_error",
        "calibration_brier_score",
        "calibration_negative_log_likelihood",
        "calibration_mean_confidence",
        "calibration_accuracy",
        "calibration_confidence_gap",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    per_class_csv_path = None
    if per_class_rows:
        per_class_csv_path = output_path / "metrics_per_class.csv"
        with per_class_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "case_id",
                    "class_id",
                    "target_present",
                    "pred_present",
                    "dice",
                    "iou",
                    "precision",
                    "recall",
                    "hd95_mm",
                    "assd_mm",
                ],
            )
            writer.writeheader()
            writer.writerows(per_class_rows)

    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "split": split,
        "case_filter": case_id,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "device": str(device),
        "upstream": upstream_provenance(),
        "metrics": _aggregate_rows(rows),
        "per_class_metrics": _aggregate_per_class_rows(per_class_rows) if per_class_rows else None,
        "metrics_per_class_csv": None if per_class_csv_path is None else str(per_class_csv_path),
        "multiclass_macro_policy": (
            "Per-case macro averages only foreground classes present in target or prediction; "
            "classes absent from both are excluded to avoid inflated scores."
            if label_mode == "multiclass"
            else None
        ),
        "note": "Only independent test split metrics may be used as final test results.",
    }
    (output_path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Evaluate orthopedic CT SegFormer3D checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="只评估当前 validation/test split 内指定病例；用于 CPU 分病例执行，方法和指标不变",
    )
    parser.add_argument(
        "--preflight-mode",
        choices=("formal", "engineering"),
        default="formal",
        help="默认 formal：拒绝 engineering split/未完成人工 QC；工程调试需显式选 engineering",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过保护性预检，仅用于定位代码问题；不得用于论文正式评估",
    )
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    if not args.skip_preflight:
        report = run_preflight(
            config_path,
            mode=args.preflight_mode,
            require_gpu=False,
        )
        print(json.dumps({"preflight": report.to_dict()}, ensure_ascii=False, indent=2))
        if not report.ready:
            raise SystemExit(2)

    output = evaluate_checkpoint(
        config_path,
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        case_id=args.case_id,
    )
    print(f"Evaluation completed: {output}")


if __name__ == "__main__":
    main()
