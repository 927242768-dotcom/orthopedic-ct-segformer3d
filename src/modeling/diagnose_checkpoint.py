"""仅用于 validation 的 SegFormer3D checkpoint 稳定性诊断入口。

该入口故意不提供 test split 参数，避免把独立 test 用于调参/诊断。
诊断只读取参数、执行 inference/backward，不执行 optimizer.step。
"""

from __future__ import annotations

import argparse
import gc
import json
from contextlib import nullcontext
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.diagnostics import (
    batchnorm_batch_stats_mode,
    batchnorm_running_diagnostics,
    head_gradient_diagnostics,
    head_parameter_diagnostics,
    logits_probability_diagnostics,
    region_loss_diagnostics,
)
from src.modeling.preflight import run_preflight
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d, upstream_provenance
from src.modeling.train import PROJECT_ROOT, _model_predictor, _resolve_project_path


def diagnose_validation_checkpoint(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    case_id: str,
    output_dir: str | Path | None = None,
    max_samples: int = 500_000,
    bn_mode: str = "running",
) -> Path:
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if max_samples <= 0:
        raise ValueError("max_samples 必须 > 0")
    if bn_mode not in {"running", "batch"}:
        raise ValueError("bn_mode 只能是 running 或 batch")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    infer_cfg = config["inference"]
    model_cfg = config["model"]
    loss_cfg = config.get("loss", {})
    seed = int(config.get("seed", 42))

    processed_root = _resolve_project_path(data_cfg["processed_root"])
    split_file = _resolve_project_path(data_cfg["split_file"])
    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "validation",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [128, 128, 128]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=seed,
    )
    case_id = str(case_id)
    if case_id not in dataset.case_ids:
        raise ValueError(f"case_id={case_id!r} 不属于 validation split")
    dataset.case_ids = [case_id]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "experiments" / f"diagnostics_{stamp}_{case_id}"
    else:
        output_path = _resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    roi = tuple(
        int(value)
        for value in infer_cfg.get("roi_size_dhw", data_cfg.get("roi_size_dhw", [128] * 3))
    )
    gradient_roi = tuple(int(value) for value in data_cfg.get("roi_size_dhw", [64] * 3))
    sw_batch_size = int(infer_cfg.get("sw_batch_size", 1))
    overlap = float(infer_cfg.get("overlap", 0.5))
    num_classes = int(model_cfg["num_classes"])
    if num_classes != 2:
        raise ValueError("当前稳定性 diagnostics 只用于 num_classes=2 的 CT-only baseline")

    head_parameters = head_parameter_diagnostics(model, num_classes=num_classes)
    batchnorm_parameters = batchnorm_running_diagnostics(model)
    case_payloads: list[dict[str, Any]] = []

    for batch in loader:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        start = time.perf_counter()
        bn_context = batchnorm_batch_stats_mode(model) if bn_mode == "batch" else nullcontext()
        with bn_context, torch.no_grad():
            logits = sliding_window_inference(
                inputs=image,
                roi_size=roi,
                sw_batch_size=sw_batch_size,
                predictor=_model_predictor(model),
                overlap=overlap,
                mode="gaussian",
            )
        inference_seconds = time.perf_counter() - start

        probability_payload = logits_probability_diagnostics(
            logits,
            label,
            max_samples=max_samples,
            seed=seed,
        )
        loss_payload = region_loss_diagnostics(
            logits,
            label,
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
        )
        del logits
        gc.collect()

        gradient_payload = head_gradient_diagnostics(
            model,
            image,
            label,
            config,
            roi_size_dhw=gradient_roi,
        )
        case_payloads.append(
            {
                "case_id": str(batch["case_id"][0]),
                "inference_seconds": float(inference_seconds),
                "logits_and_probabilities": probability_payload,
                "region_loss": loss_payload,
                "head_gradient_on_fixed_foreground_patch": gradient_payload,
            }
        )
        del image, label
        gc.collect()

    payload = {
        "diagnosed_at": datetime.now().isoformat(),
        "scope": "validation_only",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_val_dice": checkpoint.get("val_dice") if isinstance(checkpoint, dict) else None,
        "device": str(device),
        "max_samples_per_distribution": int(max_samples),
        "upstream": upstream_provenance(),
        "inference_path": {
            "dataset_training_mode": False,
            "input_channels": list(data_cfg.get("input_channels", ["ct_normalized"])),
            "sliding_window_roi_dhw": list(roi),
            "sliding_window_overlap": overlap,
            "sliding_window_mode": "gaussian",
            "patch_predictor_resize": "resize_logits_to_target(trilinear, align_corners=False)",
            "training_resize": "resize_logits_to_target(trilinear, align_corners=False)",
            "binary_prediction": "argmax over 2 logits; equivalent foreground probability >= 0.5",
        },
        "head_parameters": head_parameters,
        "batchnorm_running_statistics": batchnorm_parameters,
        "bn_inference_mode": bn_mode,
        "cases": case_payloads,
        "note": (
            "Diagnostics only: no optimizer.step was executed. Values are validation evidence and must not be written as final test performance."
        ),
    }
    output_file = output_path / "diagnostics.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Diagnose SegFormer3D validation checkpoint logits/loss/head dynamics"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument(
        "--bn-mode",
        choices=("running", "batch"),
        default="running",
        help="running=标准 eval running stats；batch=仅诊断时用当前 sliding-window batch stats，且不更新 checkpoint",
    )
    parser.add_argument(
        "--preflight-mode",
        choices=("formal", "engineering"),
        default="formal",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    if not args.skip_preflight:
        report = run_preflight(config_path, mode=args.preflight_mode, require_gpu=False)
        print(json.dumps({"preflight": report.to_dict()}, ensure_ascii=False, indent=2))
        if not report.ready:
            raise SystemExit(2)

    output = diagnose_validation_checkpoint(
        config_path,
        args.checkpoint,
        case_id=args.case_id,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        bn_mode=args.bn_mode,
    )
    print(f"Validation diagnostics completed: {output}")


if __name__ == "__main__":
    main()
