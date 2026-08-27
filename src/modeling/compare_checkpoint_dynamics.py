"""比较多个 validation checkpoint 的固定 patch 激活与参数变化。

仅用于 validation 机制诊断：不提供 test split，不执行 optimizer.step。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.diagnostics import foreground_centered_patch, sampled_tensor_stats
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import _resolve_project_path


DEFAULT_ACTIVATION_MODULES = (
    "segformer_encoder.embed_1.patch_embeddings",
    "segformer_encoder.tf_block1.1",
    "segformer_encoder.embed_2.patch_embeddings",
    "segformer_encoder.tf_block2.1",
    "segformer_encoder.embed_3.patch_embeddings",
    "segformer_encoder.tf_block3.1",
    "segformer_encoder.embed_4.patch_embeddings",
    "segformer_encoder.tf_block4.1",
    "segformer_decoder.linear_fuse.0",
    "segformer_decoder.linear_fuse.1",
    "segformer_decoder.linear_pred",
)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _activation_stats(tensor: torch.Tensor, *, seed: int) -> dict[str, Any]:
    detached = tensor.detach().float()
    payload: dict[str, Any] = {
        "shape": list(detached.shape),
        "l2_norm": float(torch.linalg.vector_norm(detached).item()),
    }
    payload.update(sampled_tensor_stats(detached, max_samples=200_000, seed=seed))
    return payload


def activation_diagnostics_on_patch(
    model: torch.nn.Module,
    image_patch: torch.Tensor,
    *,
    module_names: tuple[str, ...] = DEFAULT_ACTIVATION_MODULES,
    seed: int = 42,
) -> dict[str, Any]:
    """在固定 patch 上记录 encoder/decoder/head 的输出激活统计。"""
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise ValueError(f"找不到 activation module: {missing}")

    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is not None:
                captured[name] = tensor.detach()

        return hook

    for name in module_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    head_name = "segformer_decoder.linear_pred"
    if head_name in modules:
        def head_input_hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            tensor = _first_tensor(inputs)
            if tensor is not None:
                captured[f"{head_name}.input"] = tensor.detach()

        handles.append(modules[head_name].register_forward_pre_hook(head_input_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(image_patch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    ordered_names = list(module_names)
    if f"{head_name}.input" in captured:
        ordered_names.insert(-1, f"{head_name}.input")
    return {
        name: _activation_stats(captured[name], seed=seed + index)
        for index, name in enumerate(ordered_names)
        if name in captured
    }


def _state_group(name: str) -> str:
    parts = name.split(".")
    if name.startswith("segformer_encoder.") and len(parts) >= 2:
        return ".".join(parts[:2])
    if name.startswith("segformer_decoder.") and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def state_dict_delta(
    baseline: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """比较两个 checkpoint state_dict 的参数与 BN running-buffer 变化。"""
    common = sorted(set(baseline) & set(candidate))
    parameter_rows: list[dict[str, Any]] = []
    buffer_rows: list[dict[str, Any]] = []
    group_accumulator: dict[str, dict[str, float | int]] = {}

    for name in common:
        left = baseline[name]
        right = candidate[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            continue
        if left.shape != right.shape or not left.dtype.is_floating_point:
            continue
        left_f = left.detach().float().cpu()
        right_f = right.detach().float().cpu()
        delta = right_f - left_f
        baseline_norm = float(torch.linalg.vector_norm(left_f).item())
        delta_norm = float(torch.linalg.vector_norm(delta).item())
        relative_delta = delta_norm / max(baseline_norm, 1e-12)
        row = {
            "name": name,
            "group": _state_group(name),
            "baseline_norm": baseline_norm,
            "delta_norm": delta_norm,
            "relative_delta": relative_delta,
        }
        is_running_buffer = any(
            token in name for token in ("running_mean", "running_var")
        )
        if is_running_buffer:
            buffer_rows.append(row)
            continue
        parameter_rows.append(row)
        group = group_accumulator.setdefault(
            row["group"],
            {"squared_baseline_norm": 0.0, "squared_delta_norm": 0.0, "tensor_count": 0},
        )
        group["squared_baseline_norm"] = float(group["squared_baseline_norm"]) + baseline_norm**2
        group["squared_delta_norm"] = float(group["squared_delta_norm"]) + delta_norm**2
        group["tensor_count"] = int(group["tensor_count"]) + 1

    group_rows: list[dict[str, Any]] = []
    for group_name, values in group_accumulator.items():
        baseline_norm = float(values["squared_baseline_norm"]) ** 0.5
        delta_norm = float(values["squared_delta_norm"]) ** 0.5
        group_rows.append(
            {
                "group": group_name,
                "tensor_count": int(values["tensor_count"]),
                "baseline_norm": baseline_norm,
                "delta_norm": delta_norm,
                "relative_delta": delta_norm / max(baseline_norm, 1e-12),
            }
        )

    return {
        "parameter_groups_by_relative_delta": sorted(
            group_rows, key=lambda row: row["relative_delta"], reverse=True
        ),
        "top_parameters_by_relative_delta": sorted(
            parameter_rows, key=lambda row: row["relative_delta"], reverse=True
        )[:30],
        "batchnorm_running_buffers_by_relative_delta": sorted(
            buffer_rows, key=lambda row: row["relative_delta"], reverse=True
        ),
    }


def _parse_checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--checkpoint 必须使用 label=path")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label or not path.strip():
        raise argparse.ArgumentTypeError("--checkpoint label/path 不能为空")
    return label, Path(path.strip())


def compare_checkpoint_dynamics(
    config_path: str | Path,
    checkpoints: list[tuple[str, Path]],
    *,
    case_id: str,
    output_dir: str | Path,
) -> Path:
    if len(checkpoints) < 2:
        raise ValueError("至少需要两个 checkpoint")
    config_path = _resolve_project_path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ProcessedOrthopedicCTDataset(
        _resolve_project_path(data_cfg["processed_root"]),
        _resolve_project_path(data_cfg["split_file"]),
        "validation",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [64, 64, 64]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=seed,
    )
    case_id = str(case_id)
    if case_id not in dataset.case_ids:
        raise ValueError(f"case_id={case_id!r} 不属于 validation split")
    item = dataset[dataset.case_ids.index(case_id)]
    image = item["image"].unsqueeze(0).to(device)
    label = item["label"].unsqueeze(0).to(device)
    roi = tuple(int(v) for v in data_cfg.get("roi_size_dhw", [64, 64, 64]))
    image_patch, label_patch = foreground_centered_patch(image, label, roi)

    loaded_states: dict[str, dict[str, torch.Tensor]] = {}
    checkpoint_payloads: list[dict[str, Any]] = []
    for label_name, checkpoint_path in checkpoints:
        resolved = _resolve_project_path(checkpoint_path)
        checkpoint = torch.load(resolved, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model = build_orthopedic_segformer3d(config).to(device)
        model.load_state_dict(state_dict, strict=True)
        loaded_states[label_name] = {
            name: value.detach().cpu().clone()
            for name, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
        checkpoint_payloads.append(
            {
                "label": label_name,
                "path": str(resolved),
                "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
                "val_dice": checkpoint.get("val_dice") if isinstance(checkpoint, dict) else None,
                "activation_on_fixed_foreground_patch": activation_diagnostics_on_patch(
                    model,
                    image_patch,
                    seed=seed,
                ),
            }
        )
        del model, checkpoint

    baseline_label = checkpoints[0][0]
    deltas: dict[str, Any] = {}
    for label_name, _ in checkpoints[1:]:
        deltas[f"{baseline_label}__to__{label_name}"] = state_dict_delta(
            loaded_states[baseline_label], loaded_states[label_name]
        )

    output_path = _resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    payload = {
        "created_at": datetime.now().isoformat(),
        "scope": "validation_only_fixed_foreground_patch",
        "config": str(config_path),
        "case_id": case_id,
        "device": str(device),
        "patch_shape": list(image_patch.shape),
        "patch_target_foreground_fraction": float((label_patch > 0).float().mean().item()),
        "checkpoints": checkpoint_payloads,
        "state_deltas": deltas,
        "note": "Mechanism diagnostics only; no optimizer.step and no test split access.",
    }
    output_file = output_path / "checkpoint_dynamics.json"
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Compare validation checkpoint activation/parameter dynamics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=_parse_checkpoint_argument, required=True)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = compare_checkpoint_dynamics(
        args.config,
        args.checkpoint,
        case_id=args.case_id,
        output_dir=args.output_dir,
    )
    print(f"Checkpoint dynamics completed: {output}")


if __name__ == "__main__":
    main()
