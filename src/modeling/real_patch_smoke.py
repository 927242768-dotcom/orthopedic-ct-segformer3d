"""真实处理后病例的单 patch 训练链 smoke test。

该入口只验证：processed NIfTI → Dataset/增强 → SegFormer3D → loss → backward → optimizer.step。
它不做完整 validation/test，不输出可用于论文的性能指标。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import (
    _resolve_project_path,
    build_criterion,
    resize_logits_to_target,
    seed_everything,
)


def run_real_patch_smoke(
    config_path: str | Path,
    processed_root: str | Path,
    split_file: str | Path,
    *,
    split: str = "train",
    roi_size_dhw: tuple[int, int, int] = (36, 36, 36),
    foreground_probability: float = 1.0,
) -> dict[str, object]:
    config_path = _resolve_project_path(config_path)
    processed_root = _resolve_project_path(processed_root)
    split_file = _resolve_project_path(split_file)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    data_cfg = config["data"]
    bone_window_cfg = data_cfg.get("bone_window", {})

    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        split,
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=roi_size_dhw,
        training=True,
        foreground_probability=float(foreground_probability),
        label_mode=str(data_cfg.get("label_mode", "binary")),
        augmentation=config.get("augmentation", {}),
        hu_clip=data_cfg.get("hu_clip", [-1000.0, 2000.0]),
        bone_window_width=float(bone_window_cfg.get("width", 2000.0)),
        seed=seed,
    )
    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    label = sample["label"].unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    label = label.to(device)
    model = build_orthopedic_segformer3d(config).to(device)
    criterion = build_criterion(config).to(device)
    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-2)),
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(image)
    logits = resize_logits_to_target(logits, tuple(label.shape[-3:]))
    loss = criterion(logits, label)
    if not torch.isfinite(loss):
        raise RuntimeError(f"真实 patch loss 非有限值: {float(loss.detach().cpu())}")
    loss.backward()

    grad_sq_sum = 0.0
    grad_tensor_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError("真实 patch backward 出现 NaN/Inf gradient")
        grad_sq_sum += float(torch.sum(grad.float() ** 2).cpu())
        grad_tensor_count += 1
    if grad_tensor_count == 0:
        raise RuntimeError("真实 patch backward 未产生任何梯度")

    optimizer.step()
    channel_ranges = []
    image_cpu = image.detach().cpu().numpy()[0]
    for index, channel_name in enumerate(data_cfg.get("input_channels", ["ct_normalized"])):
        channel_ranges.append(
            {
                "channel": str(channel_name),
                "min": float(image_cpu[index].min()),
                "max": float(image_cpu[index].max()),
                "mean": float(image_cpu[index].mean()),
                "std": float(image_cpu[index].std()),
            }
        )

    result: dict[str, object] = {
        "status": "pass",
        "purpose": "engineering_smoke_only",
        "formal_metric": False,
        "case_id": str(sample["case_id"]),
        "split": split,
        "device": str(device),
        "roi_size_dhw": list(roi_size_dhw),
        "image_shape": list(image.shape),
        "label_shape": list(label.shape),
        "foreground_sampling_probability": float(foreground_probability),
        "foreground_fraction": float((label > 0).float().mean().detach().cpu()),
        "channel_ranges": channel_ranges,
        "loss": float(loss.detach().cpu()),
        "gradient_l2_norm": math.sqrt(grad_sq_sum),
        "gradient_tensor_count": grad_tensor_count,
        "note": "仅证明真实数据单 patch 训练链可运行；loss/随机权重输出不是模型性能。",
    }
    if not np.isfinite(result["gradient_l2_norm"]):
        raise RuntimeError("gradient_l2_norm 非有限值")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run one real processed CT patch training smoke step")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--roi", type=int, nargs=3, default=(36, 36, 36))
    parser.add_argument(
        "--foreground-probability",
        type=float,
        default=1.0,
        help="smoke 默认强制以前景为中心采样；不修改正式训练配置",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = run_real_patch_smoke(
        args.config,
        args.processed_root,
        args.split_file,
        split=args.split,
        roi_size_dhw=tuple(int(v) for v in args.roi),
        foreground_probability=float(args.foreground_probability),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = _resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
