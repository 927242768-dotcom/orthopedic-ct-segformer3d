from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.diagnostics import batchnorm_batch_stats_mode
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import logits_to_prediction, resize_logits_to_target


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_b = pred.bool()
    target_b = target.bool()
    intersection = float((pred_b & target_b).sum().item())
    pred_count = float(pred_b.sum().item())
    target_count = float(target_b.sum().item())
    eps = 1e-8
    return {
        "dice": (2.0 * intersection + eps) / (pred_count + target_count + eps),
        "precision": (intersection + eps) / (pred_count + eps),
        "recall": (intersection + eps) / (target_count + eps),
        "prediction_foreground_fraction": float(pred_b.float().mean().item()),
        "target_foreground_fraction": float(target_b.float().mean().item()),
        "prediction_to_target_foreground_ratio": (pred_count + eps) / (target_count + eps),
    }


def _evaluate_mode(model: torch.nn.Module, dataset: ProcessedOrthopedicCTDataset, mode: str) -> dict:
    if mode not in {"running", "batch", "train"}:
        raise ValueError(mode)

    if mode == "train":
        model.train()
        context = nullcontext()
    else:
        model.eval()
        context = batchnorm_batch_stats_mode(model) if mode == "batch" else nullcontext()

    per_case = []
    with context, torch.no_grad():
        for index in range(len(dataset.case_ids)):
            sample = dataset[index]
            image = sample["image"].unsqueeze(0)
            target = sample["label"].unsqueeze(0)
            logits = model(image)
            logits = resize_logits_to_target(logits, tuple(int(v) for v in target.shape[-3:]))
            pred = logits_to_prediction(logits)
            per_case.append({"case_id": sample["case_id"], **_metrics(pred, target)})

    keys = [
        "dice",
        "precision",
        "recall",
        "prediction_foreground_fraction",
        "target_foreground_fraction",
        "prediction_to_target_foreground_ratio",
    ]
    mean = {key: float(sum(float(item[key]) for item in per_case) / len(per_case)) for key in keys}
    return {"mode": mode, "mean": mean, "per_case": per_case}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    # 与 train.py 的 seed_everything(seed) 对齐，确保旧 checkpoint 也能复现
    # num_workers=0 时历史版本 Dataset 基于 torch.initial_seed() 选择的 patch。
    torch.manual_seed(seed)
    data_cfg = config["data"]
    val_cfg = config.get("validation", {})
    dataset = ProcessedOrthopedicCTDataset(
        Path(data_cfg["processed_root"]),
        Path(data_cfg["split_file"]),
        "validation",
        input_channels=list(data_cfg.get("input_channels", ["ct_normalized_u16"])),
        roi_size_dhw=tuple(int(v) for v in data_cfg.get("roi_size_dhw", [64, 64, 64])),
        training=True,
        foreground_probability=float(val_cfg.get("foreground_probability", 1.0)),
        patches_per_case=1,
        foreground_sampling_mode=str(val_cfg.get("foreground_sampling_mode", "fixed_per_case")),
        label_mode=str(data_cfg.get("label_mode", "binary")),
        augmentation={"enabled": False},
        hu_clip=data_cfg.get("hu_clip", [-1000.0, 2000.0]),
        seed=int(config.get("seed", 42)),
    )
    dataset.set_epoch(0)

    model = build_orthopedic_segformer3d(config).cpu()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    result = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_dice": float(checkpoint.get("val_dice", -1.0)),
        "case_count": len(dataset.case_ids),
        "results": [
            _evaluate_mode(model, dataset, "running"),
            _evaluate_mode(model, dataset, "batch"),
            _evaluate_mode(model, dataset, "train"),
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
