from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.joint_loss import RegionDiceCELoss3D
from src.modeling.train import logits_to_prediction, mean_foreground_dice


def _build_fixed_patch(config: dict) -> torch.Tensor:
    data_cfg = config["data"]
    val_cfg = config.get("validation", {})
    ds = ProcessedOrthopedicCTDataset(
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
    ds.set_epoch(0)
    return ds[0]["label"].unsqueeze(0)


def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p = pred.bool()
    t = target.bool()
    intersection = float((p & t).sum().item())
    p_count = float(p.sum().item())
    t_count = float(t.sum().item())
    eps = 1e-8
    return {
        "dice": (2.0 * intersection + eps) / (p_count + t_count + eps),
        "precision": (intersection + eps) / (p_count + eps),
        "recall": (intersection + eps) / (t_count + eps),
        "fg_ratio": (p_count + eps) / (t_count + eps),
    }


def optimize_lowres_logits(
    target: torch.Tensor,
    lowres_size: int,
    *,
    steps: int,
    lr: float,
) -> dict[str, float | int]:
    num_classes = 2
    lowres = torch.nn.Parameter(torch.zeros(1, num_classes, lowres_size, lowres_size, lowres_size))
    optimizer = torch.optim.Adam([lowres], lr=lr)
    criterion = RegionDiceCELoss3D(dice_weight=1.0, ce_weight=1.0, include_background=False)

    best = {"dice": -1.0, "step": 0}
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            lowres,
            size=tuple(int(v) for v in target.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        )
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred = logits_to_prediction(logits)
            dice = mean_foreground_dice(pred, target, num_classes)
            if dice > float(best["dice"]):
                best = {"dice": float(dice), "step": step, "loss": float(loss.item())}

    with torch.no_grad():
        logits = F.interpolate(
            lowres,
            size=tuple(int(v) for v in target.shape[-3:]),
            mode="trilinear",
            align_corners=False,
        )
        pred = logits_to_prediction(logits)
        final_metrics = _binary_metrics(pred, target)
    return {
        "lowres_size": lowres_size,
        "steps": steps,
        "best_dice": float(best["dice"]),
        "best_step": int(best["step"]),
        "best_loss": float(best.get("loss", float("nan"))),
        "final_dice": float(final_metrics["dice"]),
        "final_precision": float(final_metrics["precision"]),
        "final_recall": float(final_metrics["recall"]),
        "final_fg_ratio": float(final_metrics["fg_ratio"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    target = _build_fixed_patch(config)
    result = {
        "config": str(args.config),
        "target_shape": list(target.shape),
        "target_foreground_fraction": float((target > 0).float().mean().item()),
        "results": [
            optimize_lowres_logits(target, int(size), steps=args.steps, lr=args.lr)
            for size in args.sizes
        ],
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
