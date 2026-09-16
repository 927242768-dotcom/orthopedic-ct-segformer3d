"""检查 frozen-base refined SegFormer 在训练前是否严格继承 coarse checkpoint。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.metrics import binary_overlap_metrics
from src.modeling.model_factory import build_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]
    validation_cfg = config.get("validation", {})
    dataset = ProcessedOrthopedicCTDataset(
        data_cfg["processed_root"],
        data_cfg["split_file"],
        "validation",
        input_channels=data_cfg.get("input_channels", ["ct_normalized_u16"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [64, 64, 64]),
        training=True,
        foreground_probability=float(validation_cfg.get("foreground_probability", 1.0)),
        foreground_sampling_mode=str(
            validation_cfg.get("foreground_sampling_mode", "fixed_per_case")
        ),
        label_mode=str(data_cfg.get("label_mode", "binary")),
        augmentation={
            "enabled": True,
            "geometric": {
                "random_flip": False,
                "random_rotate_deg": 0.0,
                "random_scale_range": [1.0, 1.0],
                "transform_probability": 0.0,
            },
            "intensity": {
                "probability": 0.0,
                "gamma_range": [1.0, 1.0],
                "gaussian_noise_std_range": [0.0, 0.0],
                "hu_shift_range": [0.0, 0.0],
            },
            "hard_sampling": {"enabled": False},
        },
        hu_clip=data_cfg.get("hu_clip", [-1000.0, 2000.0]),
        seed=int(config.get("seed", 42)),
    )
    sample = dataset[0]
    image = sample["image"].unsqueeze(0)
    target = sample["label"].numpy()

    model = build_segmentation_model(config).eval()
    with torch.no_grad():
        prediction = model(image).argmax(dim=1)[0].cpu().numpy()

    dice, iou, precision, recall = binary_overlap_metrics(prediction, target)
    target_fg = float(np.asarray(target, dtype=bool).sum())
    pred_fg = float(np.asarray(prediction, dtype=bool).sum())
    print(
        {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "prediction_to_target_foreground_ratio": pred_fg / max(1.0, target_fg),
        }
    )


if __name__ == "__main__":
    main()
