import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d
from src.modeling.train import build_criterion, resize_logits_to_target


def _tiny_config() -> dict:
    return {
        "model": {
            "in_channels": 1,
            "sr_ratios": [4, 2, 1, 1],
            "embed_dims": [8, 16, 32, 64],
            "patch_kernel_size": [7, 3, 3, 3],
            "patch_stride": [4, 2, 2, 2],
            "patch_padding": [3, 1, 1, 1],
            "mlp_ratios": [2, 2, 2, 2],
            "num_heads": [1, 1, 2, 4],
            "depths": [1, 1, 1, 1],
            "decoder_head_embedding_dim": 32,
            "num_classes": 2,
            "decoder_dropout": 0.0,
        },
        "loss": {
            "type": "region_dice_ce",
            "include_background": False,
        },
    }


def _write_processed_case(root: Path, case_id: str) -> None:
    case_dir = root / case_id
    case_dir.mkdir(parents=True)

    image = np.zeros((36, 36, 36), dtype=np.float32)
    image[6:30, 6:30, 6:30] = 0.75
    label = np.zeros((36, 36, 36), dtype=np.int16)
    label[11:25, 11:25, 11:25] = 1

    affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(image, affine), str(case_dir / "image_normalized.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), str(case_dir / "label.nii.gz"))


def test_dataset_model_loss_optimizer_end_to_end_smoke(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    _write_processed_case(processed_root, "case_train")
    split_file = tmp_path / "split.json"
    split_file.write_text(
        json.dumps(
            {
                "train": ["case_train"],
                "validation": ["case_train"],
                "test": ["case_train"],
            }
        ),
        encoding="utf-8",
    )

    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "train",
        roi_size_dhw=(36, 36, 36),
        training=True,
        label_mode="binary",
        seed=123,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))

    config = _tiny_config()
    model = build_orthopedic_segformer3d(config)
    criterion = build_criterion(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["image"])
    logits = resize_logits_to_target(logits, tuple(batch["label"].shape[-3:]))
    loss = criterion(logits, batch["label"])
    loss.backward()
    optimizer.step()

    assert logits.shape == (1, 2, 36, 36, 36)
    assert torch.isfinite(loss)
    assert float(loss.detach()) > 0.0
