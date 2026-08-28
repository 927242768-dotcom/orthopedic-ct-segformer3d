"""基于冻结 baseline checkpoint 生成真实模型驱动的 hard-mining guidance。

只处理 train split，避免把 validation/test 信息泄漏进训练采样。
每个病例先固定生成 foreground/background 候选中心，再用同一个 baseline checkpoint
对对应 64^3 patch 计算：
- high_loss: 当前训练 criterion 的真实 patch loss；
- high_uncertainty: predictive entropy 的 patch 均值。

每种策略分别在 foreground/background 候选中选 Top-percent，保存稀疏中心 mask。
训练 Dataset 读取该 mask 后仍先按原 foreground_probability 决定 FG/BG 分支，
仅在该分支内部优先选择 hard center，因此不额外改变前景采样先验。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
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
from src.modeling.uncertainty import predictive_entropy


VALID_STRATEGIES = {"high_loss", "high_uncertainty"}


def _random_background_center(label: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    d, h, w = label.shape
    for _ in range(10_000):
        center = (rng.randrange(d), rng.randrange(h), rng.randrange(w))
        if label[center] == 0:
            return center
    coords = np.argwhere(label == 0)
    if len(coords) == 0:
        raise ValueError("病例不存在 background voxel，无法生成 background hard-mining 候选")
    picked = coords[rng.randrange(len(coords))]
    return tuple(int(v) for v in picked)


def generate_candidate_centers(
    label: np.ndarray,
    *,
    candidate_per_branch: int,
    seed: int,
    case_id: str,
) -> list[dict[str, Any]]:
    if candidate_per_branch < 1:
        raise ValueError("candidate_per_branch 必须 >= 1")
    foreground = np.argwhere(label > 0)
    if len(foreground) == 0:
        raise ValueError(f"{case_id}: 不存在 foreground voxel")

    case_seed = int(seed) ^ int(zlib.crc32(case_id.encode("utf-8")))
    rng = random.Random(case_seed)
    records: list[dict[str, Any]] = []

    for _ in range(candidate_per_branch):
        picked = foreground[rng.randrange(len(foreground))]
        records.append(
            {
                "branch": "foreground",
                "center_dhw": [int(v) for v in picked],
            }
        )
    for _ in range(candidate_per_branch):
        records.append(
            {
                "branch": "background",
                "center_dhw": list(_random_background_center(label, rng)),
            }
        )
    return records


def _crop_centered(
    image: np.ndarray,
    label: np.ndarray,
    center_dhw: tuple[int, int, int],
    roi_dhw: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    _, d, h, w = image.shape
    rd, rh, rw = roi_dhw
    if d < rd or h < rh or w < rw:
        raise ValueError(
            f"当前 guidance 生成要求病例尺寸 >= ROI，得到 image={image.shape[1:]}, roi={roi_dhw}"
        )
    starts: list[int] = []
    for center, current, roi in zip(center_dhw, (d, h, w), roi_dhw):
        start = int(center) - roi // 2
        starts.append(max(0, min(start, current - roi)))
    sd, sh, sw = starts
    return (
        image[:, sd : sd + rd, sh : sh + rh, sw : sw + rw],
        label[sd : sd + rd, sh : sh + rh, sw : sw + rw],
    )


def select_top_centers(
    records: list[dict[str, Any]],
    *,
    strategy: str,
    top_percent: float,
) -> list[dict[str, Any]]:
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"未知 strategy={strategy!r}")
    if not (0.0 < top_percent <= 100.0):
        raise ValueError("top_percent 必须位于 (0,100]")
    score_key = "loss_score" if strategy == "high_loss" else "uncertainty_score"
    selected: list[dict[str, Any]] = []
    for branch in ("foreground", "background"):
        branch_records = [record for record in records if record["branch"] == branch]
        if not branch_records:
            raise ValueError(f"缺少 {branch} candidate")
        keep = max(1, int(math.ceil(len(branch_records) * top_percent / 100.0)))
        ranked = sorted(branch_records, key=lambda record: float(record[score_key]), reverse=True)
        selected.extend(ranked[:keep])
    return selected


def _save_center_mask(
    selected: list[dict[str, Any]],
    *,
    shape_dhw: tuple[int, int, int],
    reference_label: Path,
    output_path: Path,
) -> None:
    mask = np.zeros(shape_dhw, dtype=np.uint8)
    for record in selected:
        center = tuple(int(v) for v in record["center_dhw"])
        mask[center] = 1
    reference = nib.load(str(reference_label))
    mask_xyz = np.transpose(mask, (2, 1, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(mask_xyz, affine=reference.affine, header=reference.header),
        str(output_path),
    )


def generate_guidance(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_root: str | Path,
    strategies: list[str],
    candidate_per_branch: int,
    top_percent: float,
    case_id: str | None = None,
) -> dict[str, Any]:
    config_path = _resolve_project_path(config_path)
    checkpoint_path = _resolve_project_path(checkpoint_path)
    output_root = _resolve_project_path(output_root)
    strategies = [str(strategy).lower() for strategy in strategies]
    invalid = sorted(set(strategies) - VALID_STRATEGIES)
    if invalid:
        raise ValueError(f"不支持的 strategies: {invalid}")
    if not strategies:
        raise ValueError("strategies 不能为空")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    processed_root = _resolve_project_path(data_cfg["processed_root"])
    split_file = _resolve_project_path(data_cfg["split_file"])
    dataset = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "train",
        input_channels=data_cfg.get("input_channels", ["ct_normalized"]),
        roi_size_dhw=data_cfg.get("roi_size_dhw", [64, 64, 64]),
        training=False,
        label_mode=str(data_cfg.get("label_mode", "binary")),
        seed=seed,
    )
    if case_id is not None:
        if case_id not in dataset.case_ids:
            raise ValueError(f"case_id={case_id!r} 不属于 train split")
        dataset.case_ids = [case_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    criterion = build_criterion(config).to(device)
    roi = tuple(int(v) for v in data_cfg.get("roi_size_dhw", [64, 64, 64]))

    case_summaries: list[dict[str, Any]] = []
    with torch.no_grad():
        for index in range(len(dataset)):
            item = dataset[index]
            current_case = str(item["case_id"])
            image = item["image"].numpy()
            label = item["label"].numpy()
            candidates = generate_candidate_centers(
                label,
                candidate_per_branch=candidate_per_branch,
                seed=seed,
                case_id=current_case,
            )
            for record in candidates:
                center = tuple(int(v) for v in record["center_dhw"])
                image_patch, label_patch = _crop_centered(image, label, center, roi)
                image_tensor = torch.from_numpy(np.ascontiguousarray(image_patch))[None].float().to(device)
                target_tensor = torch.from_numpy(np.ascontiguousarray(label_patch))[None].long().to(device)
                logits = model(image_tensor)
                logits = resize_logits_to_target(logits, tuple(int(v) for v in target_tensor.shape[-3:]))
                record["loss_score"] = float(criterion(logits, target_tensor).item())
                record["uncertainty_score"] = float(predictive_entropy(logits).mean().item())
                record["patch_foreground_fraction"] = float((target_tensor > 0).float().mean().item())

            label_path = processed_root / current_case / "label.nii.gz"
            strategy_summaries: dict[str, Any] = {}
            for strategy in strategies:
                selected = select_top_centers(
                    candidates,
                    strategy=strategy,
                    top_percent=top_percent,
                )
                case_dir = output_root / strategy / current_case
                _save_center_mask(
                    selected,
                    shape_dhw=tuple(int(v) for v in label.shape),
                    reference_label=label_path,
                    output_path=case_dir / "hard_centers.nii.gz",
                )
                payload = {
                    "generated_at": datetime.now().isoformat(),
                    "case_id": current_case,
                    "strategy": strategy,
                    "checkpoint": str(checkpoint_path),
                    "config": str(config_path),
                    "candidate_per_branch": int(candidate_per_branch),
                    "top_percent": float(top_percent),
                    "roi_size_dhw": list(roi),
                    "selected_count": len(selected),
                    "selected": selected,
                    "all_candidates": candidates,
                }
                (case_dir / "scores.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                score_key = "loss_score" if strategy == "high_loss" else "uncertainty_score"
                strategy_summaries[strategy] = {
                    "selected_count": len(selected),
                    "selected_score_mean": float(np.mean([r[score_key] for r in selected])),
                    "selected_score_min": float(np.min([r[score_key] for r in selected])),
                    "selected_score_max": float(np.max([r[score_key] for r in selected])),
                }
            case_summaries.append({"case_id": current_case, "strategies": strategy_summaries})

    summary = {
        "generated_at": datetime.now().isoformat(),
        "split": "train",
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "strategies": strategies,
        "candidate_per_branch": int(candidate_per_branch),
        "top_percent": float(top_percent),
        "case_count": len(case_summaries),
        "cases": case_summaries,
        "note": "Guidance is generated only from train split using a frozen validation-selected baseline checkpoint.",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_name = f"summary_{case_id}.json" if case_id else "summary.json"
    (output_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 v13 模型驱动的 hard-mining guidance")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["high_loss", "high_uncertainty"],
        choices=sorted(VALID_STRATEGIES),
    )
    parser.add_argument("--candidate-per-branch", type=int, default=16)
    parser.add_argument("--top-percent", type=float, default=25.0)
    parser.add_argument("--case-id", default=None)
    args = parser.parse_args()
    summary = generate_guidance(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_root=args.output_root,
        strategies=list(args.strategies),
        candidate_per_branch=args.candidate_per_branch,
        top_percent=args.top_percent,
        case_id=args.case_id,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
