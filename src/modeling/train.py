"""SegFormer3D 骨科 CT baseline / joint-loss 训练入口。

该脚本在真实公开数据完成标准化、split JSON 建立、上游 SegFormer3D 获取后运行。
不会自动下载临床数据，也不会在没有验证的情况下生成论文结果。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.joint_loss import RegionDiceCELoss3D, build_joint_loss
from src.modeling.preflight import run_preflight
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d, upstream_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def build_criterion(config: dict[str, Any]) -> torch.nn.Module:
    loss_cfg = config.get("loss", {})
    loss_type = str(loss_cfg.get("type", "region_dice_ce")).lower()
    if loss_type == "joint_orthopedic":
        return build_joint_loss(config)
    if loss_type in {"region_dice_ce", "dicece", "dice_ce"}:
        return RegionDiceCELoss3D(
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            include_background=bool(loss_cfg.get("include_background", False)),
        )
    raise ValueError(f"未知 loss.type: {loss_type}")


def resize_logits_to_target(logits: torch.Tensor, target_dhw: tuple[int, int, int]) -> torch.Tensor:
    if tuple(logits.shape[-3:]) == tuple(target_dhw):
        return logits
    return F.interpolate(logits, size=target_dhw, mode="trilinear", align_corners=False)


def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits[:, 0]) >= 0.5).long()
    return torch.argmax(logits, dim=1)


def configure_batchnorm_training_mode(
    model: torch.nn.Module,
    *,
    freeze_running_stats: bool,
) -> int:
    """按实验配置冻结 BatchNorm3d running stats，但保留 affine 参数训练。

    调用方应先执行 ``model.train()``。启用后仅把 BatchNorm3d 子模块切到 eval，
    因而 forward 使用已有 running_mean/running_var 且不再更新
    num_batches_tracked；weight/bias 的 requires_grad 不会被修改。
    """
    if not freeze_running_stats:
        return 0

    batchnorm_count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm3d):
            module.eval()
            batchnorm_count += 1
    if batchnorm_count == 0:
        raise RuntimeError("已启用 freeze_batchnorm_running_stats，但模型中没有 BatchNorm3d")
    return batchnorm_count


def summarize_foreground_fractions(fractions: list[float]) -> dict[str, float | int]:
    """汇总一个 epoch 内模型实际看到的 training patch 前景比例。"""
    if not fractions:
        raise ValueError("fractions 不能为空")
    values = np.asarray(fractions, dtype=np.float64)
    return {
        "patch_count": int(values.size),
        "foreground_fraction_mean": float(values.mean()),
        "foreground_fraction_median": float(np.median(values)),
        "foreground_fraction_std": float(values.std()),
        "foreground_fraction_min": float(values.min()),
        "foreground_fraction_max": float(values.max()),
        "foreground_fraction_q10": float(np.quantile(values, 0.10)),
        "foreground_fraction_q25": float(np.quantile(values, 0.25)),
        "foreground_fraction_q75": float(np.quantile(values, 0.75)),
        "foreground_fraction_q90": float(np.quantile(values, 0.90)),
        "foreground_patch_count": int(np.count_nonzero(values > 0.0)),
        "background_patch_count": int(np.count_nonzero(values == 0.0)),
    }


def mean_foreground_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    eps: float = 1e-5,
) -> float:
    pred = pred.long()
    target = target.long()
    class_ids = [1] if num_classes <= 2 else list(range(1, num_classes))
    scores: list[float] = []
    for class_id in class_ids:
        p = pred == class_id
        t = target == class_id
        p_sum = int(p.sum())
        t_sum = int(t.sum())
        if p_sum == 0 and t_sum == 0:
            continue
        intersection = float(torch.logical_and(p, t).sum())
        score = (2.0 * intersection + eps) / (p_sum + t_sum + eps)
        scores.append(float(score))
    return 1.0 if not scores else float(np.mean(scores))


def _autocast_context(device: torch.device, amp_enabled: bool):
    """返回与当前 PyTorch 2.1 兼容的 AMP 上下文。

    CPU 非 AMP 路径必须真正使用 nullcontext；否则 PyTorch 2.1 即使 enabled=false
    也会检查 float16 CPU autocast，并在进入上下文时报错。
    """
    if amp_enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _model_predictor(model: torch.nn.Module):
    def predictor(x: torch.Tensor) -> torch.Tensor:
        logits = model(x)
        return resize_logits_to_target(logits, tuple(x.shape[-3:]))

    return predictor


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    roi_size_dhw: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    case_scores: list[float] = []
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            start = time.perf_counter()
            with _autocast_context(device, amp_enabled):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size_dhw,
                    sw_batch_size=sw_batch_size,
                    predictor=_model_predictor(model),
                    overlap=overlap,
                    mode="gaussian",
                )
            total_time += time.perf_counter() - start
            pred = logits_to_prediction(logits)
            case_scores.append(mean_foreground_dice(pred, label, num_classes))

    return {
        "val_dice": float(np.mean(case_scores)),
        "val_dice_std": float(np.std(case_scores)),
        "val_case_count": float(len(case_scores)),
        "val_inference_seconds_total": float(total_time),
    }


class WarmupCosineRestarts:
    """按 epoch 的线性 warmup + cosine warm restarts。

    该包装器只负责学习率调度，不改变 optimizer 其他状态。
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_epochs: int,
        t0_epochs: int,
        min_lr: float,
    ) -> None:
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs 不能为负数")
        if t0_epochs <= 0:
            raise ValueError("t0_epochs 必须 > 0")
        if min_lr < 0:
            raise ValueError("min_lr 不能为负数")
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(t0_epochs),
            eta_min=float(min_lr),
        )

    def step(self, epoch: int) -> None:
        if epoch <= 0:
            raise ValueError("epoch 必须从 1 开始")
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            factor = float(epoch) / float(self.warmup_epochs)
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * factor
            return
        cosine_epoch = float(epoch - self.warmup_epochs)
        self.cosine.step(cosine_epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_epochs": self.warmup_epochs,
            "base_lrs": self.base_lrs,
            "cosine": self.cosine.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.warmup_epochs = int(state_dict["warmup_epochs"])
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.cosine.load_state_dict(state_dict["cosine"])


def build_scheduler(
    config: dict[str, Any], optimizer: torch.optim.Optimizer
) -> WarmupCosineRestarts | None:
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"", "none", "disabled"}:
        return None
    if scheduler_type != "cosine_annealing_warm_restarts":
        raise ValueError(f"暂不支持 scheduler.type={scheduler_type!r}")
    return WarmupCosineRestarts(
        optimizer,
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 0)),
        t0_epochs=int(scheduler_cfg.get("t0_epochs", 400)),
        min_lr=float(scheduler_cfg.get("min_lr", 0.0)),
    )


def save_checkpoint(
    output_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_dice: float,
    config: dict[str, Any],
    scheduler: WarmupCosineRestarts | None = None,
    best_val_dice: float | None = None,
    epochs_without_improvement: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "val_dice": float(val_dice),
            "best_val_dice": float(val_dice if best_val_dice is None else best_val_dice),
            "epochs_without_improvement": int(epochs_without_improvement),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": config,
            "upstream": upstream_provenance(),
            "git_commit": current_git_commit(),
        },
        output_path,
    )


def load_training_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupCosineRestarts | None,
    expected_config: dict[str, Any],
    device: torch.device,
) -> dict[str, int | float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None and checkpoint_config != expected_config:
        raise ValueError("resume checkpoint 中的 config 与当前训练 config 不一致")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is not None:
        if scheduler is None:
            raise ValueError("checkpoint 含 scheduler_state_dict，但当前 config 未启用 scheduler")
        scheduler.load_state_dict(scheduler_state)

    if checkpoint.get("python_random_state") is not None:
        random.setstate(checkpoint["python_random_state"])
    if checkpoint.get("numpy_random_state") is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if checkpoint.get("torch_rng_state") is not None:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    epoch = int(checkpoint["epoch"])
    return {
        "epoch": epoch,
        "start_epoch": epoch + 1,
        "best_val_dice": float(checkpoint.get("best_val_dice", checkpoint.get("val_dice", -1.0))),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
    }


def train(
    config_path: Path,
    *,
    max_epochs_override: int | None = None,
    resume_checkpoint: Path | None = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    data_cfg = config["data"]
    train_cfg = config["training"]
    infer_cfg = config["inference"]
    model_cfg = config["model"]
    validation_cfg = config.get("validation", {})
    validation_patch_mode = bool(validation_cfg.get("patch_mode", False))

    processed_root = _resolve_project_path(data_cfg["processed_root"])
    split_file = _resolve_project_path(data_cfg["split_file"])
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root 不存在: {processed_root}")
    if not split_file.exists():
        raise FileNotFoundError(f"split_file 不存在: {split_file}")

    resume_path = _resolve_project_path(resume_checkpoint) if resume_checkpoint else None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint 不存在: {resume_path}")
        run_dir = resume_path.parent.parent
        if not (run_dir / "config.yaml").exists() or not (run_dir / "split.json").exists():
            raise FileNotFoundError(f"resume run 缺少 config.yaml 或 split.json: {run_dir}")
    else:
        experiment_name = str(config.get("experiment_name", "experiment"))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "experiments" / f"{stamp}_{experiment_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(config_path, run_dir / "config.yaml")
        shutil.copy2(split_file, run_dir / "split.json")

    metadata_path = run_dir / "run_metadata.json"
    if resume_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "started_at": datetime.now().isoformat(),
            "git_commit": current_git_commit(),
            "device_requested": "cuda" if torch.cuda.is_available() else "cpu",
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "upstream": upstream_provenance(),
            "source_config": str(config_path),
            "source_split": str(split_file),
            "validation_mode": "patch" if validation_patch_mode else "full_volume",
            "validation_patch_is_engineering_proxy": validation_patch_mode,
            "training_patch_sampling_epoch_aware": True,
            "training_patches_per_case": int(train_cfg.get("patches_per_case", 1)),
            "training_foreground_sampling_mode": str(
                data_cfg.get("foreground_sampling_mode", "bernoulli")
            ),
            "training_sampling_stats_logged": True,
            "training_freeze_batchnorm_running_stats": bool(
                train_cfg.get("freeze_batchnorm_running_stats", False)
            ),
            "validation_patch_sampling_fixed_across_epochs": validation_patch_mode,
            "resume_events": [],
        }
    metadata.setdefault("resume_events", [])
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    roi = tuple(int(v) for v in data_cfg.get("roi_size_dhw", [128, 128, 128]))
    input_channels = list(data_cfg.get("input_channels", ["ct_normalized"]))
    label_mode = str(data_cfg.get("label_mode", "binary"))

    bone_window_cfg = data_cfg.get("bone_window", {})
    train_ds = ProcessedOrthopedicCTDataset(
        processed_root,
        split_file,
        "train",
        input_channels=input_channels,
        roi_size_dhw=roi,
        training=True,
        foreground_probability=float(data_cfg.get("foreground_probability", 0.7)),
        patches_per_case=int(train_cfg.get("patches_per_case", 1)),
        foreground_sampling_mode=str(data_cfg.get("foreground_sampling_mode", "bernoulli")),
        label_mode=label_mode,
        augmentation=config.get("augmentation", {}),
        hu_clip=data_cfg.get("hu_clip", [-1000.0, 2000.0]),
        bone_window_width=float(bone_window_cfg.get("width", 2000.0)),
        seed=seed,
    )
    if validation_patch_mode:
        # 仅用于 CPU/工程训练：验证集也取固定大小前景 patch，避免每个 epoch
        # 对 300–600 层整卷 CT 做 sliding-window。正式论文结果必须用 full-volume evaluation。
        val_ds = ProcessedOrthopedicCTDataset(
            processed_root,
            split_file,
            "validation",
            input_channels=input_channels,
            roi_size_dhw=roi,
            training=True,
            foreground_probability=float(validation_cfg.get("foreground_probability", 1.0)),
            label_mode=label_mode,
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
            bone_window_width=float(bone_window_cfg.get("width", 2000.0)),
            seed=seed,
        )
    else:
        val_ds = ProcessedOrthopedicCTDataset(
            processed_root,
            split_file,
            "validation",
            input_channels=input_channels,
            roi_size_dhw=roi,
            training=False,
            label_mode=label_mode,
            seed=seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(2, int(train_cfg.get("num_workers", 4)))),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_orthopedic_segformer3d(config).to(device)
    criterion = build_criterion(config).to(device)

    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = build_scheduler(config, optimizer)

    max_epochs = int(max_epochs_override or train_cfg.get("epochs", 800))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    # PyTorch 2.1 使用 torch.cuda.amp.GradScaler；CPU 路径保持 disabled。
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    accumulate = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    patience = int(train_cfg.get("early_stopping_patience", 100))

    best_dice = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    if resume_path is not None:
        resume_state = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config=config,
            device=device,
        )
        start_epoch = int(resume_state["start_epoch"])
        best_dice = float(resume_state["best_val_dice"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        metadata["resume_events"].append(
            {
                "resumed_at": datetime.now().isoformat(),
                "checkpoint": str(resume_path),
                "checkpoint_epoch": int(resume_state["epoch"]),
                "target_max_epochs": max_epochs,
                "git_commit": current_git_commit(),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    history_csv = run_dir / "history.csv"
    sampling_stats_csv = run_dir / "sampling_stats.csv"
    train_log = run_dir / "train.log"
    append_history = resume_path is not None and history_csv.exists()
    append_sampling_stats = resume_path is not None and sampling_stats_csv.exists()
    history_mode = "a" if append_history else "w"
    sampling_mode = "a" if append_sampling_stats else "w"
    last_epoch = start_epoch - 1
    with (
        history_csv.open(history_mode, encoding="utf-8", newline="") as f,
        sampling_stats_csv.open(sampling_mode, encoding="utf-8", newline="") as sampling_f,
    ):
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_dice",
                "val_dice_std",
                "val_inference_seconds_total",
                "lr",
            ],
        )
        if not append_history:
            writer.writeheader()
        sampling_writer = csv.DictWriter(
            sampling_f,
            fieldnames=[
                "epoch",
                "patch_count",
                "foreground_fraction_mean",
                "foreground_fraction_median",
                "foreground_fraction_std",
                "foreground_fraction_min",
                "foreground_fraction_max",
                "foreground_fraction_q10",
                "foreground_fraction_q25",
                "foreground_fraction_q75",
                "foreground_fraction_q90",
                "foreground_patch_count",
                "background_patch_count",
            ],
        )
        if not append_sampling_stats:
            sampling_writer.writeheader()

        for epoch in range(start_epoch, max_epochs + 1):
            if scheduler is not None:
                scheduler.step(epoch)
            train_ds.set_epoch(epoch)
            model.train()
            configure_batchnorm_training_mode(
                model,
                freeze_running_stats=bool(
                    train_cfg.get("freeze_batchnorm_running_stats", False)
                ),
            )
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            batch_count = 0
            epoch_foreground_fractions: list[float] = []

            for step, batch in enumerate(train_loader, start=1):
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                batch_foreground_fractions = (
                    (label > 0).flatten(start_dim=1).float().mean(dim=1).detach().cpu().tolist()
                )
                epoch_foreground_fractions.extend(float(v) for v in batch_foreground_fractions)

                with _autocast_context(device, amp_enabled):
                    logits = model(image)
                    logits = resize_logits_to_target(logits, tuple(label.shape[-3:]))
                    loss = criterion(logits, label) / accumulate

                scaler.scale(loss).backward()
                if step % accumulate == 0 or step == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.detach().cpu()) * accumulate
                batch_count += 1

            train_loss = running_loss / max(batch_count, 1)
            sampling_row = {
                "epoch": epoch,
                **summarize_foreground_fractions(epoch_foreground_fractions),
            }
            sampling_writer.writerow(sampling_row)
            sampling_f.flush()
            val_result = validate(
                model,
                val_loader,
                device=device,
                num_classes=int(model_cfg["num_classes"]),
                roi_size_dhw=tuple(int(v) for v in infer_cfg.get("roi_size_dhw", roi)),
                sw_batch_size=int(infer_cfg.get("sw_batch_size", 1)),
                overlap=float(infer_cfg.get("overlap", 0.5)),
                amp_enabled=amp_enabled,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_result["val_dice"],
                "val_dice_std": val_result["val_dice_std"],
                "val_inference_seconds_total": val_result["val_inference_seconds_total"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            log_line = json.dumps(row, ensure_ascii=False)
            print(log_line)
            with train_log.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")

            if val_result["val_dice"] > best_dice:
                best_dice = val_result["val_dice"]
                epochs_without_improvement = 0
                save_checkpoint(
                    run_dir / "checkpoint" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_dice=best_dice,
                    config=config,
                    scheduler=scheduler,
                    best_val_dice=best_dice,
                    epochs_without_improvement=epochs_without_improvement,
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                run_dir / "checkpoint" / "last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_dice=float(val_result["val_dice"]),
                config=config,
                scheduler=scheduler,
                best_val_dice=best_dice,
                epochs_without_improvement=epochs_without_improvement,
            )
            last_epoch = epoch

            if epochs_without_improvement >= patience:
                print(f"Early stopping: {patience} epochs without validation Dice improvement.")
                break

    summary = {
        "finished_at": datetime.now().isoformat(),
        "best_val_dice": best_dice,
        "last_epoch": last_epoch,
        "target_max_epochs": max_epochs,
        "epochs_without_improvement": epochs_without_improvement,
        "resumed": resume_path is not None,
        "run_dir": str(run_dir),
        "validation_mode": "patch" if validation_patch_mode else "full_volume",
        "training_patches_per_case": int(train_cfg.get("patches_per_case", 1)),
        "note": (
            "Engineering patch-validation proxy only; full-volume validation/test evaluation is required for formal results."
            if validation_patch_mode
            else "Validation metric only; test set evaluation must be run separately."
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_dir


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Train orthopedic CT SegFormer3D")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="从同一 run 的 checkpoint/last.pt 继续训练；--max-epochs 表示续训后的总目标 epoch",
    )
    parser.add_argument(
        "--preflight-mode",
        choices=("formal", "engineering"),
        default="formal",
        help="默认 formal：要求正式 split、人工 QC 与 CUDA；仅工程调试时显式选 engineering",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="显式允许 CPU 正式训练；不会降低数据/QC/split/task 的其它 formal 检查",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过保护性预检，仅用于定位代码问题；不得用于论文正式 run",
    )
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    if not args.skip_preflight:
        report = run_preflight(
            config_path,
            mode=args.preflight_mode,
            require_gpu=False if args.allow_cpu else None,
        )
        print(json.dumps({"preflight": report.to_dict()}, ensure_ascii=False, indent=2))
        if not report.ready:
            raise SystemExit(2)
    run_dir = train(
        config_path,
        max_epochs_override=args.max_epochs,
        resume_checkpoint=args.resume,
    )
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
