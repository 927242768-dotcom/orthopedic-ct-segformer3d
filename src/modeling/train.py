"""SegFormer3D 骨科 CT baseline / joint-loss 训练入口。

该脚本在真实公开数据完成标准化、split JSON 建立、上游 SegFormer3D 获取后运行。
不会自动下载临床数据，也不会在没有验证的情况下生成论文结果。
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
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
from src.modeling.joint_loss import (
    RegionDiceCELoss3D,
    RegionDiceWeightedCELoss3D,
    RegionTverskyCELoss3D,
    build_joint_loss,
)
from src.modeling.metrics import binary_overlap_metrics, compute_structural_metrics
from src.modeling.model_factory import build_segmentation_model, model_provenance
from src.modeling.postprocessing import postprocess_prediction
from src.modeling.preflight import run_preflight


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RunAlreadyActiveError(RuntimeError):
    """同一个实验 run 已被另一个训练进程占用。"""


def _process_is_alive(pid: int) -> bool:
    """跨平台检查 PID 是否仍存在；只做存活判断，不发送终止信号。"""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        output = result.stdout.strip()
        return bool(output) and "No tasks are running" not in output and f'"{pid}"' in output
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_run_lock(run_dir: Path) -> Path:
    """原子获取 run 级训练锁；活跃锁存在时禁止第二个训练实例写同一目录。"""
    lock_path = run_dir / "RUNNING.lock"
    payload = {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "started_at": datetime.now().isoformat(),
    }
    for _attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing: dict[str, Any] = {}
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                pass
            existing_pid = int(existing.get("pid", -1)) if existing else -1
            if _process_is_alive(existing_pid):
                raise RunAlreadyActiveError(
                    f"run 已被训练进程 PID={existing_pid} 占用: {run_dir}"
                )
            # 只有确认旧 PID 不存活时才清理 stale lock，然后原子重试一次。
            lock_path.unlink(missing_ok=True)
            continue
        try:
            os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        return lock_path
    raise RunAlreadyActiveError(f"无法获取 run 训练锁: {run_dir}")


def release_run_lock(lock_path: Path) -> None:
    """仅释放属于当前 PID 的锁，避免误删另一个训练实例的锁。"""
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return
    if int(payload.get("pid", -1)) == os.getpid():
        lock_path.unlink(missing_ok=True)


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


def find_latest_compatible_last_checkpoint(
    config: dict[str, Any],
    *,
    experiments_root: Path | None = None,
) -> Path | None:
    """查找最近一个与当前 config 完全一致的 ``checkpoint/last.pt``。

    自动续训只允许复用完整 config 一致的 run，避免把不同消融实验错误拼接。
    """
    root = experiments_root or (PROJECT_ROOT / "experiments")
    if not root.exists():
        return None

    compatible: list[Path] = []
    for checkpoint_path in root.glob("*/checkpoint/last.pt"):
        run_config_path = checkpoint_path.parent.parent / "config.yaml"
        if not run_config_path.exists():
            continue
        try:
            run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            continue
        if run_config == config:
            compatible.append(checkpoint_path)

    if not compatible:
        return None
    return max(compatible, key=lambda path: path.stat().st_mtime_ns)


def select_validation_case_subset(case_ids: list[str], max_cases: int | None) -> list[str]:
    """从完整 validation split 中固定抽取均匀分布的病例用于逐 epoch 快速验证。

    ``max_cases=None`` 时保持完整 validation；子集只影响训练内 checkpoint selector，
    完整 validation 仍可用 ``evaluate.py --split validation`` 独立运行。
    """
    if max_cases is None:
        return list(case_ids)
    max_cases = int(max_cases)
    if max_cases <= 0:
        raise ValueError("validation.max_cases 必须 > 0")
    if max_cases >= len(case_ids):
        return list(case_ids)
    indices = np.linspace(0, len(case_ids) - 1, num=max_cases, dtype=int)
    return [case_ids[int(index)] for index in indices]


def format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--:--:--"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_training_status(
    *,
    phase: str,
    epoch: int,
    max_epochs: int,
    step: int | None,
    total_steps: int | None,
    train_loss: float | None,
    val_dice: float | None,
    best_val_dice: float,
    lr: float,
    completed_training_seconds: float,
    current_epoch_seconds: float = 0.0,
    message: str | None = None,
) -> dict[str, Any]:
    """构造控制台与 ``training_status.json`` 共用的训练状态。"""
    if max_epochs <= 0:
        raise ValueError("max_epochs 必须 > 0")

    if phase == "training" and step is not None and total_steps:
        epoch_fraction = min(1.0, max(0.0, float(step) / float(total_steps)))
        completed_epoch_fraction = max(0.0, float(epoch - 1) + epoch_fraction)
    else:
        completed_epoch_fraction = max(0.0, float(epoch))

    progress = min(1.0, completed_epoch_fraction / float(max_epochs))
    elapsed = max(0.0, completed_training_seconds + current_epoch_seconds)
    if completed_epoch_fraction > 0.0 and elapsed > 0.0:
        seconds_per_epoch = elapsed / completed_epoch_fraction
        eta_seconds: float | None = max(
            0.0, seconds_per_epoch * (float(max_epochs) - completed_epoch_fraction)
        )
    else:
        eta_seconds = None

    return {
        "updated_at": datetime.now().isoformat(),
        "phase": phase,
        "epoch": int(epoch),
        "max_epochs": int(max_epochs),
        "step": None if step is None else int(step),
        "total_steps": None if total_steps is None else int(total_steps),
        "train_loss": None if train_loss is None else float(train_loss),
        "val_dice": None if val_dice is None else float(val_dice),
        "best_val_dice": float(best_val_dice),
        "lr": float(lr),
        "elapsed_seconds": float(elapsed),
        "eta_seconds": None if eta_seconds is None else float(eta_seconds),
        "progress": float(progress),
        "progress_percent": float(progress * 100.0),
        "message": message,
    }


def write_training_status(path: Path, status: dict[str, Any]) -> None:
    """原子更新状态文件，避免中断时留下半截 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp_path.replace(path)


def write_training_heartbeat(
    path: Path,
    *,
    phase: str,
    epoch: int,
    step: int | None,
    total_steps: int | None,
) -> None:
    """独立写训练进程心跳。

    心跳与训练指标状态分离：即使一个 CPU step 很慢、``training_status.json``
    很久没有变化，只要该文件仍每几秒刷新，监控界面就能确认训练进程仍存活。
    """
    write_training_status(
        path,
        {
            "updated_at": datetime.now().isoformat(),
            "pid": os.getpid(),
            "phase": str(phase),
            "epoch": int(epoch),
            "step": None if step is None else int(step),
            "total_steps": None if total_steps is None else int(total_steps),
        },
    )


def format_training_status_line(status: dict[str, Any]) -> str:
    phase = str(status["phase"]).upper()
    step_text = ""
    if status.get("step") is not None and status.get("total_steps") is not None:
        step_text = f" step {status['step']}/{status['total_steps']}"
    train_loss = status.get("train_loss")
    val_dice = status.get("val_dice")
    loss_text = "--" if train_loss is None else f"{float(train_loss):.6f}"
    val_text = "--" if val_dice is None else f"{float(val_dice):.6f}"
    eta_text = format_duration(status.get("eta_seconds"))
    return (
        f"[{phase}] epoch {status['epoch']}/{status['max_epochs']}{step_text} | "
        f"train loss={loss_text} | val Dice={val_text} | "
        f"best Dice={float(status['best_val_dice']):.6f} | lr={float(status['lr']):.3e} | "
        f"运行={format_duration(float(status['elapsed_seconds']))} | "
        f"进度={float(status['progress_percent']):.1f}% | ETA={eta_text}"
    )


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
    if loss_type in {"region_tversky_ce", "tversky_ce"}:
        return RegionTverskyCELoss3D(
            tversky_weight=float(loss_cfg.get("tversky_weight", 1.0)),
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            alpha=float(loss_cfg.get("alpha", 0.65)),
            beta=float(loss_cfg.get("beta", 0.35)),
            include_background=bool(loss_cfg.get("include_background", False)),
        )
    if loss_type in {"region_dice_weighted_ce", "dice_weighted_ce"}:
        return RegionDiceWeightedCELoss3D(
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            background_weight=float(loss_cfg.get("background_weight", 1.25)),
            foreground_weight=float(loss_cfg.get("foreground_weight", 1.0)),
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


def should_freeze_batchnorm_running_stats(
    train_cfg: dict[str, Any],
    *,
    epoch: int,
) -> bool:
    """解析当前 epoch 是否应冻结 BatchNorm3d running statistics。"""
    if epoch <= 0:
        raise ValueError("epoch 必须从 1 开始")

    freeze_always = bool(train_cfg.get("freeze_batchnorm_running_stats", False))
    freeze_from_raw = train_cfg.get("freeze_batchnorm_running_stats_from_epoch")
    if freeze_from_raw is None:
        return freeze_always
    if isinstance(freeze_from_raw, bool):
        raise ValueError("freeze_batchnorm_running_stats_from_epoch 必须是正整数 epoch")

    freeze_from = int(freeze_from_raw)
    if freeze_from <= 0:
        raise ValueError("freeze_batchnorm_running_stats_from_epoch 必须 >= 1")
    if freeze_always and freeze_from != 1:
        raise ValueError(
            "不能同时启用 freeze_batchnorm_running_stats=true 和不同起点的 "
            "freeze_batchnorm_running_stats_from_epoch"
        )
    return freeze_always or epoch >= freeze_from


def should_freeze_encoder_parameters(
    train_cfg: dict[str, Any],
    *,
    epoch: int,
) -> bool:
    """解析当前 epoch 是否应冻结 SegFormer3D encoder 的可训练参数。"""
    if epoch <= 0:
        raise ValueError("epoch 必须从 1 开始")

    freeze_from_raw = train_cfg.get("freeze_encoder_parameters_from_epoch")
    if freeze_from_raw is None:
        return False
    if isinstance(freeze_from_raw, bool):
        raise ValueError("freeze_encoder_parameters_from_epoch 必须是正整数 epoch")

    freeze_from = int(freeze_from_raw)
    if freeze_from <= 0:
        raise ValueError("freeze_encoder_parameters_from_epoch 必须 >= 1")
    return epoch >= freeze_from


def configure_encoder_parameter_training(
    model: torch.nn.Module,
    *,
    freeze_parameters: bool,
) -> int:
    """冻结/恢复 SegFormer3D encoder 参数，不改变 decoder/head 的 trainability。"""
    encoder = getattr(model, "segformer_encoder", None)
    if encoder is None:
        if freeze_parameters:
            raise RuntimeError("已启用 encoder 参数冻结，但模型没有 segformer_encoder")
        return 0

    parameter_count = 0
    for parameter in encoder.parameters():
        parameter.requires_grad_(not freeze_parameters)
        parameter_count += parameter.numel()
    return parameter_count


def should_freeze_decoder_feature_parameters(
    train_cfg: dict[str, Any],
    *,
    epoch: int,
) -> bool:
    """解析当前 epoch 是否冻结 decoder feature 参数，并保留最终 segmentation head。"""
    if epoch <= 0:
        raise ValueError("epoch 必须从 1 开始")

    freeze_from_raw = train_cfg.get("freeze_decoder_feature_parameters_from_epoch")
    if freeze_from_raw is None:
        return False
    if isinstance(freeze_from_raw, bool):
        raise ValueError("freeze_decoder_feature_parameters_from_epoch 必须是正整数 epoch")

    freeze_from = int(freeze_from_raw)
    if freeze_from <= 0:
        raise ValueError("freeze_decoder_feature_parameters_from_epoch 必须 >= 1")
    return epoch >= freeze_from


def configure_decoder_feature_parameter_training(
    model: torch.nn.Module,
    *,
    freeze_parameters: bool,
) -> int:
    """冻结/恢复 decoder feature 参数，但始终保留 ``linear_pred`` head 的 trainability。"""
    decoder = getattr(model, "segformer_decoder", None)
    if decoder is None:
        if freeze_parameters:
            raise RuntimeError("已启用 decoder feature 参数冻结，但模型没有 segformer_decoder")
        return 0

    if freeze_parameters and getattr(decoder, "linear_pred", None) is None:
        raise RuntimeError("冻结 decoder feature 参数时必须存在 linear_pred segmentation head")

    parameter_count = 0
    for name, parameter in decoder.named_parameters():
        if name.startswith("linear_pred."):
            parameter.requires_grad_(True)
            continue
        parameter.requires_grad_(not freeze_parameters)
        parameter_count += parameter.numel()
    return parameter_count


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


@contextmanager
def temporary_patch_eval_mode(model: torch.nn.Module, mode: str):
    """无副作用地切换 fixed-patch 诊断模式。"""
    if mode not in {"train", "eval", "batch"}:
        raise ValueError(f"未知 patch eval mode: {mode}")

    module_training_states = [(module, bool(module.training)) for module in model.modules()]
    bn_track_states = [
        (module, bool(module.track_running_stats))
        for module in model.modules()
        if isinstance(module, torch.nn.BatchNorm3d)
    ]
    try:
        if mode == "eval":
            model.eval()
        else:
            model.train(mode == "train")
            for module, _ in bn_track_states:
                module.train(True)
                module.track_running_stats = False
        yield
    finally:
        for module, track_running_stats in bn_track_states:
            module.track_running_stats = track_running_stats
        for module, training in module_training_states:
            module.training = training


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


def evaluate_fixed_training_patches(
    model: torch.nn.Module,
    dataset: ProcessedOrthopedicCTDataset,
    criterion: torch.nn.Module,
    *,
    device: torch.device,
    modes: list[str],
    max_patches: int,
    amp_enabled: bool,
) -> dict[str, dict[str, float]]:
    """直接 forward 固定 train patch，诊断模型是否真的记住训练样本。"""
    if max_patches <= 0:
        raise ValueError("train_patch_eval.max_patches 必须 > 0")
    indices = list(range(min(len(dataset), max_patches)))
    if not indices:
        raise ValueError("train dataset 为空，无法执行 train_patch_eval")

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        per_patch: list[dict[str, float]] = []
        with temporary_patch_eval_mode(model, mode), torch.no_grad():
            for index in indices:
                sample = dataset[index]
                image = sample["image"].unsqueeze(0).to(device)
                label = sample["label"].unsqueeze(0).to(device)
                with _autocast_context(device, amp_enabled):
                    logits = model(image)
                    logits = resize_logits_to_target(logits, tuple(label.shape[-3:]))
                    total_loss = criterion(logits, label)
                    if logits.shape[1] == 1:
                        target_fg = (label > 0).float()
                        fg_prob = torch.sigmoid(logits[:, 0])
                        ce = F.binary_cross_entropy_with_logits(logits[:, 0], target_fg)
                        fg_logit = logits[:, 0]
                    else:
                        target_index = label.long()
                        probs = torch.softmax(logits, dim=1)
                        fg_prob = probs[:, 1:].sum(dim=1)
                        target_fg = (target_index > 0).float()
                        ce = F.cross_entropy(logits, target_index)
                        fg_logit = logits[:, 1:].amax(dim=1)

                pred = logits_to_prediction(logits)
                pred_fg = pred > 0
                target_bool = label > 0
                intersection = float(torch.logical_and(pred_fg, target_bool).sum().item())
                pred_count = float(pred_fg.sum().item())
                target_count = float(target_bool.sum().item())
                eps = 1e-8
                hard_dice = (2.0 * intersection + eps) / (pred_count + target_count + eps)
                precision = (intersection + eps) / (pred_count + eps)
                recall = (intersection + eps) / (target_count + eps)
                soft_intersection = float((fg_prob * target_fg).sum().item())
                soft_denominator = float(fg_prob.sum().item() + target_fg.sum().item())
                soft_dice = (2.0 * soft_intersection + eps) / (soft_denominator + eps)
                flat_prob = fg_prob.float().reshape(-1)
                quantiles = torch.quantile(
                    flat_prob,
                    torch.tensor([0.10, 0.50, 0.90], device=flat_prob.device),
                )
                per_patch.append(
                    {
                        "hard_dice": float(hard_dice),
                        "soft_dice": float(soft_dice),
                        "criterion_loss": float(total_loss.item()),
                        "ce": float(ce.item()),
                        "precision": float(precision),
                        "recall": float(recall),
                        "prediction_foreground_fraction": float(pred_fg.float().mean().item()),
                        "target_foreground_fraction": float(target_bool.float().mean().item()),
                        "prediction_to_target_foreground_ratio": float(
                            (pred_count + eps) / (target_count + eps)
                        ),
                        "logits_mean": float(logits.float().mean().item()),
                        "logits_std": float(logits.float().std().item()),
                        "foreground_logit_mean": float(fg_logit.float().mean().item()),
                        "foreground_logit_std": float(fg_logit.float().std().item()),
                        "foreground_probability_mean": float(flat_prob.mean().item()),
                        "foreground_probability_std": float(flat_prob.std().item()),
                        "foreground_probability_q10": float(quantiles[0].item()),
                        "foreground_probability_q50": float(quantiles[1].item()),
                        "foreground_probability_q90": float(quantiles[2].item()),
                    }
                )

        keys = per_patch[0].keys()
        results[mode] = {
            key: float(np.mean([patch[key] for patch in per_patch])) for key in keys
        }
        results[mode]["patch_count"] = float(len(per_patch))
    return results


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
    postprocessing_cfg: dict[str, Any] | None = None,
    direct_forward: bool = False,
) -> dict[str, float]:
    model.eval()
    case_scores: list[float] = []
    prediction_foreground_fractions: list[float] = []
    target_foreground_fractions: list[float] = []
    prediction_to_target_foreground_ratios: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    component_count_errors: list[float] = []
    total_time = 0.0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            start = time.perf_counter()
            with _autocast_context(device, amp_enabled):
                if direct_forward:
                    # patch-validation sanity 必须与训练直接 forward 的输入条件完全一致。
                    # 不能让 sliding_window_inference 把 64^3 patch pad 成更大的 ROI，
                    # 否则 tiny-overfit 的 validation 实际上已经不是同一个模型输入。
                    logits = model(image)
                    logits = resize_logits_to_target(logits, tuple(label.shape[-3:]))
                else:
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
            pred_np = postprocess_prediction(
                pred[0].detach().cpu().numpy(),
                postprocessing_cfg,
            )
            pred_for_metric = torch.from_numpy(pred_np).unsqueeze(0).to(
                device=label.device,
                dtype=pred.dtype,
            )
            case_scores.append(mean_foreground_dice(pred_for_metric, label, num_classes))

            pred_mask = pred_np > 0
            target_mask = (label[0].detach().cpu().numpy() > 0)
            prediction_foreground_fractions.append(float(pred_mask.mean()))
            target_foreground_fractions.append(float(target_mask.mean()))
            target_foreground_voxels = int(target_mask.sum())
            prediction_to_target_foreground_ratios.append(
                float(pred_mask.sum() / target_foreground_voxels)
                if target_foreground_voxels > 0
                else float("inf")
            )
            _, _, precision, recall = binary_overlap_metrics(pred_mask, target_mask)
            precisions.append(float(precision))
            recalls.append(float(recall))
            component_count_errors.append(
                float(compute_structural_metrics(pred_mask, target_mask).component_count_error)
            )

    return {
        "val_dice": float(np.mean(case_scores)),
        "val_dice_std": float(np.std(case_scores)),
        "val_case_count": float(len(case_scores)),
        "val_inference_seconds_total": float(total_time),
        "val_prediction_foreground_fraction": float(np.mean(prediction_foreground_fractions)),
        "val_target_foreground_fraction": float(np.mean(target_foreground_fractions)),
        "val_prediction_to_target_foreground_ratio": float(
            np.mean(prediction_to_target_foreground_ratios)
        ),
        "val_precision": float(np.mean(precisions)),
        "val_recall": float(np.mean(recalls)),
        "val_component_count_error": float(np.mean(component_count_errors)),
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
    training_seconds_total: float = 0.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "val_dice": float(val_dice),
        "best_val_dice": float(val_dice if best_val_dice is None else best_val_dice),
        "epochs_without_improvement": int(epochs_without_improvement),
        "training_seconds_total": float(training_seconds_total),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config,
        "model_provenance": model_provenance(config),
        "git_commit": current_git_commit(),
    }
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        torch.save(payload, temp_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(output_path)


def load_model_initialization_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """只加载模型权重，用于新实验从旧 checkpoint 初始化。

    与 resume 不同：不恢复 optimizer/scheduler/RNG/epoch/history，也不要求 config 完全一致；
    但模型 state_dict 必须严格匹配，避免静默漏层。
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "source_val_dice": (
            float(checkpoint.get("val_dice", -1.0)) if isinstance(checkpoint, dict) else -1.0
        ),
        "source_best_val_dice": (
            float(checkpoint.get("best_val_dice", checkpoint.get("val_dice", -1.0)))
            if isinstance(checkpoint, dict)
            else -1.0
        ),
    }


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
        "val_dice": float(checkpoint.get("val_dice", -1.0)),
        "best_val_dice": float(checkpoint.get("best_val_dice", checkpoint.get("val_dice", -1.0))),
        "epochs_without_improvement": int(checkpoint.get("epochs_without_improvement", 0)),
        "training_seconds_total": float(checkpoint.get("training_seconds_total", 0.0)),
    }


def train(
    config_path: Path,
    *,
    max_epochs_override: int | None = None,
    resume_checkpoint: Path | None = None,
    init_checkpoint: Path | None = None,
    auto_resume: bool = True,
    target_val_dice_override: float | None = None,
    status_every_steps: int | None = None,
    early_stopping_patience_override: int | None = None,
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

    if resume_checkpoint is not None and init_checkpoint is not None:
        raise ValueError("resume_checkpoint 与 init_checkpoint 不能同时使用")

    init_path = _resolve_project_path(init_checkpoint) if init_checkpoint else None
    if init_path is not None and not init_path.exists():
        raise FileNotFoundError(f"init checkpoint 不存在: {init_path}")

    resume_source = "manual" if resume_checkpoint is not None else None
    resume_path = _resolve_project_path(resume_checkpoint) if resume_checkpoint else None
    if resume_path is None and init_path is None and auto_resume:
        resume_path = find_latest_compatible_last_checkpoint(config)
        if resume_path is not None:
            resume_source = "auto"
            print(f"[AUTO-RESUME] 找到兼容 last.pt: {resume_path}")

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

    run_lock_path = acquire_run_lock(run_dir)
    atexit.register(release_run_lock, run_lock_path)

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
            "model_provenance": model_provenance(config),
            "source_config": str(config_path),
            "source_split": str(split_file),
            "validation_mode": "patch" if validation_patch_mode else "full_volume",
            "validation_patch_is_engineering_proxy": validation_patch_mode,
            "training_patch_sampling_epoch_aware": not bool(
                train_cfg.get("freeze_patch_sampling_across_epochs", False)
            ),
            "training_freeze_patch_sampling_across_epochs": bool(
                train_cfg.get("freeze_patch_sampling_across_epochs", False)
            ),
            "training_patches_per_case": int(train_cfg.get("patches_per_case", 1)),
            "training_max_train_cases": train_cfg.get("max_train_cases"),
            "training_foreground_sampling_mode": str(
                data_cfg.get("foreground_sampling_mode", "bernoulli")
            ),
            "training_sampling_stats_logged": True,
            "training_train_patch_eval": train_cfg.get("train_patch_eval", {}),
            "training_freeze_batchnorm_running_stats": bool(
                train_cfg.get("freeze_batchnorm_running_stats", False)
            ),
            "training_freeze_batchnorm_running_stats_from_epoch": train_cfg.get(
                "freeze_batchnorm_running_stats_from_epoch"
            ),
            "training_freeze_encoder_parameters_from_epoch": train_cfg.get(
                "freeze_encoder_parameters_from_epoch"
            ),
            "training_freeze_decoder_feature_parameters_from_epoch": train_cfg.get(
                "freeze_decoder_feature_parameters_from_epoch"
            ),
            "validation_patch_sampling_fixed_across_epochs": validation_patch_mode,
            "resume_events": [],
            "initialization": None,
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
    training_case_count_available = len(train_ds.case_ids)
    max_train_cases_raw = train_cfg.get("max_train_cases")
    if max_train_cases_raw is not None:
        max_train_cases = int(max_train_cases_raw)
        if max_train_cases <= 0:
            raise ValueError("training.max_train_cases 必须 > 0")
        if max_train_cases < len(train_ds.case_ids):
            indices = np.linspace(0, len(train_ds.case_ids) - 1, num=max_train_cases, dtype=int)
            train_ds.case_ids = [train_ds.case_ids[int(index)] for index in indices]
    metadata["training_case_count_available"] = training_case_count_available
    metadata["training_case_count_per_epoch"] = len(train_ds.case_ids)
    metadata["training_subset_policy"] = (
        "full_train"
        if len(train_ds.case_ids) == training_case_count_available
        else "fixed_evenly_spaced_subset_for_pilot"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
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
            foreground_sampling_mode=str(
                validation_cfg.get("foreground_sampling_mode", "bernoulli")
            ),
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

    validation_case_count_available = len(val_ds.case_ids)
    validation_max_cases_raw = validation_cfg.get("max_cases")
    validation_max_cases = (
        None if validation_max_cases_raw is None else int(validation_max_cases_raw)
    )
    val_ds.case_ids = select_validation_case_subset(val_ds.case_ids, validation_max_cases)
    metadata["validation_case_count_available"] = validation_case_count_available
    metadata["validation_case_count_per_epoch"] = len(val_ds.case_ids)
    metadata["validation_subset_policy"] = (
        "full_validation"
        if len(val_ds.case_ids) == validation_case_count_available
        else "fixed_evenly_spaced_subset_for_epoch_selector"
    )
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
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
    model = build_segmentation_model(config).to(device)
    if init_path is not None:
        initialization = load_model_initialization_checkpoint(
            init_path,
            model=model,
            device=device,
        )
        metadata["initialization"] = {
            **initialization,
            "initialized_at": datetime.now().isoformat(),
            "mode": "model_weights_only",
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[INIT] 从旧 checkpoint 只加载模型权重: {init_path} | "
            f"source_epoch={initialization['source_epoch']} | "
            f"source_best_val_dice={initialization['source_best_val_dice']:.6f}"
        )
    criterion = build_criterion(config).to(device)

    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-2)),
    )
    scheduler = build_scheduler(config, optimizer)

    max_epochs = int(max_epochs_override or train_cfg.get("epochs", 800))
    if max_epochs <= 0:
        raise ValueError("max_epochs 必须 > 0")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    # PyTorch 2.1 使用 torch.cuda.amp.GradScaler；CPU 路径保持 disabled。
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    accumulate = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    patience = int(
        early_stopping_patience_override
        if early_stopping_patience_override is not None
        else train_cfg.get("early_stopping_patience", 100)
    )
    if patience <= 0:
        raise ValueError("early_stopping_patience 必须 > 0")
    target_val_dice_raw = (
        target_val_dice_override
        if target_val_dice_override is not None
        else train_cfg.get("target_val_dice")
    )
    target_val_dice = None if target_val_dice_raw is None else float(target_val_dice_raw)
    if target_val_dice is not None and not 0.0 <= target_val_dice <= 1.0:
        raise ValueError("target_val_dice 必须在 [0, 1] 范围内")
    stop_on_target = bool(train_cfg.get("stop_on_target", True))
    status_interval = max(
        1,
        int(
            status_every_steps
            if status_every_steps is not None
            else train_cfg.get("status_every_steps", 10)
        ),
    )
    patch_eval_raw = train_cfg.get("train_patch_eval", {})
    if isinstance(patch_eval_raw, bool):
        patch_eval_cfg = {"enabled": patch_eval_raw}
    else:
        patch_eval_cfg = dict(patch_eval_raw or {})
    patch_eval_enabled = bool(patch_eval_cfg.get("enabled", False))
    patch_eval_modes = [str(value).lower() for value in patch_eval_cfg.get(
        "modes", ["train", "eval", "batch"]
    )]
    if any(mode not in {"train", "eval", "batch"} for mode in patch_eval_modes):
        raise ValueError("training.train_patch_eval.modes 仅支持 train/eval/batch")
    patch_eval_max_patches = int(patch_eval_cfg.get("max_patches", 1))
    if patch_eval_enabled and patch_eval_max_patches <= 0:
        raise ValueError("training.train_patch_eval.max_patches 必须 > 0")

    best_dice = -1.0
    last_val_dice = -1.0
    epochs_without_improvement = 0
    completed_training_seconds = 0.0
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
        last_val_dice = float(resume_state["val_dice"])
        best_dice = float(resume_state["best_val_dice"])
        epochs_without_improvement = int(resume_state["epochs_without_improvement"])
        completed_training_seconds = float(resume_state["training_seconds_total"])
        metadata["resume_events"].append(
            {
                "resumed_at": datetime.now().isoformat(),
                "resume_source": resume_source,
                "checkpoint": str(resume_path),
                "checkpoint_epoch": int(resume_state["epoch"]),
                "target_max_epochs": max_epochs,
                "target_val_dice": target_val_dice,
                "git_commit": current_git_commit(),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    history_csv = run_dir / "history.csv"
    sampling_stats_csv = run_dir / "sampling_stats.csv"
    train_patch_eval_jsonl = run_dir / "train_patch_eval.jsonl"
    train_log = run_dir / "train.log"
    append_history = resume_path is not None and history_csv.exists()
    append_sampling_stats = resume_path is not None and sampling_stats_csv.exists()
    history_mode = "a" if append_history else "w"
    sampling_mode = "a" if append_sampling_stats else "w"
    default_history_fieldnames = [
        "epoch",
        "train_loss",
        "val_dice",
        "val_dice_std",
        "val_inference_seconds_total",
        "val_prediction_foreground_fraction",
        "val_target_foreground_fraction",
        "val_prediction_to_target_foreground_ratio",
        "val_precision",
        "val_recall",
        "val_component_count_error",
        "train_patch_eval_train_hard_dice",
        "train_patch_eval_eval_hard_dice",
        "train_patch_eval_batch_hard_dice",
        "lr",
    ]
    if append_history:
        with history_csv.open("r", encoding="utf-8", newline="") as existing_history:
            history_fieldnames = next(csv.reader(existing_history), default_history_fieldnames)
    else:
        history_fieldnames = default_history_fieldnames
    last_epoch = start_epoch - 1
    checkpoint_path = run_dir / "checkpoint" / "last.pt"
    status_path = run_dir / "training_status.json"
    stop_request_path = run_dir / "STOP_REQUESTED"
    # 上一次安全停止留下的请求文件不能影响本次续训。
    stop_request_path.unlink(missing_ok=True)

    # 新 run 在第 1 个 epoch 前先保存 epoch=0 的一致 checkpoint。
    # 因此即使首轮中途 Ctrl+C，也能从同一 run 安全重新开始 epoch 1。
    if resume_path is None:
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=0,
            val_dice=-1.0,
            config=config,
            scheduler=scheduler,
            best_val_dice=-1.0,
            epochs_without_improvement=0,
            training_seconds_total=0.0,
        )

    if start_epoch > max_epochs:
        status = build_training_status(
            phase="already_complete",
            epoch=last_epoch,
            max_epochs=max_epochs,
            step=None,
            total_steps=None,
            train_loss=None,
            val_dice=last_val_dice,
            best_val_dice=best_dice,
            lr=float(optimizer.param_groups[0]["lr"]),
            completed_training_seconds=completed_training_seconds,
            message="last.pt 已达到或超过当前总目标 epoch，无需重复训练。",
        )
        write_training_status(status_path, status)
        print(format_training_status_line(status))
        print("=" * 84)
        print("[TRAINING READY] 当前 last.pt 已达到设定总 epoch；提高 --max-epochs 可继续续训。")
        print("=" * 84)
        summary = {
            "finished_at": datetime.now().isoformat(),
            "status": "already_complete",
            "best_val_dice": best_dice,
            "last_val_dice": last_val_dice,
            "last_epoch": last_epoch,
            "target_max_epochs": max_epochs,
            "target_val_dice": target_val_dice,
            "training_seconds_total": completed_training_seconds,
            "epochs_without_improvement": epochs_without_improvement,
            "resumed": resume_path is not None,
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        release_run_lock(run_lock_path)
        return run_dir

    if target_val_dice is not None and best_dice >= target_val_dice and stop_on_target:
        status = build_training_status(
            phase="target_reached",
            epoch=last_epoch,
            max_epochs=max_epochs,
            step=None,
            total_steps=None,
            train_loss=None,
            val_dice=last_val_dice,
            best_val_dice=best_dice,
            lr=float(optimizer.param_groups[0]["lr"]),
            completed_training_seconds=completed_training_seconds,
            message=f"best Dice 已达到目标 {target_val_dice:.6f}。",
        )
        write_training_status(status_path, status)
        print(format_training_status_line(status))
        print("!" * 84)
        print(f"[TARGET REACHED] best Dice={best_dice:.6f} >= 目标 {target_val_dice:.6f}")
        print("!" * 84)
        summary = {
            "finished_at": datetime.now().isoformat(),
            "status": "target_reached",
            "best_val_dice": best_dice,
            "last_val_dice": last_val_dice,
            "last_epoch": last_epoch,
            "target_max_epochs": max_epochs,
            "target_val_dice": target_val_dice,
            "training_seconds_total": completed_training_seconds,
            "epochs_without_improvement": epochs_without_improvement,
            "resumed": resume_path is not None,
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        release_run_lock(run_lock_path)
        return run_dir

    interrupt_state: dict[str, Any] = {
        "phase": "starting",
        "epoch": start_epoch,
        "step": None,
        "total_steps": len(train_loader),
        "train_loss": None,
        "epoch_started": None,
    }
    heartbeat_path = run_dir / "training_heartbeat.json"
    heartbeat_stop = threading.Event()

    def _heartbeat_worker() -> None:
        while not heartbeat_stop.is_set():
            try:
                write_training_heartbeat(
                    heartbeat_path,
                    phase=str(interrupt_state.get("phase") or "training"),
                    epoch=int(interrupt_state.get("epoch") or start_epoch),
                    step=interrupt_state.get("step"),
                    total_steps=interrupt_state.get("total_steps"),
                )
            except OSError:
                # 心跳失败不能中断真实训练；面板仍可通过 PID/状态文件判断。
                pass
            heartbeat_stop.wait(2.0)

    def _handle_sigint(_signum: int, _frame: Any) -> None:
        attempted_epoch = int(interrupt_state.get("epoch") or start_epoch)
        epoch_started = interrupt_state.get("epoch_started")
        current_epoch_seconds = (
            0.0 if epoch_started is None else max(0.0, time.perf_counter() - float(epoch_started))
        )
        safe_status = build_training_status(
            phase="interrupted",
            epoch=max(0, last_epoch),
            max_epochs=max_epochs,
            step=None,
            total_steps=None,
            train_loss=interrupt_state.get("train_loss"),
            val_dice=None if last_val_dice < 0 else last_val_dice,
            best_val_dice=best_dice,
            lr=float(optimizer.param_groups[0]["lr"]),
            completed_training_seconds=completed_training_seconds,
            current_epoch_seconds=0.0,
            message=(
                f"已安全中断；epoch {attempted_epoch} 的未完成部分不写入 checkpoint，"
                f"下次自动从完整 epoch {max(0, last_epoch)} 的 last.pt 继续。"
            ),
        )
        write_training_status(status_path, safe_status)
        interrupted_summary = {
            "finished_at": datetime.now().isoformat(),
            "status": "interrupted",
            "best_val_dice": best_dice,
            "last_val_dice": last_val_dice,
            "last_epoch": last_epoch,
            "interrupted_during_epoch": attempted_epoch,
            "discarded_partial_epoch_seconds": current_epoch_seconds,
            "target_max_epochs": max_epochs,
            "target_val_dice": target_val_dice,
            "training_seconds_total": completed_training_seconds,
            "epochs_without_improvement": epochs_without_improvement,
            "resumed": resume_path is not None,
            "resume_checkpoint": str(checkpoint_path),
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(interrupted_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n" + "!" * 84)
        print("[SAFE STOP] 收到 Ctrl+C，未保存半个 epoch 的状态。")
        print(f"[SAFE STOP] 保留完整 checkpoint: {checkpoint_path}")
        print("[SAFE STOP] 再次运行同一 config 会自动从 last.pt 续训。")
        print("!" * 84)
        release_run_lock(run_lock_path)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _handle_sigint)
    except ValueError:
        # 非主线程运行时无法注册 signal；last.pt 仍按完整 epoch 保存。
        pass

    threading.Thread(
        target=_heartbeat_worker,
        name="training-heartbeat",
        daemon=True,
    ).start()

    with (
        history_csv.open(history_mode, encoding="utf-8", newline="") as f,
        sampling_stats_csv.open(sampling_mode, encoding="utf-8", newline="") as sampling_f,
    ):
        writer = csv.DictWriter(
            f,
            fieldnames=history_fieldnames,
            extrasaction="ignore",
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

        stop_reason = "completed"
        for epoch in range(start_epoch, max_epochs + 1):
            epoch_started = time.perf_counter()
            interrupt_state.update(
                {
                    "phase": "training",
                    "epoch": epoch,
                    "step": 0,
                    "total_steps": len(train_loader),
                    "train_loss": None,
                    "epoch_started": epoch_started,
                }
            )
            if scheduler is not None:
                scheduler.step(epoch)
            if bool(train_cfg.get("freeze_patch_sampling_across_epochs", False)):
                # tiny-overfit/sanity 专用：每个 epoch 重放完全相同的 crop，
                # 用于判断 model/loss/label 链路能否真正记住固定样本。
                train_ds.set_epoch(0)
            else:
                train_ds.set_epoch(epoch)
            model.train()
            if "freeze_encoder_parameters_from_epoch" in train_cfg:
                configure_encoder_parameter_training(
                    model,
                    freeze_parameters=should_freeze_encoder_parameters(
                        train_cfg,
                        epoch=epoch,
                    ),
                )
            if "freeze_decoder_feature_parameters_from_epoch" in train_cfg:
                configure_decoder_feature_parameter_training(
                    model,
                    freeze_parameters=should_freeze_decoder_feature_parameters(
                        train_cfg,
                        epoch=epoch,
                    ),
                )
            configure_batchnorm_training_mode(
                model,
                freeze_running_stats=should_freeze_batchnorm_running_stats(
                    train_cfg,
                    epoch=epoch,
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
                live_train_loss = running_loss / max(batch_count, 1)
                interrupt_state["step"] = step
                interrupt_state["train_loss"] = live_train_loss
                if stop_request_path.exists():
                    _handle_sigint(signal.SIGINT, None)
                if step % status_interval == 0 or step == len(train_loader):
                    live_status = build_training_status(
                        phase="training",
                        epoch=epoch,
                        max_epochs=max_epochs,
                        step=step,
                        total_steps=len(train_loader),
                        train_loss=live_train_loss,
                        val_dice=None if last_val_dice < 0 else last_val_dice,
                        best_val_dice=best_dice,
                        lr=float(optimizer.param_groups[0]["lr"]),
                        completed_training_seconds=completed_training_seconds,
                        current_epoch_seconds=time.perf_counter() - epoch_started,
                        message="训练中；val Dice 为上一完整 epoch 的最近值。",
                    )
                    write_training_status(status_path, live_status)
                    print(format_training_status_line(live_status), flush=True)

            train_loss = running_loss / max(batch_count, 1)
            patch_eval_result = {}
            if patch_eval_enabled:
                patch_eval_result = evaluate_fixed_training_patches(
                    model, train_ds, criterion,
                    device=device,
                    modes=patch_eval_modes,
                    max_patches=patch_eval_max_patches,
                    amp_enabled=amp_enabled,
                )
                record = {"epoch": epoch, "modes": patch_eval_result}
                with train_patch_eval_jsonl.open("a", encoding="utf-8") as patch_file:
                    patch_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                parts = [
                    f"{mode}:hard={stats['hard_dice']:.4f}/soft={stats['soft_dice']:.4f}"
                    for mode, stats in patch_eval_result.items()
                ]
                print("[TRAIN-PATCH] " + " | ".join(parts), flush=True)
            interrupt_state["phase"] = "validating"
            interrupt_state["step"] = None
            sampling_row = {
                "epoch": epoch,
                **summarize_foreground_fractions(epoch_foreground_fractions),
            }
            sampling_writer.writerow(sampling_row)
            sampling_f.flush()
            validating_status = build_training_status(
                phase="validating",
                epoch=epoch,
                max_epochs=max_epochs,
                step=None,
                total_steps=None,
                train_loss=train_loss,
                val_dice=None if last_val_dice < 0 else last_val_dice,
                best_val_dice=best_dice,
                lr=float(optimizer.param_groups[0]["lr"]),
                completed_training_seconds=completed_training_seconds,
                current_epoch_seconds=time.perf_counter() - epoch_started,
                message=(
                    f"正在 validation：本轮固定 {len(val_ds.case_ids)} 个 patch，direct forward。"
                    if validation_patch_mode
                    else f"正在 validation：本轮固定 {len(val_ds.case_ids)} 个完整体病例；完成后写入 Val Dice。"
                ),
            )
            write_training_status(status_path, validating_status)
            print(format_training_status_line(validating_status), flush=True)
            val_result = validate(
                model,
                val_loader,
                device=device,
                num_classes=int(model_cfg["num_classes"]),
                roi_size_dhw=tuple(int(v) for v in infer_cfg.get("roi_size_dhw", roi)),
                sw_batch_size=int(infer_cfg.get("sw_batch_size", 1)),
                overlap=float(infer_cfg.get("overlap", 0.5)),
                amp_enabled=amp_enabled,
                postprocessing_cfg=infer_cfg.get("postprocessing", {}),
                direct_forward=validation_patch_mode,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_dice": val_result["val_dice"],
                "val_dice_std": val_result["val_dice_std"],
                "val_inference_seconds_total": val_result["val_inference_seconds_total"],
                "val_prediction_foreground_fraction": val_result[
                    "val_prediction_foreground_fraction"
                ],
                "val_target_foreground_fraction": val_result["val_target_foreground_fraction"],
                "val_prediction_to_target_foreground_ratio": val_result[
                    "val_prediction_to_target_foreground_ratio"
                ],
                "val_precision": val_result["val_precision"],
                "val_recall": val_result["val_recall"],
                "val_component_count_error": val_result["val_component_count_error"],
                "train_patch_eval_train_hard_dice": patch_eval_result.get("train", {}).get("hard_dice"),
                "train_patch_eval_eval_hard_dice": patch_eval_result.get("eval", {}).get("hard_dice"),
                "train_patch_eval_batch_hard_dice": patch_eval_result.get("batch", {}).get("hard_dice"),
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(
                "[VAL] "
                f"Dice={val_result['val_dice']:.6f} | "
                f"Precision={val_result['val_precision']:.6f} | "
                f"Recall={val_result['val_recall']:.6f} | "
                f"fg ratio={val_result['val_prediction_to_target_foreground_ratio']:.6f} | "
                f"component error={val_result['val_component_count_error']:.6f}",
                flush=True,
            )
            log_line = json.dumps(row, ensure_ascii=False)
            with train_log.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")

            interrupt_state["phase"] = "checkpointing"
            epoch_seconds = time.perf_counter() - epoch_started
            completed_training_seconds += epoch_seconds
            last_val_dice = float(val_result["val_dice"])

            if last_val_dice > best_dice:
                best_dice = last_val_dice
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
                    training_seconds_total=completed_training_seconds,
                )
            else:
                epochs_without_improvement += 1

            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_dice=last_val_dice,
                config=config,
                scheduler=scheduler,
                best_val_dice=best_dice,
                epochs_without_improvement=epochs_without_improvement,
                training_seconds_total=completed_training_seconds,
            )
            last_epoch = epoch

            epoch_status = build_training_status(
                phase="epoch_complete",
                epoch=epoch,
                max_epochs=max_epochs,
                step=len(train_loader),
                total_steps=len(train_loader),
                train_loss=train_loss,
                val_dice=last_val_dice,
                best_val_dice=best_dice,
                lr=float(optimizer.param_groups[0]["lr"]),
                completed_training_seconds=completed_training_seconds,
                message="完整 epoch 已保存到 last.pt。",
            )
            epoch_status.update(
                {
                    "val_precision": float(val_result["val_precision"]),
                    "val_recall": float(val_result["val_recall"]),
                    "val_prediction_to_target_foreground_ratio": float(
                        val_result["val_prediction_to_target_foreground_ratio"]
                    ),
                    "val_component_count_error": float(val_result["val_component_count_error"]),
                }
            )
            write_training_status(status_path, epoch_status)
            print(format_training_status_line(epoch_status), flush=True)

            if target_val_dice is not None and best_dice >= target_val_dice:
                print("!" * 84)
                print(
                    f"[TARGET REACHED] best Dice={best_dice:.6f} >= "
                    f"目标 {target_val_dice:.6f}"
                )
                print(f"[TARGET REACHED] checkpoint: {checkpoint_path}")
                print("!" * 84)
                if stop_on_target:
                    stop_reason = "target_reached"
                    break

            if epochs_without_improvement >= patience:
                stop_reason = "early_stopped"
                print(
                    f"[EARLY STOP] {patience} epochs without validation Dice improvement."
                )
                break

    final_status = build_training_status(
        phase=stop_reason,
        epoch=last_epoch,
        max_epochs=max_epochs,
        step=None,
        total_steps=None,
        train_loss=None,
        val_dice=None if last_val_dice < 0 else last_val_dice,
        best_val_dice=best_dice,
        lr=float(optimizer.param_groups[0]["lr"]),
        completed_training_seconds=completed_training_seconds,
        message=(
            "训练目标已达到。"
            if stop_reason == "target_reached"
            else "early stopping 已触发。"
            if stop_reason == "early_stopped"
            else "已完成设定总 epoch。"
        ),
    )
    write_training_status(status_path, final_status)

    summary = {
        "finished_at": datetime.now().isoformat(),
        "status": stop_reason,
        "best_val_dice": best_dice,
        "last_val_dice": last_val_dice,
        "last_epoch": last_epoch,
        "target_max_epochs": max_epochs,
        "target_val_dice": target_val_dice,
        "stop_on_target": stop_on_target,
        "training_seconds_total": completed_training_seconds,
        "epochs_without_improvement": epochs_without_improvement,
        "resumed": resume_path is not None,
        "resume_source": resume_source,
        "resume_checkpoint": str(checkpoint_path),
        "training_status": str(status_path),
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
    release_run_lock(run_lock_path)
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
        help="手动指定同一 run 的 checkpoint/last.pt；默认会自动发现最近兼容 last.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="从旧 checkpoint 只加载模型权重并新建 run；不恢复 optimizer/epoch/history",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="明确新建 run，不自动续训历史 last.pt",
    )
    parser.add_argument(
        "--target-dice",
        type=float,
        default=None,
        help="覆盖 training.target_val_dice；达到目标时明显提示，并默认安全停止",
    )
    parser.add_argument(
        "--status-every-steps",
        type=int,
        default=None,
        help="训练中每多少 step 刷新一次实时状态；默认读取 config，未配置时为 10",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="覆盖 config 中的 early_stopping_patience；例如设很大可显式跑满总 Epoch",
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
    if args.resume is not None and args.fresh:
        parser.error("--resume 与 --fresh 不能同时使用")
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error("--resume 与 --init-checkpoint 不能同时使用")

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
    try:
        run_dir = train(
            config_path,
            max_epochs_override=args.max_epochs,
            resume_checkpoint=args.resume,
            init_checkpoint=args.init_checkpoint,
            auto_resume=not args.fresh and args.init_checkpoint is None,
            target_val_dice_override=args.target_dice,
            status_every_steps=args.status_every_steps,
            early_stopping_patience_override=args.early_stopping_patience,
        )
    except KeyboardInterrupt:
        print("训练已安全中断；重新运行同一 config 将自动从 last.pt 继续。")
        return
    except Exception as exc:
        # 给面板留下明确的终止原因。只写当前 config 对应 run 的状态/失败记录，
        # 不覆盖历史 history、checkpoint 或既有 summary。
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            latest_checkpoint = find_latest_compatible_last_checkpoint(config)
            if latest_checkpoint is not None:
                failed_run_dir = latest_checkpoint.parent.parent
                message = f"{type(exc).__name__}: {exc}"
                data_markers = (
                    "读取输入通道",
                    "读取 label",
                    "image/label shape 不一致",
                    "nifti",
                    ".nii.gz",
                    "文件损坏",
                    "file not found",
                    "no such file",
                )
                phase = (
                    "data_corrupt"
                    if any(marker in message.lower() for marker in data_markers)
                    else "failed"
                )
                failure_payload = {
                    "updated_at": datetime.now().isoformat(),
                    "phase": phase,
                    "message": message,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run_dir": str(failed_run_dir),
                    "checkpoint_preserved": str(latest_checkpoint),
                }
                write_training_status(
                    failed_run_dir / "training_status.json",
                    failure_payload,
                )
                (failed_run_dir / "failure.json").write_text(
                    json.dumps(failure_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as status_exc:
            print(f"[WARN] 写异常状态失败: {status_exc}", file=sys.stderr)
        raise

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        print(
            f"Run finished: status={summary.get('status')} | "
            f"epoch={summary.get('last_epoch')} | best Dice={summary.get('best_val_dice')} | "
            f"run={run_dir}"
        )
    else:
        print(f"Run finished: {run_dir}")


if __name__ == "__main__":
    main()
