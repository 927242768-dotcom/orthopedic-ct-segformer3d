"""SegFormer3D 上游模型适配器。

本文件不复制上游 SegFormer3D 核心实现，而是在用户通过
`env/fetch_segformer3d.ps1` 获取官方仓库后，动态加载其模型构建函数。

这样可以：
- 清楚区分第三方基础架构与本项目自研模块；
- 保留上游许可证/提交历史；
- 让骨科 CT 的输入通道、类别数、decoder 等通过本项目配置覆盖。
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPSTREAM_HOME = PROJECT_ROOT / "third_party" / "SegFormer3D"


class SegFormer3DUpstreamNotFound(RuntimeError):
    """上游 SegFormer3D 未获取或目录不完整。"""


def ensure_upstream_available(upstream_home: str | Path | None = None) -> Path:
    home = Path(upstream_home) if upstream_home is not None else DEFAULT_UPSTREAM_HOME
    architecture_file = home / "architectures" / "segformer3d.py"
    license_file = home / "LICENSE"

    if not architecture_file.exists():
        raise SegFormer3DUpstreamNotFound(
            "未找到 SegFormer3D 官方实现。请在项目根目录运行 "
            "`powershell -ExecutionPolicy Bypass -File .\\env\\fetch_segformer3d.ps1`，"
            f"期望文件：{architecture_file}"
        )
    if not license_file.exists():
        raise SegFormer3DUpstreamNotFound(
            f"发现上游代码但 LICENSE 缺失，请人工核对后再继续：{home}"
        )
    return home.resolve()


def load_upstream_module(upstream_home: str | Path | None = None):
    home = ensure_upstream_available(upstream_home)
    home_str = str(home)
    if home_str not in sys.path:
        sys.path.insert(0, home_str)

    try:
        return importlib.import_module("architectures.segformer3d")
    except Exception as exc:  # pragma: no cover - 依赖上游环境
        raise RuntimeError(
            "SegFormer3D 上游模块导入失败。请先完成项目 .venv 环境安装，"
            f"原始异常：{type(exc).__name__}: {exc}"
        ) from exc


def _extract_model_parameters(config: dict[str, Any]) -> dict[str, Any]:
    """把本项目 YAML 风格配置转换为上游 `model_parameters` 结构。"""
    model_cfg = config.get("model", config.get("model_parameters", {}))
    required = {
        "in_channels",
        "sr_ratios",
        "embed_dims",
        "patch_kernel_size",
        "patch_stride",
        "patch_padding",
        "mlp_ratios",
        "num_heads",
        "depths",
        "decoder_head_embedding_dim",
        "num_classes",
        "decoder_dropout",
    }
    missing = sorted(required - set(model_cfg))
    if missing:
        raise ValueError(f"SegFormer3D 模型配置缺少字段：{missing}")

    return {key: model_cfg[key] for key in required}


def _group_count(num_channels: int, requested_groups: int) -> int:
    """选择不超过 requested_groups 且能整除通道数的最大 GroupNorm group 数。"""
    if requested_groups <= 0:
        raise ValueError("groupnorm_num_groups 必须 > 0")
    for groups in range(min(int(requested_groups), int(num_channels)), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


def replace_batchnorm3d_with_groupnorm(
    module: torch.nn.Module,
    *,
    num_groups: int = 16,
) -> int:
    """递归把 BatchNorm3d 替换为 GroupNorm，返回替换数量。"""
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.BatchNorm3d):
            groups = _group_count(int(child.num_features), int(num_groups))
            replacement = torch.nn.GroupNorm(
                groups,
                int(child.num_features),
                eps=float(child.eps),
                affine=bool(child.affine),
            )
            if child.affine:
                with torch.no_grad():
                    replacement.weight.copy_(child.weight.detach())
                    replacement.bias.copy_(child.bias.detach())
            setattr(module, name, replacement)
            replaced += 1
        else:
            replaced += replace_batchnorm3d_with_groupnorm(
                child,
                num_groups=num_groups,
            )
    return replaced


def configure_normalization(model: torch.nn.Module, config: dict[str, Any]) -> int:
    """按项目配置替换上游 BN；默认 batchnorm 保持上游结构不变。"""
    model_cfg = config.get("model", {})
    normalization = str(model_cfg.get("normalization", "batchnorm")).lower()
    if normalization in {"batchnorm", "batchnorm3d", "bn", "upstream"}:
        return 0
    if normalization not in {"groupnorm", "groupnorm3d", "gn"}:
        raise ValueError(f"未知 model.normalization: {normalization}")

    scope = str(model_cfg.get("groupnorm_scope", "all")).lower()
    if scope == "all":
        target = model
    elif scope == "decoder":
        target = getattr(model, "segformer_decoder", None)
        if target is None:
            raise RuntimeError("groupnorm_scope=decoder 但模型没有 segformer_decoder")
    else:
        raise ValueError("model.groupnorm_scope 仅支持 all 或 decoder")

    replaced = replace_batchnorm3d_with_groupnorm(
        target,
        num_groups=int(model_cfg.get("groupnorm_num_groups", 16)),
    )
    if replaced == 0:
        raise RuntimeError(f"启用 GroupNorm 但 scope={scope!r} 内没有 BatchNorm3d 可替换")
    return replaced


def build_orthopedic_segformer3d(
    config: dict[str, Any],
    *,
    upstream_home: str | Path | None = None,
):
    """构建骨科 CT SegFormer3D baseline。

    返回值是上游 `SegFormer3D` 实例；本项目的联合损失、数据处理、训练与
    不确定性精修在其外部实现，避免修改上游文件后难以追踪来源。
    """
    module = load_upstream_module(upstream_home)
    model_parameters = _extract_model_parameters(config)
    upstream_config = {"model_parameters": model_parameters}
    model = module.build_segformer3d_model(upstream_config)
    configure_normalization(model, config)
    return model


def upstream_provenance(upstream_home: str | Path | None = None) -> dict[str, str]:
    """返回用于实验日志的上游来源信息。"""
    home = ensure_upstream_available(upstream_home)
    return {
        "repository": "https://github.com/OSUPCVLab/SegFormer3D",
        "local_path": str(home),
        "license": "GPL-3.0 (以克隆仓库 LICENSE 为准)",
    }
