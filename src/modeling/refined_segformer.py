"""SegFormer3D 的轻量高分辨率输入引导 refinement wrapper。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.modeling.baselines import TinyUNet3D
from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d


def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(int(channels), int(requested)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class SegFormer3DInputRefinement(nn.Module):
    """用原始 CT 引导半/全分辨率 residual logits，补偿 1/4 decoder 的边界损失。"""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        in_channels: int,
        num_classes: int,
        refinement_channels: int = 16,
        refinement_downsample: int = 2,
        groupnorm_groups: int = 8,
        freeze_base_model: bool = False,
        refinement_arch: str = "conv",
        fusion_mode: str = "residual",
        fusion_alpha_init: float = 0.25,
        coarse_scale_init: float = 0.25,
    ) -> None:
        super().__init__()
        if refinement_downsample not in {1, 2, 4}:
            raise ValueError("refinement_downsample 仅支持 1/2/4")
        if refinement_channels <= 0:
            raise ValueError("refinement_channels 必须 > 0")
        self.base_model = base_model
        self.num_classes = int(num_classes)
        self.refinement_downsample = int(refinement_downsample)
        self.freeze_base_model = bool(freeze_base_model)
        self.refinement_arch = str(refinement_arch).strip().lower()
        self.fusion_mode = str(fusion_mode).strip().lower()
        if self.fusion_mode not in {"residual", "gated_direct", "direct_plus_coarse"}:
            raise ValueError("fusion_mode 仅支持 residual / gated_direct / direct_plus_coarse")
        if self.fusion_mode in {"gated_direct", "direct_plus_coarse"} and not self.freeze_base_model:
            raise ValueError("direct fusion 要求 freeze_base_model=true，避免 coarse 同时漂移")
        if self.freeze_base_model:
            for parameter in self.base_model.parameters():
                parameter.requires_grad_(False)
            self.base_model.eval()

        channels = int(refinement_channels)
        refinement_input_channels = (
            int(in_channels)
            if self.fusion_mode in {"gated_direct", "direct_plus_coarse"}
            else int(in_channels) + self.num_classes
        )
        if self.refinement_arch == "conv":
            groups = _group_count(channels, int(groupnorm_groups))
            self.refinement = nn.Sequential(
                nn.Conv3d(refinement_input_channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(groups, channels),
                nn.GELU(),
                nn.Conv3d(channels, self.num_classes, 1),
            )
            final_conv = self.refinement[-1]
        elif self.refinement_arch in {"tiny_unet", "tiny_unet3d", "unet"}:
            self.refinement = TinyUNet3D(
                refinement_input_channels,
                self.num_classes,
                base_channels=channels,
            )
            final_conv = self.refinement.head
        else:
            raise ValueError(
                "refinement_arch 仅支持 conv 或 tiny_unet3d，"
                f"当前={self.refinement_arch!r}"
            )

        # residual head 从严格 0 开始，warm-start 时初始输出与 coarse 模型完全一致。
        nn.init.zeros_(final_conv.weight)
        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

        if self.fusion_mode == "gated_direct":
            alpha = float(fusion_alpha_init)
            if not 0.0 < alpha < 1.0:
                raise ValueError("fusion_alpha_init 必须位于 (0,1)")
            self.fusion_logit = nn.Parameter(
                torch.tensor(math.log(alpha / (1.0 - alpha)), dtype=torch.float32)
            )
        else:
            self.register_parameter("fusion_logit", None)

        if self.fusion_mode == "direct_plus_coarse":
            self.coarse_scale = nn.Parameter(
                torch.tensor(float(coarse_scale_init), dtype=torch.float32)
            )
        else:
            self.register_parameter("coarse_scale", None)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base_model:
            # batch=1 时禁止 coarse SegFormer 的 BN running stats 再漂移。
            self.base_model.eval()
            self.refinement.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse = self.base_model(x)
        full_size = tuple(int(v) for v in x.shape[-3:])
        if tuple(coarse.shape[-3:]) != full_size:
            coarse = F.interpolate(coarse, size=full_size, mode="trilinear", align_corners=False)

        if self.refinement_downsample == 1:
            refine_size = full_size
        else:
            refine_size = tuple(
                max(1, int((value + self.refinement_downsample - 1) // self.refinement_downsample))
                for value in full_size
            )

        image_ref = (
            x
            if tuple(x.shape[-3:]) == refine_size
            else F.interpolate(x, size=refine_size, mode="trilinear", align_corners=False)
        )
        coarse_ref = (
            coarse
            if tuple(coarse.shape[-3:]) == refine_size
            else F.interpolate(coarse, size=refine_size, mode="trilinear", align_corners=False)
        )
        refinement_input = (
            image_ref
            if self.fusion_mode in {"gated_direct", "direct_plus_coarse"}
            else torch.cat([image_ref, coarse_ref], dim=1)
        )
        refined_logits = self.refinement(refinement_input)
        if tuple(refined_logits.shape[-3:]) != full_size:
            refined_logits = F.interpolate(
                refined_logits,
                size=full_size,
                mode="trilinear",
                align_corners=False,
            )
        if self.fusion_mode == "gated_direct":
            alpha = torch.sigmoid(self.fusion_logit).to(dtype=coarse.dtype)
            return coarse + alpha * (refined_logits - coarse.detach())
        if self.fusion_mode == "direct_plus_coarse":
            scale = self.coarse_scale.to(dtype=coarse.dtype)
            return refined_logits + scale * coarse.detach()
        return coarse + refined_logits


def _load_base_checkpoint(base_model: nn.Module, checkpoint_value: str | Path) -> dict[str, Any]:
    """只给 coarse SegFormer 加载已有权重，不恢复 optimizer/scheduler。"""
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"refined SegFormer base checkpoint 不存在: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    base_model.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "val_dice": float(checkpoint.get("val_dice", -1.0)) if isinstance(checkpoint, dict) else -1.0,
    }


def build_refined_segformer3d(config: dict[str, Any]) -> SegFormer3DInputRefinement:
    model_cfg = config.get("model", {})
    base_model = build_orthopedic_segformer3d(config)
    base_checkpoint = model_cfg.get("base_init_checkpoint")
    if base_checkpoint:
        info = _load_base_checkpoint(base_model, base_checkpoint)
        print(
            "[REFINED-SEGFORMER] coarse base checkpoint loaded: "
            f"epoch={info['epoch']} val_dice={info['val_dice']:.6f}"
        )
    return SegFormer3DInputRefinement(
        base_model,
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        refinement_channels=int(model_cfg.get("refinement_channels", 16)),
        refinement_downsample=int(model_cfg.get("refinement_downsample", 2)),
        groupnorm_groups=int(model_cfg.get("refinement_groupnorm_groups", 8)),
        freeze_base_model=bool(model_cfg.get("freeze_base_model", False)),
        refinement_arch=str(model_cfg.get("refinement_arch", "conv")),
        fusion_mode=str(model_cfg.get("fusion_mode", "residual")),
        fusion_alpha_init=float(model_cfg.get("fusion_alpha_init", 0.25)),
        coarse_scale_init=float(model_cfg.get("coarse_scale_init", 0.25)),
    )
