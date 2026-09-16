"""SegFormer3D 多尺度高分辨率解码器。

核心思想：保留 SegFormer3D encoder 作为语义/长程上下文主干，但不再使用其
1/4 分辨率 logits 作为 coarse prior。直接读取 encoder 的 C1-C4 多尺度特征，
通过 U-Net 风格逐级融合恢复空间分辨率，并引入浅层 CT 分支补充骨边界细节。

该实现完全位于本项目内，不修改 third_party/SegFormer3D 上游源码，便于论文中
清楚区分“上游 SegFormer3D encoder”和“本项目高分辨率 decoder”两部分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.modeling.segformer3d_adapter import build_orthopedic_segformer3d


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(int(channels), int(requested)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvBlock3D(nn.Module):
    """适合 batch=1 的 Conv3D + GroupNorm 双卷积块。"""

    def __init__(self, in_channels: int, out_channels: int, *, groups: int) -> None:
        super().__init__()
        norm_groups = _group_count(out_channels, groups)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _Project3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, groups: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels, groups), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _FusionBlock3D(nn.Module):
    """把低分辨率 decoder feature 上采样后与 skip feature 做特征级融合。"""

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        groups: int,
    ) -> None:
        super().__init__()
        self.block = _ConvBlock3D(
            decoder_channels + skip_channels,
            out_channels,
            groups=groups,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-3:]) != tuple(skip.shape[-3:]):
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SegFormer3DHighResolutionDecoder(nn.Module):
    """SegFormer3D encoder + multi-scale feature fusion + full-resolution decoder。

    ``encoder`` 必须返回 C1-C4 四级 3D feature maps。默认 SegFormer3D 配置下，
    对 64^3 patch 的空间尺度约为 16^3/8^3/4^3/2^3。decoder 逐级恢复到 C1，
    再结合 1/2 与 1x 浅层 CT feature 恢复到原始输入分辨率。
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        in_channels: int,
        num_classes: int,
        embed_dims: Sequence[int],
        decoder_channels: Sequence[int] = (64, 64, 48, 32, 24, 16),
        shallow_channels: int = 16,
        groupnorm_groups: int = 8,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if len(embed_dims) != 4:
            raise ValueError("embed_dims 必须包含 SegFormer3D C1-C4 四级通道数")
        if len(decoder_channels) != 6:
            raise ValueError("decoder_channels 必须为 6 个值: C4/C3/C2/C1/half/full")
        if min(int(value) for value in decoder_channels) <= 0:
            raise ValueError("decoder_channels 必须全部 > 0")
        if shallow_channels <= 0:
            raise ValueError("shallow_channels 必须 > 0")

        self.encoder = encoder
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

        c1, c2, c3, c4 = [int(value) for value in embed_dims]
        d4, d3, d2, d1, d_half, d_full = [int(value) for value in decoder_channels]
        groups = int(groupnorm_groups)
        shallow = int(shallow_channels)

        # 原始 CT 的 full/half-resolution shallow spatial path，专门补偿 patch embedding
        # stride=4 对骨皮质、椎弓根等细边界造成的空间细节损失。
        self.shallow_full = _ConvBlock3D(int(in_channels), shallow, groups=groups)
        self.shallow_half = nn.Sequential(
            nn.Conv3d(int(in_channels), shallow, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(shallow, groups), shallow),
            nn.GELU(),
        )

        self.c4_project = _Project3D(c4, d4, groups=groups)
        self.fuse_c3 = _FusionBlock3D(d4, c3, d3, groups=groups)
        self.fuse_c2 = _FusionBlock3D(d3, c2, d2, groups=groups)
        self.fuse_c1 = _FusionBlock3D(d2, c1, d1, groups=groups)
        self.fuse_half = _FusionBlock3D(d1, shallow, d_half, groups=groups)
        self.fuse_full = _FusionBlock3D(d_half, shallow, d_full, groups=groups)
        self.head = nn.Conv3d(d_full, int(num_classes), kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow_full = self.shallow_full(x)
        shallow_half = self.shallow_half(x)

        features = self.encoder(x)
        if not isinstance(features, (list, tuple)) or len(features) != 4:
            raise RuntimeError("SegFormer3D encoder 必须返回 C1-C4 四级 feature maps")
        c1, c2, c3, c4 = features

        decoded = self.c4_project(c4)
        decoded = self.fuse_c3(decoded, c3)
        decoded = self.fuse_c2(decoded, c2)
        decoded = self.fuse_c1(decoded, c1)
        decoded = self.fuse_half(decoded, shallow_half)
        decoded = self.fuse_full(decoded, shallow_full)
        logits = self.head(decoded)

        # 奇数尺寸或自定义 patch_stride 下仍保证最终 logits 与输入严格同尺寸。
        if tuple(logits.shape[-3:]) != tuple(x.shape[-3:]):
            logits = F.interpolate(logits, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return logits


def _resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _extract_encoder_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """兼容 plain SegFormer、旧 refined wrapper 与本模型 checkpoint 的 encoder key。"""
    prefixes = (
        "segformer_encoder.",
        "base_model.segformer_encoder.",
        "encoder.",
    )
    for prefix in prefixes:
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    raise ValueError("checkpoint 中未找到可识别的 SegFormer3D encoder 权重")


def load_encoder_initialization(encoder: nn.Module, checkpoint_value: str | Path) -> dict[str, Any]:
    """只加载 SegFormer3D encoder 权重，不恢复旧 decoder/optimizer/epoch。"""
    checkpoint_path = _resolve_checkpoint(checkpoint_value)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"encoder_init_checkpoint 不存在: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint model_state_dict 必须为 mapping")
    encoder_state = _extract_encoder_state_dict(state_dict)
    encoder.load_state_dict(encoder_state, strict=True)
    return {
        "checkpoint": str(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1,
        "source_val_dice": float(checkpoint.get("val_dice", -1.0)) if isinstance(checkpoint, dict) else -1.0,
    }


def build_segformer3d_hr_decoder(config: dict[str, Any]) -> SegFormer3DHighResolutionDecoder:
    """从项目 config 构建改进 SegFormer3D 主模型。"""
    model_cfg = config.get("model", {})
    base_model = build_orthopedic_segformer3d(config)
    encoder = base_model.segformer_encoder

    encoder_init = model_cfg.get("encoder_init_checkpoint")
    if encoder_init:
        info = load_encoder_initialization(encoder, encoder_init)
        print(
            "[SEGFORMER3D-HR] encoder initialization loaded: "
            f"epoch={info['source_epoch']} val_dice={info['source_val_dice']:.6f}"
        )

    return SegFormer3DHighResolutionDecoder(
        encoder,
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        embed_dims=tuple(int(value) for value in model_cfg["embed_dims"]),
        decoder_channels=tuple(
            int(value)
            for value in model_cfg.get("hr_decoder_channels", [64, 64, 48, 32, 24, 16])
        ),
        shallow_channels=int(model_cfg.get("hr_shallow_channels", 16)),
        groupnorm_groups=int(model_cfg.get("hr_groupnorm_groups", 8)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
    )
