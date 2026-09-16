"""统一构建 3D segmentation 模型，便于在同一训练链路做架构对照。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.modeling.baselines import build_tiny_unet3d
from src.modeling.refined_segformer import build_refined_segformer3d
from src.modeling.segformer3d_hr_decoder import build_segformer3d_hr_decoder
from src.modeling.segformer3d_adapter import (
    build_orthopedic_segformer3d,
    upstream_provenance,
)


def build_segmentation_model(config: dict[str, Any]) -> torch.nn.Module:
    """按 ``model.name`` 构建模型；未知名称直接失败，禁止静默回退。"""
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "segformer3d")).strip().lower()
    if name in {"segformer3d", "segformer_3d"}:
        return build_orthopedic_segformer3d(config)
    if name in {"tiny_unet3d", "tiny-unet3d", "tiny_unet"}:
        return build_tiny_unet3d(config)
    if name in {"segformer3d_refined", "segformer_refined", "refined_segformer3d"}:
        return build_refined_segformer3d(config)
    if name in {"segformer3d_hr", "segformer3d_hr_decoder", "segformer3d_highres"}:
        return build_segformer3d_hr_decoder(config)
    raise ValueError(f"未知 model.name: {name!r}")


def model_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """记录当前模型来源，避免把本地 baseline 错标为 SegFormer3D 上游模型。"""
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "segformer3d")).strip().lower()
    if name in {"segformer3d", "segformer_3d"}:
        return {
            "model_name": "segformer3d",
            "source": "third_party",
            **upstream_provenance(),
        }
    if name in {"tiny_unet3d", "tiny-unet3d", "tiny_unet"}:
        return {
            "model_name": "tiny_unet3d",
            "source": "local_sanity_baseline",
            "implementation": str(Path(__file__).resolve().with_name("baselines.py")),
            "purpose": "micro_overfit_pipeline_isolation_only",
        }
    if name in {"segformer3d_refined", "segformer_refined", "refined_segformer3d"}:
        return {
            "model_name": "segformer3d_refined",
            "source": "third_party_plus_local_refinement",
            **upstream_provenance(),
            "refinement_implementation": str(
                Path(__file__).resolve().with_name("refined_segformer.py")
            ),
            "purpose": "high_resolution_input_guided_boundary_refinement",
        }
    if name in {"segformer3d_hr", "segformer3d_hr_decoder", "segformer3d_highres"}:
        return {
            "model_name": "segformer3d_hr_decoder",
            "source": "third_party_encoder_plus_local_high_resolution_decoder",
            **upstream_provenance(),
            "decoder_implementation": str(
                Path(__file__).resolve().with_name("segformer3d_hr_decoder.py")
            ),
            "purpose": "multiscale_segformer_feature_fusion_with_full_resolution_boundary_decoder",
        }
    raise ValueError(f"未知 model.name: {name!r}")
