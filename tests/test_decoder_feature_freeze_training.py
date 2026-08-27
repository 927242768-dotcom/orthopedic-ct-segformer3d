from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.modeling.train import (
    configure_decoder_feature_parameter_training,
    should_freeze_decoder_feature_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TinyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_c1 = torch.nn.Conv3d(2, 2, kernel_size=1)
        self.linear_fuse = torch.nn.Sequential(
            torch.nn.Conv3d(2, 2, kernel_size=1),
            torch.nn.BatchNorm3d(2),
        )
        self.linear_pred = torch.nn.Conv3d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_c1(x)
        x = self.linear_fuse(x)
        return self.linear_pred(x)


class _TinySegFormerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segformer_encoder = torch.nn.Conv3d(1, 2, kernel_size=1)
        self.segformer_decoder = _TinyDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segformer_decoder(self.segformer_encoder(x))


def test_decoder_feature_freeze_from_epoch2_policy_supports_resume() -> None:
    train_cfg = {"freeze_decoder_feature_parameters_from_epoch": 2}

    assert should_freeze_decoder_feature_parameters({}, epoch=1) is False
    assert should_freeze_decoder_feature_parameters(train_cfg, epoch=1) is False
    assert should_freeze_decoder_feature_parameters(train_cfg, epoch=2) is True
    assert should_freeze_decoder_feature_parameters(train_cfg, epoch=3) is True


def test_decoder_feature_freeze_keeps_only_linear_pred_trainable() -> None:
    torch.manual_seed(0)
    model = _TinySegFormerLike()
    frozen_numel = configure_decoder_feature_parameter_training(
        model,
        freeze_parameters=True,
    )

    decoder = model.segformer_decoder
    expected_frozen_numel = sum(
        p.numel()
        for name, p in decoder.named_parameters()
        if not name.startswith("linear_pred.")
    )
    assert frozen_numel == expected_frozen_numel
    assert all(
        not p.requires_grad
        for name, p in decoder.named_parameters()
        if not name.startswith("linear_pred.")
    )
    assert all(p.requires_grad for p in decoder.linear_pred.parameters())
    assert all(p.requires_grad for p in model.segformer_encoder.parameters())

    model(torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)).sum().backward()

    assert all(
        p.grad is None
        for name, p in decoder.named_parameters()
        if not name.startswith("linear_pred.")
    )
    assert all(p.grad is not None for p in decoder.linear_pred.parameters())


def test_decoder_feature_trainability_can_be_restored() -> None:
    model = _TinySegFormerLike()
    configure_decoder_feature_parameter_training(model, freeze_parameters=True)
    configure_decoder_feature_parameter_training(model, freeze_parameters=False)

    assert all(p.requires_grad for p in model.segformer_decoder.parameters())


def test_v11_config_diff_is_only_experiment_name_and_decoder_feature_freeze() -> None:
    v10_path = (
        PROJECT_ROOT
        / "configs"
        / "orthopedic_ct_cpu_binary_encoder_freeze_after_e1_v10.yaml"
    )
    v11_path = (
        PROJECT_ROOT
        / "configs"
        / "orthopedic_ct_cpu_binary_decoder_feature_freeze_after_e1_v11.yaml"
    )
    v10 = yaml.safe_load(v10_path.read_text(encoding="utf-8"))
    v11 = yaml.safe_load(v11_path.read_text(encoding="utf-8"))

    assert v10["experiment_name"] == "cpu_binary_encoder_freeze_after_e1_v10_roi64"
    assert v11["experiment_name"] == "cpu_binary_decoder_feature_freeze_after_e1_v11_roi64"
    assert "freeze_decoder_feature_parameters_from_epoch" not in v10["training"]
    assert v11["training"]["freeze_decoder_feature_parameters_from_epoch"] == 2
    assert v11["training"]["freeze_encoder_parameters_from_epoch"] == 2
    assert v11["training"]["freeze_batchnorm_running_stats_from_epoch"] == 2

    normalized_v10 = deepcopy(v10)
    normalized_v11 = deepcopy(v11)
    normalized_v10.pop("experiment_name")
    normalized_v11.pop("experiment_name")
    normalized_v11["training"].pop("freeze_decoder_feature_parameters_from_epoch")
    assert normalized_v11 == normalized_v10
