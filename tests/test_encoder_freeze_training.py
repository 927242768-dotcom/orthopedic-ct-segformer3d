from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.modeling.train import (
    configure_encoder_parameter_training,
    should_freeze_encoder_parameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TinySegFormerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.segformer_encoder = torch.nn.Conv3d(1, 2, kernel_size=1)
        self.segformer_decoder = torch.nn.Conv3d(2, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segformer_decoder(self.segformer_encoder(x))


def test_encoder_freeze_from_epoch2_policy_supports_resume() -> None:
    train_cfg = {"freeze_encoder_parameters_from_epoch": 2}

    assert should_freeze_encoder_parameters({}, epoch=1) is False
    assert should_freeze_encoder_parameters(train_cfg, epoch=1) is False
    assert should_freeze_encoder_parameters(train_cfg, epoch=2) is True
    assert should_freeze_encoder_parameters(train_cfg, epoch=3) is True


def test_encoder_freeze_only_disables_encoder_gradients() -> None:
    torch.manual_seed(0)
    model = _TinySegFormerLike()
    frozen_numel = configure_encoder_parameter_training(model, freeze_parameters=True)

    assert frozen_numel == sum(p.numel() for p in model.segformer_encoder.parameters())
    assert all(not p.requires_grad for p in model.segformer_encoder.parameters())
    assert all(p.requires_grad for p in model.segformer_decoder.parameters())

    model(torch.ones((1, 1, 2, 2, 2), dtype=torch.float32)).sum().backward()

    assert all(p.grad is None for p in model.segformer_encoder.parameters())
    assert all(p.grad is not None for p in model.segformer_decoder.parameters())


def test_encoder_trainability_can_be_restored_for_epoch1_behavior() -> None:
    model = _TinySegFormerLike()
    configure_encoder_parameter_training(model, freeze_parameters=True)
    configure_encoder_parameter_training(model, freeze_parameters=False)

    assert all(p.requires_grad for p in model.segformer_encoder.parameters())
    assert all(p.requires_grad for p in model.segformer_decoder.parameters())


def test_v10_config_diff_is_only_experiment_name_and_encoder_freeze_epoch() -> None:
    v9_path = (
        PROJECT_ROOT / "configs" / "orthopedic_ct_cpu_binary_bn_freeze_after_e1_v9.yaml"
    )
    v10_path = (
        PROJECT_ROOT
        / "configs"
        / "orthopedic_ct_cpu_binary_encoder_freeze_after_e1_v10.yaml"
    )
    v9 = yaml.safe_load(v9_path.read_text(encoding="utf-8"))
    v10 = yaml.safe_load(v10_path.read_text(encoding="utf-8"))

    assert v9["experiment_name"] == "cpu_binary_bn_freeze_after_e1_v9_roi64"
    assert v10["experiment_name"] == "cpu_binary_encoder_freeze_after_e1_v10_roi64"
    assert "freeze_encoder_parameters_from_epoch" not in v9["training"]
    assert v10["training"]["freeze_encoder_parameters_from_epoch"] == 2
    assert v10["training"]["freeze_batchnorm_running_stats_from_epoch"] == 2

    normalized_v9 = deepcopy(v9)
    normalized_v10 = deepcopy(v10)
    normalized_v9.pop("experiment_name")
    normalized_v10.pop("experiment_name")
    normalized_v10["training"].pop("freeze_encoder_parameters_from_epoch")
    assert normalized_v10 == normalized_v9
