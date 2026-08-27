from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.modeling.train import configure_batchnorm_training_mode


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_model() -> torch.nn.Sequential:
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Conv3d(1, 2, kernel_size=1, bias=False),
        torch.nn.BatchNorm3d(2),
    )


def test_freeze_batchnorm_running_stats_keeps_affine_trainable() -> None:
    model = _small_model()
    model.train()
    batchnorm = model[1]
    before_mean = batchnorm.running_mean.detach().clone()
    before_var = batchnorm.running_var.detach().clone()
    before_batches = int(batchnorm.num_batches_tracked.item())

    count = configure_batchnorm_training_mode(model, freeze_running_stats=True)

    assert count == 1
    assert model.training is True
    assert model[0].training is True
    assert batchnorm.training is False
    assert batchnorm.weight.requires_grad is True
    assert batchnorm.bias.requires_grad is True

    output = model(torch.ones((1, 1, 4, 4, 4), dtype=torch.float32))
    output.sum().backward()

    assert torch.equal(batchnorm.running_mean, before_mean)
    assert torch.equal(batchnorm.running_var, before_var)
    assert int(batchnorm.num_batches_tracked.item()) == before_batches
    assert batchnorm.weight.grad is not None
    assert batchnorm.bias.grad is not None
    assert float(batchnorm.weight.grad.abs().sum().item()) > 0.0
    assert float(batchnorm.bias.grad.abs().sum().item()) > 0.0


def test_default_batchnorm_training_behavior_is_unchanged() -> None:
    model = _small_model()
    model.train()
    batchnorm = model[1]
    before_mean = batchnorm.running_mean.detach().clone()
    before_batches = int(batchnorm.num_batches_tracked.item())

    count = configure_batchnorm_training_mode(model, freeze_running_stats=False)
    model(torch.ones((1, 1, 4, 4, 4), dtype=torch.float32))

    assert count == 0
    assert batchnorm.training is True
    assert int(batchnorm.num_batches_tracked.item()) == before_batches + 1
    assert not torch.equal(batchnorm.running_mean, before_mean)


def test_disabled_helper_does_not_change_eval_inference_state() -> None:
    model = _small_model()
    model.eval()

    count = configure_batchnorm_training_mode(model, freeze_running_stats=False)

    assert count == 0
    assert model.training is False
    assert model[0].training is False
    assert model[1].training is False


def test_v8_config_diff_is_only_experiment_name_and_bn_option() -> None:
    v6_path = PROJECT_ROOT / "configs" / "orthopedic_ct_cpu_binary_balanced_lr_v6.yaml"
    v8_path = PROJECT_ROOT / "configs" / "orthopedic_ct_cpu_binary_bn_frozen_v8.yaml"
    v6 = yaml.safe_load(v6_path.read_text(encoding="utf-8"))
    v8 = yaml.safe_load(v8_path.read_text(encoding="utf-8"))

    assert v6["experiment_name"] == "cpu_binary_balanced_lr_v6_roi64"
    assert v8["experiment_name"] == "cpu_binary_bn_frozen_v8_roi64"
    assert "freeze_batchnorm_running_stats" not in v6["training"]
    assert v8["training"]["freeze_batchnorm_running_stats"] is True

    normalized_v6 = deepcopy(v6)
    normalized_v8 = deepcopy(v8)
    normalized_v6.pop("experiment_name")
    normalized_v8.pop("experiment_name")
    normalized_v8["training"].pop("freeze_batchnorm_running_stats")
    assert normalized_v8 == normalized_v6
