from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from src.modeling.train import (
    configure_batchnorm_training_mode,
    should_freeze_batchnorm_running_stats,
)


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


def test_freeze_from_epoch2_keeps_epoch1_normal_then_anchors_running_stats() -> None:
    model = _small_model()
    batchnorm = model[1]
    train_cfg = {"freeze_batchnorm_running_stats_from_epoch": 2}
    image = torch.linspace(-2.0, 3.0, steps=64).reshape(1, 1, 4, 4, 4)

    initial_mean = batchnorm.running_mean.detach().clone()
    initial_var = batchnorm.running_var.detach().clone()
    initial_batches = int(batchnorm.num_batches_tracked.item())

    model.train()
    assert should_freeze_batchnorm_running_stats(train_cfg, epoch=1) is False
    configure_batchnorm_training_mode(model, freeze_running_stats=False)
    model(image)

    epoch1_mean = batchnorm.running_mean.detach().clone()
    epoch1_var = batchnorm.running_var.detach().clone()
    epoch1_batches = int(batchnorm.num_batches_tracked.item())
    assert batchnorm.training is True
    assert not torch.equal(epoch1_mean, initial_mean)
    assert not torch.equal(epoch1_var, initial_var)
    assert epoch1_batches == initial_batches + 1

    # 模拟 epoch1 validation 将全模型切到 eval，epoch2 必须先恢复 train，再冻结 BN。
    model.eval()
    model.train()
    assert should_freeze_batchnorm_running_stats(train_cfg, epoch=2) is True
    count = configure_batchnorm_training_mode(model, freeze_running_stats=True)
    assert count == 1
    assert model.training is True
    assert model[0].training is True
    assert batchnorm.training is False

    model.zero_grad(set_to_none=True)
    model(image).sum().backward()
    assert torch.equal(batchnorm.running_mean, epoch1_mean)
    assert torch.equal(batchnorm.running_var, epoch1_var)
    assert int(batchnorm.num_batches_tracked.item()) == epoch1_batches
    assert batchnorm.weight.grad is not None
    assert batchnorm.bias.grad is not None
    assert float(batchnorm.weight.grad.abs().sum().item()) > 0.0
    assert float(batchnorm.bias.grad.abs().sum().item()) > 0.0


def test_freeze_policy_supports_default_legacy_and_resume_epoch() -> None:
    assert should_freeze_batchnorm_running_stats({}, epoch=1) is False
    assert should_freeze_batchnorm_running_stats(
        {"freeze_batchnorm_running_stats": True}, epoch=1
    ) is True
    train_cfg = {"freeze_batchnorm_running_stats_from_epoch": 2}
    assert should_freeze_batchnorm_running_stats(train_cfg, epoch=1) is False
    # resume 到 epoch2 时训练循环会直接以 start_epoch=2 调用同一判定。
    assert should_freeze_batchnorm_running_stats(train_cfg, epoch=2) is True
    assert should_freeze_batchnorm_running_stats(train_cfg, epoch=3) is True


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


def test_v9_config_diff_is_only_experiment_name_and_bn_freeze_epoch() -> None:
    v6_path = PROJECT_ROOT / "configs" / "orthopedic_ct_cpu_binary_balanced_lr_v6.yaml"
    v9_path = (
        PROJECT_ROOT / "configs" / "orthopedic_ct_cpu_binary_bn_freeze_after_e1_v9.yaml"
    )
    v6 = yaml.safe_load(v6_path.read_text(encoding="utf-8"))
    v9 = yaml.safe_load(v9_path.read_text(encoding="utf-8"))

    assert v6["experiment_name"] == "cpu_binary_balanced_lr_v6_roi64"
    assert v9["experiment_name"] == "cpu_binary_bn_freeze_after_e1_v9_roi64"
    assert "freeze_batchnorm_running_stats" not in v9["training"]
    assert v9["training"]["freeze_batchnorm_running_stats_from_epoch"] == 2

    normalized_v6 = deepcopy(v6)
    normalized_v9 = deepcopy(v9)
    normalized_v6.pop("experiment_name")
    normalized_v9.pop("experiment_name")
    normalized_v9["training"].pop("freeze_batchnorm_running_stats_from_epoch")
    assert normalized_v9 == normalized_v6
