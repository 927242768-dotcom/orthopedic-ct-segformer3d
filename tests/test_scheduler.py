import torch

from src.modeling.train import WarmupCosineRestarts, build_scheduler


def test_warmup_cosine_scheduler_uses_config_and_changes_lr() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    config = {
        "scheduler": {
            "type": "cosine_annealing_warm_restarts",
            "warmup_epochs": 2,
            "t0_epochs": 4,
            "min_lr": 1e-5,
        }
    }

    scheduler = build_scheduler(config, optimizer)
    assert isinstance(scheduler, WarmupCosineRestarts)

    scheduler.step(1)
    assert abs(optimizer.param_groups[0]["lr"] - 5e-4) < 1e-10
    scheduler.step(2)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-3) < 1e-10
    scheduler.step(3)
    assert 1e-5 <= optimizer.param_groups[0]["lr"] <= 1e-3
    assert scheduler.state_dict()["warmup_epochs"] == 2
