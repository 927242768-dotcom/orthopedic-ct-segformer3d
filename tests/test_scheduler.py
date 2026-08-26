import torch

from src.modeling.train import (
    WarmupCosineRestarts,
    build_scheduler,
    load_training_checkpoint,
    save_checkpoint,
)


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


def test_training_checkpoint_round_trip_restores_resume_state(tmp_path) -> None:
    config = {
        "scheduler": {
            "type": "cosine_annealing_warm_restarts",
            "warmup_epochs": 2,
            "t0_epochs": 4,
            "min_lr": 1e-5,
        }
    }
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(config, optimizer)
    assert isinstance(scheduler, WarmupCosineRestarts)

    x = torch.tensor([[1.0, 2.0, 3.0]])
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    scheduler.step(3)

    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        val_dice=0.4,
        best_val_dice=0.5,
        epochs_without_improvement=2,
        config=config,
        scheduler=scheduler,
    )

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = build_scheduler(config, restored_optimizer)
    assert isinstance(restored_scheduler, WarmupCosineRestarts)

    state = load_training_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config=config,
        device=torch.device("cpu"),
    )

    assert state == {
        "epoch": 3,
        "start_epoch": 4,
        "best_val_dice": 0.5,
        "epochs_without_improvement": 2,
    }
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert restored_scheduler.state_dict() == scheduler.state_dict()
