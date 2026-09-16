import json
import os
from pathlib import Path

import torch
import yaml

from src.modeling.train import (
    RunAlreadyActiveError,
    acquire_run_lock,
    build_training_status,
    find_latest_compatible_last_checkpoint,
    load_model_initialization_checkpoint,
    format_duration,
    format_training_status_line,
    release_run_lock,
    write_training_heartbeat,
    write_training_status,
)


def _make_run(root: Path, name: str, config: dict, *, mtime: int) -> Path:
    run_dir = root / name
    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    checkpoint_path = checkpoint_dir / "last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    os.utime(checkpoint_path, (mtime, mtime))
    return checkpoint_path


def test_auto_resume_uses_latest_exact_config_match(tmp_path: Path) -> None:
    expected_config = {"experiment_name": "same", "training": {"epochs": 20}}
    other_config = {"experiment_name": "other", "training": {"epochs": 20}}

    older = _make_run(tmp_path, "older", expected_config, mtime=100)
    newest = _make_run(tmp_path, "newest", expected_config, mtime=300)
    _make_run(tmp_path, "wrong_config", other_config, mtime=500)

    selected = find_latest_compatible_last_checkpoint(
        expected_config, experiments_root=tmp_path
    )

    assert selected == newest
    assert selected != older


def test_auto_resume_returns_none_without_compatible_checkpoint(tmp_path: Path) -> None:
    _make_run(tmp_path, "run", {"training": {"epochs": 10}}, mtime=100)

    assert (
        find_latest_compatible_last_checkpoint(
            {"training": {"epochs": 20}}, experiments_root=tmp_path
        )
        is None
    )


def test_model_initialization_checkpoint_loads_weights_only(tmp_path: Path) -> None:
    source = torch.nn.Conv3d(1, 2, kernel_size=1, bias=True)
    with torch.no_grad():
        source.weight.fill_(0.25)
        source.bias.fill_(0.75)
    checkpoint_path = tmp_path / "source.pt"
    torch.save(
        {
            "epoch": 4,
            "val_dice": 0.28,
            "best_val_dice": 0.31,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": {"must_not_be_used": True},
        },
        checkpoint_path,
    )

    target = torch.nn.Conv3d(1, 2, kernel_size=1, bias=True)
    metadata = load_model_initialization_checkpoint(
        checkpoint_path,
        model=target,
        device=torch.device("cpu"),
    )

    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
    assert metadata["source_epoch"] == 4
    assert metadata["source_val_dice"] == 0.28
    assert metadata["source_best_val_dice"] == 0.31


def test_training_status_contains_progress_runtime_and_eta(tmp_path: Path) -> None:
    status = build_training_status(
        phase="training",
        epoch=3,
        max_epochs=10,
        step=5,
        total_steps=10,
        train_loss=1.25,
        val_dice=0.42,
        best_val_dice=0.5,
        lr=1e-4,
        completed_training_seconds=120.0,
        current_epoch_seconds=20.0,
    )

    assert status["progress_percent"] == 25.0
    assert status["elapsed_seconds"] == 140.0
    assert status["eta_seconds"] is not None
    assert status["eta_seconds"] > 0

    line = format_training_status_line(status)
    assert "epoch 3/10" in line
    assert "train loss=1.250000" in line
    assert "val Dice=0.420000" in line
    assert "best Dice=0.500000" in line
    assert "进度=25.0%" in line
    assert "ETA=" in line

    status_path = tmp_path / "training_status.json"
    write_training_status(status_path, status)
    saved = json.loads(status_path.read_text(encoding="utf-8"))
    assert saved["phase"] == "training"
    assert not (tmp_path / "training_status.json.tmp").exists()


def test_training_heartbeat_is_atomic_and_contains_pid(tmp_path: Path) -> None:
    heartbeat_path = tmp_path / "training_heartbeat.json"
    write_training_heartbeat(
        heartbeat_path,
        phase="training",
        epoch=7,
        step=120,
        total_steps=610,
    )

    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["phase"] == "training"
    assert payload["epoch"] == 7
    assert payload["step"] == 120
    assert payload["total_steps"] == 610
    assert not (tmp_path / "training_heartbeat.json.tmp").exists()


def test_run_lock_rejects_second_active_process(tmp_path: Path) -> None:
    lock_path = acquire_run_lock(tmp_path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        try:
            acquire_run_lock(tmp_path)
        except RunAlreadyActiveError:
            pass
        else:
            raise AssertionError("active run lock should reject a second acquire")
    finally:
        release_run_lock(lock_path)
    assert not lock_path.exists()


def test_run_lock_recovers_stale_pid(tmp_path: Path) -> None:
    stale_lock = tmp_path / "RUNNING.lock"
    stale_lock.write_text(
        json.dumps({"pid": 99999999, "hostname": "stale", "started_at": "old"}),
        encoding="utf-8",
    )

    lock_path = acquire_run_lock(tmp_path)
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
    finally:
        release_run_lock(lock_path)


def test_format_duration_is_stable() -> None:
    assert format_duration(0) == "00:00:00"
    assert format_duration(3661) == "01:01:01"
    assert format_duration(None) == "--:--:--"
