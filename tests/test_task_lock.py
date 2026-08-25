import json
from pathlib import Path

import yaml

from src.modeling.task_lock import compile_task_config, validate_task_spec


def _write_schema(path: Path, labels: list[int]) -> None:
    path.write_text(
        json.dumps({"labels": {str(value): f"V{value}" for value in labels}}),
        encoding="utf-8",
    )


def _write_spec(
    path: Path,
    schema_path: Path,
    *,
    task_type: str,
    labels: list[int],
    num_classes: int | None,
    locked: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "task_spec_version": 1,
                "task_id": "vertebra_test",
                "task_locked": locked,
                "dataset_name": "CTSpine1K",
                "task_type": task_type,
                "label_schema_path": str(schema_path),
                "foreground_labels": labels,
                "num_classes": num_classes,
                "processed_root": "data/processed_ctspine1k_real",
                "split_file": "data/splits/formal.json",
            }
        ),
        encoding="utf-8",
    )


def test_unlocked_template_is_not_ready() -> None:
    report = validate_task_spec("configs/task_specs/vertebra_task_template.json")
    assert report.ready is False
    codes = {issue.code for issue in report.issues}
    assert "task_not_locked" in codes
    assert "task_id_unset" in codes
    assert "invalid_task_type" in codes


def test_binary_task_compiles_locked_config(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    spec = tmp_path / "task.json"
    output = tmp_path / "locked_binary.yaml"
    _write_schema(schema, [1, 2, 3])
    _write_spec(
        spec,
        schema,
        task_type="binary_semantic",
        labels=[1, 2, 3],
        num_classes=2,
    )

    report = validate_task_spec(spec)
    assert report.ready is True
    assert report.label_mode == "binary"
    assert report.num_classes == 2

    compile_task_config(spec, "configs/orthopedic_ct_baseline.yaml", output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["data"]["label_mode"] == "binary"
    assert config["data"]["num_classes"] == 2
    assert config["model"]["num_classes"] == 2
    assert config["task"]["task_locked"] is True
    assert config["task"]["foreground_labels"] == [1, 2, 3]
    assert len(config["task"]["task_spec_sha256"]) == 64


def test_multiclass_task_requires_max_label_plus_background(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    spec = tmp_path / "task.json"
    output = tmp_path / "locked_multiclass.yaml"
    _write_schema(schema, [1, 2, 3])
    _write_spec(
        spec,
        schema,
        task_type="multiclass_semantic",
        labels=[1, 2, 3],
        num_classes=4,
    )

    report = validate_task_spec(spec)
    assert report.ready is True
    assert report.label_mode == "multiclass"
    assert report.num_classes == 4

    compile_task_config(spec, "configs/orthopedic_ct_baseline.yaml", output)
    config = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert config["data"]["label_mode"] == "multiclass"
    assert config["data"]["num_classes"] == 4
    assert config["model"]["num_classes"] == 4


def test_inconsistent_or_instance_task_is_rejected(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    _write_schema(schema, [1, 2, 3])

    inconsistent = tmp_path / "bad_classes.json"
    _write_spec(
        inconsistent,
        schema,
        task_type="multiclass_semantic",
        labels=[1, 2, 3],
        num_classes=3,
    )
    report = validate_task_spec(inconsistent)
    assert report.ready is False
    assert "num_classes_inconsistent" in {issue.code for issue in report.issues}

    instance = tmp_path / "instance.json"
    _write_spec(
        instance,
        schema,
        task_type="instance",
        labels=[1, 2, 3],
        num_classes=None,
    )
    instance_report = validate_task_spec(instance)
    assert instance_report.ready is False
    assert "instance_not_supported" in {issue.code for issue in instance_report.issues}
