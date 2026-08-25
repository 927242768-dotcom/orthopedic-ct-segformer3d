"""正式骨科 CT 任务规格锁定与训练配置编译。

该模块不替项目组决定任务。它只在 ``task_locked=true`` 且任务定义完整时，
把已确认的 binary/multiclass 语义任务写入新的 YAML，并记录 task-spec 指纹，
避免手工修改 ``label_mode`` / ``num_classes`` / split 路径造成不可追溯实验。

当前 SegFormer3D 训练链只支持 binary / multiclass semantic segmentation；
instance segmentation 必须先实现实例级数据与评价链，不能用语义分割配置冒充。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_TASK_TYPES = {"binary_semantic", "multiclass_semantic", "instance"}


@dataclass(frozen=True)
class TaskSpecIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class TaskSpecReport:
    ready: bool
    task_id: str
    task_type: str
    label_mode: str | None
    num_classes: int | None
    foreground_labels: list[int]
    task_spec_sha256: str
    issues: list[TaskSpecIssue]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issue_counts"] = {
            "error": sum(issue.severity == "error" for issue in self.issues),
            "warning": sum(issue.severity == "warning" for issue in self.issues),
        }
        return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _issue(issues: list[TaskSpecIssue], severity: str, code: str, message: str) -> None:
    issues.append(TaskSpecIssue(severity=severity, code=code, message=message))


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task spec 必须是 JSON object")
    return payload


def validate_task_spec(task_spec_path: str | Path) -> TaskSpecReport:
    path = _resolve(task_spec_path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = _load_json(path)
    issues: list[TaskSpecIssue] = []

    task_id = str(payload.get("task_id", "")).strip()
    if not task_id or task_id.upper() == "TBD":
        _issue(issues, "error", "task_id_unset", "task_id 尚未确定")

    task_locked = bool(payload.get("task_locked", False))
    if not task_locked:
        _issue(
            issues,
            "error",
            "task_not_locked",
            "task_locked=false：该规格只能作为模板，禁止生成正式训练配置",
        )

    task_type = str(payload.get("task_type", "")).strip()
    if task_type not in _ALLOWED_TASK_TYPES:
        _issue(
            issues,
            "error",
            "invalid_task_type",
            f"task_type 必须为 {sorted(_ALLOWED_TASK_TYPES)}，当前={task_type!r}",
        )

    raw_labels = payload.get("foreground_labels", [])
    labels: list[int] = []
    if not isinstance(raw_labels, list) or not raw_labels:
        _issue(issues, "error", "foreground_labels_missing", "foreground_labels 必须是非空列表")
    else:
        try:
            labels = sorted({int(value) for value in raw_labels})
        except (TypeError, ValueError):
            _issue(issues, "error", "foreground_labels_invalid", "foreground_labels 必须为整数")
            labels = []
        if labels and any(value <= 0 for value in labels):
            _issue(issues, "error", "foreground_labels_nonpositive", "前景标签必须全部 > 0")

    label_mode: str | None = None
    expected_num_classes: int | None = None
    if task_type == "binary_semantic":
        label_mode = "binary"
        expected_num_classes = 2
    elif task_type == "multiclass_semantic":
        label_mode = "multiclass"
        if labels:
            expected_num_classes = max(labels) + 1
    elif task_type == "instance":
        _issue(
            issues,
            "error",
            "instance_not_supported",
            "当前训练链只支持 binary/multiclass semantic；instance 任务必须先补实例级实现",
        )

    raw_num_classes = payload.get("num_classes")
    declared_num_classes: int | None = None
    if raw_num_classes is not None:
        try:
            declared_num_classes = int(raw_num_classes)
        except (TypeError, ValueError):
            _issue(issues, "error", "num_classes_invalid", "num_classes 必须为整数")
    if expected_num_classes is not None:
        if declared_num_classes is None:
            _issue(
                issues,
                "error",
                "num_classes_missing",
                f"当前任务应显式设置 num_classes={expected_num_classes}",
            )
        elif declared_num_classes != expected_num_classes:
            _issue(
                issues,
                "error",
                "num_classes_inconsistent",
                f"task_type/foreground_labels 推导 num_classes={expected_num_classes}，但规格填写 {declared_num_classes}",
            )

    schema_path = payload.get("label_schema_path")
    if schema_path:
        resolved_schema = _resolve(str(schema_path))
        if not resolved_schema.exists():
            _issue(issues, "error", "label_schema_missing", f"label schema 不存在: {resolved_schema}")
        else:
            try:
                schema = _load_json(resolved_schema)
                schema_labels = {
                    int(key) for key in (schema.get("labels") or {}).keys()
                }
                missing = sorted(set(labels) - schema_labels)
                if missing:
                    _issue(
                        issues,
                        "error",
                        "labels_not_in_schema",
                        f"以下前景标签不在 label schema 中: {missing}",
                    )
            except Exception as exc:
                _issue(
                    issues,
                    "error",
                    "label_schema_invalid",
                    f"label schema 无法解析: {type(exc).__name__}: {exc}",
                )

    for key in ("dataset_name", "processed_root", "split_file"):
        value = str(payload.get(key, "")).strip()
        if not value or "TBD" in value.upper():
            _issue(issues, "error", f"{key}_unset", f"{key} 尚未确定")

    return TaskSpecReport(
        ready=not any(issue.severity == "error" for issue in issues),
        task_id=task_id,
        task_type=task_type,
        label_mode=label_mode,
        num_classes=expected_num_classes,
        foreground_labels=labels,
        task_spec_sha256=_sha256(path),
        issues=issues,
    )


def compile_task_config(
    task_spec_path: str | Path,
    base_config_path: str | Path,
    output_path: str | Path,
) -> Path:
    spec_path = _resolve(task_spec_path)
    base_path = _resolve(base_config_path)
    output = _resolve(output_path)
    report = validate_task_spec(spec_path)
    if not report.ready:
        messages = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
        raise ValueError(f"task spec 未通过，不能编译正式 config: {messages}")
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if output.exists():
        raise FileExistsError(output)

    spec = _load_json(spec_path)
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("base config 必须为 YAML mapping")
    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})
    if not isinstance(data_cfg, dict) or not isinstance(model_cfg, dict):
        raise ValueError("config.data / config.model 必须为 mapping")

    data_cfg["dataset_name"] = str(spec["dataset_name"])
    data_cfg["processed_root"] = str(spec["processed_root"])
    data_cfg["split_file"] = str(spec["split_file"])
    data_cfg["label_mode"] = str(report.label_mode)
    data_cfg["num_classes"] = int(report.num_classes or 0)
    model_cfg["num_classes"] = int(report.num_classes or 0)
    config["task"] = {
        "task_id": report.task_id,
        "task_type": report.task_type,
        "task_locked": True,
        "task_spec": str(spec_path),
        "task_spec_sha256": report.task_spec_sha256,
        "label_schema_path": spec.get("label_schema_path"),
        "foreground_labels": report.foreground_labels,
        "note": "Generated from an explicitly locked task spec; do not hand-edit task-defining fields.",
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Validate/compile a locked orthopedic CT task spec")
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--output-config", type=Path, default=None)
    args = parser.parse_args()

    report = validate_task_spec(args.task_spec)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if not report.ready:
        raise SystemExit(2)
    if args.output_config is not None:
        if args.base_config is None:
            parser.error("指定 --output-config 时必须同时指定 --base-config")
        output = compile_task_config(args.task_spec, args.base_config, args.output_config)
        print(f"Locked config written: {output}")


if __name__ == "__main__":
    main()
