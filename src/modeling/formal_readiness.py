"""正式 3D CT 实验的一站式就绪检查。

该模块把三类已有保护合并到一份 machine-readable report：
1. task spec 是否已由项目组明确锁定；
2. GPU / CUDA 环境是否满足正式 3D 训练最低条件；
3. 数据、split、人工 QC 与训练配置是否通过 formal preflight。

它不会替项目组选择任务，也不会修改任务规格、split、驱动或训练环境。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from src.modeling.gpu_environment import GPUEnvironmentReport, inspect_gpu_environment
from src.modeling.preflight import PreflightReport, run_preflight
from src.modeling.task_lock import TaskSpecReport, validate_task_spec

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FormalReadinessBlocker:
    category: str
    code: str
    message: str


@dataclass(frozen=True)
class FormalReadinessReport:
    ready: bool
    task_spec: str
    config: str
    minimum_memory_gb: float
    processed_root_override: str | None
    split_file_override: str | None
    task: dict[str, Any]
    gpu: dict[str, Any]
    preflight: dict[str, Any]
    blockers: list[FormalReadinessBlocker]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocker_count"] = len(self.blockers)
        return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _config_task_binding_blockers(
    config_path: str | Path,
    task_report: TaskSpecReport,
) -> list[FormalReadinessBlocker]:
    """确认正式 config 确实由当前已锁定 task spec 编译而来。"""
    if not task_report.ready:
        return []

    path = _resolve(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return [
            FormalReadinessBlocker(
                category="config",
                code="config_not_mapping",
                message="正式 config 必须是 YAML mapping",
            )
        ]

    task_cfg = payload.get("task")
    if not isinstance(task_cfg, dict) or task_cfg.get("task_locked") is not True:
        return [
            FormalReadinessBlocker(
                category="config",
                code="config_not_compiled_from_task_spec",
                message="config 缺少已锁定 task 元数据；请使用 task_lock 编译正式 config，而不是手工改 baseline",
            )
        ]

    blockers: list[FormalReadinessBlocker] = []
    if str(task_cfg.get("task_id", "")).strip() != task_report.task_id:
        blockers.append(
            FormalReadinessBlocker(
                category="config",
                code="config_task_id_mismatch",
                message=(
                    f"config.task_id={task_cfg.get('task_id')!r} 与当前 task spec "
                    f"{task_report.task_id!r} 不一致"
                ),
            )
        )
    if str(task_cfg.get("task_spec_sha256", "")).strip() != task_report.task_spec_sha256:
        blockers.append(
            FormalReadinessBlocker(
                category="config",
                code="config_task_fingerprint_mismatch",
                message="config 中 task_spec_sha256 与当前 task spec 不一致；任务规格可能在编译后被修改",
            )
        )
    return blockers


def combine_readiness_reports(
    task_report: TaskSpecReport,
    gpu_report: GPUEnvironmentReport,
    preflight_report: PreflightReport,
    *,
    config_binding_blockers: list[FormalReadinessBlocker] | None = None,
    require_gpu: bool = True,
) -> list[FormalReadinessBlocker]:
    """把三份报告中的正式 blocker 归一化，方便 CI/脚本统一消费。"""
    blockers: list[FormalReadinessBlocker] = []
    for issue in task_report.issues:
        if issue.severity == "error":
            blockers.append(
                FormalReadinessBlocker(
                    category="task",
                    code=issue.code,
                    message=issue.message,
                )
            )

    if require_gpu and not gpu_report.ready:
        blockers.append(
            FormalReadinessBlocker(
                category="gpu",
                code="gpu_environment_not_ready",
                message="；".join(gpu_report.issues) or "GPU/CUDA 环境未通过",
            )
        )

    for issue in preflight_report.issues:
        if issue.severity == "error":
            suffix = f" [case={issue.case_id}]" if issue.case_id else ""
            blockers.append(
                FormalReadinessBlocker(
                    category="preflight",
                    code=issue.code,
                    message=issue.message + suffix,
                )
            )

    blockers.extend(config_binding_blockers or [])
    return blockers


def run_formal_readiness(
    task_spec_path: str | Path,
    config_path: str | Path,
    *,
    minimum_memory_gb: float = 8.0,
    processed_root_override: str | Path | None = None,
    split_file_override: str | Path | None = None,
    allow_cpu: bool = False,
) -> FormalReadinessReport:
    if minimum_memory_gb <= 0:
        raise ValueError("minimum_memory_gb 必须 > 0")

    task_report = validate_task_spec(task_spec_path)
    gpu_report = inspect_gpu_environment(minimum_memory_gb=minimum_memory_gb)
    # GPU 由 gpu_environment 做更严格的一致验收；这里关闭 preflight 内的重复 GPU blocker。
    preflight_report = run_preflight(
        config_path,
        mode="formal",
        require_gpu=False,
        require_human_qc=True,
        processed_root_override=processed_root_override,
        split_file_override=split_file_override,
    )
    binding_blockers = _config_task_binding_blockers(config_path, task_report)
    blockers = combine_readiness_reports(
        task_report,
        gpu_report,
        preflight_report,
        config_binding_blockers=binding_blockers,
        require_gpu=not allow_cpu,
    )

    return FormalReadinessReport(
        ready=not blockers,
        task_spec=str(_resolve(task_spec_path)),
        config=str(_resolve(config_path)),
        minimum_memory_gb=float(minimum_memory_gb),
        processed_root_override=(
            None if processed_root_override is None else str(_resolve(processed_root_override))
        ),
        split_file_override=(
            None if split_file_override is None else str(_resolve(split_file_override))
        ),
        task=task_report.to_dict(),
        gpu=gpu_report.to_dict(),
        preflight=preflight_report.to_dict(),
        blockers=blockers,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="骨科 CT 正式实验一站式 readiness 检查")
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=Path("configs/task_specs/vertebra_task_template.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/orthopedic_ct_baseline.yaml"),
    )
    parser.add_argument("--minimum-memory-gb", type=float, default=8.0)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="允许正式 readiness 不把无 CUDA 作为 blocker；CPU 训练仍需自行承担更长耗时",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_arg_parser().parse_args()
    report = run_formal_readiness(
        args.task_spec,
        args.config,
        minimum_memory_gb=args.minimum_memory_gb,
        processed_root_override=args.processed_root,
        split_file_override=args.split_file,
        allow_cpu=args.allow_cpu,
    )
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report.ready else 2)


if __name__ == "__main__":
    main()
