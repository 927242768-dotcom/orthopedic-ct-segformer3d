"""正式训练/评估前的数据与配置预检。

核心目标不是替代训练，而是在消耗 GPU 资源或产生论文结果之前，提前阻止：
- train/validation/test 病例泄漏；
- CTSpine1K ``test_private`` 被误放入 train/validation；
- engineering-smoke split 被误当正式实验；
- input channel / model.in_channels / num_classes 不一致；
- NIfTI shape/spacing 与配置不一致；
- multiclass 标签值超出模型类别范围；
- 尚未完成项目要求的人工 QC 就启动 formal run。

``mode=engineering`` 只检查工程可运行性，不把缺 GPU/人工签字当成 blocker。
``mode=formal`` 按当前项目科研规范执行更严格检查。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PreflightIssue:
    severity: str
    code: str
    message: str
    case_id: str | None = None


@dataclass(frozen=True)
class PreflightReport:
    mode: str
    ready: bool
    config: str
    processed_root: str | None
    split_file: str | None
    checked_case_count: int
    split_counts: dict[str, int]
    pipeline_versions: dict[str, int]
    label_values_union: list[int]
    cuda_available: bool
    cuda_device: str | None
    issues: list[PreflightIssue]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issue_counts"] = {
            "error": sum(issue.severity == "error" for issue in self.issues),
            "warning": sum(issue.severity == "warning" for issue in self.issues),
        }
        return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _issue(
    issues: list[PreflightIssue],
    severity: str,
    code: str,
    message: str,
    case_id: str | None = None,
) -> None:
    issues.append(PreflightIssue(severity=severity, code=code, message=message, case_id=case_id))


def _channel_path(case_dir: Path, channel: str) -> Path:
    if channel == "ct_normalized":
        return case_dir / "image_normalized.nii.gz"
    if channel == "bone_window":
        return case_dir / "image_bone_window.nii.gz"
    return case_dir / f"{channel}.nii.gz"


def _safe_metadata(case_dir: Path) -> dict[str, Any] | None:
    path = case_dir / "metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _label_values(case_dir: Path, metadata: dict[str, Any] | None) -> list[int]:
    if metadata is not None:
        label = metadata.get("label")
        if isinstance(label, dict):
            values = label.get("label_values_after")
            if isinstance(values, list) and values:
                return sorted({int(value) for value in values})

    label_path = case_dir / "label.nii.gz"
    image = nib.load(str(label_path))
    values = np.asarray(image.dataobj)
    return sorted({int(value) for value in np.unique(np.rint(values))})


def _source_split_map(processed_root: Path) -> dict[str, str]:
    manifest_path = processed_root / "ctspine1k_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    result: dict[str, str] = {}
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            case_id = item.get("case_id")
            source_split = item.get("source_split")
            if case_id and source_split:
                result[str(case_id)] = str(source_split)
    return result


def _human_qc_map(processed_root: Path) -> dict[str, dict[str, str]]:
    path = processed_root / "manual_qc_review.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {
            str(row.get("case_id", "")): {str(key): str(value or "") for key, value in row.items()}
            for row in rows
            if row.get("case_id")
        }


def _accepted_review(row: dict[str, str]) -> bool:
    status = row.get("review_status", "").strip().lower()
    accepted = {"pass", "passed", "approved", "ok", "通过", "合格"}
    required_flags = ("orientation_ok", "spacing_ok", "label_alignment_ok", "bone_window_ok")
    truthy = {"1", "true", "yes", "y", "ok", "pass", "通过", "是"}
    return status in accepted and all(row.get(field, "").strip().lower() in truthy for field in required_flags)


def _iter_unique_cases(split_payload: dict[str, Any]) -> Iterable[tuple[str, str]]:
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        for value in split_payload.get(split, []) or []:
            case_id = str(value)
            if case_id in seen:
                continue
            seen.add(case_id)
            yield split, case_id


def run_preflight(
    config_path: str | Path,
    *,
    mode: str = "formal",
    require_gpu: bool | None = None,
    require_human_qc: bool | None = None,
    processed_root_override: str | Path | None = None,
    split_file_override: str | Path | None = None,
) -> PreflightReport:
    if mode not in {"formal", "engineering"}:
        raise ValueError("mode 只能为 formal 或 engineering")
    if require_gpu is None:
        require_gpu = mode == "formal"
    if require_human_qc is None:
        require_human_qc = mode == "formal"

    issues: list[PreflightIssue] = []
    config_path = _resolve(config_path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config YAML 必须为 mapping")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    processed_root = _resolve(
        processed_root_override
        if processed_root_override is not None
        else data_cfg.get("processed_root", "data/processed")
    )
    split_file = _resolve(
        split_file_override
        if split_file_override is not None
        else data_cfg.get("split_file", "data/splits/TBD.json")
    )

    if not processed_root.is_dir():
        _issue(issues, "error", "processed_root_missing", f"processed_root 不存在: {processed_root}")
    if not split_file.is_file():
        _issue(issues, "error", "split_file_missing", f"split_file 不存在: {split_file}")

    split_payload: dict[str, Any] = {}
    if split_file.is_file():
        try:
            loaded = json.loads(split_file.read_text(encoding="utf-8-sig"))
            if not isinstance(loaded, dict):
                raise TypeError("split JSON 不是 object")
            split_payload = loaded
        except Exception as exc:
            _issue(issues, "error", "split_parse_error", f"split JSON 解析失败: {type(exc).__name__}: {exc}")

    split_counts: dict[str, int] = {}
    memberships: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        values = split_payload.get(split, []) if split_payload else []
        if not isinstance(values, list):
            _issue(issues, "error", "split_not_list", f"split {split!r} 必须是 list")
            values = []
        case_ids = [str(value) for value in values]
        split_counts[split] = len(case_ids)
        if split in {"train", "validation"} and not case_ids:
            _issue(issues, "error", "required_split_empty", f"split {split!r} 为空")
        if len(case_ids) != len(set(case_ids)):
            _issue(issues, "error", "duplicate_within_split", f"split {split!r} 内存在重复 case_id")
        for case_id in case_ids:
            memberships.setdefault(case_id, []).append(split)

    for case_id, member_splits in memberships.items():
        if len(member_splits) > 1:
            _issue(
                issues,
                "error",
                "case_split_leakage",
                f"同一 case 同时出现在 {member_splits}",
                case_id,
            )

    meta = split_payload.get("_meta", {}) if split_payload else {}
    if mode == "formal" and isinstance(meta, dict) and meta.get("formal_experiment") is False:
        _issue(
            issues,
            "error",
            "engineering_split_for_formal_run",
            "split 明确标记 formal_experiment=false，禁止用于正式论文训练/评估",
        )

    input_channels = [str(value) for value in data_cfg.get("input_channels", ["ct_normalized"])]
    model_in_channels = int(model_cfg.get("in_channels", len(input_channels)))
    if model_in_channels != len(input_channels):
        _issue(
            issues,
            "error",
            "input_channel_mismatch",
            f"model.in_channels={model_in_channels}，但 data.input_channels={input_channels}",
        )

    model_num_classes = int(model_cfg.get("num_classes", data_cfg.get("num_classes", 2)))
    if "num_classes" in data_cfg and int(data_cfg["num_classes"]) != model_num_classes:
        _issue(
            issues,
            "error",
            "num_classes_mismatch",
            f"data.num_classes={data_cfg['num_classes']} 与 model.num_classes={model_num_classes} 不一致",
        )
    label_mode = str(data_cfg.get("label_mode", "binary"))
    if label_mode not in {"binary", "multiclass"}:
        _issue(issues, "error", "invalid_label_mode", f"label_mode={label_mode!r} 非法")

    target_spacing = tuple(float(v) for v in data_cfg.get("target_spacing_xyz_mm", [1.0, 1.0, 1.0]))
    if len(target_spacing) != 3 or any(value <= 0 for value in target_spacing):
        _issue(issues, "error", "invalid_target_spacing", f"target spacing 非法: {target_spacing}")

    source_splits = _source_split_map(processed_root) if processed_root.is_dir() else {}
    qc_rows = _human_qc_map(processed_root) if processed_root.is_dir() else {}
    if mode == "formal" and require_human_qc and processed_root.is_dir() and not qc_rows:
        _issue(issues, "error", "human_qc_missing", "formal run 要求 manual_qc_review.csv 人工审核记录")

    pipeline_versions: dict[str, int] = {}
    label_union: set[int] = set()
    checked_case_count = 0

    if processed_root.is_dir():
        for split, case_id in _iter_unique_cases(split_payload):
            case_dir = processed_root / case_id
            if not case_dir.is_dir():
                _issue(issues, "error", "case_dir_missing", f"病例目录不存在: {case_dir}", case_id)
                continue

            label_path = case_dir / "label.nii.gz"
            if not label_path.exists():
                _issue(issues, "error", "label_missing", "label.nii.gz 不存在", case_id)
                continue

            missing_channels = [
                channel for channel in input_channels if not _channel_path(case_dir, channel).exists()
            ]
            if missing_channels:
                _issue(
                    issues,
                    "error",
                    "input_channel_missing",
                    f"缺少输入通道: {missing_channels}",
                    case_id,
                )
                continue

            try:
                label_img = nib.load(str(label_path))
                label_shape = tuple(int(v) for v in label_img.shape[:3])
                label_spacing = tuple(float(v) for v in label_img.header.get_zooms()[:3])
            except Exception as exc:
                _issue(issues, "error", "label_read_error", f"label 读取失败: {exc}", case_id)
                continue

            for channel in input_channels:
                path = _channel_path(case_dir, channel)
                try:
                    image = nib.load(str(path))
                    shape = tuple(int(v) for v in image.shape[:3])
                    spacing = tuple(float(v) for v in image.header.get_zooms()[:3])
                except Exception as exc:
                    _issue(issues, "error", "channel_read_error", f"{channel} 读取失败: {exc}", case_id)
                    continue
                if shape != label_shape:
                    _issue(
                        issues,
                        "error",
                        "shape_mismatch",
                        f"{channel} shape={shape} 与 label={label_shape} 不一致",
                        case_id,
                    )
                if not np.allclose(spacing, label_spacing, rtol=0.0, atol=1e-5):
                    _issue(
                        issues,
                        "error",
                        "spacing_mismatch",
                        f"{channel} spacing={spacing} 与 label={label_spacing} 不一致",
                        case_id,
                    )

            if len(target_spacing) == 3 and not np.allclose(
                label_spacing, target_spacing, rtol=0.0, atol=1e-4
            ):
                _issue(
                    issues,
                    "error" if mode == "formal" else "warning",
                    "target_spacing_mismatch",
                    f"label spacing={label_spacing} 与 config target={target_spacing} 不一致",
                    case_id,
                )

            metadata = _safe_metadata(case_dir)
            if metadata is None:
                _issue(
                    issues,
                    "error" if mode == "formal" else "warning",
                    "metadata_missing_or_invalid",
                    "metadata.json 缺失或无法解析",
                    case_id,
                )
            else:
                version = str(metadata.get("pipeline_version", "unknown"))
                pipeline_versions[version] = pipeline_versions.get(version, 0) + 1

            try:
                values = _label_values(case_dir, metadata)
                label_union.update(values)
            except Exception as exc:
                _issue(issues, "error", "label_values_error", f"标签类别读取失败: {exc}", case_id)
                values = []

            if values and min(values) < 0:
                _issue(issues, "error", "negative_label", f"标签包含负值: {values[:10]}", case_id)
            if label_mode == "multiclass" and values and max(values) >= model_num_classes:
                _issue(
                    issues,
                    "error",
                    "multiclass_out_of_range",
                    f"max label={max(values)}，但 model.num_classes={model_num_classes}；需至少 {max(values)+1}",
                    case_id,
                )

            source_split = source_splits.get(case_id)
            if source_split == "test_private" and split in {"train", "validation"}:
                _issue(
                    issues,
                    "error",
                    "private_test_leakage",
                    f"官方 source_split=test_private 被放入 {split}",
                    case_id,
                )

            if mode == "formal" and require_human_qc:
                row = qc_rows.get(case_id)
                if row is None:
                    _issue(issues, "error", "human_qc_case_missing", "人工 QC CSV 无该病例", case_id)
                elif not _accepted_review(row):
                    _issue(
                        issues,
                        "error",
                        "human_qc_not_approved",
                        "人工 QC 尚未同时确认 orientation/spacing/alignment/bone-window 并标记通过",
                        case_id,
                    )

            checked_case_count += 1

    if mode == "formal" and len(pipeline_versions) > 1:
        _issue(
            issues,
            "error",
            "mixed_pipeline_versions",
            f"正式 split 混用了多个 preprocessing pipeline: {pipeline_versions}",
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    if require_gpu and not cuda_available:
        _issue(
            issues,
            "error",
            "gpu_unavailable",
            "当前 PyTorch 环境没有可用 CUDA GPU；正式 3D CT baseline 不应在本 CPU 环境启动",
        )

    ready = not any(issue.severity == "error" for issue in issues)
    return PreflightReport(
        mode=mode,
        ready=ready,
        config=str(config_path),
        processed_root=str(processed_root),
        split_file=str(split_file),
        checked_case_count=checked_case_count,
        split_counts=split_counts,
        pipeline_versions=pipeline_versions,
        label_values_union=sorted(label_union),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        issues=issues,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="骨科 CT SegFormer3D 正式训练/评估前预检")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("formal", "engineering"), default="formal")
    parser.add_argument("--processed-root", type=Path, default=None, help="可覆盖 config 中 processed_root，用于工程预检")
    parser.add_argument("--split-file", type=Path, default=None, help="可覆盖 config 中 split_file，用于工程预检")
    parser.add_argument("--allow-cpu", action="store_true", help="仅用于工程预检；formal 模式默认要求 CUDA GPU")
    parser.add_argument(
        "--skip-human-qc",
        action="store_true",
        help="跳过人工 QC blocker；只应用于工程排查，不应用于正式论文 run",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    report = run_preflight(
        args.config,
        mode=args.mode,
        require_gpu=False if args.allow_cpu else None,
        require_human_qc=False if args.skip_human_qc else None,
        processed_root_override=args.processed_root,
        split_file_override=args.split_file,
    )
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    if args.output is not None:
        output = _resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)
    if not report.ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
