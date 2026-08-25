"""CTSpine1K 公开数据 → 本项目标准病例目录的批处理入口。

支持两种常见本地布局：

1. Hugging Face 原始仓库布局::

    raw_data/volumes/<sub-dataset>/*.nii.gz
    raw_data/labels/<sub-dataset>/*_seg.nii.gz

2. ``download_ctspine1k_sample.ps1`` 的小样本布局::

    <sub-dataset>/volumes/*.nii.gz
    <sub-dataset>/labels/*_seg.nii.gz

该模块只做可审计的数据发现、image/label 配对、官方 split 标记与标准化处理。
不会把 CTSpine1K 的 ``test_public`` 自动解释为 validation，也不会把 ``test_private``
自动解释为最终 test；论文实验 split 仍需在研究方案中显式固定，避免误用 benchmark 划分。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.preprocessing.nifti_pipeline import process_nifti_case
from src.preprocessing.qc_visualization import generate_case_qc

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RECOGNIZED_SPLITS = {
    "trainset": "trainset",
    "test_public": "test_public",
    "test_private": "test_private",
}


@dataclass(frozen=True)
class CTSpine1KCase:
    case_id: str
    source_name: str
    sub_dataset: str
    source_split: str
    image_path: Path
    label_path: Path


def _nifti_stem(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    if lower.endswith(".nii"):
        return name[:-4]
    raise ValueError(f"不是 NIfTI 文件: {path}")


def _iter_nifti(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.nii.gz")
    yield from root.rglob("*.nii")


def parse_official_split(path: str | Path) -> dict[str, str]:
    """解析 CTSpine1K 官方 ``data_split.txt`` 的三段 split 标记。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    mapping: dict[str, str] = {}
    current: str | None = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            section = line[:-1].strip().lower()
            current = _RECOGNIZED_SPLITS.get(section)
            continue
        if current is None:
            continue

        key = Path(line).name.lower()
        previous = mapping.get(key)
        if previous is not None and previous != current:
            raise RuntimeError(f"官方 split 文件中同一病例跨 split: {line}: {previous} vs {current}")
        mapping[key] = current

    if not mapping:
        raise RuntimeError(f"未在 split 文件中解析到 trainset/test_public/test_private: {path}")
    return mapping


def _find_volume_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    direct = root / "raw_data" / "volumes"
    if direct.is_dir():
        candidates.append(direct)

    for path in root.rglob("volumes"):
        if path.is_dir() and path not in candidates:
            candidates.append(path)
    return sorted(candidates)


def _label_dir_for_volume_dir(volume_dir: Path) -> Path:
    if volume_dir.name != "volumes":
        raise ValueError(f"非法 volumes 目录: {volume_dir}")
    sibling = volume_dir.parent / "labels"
    if sibling.is_dir():
        return sibling

    if volume_dir.parent.name == "raw_data":
        sibling = volume_dir.parent / "labels"
        if sibling.is_dir():
            return sibling
    raise FileNotFoundError(f"找不到与 {volume_dir} 对应的 labels 目录")


def _sub_dataset_for_file(file_path: Path, data_root: Path) -> str:
    """兼容 HF 嵌套布局与按 sub-dataset 下载的小样本布局。"""
    relative_parent = file_path.parent.relative_to(data_root)
    if relative_parent.parts:
        return relative_parent.parts[0]
    # 小样本布局：<sub-dataset>/volumes/*.nii.gz 或 labels/*.nii.gz
    return data_root.parent.name


def _safe_case_id(sub_dataset: str, source_name: str) -> str:
    raw = f"ctspine1k-{sub_dataset}-{source_name}".lower()
    safe = _SAFE_RE.sub("_", raw).strip("_")
    if not safe:
        raise ValueError(f"无法生成 case_id: {sub_dataset}/{source_name}")
    return safe


def discover_ctspine1k_cases(
    root: str | Path,
    *,
    split_file: str | Path | None = None,
) -> list[CTSpine1KCase]:
    """发现并严格配对 CTSpine1K NIfTI image/label。"""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)

    split_mapping = parse_official_split(split_file) if split_file is not None else {}
    cases: list[CTSpine1KCase] = []
    seen_case_ids: set[str] = set()

    volume_roots = _find_volume_dirs(root)
    if not volume_roots:
        raise RuntimeError(f"没有发现 CTSpine1K volumes 目录: {root}")

    for volume_root in volume_roots:
        label_root = _label_dir_for_volume_dir(volume_root)
        volume_files = [path for path in _iter_nifti(volume_root) if path.is_file()]
        label_files = [path for path in _iter_nifti(label_root) if path.is_file()]

        label_index: dict[tuple[str, str], list[Path]] = {}
        for label in label_files:
            label_stem = _nifti_stem(label)
            if not label_stem.lower().endswith("_seg"):
                continue
            source_name = label_stem[:-4]
            sub_dataset = _sub_dataset_for_file(label, label_root)
            label_index.setdefault((sub_dataset.lower(), source_name.lower()), []).append(label)

        for image in sorted(volume_files):
            source_name = _nifti_stem(image)
            sub_dataset = _sub_dataset_for_file(image, volume_root)
            matches = label_index.get((sub_dataset.lower(), source_name.lower()), [])
            if len(matches) != 1:
                raise RuntimeError(
                    f"{image}: 期望唯一匹配 *_seg label，但找到 {len(matches)} 个: {matches[:3]}"
                )

            case_id = _safe_case_id(sub_dataset, source_name)
            if case_id in seen_case_ids:
                raise RuntimeError(f"发现重复 case_id={case_id}，请检查重复下载或目录嵌套")
            seen_case_ids.add(case_id)
            cases.append(
                CTSpine1KCase(
                    case_id=case_id,
                    source_name=image.name,
                    sub_dataset=sub_dataset,
                    source_split=split_mapping.get(image.name.lower(), "unknown"),
                    image_path=image,
                    label_path=matches[0],
                )
            )

    if not cases:
        raise RuntimeError("没有发现可配对的 CTSpine1K image/label 病例")
    return sorted(cases, key=lambda item: item.case_id)


def prepare_ctspine1k_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split_file: str | Path | None = None,
    limit: int | None = None,
    target_spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_clip: tuple[float, float] = (-1000.0, 2000.0),
    bone_window: tuple[float, float] = (500.0, 2000.0),
    overwrite: bool = False,
    dry_run: bool = False,
    generate_qc: bool = False,
) -> dict[str, object]:
    """批量标准化 CTSpine1K，并可选生成逐例 QC contact sheet。"""
    cases = discover_ctspine1k_cases(source_root, split_file=split_file)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须 > 0")
        cases = cases[:limit]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    processed = 0
    skipped = 0
    qc_generated = 0

    for case in cases:
        case_dir = output_root / case.case_id
        entry = {
            **asdict(case),
            "image_path": str(case.image_path),
            "label_path": str(case.label_path),
        }
        manifest.append(entry)

        if dry_run:
            continue

        required = case_dir / "image_normalized.nii.gz"
        label_output = case_dir / "label.nii.gz"
        if case_dir.exists() and not overwrite:
            if required.exists() and label_output.exists():
                skipped += 1
                if generate_qc and not (case_dir / "qc_contact_sheet.png").exists():
                    generate_case_qc(case_dir)
                    qc_generated += 1
                continue
            raise RuntimeError(
                f"输出目录已存在但不完整: {case_dir}；请人工检查或显式使用 --overwrite"
            )

        try:
            process_nifti_case(
                case.image_path,
                case_dir,
                label_path=case.label_path,
                target_spacing_xyz=target_spacing_xyz,
                hu_clip=hu_clip,
                bone_window=bone_window,
            )
            processed += 1
            if generate_qc:
                generate_case_qc(case_dir)
                qc_generated += 1
        except Exception as exc:
            failures.append(
                {
                    "case_id": case.case_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    split_counts: dict[str, int] = {}
    for case in cases:
        split_counts[case.source_split] = split_counts.get(case.source_split, 0) + 1

    summary = {
        "source_root": str(Path(source_root)),
        "output_root": str(output_root),
        "split_file": None if split_file is None else str(Path(split_file)),
        "dry_run": dry_run,
        "discovered_case_count": len(cases),
        "processed_count": processed,
        "skipped_count": skipped,
        "qc_generated_count": qc_generated,
        "failure_count": len(failures),
        "failures": failures,
        "source_split_counts": split_counts,
        "target_spacing_xyz": list(target_spacing_xyz),
        "hu_clip": list(hu_clip),
        "bone_window": list(bone_window),
        "split_policy_note": (
            "仅保留 CTSpine1K 官方 trainset/test_public/test_private 标记；"
            "本工具不自动把 public/private test 重解释为 validation/test。"
        ),
    }

    (output_root / "ctspine1k_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "batch_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量标准化 CTSpine1K 公开脊柱 CT 数据")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=None, help="官方 data_split.txt，可选")
    parser.add_argument("--limit", type=int, default=None, help="首轮真实 QC 建议先处理 1—10 例")
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=2000.0)
    parser.add_argument("--bone-center", type=float, default=500.0)
    parser.add_argument("--bone-width", type=float, default=2000.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qc", action="store_true", help="标准化后同时生成三视图/骨窗/label QC 图")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    summary = prepare_ctspine1k_dataset(
        args.source_root,
        args.output_root,
        split_file=args.split_file,
        limit=args.limit,
        target_spacing_xyz=tuple(args.spacing),
        hu_clip=(args.hu_min, args.hu_max),
        bone_window=(args.bone_center, args.bone_width),
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        generate_qc=args.qc,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
