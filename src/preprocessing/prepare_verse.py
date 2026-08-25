"""VerSe 完整公开数据 → 本项目标准病例目录的批处理入口。

该脚本不负责下载数据，只处理用户已经按 VerSe 官方结构解压的数据：
- 自动发现 CT 与 vertebra mask；
- 识别 training/validation/test 来源 split；
- 用 ``process_nifti_case`` 做物理空间校验与标准化；
- 输出 manifest、official split 和 batch QC summary；
- 同一 patient group 若跨 source split 会直接失败，防止泄漏。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.preprocessing.nifti_pipeline import process_nifti_case
from src.preprocessing.qc_visualization import generate_case_qc

_SUBJECT_RE = re.compile(r"(sub-verse\d+)", re.IGNORECASE)
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class VerseCase:
    case_id: str
    patient_group: str
    source_split: str
    image_path: Path
    label_path: Path


def _iter_nifti(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.nii.gz")
    yield from root.rglob("*.nii")


def _infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    validation_names = {"validation", "02_validation", "verse19validation", "verse20validation"}
    test_names = {"test", "03_test", "verse19test", "verse20test"}
    train_names = {"train", "training", "01_training", "verse19training", "verse20training"}
    if parts & validation_names:
        return "validation"
    if parts & test_names:
        return "test"
    if parts & train_names:
        return "train"
    return "unknown"


def _subject_id(path: Path) -> str:
    match = _SUBJECT_RE.search(path.as_posix())
    if match is None:
        raise ValueError(f"无法从路径提取 VerSe subject id: {path}")
    return match.group(1).lower()


def _case_id_from_image(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    if name.endswith("_ct"):
        name = name[:-3]
    safe = _SAFE_RE.sub("_", name).strip("_")
    if not safe:
        raise ValueError(f"无法生成 case_id: {path}")
    return safe.lower()


def discover_verse_cases(root: str | Path) -> list[VerseCase]:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)

    all_nifti = list(_iter_nifti(root))
    labels = [path for path in all_nifti if "_seg-vert_msk.nii" in path.name.lower()]
    images = [
        path
        for path in all_nifti
        if path.name.lower().endswith("_ct.nii.gz") or path.name.lower().endswith("_ct.nii")
    ]

    label_index: dict[tuple[str, str, str], list[Path]] = {}
    for label in labels:
        subject = _subject_id(label)
        split = _infer_split(label)
        expected_image_name = label.name.replace("_seg-vert_msk.nii.gz", "_ct.nii.gz").replace(
            "_seg-vert_msk.nii", "_ct.nii"
        )
        key = (subject, split, expected_image_name.lower())
        label_index.setdefault(key, []).append(label)

    cases: list[VerseCase] = []
    seen_case_ids: set[str] = set()
    for image in sorted(images):
        subject = _subject_id(image)
        split = _infer_split(image)
        key = (subject, split, image.name.lower())
        matches = label_index.get(key, [])
        if len(matches) != 1:
            raise RuntimeError(
                f"{image}: 期望唯一匹配 vertebra mask，但找到 {len(matches)} 个: {matches[:3]}"
            )

        case_id = _case_id_from_image(image)
        if case_id in seen_case_ids:
            raise RuntimeError(
                f"发现重复 case_id={case_id}。可能同时解压了含重复病例的数据包，需先去重。"
            )
        seen_case_ids.add(case_id)
        cases.append(
            VerseCase(
                case_id=case_id,
                patient_group=subject,
                source_split=split,
                image_path=image,
                label_path=matches[0],
            )
        )

    if not cases:
        raise RuntimeError("没有发现符合 VerSe 命名规则的 CT + vertebra mask 病例")

    group_split: dict[str, str] = {}
    for case in cases:
        previous = group_split.setdefault(case.patient_group, case.source_split)
        if previous != case.source_split:
            raise RuntimeError(
                f"patient group {case.patient_group} 跨 source split: {previous} vs {case.source_split}"
            )
    return cases


def build_official_split(cases: list[VerseCase]) -> dict[str, object]:
    split = {"train": [], "validation": [], "test": []}
    unknown = []
    for case in cases:
        if case.source_split in split:
            split[case.source_split].append(case.case_id)
        else:
            unknown.append(case.case_id)
    return {
        **split,
        "meta": {
            "source": "VerSe complete restructured dataset",
            "split_policy": "preserve source split; patient-group leakage checked",
            "case_count": len(cases),
            "patient_group_count": len({case.patient_group for case in cases}),
            "unknown_split_cases": unknown,
        },
    }


def prepare_verse_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
    target_spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_clip: tuple[float, float] = (-1000.0, 2000.0),
    bone_window: tuple[float, float] = (500.0, 2000.0),
    overwrite: bool = False,
    dry_run: bool = False,
    generate_qc: bool = False,
) -> dict[str, object]:
    cases = discover_verse_cases(source_root)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须 > 0")
        cases = cases[:limit]

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    failures = []
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
        if case_dir.exists() and not overwrite:
            required = case_dir / "image_normalized.nii.gz"
            label = case_dir / "label.nii.gz"
            if required.exists() and label.exists():
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

    split_payload = build_official_split(cases)
    summary = {
        "source_root": str(Path(source_root)),
        "output_root": str(output_root),
        "dry_run": dry_run,
        "discovered_case_count": len(cases),
        "processed_count": processed,
        "skipped_count": skipped,
        "qc_generated_count": qc_generated,
        "failure_count": len(failures),
        "failures": failures,
        "target_spacing_xyz": list(target_spacing_xyz),
        "hu_clip": list(hu_clip),
        "bone_window": list(bone_window),
    }

    (output_root / "verse_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "verse_official_split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "batch_qc_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="批量标准化 VerSe 公开脊柱 CT 数据")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="首轮 QC 可先处理 10 例")
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
    args = build_arg_parser().parse_args()
    summary = prepare_verse_dataset(
        args.source_root,
        args.output_root,
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
