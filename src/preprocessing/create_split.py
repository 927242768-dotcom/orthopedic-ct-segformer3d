"""为标准化病例生成可复现的患者级 train/validation/test split。

默认把每个 `case_xxx` 视为一个患者。如果同一患者存在多次扫描，必须提供
`--group-map` JSON，把多个 case 映射到同一 patient_group，避免数据泄漏。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def discover_valid_cases(processed_root: Path) -> list[str]:
    cases = []
    for path in sorted(processed_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / "image_normalized.nii.gz").exists() and (path / "label.nii.gz").exists():
            cases.append(path.name)
    return cases


def load_group_map(path: Path | None, case_ids: list[str]) -> dict[str, str]:
    if path is None:
        return {case_id: case_id for case_id in case_ids}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {str(k): str(v) for k, v in payload.items()}
    missing = sorted(set(case_ids) - set(mapping))
    if missing:
        raise ValueError(f"group-map 缺少 {len(missing)} 个病例，例如: {missing[:5]}")
    return {case_id: mapping[case_id] for case_id in case_ids}


def split_groups(
    case_ids: list[str],
    group_map: dict[str, str],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("train_ratio/val_ratio 必须位于 (0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须 < 1")

    grouped: dict[str, list[str]] = defaultdict(list)
    for case_id in case_ids:
        grouped[group_map[case_id]].append(case_id)

    groups = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(groups)
    if n < 3:
        raise ValueError("至少需要 3 个患者组才能建立 train/validation/test")

    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    result = {"train": [], "validation": [], "test": []}
    for case_id in sorted(case_ids):
        group = group_map[case_id]
        if group in train_groups:
            result["train"].append(case_id)
        elif group in val_groups:
            result["validation"].append(case_id)
        elif group in test_groups:
            result["test"].append(case_id)
        else:  # pragma: no cover
            raise AssertionError(f"未分配 patient group: {group}")

    # 最终安全检查：同一个 group 不能跨 split。
    split_of_group: dict[str, str] = {}
    for split_name, ids in result.items():
        for case_id in ids:
            group = group_map[case_id]
            previous = split_of_group.setdefault(group, split_name)
            if previous != split_name:
                raise AssertionError(f"patient-level 泄漏: group={group}: {previous} vs {split_name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level CT split JSON")
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = discover_valid_cases(args.processed_root)
    if not cases:
        raise SystemExit("没有发现同时包含 image_normalized.nii.gz 与 label.nii.gz 的病例")
    groups = load_group_map(args.group_map, cases)
    result = split_groups(
        cases,
        groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output = {
        **result,
        "meta": {
            "created_at": datetime.now().isoformat(),
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
            "case_count": len(cases),
            "patient_group_count": len(set(groups.values())),
            "group_map_used": args.group_map is not None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
