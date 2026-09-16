from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.prepare_ctspine1k import discover_ctspine1k_cases
from src.preprocessing.prepare_ctspine1k_compact import (
    DEFAULT_HU_CLIP,
    DEFAULT_SOURCE_ROOT,
    DEFAULT_SPACING,
    _process_case,
)


def _verify_gzip(path: Path) -> None:
    """完整读取 gzip 流，强制触发 CRC 校验。"""
    with gzip.open(path, "rb") as stream:
        while stream.read(4 * 1024 * 1024):
            pass


def repair_case(
    case_id: str,
    *,
    source_root: Path,
    output_root: Path,
    spacing_xyz: tuple[float, float, float],
    hu_clip: tuple[float, float],
) -> dict:
    split_file = source_root / "metadata" / "data_split.txt"
    cases = discover_ctspine1k_cases(source_root, split_file=split_file)
    matches = [case for case in cases if case.case_id == case_id]
    if len(matches) != 1:
        raise RuntimeError(f"case_id={case_id!r} 匹配数量异常: {len(matches)}")

    case = matches[0]
    _process_case(
        case,
        output_root,
        spacing_xyz=spacing_xyz,
        hu_clip=hu_clip,
    )

    case_dir = output_root / case_id
    image_path = case_dir / "image_normalized_u16.nii.gz"
    label_path = case_dir / "label.nii.gz"
    metadata_path = case_dir / "metadata.json"

    _verify_gzip(image_path)
    _verify_gzip(label_path)

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_shape = tuple(int(v) for v in image.shape[:3])
    label_shape = tuple(int(v) for v in label.shape[:3])
    if image_shape != label_shape:
        raise RuntimeError(f"修复后 image/label shape 不一致: {image_shape} vs {label_shape}")

    image_data = np.asarray(image.dataobj, dtype=np.uint16)
    label_data = np.asarray(label.dataobj, dtype=np.uint8)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    payload = {
        "case_id": case_id,
        "source_split": case.source_split,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "shape_xyz": list(image_shape),
        "image_min": int(image_data.min()),
        "image_max": int(image_data.max()),
        "label_min": int(label_data.min()),
        "label_max": int(label_data.max()),
        "pipeline_version": metadata.get("pipeline_version"),
        "gzip_crc_verified": True,
    }
    return payload


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="原子重建一个 CTSpine1K 紧凑缓存病例并验证 gzip CRC")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=DEFAULT_SPACING)
    parser.add_argument("--hu-min", type=float, default=DEFAULT_HU_CLIP[0])
    parser.add_argument("--hu-max", type=float, default=DEFAULT_HU_CLIP[1])
    args = parser.parse_args()

    payload = repair_case(
        str(args.case_id),
        source_root=Path(args.source_root),
        output_root=Path(args.output_root),
        spacing_xyz=tuple(float(v) for v in args.spacing),
        hu_clip=(float(args.hu_min), float(args.hu_max)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
