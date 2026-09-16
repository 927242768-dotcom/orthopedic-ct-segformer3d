from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import nibabel as nib
import numpy as np

EXPECTED_PIPELINE = "0.6.0-compact-u16-fixed-hu-minmax-1p5mm-nibsafe"
EXPECTED_SPLITS = {"train": 610, "validation": 197, "test": 0}


def verify_gzip(path: Path) -> None:
    with gzip.open(path, "rb") as stream:
        while stream.read(8 * 1024 * 1024):
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    split = json.loads((root / "split.json").read_text(encoding="utf-8-sig"))
    issues: list[dict[str, str]] = []
    counts = {key: len(split.get(key, [])) for key in ("train", "validation", "test")}
    if counts != EXPECTED_SPLITS:
        issues.append({"case_id": "<split>", "error": f"split counts {counts} != {EXPECTED_SPLITS}"})
    train = [str(x) for x in split.get("train", [])]
    val = [str(x) for x in split.get("validation", [])]
    test = [str(x) for x in split.get("test", [])]
    if test:
        issues.append({"case_id": "<split>", "error": "test split must stay empty"})
    if set(train) & set(val):
        issues.append({"case_id": "<split>", "error": "train/validation overlap"})
    case_ids = train + val
    if len(case_ids) != len(set(case_ids)):
        issues.append({"case_id": "<split>", "error": "duplicate case_id across allowed splits"})

    checked = 0
    for case_id in case_ids:
        case_dir = root / case_id
        meta_path = case_dir / "metadata.json"
        image_path = case_dir / "image_normalized_u16.nii.gz"
        label_path = case_dir / "label.nii.gz"
        try:
            for required in (meta_path, image_path, label_path):
                if not required.is_file():
                    raise FileNotFoundError(required)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("pipeline_version") != EXPECTED_PIPELINE:
                raise RuntimeError(f"pipeline mismatch: {meta.get('pipeline_version')}")
            if meta.get("source_split") == "test_private":
                raise RuntimeError("test_private leakage")
            qc = meta.get("qc", {})
            if qc.get("status") != "pass":
                raise RuntimeError(f"qc status={qc.get('status')}")
            if qc.get("cache_crc_verified") is not True:
                raise RuntimeError("metadata cache_crc_verified != true")
            if qc.get("cache_image_label_affine_match") is not True:
                raise RuntimeError("metadata affine match != true")
            stats = meta.get("source", {}).get("image", {}).get("hu_stats", {})
            p01, p50, p99 = (float(stats[k]) for k in ("p01", "p50", "p99"))
            if abs(p50) > 10000 or max(abs(p01), abs(p99)) > 100000:
                raise RuntimeError(f"implausible source HU stats: p01={p01}, p50={p50}, p99={p99}")
            spacing = tuple(float(v) for v in meta.get("processed", {}).get("spacing_xyz_mm", []))
            if len(spacing) != 3 or not np.allclose(spacing, (1.5, 1.5, 1.5), rtol=0.0, atol=1e-4):
                raise RuntimeError(f"spacing={spacing}")
            verify_gzip(image_path)
            verify_gzip(label_path)
            image = nib.load(str(image_path))
            label = nib.load(str(label_path))
            if image.shape[:3] != label.shape[:3]:
                raise RuntimeError(f"shape mismatch image={image.shape} label={label.shape}")
            if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
                raise RuntimeError("fresh affine mismatch")
            checked += 1
            if checked % 50 == 0 or checked == len(case_ids):
                print(f"[AUDIT] checked={checked}/{len(case_ids)} issues={len(issues)}", flush=True)
        except Exception as exc:
            issues.append({"case_id": case_id, "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "ready": not issues and checked == 807,
        "checked_case_count": checked,
        "split_counts": counts,
        "issue_count": len(issues),
        "issues": issues,
        "test_private_used": False,
        "fresh_gzip_crc_verified": not issues and checked == 807,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
