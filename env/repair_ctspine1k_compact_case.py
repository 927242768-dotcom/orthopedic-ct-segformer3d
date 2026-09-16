from __future__ import annotations

import argparse
import gzip
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np

from src.preprocessing.prepare_ctspine1k import CTSpine1KCase
from src.preprocessing.prepare_ctspine1k_compact import _process_case


def verify_gzip_nifti(path: Path) -> dict[str, object]:
    with gzip.open(path, "rb") as stream:
        while stream.read(4 * 1024 * 1024):
            pass
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair one CTSpine1K compact cache case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(r"H:\CTSpine1K_compact_1p5mm"),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("experiments/repair_ctspine1k_compact_case.json"),
    )
    args = parser.parse_args()

    result: dict[str, object] = {
        "started_at": datetime.now().isoformat(),
        "case_id": args.case_id,
        "cache_root": str(args.cache_root),
        "status": "running",
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        manifest_path = args.cache_root / "ctspine1k_manifest.json"
        items = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = next(entry for entry in items if entry["case_id"] == args.case_id)
        case = CTSpine1KCase(
            case_id=item["case_id"],
            source_name=item["source_name"],
            sub_dataset=item["sub_dataset"],
            source_split=item["source_split"],
            image_path=Path(item["image_path"]),
            label_path=Path(item["label_path"]),
        )

        _process_case(
            case,
            args.cache_root,
            spacing_xyz=(1.5, 1.5, 1.5),
            hu_clip=(-1000.0, 2000.0),
        )

        case_dir = args.cache_root / args.case_id
        image_info = verify_gzip_nifti(case_dir / "image_normalized_u16.nii.gz")
        label_info = verify_gzip_nifti(case_dir / "label.nii.gz")
        result.update(
            {
                "status": "repaired",
                "finished_at": datetime.now().isoformat(),
                "image": image_info,
                "label": label_info,
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "finished_at": datetime.now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        args.result_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise

    args.result_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
