from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _verify_gzip(path: Path, chunk_size: int = 8 * 1024 * 1024) -> int:
    """完整读取 gzip 数据并返回解压后的字节数；读取结束时会强制校验 CRC。"""
    total = 0
    with gzip.open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
    return total


def audit_cache(
    root: Path,
    *,
    splits: tuple[str, ...],
    status_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    split_path = root / "split.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    case_ids: list[tuple[str, str]] = []
    for split in splits:
        values = payload.get(split)
        if not isinstance(values, list):
            raise RuntimeError(f"split={split!r} 不存在或不是 list")
        case_ids.extend((split, str(case_id)) for case_id in values)

    required_names = ("image_normalized_u16.nii.gz", "label.nii.gz")
    started = time.perf_counter()
    failures: list[dict[str, str]] = []
    verified_files = 0
    total_files = len(case_ids) * len(required_names)

    def update(current_case: str | None, state: str) -> None:
        elapsed = time.perf_counter() - started
        checked_cases = verified_files // len(required_names)
        rate = verified_files / elapsed if elapsed > 0 else 0.0
        remaining_files = max(0, total_files - verified_files)
        eta = None if rate <= 0 else remaining_files / rate
        _write_json_atomic(
            status_path,
            {
                "updated_at": datetime.now().isoformat(),
                "state": state,
                "root": str(root),
                "splits": list(splits),
                "case_count": len(case_ids),
                "total_file_count": total_files,
                "verified_file_count": verified_files,
                "estimated_checked_case_count": checked_cases,
                "failure_count": len(failures),
                "current_case": current_case,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "pid": os.getpid(),
            },
        )

    update(None, "running")
    for split, case_id in case_ids:
        case_dir = root / case_id
        for name in required_names:
            path = case_dir / name
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                _verify_gzip(path)
            except Exception as exc:
                failures.append(
                    {
                        "split": split,
                        "case_id": case_id,
                        "file": str(path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            finally:
                verified_files += 1
        if verified_files % 20 == 0 or failures:
            update(case_id, "running")

    elapsed = time.perf_counter() - started
    result = {
        "finished_at": datetime.now().isoformat(),
        "state": "completed" if not failures else "completed_with_failures",
        "root": str(root),
        "splits": list(splits),
        "case_count": len(case_ids),
        "total_file_count": total_files,
        "verified_file_count": verified_files,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "note": "仅做 gzip CRC/流完整性检查；未执行模型推理，未读取 test split。",
    }
    _write_json_atomic(report_path, result)
    update(None, result["state"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CTSpine1K compact-cache gzip CRC integrity")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result = audit_cache(
        Path(args.root),
        splits=tuple(str(v) for v in args.splits),
        status_path=Path(args.status),
        report_path=Path(args.report),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
