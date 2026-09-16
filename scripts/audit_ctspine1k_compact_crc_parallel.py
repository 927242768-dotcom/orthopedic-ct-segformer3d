from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _verify_one(split: str, case_id: str, path: Path) -> dict[str, str] | None:
    try:
        if not path.is_file():
            raise FileNotFoundError(path)
        with gzip.open(path, "rb") as stream:
            while stream.read(8 * 1024 * 1024):
                pass
        return None
    except Exception as exc:
        return {
            "split": split,
            "case_id": case_id,
            "file": str(path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def audit_cache(
    root: Path,
    *,
    splits: tuple[str, ...],
    status_path: Path,
    report_path: Path,
    workers: int,
) -> dict[str, Any]:
    split_payload = json.loads((root / "split.json").read_text(encoding="utf-8"))
    tasks: list[tuple[str, str, Path]] = []
    for split in splits:
        values = split_payload.get(split)
        if not isinstance(values, list):
            raise RuntimeError(f"split={split!r} 不存在或不是 list")
        for raw_case_id in values:
            case_id = str(raw_case_id)
            case_dir = root / case_id
            tasks.append((split, case_id, case_dir / "image_normalized_u16.nii.gz"))
            tasks.append((split, case_id, case_dir / "label.nii.gz"))

    started = time.perf_counter()
    verified = 0
    failures: list[dict[str, str]] = []
    total = len(tasks)

    def update(current_case: str | None, state: str) -> None:
        elapsed = time.perf_counter() - started
        rate = verified / elapsed if elapsed > 0 else 0.0
        eta = None if rate <= 0 else max(0.0, (total - verified) / rate)
        _write_json_atomic(
            status_path,
            {
                "updated_at": datetime.now().isoformat(),
                "state": state,
                "root": str(root),
                "splits": list(splits),
                "workers": workers,
                "case_count": total // 2,
                "total_file_count": total,
                "verified_file_count": verified,
                "estimated_checked_case_count": verified // 2,
                "failure_count": len(failures),
                "current_case": current_case,
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
                "pid": os.getpid(),
            },
        )

    update(None, "running")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_verify_one, split, case_id, path): (split, case_id, path)
            for split, case_id, path in tasks
        }
        for future in as_completed(futures):
            _split, case_id, _path = futures[future]
            failure = future.result()
            if failure is not None:
                failures.append(failure)
            verified += 1
            if verified % 20 == 0 or failure is not None:
                update(case_id, "running")

    result = {
        "finished_at": datetime.now().isoformat(),
        "state": "completed" if not failures else "completed_with_failures",
        "root": str(root),
        "splits": list(splits),
        "workers": workers,
        "case_count": total // 2,
        "total_file_count": total,
        "verified_file_count": verified,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "仅做 gzip CRC/流完整性检查；未执行模型推理，未读取 test split。",
    }
    _write_json_atomic(report_path, result)
    update(None, result["state"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel gzip CRC audit for CTSpine1K compact cache")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    workers = max(1, min(int(args.workers), 8))
    result = audit_cache(
        Path(args.root),
        splits=tuple(str(v) for v in args.splits),
        status_path=Path(args.status),
        report_path=Path(args.report),
        workers=workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
