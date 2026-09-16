"""将完整 CTSpine1K raw NIfTI 数据断点续传到 H:\\CTSpine1K。

只下载训练真正需要的 rawdata（volume + label）和 metadata，避免 Arrow 格式的巨大额外占用。
重复运行会复用已完成文件并继续未完成下载。
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "alexanderdann/CTSpine1K"
DESTINATION = Path("H:/CTSpine1K")
STATUS_PATH = DESTINATION / "_download_status.json"
MIN_INITIAL_FREE_GB = 175.0
ALLOW_PATTERNS = [
    "raw_data/**",
    "metadata/**",
    "README.md",
    "CTSpine1K.py",
]


def write_status(**payload: object) -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    status = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "repo_id": REPO_ID,
        "destination": str(DESTINATION),
        **payload,
    }
    temp = STATUS_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATUS_PATH)


def free_gb() -> float:
    return shutil.disk_usage(DESTINATION.parent).free / (1024**3)


def main() -> int:
    if not DESTINATION.parent.exists():
        print("[ERROR] H: 盘不存在或当前不可访问。")
        return 2

    already_started = DESTINATION.exists() and any(DESTINATION.iterdir())
    available = free_gb()
    if not already_started and available < MIN_INITIAL_FREE_GB:
        print(
            f"[ERROR] H: 当前仅剩 {available:.1f} GB，完整 raw CTSpine1K 需要约 162 GB，"
            "为避免下载到一半空间不足，本下载器要求首次启动至少 175 GB 空闲。"
        )
        return 3

    print("=" * 76)
    print("CTSpine1K 完整数据下载（Raw NIfTI，可断点续传）")
    print(f"目标目录 : {DESTINATION}")
    print(f"当前 H: 空闲 : {available:.1f} GB")
    print("官方规模 : 1005 CT；raw 仓库约 162 GB")
    print("下载内容 : raw_data/volumes + raw_data/labels + metadata")
    print("并发数   : 2（优先稳定性）")
    print("说明     : 中断后重新运行本脚本会续传，不需要从头下载。")
    print("=" * 76, flush=True)

    write_status(
        state="downloading",
        started_or_resumed_at=datetime.now().isoformat(timespec="seconds"),
        free_gb_before=available,
        note="Raw NIfTI download in progress; rerun to resume if interrupted.",
    )

    started = time.perf_counter()
    try:
        result = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=DESTINATION,
            allow_patterns=ALLOW_PATTERNS,
            max_workers=2,
        )
    except KeyboardInterrupt:
        elapsed = time.perf_counter() - started
        write_status(
            state="interrupted",
            elapsed_seconds_this_run=elapsed,
            free_gb_after=free_gb(),
            note="Download interrupted safely. Run again to resume.",
        )
        print("\n[SAFE STOP] 下载已中断；已完成文件保留，重新运行即可续传。")
        return 130
    except Exception as exc:
        elapsed = time.perf_counter() - started
        write_status(
            state="error",
            elapsed_seconds_this_run=elapsed,
            free_gb_after=free_gb(),
            error=f"{type(exc).__name__}: {exc}",
            note="Run again after network recovery; snapshot_download resumes existing files.",
        )
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        print("网络恢复后重新运行即可续传。")
        return 1

    elapsed = time.perf_counter() - started
    remaining = free_gb()
    write_status(
        state="completed",
        completed_at=datetime.now().isoformat(timespec="seconds"),
        elapsed_seconds_this_run=elapsed,
        free_gb_after=remaining,
        snapshot_path=str(result),
        note="Full raw CTSpine1K snapshot completed. Preprocessing/split validation is still required before training.",
    )
    print("\n" + "=" * 76)
    print("[COMPLETED] CTSpine1K raw 数据下载完成。")
    print(f"目录: {result}")
    print(f"H: 剩余空间: {remaining:.1f} GB")
    print("下一步：建立完整 split、训练专用预处理/缓存，再启动 1000 例量级训练。")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
