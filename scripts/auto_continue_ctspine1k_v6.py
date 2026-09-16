"""CTSpine1K v6 自动续跑链。

只在 v6 cache 完整达到 train610 + validation197、test=0 且 failure=0 后：
1. 等待高占用 Qwen 评测结束，避免 CPU/RAM 争抢；
2. 用 SegFormer3D-HR 稳定化 best 权重初始化全规模 train610/validation197；
3. 训练结束后自动用 best.pt 跑 validation197 full-volume sliding-window evaluation。

该脚本绝不读取/生成 test_private，也不会执行 test split 评估。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CACHE_ROOT = Path(r"H:\CTSpine1K_compact_1p5mm_nibsafe_v6")
CACHE_STATUS = CACHE_ROOT / "preprocess_status.json"
CACHE_SPLIT = CACHE_ROOT / "split.json"
CONFIG = PROJECT_ROOT / "configs" / "orthopedic_ct_full_large_scale_v6_segformer3d_hr_formal610_val197.yaml"
INIT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "20260915_151039_ctspine1k_v6_pilot32_val12_segformer3d_hr_stabilize_ft"
    / "checkpoint"
    / "best.pt"
)
CHAIN_DIR = PROJECT_ROOT / "experiments" / "auto_continue_ctspine1k_v6"
CHAIN_STATUS = CHAIN_DIR / "status.json"
CHAIN_LOG = CHAIN_DIR / "chain.log"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _log(message: str) -> None:
    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    with CHAIN_LOG.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def _status(phase: str, **extra: Any) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(),
        "phase": phase,
        **extra,
    }
    _atomic_json(CHAIN_STATUS, payload)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root 必须为 object: {path}")
    return payload


def _cache_snapshot() -> dict[str, Any]:
    if not CACHE_STATUS.exists():
        return {"ready": False, "reason": "preprocess_status_missing"}
    status = _read_json(CACHE_STATUS)
    completed = int(status.get("completed_case_count", 0))
    failures = int(status.get("failure_count", 0))
    state = str(status.get("state", "unknown"))

    split_counts = {"train": -1, "validation": -1, "test": -1}
    if CACHE_SPLIT.exists():
        split = _read_json(CACHE_SPLIT)
        split_counts = {
            key: len(split.get(key, [])) if isinstance(split.get(key, []), list) else -1
            for key in ("train", "validation", "test")
        }

    ready = (
        completed == 807
        and failures == 0
        and split_counts == {"train": 610, "validation": 197, "test": 0}
        and state in {"completed", "completed_with_failures"}
    )
    return {
        "ready": ready,
        "state": state,
        "completed": completed,
        "failures": failures,
        "split_counts": split_counts,
        "eta_seconds": status.get("eta_seconds"),
        "current_case": status.get("current_case"),
        "free_disk_gb": status.get("free_gb"),
    }


def _qwen_eval_processes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "evaluate_fixed60_unified.py" not in cmdline:
                continue
            memory = process.info.get("memory_info")
            rows.append(
                {
                    "pid": int(process.info["pid"]),
                    "private_or_rss_gb": (
                        None if memory is None else round(float(memory.rss) / 1024**3, 3)
                    ),
                    "cmdline": cmdline,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rows


def _free_ram_gb() -> float:
    return float(psutil.virtual_memory().available) / 1024**3


def _run_logged(command: list[str], log_name: str) -> int:
    log_path = CHAIN_DIR / log_name
    _log("执行: " + " ".join(command))
    with log_path.open("a", encoding="utf-8", errors="replace") as stream:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return int(process.wait())


def _find_fullscale_run(started_after: float) -> Path | None:
    candidates: list[Path] = []
    for path in (PROJECT_ROOT / "experiments").glob(
        "*_ctspine1k_v6_segformer3d_hr_full610_val197"
    ):
        if not path.is_dir():
            continue
        summary = path / "summary.json"
        if summary.exists() and summary.stat().st_mtime >= started_after:
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime_ns)


def _validate_static_inputs() -> None:
    for path in (PYTHON, CONFIG, INIT_CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)


def run_chain(*, poll_seconds: int, check_only: bool) -> int:
    _validate_static_inputs()
    snapshot = _cache_snapshot()
    if check_only:
        print(
            json.dumps(
                {
                    "cache": snapshot,
                    "qwen_eval": _qwen_eval_processes(),
                    "free_ram_gb": round(_free_ram_gb(), 3),
                    "config": str(CONFIG),
                    "init_checkpoint": str(INIT_CHECKPOINT),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    _log("自动续跑链启动；test_private/test split 明确禁用。")
    while True:
        snapshot = _cache_snapshot()
        if int(snapshot.get("failures", 0)) > 0:
            _status("blocked_cache_failure", cache=snapshot)
            _log(f"缓存出现 failure，停止自动链: {snapshot}")
            return 2
        if snapshot.get("ready"):
            break
        _status("waiting_cache", cache=snapshot)
        time.sleep(poll_seconds)

    _log("缓存已达到 807/807 且 train610/validation197/test0。")

    # 等待 Qwen 评测结束，避免 CPU/RAM 争抢把两个长任务都拖慢。
    while True:
        qwen = _qwen_eval_processes()
        free_ram = _free_ram_gb()
        if not qwen and free_ram >= 3.0:
            break
        _status(
            "waiting_resources",
            cache=_cache_snapshot(),
            qwen_eval=qwen,
            free_ram_gb=free_ram,
        )
        time.sleep(poll_seconds)

    train_started = time.time()
    _status("training_starting", free_ram_gb=_free_ram_gb())
    train_command = [
        str(PYTHON),
        "-m",
        "src.modeling.train",
        "--config",
        str(CONFIG),
        "--init-checkpoint",
        str(INIT_CHECKPOINT),
        "--preflight-mode",
        "engineering",
        "--allow-cpu",
    ]
    train_code = _run_logged(train_command, "fullscale_train.log")
    if train_code != 0:
        _status("training_failed", return_code=train_code)
        _log(f"全规模训练退出码={train_code}，自动链停止。")
        return train_code

    run_dir = _find_fullscale_run(train_started)
    if run_dir is None:
        _status("training_run_not_found")
        _log("训练进程成功结束，但未定位到新 full610 run。")
        return 3
    best_checkpoint = run_dir / "checkpoint" / "best.pt"
    if not best_checkpoint.exists():
        _status("best_checkpoint_missing", run_dir=str(run_dir))
        _log(f"训练 run 缺少 best.pt: {run_dir}")
        return 4

    evaluation_dir = run_dir / "full_volume_validation197"
    _status(
        "full_volume_validation_starting",
        run_dir=str(run_dir),
        best_checkpoint=str(best_checkpoint),
        evaluation_dir=str(evaluation_dir),
    )
    eval_command = [
        str(PYTHON),
        "-m",
        "src.modeling.evaluate",
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(best_checkpoint),
        "--split",
        "validation",
        "--output-dir",
        str(evaluation_dir),
        "--resume",
        "--preflight-mode",
        "engineering",
    ]
    eval_code = _run_logged(eval_command, "full_volume_validation197.log")
    if eval_code != 0:
        _status(
            "full_volume_validation_failed",
            return_code=eval_code,
            run_dir=str(run_dir),
            evaluation_dir=str(evaluation_dir),
        )
        _log(f"validation197 full-volume 退出码={eval_code}。")
        return eval_code

    _status(
        "completed",
        run_dir=str(run_dir),
        best_checkpoint=str(best_checkpoint),
        evaluation_dir=str(evaluation_dir),
        note=(
            "完成 train610/validation197 全规模开发训练与 validation197 full-volume evaluation；"
            "在人工 QC/formal gate 完成前不标记为最终论文结果。"
        ),
    )
    _log("自动续跑链完成。")
    return 0


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Auto-continue CTSpine1K v6 full-scale pipeline")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error("--poll-seconds 必须 >= 10")
    raise SystemExit(run_chain(poll_seconds=args.poll_seconds, check_only=args.check_only))


if __name__ == "__main__":
    main()
