from __future__ import annotations

import csv
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import torch

try:
    import psutil
except ImportError:  # 本地面板可降级；训练本身不依赖 psutil。
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_V1 = PROJECT_ROOT / "configs" / "orthopedic_ct_full_large_scale_v1.yaml"
CONFIG_V2 = PROJECT_ROOT / "configs" / "orthopedic_ct_full_large_scale_v2.yaml"
CONFIG_V3 = PROJECT_ROOT / "configs" / "orthopedic_ct_full_large_scale_v3_fixed_hu.yaml"
CONFIG_V4B = PROJECT_ROOT / "configs" / "orthopedic_ct_full_large_scale_v4b_residual_fp_pilot32.yaml"
V3_BEST = PROJECT_ROOT / "experiments" / "20260909_232528_full_ctspine1k_large_scale_v3_fixed_hu" / "checkpoint" / "best.pt"
V4B_GUIDANCE_ROOT = PROJECT_ROOT / "experiments" / "hard_mining_guidance_v4b_residual_fp_pilot32"
RAW_ROOT = Path("H:/CTSpine1K")
LEGACY_CACHE_ROOT = Path("H:/CTSpine1K_compact_1p5mm")
CACHE_ROOT = Path("H:/CTSpine1K_compact_1p5mm_fixed_hu")
EXPERIMENT_V1 = "full_ctspine1k_large_scale_v1"
EXPERIMENT_V2 = "full_ctspine1k_large_scale_v2"
EXPERIMENT_V3 = "full_ctspine1k_large_scale_v3_fixed_hu"
EXPERIMENT_V4B = "full_ctspine1k_v4b_residual_fp_pilot32"
SUPPORTED_EXPERIMENTS = (EXPERIMENT_V1, EXPERIMENT_V2, EXPERIMENT_V3, EXPERIMENT_V4B)
EXPECTED_RAW = 1005
EXPECTED_TRAIN = 610
EXPECTED_VAL = 197
EXPECTED_TEST = 198
QUICK_VAL_CASES = 12


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def fmt_num(value, digits: int = 6) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def fmt_duration(seconds) -> str:
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "--:--:--"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def latest_run_dir(experiment_name: str | None = None) -> Path | None:
    root = PROJECT_ROOT / "experiments"
    if not root.exists():
        return None
    names = (experiment_name,) if experiment_name is not None else SUPPORTED_EXPERIMENTS
    candidates: list[Path] = []
    for name in names:
        candidates.extend(p for p in root.glob(f"*_{name}") if p.is_dir())
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def run_experiment_name(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    for name in SUPPORTED_EXPERIMENTS:
        if run_dir.name.endswith(f"_{name}"):
            return name
    config = read_json(run_dir / "run_metadata.json")
    source = str(config.get("source_config") or "")
    if source.endswith("orthopedic_ct_full_large_scale_v4b_residual_fp_pilot32.yaml"):
        return EXPERIMENT_V4B
    if source.endswith("orthopedic_ct_full_large_scale_v3_fixed_hu.yaml"):
        return EXPERIMENT_V3
    if source.endswith("orthopedic_ct_full_large_scale_v2.yaml"):
        return EXPERIMENT_V2
    if source.endswith("orthopedic_ct_full_large_scale_v1.yaml"):
        return EXPERIMENT_V1
    return None


def latest_full_validation_dir() -> Path | None:
    root = PROJECT_ROOT / "experiments"
    if not root.exists():
        return None
    candidates = [p for p in root.glob("full_validation_ctspine1k_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def resumable_full_validation_dir(checkpoint: Path, config_path: Path) -> Path | None:
    """查找同一 checkpoint/config 尚未完成的 197 例 validation，供面板断点续跑。"""
    root = PROJECT_ROOT / "experiments"
    if not root.exists():
        return None
    checkpoint_text = str(checkpoint.resolve())
    config_text = str(config_path.resolve())
    candidates: list[Path] = []
    for folder in root.glob("full_validation_ctspine1k_*"):
        if not folder.is_dir():
            continue
        status = read_json(folder / "evaluation_status.json")
        if not status:
            continue
        if str(status.get("checkpoint") or "") != checkpoint_text:
            continue
        if str(status.get("config") or "") != config_text:
            continue
        expected = int(status.get("expected_case_count") or 0)
        completed = int(status.get("completed_case_count") or 0)
        if expected == EXPECTED_VAL and 0 <= completed < EXPECTED_VAL:
            candidates.append(folder)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)


def _raw_nifti_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.rglob("*.nii.gz"))


def _cache_complete_count() -> int:
    status = read_json(CACHE_ROOT / "preprocess_status.json")
    if int(status.get("total_case_count") or 0) == EXPECTED_RAW:
        value = int(status.get("completed_case_count") or 0)
        if 0 <= value <= EXPECTED_RAW:
            return value
    if not CACHE_ROOT.exists():
        return 0
    count = 0
    for case_dir in CACHE_ROOT.iterdir():
        if not case_dir.is_dir():
            continue
        if all(
            (case_dir / name).exists()
            for name in ("image_normalized_u16.nii.gz", "label.nii.gz", "metadata.json")
        ):
            count += 1
    return count


def _cache_split_counts() -> tuple[int, int, int]:
    payload = read_json(CACHE_ROOT / "split.json")
    return (
        len(payload.get("train", []) or []),
        len(payload.get("validation", []) or []),
        len(payload.get("test", []) or []),
    )


def _v4b_guidance_status() -> tuple[bool, int, int, str]:
    """轻量检查 v4-B 32例 train-only residual-FP guidance 是否完整。"""
    expected = 32
    summary = read_json(V4B_GUIDANCE_ROOT / "summary.json")
    status = read_json(V4B_GUIDANCE_ROOT / "guidance_status.json")
    phase = str(status.get("phase") or ("completed" if summary else "not_started"))
    if str(summary.get("split") or "") != "train":
        completed = int(status.get("completed_case_count") or 0)
        return False, completed, expected, phase
    cases = summary.get("cases") or []
    if int(summary.get("case_count") or 0) != expected or len(cases) != expected:
        completed = int(status.get("completed_case_count") or len(cases))
        return False, completed, expected, phase
    completed_files = 0
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if case_id and (V4B_GUIDANCE_ROOT / case_id / "hard_centers.nii.gz").exists():
            completed_files += 1
    ready = completed_files == expected and phase == "completed"
    return ready, completed_files, expected, phase


def _folder_size_gb(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return total / 1024**3


def _metric_mean(summary: dict, key: str):
    try:
        return summary["metrics"][key]["mean"]
    except (KeyError, TypeError):
        return None


def _display_phase(phase: str) -> str:
    """把内部状态翻译成面板上明确、可区分的中文状态。"""
    mapping = {
        "starting": "正在启动",
        "training": "正在训练",
        "validating": "正在 validation",
        "checkpointing": "正在保存 checkpoint",
        "epoch_complete": "Epoch 已完成",
        "early_stopped": "Early Stopping 正常停止",
        "interrupted": "已安全停止",
        "completed": "已完成",
        "already_complete": "已完成",
        "target_reached": "达到目标",
        "data_corrupt": "数据文件损坏",
        "failed": "异常退出",
    }
    return mapping.get(str(phase), str(phase) if phase else "未开始")


def _latest_history_row(run_dir: Path) -> dict[str, str]:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return {}
    try:
        with history_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    return rows[-1] if rows else {}


def _v2_pilot_gate(run_dir: Path, label: str = "v2") -> tuple[bool, str] | None:
    """读取前 5 轮历史，给出是否值得继续长训的保守门禁。"""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return None
    try:
        with history_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if len(rows) < 5:
        return None
    # 阶段门禁永远只看最初 5 Epoch，避免后续续训后反过来改变首阶段判断。
    rows = rows[:5]

    def number(row: dict, key: str) -> float | None:
        try:
            value = row.get(key)
            return None if value in {None, ""} else float(value)
        except (TypeError, ValueError):
            return None

    first = rows[0]
    last = rows[-1]
    val_values = [number(row, "val_dice") for row in rows]
    val_values = [value for value in val_values if value is not None]
    if not val_values:
        return False, f"[{label} 5轮门禁：停止] history.csv 没有可用 Val Dice，请先检查训练/验证链路。"

    first_val = number(first, "val_dice") or 0.0
    best_val = max(val_values)
    last_val = number(last, "val_dice") or 0.0
    first_loss = number(first, "train_loss")
    last_loss = number(last, "train_loss")
    pred_fg = number(last, "val_prediction_foreground_fraction")
    target_fg = number(last, "val_target_foreground_fraction")
    component_error = number(last, "val_component_count_error")
    foreground_ratio = (
        pred_fg / target_fg
        if pred_fg is not None and target_fg is not None and target_fg > 0.0
        else None
    )
    improvement = best_val - first_val

    blockers: list[str] = []
    if best_val < 0.08:
        blockers.append(f"Best Dice={best_val:.4f}<0.08")
    if improvement < 0.02:
        blockers.append(f"相对首轮提升仅 {improvement:+.4f}<0.02")
    if foreground_ratio is not None and foreground_ratio > 3.0:
        blockers.append(f"预测/真值前景比={foreground_ratio:.2f}>3")
    if component_error is not None and component_error > 300.0:
        blockers.append(f"component error={component_error:.1f}>300，仍明显碎片化")

    metrics = (
        f"Train Loss {fmt_num(first_loss)}→{fmt_num(last_loss)}；"
        f"Val Dice 首轮={first_val:.4f}，末轮={last_val:.4f}，Best={best_val:.4f}"
    )
    if foreground_ratio is not None:
        metrics += f"；预测/真值前景比={foreground_ratio:.2f}"
    if component_error is not None:
        metrics += f"；component error={component_error:.1f}"

    if blockers:
        return False, (
            f"[{label} 5轮门禁：停止并排查] " + "；".join(blockers) + "。" + metrics
            + "。不要直接继续到15/30/40轮。"
        )
    return True, (
        f"[{label} 5轮门禁：趋势通过] " + metrics
        + "。可以先做一次完整197例 validation；若全量指标也明显优于 v1，再继续到15轮。"
    )


def _file_age_seconds(path: Path) -> float | None:
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _cache_ready_lightweight(cache_root: Path = CACHE_ROOT) -> bool:
    """只读两个小 JSON 判断指定缓存是否可训练，不递归扫描 H 盘。"""
    status = read_json(cache_root / "preprocess_status.json")
    split = read_json(cache_root / "split.json")
    return (
        int(status.get("completed_case_count") or 0) == EXPECTED_RAW
        and int(status.get("failure_count") or 0) == 0
        and len(split.get("train", []) or []) == EXPECTED_TRAIN
        and len(split.get("validation", []) or []) == EXPECTED_VAL
        and len(split.get("test", []) or []) == EXPECTED_TEST
    )


def _evaluation_process_snapshot() -> dict:
    """查找完整 validation 评估进程，避免重复启动第二个 197 例验证。"""
    if psutil is None:
        return {}
    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "status", "memory_info", "create_time"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "src.modeling.evaluate" not in cmdline or "--split validation" not in cmdline:
                continue
            cpu_times = process.cpu_times()
            memory_info = process.info.get("memory_info")
            matches.append(
                {
                    "pid": int(process.info["pid"]),
                    "status": str(process.info.get("status") or "unknown"),
                    "rss_gb": 0.0 if memory_info is None else float(memory_info.rss) / 1024**3,
                    "cpu_seconds": float(cpu_times.user + cpu_times.system),
                    "create_time": float(process.info.get("create_time") or 0.0),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    if not matches:
        return {}
    return max(matches, key=lambda item: (item["rss_gb"], item["cpu_seconds"]))


def _training_process_snapshot() -> dict:
    """查找 v1/v2 完整数据训练进程；优先返回真正占内存/CPU的子进程。"""
    if psutil is None:
        return {}
    matches = []
    for process in psutil.process_iter(["pid", "cmdline", "status", "memory_info", "create_time"]):
        try:
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "src.modeling.train" not in cmdline:
                continue
            if not any(
                config_name in cmdline
                for config_name in (
                    "orthopedic_ct_full_large_scale_v1.yaml",
                    "orthopedic_ct_full_large_scale_v2.yaml",
                    "orthopedic_ct_full_large_scale_v3_fixed_hu.yaml",
                    "orthopedic_ct_full_large_scale_v4b_residual_fp_pilot32.yaml",
                )
            ):
                continue
            if "orthopedic_ct_full_large_scale_v4b_residual_fp_pilot32.yaml" in cmdline:
                experiment = EXPERIMENT_V4B
            elif "orthopedic_ct_full_large_scale_v3_fixed_hu.yaml" in cmdline:
                experiment = EXPERIMENT_V3
            elif "orthopedic_ct_full_large_scale_v2.yaml" in cmdline:
                experiment = EXPERIMENT_V2
            else:
                experiment = EXPERIMENT_V1
            cpu_times = process.cpu_times()
            memory_info = process.info.get("memory_info")
            matches.append(
                {
                    "pid": int(process.info["pid"]),
                    "status": str(process.info.get("status") or "unknown"),
                    "rss_gb": 0.0 if memory_info is None else float(memory_info.rss) / 1024**3,
                    "cpu_seconds": float(cpu_times.user + cpu_times.system),
                    "create_time": float(process.info.get("create_time") or 0.0),
                    "experiment": experiment,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
    if not matches:
        return {}
    return max(matches, key=lambda item: (item["rss_gb"], item["cpu_seconds"]))


class TrainingDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CTSpine1K 1005例训练控制台（v1/v2/v3 + v4-B residual FP）")
        self.geometry("1120x870")
        self.minsize(980, 760)

        self.train_process: subprocess.Popen[str] | None = None
        self.prep_process: subprocess.Popen[str] | None = None
        self.guidance_process: subprocess.Popen[str] | None = None
        self.eval_process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[tuple[str, str, str]] = queue.Queue()
        self.monitor_queue: queue.Queue[dict] = queue.Queue(maxsize=4)
        self._monitor_stop = threading.Event()
        self._monitor_force = threading.Event()
        self._cache_ready = _cache_ready_lightweight()
        self._last_data_refresh = 0.0
        self._eval_output_dir: Path | None = None
        self._last_process_cpu_seconds: float | None = None
        self._last_process_sample_at: float | None = None
        self._ui_pulse = False

        self.target_var = tk.StringVar(value="0.80")
        self.epochs_var = tk.StringVar(value="40")

        self.raw_var = tk.StringVar(value="检查中...")
        self.split_var = tk.StringVar(value="610 train / 197 validation / 198 test")
        self.cache_var = tk.StringVar(value="检查中...")
        self.disk_var = tk.StringVar(value="检查中...")
        self.device_var = tk.StringVar(value=self._device_text())

        self.run_var = tk.StringVar(value="尚未检测到 v1/v2/v3/v4-B 完整数据训练")
        self.phase_var = tk.StringVar(value="未开始")
        self.epoch_var = tk.StringVar(value="--")
        self.loss_var = tk.StringVar(value="--")
        self.val_var = tk.StringVar(value="--")
        self.best_var = tk.StringVar(value="--")
        self.precision_var = tk.StringVar(value="--")
        self.recall_var = tk.StringVar(value="--")
        self.fg_ratio_var = tk.StringVar(value="--")
        self.component_error_var = tk.StringVar(value="--")
        self.lr_var = tk.StringVar(value="--")
        self.elapsed_var = tk.StringVar(value="--:--:--")
        self.progress_var = tk.StringVar(value="0.0%")
        self.eta_var = tk.StringVar(value="--:--:--")
        self.judgement_var = tk.StringVar(value="等待紧凑训练缓存准备完成。")
        self.process_state_var = tk.StringVar(value="⚪ 训练进程：未检测到")
        self.train_pid_var = tk.StringVar(value="--")
        self.step_var = tk.StringVar(value="--")
        self.status_age_var = tk.StringVar(value="--")
        self.ui_heartbeat_var = tk.StringVar(value="● 面板响应正常")

        self.prep_phase_var = tk.StringVar(value="未准备")
        self.prep_progress_var = tk.StringVar(value="0 / 1005")
        self.prep_eta_var = tk.StringVar(value="--:--:--")
        guidance_ready, guidance_done, guidance_expected, guidance_phase = _v4b_guidance_status()
        self.v4b_guidance_var = tk.StringVar(
            value=(
                f"{'✓ 可训练' if guidance_ready else guidance_phase} | "
                f"{guidance_done}/{guidance_expected} train cases"
            )
        )

        self.full_eval_phase_var = tk.StringVar(value="尚未运行完整验证")
        self.full_eval_dice_var = tk.StringVar(value="--")
        self.full_eval_hd95_var = tk.StringVar(value="--")
        self.full_eval_assd_var = tk.StringVar(value="--")
        self.full_eval_cases_var = tk.StringVar(value="-- / 197")
        self.full_eval_verdict_var = tk.StringVar(
            value="训练内只用固定子集看趋势；真正判断到位要用 best.pt 跑完整197例 validation。"
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(250, self._poll_output)
        self.after(250, self._poll_monitor)
        self.after(500, self._pulse_ui_heartbeat)
        threading.Thread(target=self._background_monitor, name="dashboard-monitor", daemon=True).start()

    @staticmethod
    def _device_text() -> str:
        if torch.cuda.is_available():
            return f"CUDA：{torch.cuda.get_device_name(0)} | PyTorch {torch.__version__}"
        return f"CPU-only | PyTorch {torch.__version__}（610例3D训练会很慢）"

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        ttk.Label(
            root,
            text="CTSpine1K 1005例训练控制台｜v3稳定基线 + v4-B residual FP",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            root,
            text=(
                "完整数据采用官方隔离：610例训练 / 197例验证 / 198例最终test。"
                "test_private 绝不会进入训练或模型选择。"
            ),
        ).pack(anchor="w", pady=(4, 10))

        data_box = ttk.LabelFrame(root, text="① 数据与训练缓存", padding=10)
        data_box.pack(fill="x")
        grid = ttk.Frame(data_box)
        grid.pack(fill="x")
        data_fields = [
            ("原始数据", self.raw_var),
            ("官方划分", self.split_var),
            ("v3修正版缓存", self.cache_var),
            ("H盘空间", self.disk_var),
            ("训练设备", self.device_var),
        ]
        for row_index, (name, var) in enumerate(data_fields):
            ttk.Label(grid, text=f"{name}：", font=("Microsoft YaHei UI", 10, "bold")).grid(
                row=row_index, column=0, sticky="nw", padx=(0, 8), pady=2
            )
            ttk.Label(grid, textvariable=var, wraplength=850, justify="left").grid(
                row=row_index, column=1, sticky="w", pady=2
            )
        grid.columnconfigure(1, weight=1)

        prep_row = ttk.Frame(data_box)
        prep_row.pack(fill="x", pady=(8, 0))
        ttk.Button(
            prep_row,
            text="生成 / 继续 v3 修正版1005例缓存",
            command=self._start_preprocessing,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(prep_row, text="停止预处理", command=self._stop_preprocessing).pack(
            side="left", padx=(0, 8)
        )
        ttk.Button(prep_row, text="打开 H盘数据目录", command=self._open_data_dir).pack(side="left")
        ttk.Label(prep_row, text="预处理状态：").pack(side="left", padx=(18, 2))
        ttk.Label(prep_row, textvariable=self.prep_phase_var).pack(side="left")
        ttk.Label(prep_row, textvariable=self.prep_progress_var).pack(side="left", padx=(10, 0))
        ttk.Label(prep_row, text="ETA ").pack(side="left", padx=(10, 0))
        ttk.Label(prep_row, textvariable=self.prep_eta_var).pack(side="left")

        judge_box = ttk.LabelFrame(root, text="② 怎么知道训练真的到位", padding=10)
        judge_box.pack(fill="x", pady=(9, 0))
        ttk.Label(
            judge_box,
            text=(
                "v1 每个 epoch 用固定 12/197 例快速验证；v2/v3 用固定24例；v4-B pilot 用固定12例，并记录 Precision/Recall、fg ratio 与 component error。\n"
                "真正判断：训练暂停/结束后点“完整197例验证”，看 mean Dice、HD95、ASSD。"
                "只有完整验证达到你设的目标并且指标稳定，才叫验证集上基本到位。\n"
                "最终 test 198例不在这个窗口提供按钮，防止反复看 test 后继续调参造成数据泄漏。"
            ),
            justify="left",
            wraplength=1030,
        ).pack(anchor="w")

        controls = ttk.LabelFrame(root, text="③ 训练控制", padding=10)
        controls.pack(fill="x", pady=(9, 0))
        row = ttk.Frame(controls)
        row.pack(fill="x")
        ttk.Label(row, text="目标完整 Val Dice：").pack(side="left")
        ttk.Entry(row, textvariable=self.target_var, width=8).pack(side="left", padx=(0, 12))
        ttk.Label(row, text="总目标 Epoch：").pack(side="left")
        ttk.Entry(row, textvariable=self.epochs_var, width=8).pack(side="left", padx=(0, 14))
        ttk.Button(row, text="v1 开始 / 继续（baseline）", command=self._start_training).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(
            row,
            text="忽略早停，跑满总 Epoch",
            command=lambda: self._start_training(ignore_early_stopping=True),
        ).pack(side="left", padx=(0, 7))
        ttk.Button(row, text="安全停止训练", command=self._request_train_stop).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(row, text="打开实验目录", command=self._open_run_dir).pack(side="left")

        v2_row = ttk.Frame(controls)
        v2_row.pack(fill="x", pady=(7, 0))
        ttk.Button(v2_row, text="开始 v2 5轮试训", command=self._start_v2_pilot).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(v2_row, text="继续 v2 到总 Epoch", command=self._continue_v2).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(v2_row, text="完整197例验证 当前 best.pt", command=self._start_full_validation).pack(
            side="left", padx=(0, 7)
        )
        ttk.Label(
            v2_row,
            text="v2 仅保留历史复现；5轮门禁已判定不通过。",
        ).pack(side="left", padx=(8, 0))

        v3_row = ttk.Frame(controls)
        v3_row.pack(fill="x", pady=(7, 0))
        ttk.Button(v3_row, text="开始 v3 修正版 5轮试训", command=self._start_v3_pilot).pack(
            side="left", padx=(0, 7)
        )
        ttk.Button(v3_row, text="继续 v3 到总 Epoch", command=self._continue_v3).pack(
            side="left", padx=(0, 7)
        )
        ttk.Label(
            v3_row,
            text="v3 使用修正后的固定HU缓存；缓存未到1005/1005时禁止启动。",
        ).pack(side="left", padx=(8, 0))

        v4b_row = ttk.Frame(controls)
        v4b_row.pack(fill="x", pady=(7, 0))
        ttk.Button(
            v4b_row,
            text="① 生成/检查 v4-B 32例 residual-FP guidance",
            command=self._start_v4b_guidance,
        ).pack(side="left", padx=(0, 7))
        ttk.Button(
            v4b_row,
            text="② 开始 v4-B 2轮试训",
            command=self._start_v4b_pilot,
        ).pack(side="left", padx=(0, 7))
        ttk.Label(v4b_row, text="Guidance：").pack(side="left", padx=(8, 2))
        ttk.Label(v4b_row, textvariable=self.v4b_guidance_var).pack(side="left")
        ttk.Label(
            v4b_row,
            text="（自动从 v3 Epoch4 best.pt 初始化；不会 resume v3；test 198 不参与）",
        ).pack(side="left", padx=(8, 0))

        ttk.Label(
            controls,
            text=(
                "默认目标 0.80 是可修改的工程目标，不是预先宣称能达到的成绩。"
                "完整验证没过目标时，不应因为快速12例指标好看就说模型到位。"
            ),
            wraplength=1030,
        ).pack(anchor="w", pady=(7, 0))

        status = ttk.LabelFrame(root, text="④ 实时训练状态（v4-B pilot=12例快速 validation）", padding=10)
        status.pack(fill="x", pady=(9, 0))

        alive_row = ttk.Frame(status)
        alive_row.pack(fill="x", pady=(0, 7))
        self.process_state_label = tk.Label(
            alive_row,
            textvariable=self.process_state_var,
            font=("Microsoft YaHei UI", 12, "bold"),
            anchor="w",
            fg="#666666",
        )
        self.process_state_label.pack(side="left", fill="x", expand=True)
        ttk.Label(alive_row, textvariable=self.ui_heartbeat_var).pack(side="right")

        status_grid = ttk.Frame(status)
        status_grid.pack(fill="x")
        fields = [
            ("状态", self.phase_var),
            ("Epoch", self.epoch_var),
            ("Step", self.step_var),
            ("Train loss", self.loss_var),
            ("快速 Val Dice", self.val_var),
            ("快速 Best Dice", self.best_var),
            ("Precision", self.precision_var),
            ("Recall", self.recall_var),
            ("fg ratio", self.fg_ratio_var),
            ("component error", self.component_error_var),
            ("学习率", self.lr_var),
            ("运行时间", self.elapsed_var),
            ("进度", self.progress_var),
            ("ETA", self.eta_var),
            ("PID", self.train_pid_var),
            ("指标状态更新", self.status_age_var),
        ]
        for i, (name, var) in enumerate(fields):
            r, c = divmod(i, 3)
            cell = ttk.Frame(status_grid)
            cell.grid(row=r, column=c, sticky="ew", padx=7, pady=3)
            status_grid.columnconfigure(c, weight=1)
            ttk.Label(cell, text=f"{name}：", font=("Microsoft YaHei UI", 10, "bold")).pack(
                side="left"
            )
            ttk.Label(cell, textvariable=var).pack(side="left")
        ttk.Separator(status).pack(fill="x", pady=6)
        ttk.Label(status, textvariable=self.judgement_var, wraplength=1030, justify="left").pack(
            anchor="w"
        )
        ttk.Label(status, textvariable=self.run_var, wraplength=1030, justify="left").pack(
            anchor="w", pady=(5, 0)
        )

        full_box = ttk.LabelFrame(root, text="⑤ 完整197例验证结果", padding=10)
        full_box.pack(fill="x", pady=(9, 0))
        full_grid = ttk.Frame(full_box)
        full_grid.pack(fill="x")
        full_fields = [
            ("状态", self.full_eval_phase_var),
            ("病例", self.full_eval_cases_var),
            ("Mean Dice", self.full_eval_dice_var),
            ("Mean HD95", self.full_eval_hd95_var),
            ("Mean ASSD", self.full_eval_assd_var),
        ]
        for i, (name, var) in enumerate(full_fields):
            cell = ttk.Frame(full_grid)
            cell.grid(row=0, column=i, sticky="ew", padx=7, pady=2)
            full_grid.columnconfigure(i, weight=1)
            ttk.Label(cell, text=f"{name}：", font=("Microsoft YaHei UI", 9, "bold")).pack(
                side="left"
            )
            ttk.Label(cell, textvariable=var).pack(side="left")
        ttk.Label(
            full_box,
            textvariable=self.full_eval_verdict_var,
            wraplength=1030,
            justify="left",
        ).pack(anchor="w", pady=(5, 0))

        log_box = ttk.LabelFrame(root, text="运行日志", padding=6)
        log_box.pack(fill="both", expand=True, pady=(9, 0))
        self.log = tk.Text(log_box, height=11, wrap="word", font=("Consolas", 9))
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_box, orient="vertical", command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set, state="disabled")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _other_process_running(self, except_kind: str | None = None) -> bool:
        pairs = {
            "train": self.train_process,
            "prep": self.prep_process,
            "guidance": self.guidance_process,
            "eval": self.eval_process,
        }
        for kind, proc in pairs.items():
            if kind == except_kind:
                continue
            if proc is not None and proc.poll() is None:
                return True
        return False

    def _launch_process(self, kind: str, cmd: list[str], banner: str) -> subprocess.Popen[str] | None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
        except OSError as exc:
            messagebox.showerror("启动失败", str(exc))
            return None
        self._append_log(f"\n===== {banner} =====\n")
        threading.Thread(target=self._reader_thread, args=(kind, process), daemon=True).start()
        return process

    def _launch_training_detached(self, cmd: list[str]) -> int | None:
        """把正式训练和 Tk 面板进程彻底解耦。

        训练 stdout/stderr 直接追加到持久日志。面板关闭、重启或短暂卡顿时，
        训练进程不会因为 stdout PIPE / 父进程生命周期而一起退出。
        """
        log_dir = PROJECT_ROOT / "experiments" / "launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "full_ctspine1k_train.log"
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        try:
            with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
                log_file.write(
                    f"\n===== {datetime.now().isoformat()} 面板启动/续训 =====\n"
                    f"command: {' '.join(cmd)}\n"
                )
                process = subprocess.Popen(
                    cmd,
                    cwd=PROJECT_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                    close_fds=True,
                )
        except OSError as exc:
            messagebox.showerror("训练启动失败", str(exc))
            return None
        self._append_log(
            "\n===== 已独立启动完整 CTSpine1K 训练 =====\n"
            f"启动 PID：{process.pid}\n"
            f"启动日志：{log_path}\n"
            "训练已与面板解耦；现在可以关闭/重开面板，不会结束训练。\n"
        )
        return int(process.pid)

    def _reader_thread(self, kind: str, process: subprocess.Popen[str]) -> None:
        if process.stdout is not None:
            for line in process.stdout:
                self.output_queue.put((kind, "line", line))
        code = process.wait()
        self.output_queue.put((kind, "exit", str(code)))

    def _poll_output(self) -> None:
        try:
            while True:
                kind, event, value = self.output_queue.get_nowait()
                if event == "line":
                    self._append_log(value)
                elif event == "exit":
                    self._append_log(f"\n===== {kind} 进程结束，退出码 {value} =====\n")
                    if kind == "train":
                        self.train_process = None
                        self._monitor_force.set()
                    elif kind == "prep":
                        self.prep_process = None
                        self._monitor_force.set()
                    elif kind == "guidance":
                        self.guidance_process = None
                        ready, done, expected, phase = _v4b_guidance_status()
                        self.v4b_guidance_var.set(
                            f"{'✓ 可训练' if ready else phase} | {done}/{expected} train cases"
                        )
                        self._monitor_force.set()
                    elif kind == "eval":
                        self.eval_process = None
                        self._monitor_force.set()
                        self._load_full_validation_result()
        except queue.Empty:
            pass
        self.after(300, self._poll_output)

    def _start_preprocessing(self) -> None:
        prep_status = read_json(CACHE_ROOT / "preprocess_status.json")
        prep_age = _file_age_seconds(CACHE_ROOT / "preprocess_status.json")
        external_preparing = (
            str(prep_status.get("state") or "") == "preparing"
            and prep_age is not None
            and prep_age < 120
        )
        if (self.prep_process is not None and self.prep_process.poll() is None) or external_preparing:
            messagebox.showinfo("正在预处理", "v3 修正版紧凑缓存正在生成，不会重复启动第二个进程。")
            return
        active_train = _training_process_snapshot()
        if active_train:
            messagebox.showwarning(
                "训练正在运行",
                f"检测到训练进程 PID {active_train.get('pid')}，训练期间不要重新生成缓存。",
            )
            return
        if self._other_process_running(except_kind="prep"):
            messagebox.showwarning("当前有任务运行", "请先停止训练/验证，再生成训练缓存。")
            return
        if not RAW_ROOT.exists():
            messagebox.showerror("原始数据不存在", f"没有找到 {RAW_ROOT}")
            return
        cmd = [
            str(PYTHON),
            "-m",
            "src.preprocessing.prepare_ctspine1k_compact",
            "--source-root",
            str(RAW_ROOT),
            "--output-root",
            str(CACHE_ROOT),
        ]
        self.prep_process = self._launch_process("prep", cmd, "开始 / 继续生成 v3 修正版1005例缓存")
        if self.prep_process is not None:
            self.prep_phase_var.set("preparing")

    def _stop_preprocessing(self) -> None:
        prep_status = read_json(CACHE_ROOT / "preprocess_status.json")
        prep_age = _file_age_seconds(CACHE_ROOT / "preprocess_status.json")
        attached_running = self.prep_process is not None and self.prep_process.poll() is None
        external_running = (
            str(prep_status.get("state") or "") == "preparing"
            and prep_age is not None
            and prep_age < 120
        )
        if not attached_running and not external_running:
            messagebox.showinfo("没有运行中的预处理", "当前没有正在运行的 v3 修正版全量预处理。")
            return
        try:
            CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            (CACHE_ROOT / "PREPROCESS_STOP_REQUESTED").write_text(
                f"requested_at={datetime.now().isoformat()}\n", encoding="utf-8"
            )
            self.prep_phase_var.set("正在安全停止")
            self._append_log("\n[面板] 已请求预处理在当前病例结束后安全停止。\n")
        except OSError as exc:
            messagebox.showerror("停止失败", str(exc))

    def _validate_inputs(self) -> tuple[float, int] | None:
        try:
            target = float(self.target_var.get().strip())
            epochs = int(self.epochs_var.get().strip())
        except ValueError:
            messagebox.showerror("参数错误", "目标 Dice 必须是 0~1 的小数，总 Epoch 必须是正整数。")
            return None
        if not 0 <= target <= 1 or epochs <= 0:
            messagebox.showerror("参数错误", "目标 Dice 必须在 0~1，总 Epoch 必须大于 0。")
            return None
        return target, epochs

    def _start_v4b_guidance(self) -> None:
        ready, done, expected, phase = _v4b_guidance_status()
        self.v4b_guidance_var.set(
            f"{'✓ 可训练' if ready else phase} | {done}/{expected} train cases"
        )
        if ready:
            messagebox.showinfo(
                "v4-B guidance 已完整",
                f"已经完成 {done}/{expected} 个 train 病例。可以直接点击“② 开始 v4-B 2轮试训”。",
            )
            return
        if self.guidance_process is not None and self.guidance_process.poll() is None:
            messagebox.showinfo("正在生成 guidance", "v4-B residual-FP guidance 已在运行。")
            return
        active_train = _training_process_snapshot()
        if active_train:
            messagebox.showwarning(
                "训练正在运行",
                f"检测到训练进程 PID {active_train.get('pid')}，请先安全停止/等待训练结束。",
            )
            return
        if self._other_process_running(except_kind="guidance"):
            messagebox.showwarning("当前有任务运行", "请先等待预处理/验证结束后再生成 guidance。")
            return
        if not _cache_ready_lightweight(CACHE_ROOT):
            messagebox.showwarning("fixed-HU 缓存未就绪", str(CACHE_ROOT))
            return
        if not V3_BEST.exists():
            messagebox.showerror("v3 best.pt 不存在", str(V3_BEST))
            return
        cmd = [
            str(PYTHON),
            "-m",
            "src.modeling.generate_residual_fp_guidance",
            "--config",
            str(CONFIG_V4B.relative_to(PROJECT_ROOT)),
            "--checkpoint",
            str(V3_BEST),
            "--output-root",
            str(V4B_GUIDANCE_ROOT),
            "--max-cases",
            "32",
            "--max-centers-per-case",
            "2",
            "--min-center-distance-voxels",
            "48",
            "--resume",
        ]
        self.guidance_process = self._launch_process(
            "guidance",
            cmd,
            "生成 v4-B train-only residual-FP guidance（running BN + ROI96 + pp34816）",
        )
        if self.guidance_process is not None:
            self.v4b_guidance_var.set(f"running | {done}/{expected} train cases")
            self._monitor_force.set()

    def _start_v4b_pilot(self) -> None:
        ready, done, expected, phase = _v4b_guidance_status()
        if not ready:
            messagebox.showwarning(
                "v4-B guidance 尚未完整",
                f"当前 {done}/{expected}，状态={phase}。请先点击“① 生成/检查 v4-B 32例 residual-FP guidance”。",
            )
            return
        if not V3_BEST.exists():
            messagebox.showerror("v3 best.pt 不存在", str(V3_BEST))
            return
        run_dir = latest_run_dir(EXPERIMENT_V4B)
        init_checkpoint: Path | None = V3_BEST
        if run_dir is not None:
            summary = read_json(run_dir / "summary.json")
            status = read_json(run_dir / "training_status.json")
            last_epoch = int(summary.get("last_epoch") or status.get("epoch") or 0)
            final_status = str(summary.get("status") or status.get("phase") or "")
            if last_epoch >= 2 or final_status in {"completed", "target_reached"}:
                messagebox.showinfo(
                    "v4-B 2轮 pilot 已完成",
                    f"检测到 {run_dir.name}\nepoch={last_epoch}，状态={final_status or '未知'}。\n"
                    "先查看 Precision / Recall / fg ratio / Dice，再决定是否进入 610例短训。",
                )
                return
            # 已有同 config 的未完成 run：让 train.py 按 exact config 自动 resume last.pt。
            init_checkpoint = None
        self._start_training(
            config_path=CONFIG_V4B,
            experiment_name=EXPERIMENT_V4B,
            max_epochs_override=2,
            fresh=False,
            init_checkpoint=init_checkpoint,
        )

    def _start_training(
        self,
        ignore_early_stopping: bool = False,
        *,
        config_path: Path = CONFIG_V1,
        experiment_name: str = EXPERIMENT_V1,
        max_epochs_override: int | None = None,
        fresh: bool = False,
        init_checkpoint: Path | None = None,
    ) -> None:
        if self.train_process is not None and self.train_process.poll() is None:
            messagebox.showinfo("正在训练", "训练进程已经在运行。")
            return
        active_train = _training_process_snapshot()
        if active_train:
            self._monitor_force.set()
            messagebox.showinfo(
                "已检测到正在训练",
                f"当前已经有 {active_train.get('experiment') or '完整数据'} 训练进程 "
                f"PID {active_train.get('pid')}。\n面板会直接监控它，不会重复启动第二个训练进程。",
            )
            return
        if self._other_process_running(except_kind="train"):
            messagebox.showwarning("当前有任务运行", "请等预处理/完整验证结束后再训练。")
            return

        run_dir = latest_run_dir(experiment_name)
        if run_dir is not None:
            summary = read_json(run_dir / "summary.json")
            if str(summary.get("status") or "") == "early_stopped" and not ignore_early_stopping:
                messagebox.showwarning(
                    "当前实验已 Early Stopped",
                    f"{experiment_name} 已经因为连续多轮 Val Dice 没有改善而正常 Early Stopping。\n\n"
                    "普通“开始/继续”不会再自动多跑一轮。若只是做历史复现实验，可用高级忽略早停；"
                    "若目标是提升效果，应修改方案或进入下一阶段，而不是硬续。",
                )
                return

        cache_root = (
            CACHE_ROOT
            if experiment_name in {EXPERIMENT_V3, EXPERIMENT_V4B}
            else LEGACY_CACHE_ROOT
        )
        cache_ready = _cache_ready_lightweight(cache_root)
        if experiment_name in {EXPERIMENT_V3, EXPERIMENT_V4B}:
            self._cache_ready = cache_ready
        self._monitor_force.set()
        if not cache_ready:
            version_text = (
                "v3/v4-B fixed-HU"
                if experiment_name in {EXPERIMENT_V3, EXPERIMENT_V4B}
                else "v1/v2 历史"
            )
            messagebox.showwarning(
                "训练缓存未就绪",
                f"{version_text} 1005例缓存尚未完整准备好：{cache_root}。\n"
                "v3/v4-B 必须等 fixed-HU 缓存 1005/1005 且 failure=0 后才能启动。",
            )
            return
        values = self._validate_inputs()
        if values is None:
            return
        target, requested_epochs = values
        epochs = int(max_epochs_override or requested_epochs)
        if not torch.cuda.is_available():
            self._append_log(
                "\n[提示] 当前是 CPU-only。训练会较慢；是否仍在运行请看绿色进程状态与心跳。\n"
            )

        cmd = [
            str(PYTHON),
            "-m",
            "src.modeling.train",
            "--config",
            str(config_path.relative_to(PROJECT_ROOT)),
            "--max-epochs",
            str(epochs),
            "--target-dice",
            str(target),
            "--preflight-mode",
            "engineering",
            "--status-every-steps",
            "5",
        ]
        if init_checkpoint is not None:
            init_checkpoint = Path(init_checkpoint)
            if not init_checkpoint.exists():
                messagebox.showerror("初始化 checkpoint 不存在", str(init_checkpoint))
                return
            cmd.extend(["--init-checkpoint", str(init_checkpoint)])
            self._append_log(
                f"\n[面板] {experiment_name} 从 {init_checkpoint} 仅初始化模型权重；"
                "新建 optimizer/history/epoch，不会 resume v3。\n"
            )
        if fresh:
            cmd.append("--fresh")
            self._append_log(
                f"\n[面板] {experiment_name} 本次明确 fresh training，不继承 v1 或其它 run 的 optimizer/scheduler。\n"
            )
        if ignore_early_stopping:
            cmd.extend(["--early-stopping-patience", "1000000"])
            self._append_log(
                "\n[面板] 本次显式忽略 Early Stopping，只在达到总 Epoch / 手动停止 / 异常时结束。\n"
            )
        launched_pid = self._launch_training_detached(cmd)
        if launched_pid is not None:
            self.train_process = None
            self.phase_var.set("启动中")
            self.process_state_var.set(
                f"🟡 {experiment_name}：正在启动训练（PID {launched_pid}）"
            )
            self.train_pid_var.set(str(launched_pid))
            self._monitor_force.set()

    def _start_v2_pilot(self) -> None:
        run_dir = latest_run_dir(EXPERIMENT_V2)
        fresh = run_dir is None
        if run_dir is not None:
            summary = read_json(run_dir / "summary.json")
            status = read_json(run_dir / "training_status.json")
            last_epoch = int(summary.get("last_epoch") or status.get("epoch") or 0)
            final_status = str(summary.get("status") or status.get("phase") or "")
            if last_epoch >= 5 or final_status in {"completed", "early_stopped", "target_reached"}:
                messagebox.showinfo(
                    "v2 5轮试训已存在",
                    f"检测到已有 v2 run：{run_dir.name}\n"
                    f"当前 epoch={last_epoch}，状态={final_status or '未知'}。\n"
                    "不会再新建重复 v2；请先看指标，再决定是否点“继续 v2 到总 Epoch”。",
                )
                return
        self._start_training(
            config_path=CONFIG_V2,
            experiment_name=EXPERIMENT_V2,
            max_epochs_override=5,
            fresh=fresh,
        )

    def _continue_v2(self) -> None:
        run_dir = latest_run_dir(EXPERIMENT_V2)
        if run_dir is None:
            messagebox.showinfo("还没有 v2", "请先点击“开始 v2 5轮试训”。")
            return
        values = self._validate_inputs()
        if values is None:
            return
        _target, epochs = values
        summary = read_json(run_dir / "summary.json")
        status = read_json(run_dir / "training_status.json")
        last_epoch = int(summary.get("last_epoch") or status.get("epoch") or 0)
        if last_epoch < 5:
            messagebox.showwarning(
                "v2 首阶段尚未完成",
                f"当前 v2 只到 epoch {last_epoch}/5。请先完成 5 Epoch 试训，不能直接跳到长训练。",
            )
            return
        gate = _v2_pilot_gate(run_dir)
        if gate is None:
            messagebox.showwarning(
                "无法判断 v2 5轮趋势",
                "没有读到完整的前5轮 history.csv 指标，先不要继续长训练。",
            )
            return
        gate_passed, gate_message = gate
        if not gate_passed:
            messagebox.showwarning("v2 5轮门禁未通过", gate_message)
            self.judgement_var.set(gate_message)
            return
        if epochs <= last_epoch:
            messagebox.showinfo(
                "总 Epoch 未增加",
                f"当前 v2 已到 epoch {last_epoch}；请把“总目标 Epoch”设置为更大的值（建议先到15，再决定30/40）。",
            )
            return
        self.judgement_var.set(gate_message)
        self._start_training(
            config_path=CONFIG_V2,
            experiment_name=EXPERIMENT_V2,
            max_epochs_override=epochs,
            fresh=False,
        )

    def _start_v3_pilot(self) -> None:
        run_dir = latest_run_dir(EXPERIMENT_V3)
        fresh = run_dir is None
        if run_dir is not None:
            summary = read_json(run_dir / "summary.json")
            status = read_json(run_dir / "training_status.json")
            last_epoch = int(summary.get("last_epoch") or status.get("epoch") or 0)
            final_status = str(summary.get("status") or status.get("phase") or "")
            if last_epoch >= 5 or final_status in {"completed", "early_stopped", "target_reached"}:
                messagebox.showinfo(
                    "v3 5轮试训已存在",
                    f"检测到已有 v3 run：{run_dir.name}\n"
                    f"当前 epoch={last_epoch}，状态={final_status or '未知'}。\n"
                    "不会重复新建 v3；请先看5轮门禁，再决定是否继续。",
                )
                return
        self._start_training(
            config_path=CONFIG_V3,
            experiment_name=EXPERIMENT_V3,
            max_epochs_override=5,
            fresh=fresh,
        )

    def _continue_v3(self) -> None:
        run_dir = latest_run_dir(EXPERIMENT_V3)
        if run_dir is None:
            messagebox.showinfo("还没有 v3", "请先点击“开始 v3 修正版 5轮试训”。")
            return
        values = self._validate_inputs()
        if values is None:
            return
        _target, epochs = values
        summary = read_json(run_dir / "summary.json")
        status = read_json(run_dir / "training_status.json")
        last_epoch = int(summary.get("last_epoch") or status.get("epoch") or 0)
        if last_epoch < 5:
            messagebox.showwarning(
                "v3 首阶段尚未完成",
                f"当前 v3 只到 epoch {last_epoch}/5。请先完成 5 Epoch 试训，不能直接跳到长训练。",
            )
            return
        gate = _v2_pilot_gate(run_dir, label="v3")
        if gate is None:
            messagebox.showwarning(
                "无法判断 v3 5轮趋势",
                "没有读到完整的前5轮 history.csv 指标，先不要继续长训练。",
            )
            return
        gate_passed, gate_message = gate
        if not gate_passed:
            messagebox.showwarning("v3 5轮门禁未通过", gate_message)
            self.judgement_var.set(gate_message)
            return
        if epochs <= last_epoch:
            messagebox.showinfo(
                "总 Epoch 未增加",
                f"当前 v3 已到 epoch {last_epoch}；请把“总目标 Epoch”设置为更大的值（建议先到15）。",
            )
            return
        self.judgement_var.set(gate_message)
        self._start_training(
            config_path=CONFIG_V3,
            experiment_name=EXPERIMENT_V3,
            max_epochs_override=epochs,
            fresh=False,
        )

    def _request_train_stop(self) -> None:
        run_dir = latest_run_dir()
        if run_dir is None:
            messagebox.showinfo("没有训练 run", "还没有找到完整数据训练 run。")
            return
        status = read_json(run_dir / "training_status.json")
        phase = str(status.get("phase", ""))
        if phase in {"completed", "target_reached", "early_stopped", "interrupted", "already_complete"}:
            messagebox.showinfo("无需停止", "当前 run 已经不在训练。")
            return
        try:
            (run_dir / "STOP_REQUESTED").write_text(
                f"requested_at={datetime.now().isoformat()}\n", encoding="utf-8"
            )
            self.phase_var.set("正在安全停止")
            self._append_log(
                "\n[面板] 已发送安全停止请求；当前未完成 epoch 会丢弃，最近完整 last.pt 保留。\n"
            )
        except OSError as exc:
            messagebox.showerror("停止失败", str(exc))

    def _start_full_validation(self) -> None:
        if self.eval_process is not None and self.eval_process.poll() is None:
            messagebox.showinfo("正在完整验证", "197例完整 validation 已经在运行。")
            return
        active_eval = _evaluation_process_snapshot()
        if active_eval:
            self._monitor_force.set()
            messagebox.showinfo(
                "正在完整验证",
                f"检测到 197例 validation 进程 PID {active_eval.get('pid')}，不会重复启动。",
            )
            return
        active_train = _training_process_snapshot()
        if active_train:
            messagebox.showwarning(
                "训练仍在运行",
                f"检测到训练进程 PID {active_train.get('pid')}。请先安全停止/等待训练结束，再跑完整197例验证。",
            )
            return
        if self._other_process_running(except_kind="eval"):
            messagebox.showwarning("当前有任务运行", "请先安全停止训练/预处理，再运行完整验证。")
            return
        run_dir = latest_run_dir()
        if run_dir is None:
            messagebox.showwarning("还没有 checkpoint", "请先至少完成一个训练 epoch。")
            return
        experiment_for_cache = run_experiment_name(run_dir)
        cache_root = CACHE_ROOT if experiment_for_cache == EXPERIMENT_V3 else LEGACY_CACHE_ROOT
        cache_ready = _cache_ready_lightweight(cache_root)
        if experiment_for_cache == EXPERIMENT_V3:
            self._cache_ready = cache_ready
        self._monitor_force.set()
        if not cache_ready:
            messagebox.showwarning(
                "缓存未就绪",
                f"当前 run 对应的1005例缓存尚未完整准备完成：{cache_root}",
            )
            return
        checkpoint = run_dir / "checkpoint" / "best.pt"
        if not checkpoint.exists():
            messagebox.showwarning("best.pt 不存在", "请先至少完成一个带验证的训练 epoch。")
            return

        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            messagebox.showerror("run 配置缺失", f"没有找到 {config_path}")
            return
        experiment = run_experiment_name(run_dir) or "unknown"
        resume_dir = resumable_full_validation_dir(checkpoint, config_path)
        resume_eval = resume_dir is not None
        if resume_dir is not None:
            output_dir = resume_dir
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                PROJECT_ROOT / "experiments" / f"full_validation_ctspine1k_{experiment}_{stamp}"
            )
        self._eval_output_dir = output_dir
        cmd = [
            str(PYTHON),
            "-m",
            "src.modeling.evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--split",
            "validation",
            "--output-dir",
            str(output_dir),
            "--preflight-mode",
            "engineering",
        ]
        if resume_eval:
            cmd.append("--resume")
        banner = "继续完整197例validation（断点续跑）" if resume_eval else "开始完整197例validation"
        self.eval_process = self._launch_process("eval", cmd, banner)
        if self.eval_process is not None:
            self.full_eval_phase_var.set("running")
            self.full_eval_verdict_var.set("正在跑完整197例 validation；CPU 环境下可能需要较长时间。")

    def _open_data_dir(self) -> None:
        target = CACHE_ROOT if CACHE_ROOT.exists() else RAW_ROOT
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _open_run_dir(self) -> None:
        run_dir = latest_run_dir()
        target = run_dir if run_dir is not None else PROJECT_ROOT / "experiments"
        try:
            os.startfile(target)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc))

    def _put_monitor_snapshot(self, snapshot: dict) -> None:
        try:
            self.monitor_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self.monitor_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.monitor_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _background_monitor(self) -> None:
        """所有 H 盘扫描和进程探测都在后台线程执行，绝不阻塞 Tk 主线程。"""
        last_heavy_scan = 0.0
        while not self._monitor_stop.is_set():
            forced = self._monitor_force.is_set()
            self._monitor_force.clear()
            now = time.monotonic()

            guidance_ready, guidance_done, guidance_expected, guidance_phase = _v4b_guidance_status()
            snapshot: dict = {
                "prep": read_json(CACHE_ROOT / "preprocess_status.json"),
                "process": _training_process_snapshot(),
                "v4b_guidance": {
                    "ready": guidance_ready,
                    "done": guidance_done,
                    "expected": guidance_expected,
                    "phase": guidance_phase,
                },
            }

            run_dir = latest_run_dir()
            if run_dir is not None:
                status_path = run_dir / "training_status.json"
                heartbeat_path = run_dir / "training_heartbeat.json"
                snapshot["run_dir"] = str(run_dir)
                snapshot["train_status"] = read_json(status_path)
                snapshot["train_summary"] = read_json(run_dir / "summary.json")
                snapshot["history_latest"] = _latest_history_row(run_dir)
                snapshot["heartbeat"] = read_json(heartbeat_path)
                snapshot["status_age"] = _file_age_seconds(status_path)
                snapshot["heartbeat_age"] = _file_age_seconds(heartbeat_path)

            eval_dir = self._eval_output_dir or latest_full_validation_dir()
            if eval_dir is not None:
                snapshot["full_validation_status"] = read_json(eval_dir / "evaluation_status.json")
                summary_path = eval_dir / "summary.json"
                if summary_path.exists():
                    snapshot["full_validation_summary"] = read_json(summary_path)

            # 先把轻量训练状态立即交给 UI；H 盘慢扫描不能拖住“是否还在训练”的判断。
            self._put_monitor_snapshot(dict(snapshot))

            # 递归统计原始数据/缓存体积是最容易卡 UI 的操作，只在后台且低频执行。
            if forced or now - last_heavy_scan >= 30.0:
                volumes = _raw_nifti_count(RAW_ROOT / "raw_data" / "volumes")
                labels = _raw_nifti_count(RAW_ROOT / "raw_data" / "labels")
                cache_count = _cache_complete_count()
                train_count, val_count, test_count = _cache_split_counts()
                try:
                    free_gb = shutil.disk_usage("H:/").free / 1024**3
                except OSError:
                    free_gb = 0.0
                snapshot["data"] = {
                    "volumes": volumes,
                    "labels": labels,
                    "cache_count": cache_count,
                    "train_count": train_count,
                    "val_count": val_count,
                    "test_count": test_count,
                    "free_gb": free_gb,
                    "cache_gb": _folder_size_gb(CACHE_ROOT),
                }
                last_heavy_scan = now
                self._put_monitor_snapshot(snapshot)

            self._monitor_force.wait(1.0)

    def _poll_monitor(self) -> None:
        latest = None
        try:
            while True:
                latest = self.monitor_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self._apply_monitor_snapshot(latest)
        self.after(250, self._poll_monitor)

    def _pulse_ui_heartbeat(self) -> None:
        self._ui_pulse = not self._ui_pulse
        symbol = "●" if self._ui_pulse else "○"
        self.ui_heartbeat_var.set(f"{symbol} 面板响应正常 {time.strftime('%H:%M:%S')}")
        self.after(500, self._pulse_ui_heartbeat)

    def _apply_monitor_snapshot(self, snapshot: dict) -> None:
        data = snapshot.get("data")
        if isinstance(data, dict):
            volumes = int(data.get("volumes") or 0)
            labels = int(data.get("labels") or 0)
            raw_ok = volumes == EXPECTED_RAW and labels == EXPECTED_RAW
            self.raw_var.set(
                f"volume {volumes}/{EXPECTED_RAW}，label {labels}/{EXPECTED_RAW} "
                + ("✓ 完整" if raw_ok else "✗ 不完整")
            )
            train_count = int(data.get("train_count") or 0)
            val_count = int(data.get("val_count") or 0)
            test_count = int(data.get("test_count") or 0)
            cache_count = int(data.get("cache_count") or 0)
            split_ready = (
                train_count == EXPECTED_TRAIN
                and val_count == EXPECTED_VAL
                and test_count == EXPECTED_TEST
            )
            self._cache_ready = raw_ok and cache_count == EXPECTED_RAW and split_ready
            prep_state = read_json(CACHE_ROOT / "preprocess_status.json").get("state", "未开始")
            self.cache_var.set(
                f"{cache_count}/{EXPECTED_RAW} 例 | split {train_count}/{val_count}/{test_count} | "
                f"state={prep_state} " + ("✓ 可训练" if self._cache_ready else "尚未就绪")
            )
            self.split_var.set(
                f"train={EXPECTED_TRAIN} / validation={EXPECTED_VAL} / test={EXPECTED_TEST} "
                "（官方 trainset / test_public / test_private）"
            )
            self.disk_var.set(
                f"H盘剩余 {float(data.get('free_gb') or 0.0):.2f} GB | "
                f"紧凑缓存当前 {float(data.get('cache_gb') or 0.0):.2f} GB"
            )

        guidance = snapshot.get("v4b_guidance") or {}
        if isinstance(guidance, dict) and guidance:
            ready = bool(guidance.get("ready"))
            done = int(guidance.get("done") or 0)
            expected = int(guidance.get("expected") or 32)
            guidance_phase = str(guidance.get("phase") or "not_started")
            self.v4b_guidance_var.set(
                f"{'✓ 可训练' if ready else guidance_phase} | {done}/{expected} train cases"
            )

        prep = snapshot.get("prep") or {}
        if isinstance(prep, dict):
            state = str(prep.get("state") or "未准备")
            completed = int(prep.get("completed_case_count") or 0)
            total = int(prep.get("total_case_count") or EXPECTED_RAW)
            self.prep_phase_var.set(state)
            self.prep_progress_var.set(f"{completed} / {total}")
            self.prep_eta_var.set(fmt_duration(prep.get("eta_seconds")))
            if completed == EXPECTED_RAW and int(prep.get("failure_count") or 0) == 0:
                self._cache_ready = _cache_ready_lightweight()

        status = snapshot.get("train_status") or {}
        summary = snapshot.get("train_summary") or {}
        history_latest = snapshot.get("history_latest") or {}
        heartbeat = snapshot.get("heartbeat") or {}
        process = snapshot.get("process") or {}
        status_age = snapshot.get("status_age")
        heartbeat_age = snapshot.get("heartbeat_age")

        phase = str(status.get("phase") or summary.get("status") or "未开始")
        self.phase_var.set(_display_phase(phase))
        epoch = status.get("epoch", summary.get("last_epoch"))
        max_epochs = status.get("max_epochs", summary.get("target_max_epochs"))
        self.epoch_var.set(
            f"{epoch if epoch is not None else '--'} / {max_epochs if max_epochs is not None else '--'}"
        )
        step = heartbeat.get("step") if heartbeat_age is not None and heartbeat_age < 8 else None
        total_steps = heartbeat.get("total_steps") if step is not None else None
        if step is None:
            step = status.get("step")
            total_steps = status.get("total_steps")
        self.step_var.set(
            "--" if step is None else f"{step} / {total_steps if total_steps is not None else '--'}"
        )
        self.loss_var.set(fmt_num(status.get("train_loss")))
        self.val_var.set(fmt_num(status.get("val_dice", summary.get("last_val_dice"))))
        best = status.get("best_val_dice", summary.get("best_val_dice"))
        self.best_var.set(fmt_num(best))
        self.precision_var.set(fmt_num(history_latest.get("val_precision")))
        self.recall_var.set(fmt_num(history_latest.get("val_recall")))
        self.fg_ratio_var.set(fmt_num(history_latest.get("val_prediction_to_target_foreground_ratio"), digits=4))
        self.component_error_var.set(fmt_num(history_latest.get("val_component_count_error"), digits=3))
        self.lr_var.set("--" if status.get("lr") is None else f"{float(status['lr']):.3e}")
        self.elapsed_var.set(
            fmt_duration(status.get("elapsed_seconds", summary.get("training_seconds_total")))
        )
        self.progress_var.set(
            "--"
            if status.get("progress_percent") is None
            else f"{float(status['progress_percent']):.1f}%"
        )
        self.eta_var.set(fmt_duration(status.get("eta_seconds")))
        self.status_age_var.set("--" if status_age is None else f"{float(status_age):.0f} 秒前")
        if snapshot.get("run_dir"):
            self.run_var.set(f"当前 run：{snapshot['run_dir']}")

        process_alive = bool(process)
        pid = process.get("pid") or heartbeat.get("pid")
        self.train_pid_var.set("--" if pid is None else str(pid))

        cpu_active = False
        if process_alive:
            cpu_seconds = float(process.get("cpu_seconds") or 0.0)
            sample_at = time.monotonic()
            if self._last_process_cpu_seconds is not None and self._last_process_sample_at is not None:
                cpu_active = cpu_seconds > self._last_process_cpu_seconds + 0.02
            self._last_process_cpu_seconds = cpu_seconds
            self._last_process_sample_at = sample_at
            if phase == "validating":
                activity = "正在 validation"
            elif phase == "checkpointing":
                activity = "正在保存 checkpoint"
            else:
                activity = "正在训练计算" if cpu_active else "训练进程存活"
            self.process_state_var.set(
                f"🟢 {activity} | PID {process.get('pid')} | "
                f"RAM {float(process.get('rss_gb') or 0.0):.2f} GB"
            )
            self.process_state_label.configure(fg="#16803a")
        elif heartbeat_age is not None and heartbeat_age < 8:
            self.process_state_var.set(
                f"🟢 训练进程：心跳正常 | PID {heartbeat.get('pid')} | {float(heartbeat_age):.1f} 秒前"
            )
            self.process_state_label.configure(fg="#16803a")
        elif phase in {"training", "validating", "checkpointing", "starting"}:
            self.process_state_var.set("🔴 训练状态：未检测到活跃进程，请检查日志")
            self.process_state_label.configure(fg="#b42318")
        else:
            self.process_state_var.set(f"⚪ 训练进程：未运行 | 状态 {phase}")
            self.process_state_label.configure(fg="#666666")

        # training_status.json 是当前运行状态的权威来源；summary 只在状态文件缺失时兜底。
        final_phase = phase
        current_run_dir = Path(snapshot["run_dir"]) if snapshot.get("run_dir") else None
        current_experiment = run_experiment_name(current_run_dir)
        quick_val_cases = 24 if current_experiment in {EXPERIMENT_V2, EXPERIMENT_V3} else QUICK_VAL_CASES
        if process_alive or (heartbeat_age is not None and heartbeat_age < 8):
            if phase == "validating":
                verdict = (
                    f"[正在 validation] 当前 epoch 训练部分已结束，正在跑固定 {quick_val_cases}/197 例完整体快速验证。"
                    "这一步 CPU 会明显慢于普通训练 step。"
                )
            elif status_age is not None and float(status_age) > 20:
                verdict = (
                    f"[正在训练] 进程/心跳正常，但指标文件已 {float(status_age):.0f} 秒没更新。"
                    "这通常表示当前一个 3D step 较慢，不是界面卡死。"
                )
            else:
                verdict = (
                    f"[正在训练] 进程存活；本 run 每轮使用固定 {quick_val_cases}/197 例做快速 validation。"
                )
        elif final_phase == "early_stopped":
            verdict = (
                "[Early Stopping 正常停止] 连续多轮快速 Val Dice 未改善。这不是崩溃；"
                "普通开始/继续不会自动多跑一轮。"
            )
        elif final_phase in {"completed", "already_complete"}:
            if current_experiment == EXPERIMENT_V2 and current_run_dir:
                gate = _v2_pilot_gate(current_run_dir, label="v2")
            elif current_experiment == EXPERIMENT_V3 and current_run_dir:
                gate = _v2_pilot_gate(current_run_dir, label="v3")
            else:
                gate = None
            if gate is not None:
                verdict = gate[1]
            else:
                verdict = "[已完成] 当前训练阶段完成；用 best.pt 做完整197例 validation 再决定下一阶段。"
        elif final_phase == "target_reached":
            verdict = "[达到目标] 已按设定目标安全停止。先锁定 checkpoint，并做完整197例 validation 确认。"
        elif final_phase == "interrupted":
            verdict = "[已安全停止] last.pt 已保留；再次点击对应版本继续训练会从完整 epoch 续。"
        elif final_phase == "data_corrupt":
            verdict = "[数据文件损坏] 训练因病例文件读取/shape 异常退出；请查看 failure.json 中的 case/path。"
        elif final_phase == "failed":
            verdict = "[异常退出] 训练进程异常结束；checkpoint/history 未被覆盖，请查看 failure.json 和启动日志。"
        else:
            verdict = "当前没有检测到正在运行的完整数据训练进程。"
        self.judgement_var.set(verdict)

        full_summary = snapshot.get("full_validation_summary")
        if isinstance(full_summary, dict) and full_summary:
            self._apply_full_validation_summary(full_summary)
        else:
            full_status = snapshot.get("full_validation_status")
            if isinstance(full_status, dict) and full_status:
                expected = int(full_status.get("expected_case_count") or EXPECTED_VAL)
                completed = int(full_status.get("completed_case_count") or 0)
                phase_text = str(full_status.get("phase") or "running")
                self.full_eval_phase_var.set("正在 validation" if phase_text == "running" else phase_text)
                self.full_eval_cases_var.set(f"{completed} / {expected}")
                self.full_eval_verdict_var.set(
                    f"完整 validation 已完成 {completed}/{expected} 例；每病例完成后都会落盘，可安全断点续跑。"
                )

    def _apply_full_validation_summary(self, summary: dict) -> None:
        dice = _metric_mean(summary, "dice")
        hd95 = _metric_mean(summary, "hd95_mm")
        assd = _metric_mean(summary, "assd_mm")
        case_count = summary.get("metrics", {}).get("case_count")
        self.full_eval_phase_var.set("completed")
        self.full_eval_cases_var.set(f"{case_count if case_count is not None else '--'} / {EXPECTED_VAL}")
        self.full_eval_dice_var.set(fmt_num(dice))
        self.full_eval_hd95_var.set("--" if hd95 is None else f"{float(hd95):.2f} mm")
        self.full_eval_assd_var.set("--" if assd is None else f"{float(assd):.2f} mm")

        try:
            target = float(self.target_var.get().strip())
        except ValueError:
            target = None
        if case_count != EXPECTED_VAL:
            verdict = "[结果不完整] 本次没有覆盖全部197个 validation 病例，不能作为完整验证结论。"
        elif target is not None and dice is not None and float(dice) >= target:
            verdict = (
                f"[完整验证达到目标] 197例 mean Dice={float(dice):.4f} >= 目标 {target:.4f}。"
                "这说明 validation 层面达到你设定的目标；之后应先锁定参数/checkpoint，再做一次最终 test。"
            )
        elif dice is not None:
            verdict = (
                f"[完整验证未达到目标] 197例 mean Dice={float(dice):.4f}。"
                "不要因为12例快速指标较高就说训练到位；应继续训练或改方案，且不要查看test。"
            )
        else:
            verdict = "完整验证 summary 已生成，但没有可用 mean Dice，请检查日志。"
        self.full_eval_verdict_var.set(verdict)

    def _load_full_validation_result(self) -> None:
        eval_dir = self._eval_output_dir or latest_full_validation_dir()
        if eval_dir is None:
            return
        summary = read_json(eval_dir / "summary.json")
        if summary:
            self._apply_full_validation_summary(summary)
        else:
            self.full_eval_phase_var.set("未生成summary")

    def _on_close(self) -> None:
        running = []
        if self.prep_process is not None and self.prep_process.poll() is None:
            running.append("预处理")
        if self.train_process is not None and self.train_process.poll() is None:
            running.append("训练")
        if self.guidance_process is not None and self.guidance_process.poll() is None:
            running.append("v4-B guidance")
        if self.eval_process is not None and self.eval_process.poll() is None:
            running.append("完整验证")
        if running:
            messagebox.showwarning(
                "仍有任务运行",
                "以下任务仍在运行：" + "、".join(running) + "。请先安全停止/等待结束后再关闭面板。",
            )
            return
        self._monitor_stop.set()
        self._monitor_force.set()
        self.destroy()


if __name__ == "__main__":
    TrainingDashboard().mainloop()
