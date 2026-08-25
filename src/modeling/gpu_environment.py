"""正式 3D CT 训练前的 GPU / CUDA 环境只读验收。

不安装、不卸载、不修改任何驱动或 Python 环境；仅收集 PyTorch CUDA 状态、
可见设备、显存和 ``nvidia-smi`` 可达性，并给出 machine-readable report。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GPUDeviceInfo:
    index: int
    name: str
    total_memory_gb: float
    compute_capability: str | None


@dataclass(frozen=True)
class GPUEnvironmentReport:
    ready: bool
    python_executable: str
    python_version: str
    project_venv: bool
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    device_count: int
    devices: list[GPUDeviceInfo]
    nvidia_smi_available: bool
    nvidia_smi_summary: str | None
    minimum_memory_gb: float
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_gpu_environment(
    *,
    torch_cuda_version: str | None,
    cuda_available: bool,
    devices: list[GPUDeviceInfo],
    nvidia_smi_available: bool,
    minimum_memory_gb: float,
) -> list[str]:
    issues: list[str] = []
    if torch_cuda_version is None:
        issues.append("当前 PyTorch 不是 CUDA build")
    if not cuda_available:
        issues.append("torch.cuda.is_available()=False")
    if not devices:
        issues.append("PyTorch 没有可见 CUDA device")
    if not nvidia_smi_available:
        issues.append("nvidia-smi 不可用，无法确认驱动侧 GPU 状态")
    if devices and max(device.total_memory_gb for device in devices) < minimum_memory_gb:
        issues.append(
            f"最大可见 GPU 显存低于当前最低验收线 {minimum_memory_gb:g} GiB；"
            "需降低 ROI/batch 或使用更大显存设备"
        )
    return issues


def _nvidia_smi_summary() -> tuple[bool, str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return False, None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return False, None
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout).strip() or None
    return True, completed.stdout.strip() or None


def inspect_gpu_environment(minimum_memory_gb: float = 8.0) -> GPUEnvironmentReport:
    if minimum_memory_gb <= 0:
        raise ValueError("minimum_memory_gb 必须 > 0")

    devices: list[GPUDeviceInfo] = []
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = None
            try:
                major, minor = torch.cuda.get_device_capability(index)
                capability = f"{major}.{minor}"
            except Exception:
                pass
            devices.append(
                GPUDeviceInfo(
                    index=index,
                    name=str(properties.name),
                    total_memory_gb=float(properties.total_memory / (1024**3)),
                    compute_capability=capability,
                )
            )

    smi_available, smi_summary = _nvidia_smi_summary()
    torch_cuda_version = None if torch.version.cuda is None else str(torch.version.cuda)
    issues = assess_gpu_environment(
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        devices=devices,
        nvidia_smi_available=smi_available,
        minimum_memory_gb=float(minimum_memory_gb),
    )

    executable = Path(sys.executable).resolve()
    expected_venv = (PROJECT_ROOT / ".venv").resolve()
    project_venv = expected_venv in executable.parents
    if not project_venv:
        issues.append(f"当前 Python 不在项目 .venv: {executable}")

    return GPUEnvironmentReport(
        ready=not issues,
        python_executable=str(executable),
        python_version=sys.version.split()[0],
        project_venv=project_venv,
        torch_version=str(torch.__version__),
        torch_cuda_version=torch_cuda_version,
        cuda_available=cuda_available,
        device_count=len(devices),
        devices=devices,
        nvidia_smi_available=smi_available,
        nvidia_smi_summary=smi_summary,
        minimum_memory_gb=float(minimum_memory_gb),
        issues=issues,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Read-only GPU/CUDA readiness check")
    parser.add_argument("--minimum-memory-gb", type=float, default=8.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = inspect_gpu_environment(args.minimum_memory_gb)
    text = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report.ready else 2)


if __name__ == "__main__":
    main()
