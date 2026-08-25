from src.modeling.gpu_environment import (
    GPUDeviceInfo,
    assess_gpu_environment,
    inspect_gpu_environment,
)


def test_assess_gpu_environment_accepts_healthy_cuda_device() -> None:
    issues = assess_gpu_environment(
        torch_cuda_version="11.8",
        cuda_available=True,
        devices=[
            GPUDeviceInfo(
                index=0,
                name="Test GPU",
                total_memory_gb=16.0,
                compute_capability="8.6",
            )
        ],
        nvidia_smi_available=True,
        minimum_memory_gb=8.0,
    )
    assert issues == []


def test_assess_gpu_environment_reports_cpu_build_and_low_memory() -> None:
    issues = assess_gpu_environment(
        torch_cuda_version=None,
        cuda_available=False,
        devices=[],
        nvidia_smi_available=False,
        minimum_memory_gb=8.0,
    )
    assert "当前 PyTorch 不是 CUDA build" in issues
    assert "torch.cuda.is_available()=False" in issues
    assert "PyTorch 没有可见 CUDA device" in issues
    assert "nvidia-smi 不可用，无法确认驱动侧 GPU 状态" in issues

    low_memory = assess_gpu_environment(
        torch_cuda_version="11.8",
        cuda_available=True,
        devices=[
            GPUDeviceInfo(
                index=0,
                name="Small GPU",
                total_memory_gb=6.0,
                compute_capability="7.5",
            )
        ],
        nvidia_smi_available=True,
        minimum_memory_gb=8.0,
    )
    assert any("显存低于" in issue for issue in low_memory)


def test_inspect_gpu_environment_returns_machine_readable_report() -> None:
    report = inspect_gpu_environment(minimum_memory_gb=8.0)
    payload = report.to_dict()
    assert payload["minimum_memory_gb"] == 8.0
    assert isinstance(payload["cuda_available"], bool)
    assert isinstance(payload["devices"], list)
    assert isinstance(payload["issues"], list)
    assert payload["project_venv"] is True
