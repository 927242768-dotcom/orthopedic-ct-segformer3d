from types import SimpleNamespace

from src.modeling.formal_readiness import (
    FormalReadinessBlocker,
    combine_readiness_reports,
)


def _report(*, ready: bool, issues: list[object]) -> SimpleNamespace:
    return SimpleNamespace(ready=ready, issues=issues)


def test_combine_readiness_reports_collects_all_error_categories() -> None:
    task = _report(
        ready=False,
        issues=[
            SimpleNamespace(severity="error", code="task_not_locked", message="not locked"),
            SimpleNamespace(severity="warning", code="note", message="warning only"),
        ],
    )
    gpu = _report(ready=False, issues=["cuda unavailable", "no device"])
    preflight = _report(
        ready=False,
        issues=[
            SimpleNamespace(
                severity="error",
                code="human_qc_not_approved",
                message="qc unsigned",
                case_id="case_001",
            ),
            SimpleNamespace(
                severity="warning",
                code="non_blocking",
                message="warning",
                case_id=None,
            ),
        ],
    )

    blockers = combine_readiness_reports(
        task,
        gpu,
        preflight,
        config_binding_blockers=[
            FormalReadinessBlocker(
                category="config",
                code="config_task_fingerprint_mismatch",
                message="fingerprint mismatch",
            )
        ],
    )

    assert [(item.category, item.code) for item in blockers] == [
        ("task", "task_not_locked"),
        ("gpu", "gpu_environment_not_ready"),
        ("preflight", "human_qc_not_approved"),
        ("config", "config_task_fingerprint_mismatch"),
    ]
    assert "case=case_001" in blockers[2].message


def test_combine_readiness_reports_can_explicitly_allow_cpu() -> None:
    task = _report(ready=True, issues=[])
    gpu = _report(ready=False, issues=["cuda unavailable"])
    preflight = _report(ready=True, issues=[])

    assert combine_readiness_reports(task, gpu, preflight, require_gpu=False) == []


def test_combine_readiness_reports_returns_empty_when_all_ready() -> None:
    task = _report(ready=True, issues=[])
    gpu = _report(ready=True, issues=[])
    preflight = _report(ready=True, issues=[])

    assert combine_readiness_reports(task, gpu, preflight) == []
