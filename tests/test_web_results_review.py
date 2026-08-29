import csv
import importlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi.testclient import TestClient

webapp = importlib.import_module("web.backend.app")


def _write_image(path: Path, array: np.ndarray) -> None:
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((0.8, 1.0, 1.2))
    image.SetOrigin((10.0, 20.0, -30.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def _prepare_evaluation(processed_root: Path, experiments_root: Path) -> tuple[str, str]:
    case_id = "ctspine1k-msd-t10-eval_test"
    case_dir = processed_root / case_id
    case_dir.mkdir(parents=True)
    bone = np.zeros((12, 14, 16), dtype=np.float32)
    bone[2:10, 3:12, 4:14] = 0.8
    _write_image(case_dir / "image_bone_window.nii.gz", bone)
    label = np.zeros_like(bone, dtype=np.int16)
    label[3:10, 4:12, 5:14] = 24
    _write_image(case_dir / "label.nii.gz", label)

    evaluation_id = "evaluation_test"
    evaluation_dir = experiments_root / evaluation_id
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "evaluated_at": "2026-08-16T19:30:00",
                "split": "test",
                "device": "cpu",
                "checkpoint": "checkpoint.pt",
                "config": "config.yaml",
                "metrics": {"case_count": 1},
            }
        ),
        encoding="utf-8",
    )
    with (evaluation_dir / "metrics_per_case.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["case_id", "dice", "hd95_mm", "inference_seconds"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": case_id,
                "dice": "0.8125",
                "hd95_mm": "2.5",
                "inference_seconds": "1.25",
            }
        )

    prediction = np.zeros_like(bone, dtype=np.int16)
    prediction[4:9, 5:11, 6:13] = 24
    _write_image(
        evaluation_dir / "predictions" / case_id / "prediction.nii.gz",
        prediction,
    )
    uncertainty = np.zeros_like(bone, dtype=np.float32)
    uncertainty[5:8, 6:10, 7:12] = 0.9
    _write_image(
        evaluation_dir / "uncertainty" / case_id / "predictive_entropy.nii.gz",
        uncertainty,
    )
    return evaluation_id, case_id


def test_results_review_lists_evaluation_and_serves_prediction_uncertainty_mpr(
    tmp_path: Path, monkeypatch
) -> None:
    processed_root = tmp_path / "processed"
    experiments_root = tmp_path / "experiments"
    processed_root.mkdir()
    experiments_root.mkdir()
    evaluation_id, case_id = _prepare_evaluation(processed_root, experiments_root)
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", processed_root)
    monkeypatch.setattr(webapp, "EXPERIMENTS_ROOT", experiments_root)

    with TestClient(webapp.app) as client:
        page = client.get("/results-review")
        assert page.status_code == 200
        assert "模型评估结果复核" in page.text

        listing = client.get("/api/research/evaluations")
        assert listing.status_code == 200
        payload = listing.json()
        assert payload["total"] == 1
        assert payload["evaluations"][0]["evaluation_id"] == evaluation_id
        assert payload["evaluations"][0]["case_count"] == 1

        detail = client.get(f"/api/research/evaluations/{evaluation_id}")
        assert detail.status_code == 200
        case = detail.json()["cases"][0]
        assert case["case_id"] == case_id
        assert case["dice"] == 0.8125
        assert case["prediction_available"] is True
        assert case["uncertainty_available"] is True

        prediction = client.get(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mpr",
            params={"mode": "prediction", "plane": "coronal", "position": 0.5},
        )
        assert prediction.status_code == 200
        assert prediction.headers["content-type"].startswith("image/png")
        assert prediction.headers["x-overlay-mode"] == "prediction"
        assert prediction.content.startswith(b"\x89PNG\r\n\x1a\n")

        uncertainty = client.get(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mpr",
            params={"mode": "uncertainty", "plane": "sagittal", "position": 0.5},
        )
        assert uncertainty.status_code == 200
        assert uncertainty.headers["x-overlay-mode"] == "uncertainty"
        assert uncertainty.content.startswith(b"\x89PNG\r\n\x1a\n")

        built_prediction = client.post(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh/build",
            params={"source": "prediction", "simplify_mm": 2.0},
        )
        assert built_prediction.status_code == 200
        prediction_payload = built_prediction.json()
        assert prediction_payload["source"] == "prediction"
        assert prediction_payload["summary"]["selection"] == "foreground_gt_0"
        assert prediction_payload["summary"]["simplification"]["method"] == (
            "vertex_clustering_feature_weighted"
        )

        prediction_summary = client.get(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh/summary",
            params={"source": "prediction", "simplify_mm": 2.0},
        )
        assert prediction_summary.status_code == 200
        assert prediction_summary.json()["vertex_count"] > 0

        prediction_mesh = client.get(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh",
            params={"source": "prediction", "simplify_mm": 2.0},
        )
        assert prediction_mesh.status_code == 200
        assert prediction_mesh.content.startswith(b"ply")

        built_gt = client.post(
            f"/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh/build",
            params={"source": "gt", "simplify_mm": 2.0},
        )
        assert built_gt.status_code == 200
        assert built_gt.json()["source"] == "gt"


def test_results_review_empty_root_and_unknown_case_are_safe(tmp_path: Path, monkeypatch) -> None:
    processed_root = tmp_path / "processed"
    experiments_root = tmp_path / "experiments"
    processed_root.mkdir()
    experiments_root.mkdir()
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", processed_root)
    monkeypatch.setattr(webapp, "EXPERIMENTS_ROOT", experiments_root)

    with TestClient(webapp.app) as client:
        empty = client.get("/api/research/evaluations")
        assert empty.status_code == 200
        assert empty.json()["total"] == 0

    evaluation_id, case_id = _prepare_evaluation(processed_root, experiments_root)
    with TestClient(webapp.app) as client:
        unknown = client.get(
            f"/api/research/evaluations/{evaluation_id}/cases/not_in_metrics/mpr"
        )
        assert unknown.status_code == 404
        assert "不存在此 case_id" in unknown.json()["detail"]

        missing_eval = client.get("/api/research/evaluations/../bad")
        assert missing_eval.status_code in {400, 404}
