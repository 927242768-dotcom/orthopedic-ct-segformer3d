import importlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi.testclient import TestClient

webapp = importlib.import_module("web.backend.app")


def _prepare_research_case(root: Path) -> str:
    case_id = "ctspine1k-msd-t10-mesh_test"
    case_dir = root / case_id
    case_dir.mkdir(parents=True)

    label = np.zeros((14, 16, 18), dtype=np.int16)
    label[3:11, 4:13, 5:15] = 7
    image = sitk.GetImageFromArray(label)
    image.SetSpacing((0.8, 1.0, 1.2))
    image.SetOrigin((10.0, 20.0, -30.0))
    sitk.WriteImage(image, str(case_dir / "label.nii.gz"))
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pipeline_version": "0.3.0",
                "label": {"label_values_after": [0, 7]},
            }
        ),
        encoding="utf-8",
    )
    (root / "ctspine1k_manifest.json").write_text(
        json.dumps([{"case_id": case_id, "source_split": "trainset"}]),
        encoding="utf-8",
    )
    return case_id


def test_research_3d_api_builds_and_serves_local_physical_mesh(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    case_id = _prepare_research_case(root)
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", root)

    with TestClient(webapp.app) as client:
        page = client.get("/research-3d")
        assert page.status_code == 200
        assert "真实骨结构 3D 工程查看器" in page.text

        cases = client.get("/api/research/cases")
        assert cases.status_code == 200
        payload = cases.json()
        assert payload["total"] == 1
        assert payload["cases"][0]["case_id"] == case_id
        assert payload["cases"][0]["source_split"] == "trainset"
        assert payload["cases"][0]["label_values"] == [0, 7]
        assert payload["cases"][0]["label_items"] == [
            {"value": 7, "name": "C7", "display": "C7 (7)"}
        ]
        assert payload["cases"][0]["label_schema_id"] == "ctspine1k_verse_vertebrae_1_25"
        assert payload["cases"][0]["formal_task_locked"] is False

        missing = client.get(f"/api/research/cases/{case_id}/mesh")
        assert missing.status_code == 404

        built = client.post(f"/api/research/cases/{case_id}/mesh/build")
        assert built.status_code == 200
        built_payload = built.json()
        assert built_payload["status"] == "built"
        assert built_payload["summary"]["vertex_count"] > 0
        assert built_payload["summary"]["face_count"] > 0

        mesh = client.get(f"/api/research/cases/{case_id}/mesh")
        assert mesh.status_code == 200
        assert mesh.content.startswith(b"ply")

        summary = client.get(f"/api/research/cases/{case_id}/mesh/summary")
        assert summary.status_code == 200
        assert summary.json()["selection"] == "foreground_gt_0"

        class_mesh = client.post(f"/api/research/cases/{case_id}/mesh/build?class_id=7")
        assert class_mesh.status_code == 200
        assert class_mesh.json()["summary"]["selection"] == "class_7"

        simplified = client.post(
            f"/api/research/cases/{case_id}/mesh/build?simplify_mm=1.5"
        )
        assert simplified.status_code == 200
        simplified_payload = simplified.json()
        assert simplified_payload["simplify_mm"] == 1.5
        assert simplified_payload["summary"]["simplification"]["cluster_size_mm"] == 1.5
        assert simplified_payload["summary"]["vertex_count"] < built_payload["summary"]["vertex_count"]
        simplified_mesh = client.get(
            f"/api/research/cases/{case_id}/mesh?simplify_mm=1.5"
        )
        assert simplified_mesh.status_code == 200

        sdf = client.post(
            f"/api/research/cases/{case_id}/mesh/build?surface=sdf&sdf_sigma_mm=0.4"
        )
        assert sdf.status_code == 200
        sdf_payload = sdf.json()
        assert sdf_payload["surface"] == "sdf"
        assert sdf_payload["sdf_sigma_mm"] == 0.4
        assert sdf_payload["summary"]["surface_method"] == "sdf_smoothed_zero_level"
        assert sdf_payload["summary"]["metrics"]["component_count_preserved"] is True
        assert sdf_payload["summary"]["vertex_count"] > 0

        sdf_summary = client.get(
            f"/api/research/cases/{case_id}/mesh/summary?surface=sdf&sdf_sigma_mm=0.4"
        )
        assert sdf_summary.status_code == 200
        sdf_mesh = client.get(
            f"/api/research/cases/{case_id}/mesh?surface=sdf&sdf_sigma_mm=0.4"
        )
        assert sdf_mesh.status_code == 200
        assert sdf_mesh.content.startswith(b"ply")

        sdf_with_simplification = client.post(
            f"/api/research/cases/{case_id}/mesh/build?surface=sdf&sdf_sigma_mm=0.4&simplify_mm=1.5"
        )
        assert sdf_with_simplification.status_code == 422
        assert "暂不与 vertex-clustering 同时启用" in sdf_with_simplification.json()["detail"]


def test_research_3d_refuses_existing_sdf_artifact_when_component_guard_failed(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    case_id = _prepare_research_case(root)
    case_dir = root / case_id
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", root)

    ply = case_dir / "mesh_foreground_sdf0p8.ply"
    ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    ply.with_suffix(".json").write_text(
        json.dumps(
            {
                "metrics": {
                    "original_components": 2,
                    "smoothed_components": 3,
                    "component_count_preserved": False,
                }
            }
        ),
        encoding="utf-8",
    )

    with TestClient(webapp.app) as client:
        summary = client.get(
            f"/api/research/cases/{case_id}/mesh/summary?surface=sdf&sdf_sigma_mm=0.8"
        )
        assert summary.status_code == 422
        assert "拒绝加载" in summary.json()["detail"]

        mesh = client.get(
            f"/api/research/cases/{case_id}/mesh?surface=sdf&sdf_sigma_mm=0.8"
        )
        assert mesh.status_code == 422
        assert "2 -> 3" in mesh.json()["detail"]


def test_research_3d_rejects_missing_label_class(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    case_id = _prepare_research_case(root)
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", root)

    with TestClient(webapp.app) as client:
        response = client.post(f"/api/research/cases/{case_id}/mesh/build?class_id=99")
    assert response.status_code == 422
    assert "不存在" in response.json()["detail"]


def test_physical_measurement_api_returns_mm_and_degrees() -> None:
    with TestClient(webapp.app) as client:
        distance = client.post(
            "/api/research/measure/distance",
            json={
                "point_a": {"x": 0, "y": 0, "z": 0},
                "point_b": {"x": 3, "y": 4, "z": 0},
            },
        )
        assert distance.status_code == 200
        assert distance.json()["distance_mm"] == 5.0

        angle = client.post(
            "/api/research/measure/angle",
            json={
                "point_a": {"x": 1, "y": 0, "z": 0},
                "vertex_b": {"x": 0, "y": 0, "z": 0},
                "point_c": {"x": 0, "y": 1, "z": 0},
            },
        )
        assert angle.status_code == 200
        assert angle.json()["angle_degrees"] == 90.0

        degenerate = client.post(
            "/api/research/measure/angle",
            json={
                "point_a": {"x": 0, "y": 0, "z": 0},
                "vertex_b": {"x": 0, "y": 0, "z": 0},
                "point_c": {"x": 0, "y": 1, "z": 0},
            },
        )
        assert degenerate.status_code == 422
