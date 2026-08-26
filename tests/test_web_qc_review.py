import csv
import importlib
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from fastapi.testclient import TestClient

webapp = importlib.import_module("web.backend.app")


def _prepare_qc_root(root: Path) -> None:
    case_id = "ctspine1k-msd-t10-case_test"
    case_dir = root / case_id
    case_dir.mkdir(parents=True)
    # FileResponse 测试只需要存在的 contact sheet；交互式 MPR 使用真实合成 NIfTI。
    (case_dir / "qc_contact_sheet.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    bone = np.zeros((10, 12, 14), dtype=np.float32)
    bone[2:8, 3:10, 4:12] = 0.65
    label = np.zeros((10, 12, 14), dtype=np.int16)
    label[3:7, 4:9, 5:11] = 7
    bone_image = sitk.GetImageFromArray(bone)
    bone_image.SetSpacing((1.0, 1.0, 1.0))
    label_image = sitk.GetImageFromArray(label)
    label_image.CopyInformation(bone_image)
    sitk.WriteImage(bone_image, str(case_dir / "image_bone_window.nii.gz"))
    sitk.WriteImage(label_image, str(case_dir / "label.nii.gz"))

    fieldnames = [
        "case_id",
        "qc_image",
        "auto_has_label",
        "auto_label_values",
        "orientation_ok",
        "spacing_ok",
        "label_alignment_ok",
        "bone_window_ok",
        "review_status",
        "reviewer",
        "notes",
    ]
    with (root / "manual_qc_review.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "case_id": case_id,
                "qc_image": str(case_dir / "qc_contact_sheet.png"),
                "auto_has_label": "True",
                "auto_label_values": "[0, 1]",
                "orientation_ok": "",
                "spacing_ok": "",
                "label_alignment_ok": "",
                "bone_window_ok": "",
                "review_status": "",
                "reviewer": "",
                "notes": "",
            }
        )


def test_qc_review_api_lists_and_saves_manual_review(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    _prepare_qc_root(root)
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", root)

    with TestClient(webapp.app) as client:
        page = client.get("/qc-review")
        assert page.status_code == 200
        assert "真实 CT 人工 QC 审核" in page.text

        listing = client.get("/api/research/qc")
        assert listing.status_code == 200
        payload = listing.json()
        assert payload["total"] == 1
        assert payload["pending"] == 1
        assert payload["cases"][0]["review_status"] == ""
        assert payload["cases"][0]["auto_label_display"] == "C1 (1)"
        assert payload["cases"][0]["auto_label_items"] == [
            {"value": 1, "name": "C1", "display": "C1 (1)"}
        ]

        image = client.get("/api/research/qc/ctspine1k-msd-t10-case_test/image")
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/png")

        mpr = client.get(
            "/api/research/qc/ctspine1k-msd-t10-case_test/mpr"
            "?plane=coronal&position=0.5&overlay=true"
        )
        assert mpr.status_code == 200
        assert mpr.headers["content-type"].startswith("image/png")
        assert mpr.headers["x-mpr-plane"] == "coronal"
        assert int(mpr.headers["x-mpr-index"]) >= 0
        assert mpr.headers["x-label-overlay"] == "true"
        assert len(mpr.content) > 50

        review = {
            "orientation_ok": True,
            "spacing_ok": True,
            "label_alignment_ok": True,
            "bone_window_ok": True,
            "review_status": "pass",
            "reviewer": "tester",
            "notes": "checked",
        }
        saved = client.post(
            "/api/research/qc/ctspine1k-msd-t10-case_test",
            json=review,
        )
        assert saved.status_code == 200
        assert saved.json()["case"]["review_status"] == "pass"

        listing_after = client.get("/api/research/qc").json()
        assert listing_after["reviewed"] == 1
        assert listing_after["passed"] == 1
        assert listing_after["pending"] == 0

    with (root / "manual_qc_review.csv").open("r", encoding="utf-8-sig", newline="") as file:
        row = next(csv.DictReader(file))
    assert row["orientation_ok"] == "yes"
    assert row["review_status"] == "pass"
    assert row["reviewer"] == "tester"
    assert row["notes"] == "checked"


def test_qc_review_rejects_pass_when_any_check_failed(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "processed"
    root.mkdir()
    _prepare_qc_root(root)
    monkeypatch.setattr(webapp, "RESEARCH_PROCESSED_ROOT", root)

    with TestClient(webapp.app) as client:
        response = client.post(
            "/api/research/qc/ctspine1k-msd-t10-case_test",
            json={
                "orientation_ok": True,
                "spacing_ok": True,
                "label_alignment_ok": False,
                "bone_window_ok": True,
                "review_status": "pass",
                "reviewer": "tester",
                "notes": "",
            },
        )
    assert response.status_code == 422
    assert "四项人工检查" in response.json()["detail"]


def test_qc_review_frontend_collapses_case_list_and_enters_review_area() -> None:
    with TestClient(webapp.app) as client:
        page = client.get("/qc-review")
        script = client.get("/static/qc_review.js")
        styles = client.get("/static/qc_review.css")

    assert page.status_code == 200
    assert script.status_code == 200
    assert styles.status_code == 200
    assert page.headers["cache-control"] == "no-store, max-age=0"

    assert "/static/qc_review.css?v=20260825-3" in page.text
    assert "/static/qc_review.js?v=20260825-3" in page.text
    assert 'id="qcLayout"' in page.text
    assert 'id="caseSidebar"' in page.text
    assert 'id="qcMain"' in page.text
    assert 'id="caseListToggleBtn"' in page.text
    assert 'aria-controls="caseSidebar"' in page.text

    assert "function setCaseListCollapsed" in script.text
    assert "function enterReviewArea" in script.text
    assert "scrollIntoView" in script.text
    assert 'button.addEventListener("click", () => selectCase(index));' in script.text
    assert "sidebar.hidden = collapsed" in script.text
    assert "setCaseListCollapsed(true);" in script.text
    assert "enterReviewArea({ smooth });" in script.text
    assert "selectCase(qcState.index - 1)" in script.text
    assert "selectCase(qcState.index + 1)" in script.text
    assert '"显示病例列表"' in script.text
    assert '"收起病例列表"' in script.text

    assert ".qc-layout > .card { grid-column: auto; }" in styles.text
    assert ".qc-layout.case-list-collapsed" in styles.text
    assert ".qc-sidebar[hidden] { display: none !important; }" in styles.text
    assert ".qc-layout.case-list-collapsed .qc-main { grid-column: 1 / -1; }" in styles.text
    assert "@media (max-width: 980px)" in styles.text
