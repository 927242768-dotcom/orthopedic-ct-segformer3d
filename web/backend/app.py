"""Θ¬¿τºæ CT τºæτáöσ₧ï Web Φ╛àσè⌐σêåµ₧ÉσÄƒσ₧ïσÉÄτ½»πÇé

σ╜ôσëìσ«₧τÄ░µÅÉΣ╛¢∩╝Ü
- µ£¼σ£░σüÑσ║╖µúÇµƒÑ∩╝¢
- σñÜµûçΣ╗╢τùàΣ╛ïΣ╕èΣ╝á∩╝¢
- DICOM/NIfTI σƒ║τíÇΦ»åσê½Σ╕ÄΦ┤¿µÄºµæÿΦªü∩╝¢
- Σ╕¡σñ«Φ╜┤Σ╜ìΘ¬¿τ¬ùΘóäΦºê∩╝¢
- µ¿íσ₧ïµÄ¿τÉåµÄÑσÅúσìáΣ╜ì∩╝êσÅ¬µ£ëτ£ƒσ«₧ checkpoint Σ╕ÄµÄ¿τÉåΘÇéΘàìσ«îµêÉσÉÄµëìσÉ»τö¿∩╝ëπÇé

σ«ëσà¿Φ╛╣τòî∩╝Ü
- Θ╗ÿΦ«ñΣ╗àσ╗║Φ««τ╗æσ«Ü 127.0.0.1∩╝¢
- Σ╕ìΣ┐¥σ¡ÿΣ╕èΣ╝áµûçΣ╗╢τÜäσÄƒσºïµûçΣ╗╢σÉì∩╝îΘÖìΣ╜ÄµäÅσñûµÜ┤Θ£▓Φ║½Σ╗╜Σ┐íµü»ΘúÄΘÖ⌐∩╝¢
- µ£¼τ│╗τ╗ƒΣ╕║τºæτáöσÄƒσ₧ï∩╝îΣ╕ìµÅÉΣ╛¢τï¼τ½ïΣ╕┤σ║èΦ»èµû¡τ╗ôΦ«║πÇé
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import imageio.v3 as iio
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_schema import label_items, load_label_schema  # noqa: E402
from src.preprocessing.dicom_pipeline import (  # noqa: E402
    choose_ct_series,
    discover_dicom_series,
    inspect_series,
    read_dicom_series_with_sitk,
)
from src.reconstruction.export_mesh import export_nifti_mask_mesh  # noqa: E402
from src.reconstruction.measurement import angle_degrees, distance_mm  # noqa: E402
from src.reconstruction.sdf_surface import export_nifti_sdf_surface  # noqa: E402
from src.sitk_compat import sitk_io_path  # noqa: E402

FRONTEND_DIR = PROJECT_ROOT / "web" / "frontend"
RUNTIME_DIR = PROJECT_ROOT / "web" / "runtime"
CASES_DIR = RUNTIME_DIR / "cases"
MODEL_DIR = PROJECT_ROOT / "models"
RESEARCH_PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed_ctspine1k_real"
LARGE_SCALE_PROCESSED_ROOT = Path("H:/CTSpine1K_compact_1p5mm_nibsafe_v6")
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

MAX_FILES_PER_CASE = 4000
MAX_TOTAL_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB∩╝îΘªûτëêµ£¼µ£║ΘÖÉσê╢
CHUNK_SIZE = 1024 * 1024

app = FastAPI(
    title="Θ¬¿τºæ CT µÖ║Φâ╜Φ╛àσè⌐σêåµ₧Éτáöτ⌐╢σ╣│σÅ░",
    version="0.1.0",
    description="τºæτáöσÄƒσ₧ï∩╝ÜDICOM/NIfTI Φ┤¿µÄºπÇüΘóäΦºêπÇüσÉÄτ╗¡σêåσë▓Σ╕ÄΣ╕ëτ╗┤Θçìσ╗║πÇé",
)

FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
CASES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


class ManualQCReviewRequest(BaseModel):
    orientation_ok: bool
    spacing_ok: bool
    label_alignment_ok: bool
    bone_window_ok: bool
    review_status: Literal["pass", "needs_review", "fail"]
    reviewer: str
    notes: str = ""


class PhysicalPoint3D(BaseModel):
    x: float
    y: float
    z: float

    def xyz(self) -> tuple[float, float, float]:
        return float(self.x), float(self.y), float(self.z)


class DistanceMeasurementRequest(BaseModel):
    point_a: PhysicalPoint3D
    point_b: PhysicalPoint3D


class AngleMeasurementRequest(BaseModel):
    point_a: PhysicalPoint3D
    vertex_b: PhysicalPoint3D
    point_c: PhysicalPoint3D


def _case_dir(case_id: str) -> Path:
    if not case_id.startswith("case_") or any(ch in case_id for ch in "/\\.."):
        raise HTTPException(status_code=400, detail="Θ¥₧µ│ò case_id")
    path = CASES_DIR / case_id
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="τùàΣ╛ïΣ╕ìσ¡ÿσ£¿")
    return path


def _research_case_dir(case_id: str) -> Path:
    """σÅ¬σàüΦ«╕Φ«┐Θù«µ£¼Θí╣τ¢«σ╖▓µáçσçåσîûσà¼σ╝Çτáöτ⌐╢τùàΣ╛ï∩╝îΣ╕ìµÄÑσÅùΣ╗╗µäÅµûçΣ╗╢Φ╖»σ╛äπÇé"""
    if not case_id or any(ch in case_id for ch in "/\\") or ".." in case_id:
        raise HTTPException(status_code=400, detail="Θ¥₧µ│ò research case_id")
    for root in (RESEARCH_PROCESSED_ROOT, LARGE_SCALE_PROCESSED_ROOT):
        path = root / case_id
        if root.is_dir() and path.is_dir():
            return path
    raise HTTPException(status_code=404, detail="τáöτ⌐╢τùàΣ╛ïΣ╕ìσ¡ÿσ£¿")


def _evaluation_id(evaluation_dir: Path) -> str:
    return "::".join(evaluation_dir.relative_to(EXPERIMENTS_ROOT).parts)


def _evaluation_dir(evaluation_id: str) -> Path:
    """σÅ¬σàüΦ«╕Φ»╗σÅû experiments µá╣τ¢«σ╜òσåàµ£ÇσñÜΣ╕ñσ▒éτÜäτ£ƒσ«₧Φ»äΣ╝░τ¢«σ╜òπÇé"""
    if not evaluation_id or any(ch in evaluation_id for ch in "/\\") or ".." in evaluation_id:
        raise HTTPException(status_code=400, detail="Θ¥₧µ│ò evaluation_id")
    parts = evaluation_id.split("::")
    if not 1 <= len(parts) <= 2 or any(not part for part in parts):
        raise HTTPException(status_code=400, detail="Θ¥₧µ│ò evaluation_id")
    path = EXPERIMENTS_ROOT.joinpath(*parts)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="Φ»äΣ╝░τ¢«σ╜òΣ╕ìσ¡ÿσ£¿")
    if not (path / "summary.json").exists() or not (path / "metrics_per_case.csv").exists():
        raise HTTPException(status_code=422, detail="τ¢«σ╜òΣ╕ìµÿ»σ«îµò┤ evaluate.py Φ╛ôσç║")
    return path


def _iter_evaluation_dirs() -> list[Path]:
    if not EXPERIMENTS_ROOT.is_dir():
        return []
    found: list[Path] = []
    for top in EXPERIMENTS_ROOT.iterdir():
        if not top.is_dir():
            continue
        if (top / "summary.json").exists() and (top / "metrics_per_case.csv").exists():
            found.append(top)
        for child in top.iterdir():
            if child.is_dir() and (child / "summary.json").exists() and (child / "metrics_per_case.csv").exists():
                found.append(child)
    return found


def _read_metrics_per_case(evaluation_dir: Path) -> list[dict[str, object]]:
    path = evaluation_dir / "metrics_per_case.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    parsed: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {}
        for key, value in row.items():
            text = str(value or "").strip()
            if key == "case_id":
                item[key] = text
                continue
            try:
                item[key] = float(text)
            except ValueError:
                item[key] = text
        case_id = str(item.get("case_id", ""))
        if not case_id:
            continue
        item["prediction_available"] = (
            evaluation_dir / "predictions" / case_id / "prediction.nii.gz"
        ).exists()
        item["uncertainty_available"] = (
            evaluation_dir / "uncertainty" / case_id / "predictive_entropy.nii.gz"
        ).exists()
        parsed.append(item)
    return parsed


def _evaluation_summary(evaluation_dir: Path) -> dict[str, object]:
    try:
        summary = json.loads((evaluation_dir / "summary.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=422, detail=f"summary.json µùáµ│òΦºúµ₧É: {exc}") from exc
    rows = _read_metrics_per_case(evaluation_dir)
    return {
        "evaluation_id": _evaluation_id(evaluation_dir),
        "display_name": evaluation_dir.parent.name + " / " + evaluation_dir.name if evaluation_dir.parent != EXPERIMENTS_ROOT else evaluation_dir.name,
        "split": summary.get("split"),
        "evaluated_at": summary.get("evaluated_at"),
        "device": summary.get("device"),
        "checkpoint": summary.get("checkpoint"),
        "config": summary.get("config"),
        "case_count": len(rows),
        "metrics": summary.get("metrics"),
        "per_class_metrics": summary.get("per_class_metrics"),
        "research_only": True,
    }


def _manual_qc_csv_path() -> Path:
    return RESEARCH_PROCESSED_ROOT / "manual_qc_review.csv"


def _read_manual_qc_rows() -> tuple[list[str], list[dict[str, str]]]:
    path = _manual_qc_csv_path()
    if not path.exists():
        raise HTTPException(status_code=404, detail="manual_qc_review.csv σ░Üµ£¬τöƒµêÉ")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in reader]
    if "case_id" not in fieldnames:
        raise HTTPException(status_code=500, detail="manual_qc_review.csv τ╝║σ░æ case_id")
    return fieldnames, rows


def _bool_from_qc_cell(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "y", "ok", "pass", "µÿ»", "ΘÇÜΦ┐ç"}:
        return True
    if normalized in {"no", "false", "0", "n", "fail", "σÉª", "Σ╕ìΘÇÜΦ┐ç"}:
        return False
    return None


def _parse_label_values_cell(value: str) -> list[int]:
    try:
        payload = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    try:
        return sorted({int(item) for item in payload})
    except (TypeError, ValueError):
        return []


def _public_qc_row(row: dict[str, str]) -> dict[str, object]:
    label_values = _parse_label_values_cell(row.get("auto_label_values", ""))
    readable_labels = label_items(label_values, include_background=False)
    return {
        "case_id": row.get("case_id", ""),
        "auto_has_label": row.get("auto_has_label", ""),
        "auto_label_values": row.get("auto_label_values", ""),
        "auto_label_items": readable_labels,
        "auto_label_display": ", ".join(str(item["display"]) for item in readable_labels),
        "orientation_ok": _bool_from_qc_cell(row.get("orientation_ok", "")),
        "spacing_ok": _bool_from_qc_cell(row.get("spacing_ok", "")),
        "label_alignment_ok": _bool_from_qc_cell(row.get("label_alignment_ok", "")),
        "bone_window_ok": _bool_from_qc_cell(row.get("bone_window_ok", "")),
        "review_status": row.get("review_status", ""),
        "reviewer": row.get("reviewer", ""),
        "notes": row.get("notes", ""),
    }


def _write_manual_qc_rows(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path = _manual_qc_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _mesh_paths(
    case_dir: Path,
    class_id: int | None,
    simplify_mm: float | None = None,
) -> tuple[Path, Path]:
    if class_id is None:
        stem = "mesh_foreground"
    else:
        if class_id <= 0 or class_id > 4096:
            raise HTTPException(status_code=400, detail="class_id σ┐àΘí╗Σ╕║µ¡úµò┤µò░")
        stem = f"mesh_class_{class_id}"
    if simplify_mm is not None:
        value = float(simplify_mm)
        if not np.isfinite(value) or value <= 0 or value > 10:
            raise HTTPException(status_code=400, detail="simplify_mm σ┐àΘí╗Σ╜ìΣ║Ä (0,10] mm")
        token = f"{value:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        stem = f"{stem}_s{token}"
    ply = case_dir / f"{stem}.ply"
    return ply, ply.with_suffix(".json")


def _surface_mesh_paths(
    case_dir: Path,
    class_id: int | None,
    simplify_mm: float | None,
    *,
    surface: str,
    sdf_sigma_mm: float,
) -> tuple[Path, Path]:
    if surface == "mask":
        return _mesh_paths(case_dir, class_id, simplify_mm)
    if surface != "sdf":
        raise HTTPException(status_code=400, detail="surface σ┐àΘí╗Σ╕║ mask µêû sdf")
    if simplify_mm is not None:
        raise HTTPException(status_code=422, detail="SDF surface µÜéΣ╕ìΣ╕Ä vertex-clustering σÉîµù╢σÉ»τö¿")
    sigma = float(sdf_sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0 or sigma > 3:
        raise HTTPException(status_code=400, detail="sdf_sigma_mm σ┐àΘí╗Σ╜ìΣ║Ä (0,3] mm")
    if class_id is None:
        stem = "mesh_foreground"
    else:
        if class_id <= 0 or class_id > 4096:
            raise HTTPException(status_code=400, detail="class_id σ┐àΘí╗Σ╕║µ¡úµò┤µò░")
        stem = f"mesh_class_{class_id}"
    token = f"{sigma:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    ply = case_dir / f"{stem}_sdf{token}.ply"
    return ply, ply.with_suffix(".json")


def _evaluation_case_row(evaluation_dir: Path, case_id: str) -> dict[str, object]:
    for row in _read_metrics_per_case(evaluation_dir):
        if str(row.get("case_id")) == case_id:
            return row
    raise HTTPException(status_code=404, detail="Φ»ÑΦ»äΣ╝░Σ╕¡Σ╕ìσ¡ÿσ£¿µ¡ñ case_id")


def _evaluation_surface_mesh_paths(
    evaluation_dir: Path,
    case_id: str,
    *,
    source: Literal["prediction", "gt"],
    simplify_mm: float | None,
    surface: str,
    sdf_sigma_mm: float,
) -> tuple[Path, Path]:
    evaluation_token = _evaluation_id(evaluation_dir).replace("::", "__")
    output_dir = RUNTIME_DIR / "evaluation_reconstruction" / evaluation_token / case_id / source
    return _surface_mesh_paths(
        output_dir,
        None,
        simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )


def _read_sdf_summary_checked(json_path: Path) -> dict[str, object]:
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="SDF mesh summary σ░Üµ£¬τöƒµêÉ")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"SDF mesh summary µùáµ│òΦºúµ₧É: {exc}") from exc
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("component_count_preserved") is not True:
        before = metrics.get("original_components") if isinstance(metrics, dict) else None
        after = metrics.get("smoothed_components") if isinstance(metrics, dict) else None
        raise HTTPException(
            status_code=422,
            detail=f"SDF Φí¿Θ¥óµ£¬ΘÇÜΦ┐çΦ┐₧ΘÇÜσƒƒΣ┐¥µèñ∩╝îµïÆτ╗¥σèáΦ╜╜: {before} -> {after}",
        )
    return payload


def _mesh_query_string(
    *,
    class_id: int | None,
    simplify_mm: float | None,
    surface: str = "mask",
    sdf_sigma_mm: float = 0.4,
) -> str:
    params: list[str] = []
    if class_id is not None:
        params.append(f"class_id={int(class_id)}")
    if simplify_mm is not None:
        params.append(f"simplify_mm={float(simplify_mm):g}")
    if surface != "mask":
        params.append(f"surface={surface}")
        params.append(f"sdf_sigma_mm={float(sdf_sigma_mm):g}")
    return "" if not params else "?" + "&".join(params)


def _research_source_split_map() -> dict[str, str]:
    path = RESEARCH_PROCESSED_ROOT / "ctspine1k_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, list):
        return {}
    return {
        str(item.get("case_id")): str(item.get("source_split", "unknown"))
        for item in payload
        if isinstance(item, dict) and item.get("case_id")
    }


def _extract_research_plane_zyx(
    array_zyx: np.ndarray,
    *,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
) -> tuple[np.ndarray, int]:
    if array_zyx.ndim != 3 or min(array_zyx.shape) <= 0:
        raise HTTPException(status_code=422, detail="τáöτ⌐╢τùàΣ╛ïΣ╕ìµÿ»µ£ëµòê 3D Σ╜ôµò░µì«")
    position = float(np.clip(position, 0.0, 1.0))
    z, y, x = array_zyx.shape
    if plane == "axial":
        index = int(round(position * (z - 1)))
        out = array_zyx[index, :, :]
    elif plane == "coronal":
        index = int(round(position * (y - 1)))
        out = np.flipud(array_zyx[:, index, :])
    elif plane == "sagittal":
        index = int(round(position * (x - 1)))
        out = np.flipud(array_zyx[:, :, index])
    else:  # pragma: no cover - Literal σ╖▓ΘÖÉσê╢
        raise HTTPException(status_code=400, detail=f"µ£¬τƒÑ plane: {plane}")
    return np.asarray(out), index


def _label_rgb(label_plane: np.ndarray) -> np.ndarray:
    """Σ╕║µò┤µò░ label τöƒµêÉτ¿│σ«ÜπÇüµùáΘ£ÇσñûΘâ¿ colormap τÜä RGBπÇé"""
    labels = np.rint(label_plane).astype(np.int64, copy=False)
    rgb = np.zeros(labels.shape + (3,), dtype=np.float32)
    foreground = labels > 0
    if not np.any(foreground):
        return rgb
    values = labels[foreground]
    rgb[..., 0][foreground] = ((values * 67 + 53) % 255) / 255.0
    rgb[..., 1][foreground] = ((values * 131 + 97) % 255) / 255.0
    rgb[..., 2][foreground] = ((values * 193 + 173) % 255) / 255.0
    return rgb


def _load_bone_display(case_dir: Path) -> tuple[sitk.Image, np.ndarray]:
    bone_path = case_dir / "image_bone_window.nii.gz"
    if bone_path.exists():
        image = sitk.ReadImage(sitk_io_path(bone_path))
        return image, np.clip(sitk.GetArrayFromImage(image).astype(np.float32), 0.0, 1.0)
    compact_path = case_dir / "image_normalized_u16.nii.gz"
    if compact_path.exists():
        image = sitk.ReadImage(sitk_io_path(compact_path))
        normalized = sitk.GetArrayFromImage(image).astype(np.float32) / 65535.0
        hu = -1000.0 + 3000.0 * normalized
        bone = np.clip((hu + 500.0) / 2000.0, 0.0, 1.0)
        return image, bone
    raise HTTPException(status_code=404, detail="τáöτ⌐╢τùàΣ╛ïτ╝║σ░æσÅ»µÿ╛τñ║ CT Σ╜ôµò░µì«")


def _research_mpr_png(
    case_dir: Path,
    *,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
    overlay: bool,
    alpha: float,
) -> tuple[bytes, int]:
    label_path = case_dir / "label.nii.gz"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="τáöτ⌐╢τùàΣ╛ïτ╝║σ░æ label")
    bone_image, bone = _load_bone_display(case_dir)
    label_image = sitk.ReadImage(sitk_io_path(label_path))
    if bone_image.GetSize() != label_image.GetSize():
        raise HTTPException(status_code=422, detail="CT Σ╕Ä label size Σ╕ìΣ╕ÇΦç┤")
    label = sitk.GetArrayFromImage(label_image)
    bone_plane, index = _extract_research_plane_zyx(
        bone,
        plane=plane,
        position=position,
    )
    label_plane, label_index = _extract_research_plane_zyx(
        label,
        plane=plane,
        position=position,
    )
    if label_index != index:
        raise RuntimeError("MPR image/label index Σ╕ìΣ╕ÇΦç┤")

    gray = np.clip(bone_plane, 0.0, 1.0)
    if overlay:
        rgb = np.repeat(gray[..., None], 3, axis=2)
        mask = label_plane > 0
        if np.any(mask):
            colors = _label_rgb(label_plane)
            blend = float(np.clip(alpha, 0.0, 1.0))
            rgb[mask] = (1.0 - blend) * rgb[mask] + blend * colors[mask]
        display = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        display = np.round(gray * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    iio.imwrite(buffer, display, extension=".png")
    return buffer.getvalue(), index


def _same_sitk_geometry(reference: sitk.Image, other: sitk.Image) -> bool:
    return (
        reference.GetSize() == other.GetSize()
        and np.allclose(reference.GetSpacing(), other.GetSpacing(), rtol=0.0, atol=1e-5)
        and np.allclose(reference.GetOrigin(), other.GetOrigin(), rtol=0.0, atol=1e-3)
        and np.allclose(reference.GetDirection(), other.GetDirection(), rtol=0.0, atol=1e-5)
    )


def _evaluation_mpr_png(
    evaluation_dir: Path,
    case_dir: Path,
    case_id: str,
    *,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
    mode: Literal["raw", "prediction", "gt", "error", "uncertainty"],
    alpha: float,
) -> tuple[bytes, int]:
    bone_image, bone = _load_bone_display(case_dir)
    bone_plane, index = _extract_research_plane_zyx(bone, plane=plane, position=position)
    gray = np.clip(bone_plane, 0.0, 1.0)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    blend = float(np.clip(alpha, 0.0, 1.0))

    prediction_plane = None
    if mode in {"prediction", "error"}:
        artifact_path = evaluation_dir / "predictions" / case_id / "prediction.nii.gz"
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Φ»ÑΦ»äΣ╝░τùàΣ╛ïµ▓íµ£ë prediction.nii.gz")
        artifact_image = sitk.ReadImage(sitk_io_path(artifact_path))
        if not _same_sitk_geometry(bone_image, artifact_image):
            raise HTTPException(status_code=422, detail="prediction Σ╕ÄσñäτÉåσÉÄ CT τë⌐τÉåτ⌐║Θù┤Σ╕ìΣ╕ÇΦç┤")
        prediction_plane, pred_index = _extract_research_plane_zyx(
            sitk.GetArrayFromImage(artifact_image), plane=plane, position=position
        )
        if pred_index != index:
            raise RuntimeError("prediction MPR index Σ╕ìΣ╕ÇΦç┤")

    gt_plane = None
    if mode in {"gt", "error"}:
        label_path = case_dir / "label.nii.gz"
        if not label_path.exists():
            raise HTTPException(status_code=404, detail="Φ»ÑτùàΣ╛ïµ▓íµ£ë GT label")
        label_image = sitk.ReadImage(sitk_io_path(label_path))
        if not _same_sitk_geometry(bone_image, label_image):
            raise HTTPException(status_code=422, detail="GT Σ╕ÄσñäτÉåσÉÄ CT τë⌐τÉåτ⌐║Θù┤Σ╕ìΣ╕ÇΦç┤")
        gt_plane, gt_index = _extract_research_plane_zyx(
            sitk.GetArrayFromImage(label_image), plane=plane, position=position
        )
        if gt_index != index:
            raise RuntimeError("GT MPR index Σ╕ìΣ╕ÇΦç┤")

    if mode == "prediction" and prediction_plane is not None:
        mask = prediction_plane > 0
        rgb[mask] = (1.0 - blend) * rgb[mask] + blend * np.array([0.18, 0.62, 0.98], dtype=np.float32)
    elif mode == "gt" and gt_plane is not None:
        mask = gt_plane > 0
        rgb[mask] = (1.0 - blend) * rgb[mask] + blend * np.array([0.18, 0.78, 0.48], dtype=np.float32)
    elif mode == "error" and prediction_plane is not None and gt_plane is not None:
        pred = prediction_plane > 0
        gt = gt_plane > 0
        tp, fp, fn = pred & gt, pred & ~gt, ~pred & gt
        rgb[tp] = (1.0 - blend) * rgb[tp] + blend * np.array([0.18, 0.76, 0.48], dtype=np.float32)
        rgb[fp] = (1.0 - blend) * rgb[fp] + blend * np.array([0.95, 0.38, 0.22], dtype=np.float32)
        rgb[fn] = (1.0 - blend) * rgb[fn] + blend * np.array([0.10, 0.72, 0.92], dtype=np.float32)
    elif mode == "uncertainty":
        artifact_path = evaluation_dir / "uncertainty" / case_id / "predictive_entropy.nii.gz"
        if not artifact_path.exists():
            raise HTTPException(status_code=404, detail="Φ»ÑΦ»äΣ╝░τùàΣ╛ïµ▓íµ£ë predictive_entropy.nii.gz")
        artifact_image = sitk.ReadImage(sitk_io_path(artifact_path))
        if not _same_sitk_geometry(bone_image, artifact_image):
            raise HTTPException(status_code=422, detail="uncertainty Σ╕ÄσñäτÉåσÉÄ CT τë⌐τÉåτ⌐║Θù┤Σ╕ìΣ╕ÇΦç┤")
        uncertainty = sitk.GetArrayFromImage(artifact_image).astype(np.float32)
        unc_plane, unc_index = _extract_research_plane_zyx(uncertainty, plane=plane, position=position)
        if unc_index != index:
            raise RuntimeError("uncertainty MPR index Σ╕ìΣ╕ÇΦç┤")
        score = np.clip(unc_plane, 0.0, 1.0)
        heat = np.stack([score, np.clip(1.0 - np.abs(score - 0.5) * 2.0, 0.0, 1.0), 1.0 - score], axis=-1)
        local_alpha = (blend * score)[..., None]
        rgb = (1.0 - local_alpha) * rgb + local_alpha * heat

    display = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    iio.imwrite(buffer, display, extension=".png")
    return buffer.getvalue(), index


def _research_case_summary(case_dir: Path, source_split: str | None = None) -> dict[str, object]:
    metadata_path = case_dir / "metadata.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
    label_values: list[int] = []
    label_info = metadata.get("label")
    if isinstance(label_info, dict):
        values = label_info.get("label_values_after")
        if isinstance(values, list):
            label_values = [int(value) for value in values]
    foreground_ply, _ = _mesh_paths(case_dir, None)
    schema = load_label_schema()
    return {
        "case_id": case_dir.name,
        "source_split": source_split or "unknown",
        "pipeline_version": metadata.get("pipeline_version"),
        "label_values": label_values,
        "label_items": label_items(label_values, include_background=False),
        "label_schema_id": schema.get("schema_id"),
        "formal_task_locked": bool(schema.get("formal_task_locked", False)),
        "foreground_mesh_ready": foreground_ply.exists(),
        "research_only": True,
    }


def _safe_suffix(filename: str | None) -> str:
    """σÅ¬Σ┐¥τòÖµá╝σ╝ÅσÉÄτ╝Ç∩╝îΣ╕ìΣ┐¥σ¡ÿσÅ»Φâ╜σÉ½ PHI τÜäσÄƒµûçΣ╗╢σÉìπÇé"""
    name = (filename or "").lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    suffix = Path(name).suffix
    if suffix in {".nii", ".dcm", ".dicom"}:
        return suffix
    return ""


def _manifest_path(case_path: Path) -> Path:
    return case_path / "manifest.json"


def _read_manifest(case_path: Path) -> dict:
    path = _manifest_path(case_path)
    if not path.exists():
        raise HTTPException(status_code=500, detail="τùàΣ╛ï manifest τ╝║σñ▒")
    return json.loads(path.read_text(encoding="utf-8"))


def _window_uint8(array: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("window width σ┐àΘí╗ > 0")
    low = center - width / 2.0
    high = center + width / 2.0
    out = np.clip(array.astype(np.float32), low, high)
    out = (out - low) / max(high - low, 1e-6)
    return np.round(out * 255.0).astype(np.uint8)


def _find_nifti(case_path: Path) -> Path | None:
    upload_dir = case_path / "uploads"
    candidates = list(upload_dir.glob("*.nii")) + list(upload_dir.glob("*.nii.gz"))
    if len(candidates) == 1:
        return candidates[0]
    return None


def _inspect_nifti(path: Path) -> dict:
    img = nib.load(str(path))
    shape = list(img.shape)
    zooms = [float(v) for v in img.header.get_zooms()[:3]]
    return {
        "source_type": "nifti",
        "shape": shape,
        "spacing_xyz_mm": zooms,
        "dtype": str(img.get_data_dtype()),
        "affine": np.asarray(img.affine, dtype=float).round(6).tolist(),
    }


def _inspect_dicom(case_path: Path) -> dict:
    upload_dir = case_path / "uploads"
    grouped = discover_dicom_series(upload_dir)
    if not grouped:
        raise HTTPException(status_code=422, detail="µ£¬µúÇµ╡ïσê░σÅ»Φºúµ₧ÉτÜä DICOM series")

    series = []
    for _, files in grouped.items():
        qc = inspect_series(files)
        series.append(
            {
                "file_count": qc.file_count,
                "modality": qc.modality,
                "status": qc.status,
                "warnings": qc.warnings,
                "rows": qc.rows,
                "columns": qc.columns,
                "spacing_xy_mm": qc.spacing_xy_mm,
                "estimated_spacing_z_mm": qc.estimated_spacing_z_mm,
                "series_uid_hash": qc.series_uid_hash,
            }
        )

    selected = None
    try:
        _, _, selected_qc = choose_ct_series(grouped)
        selected = {
            "file_count": selected_qc.file_count,
            "status": selected_qc.status,
            "warnings": selected_qc.warnings,
            "series_uid_hash": selected_qc.series_uid_hash,
        }
    except Exception as exc:
        selected = {"status": "manual_review_required", "reason": str(exc)}

    return {
        "source_type": "dicom",
        "series_count": len(grouped),
        "series": series,
        "selected_series": selected,
    }


def _extract_mpr_slice_xyz(
    volume_xyz: np.ndarray,
    *,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
) -> np.ndarray:
    if volume_xyz.ndim != 3 or min(volume_xyz.shape) <= 0:
        raise HTTPException(status_code=422, detail="µùáµ│òσ╜óµêÉµ£ëµòê 3D CT Σ╜ôµò░µì«")
    position = float(np.clip(position, 0.0, 1.0))
    if plane == "axial":
        index = int(round(position * (volume_xyz.shape[2] - 1)))
        slice_2d = volume_xyz[:, :, index]
    elif plane == "coronal":
        index = int(round(position * (volume_xyz.shape[1] - 1)))
        slice_2d = volume_xyz[:, index, :]
    elif plane == "sagittal":
        index = int(round(position * (volume_xyz.shape[0] - 1)))
        slice_2d = volume_xyz[index, :, :]
    else:  # pragma: no cover - FastAPI Literal σ╖▓µïªµê¬
        raise HTTPException(status_code=400, detail=f"µ£¬τƒÑ MPR plane: {plane}")
    return np.rot90(np.asarray(slice_2d, dtype=np.float32))


def _preview_png_from_volume(
    volume_xyz: np.ndarray,
    *,
    center: float,
    width: float,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
) -> bytes:
    slice_2d = _extract_mpr_slice_xyz(volume_xyz, plane=plane, position=position)
    display = _window_uint8(slice_2d, center, width)
    buffer = io.BytesIO()
    iio.imwrite(buffer, display, extension=".png")
    return buffer.getvalue()


def _preview_from_nifti(
    path: Path,
    center: float,
    width: float,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
) -> bytes:
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asarray(img.dataobj)
    if data.ndim < 3:
        raise HTTPException(status_code=422, detail="NIfTI Σ╕ìµÿ» 3D/4D Σ╜ôµò░µì«")
    if data.ndim > 3:
        data = data[..., 0]
    return _preview_png_from_volume(
        np.asarray(data, dtype=np.float32),
        center=center,
        width=width,
        plane=plane,
        position=position,
    )


def _preview_from_dicom(
    case_path: Path,
    center: float,
    width: float,
    plane: Literal["axial", "coronal", "sagittal"],
    position: float,
) -> bytes:
    grouped = discover_dicom_series(case_path / "uploads")
    _, files, _ = choose_ct_series(grouped)
    image = read_dicom_series_with_sitk(files)
    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    if array_zyx.ndim != 3 or min(array_zyx.shape) <= 0:
        raise HTTPException(status_code=422, detail="DICOM µùáµ│òσ╜óµêÉµ£ëµòê 3D CT Σ╜ôµò░µì«")
    volume_xyz = np.transpose(array_zyx, (2, 1, 0))
    return _preview_png_from_volume(
        volume_xyz,
        center=center,
        width=width,
        plane=plane,
        position=position,
    )


@app.get("/")
def index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="σëìτ½» index.html Σ╕ìσ¡ÿσ£¿")
    return FileResponse(index_path)


@app.get("/qc-review")
def qc_review_page() -> FileResponse:
    page = FRONTEND_DIR / "qc_review.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="QC review Θí╡Θ¥óΣ╕ìσ¡ÿσ£¿")
    return FileResponse(
        page,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/research/qc")
def list_manual_qc() -> dict:
    _, rows = _read_manual_qc_rows()
    public_rows = [_public_qc_row(row) for row in rows]
    reviewed = [row for row in public_rows if str(row["review_status"]).strip()]
    passed = [row for row in reviewed if row["review_status"] == "pass"]
    return {
        "research_only": True,
        "dataset": "CTSpine1K/MSD-T10 engineering subset",
        "total": len(public_rows),
        "reviewed": len(reviewed),
        "passed": len(passed),
        "pending": len(public_rows) - len(reviewed),
        "cases": public_rows,
        "note": "µ£¼µÄÑσÅúσÅ¬Σ┐¥σ¡ÿΣ║║σ╖Ñσ«íµá╕Φ«░σ╜ò∩╝îΣ╕ìΦç¬σè¿σêñµû¡σî╗σ¡ªµ¡úτí«µÇºπÇé",
    }


@app.get("/api/research/qc/{case_id}/image")
def research_qc_image(case_id: str) -> FileResponse:
    case_dir = _research_case_dir(case_id)
    image = case_dir / "qc_contact_sheet.png"
    if not image.exists():
        raise HTTPException(status_code=404, detail="Φ»ÑτùàΣ╛ïτ╝║σ░æ qc_contact_sheet.png")
    return FileResponse(image, media_type="image/png")


@app.get("/api/research/qc/{case_id}/mpr")
def research_qc_mpr(
    case_id: str,
    plane: Literal["axial", "coronal", "sagittal"] = Query("axial"),
    position: float = Query(0.5, ge=0.0, le=1.0),
    overlay: bool = Query(True),
    alpha: float = Query(0.38, ge=0.0, le=1.0),
) -> Response:
    case_dir = _research_case_dir(case_id)
    try:
        png, index = _research_mpr_png(
            case_dir,
            plane=plane,
            position=position,
            overlay=overlay,
            alpha=alpha,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"τáöτ⌐╢ MPR τöƒµêÉσñ▒Φ┤Ñ: {type(exc).__name__}: {exc}",
        ) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-MPR-Plane": plane,
            "X-MPR-Index": str(index),
            "X-Label-Overlay": "true" if overlay else "false",
        },
    )


@app.post("/api/research/qc/{case_id}")
def save_manual_qc(case_id: str, review: ManualQCReviewRequest) -> dict:
    _research_case_dir(case_id)
    reviewer = review.reviewer.strip()
    notes = review.notes.strip()
    if not reviewer:
        raise HTTPException(status_code=422, detail="reviewer Σ╕ìΦâ╜Σ╕║τ⌐║")
    if len(reviewer) > 80:
        raise HTTPException(status_code=422, detail="reviewer Φ┐çΘò┐")
    if len(notes) > 2000:
        raise HTTPException(status_code=422, detail="notes Φ┐çΘò┐")
    if review.review_status == "pass" and not all(
        (
            review.orientation_ok,
            review.spacing_ok,
            review.label_alignment_ok,
            review.bone_window_ok,
        )
    ):
        raise HTTPException(
            status_code=422,
            detail="review_status=pass µù╢σ¢¢Θí╣Σ║║σ╖ÑµúÇµƒÑσ┐àΘí╗σà¿Θâ¿ΘÇÜΦ┐ç",
        )

    fieldnames, rows = _read_manual_qc_rows()
    target = next((row for row in rows if row.get("case_id") == case_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Σ║║σ╖Ñσ«íµá╕ CSV Σ╕¡Σ╕ìσ¡ÿσ£¿Φ»ÑτùàΣ╛ï")

    target.update(
        {
            "orientation_ok": "yes" if review.orientation_ok else "no",
            "spacing_ok": "yes" if review.spacing_ok else "no",
            "label_alignment_ok": "yes" if review.label_alignment_ok else "no",
            "bone_window_ok": "yes" if review.bone_window_ok else "no",
            "review_status": review.review_status,
            "reviewer": reviewer,
            "notes": notes,
        }
    )
    _write_manual_qc_rows(fieldnames, rows)
    return {
        "status": "saved",
        "case": _public_qc_row(target),
        "research_only": True,
    }


@app.get("/results-review")
def results_review_page() -> FileResponse:
    page = FRONTEND_DIR / "results_review.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="results review Θí╡Θ¥óΣ╕ìσ¡ÿσ£¿")
    return FileResponse(page)


@app.get("/teaching")
def teaching_page() -> FileResponse:
    page = FRONTEND_DIR / "teaching.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="teaching Θí╡Θ¥óΣ╕ìσ¡ÿσ£¿")
    return FileResponse(page)


@app.get("/api/research/evaluations")
def list_evaluations() -> dict:
    evaluations: list[dict[str, object]] = []
    for path in _iter_evaluation_dirs():
        try:
            item = _evaluation_summary(path)
            if "test" in str(item.get("split") or "").lower():
                continue
            evaluations.append(item)
        except HTTPException:
            continue
    evaluations.sort(
        key=lambda item: (str(item.get("evaluated_at") or ""), int(item.get("case_count") or 0)),
        reverse=True,
    )
    return {
        "research_only": True,
        "total": len(evaluations),
        "evaluations": evaluations,
        "note": "σÅ¬σêùσç║τúüτ¢ÿΣ╕¡τ£ƒσ«₧ evaluate.py Φ»äΣ╝░Σ║ºτë⌐∩╝îσîàσÉ½σñºσ₧ïσ«₧Θ¬îΣ╕ïτÜä full-volume Φ»äΣ╝░πÇé",
    }


@app.get("/api/research/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    summary = _evaluation_summary(evaluation_dir)
    return {
        **summary,
        "cases": _read_metrics_per_case(evaluation_dir),
        "note": "Φ┐ÖΣ║¢µîçµáçµ¥ÑΦç¬τúüτ¢ÿΣ╕èτÜäΦ»äΣ╝░Σ║ºτë⌐∩╝¢Web Σ╕ìΘçìµû░Φ«íτ«ùµêûΣ┐«µö╣Φ«║µûçτ╗ôµ₧£πÇé",
    }


@app.get("/api/research/evaluations/{evaluation_id}/cases/{case_id}/mpr")
def evaluation_case_mpr(
    evaluation_id: str,
    case_id: str,
    mode: Literal["raw", "prediction", "gt", "error", "uncertainty"] = Query("prediction"),
    plane: Literal["axial", "coronal", "sagittal"] = Query("axial"),
    position: float = Query(0.5, ge=0.0, le=1.0),
    alpha: float = Query(0.45, ge=0.0, le=1.0),
) -> Response:
    evaluation_dir = _evaluation_dir(evaluation_id)
    rows = _read_metrics_per_case(evaluation_dir)
    if not any(str(row.get("case_id")) == case_id for row in rows):
        raise HTTPException(status_code=404, detail="Φ»ÑΦ»äΣ╝░Σ╕¡Σ╕ìσ¡ÿσ£¿µ¡ñ case_id")
    case_dir = _research_case_dir(case_id)
    try:
        png, index = _evaluation_mpr_png(
            evaluation_dir,
            case_dir,
            case_id,
            plane=plane,            position=position,
            mode=mode,
            alpha=alpha,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Φ»äΣ╝░ MPR τöƒµêÉσñ▒Φ┤Ñ: {type(exc).__name__}: {exc}",
        ) from exc
    return Response(
        content=png,
        media_type="image/png",
        headers={
            "X-Evaluation-Id": evaluation_id,
            "X-Case-Id": case_id,
            "X-MPR-Plane": plane,
            "X-MPR-Index": str(index),
            "X-Overlay-Mode": mode,
        },
    )


@app.get("/api/research/evaluations/{evaluation_id}/cases/{case_id}/geometry")
def evaluation_case_geometry(evaluation_id: str, case_id: str) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    _evaluation_case_row(evaluation_dir, case_id)
    case_dir = _research_case_dir(case_id)
    image, array_zyx = _load_bone_display(case_dir)
    size_xyz = list(image.GetSize())
    spacing_xyz = [float(v) for v in image.GetSpacing()]
    origin_xyz = [float(v) for v in image.GetOrigin()]
    direction = [float(v) for v in image.GetDirection()]
    return {
        "case_id": case_id,
        "size_xyz": size_xyz,
        "shape_zyx": list(array_zyx.shape),
        "spacing_xyz_mm": spacing_xyz,
        "origin_xyz_mm": origin_xyz,
        "direction": direction,
        "coordinate_note": "µòÖσ¡ªσ¥Éµáç X/Y/Z σêåσê½σ»╣σ║ö Sagittal/Coronal/Axial σêçτëçΦ╜┤πÇé",
    }


@app.get("/api/research/evaluations/{evaluation_id}/cases/{case_id}/teaching-hit")
def evaluation_case_teaching_hit(
    evaluation_id: str,
    case_id: str,
    x: float = Query(..., ge=0.0, le=1.0),
    y: float = Query(..., ge=0.0, le=1.0),
    z: float = Query(..., ge=0.0, le=1.0),
) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    _evaluation_case_row(evaluation_dir, case_id)
    case_dir = _research_case_dir(case_id)
    label_path = case_dir / "label.nii.gz"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="Φ»ÑτùàΣ╛ïτ╝║σ░æΣ╕ôσ«╢σÅéΦÇâµáçµ│¿")
    label_image = sitk.ReadImage(sitk_io_path(label_path))
    label = sitk.GetArrayFromImage(label_image)
    z_size, y_size, x_size = label.shape
    ix = int(round(float(x) * (x_size - 1)))
    iy = int(round(float(y) * (y_size - 1)))
    iz = int(round(float(z) * (z_size - 1)))
    value = int(label[iz, iy, ix])
    return {
        "hit": value > 0,
        "label_value": value,
        "index_xyz": [ix, iy, iz],
        "size_xyz": [x_size, y_size, z_size],
        "message": "σ╖▓σæ╜Σ╕¡Σ╕ôσ«╢σÅéΦÇâτ╗ôµ₧ä" if value > 0 else "Φ»Ñτé╣µ£¬ΦÉ╜σ£¿Σ╕ôσ«╢σÅéΦÇâτ╗ôµ₧äσåà∩╝îσÅ»µëôσ╝Ç AI Φ╛àσè⌐σÉÄσåìσêñµû¡πÇé",
    }


@app.post("/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh/build")
def build_evaluation_case_mesh(
    evaluation_id: str,
    case_id: str,
    source: Literal["prediction", "gt"] = Query("prediction"),
    simplify_mm: float | None = Query(None, gt=0, le=10),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
    feature_preservation_strength: float = Query(8.0, ge=0, le=100),
) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    _evaluation_case_row(evaluation_dir, case_id)
    case_dir = _research_case_dir(case_id)
    if source == "prediction":
        input_path = evaluation_dir / "predictions" / case_id / "prediction.nii.gz"
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Φ»ÑΦ»äΣ╝░τùàΣ╛ïµ▓íµ£ë prediction.nii.gz")
    else:
        input_path = case_dir / "label.nii.gz"
        if not input_path.exists():
            raise HTTPException(status_code=404, detail="Φ»ÑτùàΣ╛ïτ╝║σ░æ label.nii.gz")

    ply_path, json_path = _evaluation_surface_mesh_paths(
        evaluation_dir,
        case_id,
        source=source,
        simplify_mm=simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    try:
        if surface == "sdf":
            summary = export_nifti_sdf_surface(
                input_path,
                ply_path,
                sigma_mm=sdf_sigma_mm,
                summary_path=json_path,
                require_component_preservation=True,
            )
        else:
            summary = export_nifti_mask_mesh(
                input_path,
                ply_path,
                summary_path=json_path,
                simplify_cluster_mm=simplify_mm,
                feature_preservation_strength=feature_preservation_strength,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"evaluation mesh τöƒµêÉσñ▒Φ┤Ñ: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "status": "built",
        "evaluation_id": evaluation_id,
        "case_id": case_id,
        "source": source,
        "surface": surface,
        "simplify_mm": simplify_mm,
        "sdf_sigma_mm": sdf_sigma_mm if surface == "sdf" else None,
        "feature_preservation_strength": (
            feature_preservation_strength if surface == "mask" and simplify_mm is not None else None
        ),
        "summary": summary,
        "research_only": True,
        "note": "σÅ¬Σ╗Äσ╖▓Σ┐¥σ¡ÿτÜäτ£ƒσ«₧ evaluation prediction µêûσ»╣σ║ö GT label µ₧äσ╗║ 3D∩╝¢Σ╕ìΘçìµû░Φ┐ÉΦíîµ¿íσ₧ïµÄ¿τÉåπÇé",
    }


@app.get("/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh")
def get_evaluation_case_mesh(
    evaluation_id: str,
    case_id: str,
    source: Literal["prediction", "gt"] = Query("prediction"),
    simplify_mm: float | None = Query(None, gt=0, le=10),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
) -> FileResponse:
    evaluation_dir = _evaluation_dir(evaluation_id)
    _evaluation_case_row(evaluation_dir, case_id)
    ply_path, _ = _evaluation_surface_mesh_paths(
        evaluation_dir,
        case_id,
        source=source,
        simplify_mm=simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="evaluation mesh σ░Üµ£¬τöƒµêÉ")
    if surface == "sdf":
        _read_sdf_summary_checked(ply_path.with_suffix(".json"))
    return FileResponse(ply_path, media_type="application/octet-stream", filename=ply_path.name)


@app.get("/api/research/evaluations/{evaluation_id}/cases/{case_id}/mesh/summary")
def get_evaluation_case_mesh_summary(
    evaluation_id: str,
    case_id: str,
    source: Literal["prediction", "gt"] = Query("prediction"),
    simplify_mm: float | None = Query(None, gt=0, le=10),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    _evaluation_case_row(evaluation_dir, case_id)
    _, json_path = _evaluation_surface_mesh_paths(
        evaluation_dir,
        case_id,
        source=source,
        simplify_mm=simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    if surface == "sdf":
        return _read_sdf_summary_checked(json_path)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="evaluation mesh summary σ░Üµ£¬τöƒµêÉ")
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"evaluation mesh summary µùáµ│òΦºúµ₧É: {exc}") from exc


@app.get("/research-3d")
def research_3d_page() -> FileResponse:
    page = FRONTEND_DIR / "research_3d.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="3D research Θí╡Θ¥óΣ╕ìσ¡ÿσ£¿")
    return FileResponse(page)


@app.get("/api/research/cases")
def list_research_cases() -> dict:
    if not RESEARCH_PROCESSED_ROOT.is_dir():
        return {"research_only": True, "cases": [], "total": 0}
    source_splits = _research_source_split_map()
    case_dirs = sorted(
        path
        for path in RESEARCH_PROCESSED_ROOT.iterdir()
        if path.is_dir()
        and (path / "label.nii.gz").exists()
        and "test" not in source_splits.get(path.name, "unknown").lower()
    )
    cases = [
        _research_case_summary(path, source_splits.get(path.name))
        for path in case_dirs
    ]
    return {
        "research_only": True,
        "dataset": "CTSpine1K/MSD-T10 engineering subset",
        "total": len(cases),
        "cases": cases,
        "note": "σ╜ôσëì 3D µ╝öτñ║Σ╜┐τö¿σà¼σ╝Çµò░µì«τ£ƒσÇ╝ label∩╝¢Σ╕ìµÿ»µ¿íσ₧ïΘóäµ╡ïµêûΦ»èµû¡τ╗ôµ₧£πÇé",
    }


@app.post("/api/research/cases/{case_id}/mesh/build")
def build_research_mesh(
    case_id: str,
    class_id: int | None = Query(None, ge=1, description="σÅ»ΘÇëµáçτ¡╛τ▒╗σê½∩╝¢Σ╕║τ⌐║σêÖσ»╝σç║µëÇµ£ë >0 σëìµÖ»"),
    simplify_mm: float | None = Query(
        None,
        gt=0,
        le=10,
        description="σÅ»ΘÇëτë⌐τÉåτ⌐║Θù┤ vertex-clustering τ╜æµá╝σñºσ░Å∩╝îσìòΣ╜ì mm",
    ),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
) -> dict:
    case_dir = _research_case_dir(case_id)
    label_path = case_dir / "label.nii.gz"
    if not label_path.exists():
        raise HTTPException(status_code=404, detail="Φ»ÑτùàΣ╛ïτ╝║σ░æ label.nii.gz")
    ply_path, json_path = _surface_mesh_paths(
        case_dir,
        class_id,
        simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    try:
        if surface == "sdf":
            summary = export_nifti_sdf_surface(
                label_path,
                ply_path,
                class_id=class_id,
                sigma_mm=sdf_sigma_mm,
                summary_path=json_path,
                require_component_preservation=True,
            )
        else:
            summary = export_nifti_mask_mesh(
                label_path,
                ply_path,
                class_id=class_id,
                summary_path=json_path,
                simplify_cluster_mm=simplify_mm,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"mesh τöƒµêÉσñ▒Φ┤Ñ: {type(exc).__name__}: {exc}",
        ) from exc
    return {
        "status": "built",
        "case_id": case_id,
        "class_id": class_id,
        "simplify_mm": simplify_mm,
        "surface": surface,
        "sdf_sigma_mm": sdf_sigma_mm if surface == "sdf" else None,
        "mesh_url": f"/api/research/cases/{case_id}/mesh"
        + _mesh_query_string(
            class_id=class_id,
            simplify_mm=simplify_mm,
            surface=surface,
            sdf_sigma_mm=sdf_sigma_mm,
        ),
        "summary": summary,
        "research_only": True,
    }


@app.get("/api/research/cases/{case_id}/mesh")
def get_research_mesh(
    case_id: str,
    class_id: int | None = Query(None, ge=1),
    simplify_mm: float | None = Query(None, gt=0, le=10),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
) -> FileResponse:
    case_dir = _research_case_dir(case_id)
    ply_path, _ = _surface_mesh_paths(
        case_dir,
        class_id,
        simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    if not ply_path.exists():
        raise HTTPException(status_code=404, detail="mesh σ░Üµ£¬τöƒµêÉ∩╝îΦ»╖σàêΦ░âτö¿ build")
    if surface == "sdf":
        _read_sdf_summary_checked(ply_path.with_suffix(".json"))
    return FileResponse(ply_path, media_type="application/octet-stream", filename=ply_path.name)


@app.get("/api/research/cases/{case_id}/mesh/summary")
def get_research_mesh_summary(
    case_id: str,
    class_id: int | None = Query(None, ge=1),
    simplify_mm: float | None = Query(None, gt=0, le=10),
    surface: Literal["mask", "sdf"] = Query("mask"),
    sdf_sigma_mm: float = Query(0.4, gt=0, le=3),
) -> dict:
    case_dir = _research_case_dir(case_id)
    _, json_path = _surface_mesh_paths(
        case_dir,
        class_id,
        simplify_mm,
        surface=surface,
        sdf_sigma_mm=sdf_sigma_mm,
    )
    if surface == "sdf":
        return _read_sdf_summary_checked(json_path)
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="mesh summary σ░Üµ£¬τöƒµêÉ")
    return json.loads(json_path.read_text(encoding="utf-8"))


@app.post("/api/research/measure/distance")
def measure_distance(request: DistanceMeasurementRequest) -> dict:
    try:
        value = distance_mm(request.point_a.xyz(), request.point_b.xyz())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "distance_mm": value,
        "point_a_xyz_mm": list(request.point_a.xyz()),
        "point_b_xyz_mm": list(request.point_b.xyz()),
        "research_only": True,
        "note": "τ║»σçáΣ╜òτë⌐τÉåΦ╖¥τª╗∩╝îΣ╕ìµÿ»Σ╕┤σ║èΦ»èµû¡τ╗ôΦ«║πÇé",
    }


@app.post("/api/research/measure/angle")
def measure_angle(request: AngleMeasurementRequest) -> dict:
    try:
        value = angle_degrees(
            request.point_a.xyz(),
            request.vertex_b.xyz(),
            request.point_c.xyz(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "angle_degrees": value,
        "point_a_xyz_mm": list(request.point_a.xyz()),
        "vertex_b_xyz_mm": list(request.vertex_b.xyz()),
        "point_c_xyz_mm": list(request.point_c.xyz()),
        "research_only": True,
        "note": "τ║»σçáΣ╜òΣ╕ëτé╣σñ╣ΦºÆ∩╝îΣ╕ìµÿ»Σ╕┤σ║èΦ»èµû¡τ╗ôΦ«║πÇé",
    }


@app.get("/api/health")
def health() -> dict:
    checkpoint_candidates = []
    if MODEL_DIR.exists():
        for pattern in ("*.pt", "*.pth", "*.ckpt", "*.bin"):
            checkpoint_candidates.extend(MODEL_DIR.rglob(pattern))

    return {
        "status": "ok",
        "app_version": app.version,
        "research_only": True,
        "project_root": str(PROJECT_ROOT),
        "runtime_ready": CASES_DIR.exists(),
        "model_checkpoint_count": len(checkpoint_candidates),
        "inference_ready": False,
        "message": "Web µò░µì«Σ╕èΣ╝á/Φ┤¿µÄº/ΘóäΦºêσ╖▓σÅ»τö¿∩╝¢µ¿íσ₧ïµÄ¿τÉåΘ£Çσ«îµêÉ SegFormer3D checkpoint Σ╕ÄµÄ¿τÉåΘÇéΘàìσÉÄσÉ»τö¿πÇé",
    }


@app.post("/api/cases/upload")
async def upload_case(
    files: Annotated[list[UploadFile], File(description="Σ╕ÇΣ╕¬τùàΣ╛ïτÜä DICOM µûçΣ╗╢µêûσìòΣ╕¬ NIfTI")],
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="µ£¬ΘÇëµï⌐µûçΣ╗╢")
    if len(files) > MAX_FILES_PER_CASE:
        raise HTTPException(status_code=413, detail=f"σìòτùàΣ╛ïµûçΣ╗╢µò░Φ╢àΦ┐ç {MAX_FILES_PER_CASE}")

    case_id = f"case_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:10]}"
    case_path = CASES_DIR / case_id
    upload_dir = case_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=False)

    saved_files = []
    total_bytes = 0

    try:
        for index, upload in enumerate(files, start=1):
            suffix = _safe_suffix(upload.filename)
            internal_name = f"file_{index:05d}{suffix}"
            target = upload_dir / internal_name
            file_bytes = 0

            with target.open("wb") as f:
                while True:
                    chunk = await upload.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    file_bytes += len(chunk)
                    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="σìòτùàΣ╛ïµÇ╗Σ╕èΣ╝áΣ╜ôΘçÅΦ╢àΦ┐ç 2 GiB")
                    f.write(chunk)

            saved_files.append(
                {
                    "internal_name": internal_name,
                    "size_bytes": file_bytes,
                    "format_hint": "nifti" if suffix in {".nii", ".nii.gz"} else "unknown_or_dicom",
                }
            )
            await upload.close()

        manifest = {
            "case_id": case_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": len(saved_files),
            "total_bytes": total_bytes,
            "files": saved_files,
            "research_only": True,
        }
        _manifest_path(case_path).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    except Exception:
        shutil.rmtree(case_path, ignore_errors=True)
        raise

    return {
        "status": "uploaded",
        "case_id": case_id,
        "file_count": len(saved_files),
        "total_bytes": total_bytes,
        "next": f"/api/cases/{case_id}/inspect",
    }


@app.get("/api/cases/{case_id}/inspect")
def inspect_case(case_id: str) -> dict:
    case_path = _case_dir(case_id)
    manifest = _read_manifest(case_path)
    nifti_path = _find_nifti(case_path)

    if nifti_path is not None:
        inspection = _inspect_nifti(nifti_path)
    else:
        inspection = _inspect_dicom(case_path)

    return {
        "case_id": case_id,
        "manifest": {
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "created_at_utc": manifest["created_at_utc"],
        },
        "inspection": inspection,
        "research_only": True,
    }


@app.get("/api/cases/{case_id}/preview")
def preview_case(
    case_id: str,
    center: float = Query(500.0, description="µÿ╛τñ║τ¬ùΣ╜ì∩╝îΣ╗àτö¿Σ║ÄΘóäΦºê"),
    width: float = Query(2000.0, gt=0, description="µÿ╛τñ║τ¬ùσ«╜∩╝îΣ╗àτö¿Σ║ÄΘóäΦºê"),
    plane: Literal["axial", "coronal", "sagittal"] = Query(
        "axial", description="MPR σ╣│Θ¥ó"
    ),
    position: float = Query(0.5, ge=0.0, le=1.0, description="µ▓┐Φ»Ñσ╣│Θ¥óτÜäσ╜ÆΣ╕ÇσîûΣ╜ìτ╜«"),
) -> Response:
    case_path = _case_dir(case_id)
    nifti_path = _find_nifti(case_path)

    try:
        if nifti_path is not None:
            png = _preview_from_nifti(nifti_path, center, width, plane, position)
        else:
            png = _preview_from_dicom(case_path, center, width, plane, position)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"ΘóäΦºêτöƒµêÉσñ▒Φ┤Ñ: {type(exc).__name__}: {exc}") from exc

    return Response(content=png, media_type="image/png")


@app.post("/api/cases/{case_id}/infer")
def infer_case(case_id: str) -> JSONResponse:
    _case_dir(case_id)
    return JSONResponse(
        status_code=501,
        content={
            "status": "not_ready",
            "case_id": case_id,
            "message": "τ£ƒσ«₧ SegFormer3D baseline/checkpoint σ░Üµ£¬σ«îµêÉ∩╝îσ╜ôσëìΣ╕ìΣ╝¬ΘÇáσêåσë▓τ╗ôµ₧£πÇéσ╛àµ¿íσ₧ïΦ«¡τ╗âΣ╕ÄµÄ¿τÉåΘÇéΘàìσ«îµêÉσÉÄσÉ»τö¿µ¡ñµÄÑσÅúπÇé",
        },
    )