"""骨科 CT 科研型 Web 辅助分析原型后端。

当前实现提供：
- 本地健康检查；
- 多文件病例上传；
- DICOM/NIfTI 基础识别与质控摘要；
- 中央轴位骨窗预览；
- 模型推理接口占位（只有真实 checkpoint 与推理适配完成后才启用）。

安全边界：
- 默认仅建议绑定 127.0.0.1；
- 不保存上传文件的原始文件名，降低意外暴露身份信息风险；
- 本系统为科研原型，不提供独立临床诊断结论。
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
MAX_TOTAL_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB，首版本机限制
CHUNK_SIZE = 1024 * 1024

app = FastAPI(
    title="骨科 CT 智能辅助分析研究平台",
    version="0.1.0",
    description="科研原型：DICOM/NIfTI 质控、预览、后续分割与三维重建。",
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
        raise HTTPException(status_code=400, detail="非法 case_id")
    path = CASES_DIR / case_id
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="病例不存在")
    return path


def _research_case_dir(case_id: str) -> Path:
    """只允许访问本项目已标准化公开研究病例，不接受任意文件路径。"""
    if not case_id or any(ch in case_id for ch in "/\\") or ".." in case_id:
        raise HTTPException(status_code=400, detail="非法 research case_id")
    for root in (RESEARCH_PROCESSED_ROOT, LARGE_SCALE_PROCESSED_ROOT):
        path = root / case_id
        if root.is_dir() and path.is_dir():
            return path
    raise HTTPException(status_code=404, detail="研究病例不存在")


def _evaluation_id(evaluation_dir: Path) -> str:
    return "::".join(evaluation_dir.relative_to(EXPERIMENTS_ROOT).parts)


def _evaluation_dir(evaluation_id: str) -> Path:
    """只允许读取 experiments 根目录内最多两层的真实评估目录。"""
    if not evaluation_id or any(ch in evaluation_id for ch in "/\\") or ".." in evaluation_id:
        raise HTTPException(status_code=400, detail="非法 evaluation_id")
    parts = evaluation_id.split("::")
    if not 1 <= len(parts) <= 2 or any(not part for part in parts):
        raise HTTPException(status_code=400, detail="非法 evaluation_id")
    path = EXPERIMENTS_ROOT.joinpath(*parts)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="评估目录不存在")
    if not (path / "summary.json").exists() or not (path / "metrics_per_case.csv").exists():
        raise HTTPException(status_code=422, detail="目录不是完整 evaluate.py 输出")
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
        raise HTTPException(status_code=422, detail=f"summary.json 无法解析: {exc}") from exc
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
        raise HTTPException(status_code=404, detail="manual_qc_review.csv 尚未生成")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
        rows = [{str(k): str(v or "") for k, v in row.items()} for row in reader]
    if "case_id" not in fieldnames:
        raise HTTPException(status_code=500, detail="manual_qc_review.csv 缺少 case_id")
    return fieldnames, rows


def _bool_from_qc_cell(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "y", "ok", "pass", "是", "通过"}:
        return True
    if normalized in {"no", "false", "0", "n", "fail", "否", "不通过"}:
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
            raise HTTPException(status_code=400, detail="class_id 必须为正整数")
        stem = f"mesh_class_{class_id}"
    if simplify_mm is not None:
        value = float(simplify_mm)
        if not np.isfinite(value) or value <= 0 or value > 10:
            raise HTTPException(status_code=400, detail="simplify_mm 必须位于 (0,10] mm")
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
        raise HTTPException(status_code=400, detail="surface 必须为 mask 或 sdf")
    if simplify_mm is not None:
        raise HTTPException(status_code=422, detail="SDF surface 暂不与 vertex-clustering 同时启用")
    sigma = float(sdf_sigma_mm)
    if not np.isfinite(sigma) or sigma <= 0 or sigma > 3:
        raise HTTPException(status_code=400, detail="sdf_sigma_mm 必须位于 (0,3] mm")
    if class_id is None:
        stem = "mesh_foreground"
    else:
        if class_id <= 0 or class_id > 4096:
            raise HTTPException(status_code=400, detail="class_id 必须为正整数")
        stem = f"mesh_class_{class_id}"
    token = f"{sigma:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    ply = case_dir / f"{stem}_sdf{token}.ply"
    return ply, ply.with_suffix(".json")


def _evaluation_case_row(evaluation_dir: Path, case_id: str) -> dict[str, object]:
    for row in _read_metrics_per_case(evaluation_dir):
        if str(row.get("case_id")) == case_id:
            return row
    raise HTTPException(status_code=404, detail="该评估中不存在此 case_id")


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
        raise HTTPException(status_code=404, detail="SDF mesh summary 尚未生成")
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"SDF mesh summary 无法解析: {exc}") from exc
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("component_count_preserved") is not True:
        before = metrics.get("original_components") if isinstance(metrics, dict) else None
        after = metrics.get("smoothed_components") if isinstance(metrics, dict) else None
        raise HTTPException(
            status_code=422,
            detail=f"SDF 表面未通过连通域保护，拒绝加载: {before} -> {after}",
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
        raise HTTPException(status_code=422, detail="研究病例不是有效 3D 体数据")
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
    else:  # pragma: no cover - Literal 已限制
        raise HTTPException(status_code=400, detail=f"未知 plane: {plane}")
    return np.asarray(out), index


def _label_rgb(label_plane: np.ndarray) -> np.ndarray:
    """为整数 label 生成稳定、无需外部 colormap 的 RGB。"""
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
    raise HTTPException(status_code=404, detail="研究病例缺少可显示 CT 体数据")


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
        raise HTTPException(status_code=404, detail="研究病例缺少 label")
    bone_image, bone = _load_bone_display(case_dir)
    label_image = sitk.ReadImage(sitk_io_path(label_path))
    if bone_image.GetSize() != label_image.GetSize():
        raise HTTPException(status_code=422, detail="CT 与 label size 不一致")
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
        raise RuntimeError("MPR image/label index 不一致")

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
            raise HTTPException(status_code=404, detail="该评估病例没有 prediction.nii.gz")
        artifact_image = sitk.ReadImage(sitk_io_path(artifact_path))
        if not _same_sitk_geometry(bone_image, artifact_image):
            raise HTTPException(status_code=422, detail="prediction 与处理后 CT 物理空间不一致")
        prediction_plane, pred_index = _extract_research_plane_zyx(
            sitk.GetArrayFromImage(artifact_image), plane=plane, position=position
        )
        if pred_index != index:
            raise RuntimeError("prediction MPR index 不一致")

    gt_plane = None
    if mode in {"gt", "error"}:
        label_path = case_dir / "label.nii.gz"
        if not label_path.exists():
            raise HTTPException(status_code=404, detail="该病例没有 GT label")
        label_image = sitk.ReadImage(sitk_io_path(label_path))
        if not _same_sitk_geometry(bone_image, label_image):
            raise HTTPException(status_code=422, detail="GT 与处理后 CT 物理空间不一致")
        gt_plane, gt_index = _extract_research_plane_zyx(
            sitk.GetArrayFromImage(label_image), plane=plane, position=position
        )
        if gt_index != index:
            raise RuntimeError("GT MPR index 不一致")

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
            raise HTTPException(status_code=404, detail="该评估病例没有 predictive_entropy.nii.gz")
        artifact_image = sitk.ReadImage(sitk_io_path(artifact_path))
        if not _same_sitk_geometry(bone_image, artifact_image):
            raise HTTPException(status_code=422, detail="uncertainty 与处理后 CT 物理空间不一致")
        uncertainty = sitk.GetArrayFromImage(artifact_image).astype(np.float32)
        unc_plane, unc_index = _extract_research_plane_zyx(uncertainty, plane=plane, position=position)
        if unc_index != index:
            raise RuntimeError("uncertainty MPR index 不一致")
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
    """只保留格式后缀，不保存可能含 PHI 的原文件名。"""
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
        raise HTTPException(status_code=500, detail="病例 manifest 缺失")
    return json.loads(path.read_text(encoding="utf-8"))


def _window_uint8(array: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("window width 必须 > 0")
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
        raise HTTPException(status_code=422, detail="未检测到可解析的 DICOM series")

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
        raise HTTPException(status_code=422, detail="无法形成有效 3D CT 体数据")
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
    else:  # pragma: no cover - FastAPI Literal 已拦截
        raise HTTPException(status_code=400, detail=f"未知 MPR plane: {plane}")
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
        raise HTTPException(status_code=422, detail="NIfTI 不是 3D/4D 体数据")
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
        raise HTTPException(status_code=422, detail="DICOM 无法形成有效 3D CT 体数据")
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
        raise HTTPException(status_code=500, detail="前端 index.html 不存在")
    return FileResponse(index_path)


@app.get("/qc-review")
def qc_review_page() -> FileResponse:
    page = FRONTEND_DIR / "qc_review.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="QC review 页面不存在")
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
        "note": "本接口只保存人工审核记录，不自动判断医学正确性。",
    }


@app.get("/api/research/qc/{case_id}/image")
def research_qc_image(case_id: str) -> FileResponse:
    case_dir = _research_case_dir(case_id)
    image = case_dir / "qc_contact_sheet.png"
    if not image.exists():
        raise HTTPException(status_code=404, detail="该病例缺少 qc_contact_sheet.png")
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
            detail=f"研究 MPR 生成失败: {type(exc).__name__}: {exc}",
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
        raise HTTPException(status_code=422, detail="reviewer 不能为空")
    if len(reviewer) > 80:
        raise HTTPException(status_code=422, detail="reviewer 过长")
    if len(notes) > 2000:
        raise HTTPException(status_code=422, detail="notes 过长")
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
            detail="review_status=pass 时四项人工检查必须全部通过",
        )

    fieldnames, rows = _read_manual_qc_rows()
    target = next((row for row in rows if row.get("case_id") == case_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="人工审核 CSV 中不存在该病例")

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
        raise HTTPException(status_code=500, detail="results review 页面不存在")
    return FileResponse(page)


@app.get("/teaching")
def teaching_page() -> FileResponse:
    page = FRONTEND_DIR / "teaching.html"
    if not page.exists():
        raise HTTPException(status_code=500, detail="teaching 页面不存在")
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
        "note": "只列出磁盘中真实 evaluate.py 评估产物，包含大型实验下的 full-volume 评估。",
    }


@app.get("/api/research/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict:
    evaluation_dir = _evaluation_dir(evaluation_id)
    summary = _evaluation_summary(evaluation_dir)
    return {
        **summary,
        "cases": _read_metrics_per_case(evaluation_dir),
        "note": "这些指标来自磁盘上的评估产物；Web 不重新计算或修改论文结果。",
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
        raise HTTPException(status_code=404, detail="该评估中不存在此 case_id")
    case_dir = _research_case_dir(case_id)
    try:
        png, index = _evaluation_mpr_png(
            evaluation_dir,
            case_dir,
            case_id,
            plane=plane,