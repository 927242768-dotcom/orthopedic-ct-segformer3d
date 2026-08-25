"""骨科 CT DICOM/NIfTI 标准化处理核心工具。

当前版本聚焦：
- DICOM series 发现与基础 QC；
- 几何排序信息检查；
- HU 恢复；
- SimpleITK 读取与重采样；
- 强度裁剪/标准化与骨窗派生；
- 元数据/QC JSON 输出。

注意：
1. 本模块用于科研数据处理，不承担临床诊断职责。
2. 临床数据必须在调用本模块前完成合法授权与脱敏。
3. 任何预处理参数变化都应形成新的 preprocessing version。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pydicom
import SimpleITK as sitk

from src.sitk_compat import sitk_io_path, sitk_io_paths


@dataclass
class SeriesQC:
    status: str
    warnings: list[str]
    file_count: int
    study_uid_hash: str | None
    series_uid_hash: str | None
    modality: str | None
    rows: int | None
    columns: int | None
    slice_positions_available: int
    duplicate_position_count: int
    spacing_xy_mm: list[float] | None
    estimated_spacing_z_mm: float | None


def _uid_hash(value: str | None) -> str | None:
    """只保存 UID 哈希，避免在普通日志中暴露原始 UID。"""
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def discover_dicom_series(root: str | Path) -> dict[str, list[Path]]:
    """扫描目录并按 SeriesInstanceUID 分组。

    不依赖扩展名，因此可以识别无 `.dcm` 后缀的 DICOM 文件。
    无法读取为 DICOM header 的文件会被跳过。
    """
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"DICOM 目录不存在: {root}")

    grouped: dict[str, list[Path]] = {}
    for path in _iter_files(root):
        try:
            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=False,
                specific_tags=[
                    "SeriesInstanceUID",
                    "Modality",
                    "SOPInstanceUID",
                ],
            )
        except Exception:
            continue

        series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
        if not series_uid:
            continue
        grouped.setdefault(series_uid, []).append(path)

    return grouped


def _safe_float_list(value: object, expected: int | None = None) -> list[float] | None:
    if value is None:
        return None
    try:
        result = [float(x) for x in value]
    except Exception:
        return None
    if expected is not None and len(result) != expected:
        return None
    return result


def _slice_position(ds: pydicom.dataset.Dataset) -> float | None:
    """优先使用 ImagePositionPatient 与 ImageOrientationPatient 计算几何位置。"""
    ipp = _safe_float_list(getattr(ds, "ImagePositionPatient", None), 3)
    iop = _safe_float_list(getattr(ds, "ImageOrientationPatient", None), 6)
    if ipp is not None and iop is not None:
        row = np.asarray(iop[:3], dtype=np.float64)
        col = np.asarray(iop[3:], dtype=np.float64)
        normal = np.cross(row, col)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal /= norm
            return float(np.dot(np.asarray(ipp, dtype=np.float64), normal))

    slice_location = getattr(ds, "SliceLocation", None)
    if slice_location is not None:
        try:
            return float(slice_location)
        except Exception:
            pass

    return None


def sort_dicom_files_by_geometry(files: Sequence[Path]) -> list[Path]:
    """按 DICOM 几何位置可靠排序切片。

    多层 CT 优先使用 ``ImagePositionPatient`` + ``ImageOrientationPatient``
    计算的物理位置；若几何字段不完整，则仅在所有切片都具有唯一
    ``InstanceNumber`` 时回退。无法得到可靠顺序时直接报错，避免把文件名/rglob
    顺序误当成切片顺序。
    """
    if not files:
        raise ValueError("series 文件列表为空")
    if len(files) == 1:
        return [Path(files[0])]

    records: list[tuple[Path, float | None, int | None]] = []
    for path in files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception as exc:
            raise RuntimeError(f"无法读取 DICOM header 以排序: {path.name}") from exc

        position = _slice_position(ds)
        instance_raw = getattr(ds, "InstanceNumber", None)
        try:
            instance_number = int(instance_raw) if instance_raw is not None else None
        except (TypeError, ValueError):
            instance_number = None
        records.append((Path(path), position, instance_number))

    positions = [record[1] for record in records]
    if all(position is not None for position in positions):
        rounded = [round(float(position), 5) for position in positions]
        if len(set(rounded)) != len(rounded):
            raise RuntimeError("检测到重复/近重复切片几何位置，拒绝自动排序")
        return [
            record[0]
            for record in sorted(records, key=lambda record: float(record[1]))
        ]

    instances = [record[2] for record in records]
    if all(instance is not None for instance in instances):
        values = [int(instance) for instance in instances]
        if len(set(values)) != len(values):
            raise RuntimeError("几何字段不完整且 InstanceNumber 重复，无法可靠排序")
        return [record[0] for record in sorted(records, key=lambda record: int(record[2]))]

    raise RuntimeError(
        "多层 DICOM 缺少完整几何位置，且无法用唯一 InstanceNumber 回退排序"
    )


def inspect_series(files: Sequence[Path]) -> SeriesQC:
    if not files:
        raise ValueError("series 文件列表为空")

    headers: list[pydicom.dataset.Dataset] = []
    warnings: list[str] = []
    unreadable = 0

    for path in files:
        try:
            headers.append(pydicom.dcmread(str(path), stop_before_pixels=True, force=False))
        except Exception as exc:
            unreadable += 1
            warnings.append(f"无法读取 DICOM header: {path.name}: {type(exc).__name__}")

    if not headers:
        return SeriesQC(
            status="fail",
            warnings=warnings or ["没有可读取的 DICOM header"],
            file_count=len(files),
            study_uid_hash=None,
            series_uid_hash=None,
            modality=None,
            rows=None,
            columns=None,
            slice_positions_available=0,
            duplicate_position_count=0,
            spacing_xy_mm=None,
            estimated_spacing_z_mm=None,
        )

    first = headers[0]
    rows_set = {int(getattr(ds, "Rows", -1)) for ds in headers}
    cols_set = {int(getattr(ds, "Columns", -1)) for ds in headers}
    modality_set = {str(getattr(ds, "Modality", "")) for ds in headers}

    if unreadable:
        warnings.append(f"{unreadable} 个文件 header 读取失败")
    if len(rows_set) > 1 or len(cols_set) > 1:
        warnings.append("同一 series 内 Rows/Columns 不一致")
    if len(modality_set) > 1:
        warnings.append("同一 series 内 Modality 不一致")
    if "CT" not in modality_set:
        warnings.append(f"该 series 不是纯 CT: {sorted(modality_set)}")

    positions = [p for ds in headers if (p := _slice_position(ds)) is not None]
    duplicate_position_count = len(positions) - len({round(p, 5) for p in positions})
    if duplicate_position_count > 0:
        warnings.append(f"检测到 {duplicate_position_count} 个重复/近重复切片位置")

    spacing_xy = _safe_float_list(getattr(first, "PixelSpacing", None), 2)
    if spacing_xy is None:
        warnings.append("缺少或无法解析 PixelSpacing")

    estimated_z = None
    if len(positions) >= 2:
        unique_sorted = sorted({round(p, 6) for p in positions})
        diffs = np.diff(unique_sorted)
        diffs = np.abs(diffs[diffs != 0])
        if len(diffs):
            estimated_z = float(np.median(diffs))
            if np.max(diffs) > 1.5 * np.median(diffs):
                warnings.append("切片间距存在明显不均匀，需检查缺层/混合 series")
    else:
        warnings.append("可用几何切片位置不足，无法可靠估计 z-spacing")

    status = "pass"
    if warnings:
        status = "warning"
    if len(rows_set) > 1 or len(cols_set) > 1 or not spacing_xy:
        status = "fail"

    return SeriesQC(
        status=status,
        warnings=warnings,
        file_count=len(files),
        study_uid_hash=_uid_hash(str(getattr(first, "StudyInstanceUID", "")) or None),
        series_uid_hash=_uid_hash(str(getattr(first, "SeriesInstanceUID", "")) or None),
        modality=str(getattr(first, "Modality", "")) or None,
        rows=None if rows_set == {-1} else min(rows_set),
        columns=None if cols_set == {-1} else min(cols_set),
        slice_positions_available=len(positions),
        duplicate_position_count=duplicate_position_count,
        spacing_xy_mm=spacing_xy,
        estimated_spacing_z_mm=estimated_z,
    )


def choose_ct_series(grouped: dict[str, list[Path]]) -> tuple[str, list[Path], SeriesQC]:
    """从目录中选择最可能的主 CT series。

    规则：先保留 Modality=CT，再优先 file_count 最大且 QC 非 fail 的 series。
    若没有合格 series，抛出异常，不静默猜测。
    """
    candidates: list[tuple[str, list[Path], SeriesQC]] = []
    for uid, files in grouped.items():
        qc = inspect_series(files)
        if qc.modality == "CT" and qc.status != "fail":
            candidates.append((uid, files, qc))

    if not candidates:
        raise RuntimeError("未发现可自动选择的合格 CT series，请人工检查输入目录")

    candidates.sort(key=lambda item: item[2].file_count, reverse=True)
    return candidates[0]


def read_dicom_series_with_sitk(files: Sequence[Path]) -> sitk.Image:
    """显式几何排序后使用 GDCM/SimpleITK 读取 series。"""
    sorted_files = sort_dicom_files_by_geometry(files)
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOff()
    reader.SetFileNames(sitk_io_paths(list(sorted_files)))
    return reader.Execute()


def resample_image(
    image: sitk.Image,
    target_spacing_xyz: Sequence[float],
    *,
    is_label: bool = False,
) -> sitk.Image:
    if len(target_spacing_xyz) != 3:
        raise ValueError("target_spacing_xyz 必须包含 3 个值")
    target_spacing = tuple(float(v) for v in target_spacing_xyz)
    if any(v <= 0 for v in target_spacing):
        raise ValueError("target spacing 必须全部 > 0")

    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def clip_and_normalize_hu_with_stats(
    hu_array: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """HU 截断后做逐病例 z-score，并返回可复现的归一化参数。"""
    if not math.isfinite(clip_min) or not math.isfinite(clip_max) or clip_min >= clip_max:
        raise ValueError("HU clip 参数非法")
    clipped = np.clip(hu_array.astype(np.float32), clip_min, clip_max)
    mean = float(clipped.mean())
    std = float(clipped.std())
    if std < 1e-6:
        std = 1.0
    normalized = (clipped - mean) / std
    return normalized, {
        "method": "clip_then_case_zscore",
        "clip_min_hu": float(clip_min),
        "clip_max_hu": float(clip_max),
        "clipped_mean_hu": mean,
        "clipped_std_hu": std,
    }


def clip_and_normalize_hu(
    hu_array: np.ndarray,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    normalized, _ = clip_and_normalize_hu_with_stats(hu_array, clip_min, clip_max)
    return normalized


def apply_window(hu_array: np.ndarray, center: float, width: float) -> np.ndarray:
    if width <= 0:
        raise ValueError("window width 必须 > 0")
    lower = center - width / 2.0
    upper = center + width / 2.0
    x = np.clip(hu_array.astype(np.float32), lower, upper)
    return (x - lower) / (upper - lower)


def image_stats(array: np.ndarray) -> dict[str, float]:
    arr = np.asarray(array, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(np.percentile(arr, 1)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
    }


def process_dicom_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    target_spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    hu_clip: tuple[float, float] = (-1000.0, 2000.0),
    bone_window: tuple[float, float] | None = None,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = discover_dicom_series(input_dir)
    if not grouped:
        raise RuntimeError("输入目录中未发现 DICOM series")

    selected_uid, files, qc = choose_ct_series(grouped)
    if qc.status == "fail":
        raise RuntimeError(f"DICOM QC 失败: {qc.warnings}")

    # SimpleITK 会根据 DICOM 几何信息组织体数据。
    image = read_dicom_series_with_sitk(files)
    image = sitk.Cast(image, sitk.sitkFloat32)

    original_meta = {
        "shape_xyz": list(image.GetSize()),
        "spacing_xyz_mm": [float(x) for x in image.GetSpacing()],
        "origin_xyz_mm": [float(x) for x in image.GetOrigin()],
        "direction": [float(x) for x in image.GetDirection()],
    }

    # GDCM 对常规 CT 通常已应用 Rescale Slope/Intercept；这里以实际读出数组统计进行 QC。
    original_array = sitk.GetArrayFromImage(image)  # z, y, x
    original_stats = image_stats(original_array)

    resampled = resample_image(image, target_spacing_xyz, is_label=False)
    hu_resampled = sitk.GetArrayFromImage(resampled).astype(np.float32)
    normalized, normalization_stats = clip_and_normalize_hu_with_stats(hu_resampled, *hu_clip)

    normalized_image = sitk.GetImageFromArray(normalized)
    normalized_image.CopyInformation(resampled)
    sitk.WriteImage(normalized_image, sitk_io_path(output_dir / "image_normalized.nii.gz"), True)

    if bone_window is not None:
        center, width = bone_window
        bone = apply_window(hu_resampled, center, width)
        bone_image = sitk.GetImageFromArray(bone.astype(np.float32))
        bone_image.CopyInformation(resampled)
        sitk.WriteImage(bone_image, sitk_io_path(output_dir / "image_bone_window.nii.gz"), True)

    result = {
        "pipeline_version": "0.3.0",
        "source_type": "dicom",
        "selected_series_uid_hash": _uid_hash(selected_uid),
        "series_count_detected": len(grouped),
        "qc": asdict(qc),
        "original": original_meta,
        "original_intensity_stats": original_stats,
        "processed": {
            "shape_xyz": list(resampled.GetSize()),
            "spacing_xyz_mm": [float(x) for x in resampled.GetSpacing()],
            "hu_clip": list(hu_clip),
            "normalized_stats": image_stats(normalized),
            "normalization": normalization_stats,
            "bone_window": None if bone_window is None else list(bone_window),
        },
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with (output_dir / "qc.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(qc), f, ensure_ascii=False, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="骨科 CT DICOM 标准化处理")
    parser.add_argument("input_dir", type=Path, help="包含一个或多个 DICOM series 的目录")
    parser.add_argument("output_dir", type=Path, help="处理结果目录")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
        help="目标 spacing，单位 mm，按 x y z 输入",
    )
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=2000.0)
    parser.add_argument("--bone-center", type=float, default=None)
    parser.add_argument("--bone-width", type=float, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bone_window = None
    if args.bone_center is not None or args.bone_width is not None:
        if args.bone_center is None or args.bone_width is None:
            raise SystemExit("--bone-center 与 --bone-width 必须同时提供")
        bone_window = (args.bone_center, args.bone_width)

    result = process_dicom_directory(
        args.input_dir,
        args.output_dir,
        target_spacing_xyz=args.spacing,
        hu_clip=(args.hu_min, args.hu_max),
        bone_window=bone_window,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
