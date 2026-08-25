"""公开骨科 CT NIfTI 病例标准化处理。

用于 VerSe、CTSpine1K、TotalSegmentator 等已经整理为 NIfTI 的公开数据。
核心原则：
- 图像按物理空间重采样；
- 标签只用 nearest-neighbor，并严格检查原始 image/label 几何一致性；
- 输出格式与训练 Dataset 完全一致；
- 记录可审计 metadata/qc，不把未验证数据静默带入训练。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import SimpleITK as sitk

from src.preprocessing.dicom_pipeline import (
    apply_window,
    clip_and_normalize_hu_with_stats,
    image_stats,
    resample_image,
)
from src.sitk_compat import sitk_io_path


def _geometry_payload(image: sitk.Image) -> dict[str, object]:
    return {
        "shape_xyz": [int(x) for x in image.GetSize()],
        "spacing_xyz_mm": [float(x) for x in image.GetSpacing()],
        "origin_xyz_mm": [float(x) for x in image.GetOrigin()],
        "direction": [float(x) for x in image.GetDirection()],
    }


def _allclose(a: Sequence[float], b: Sequence[float], atol: float) -> bool:
    return len(a) == len(b) and bool(np.allclose(a, b, rtol=0.0, atol=atol))


def compare_image_label_geometry(image: sitk.Image, label: sitk.Image) -> dict[str, object]:
    """检查图像与标签是否处于同一物理空间。

    公开分割数据的 label 应与 image 在 size/spacing/origin/direction 上对齐。
    这里不自动“猜测”错位标签应如何修正；若失败，由调用方阻止进入训练。
    """
    size_match = tuple(image.GetSize()) == tuple(label.GetSize())
    spacing_match = _allclose(image.GetSpacing(), label.GetSpacing(), atol=1e-5)
    origin_match = _allclose(image.GetOrigin(), label.GetOrigin(), atol=1e-3)
    direction_match = _allclose(image.GetDirection(), label.GetDirection(), atol=1e-5)
    aligned = size_match and spacing_match and origin_match and direction_match
    return {
        "aligned": aligned,
        "size_match": size_match,
        "spacing_match": spacing_match,
        "origin_match": origin_match,
        "direction_match": direction_match,
    }


def _resample_label_to_reference(label: sitk.Image, reference: sitk.Image) -> sitk.Image:
    return sitk.Resample(
        label,
        reference,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        label.GetPixelID(),
    )


def _validate_label_values(label_array: np.ndarray) -> tuple[list[int], list[str]]:
    warnings: list[str] = []
    if not np.isfinite(label_array).all():
        raise ValueError("标签包含 NaN/Inf")

    rounded = np.rint(label_array)
    if not np.allclose(label_array, rounded, rtol=0.0, atol=1e-5):
        raise ValueError("标签包含非整数类别值，疑似使用了错误插值或标签文件异常")

    unique = sorted(int(v) for v in np.unique(rounded))
    if unique and unique[0] < 0:
        warnings.append("标签包含负类别值，请确认数据集标签定义")
    if unique == [0]:
        warnings.append("标签只有背景 0，当前病例没有前景类别")
    return unique, warnings


def process_nifti_case(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    label_path: str | Path | None = None,
    target_spacing_xyz: Sequence[float] = (1.0, 1.0, 1.0),
    hu_clip: tuple[float, float] = (-1000.0, 2000.0),
    bone_window: tuple[float, float] | None = None,
) -> dict[str, object]:
    """把一个 NIfTI CT（及可选标签）转换为项目标准病例目录。"""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    if label_path is not None and not Path(label_path).exists():
        raise FileNotFoundError(label_path)
    if len(target_spacing_xyz) != 3 or any(float(v) <= 0 for v in target_spacing_xyz):
        raise ValueError("target_spacing_xyz 必须为 3 个正数")
    if not all(math.isfinite(float(v)) for v in target_spacing_xyz):
        raise ValueError("target spacing 必须为有限数")

    output_dir.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(sitk_io_path(image_path))
    if image.GetDimension() != 3:
        raise ValueError(f"只支持 3D CT NIfTI，当前维度={image.GetDimension()}")
    image = sitk.Cast(image, sitk.sitkFloat32)
    original_geometry = _geometry_payload(image)
    original_array = sitk.GetArrayFromImage(image).astype(np.float32)
    original_stats = image_stats(original_array)

    resampled = resample_image(image, target_spacing_xyz, is_label=False)
    hu_resampled = sitk.GetArrayFromImage(resampled).astype(np.float32)
    normalized, normalization_stats = clip_and_normalize_hu_with_stats(
        hu_resampled, *hu_clip
    )

    normalized_image = sitk.GetImageFromArray(normalized.astype(np.float32))
    normalized_image.CopyInformation(resampled)
    sitk.WriteImage(normalized_image, sitk_io_path(output_dir / "image_normalized.nii.gz"), True)

    if bone_window is not None:
        center, width = bone_window
        bone = apply_window(hu_resampled, center, width)
        bone_image = sitk.GetImageFromArray(bone.astype(np.float32))
        bone_image.CopyInformation(resampled)
        sitk.WriteImage(bone_image, sitk_io_path(output_dir / "image_bone_window.nii.gz"), True)

    warnings: list[str] = []
    label_info: dict[str, object] | None = None
    if label_path is not None:
        label = sitk.ReadImage(sitk_io_path(Path(label_path)))
        if label.GetDimension() != 3:
            raise ValueError(f"只支持 3D 标签，当前维度={label.GetDimension()}")

        alignment = compare_image_label_geometry(image, label)
        if not bool(alignment["aligned"]):
            raise ValueError(f"image/label 物理空间不一致: {alignment}")

        label_array = sitk.GetArrayFromImage(label)
        label_values, label_warnings = _validate_label_values(label_array)
        warnings.extend(label_warnings)

        label = sitk.Cast(label, sitk.sitkInt16)
        resampled_label = _resample_label_to_reference(label, resampled)
        out_label_array = sitk.GetArrayFromImage(resampled_label)
        out_values, out_warnings = _validate_label_values(out_label_array)
        warnings.extend(out_warnings)
        if not set(out_values).issubset(set(label_values)):
            raise RuntimeError("nearest-neighbor 重采样后出现原标签不存在的类别值")

        sitk.WriteImage(resampled_label, sitk_io_path(output_dir / "label.nii.gz"), True)
        label_info = {
            "source_path_name": Path(label_path).name,
            "geometry_alignment": alignment,
            "label_values_before": label_values,
            "label_values_after": out_values,
        }

    qc_status = "warning" if warnings else "pass"
    result: dict[str, object] = {
        "pipeline_version": "0.3.0",
        "source_type": "nifti",
        "source_image_name": image_path.name,
        "qc": {
            "status": qc_status,
            "warnings": warnings,
            "image_label_alignment": None if label_info is None else label_info["geometry_alignment"],
        },
        "original": original_geometry,
        "original_intensity_stats": original_stats,
        "processed": {
            "shape_xyz": [int(x) for x in resampled.GetSize()],
            "spacing_xyz_mm": [float(x) for x in resampled.GetSpacing()],
            "hu_clip": [float(hu_clip[0]), float(hu_clip[1])],
            "normalized_stats": image_stats(normalized),
            "normalization": normalization_stats,
            "bone_window": None if bone_window is None else [float(x) for x in bone_window],
        },
        "label": label_info,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "qc.json").write_text(
        json.dumps(result["qc"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="骨科 CT NIfTI 标准化处理")
    parser.add_argument("image", type=Path, help="原始 CT NIfTI (.nii/.nii.gz)")
    parser.add_argument("output_dir", type=Path, help="标准化病例输出目录")
    parser.add_argument("--label", type=Path, default=None, help="可选分割标签 NIfTI")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=(1.0, 1.0, 1.0),
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

    result = process_nifti_case(
        args.image,
        args.output_dir,
        label_path=args.label,
        target_spacing_xyz=args.spacing,
        hu_clip=(args.hu_min, args.hu_max),
        bone_window=bone_window,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
