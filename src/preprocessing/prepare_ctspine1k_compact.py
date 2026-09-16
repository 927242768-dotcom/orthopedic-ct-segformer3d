"""为完整 CTSpine1K 生成低占用训练缓存。

目标：在保留原始 H:\CTSpine1K raw NIfTI 的前提下，避免再生成一套体积很大的
1 mm float32 + bone-window 数据。完整数据训练缓存采用：
- 1.5 mm isotropic resampling；
- CT clip 到 [-1000, 2000] HU 后，按固定 HU 上下限线性映射到 [0,1]；
- [0,1] CT 量化为 uint16（训练时恢复 float32/65535）；
- label 采用 nearest-neighbor，保存 uint8；
- 仅保存训练必需的 image/label/metadata，不生成 bone-window、mesh、QC 图片。

官方 source split 映射：
- trainset -> train
- test_public -> validation
- test_private -> test

该缓存属于大规模工程训练输入。正式论文结果仍应在人工 QC / 参数锁定后生成。
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from src.preprocessing.dicom_pipeline import resample_image
from src.preprocessing.nifti_pipeline import compare_image_label_geometry
from src.preprocessing.prepare_ctspine1k import CTSpine1KCase, discover_ctspine1k_cases
from src.sitk_compat import sitk_io_path

DEFAULT_SOURCE_ROOT = Path("H:/CTSpine1K")
DEFAULT_OUTPUT_ROOT = Path("H:/CTSpine1K_compact_1p5mm_nibsafe_v6")
DEFAULT_SPACING = (1.5, 1.5, 1.5)
DEFAULT_HU_CLIP = (-1000.0, 2000.0)
DEFAULT_INCLUDED_SOURCE_SPLITS = ("trainset", "test_public")
PIPELINE_VERSION = "0.6.0-compact-u16-fixed-hu-minmax-1p5mm-nibsafe"
STOP_FILE_NAME = "PREPROCESS_STOP_REQUESTED"


def _json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def _verify_gzip_crc(path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """完整读取 gzip 流，确保文件在落盘前通过 CRC 校验。"""
    with gzip.open(path, "rb") as stream:
        while stream.read(chunk_size):
            pass


def _write_image_atomic(image: sitk.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp.nii.gz")
    try:
        sitk.WriteImage(image, sitk_io_path(temp), True)
        _verify_gzip_crc(temp)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def _percentile_stats(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"期望 3D 体数据，实际 shape={values.shape}")
    if not np.isfinite(values).all():
        finite_ratio = float(np.isfinite(values).mean())
        raise ValueError(f"体数据包含 NaN/Inf，finite_ratio={finite_ratio:.8f}")
    p001, p01, p50, p99, p999 = np.percentile(values, [0.1, 1.0, 50.0, 99.0, 99.9])
    return {
        "min": float(values.min()),
        "p001": float(p001),
        "p01": float(p01),
        "p50": float(p50),
        "p99": float(p99),
        "p999": float(p999),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _nib_geometry_payload(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    affine_ras = np.asarray(image.affine, dtype=np.float64)
    if affine_ras.shape != (4, 4) or not np.isfinite(affine_ras).all():
        raise ValueError("NIfTI affine 非法或包含 NaN/Inf")
    header = image.header
    return {
        "shape_xyz": [int(v) for v in image.shape[:3]],
        "zooms_xyz_mm": [float(v) for v in header.get_zooms()[:3]],
        "affine_ras": affine_ras.tolist(),
        "qform_code": int(header["qform_code"]),
        "sform_code": int(header["sform_code"]),
    }


def _nib_to_sitk(
    image: nib.spatialimages.SpatialImage,
    data_xyz: np.ndarray,
    *,
    pixel_type: int | None = None,
) -> sitk.Image:
    """把 nibabel 的 XYZ voxel + RAS affine 无损映射到 SimpleITK 的 LPS 几何。"""
    if data_xyz.ndim != 3:
        raise ValueError(f"只支持 3D NIfTI，当前 shape={data_xyz.shape}")

    affine_ras = np.asarray(image.affine, dtype=np.float64)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = ras_to_lps @ affine_ras
    basis = affine_lps[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError(f"NIfTI affine 推导 spacing 失败: {spacing.tolist()}")

    direction = basis / spacing[np.newaxis, :]
    gram = direction.T @ direction
    if not np.allclose(gram, np.eye(3), rtol=0.0, atol=2e-4):
        raise ValueError(
            "NIfTI affine 含明显 shear/非正交方向，拒绝静默投影到 SimpleITK direction: "
            f"gram={gram.tolist()}"
        )
    determinant = float(np.linalg.det(direction))
    if not np.isclose(abs(determinant), 1.0, rtol=0.0, atol=2e-4):
        raise ValueError(f"NIfTI direction determinant 异常: {determinant}")

    array_zyx = np.ascontiguousarray(np.transpose(data_xyz, (2, 1, 0)))
    sitk_image = sitk.GetImageFromArray(array_zyx)
    sitk_image.SetSpacing(tuple(float(v) for v in spacing))
    sitk_image.SetOrigin(tuple(float(v) for v in affine_lps[:3, 3]))
    sitk_image.SetDirection(tuple(float(v) for v in direction.reshape(-1)))
    if pixel_type is not None:
        sitk_image = sitk.Cast(sitk_image, pixel_type)
    return sitk_image


def _assert_sitk_matches_nib_geometry(sitk_image: sitk.Image, nib_image: nib.spatialimages.SpatialImage) -> None:
    """校验 SimpleITK 读取后的 LPS 几何与 NIfTI affine 一致。"""
    reference = _nib_to_sitk(nib_image, np.zeros((1, 1, 1), dtype=np.uint8))
    if not np.allclose(sitk_image.GetSpacing(), reference.GetSpacing(), rtol=0.0, atol=2e-4):
        raise ValueError(f"SimpleITK/nibabel spacing 不一致: {sitk_image.GetSpacing()} vs {reference.GetSpacing()}")
    if not np.allclose(sitk_image.GetOrigin(), reference.GetOrigin(), rtol=0.0, atol=2e-3):
        raise ValueError(f"SimpleITK/nibabel origin 不一致: {sitk_image.GetOrigin()} vs {reference.GetOrigin()}")
    if not np.allclose(sitk_image.GetDirection(), reference.GetDirection(), rtol=0.0, atol=2e-4):
        raise ValueError("SimpleITK/nibabel direction 不一致")


def _ct_stats_implausible(stats: dict[str, float]) -> bool:
    p01 = float(stats["p01"])
    p50 = float(stats["p50"])
    p99 = float(stats["p99"])
    return abs(p50) > 10000.0 or max(abs(p01), abs(p99)) > 100000.0


def _read_ct_via_zlib_fallback(path: Path, temp_dir: Path) -> sitk.Image:
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f".__zlib_fallback_{os.getpid()}_{path.stem}.nii"
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        with path.open("rb") as src, temp_path.open("wb") as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(decoder.decompress(chunk))
            dst.write(decoder.flush())
        if not decoder.eof or decoder.unused_data:
            raise RuntimeError("gzip fallback 解压未完整结束或存在额外 member")
        image = sitk.ReadImage(sitk_io_path(temp_path))
        return sitk.Cast(image, sitk.sitkFloat32)
    finally:
        temp_path.unlink(missing_ok=True)


def _load_ct_nibsafe(path: Path, *, temp_dir: Path | None = None) -> tuple[sitk.Image, dict[str, Any]]:
    """稳定读取压缩 NIfTI CT；异常 gzip 读取时使用顺序 zlib fallback。"""
    nib_image = nib.load(str(path))
    if len(nib_image.shape) != 3:
        raise ValueError(f"只支持 3D CT NIfTI，当前 shape={nib_image.shape}")
    reader = "SimpleITK.ReadImage(float32; scaling-applied; nibabel-affine-audited)"
    try:
        sitk_image = sitk.ReadImage(sitk_io_path(path))
        sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
        stats = _percentile_stats(sitk.GetArrayFromImage(sitk_image))
        if _ct_stats_implausible(stats):
            raise RuntimeError(f"SimpleITK CT statistics implausible: {stats}")
    except Exception:
        fallback_dir = path.parent if temp_dir is None else temp_dir
        sitk_image = _read_ct_via_zlib_fallback(path, fallback_dir)
        stats = _percentile_stats(sitk.GetArrayFromImage(sitk_image))
        if _ct_stats_implausible(stats):
            raise RuntimeError(f"zlib fallback 后 CT statistics 仍异常: {stats}")
        reader = "zlib-sequential->SimpleITK(float32; scaling-applied; nibabel-affine-audited)"
    if sitk_image.GetDimension() != 3:
        raise ValueError(f"读取后不是 3D CT: dimension={sitk_image.GetDimension()}")
    _assert_sitk_matches_nib_geometry(sitk_image, nib_image)
    return sitk_image, {"reader": reader, "geometry": _nib_geometry_payload(nib_image), "hu_stats": stats}


def _load_label_nibsafe(path: Path) -> tuple[sitk.Image, dict[str, Any]]:
    nib_image = nib.load(str(path))
    if len(nib_image.shape) != 3:
        raise ValueError(f"只支持 3D label NIfTI，当前 shape={nib_image.shape}")
    sitk_image = sitk.ReadImage(sitk_io_path(path))
    if sitk_image.GetDimension() != 3:
        raise ValueError(f"SimpleITK 读取后不是 3D label: dimension={sitk_image.GetDimension()}")
    _assert_sitk_matches_nib_geometry(sitk_image, nib_image)
    data_float = sitk.GetArrayFromImage(sitk_image).astype(np.float32, copy=False)
    if not np.isfinite(data_float).all():
        raise ValueError("label 包含 NaN/Inf")
    rounded = np.rint(data_float)
    if not np.allclose(data_float, rounded, rtol=0.0, atol=1e-5):
        raise ValueError("label 包含非整数类别值")
    if rounded.min() < 0 or rounded.max() > 255:
        raise ValueError(f"label 超出 uint8 范围: {float(rounded.min())}..{float(rounded.max())}")
    values = sorted(int(v) for v in np.unique(rounded.astype(np.uint8, copy=False)))
    sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
    return sitk_image, {
        "reader": "SimpleITK.ReadImage(uint8; nibabel-affine-audited)",
        "geometry": _nib_geometry_payload(nib_image),
        "label_values": values,
    }


def _decode_quantized_hu(array_u16: np.ndarray, hu_clip: tuple[float, float]) -> np.ndarray:
    hu_min, hu_max = map(float, hu_clip)
    return array_u16.astype(np.float32) / 65535.0 * (hu_max - hu_min) + hu_min


def _validate_written_case(
    case_dir: Path,
    *,
    hu_clip: tuple[float, float],
    expected_shape_xyz: tuple[int, int, int],
) -> dict[str, Any]:
    image_path = case_dir / "image_normalized_u16.nii.gz"
    label_path = case_dir / "label.nii.gz"
    _verify_gzip_crc(image_path)
    _verify_gzip_crc(label_path)

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_shape = tuple(int(v) for v in image.shape[:3])
    label_shape = tuple(int(v) for v in label.shape[:3])
    if image_shape != expected_shape_xyz or label_shape != expected_shape_xyz:
        raise RuntimeError(
            f"cache shape 校验失败: expected={expected_shape_xyz}, image={image_shape}, label={label_shape}"
        )
    if not np.allclose(image.affine, label.affine, rtol=0.0, atol=1e-4):
        raise RuntimeError("cache image/label affine 不一致")

    stored = np.asanyarray(image.dataobj)
    if stored.dtype != np.uint16:
        raise RuntimeError(f"cache CT dtype 不是 uint16: {stored.dtype}")
    decoded_hu = _decode_quantized_hu(stored, hu_clip)
    label_values = sorted(int(v) for v in np.unique(np.asanyarray(label.dataobj)))
    return {
        "gzip_crc_verified": True,
        "shape_xyz": list(image_shape),
        "image_label_affine_match": True,
        "decoded_hu_stats": _percentile_stats(decoded_hu),
        "label_values": label_values,
    }


def _case_ready(case_dir: Path) -> bool:
    required = ("image_normalized_u16.nii.gz", "label.nii.gz", "metadata.json")
    if not all((case_dir / name).exists() for name in required):
        return False
    try:
        metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    qc = metadata.get("qc", {})
    return bool(
        metadata.get("pipeline_version") == PIPELINE_VERSION
        and qc.get("status") == "pass"
        and qc.get("cache_crc_verified") is True
    )


def _split_payload(cases: list[CTSpine1KCase]) -> dict[str, Any]:
    mapping = {
        "trainset": "train",
        "test_public": "validation",
        "test_private": "test",
    }
    payload: dict[str, Any] = {
        "_meta": {
            "purpose": "ctspine1k_full_large_scale_training",
            "formal_experiment": False,
            "dataset": "CTSpine1K/full",
            "task_id": "vertebra_binary_ctspine1k_full_v1",
            "created": datetime.now().isoformat(),
            "patient_level": True,
            "preprocessing_pipeline": PIPELINE_VERSION,
            "source_split_policy": (
                "official trainset->train, test_public->validation; "
                "test_private is excluded from v6 cache until the final model is locked"
            ),
        },
        "train": [],
        "validation": [],
        "test": [],
    }
    unknown: list[str] = []
    for case in cases:
        target = mapping.get(case.source_split)
        if target is None:
            unknown.append(f"{case.case_id}:{case.source_split}")
            continue
        payload[target].append(case.case_id)
    if unknown:
        raise RuntimeError(f"发现未知 source split: {unknown[:10]}")
    return payload


def _manifest_payload(cases: list[CTSpine1KCase]) -> list[dict[str, str]]:
    return [
        {
            "case_id": case.case_id,
            "source_name": case.source_name,
            "sub_dataset": case.sub_dataset,
            "source_split": case.source_split,
            "image_path": str(case.image_path),
            "label_path": str(case.label_path),
        }
        for case in cases
    ]


def _process_case(
    case: CTSpine1KCase,
    output_root: Path,
    *,
    spacing_xyz: tuple[float, float, float],
    hu_clip: tuple[float, float],
) -> None:
    case_dir = output_root / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    image, raw_image_info = _load_ct_nibsafe(case.image_path, temp_dir=case_dir)
    label, raw_label_info = _load_label_nibsafe(case.label_path)

    alignment = compare_image_label_geometry(image, label)
    if not bool(alignment["aligned"]):
        raise ValueError(f"image/label 物理空间不一致: {alignment}")

    hu_min, hu_max = map(float, hu_clip)
    resampled = resample_image(
        image,
        spacing_xyz,
        is_label=False,
        default_pixel_value=hu_min,
    )
    hu = sitk.GetArrayFromImage(resampled).astype(np.float32)
    resampled_hu_stats = _percentile_stats(hu)
    raw_p50 = float(raw_image_info["hu_stats"]["p50"])
    resampled_p50 = float(resampled_hu_stats["p50"])
    median_shift_hu = resampled_p50 - raw_p50
    if abs(median_shift_hu) > 300.0:
        raise RuntimeError(
            f"raw->resample HU median 漂移异常: raw={raw_p50:.3f}, "
            f"resampled={resampled_p50:.3f}, shift={median_shift_hu:.3f} HU"
        )
    # 大体积 CT 单例可接近 1 GB float32。这里故意原地 clip/normalize，
    # 避免同时保留 hu/clipped/normalized 三份完整 float32 体数据。
    clipped = np.clip(hu, hu_min, hu_max, out=hu)
    clipped_stats = _percentile_stats(clipped)
    normalization_stats = {
        "method": "clip_then_fixed_hu_minmax",
        "clip_min_hu": hu_min,
        "clip_max_hu": hu_max,
        "clipped_mean_hu": float(clipped.mean()),
        "clipped_std_hu": float(clipped.std()),
        "unit_scale_hu_per_1p0": float(hu_max - hu_min),
    }
    normalized = clipped
    normalized -= hu_min
    normalized /= hu_max - hu_min
    normalized *= 65535.0
    np.rint(normalized, out=normalized)
    quantized = normalized.astype(np.uint16)
    quantized_image = sitk.GetImageFromArray(quantized)
    quantized_image.CopyInformation(resampled)

    resampled_label = sitk.Resample(
        label,
        resampled,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    label_values = sorted(int(v) for v in np.unique(sitk.GetArrayViewFromImage(resampled_label)))
    if not label_values:
        raise ValueError("重采样后的 label 为空")
    if min(label_values) < 0 or max(label_values) > 25:
        raise ValueError(f"label 范围异常: {label_values[:10]} ... {label_values[-10:]}")

    expected_shape_xyz = tuple(int(v) for v in resampled.GetSize())
    processed_geometry = {
        "shape_xyz": [int(x) for x in resampled.GetSize()],
        "spacing_xyz_mm": [float(x) for x in resampled.GetSpacing()],
        "origin_xyz_mm": [float(x) for x in resampled.GetOrigin()],
        "direction": [float(x) for x in resampled.GetDirection()],
    }
    _write_image_atomic(quantized_image, case_dir / "image_normalized_u16.nii.gz")
    _write_image_atomic(resampled_label, case_dir / "label.nii.gz")

    # 回读 gzip/affine/HU QC 前先释放原始与重采样大对象，避免并行构建时峰值内存叠加。
    del image, label, resampled, hu, clipped, normalized, quantized
    del quantized_image, resampled_label

    cache_qc = _validate_written_case(
        case_dir,
        hu_clip=hu_clip,
        expected_shape_xyz=expected_shape_xyz,
    )
    decoded_stats = cache_qc["decoded_hu_stats"]
    decoded_drift = {
        key: float(decoded_stats[key]) - float(clipped_stats[key])
        for key in ("p01", "p50", "p99")
    }
    if max(abs(v) for v in decoded_drift.values()) > 0.1:
        raise RuntimeError(f"quantized cache HU percentile drift: {decoded_drift}")

    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "source_type": "ctspine1k_full_raw_nifti",
        "source_image_name": case.source_name,
        "source_split": case.source_split,
        "sub_dataset": case.sub_dataset,
        "source": {
            "image": raw_image_info,
            "label": raw_label_info,
        },
        "processed": {
            **processed_geometry,
            "hu_clip": [float(hu_clip[0]), float(hu_clip[1])],
            "resampled_hu_stats": resampled_hu_stats,
            "clipped_hu_stats": clipped_stats,
            "raw_to_resampled_median_shift_hu": median_shift_hu,
            "normalization": normalization_stats,
            "storage": "uint16_quantized_fixed_hu_minmax_ct",
            "quantization_scale": 65535,
            "training_decode": "float32(image_normalized_u16) / 65535.0",
            "decoded_semantics": "0.0=-1000HU, 1.0=2000HU, linear in between",
        },
        "label": {
            "geometry_alignment": alignment,
            "label_values_before": raw_label_info["label_values"],
            "label_values_after": label_values,
            "storage": "uint8_nearest_neighbor",
        },
        "qc": {
            "status": "pass",
            "human_review": False,
            "raw_reader": "nibabel",
            "cache_crc_verified": bool(cache_qc["gzip_crc_verified"]),
            "cache_image_label_affine_match": bool(cache_qc["image_label_affine_match"]),
            "cache_decoded_hu_stats": decoded_stats,
            "cache_quantization_percentile_drift_hu": decoded_drift,
        },
    }
    _json_atomic(case_dir / "metadata.json", metadata)


def _init_preprocess_worker(sitk_threads: int) -> None:
    """限制每个并行进程内部的 ITK 线程数，避免 2 个病例各自再开满 16 线程。"""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(max(1, int(sitk_threads)))


def _process_case_worker(
    case: CTSpine1KCase,
    output_root: str,
    spacing_xyz: tuple[float, float, float],
    hu_clip: tuple[float, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    _process_case(
        case,
        Path(output_root),
        spacing_xyz=spacing_xyz,
        hu_clip=hu_clip,
    )
    return {
        "case_id": case.case_id,
        "elapsed_seconds": time.perf_counter() - started,
    }


def prepare_compact_dataset(
    source_root: str | Path = DEFAULT_SOURCE_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    spacing_xyz: tuple[float, float, float] = DEFAULT_SPACING,
    hu_clip: tuple[float, float] = DEFAULT_HU_CLIP,
    limit: int | None = None,
    min_free_gb: float = 12.0,
    metadata_only: bool = False,
    include_source_splits: tuple[str, ...] = DEFAULT_INCLUDED_SOURCE_SPLITS,
    case_ids: tuple[str, ...] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    source_root = Path(source_root)
    output_root = Path(output_root)
    workers = int(workers)
    if workers <= 0:
        raise ValueError("workers 必须 >= 1")
    split_file = source_root / "metadata" / "data_split.txt"
    if not split_file.exists():
        raise FileNotFoundError(split_file)

    all_cases = discover_ctspine1k_cases(source_root, split_file=split_file)
    if len(all_cases) != 1005 and limit is None:
        raise RuntimeError(f"完整 CTSpine1K 应发现 1005 例，当前={len(all_cases)}")

    requested_splits = tuple(str(v) for v in include_source_splits)
    allowed_splits = {"trainset", "test_public", "test_private"}
    invalid_splits = sorted(set(requested_splits) - allowed_splits)
    if invalid_splits:
        raise ValueError(f"未知 source split: {invalid_splits}")
    if "test_private" in requested_splits:
        raise ValueError("v6 预处理阶段禁止读取/生成 test_private；最终模型锁定后再单独处理")

    if len(all_cases) == 1005:
        official_counts = {
            split: sum(case.source_split == split for case in all_cases)
            for split in ("trainset", "test_public", "test_private")
        }
        expected_counts = {"trainset": 610, "test_public": 197, "test_private": 198}
        if official_counts != expected_counts:
            raise RuntimeError(f"官方 split 数量异常: actual={official_counts}, expected={expected_counts}")

    eligible_cases = [case for case in all_cases if case.source_split in requested_splits]
    # canonical split/manifest 必须稳定描述整个允许范围，不能被 --case-id/--limit 的
    # 局部修复任务缩写。这样单例重建不会悄悄污染后续正式训练的数据划分。
    output_root.mkdir(parents=True, exist_ok=True)
    canonical_split_payload = _split_payload(eligible_cases)
    _json_atomic(output_root / "split.json", canonical_split_payload)
    _json_atomic(output_root / "ctspine1k_manifest.json", _manifest_payload(eligible_cases))

    cases = list(eligible_cases)
    if case_ids is not None:
        requested_case_ids = {str(v) for v in case_ids}
        found_case_ids = {case.case_id for case in cases}
        missing = sorted(requested_case_ids - found_case_ids)
        if missing:
            raise ValueError(f"请求病例不在允许的 train/validation 范围内或不存在: {missing[:10]}")
        cases = [case for case in cases if case.case_id in requested_case_ids]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 必须 > 0")
        cases = cases[:limit]
    if not cases:
        raise RuntimeError("过滤后没有可处理病例")

    stop_file = output_root / STOP_FILE_NAME
    stop_file.unlink(missing_ok=True)

    run_split_payload = _split_payload(cases)

    status_path = output_root / "preprocess_status.json"
    started = time.perf_counter()
    processed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    total = len(cases)

    if metadata_only:
        completed = sum(1 for case in cases if _case_ready(output_root / case.case_id))
        payload = {
            "updated_at": datetime.now().isoformat(),
            "state": "ready_to_preprocess" if completed < total else "completed",
            "pipeline_version": PIPELINE_VERSION,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "total_case_count": total,
            "processed_this_run": 0,
            "skipped_existing": completed,
            "completed_case_count": completed,
            "failure_count": 0,
            "failures": [],
            "current_case": None,
            "active_cases": [],
            "workers": workers,
            "progress_percent": 100.0 * completed / max(total, 1),
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "free_gb": shutil.disk_usage(output_root.anchor or str(output_root)).free / 1024**3,
            "target_spacing_xyz_mm": list(spacing_xyz),
            "hu_clip": list(hu_clip),
            "split_counts": {
                "train": len(run_split_payload["train"]),
                "validation": len(run_split_payload["validation"]),
                "test": len(run_split_payload["test"]),
            },
            "note": "canonical train610/validation197 split 已建立；等待生成或继续紧凑训练缓存。",
        }
        _json_atomic(status_path, payload)
        return payload

    def update_status(
        state: str,
        current_case: str | None = None,
        note: str | None = None,
        active_cases: list[str] | None = None,
    ) -> dict[str, Any]:
        completed = processed + skipped
        elapsed = time.perf_counter() - started
        rate = processed / elapsed if processed > 0 and elapsed > 0 else None
        eta = None if rate in (None, 0) else max(0.0, (total - completed) / rate)
        free_gb = shutil.disk_usage(output_root.anchor or str(output_root)).free / 1024**3
        payload = {
            "updated_at": datetime.now().isoformat(),
            "state": state,
            "pipeline_version": PIPELINE_VERSION,
            "source_root": str(source_root),
            "output_root": str(output_root),
            "total_case_count": total,
            "processed_this_run": processed,
            "skipped_existing": skipped,
            "completed_case_count": completed,
            "failure_count": len(failures),
            "failures": failures[-20:],
            "current_case": current_case,
            "active_cases": [] if active_cases is None else list(active_cases),
            "workers": workers,
            "progress_percent": 100.0 * completed / max(total, 1),
            "elapsed_seconds": elapsed,
            "eta_seconds": eta,
            "cases_per_second": rate,
            "free_gb": free_gb,
            "target_spacing_xyz_mm": list(spacing_xyz),
            "hu_clip": list(hu_clip),
            "split_counts": {
                "train": len(run_split_payload["train"]),
                "validation": len(run_split_payload["validation"]),
                "test": len(run_split_payload["test"]),
            },
            "note": note,
        }
        _json_atomic(status_path, payload)
        return payload

    pending_cases: list[CTSpine1KCase] = []
    for case in cases:
        if _case_ready(output_root / case.case_id):
            skipped += 1
        else:
            pending_cases.append(case)

    update_status(
        "preparing",
        note=f"待处理 {len(pending_cases)} 例；已验证并跳过 {skipped} 例。",
    )

    if workers == 1:
        for case in pending_cases:
            if stop_file.exists():
                stop_file.unlink(missing_ok=True)
                final = update_status(
                    "interrupted",
                    case.case_id,
                    "用户请求停止；已完成病例保留，下次继续时自动跳过。",
                )
                print(json.dumps(final, ensure_ascii=False))
                return final

            free_gb = shutil.disk_usage(output_root.anchor or str(output_root)).free / 1024**3
            if free_gb < min_free_gb:
                final = update_status(
                    "stopped_low_disk",
                    case.case_id,
                    f"H 盘剩余空间低于安全阈值 {min_free_gb:.1f} GB，已停止。",
                )
                print(json.dumps(final, ensure_ascii=False))
                return final

            try:
                _process_case(case, output_root, spacing_xyz=spacing_xyz, hu_clip=hu_clip)
                processed += 1
            except Exception as exc:
                failures.append(
                    {
                        "case_id": case.case_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            status = update_status("preparing", case.case_id)
            print(
                f"[PREP] complete={status['completed_case_count']}/{total} | "
                f"failed={status['failure_count']} | free={status['free_gb']:.1f} GB | "
                f"case={case.case_id}",
                flush=True,
            )
    else:
        logical_cpus = max(1, os.cpu_count() or 1)
        sitk_threads = max(1, logical_cpus // (2 * workers))
        active: dict[Future[dict[str, Any]], CTSpine1KCase] = {}
        next_index = 0
        stop_reason: str | None = None
        stop_note: str | None = None

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_preprocess_worker,
            initargs=(sitk_threads,),
        ) as executor:
            while next_index < len(pending_cases) or active:
                while (
                    stop_reason is None
                    and next_index < len(pending_cases)
                    and len(active) < workers
                ):
                    if stop_file.exists():
                        stop_reason = "interrupted"
                        stop_note = "检测到安全停止请求；停止提交新病例，等待在途病例原子写完。"
                        break
                    free_gb = shutil.disk_usage(output_root.anchor or str(output_root)).free / 1024**3
                    if free_gb < min_free_gb:
                        stop_reason = "stopped_low_disk"
                        stop_note = f"H 盘剩余空间低于安全阈值 {min_free_gb:.1f} GB；停止提交新病例。"
                        break
                    case = pending_cases[next_index]
                    next_index += 1
                    try:
                        future = executor.submit(
                            _process_case_worker,
                            case,
                            str(output_root),
                            spacing_xyz,
                            hu_clip,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "case_id": case.case_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        stop_reason = "failed_worker_pool"
                        stop_note = "并行 worker pool 无法继续提交任务。"
                        break
                    active[future] = case

                if not active:
                    break

                done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                for future in done:
                    case = active.pop(future)
                    try:
                        future.result()
                        processed += 1
                    except Exception as exc:
                        failures.append(
                            {
                                "case_id": case.case_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        if type(exc).__name__ == "BrokenProcessPool":
                            stop_reason = "failed_worker_pool"
                            stop_note = "并行 worker 异常退出；保留已完成病例，需检查内存或底层库错误。"

                    active_ids = [item.case_id for item in active.values()]
                    status = update_status(
                        "preparing" if stop_reason is None else stop_reason,
                        case.case_id,
                        stop_note,
                        active_cases=active_ids,
                    )
                    print(
                        f"[PREP] complete={status['completed_case_count']}/{total} | "
                        f"failed={status['failure_count']} | active={len(active_ids)} | "
                        f"free={status['free_gb']:.1f} GB | case={case.case_id}",
                        flush=True,
                    )

        if stop_reason is not None:
            stop_file.unlink(missing_ok=True)
            final = update_status(stop_reason, note=stop_note, active_cases=[])
            _json_atomic(output_root / "preprocess_summary.json", final)
            print(json.dumps(final, ensure_ascii=False))
            return final

    state = "completed" if not failures else "completed_with_failures"
    if failures:
        final_note = "预处理结束，但存在失败病例；修复前不要开始完整训练。"
    elif total == 807 and requested_splits == DEFAULT_INCLUDED_SOURCE_SPLITS and case_ids is None:
        final_note = "train610 + validation197 共807例 nibsafe v6 缓存已就绪；test198 未读取/未生成。"
    else:
        final_note = f"指定范围预处理完成：{total} 例；test_private 未参与。"
    final = update_status(state, note=final_note)
    _json_atomic(output_root / "preprocess_summary.json", final)
    return final


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Prepare compact full CTSpine1K training cache")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--spacing", type=float, nargs=3, default=DEFAULT_SPACING)
    parser.add_argument("--hu-min", type=float, default=DEFAULT_HU_CLIP[0])
    parser.add_argument("--hu-max", type=float, default=DEFAULT_HU_CLIP[1])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-free-gb", type=float, default=12.0)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-source-splits", nargs="+", default=list(DEFAULT_INCLUDED_SOURCE_SPLITS))
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    result = prepare_compact_dataset(
        args.source_root,
        args.output_root,
        spacing_xyz=tuple(float(v) for v in args.spacing),
        hu_clip=(float(args.hu_min), float(args.hu_max)),
        limit=args.limit,
        min_free_gb=float(args.min_free_gb),
        metadata_only=bool(args.metadata_only),
        include_source_splits=tuple(str(v) for v in args.include_source_splits),
        case_ids=None if args.case_id is None else tuple(str(v) for v in args.case_id),
        workers=int(args.workers),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["state"] in {"completed_with_failures", "stopped_low_disk"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
