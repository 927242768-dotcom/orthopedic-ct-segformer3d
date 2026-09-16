from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from src.modeling.dataset import ProcessedOrthopedicCTDataset
from src.modeling.train import select_validation_case_subset
from src.preprocessing.prepare_ctspine1k_compact import (
    PIPELINE_VERSION,
    _load_ct_nibsafe,
    prepare_compact_dataset,
)


def _write_sitk(path: Path, array_zyx: np.ndarray, spacing=(1.5, 1.5, 1.5)) -> None:
    image = sitk.GetImageFromArray(array_zyx)
    image.SetSpacing(tuple(float(v) for v in spacing))
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)


def test_validation_subset_is_fixed_evenly_spaced() -> None:
    cases = [f"case_{index:03d}" for index in range(197)]
    subset = select_validation_case_subset(cases, 12)
    assert len(subset) == 12
    assert subset == select_validation_case_subset(cases, 12)
    assert subset[0] == cases[0]
    assert subset[-1] == cases[-1]
    assert len(set(subset)) == 12
    assert select_validation_case_subset(cases, None) == cases
    assert select_validation_case_subset(cases, 500) == cases


def test_compact_preprocessing_and_u16_dataset_decode(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    output_root = tmp_path / "compact"
    volume = np.linspace(-1000.0, 2000.0, 16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    label = np.zeros((16, 16, 16), dtype=np.uint8)
    label[4:12, 4:12, 4:12] = 20

    _write_sitk(source_root / "raw_data" / "volumes" / "TEST" / "case_001.nii.gz", volume)
    _write_sitk(source_root / "raw_data" / "labels" / "TEST" / "case_001_seg.nii.gz", label)
    metadata_dir = source_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "data_split.txt").write_text("trainset:\ncase_001.nii.gz\n", encoding="utf-8")

    summary = prepare_compact_dataset(
        source_root,
        output_root,
        spacing_xyz=(1.5, 1.5, 1.5),
        limit=1,
        min_free_gb=0.0,
    )
    assert summary["state"] == "completed"
    assert summary["failure_count"] == 0

    split = json.loads((output_root / "split.json").read_text(encoding="utf-8"))
    assert len(split["train"]) == 1
    assert split["validation"] == []
    assert split["test"] == []
    case_id = split["train"][0]
    case_dir = output_root / case_id
    assert (case_dir / "image_normalized_u16.nii.gz").exists()
    assert (case_dir / "label.nii.gz").exists()

    stored = np.asarray(nib.load(str(case_dir / "image_normalized_u16.nii.gz")).dataobj)
    assert stored.dtype == np.uint16
    assert int(stored.min()) == 0
    assert int(stored.max()) == 65535
    # 固定 HU min-max 映射应保留连续灰度，不能再出现旧 0.4.0 中
    # “z-score 后错误 clip 到 [0,1]”导致的大面积 0/65535 饱和。
    assert float(np.mean(stored == 0)) < 0.01
    assert float(np.mean(stored == 65535)) < 0.01
    decoded = stored.astype(np.float32) / 65535.0
    assert 0.49 < float(decoded.mean()) < 0.51

    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    normalization = metadata["processed"]["normalization"]
    assert normalization["method"] == "clip_then_fixed_hu_minmax"
    assert metadata["processed"]["decoded_semantics"] == (
        "0.0=-1000HU, 1.0=2000HU, linear in between"
    )
    assert metadata["pipeline_version"] == PIPELINE_VERSION
    assert metadata["qc"]["status"] == "pass"
    assert metadata["qc"]["raw_reader"] == "nibabel"
    assert metadata["qc"]["cache_crc_verified"] is True

    dataset = ProcessedOrthopedicCTDataset(
        output_root,
        output_root / "split.json",
        "train",
        input_channels=["ct_normalized_u16"],
        roi_size_dhw=(8, 8, 8),
        training=True,
        foreground_probability=1.0,
        patches_per_case=1,
        label_mode="binary",
        seed=42,
    )
    sample = dataset[0]
    image = sample["image"]
    target = sample["label"]
    assert tuple(image.shape) == (1, 8, 8, 8)
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert set(target.unique().tolist()) == {0, 1}


def test_nibsafe_reader_applies_scaling_and_preserves_geometry(tmp_path: Path) -> None:
    raw = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    affine_ras = np.array(
        [
            [-0.7, 0.0, 0.0, 10.0],
            [0.0, 0.8, 0.0, 20.0],
            [0.0, 0.0, 2.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    nifti = nib.Nifti1Image(raw, affine_ras)
    nifti.header["scl_slope"] = 2.0
    nifti.header["scl_inter"] = -1000.0
    path = tmp_path / "scaled_ct.nii.gz"
    nib.save(nifti, str(path))

    expected = nib.load(str(path)).get_fdata(dtype=np.float32)
    image, info = _load_ct_nibsafe(path)
    actual_xyz = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))

    assert np.allclose(actual_xyz, expected, rtol=0.0, atol=1e-5)
    assert np.allclose(image.GetSpacing(), (0.7, 0.8, 2.5), rtol=0.0, atol=1e-6)
    assert np.allclose(image.GetOrigin(), (-10.0, -20.0, 30.0), rtol=0.0, atol=1e-6)
    assert info["reader"] == "SimpleITK.ReadImage(float32; scaling-applied; nibabel-affine-audited)"
    assert np.isclose(info["hu_stats"]["p50"], float(np.percentile(expected, 50.0)))
