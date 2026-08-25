from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.reconstruction.sdf_surface import (
    export_nifti_sdf_surface,
    sdf_surface_from_mask,
    signed_distance_field_mm,
    smooth_sdf_mm,
)


def test_signed_distance_and_sdf_surface_are_physical_and_finite() -> None:
    mask = np.zeros((24, 26, 28), dtype=np.uint8)
    mask[6:18, 7:20, 8:22] = 1
    spacing = (0.8, 1.0, 1.5)

    sdf = signed_distance_field_mm(mask, spacing_xyz_mm=spacing)
    assert sdf.shape == mask.shape
    assert float(sdf[12, 13, 14]) < 0.0
    assert float(sdf[0, 0, 0]) > 0.0

    mesh, metrics = sdf_surface_from_mask(
        mask,
        spacing_xyz_mm=spacing,
        origin_xyz_mm=(10.0, 20.0, -30.0),
        sigma_mm=0.5,
    )
    assert mesh.vertex_count > 0
    assert mesh.face_count > 0
    assert np.isfinite(mesh.vertices_xyz_mm).all()
    assert metrics.component_count_preserved is True
    assert metrics.original_components == 1
    assert metrics.smoothed_components == 1
    assert 0.0 <= metrics.vertex_hd95_mm < 2.0


def test_sdf_smoothing_preserves_two_separated_components_for_small_sigma() -> None:
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[5:11, 5:11, 5:11] = 1
    mask[21:27, 21:27, 21:27] = 1

    _, metrics = sdf_surface_from_mask(
        mask,
        spacing_xyz_mm=(1.0, 1.0, 1.0),
        sigma_mm=0.5,
    )
    assert metrics.original_components == 2
    assert metrics.smoothed_components == 2
    assert metrics.component_count_preserved is True


def test_smooth_sdf_zero_sigma_returns_copy_and_negative_sigma_rejected() -> None:
    sdf = np.linspace(-2, 2, 5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    copied = smooth_sdf_mm(sdf, spacing_xyz_mm=(1, 1, 1), sigma_mm=0.0)
    assert np.array_equal(copied, sdf)
    assert copied is not sdf

    try:
        smooth_sdf_mm(sdf, spacing_xyz_mm=(1, 1, 1), sigma_mm=-0.1)
    except ValueError as exc:
        assert "sigma_mm" in str(exc)
    else:
        raise AssertionError("negative sigma should be rejected")


def test_export_nifti_sdf_surface_rejects_component_change_by_default(tmp_path: Path) -> None:
    label = np.zeros((32, 32, 32), dtype=np.int16)
    label[5:15, 5:15, 5:15] = 1
    label[25, 25, 25] = 1
    image = sitk.GetImageFromArray(label)
    source = tmp_path / "two_components.nii.gz"
    sitk.WriteImage(image, str(source))

    try:
        export_nifti_sdf_surface(
            source,
            tmp_path / "rejected.ply",
            sigma_mm=0.8,
        )
    except ValueError as exc:
        assert "连通域数量" in str(exc)
    else:
        raise AssertionError("component-count change should be rejected by default")

    accepted = export_nifti_sdf_surface(
        source,
        tmp_path / "allowed_for_sweep.ply",
        sigma_mm=0.8,
        require_component_preservation=False,
    )
    assert accepted["metrics"]["component_count_preserved"] is False


def test_export_nifti_sdf_surface_writes_ply_and_summary(tmp_path: Path) -> None:
    label = np.zeros((20, 22, 24), dtype=np.int16)
    label[5:15, 6:17, 7:19] = 24
    image = sitk.GetImageFromArray(label)
    image.SetSpacing((0.8, 1.0, 1.2))
    image.SetOrigin((5.0, 10.0, -20.0))
    source = tmp_path / "label.nii.gz"
    output = tmp_path / "sdf_class24.ply"
    sitk.WriteImage(image, str(source))

    summary = export_nifti_sdf_surface(
        source,
        output,
        class_id=24,
        sigma_mm=0.5,
    )
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert summary["selection"] == "class_24"
    assert summary["metrics"]["component_count_preserved"] is True
    assert summary["metrics"]["sdf_vertex_count"] > 0
