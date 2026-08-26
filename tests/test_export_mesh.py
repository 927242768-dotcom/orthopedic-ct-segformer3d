from pathlib import Path

import numpy as np
import SimpleITK as sitk

from src.reconstruction.export_mesh import export_nifti_mask_mesh


def test_export_nifti_mask_mesh_writes_physical_ply_and_summary(tmp_path: Path) -> None:
    label = np.zeros((12, 14, 16), dtype=np.int16)
    label[3:9, 4:11, 5:13] = 7
    image = sitk.GetImageFromArray(label)
    image.SetSpacing((0.8, 1.2, 2.0))
    image.SetOrigin((10.0, -20.0, 30.0))

    source = tmp_path / "label.nii.gz"
    output = tmp_path / "class7.ply"
    sitk.WriteImage(image, str(source))

    summary = export_nifti_mask_mesh(source, output, class_id=7)

    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert summary["selection"] == "class_7"
    assert summary["foreground_voxels"] == int((label == 7).sum())
    assert summary["vertex_count"] > 0
    assert summary["face_count"] > 0
    assert summary["surface_area_mm2"] > 0
    assert np.allclose(summary["spacing_xyz_mm"], [0.8, 1.2, 2.0], atol=1e-6)


def test_export_mesh_can_write_traceable_simplified_mesh(tmp_path: Path) -> None:
    label = np.zeros((24, 26, 28), dtype=np.int16)
    label[4:20, 5:21, 6:22] = 1
    image = sitk.GetImageFromArray(label)
    image.SetSpacing((1.0, 1.0, 1.0))
    source = tmp_path / "label.nii.gz"
    output = tmp_path / "simplified.ply"
    sitk.WriteImage(image, str(source))

    summary = export_nifti_mask_mesh(
        source,
        output,
        simplify_cluster_mm=2.0,
    )

    simplification = summary["simplification"]
    assert simplification is not None
    assert simplification["method"] == "vertex_clustering"
    assert simplification["cluster_size_mm"] == 2.0
    assert simplification["vertex_reduction_fraction"] > 0.0
    assert simplification["face_reduction_fraction"] > 0.0
    assert simplification["vertex_hd95_mm"] >= 0.0
    assert summary["vertex_count"] < simplification["original_vertex_count"]
    assert summary["face_count"] < simplification["original_face_count"]


def test_export_mesh_records_feature_preservation_candidate(tmp_path: Path) -> None:
    label = np.zeros((24, 26, 28), dtype=np.int16)
    label[4:20, 5:21, 6:22] = 1
    image = sitk.GetImageFromArray(label)
    source = tmp_path / "label.nii.gz"
    output = tmp_path / "feature_weighted.ply"
    sitk.WriteImage(image, str(source))

    summary = export_nifti_mask_mesh(
        source,
        output,
        simplify_cluster_mm=2.0,
        feature_preservation_strength=8.0,
    )

    simplification = summary["simplification"]
    assert simplification is not None
    assert simplification["method"] == "vertex_clustering_feature_weighted"
    assert simplification["feature_preservation_strength"] == 8.0
    assert simplification["vertex_reduction_fraction"] > 0.0
    assert simplification["vertex_hd95_mm"] >= 0.0
