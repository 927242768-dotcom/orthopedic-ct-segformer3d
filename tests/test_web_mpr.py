import numpy as np

from web.backend.app import _extract_mpr_slice_xyz, _preview_png_from_volume


def test_mpr_extracts_three_planes_at_normalized_position() -> None:
    volume = np.arange(10 * 12 * 14, dtype=np.float32).reshape(10, 12, 14)

    axial = _extract_mpr_slice_xyz(volume, plane="axial", position=0.5)
    coronal = _extract_mpr_slice_xyz(volume, plane="coronal", position=0.5)
    sagittal = _extract_mpr_slice_xyz(volume, plane="sagittal", position=0.5)

    assert axial.shape == (12, 10)
    assert coronal.shape == (14, 10)
    assert sagittal.shape == (14, 12)


def test_mpr_preview_encodes_png() -> None:
    volume = np.linspace(-1000.0, 2000.0, 8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    png = _preview_png_from_volume(
        volume,
        center=500.0,
        width=2000.0,
        plane="axial",
        position=0.5,
    )

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
