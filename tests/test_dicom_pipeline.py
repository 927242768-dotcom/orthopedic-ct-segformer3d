from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.preprocessing.dicom_pipeline import (
    apply_window,
    clip_and_normalize_hu,
    sort_dicom_files_by_geometry,
)


def test_apply_window_clips_and_scales() -> None:
    hu = np.asarray([-1000.0, 0.0, 500.0, 1500.0, 3000.0], dtype=np.float32)
    out = apply_window(hu, center=500.0, width=2000.0)
    assert out.dtype == np.float32
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert np.isclose(out[0], 0.0)
    assert np.isclose(out[-1], 1.0)
    assert np.isclose(out[2], 0.5)


def test_clip_and_normalize_hu_is_finite() -> None:
    hu = np.asarray([-5000, -1000, 0, 1000, 5000], dtype=np.float32)
    out = clip_and_normalize_hu(hu, -1000.0, 2000.0)
    assert np.isfinite(out).all()
    assert abs(float(out.mean())) < 1e-5
    assert np.isclose(float(out.std()), 1.0, atol=1e-5)


def _write_dicom_header(path: Path, *, z: float, instance_number: int) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.999.1"
    ds.Modality = "CT"
    ds.Rows = 2
    ds.Columns = 2
    ds.PixelSpacing = [1.0, 1.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.ImagePositionPatient = [0.0, 0.0, z]
    ds.InstanceNumber = instance_number
    ds.save_as(str(path), write_like_original=False)


def test_sort_dicom_files_uses_geometry_not_input_order(tmp_path: Path) -> None:
    high = tmp_path / "slice_a.dcm"
    low = tmp_path / "slice_b.dcm"
    middle = tmp_path / "slice_c.dcm"
    _write_dicom_header(high, z=5.0, instance_number=3)
    _write_dicom_header(low, z=-5.0, instance_number=1)
    _write_dicom_header(middle, z=0.0, instance_number=2)

    ordered = sort_dicom_files_by_geometry([high, low, middle])
    assert ordered == [low, middle, high]
