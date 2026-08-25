"""为标准化骨科 CT 病例生成可人工审核的三视图 QC 图。

目标不是替代人工判断，而是把 P0 阶段必须检查的内容固定成统一输出：
- axial / coronal / sagittal 三视图；
- 标准化 CT；
- 骨窗；
- label overlay；
- 记录切片索引、spacing、shape、标签值，便于逐例复核。

默认读取 ``process_nifti_case`` 生成的病例目录。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import SimpleITK as sitk

from src.sitk_compat import sitk_io_path

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


_ORIENTATIONS = ("axial", "coronal", "sagittal")


def _read_array(path: Path) -> tuple[sitk.Image, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    image = sitk.ReadImage(sitk_io_path(path))
    if image.GetDimension() != 3:
        raise ValueError(f"只支持 3D NIfTI: {path}，当前维度={image.GetDimension()}")
    return image, sitk.GetArrayFromImage(image)


def _assert_same_geometry(reference: sitk.Image, other: sitk.Image, *, name: str) -> None:
    if tuple(reference.GetSize()) != tuple(other.GetSize()):
        raise ValueError(f"{name} 与 image_normalized size 不一致")
    if not np.allclose(reference.GetSpacing(), other.GetSpacing(), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} 与 image_normalized spacing 不一致")
    if not np.allclose(reference.GetOrigin(), other.GetOrigin(), rtol=0.0, atol=1e-3):
        raise ValueError(f"{name} 与 image_normalized origin 不一致")
    if not np.allclose(reference.GetDirection(), other.GetDirection(), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} 与 image_normalized direction 不一致")


def _review_indices(shape_zyx: tuple[int, int, int], label: np.ndarray | None) -> dict[str, int]:
    z, y, x = shape_zyx
    if label is not None and np.any(label > 0):
        foreground = np.argwhere(label > 0)
        z_idx, y_idx, x_idx = np.rint(np.median(foreground, axis=0)).astype(int)
    else:
        z_idx, y_idx, x_idx = z // 2, y // 2, x // 2

    return {
        "axial": int(np.clip(z_idx, 0, z - 1)),
        "coronal": int(np.clip(y_idx, 0, y - 1)),
        "sagittal": int(np.clip(x_idx, 0, x - 1)),
    }


def _plane(array: np.ndarray, orientation: str, index: int) -> np.ndarray:
    if orientation == "axial":
        return array[index, :, :]
    if orientation == "coronal":
        return np.flipud(array[:, index, :])
    if orientation == "sagittal":
        return np.flipud(array[:, :, index])
    raise ValueError(f"未知 orientation: {orientation}")


def _label_values(label: np.ndarray | None) -> list[int]:
    if label is None:
        return []
    return sorted(int(v) for v in np.unique(label))


def generate_case_qc(
    case_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    dpi: int = 140,
) -> dict[str, object]:
    """生成单病例 3×3 QC contact sheet，并返回可写入审核清单的摘要。"""
    case_dir = Path(case_dir)
    normalized_path = case_dir / "image_normalized.nii.gz"
    bone_path = case_dir / "image_bone_window.nii.gz"
    label_path = case_dir / "label.nii.gz"

    normalized_image, normalized = _read_array(normalized_path)
    if normalized.ndim != 3:
        raise ValueError(f"标准化 CT 数组必须为 3D，当前 shape={normalized.shape}")

    if bone_path.exists():
        bone_image, bone = _read_array(bone_path)
        _assert_same_geometry(normalized_image, bone_image, name="image_bone_window")
    else:
        bone = normalized

    label: np.ndarray | None = None
    if label_path.exists():
        label_image, label = _read_array(label_path)
        _assert_same_geometry(normalized_image, label_image, name="label")

    indices = _review_indices(tuple(int(v) for v in normalized.shape), label)
    output = Path(output_path) if output_path is not None else case_dir / "qc_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    column_titles = ("Normalized CT", "Bone window", "Label overlay")
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row, orientation in enumerate(_ORIENTATIONS):
        index = indices[orientation]
        normalized_plane = _plane(normalized, orientation, index)
        bone_plane = _plane(bone, orientation, index)
        label_plane = None if label is None else _plane(label, orientation, index)

        axes[row, 0].imshow(normalized_plane, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 1].imshow(bone_plane, cmap="gray", vmin=0.0, vmax=1.0)
        axes[row, 2].imshow(bone_plane, cmap="gray", vmin=0.0, vmax=1.0)
        if label_plane is not None and np.any(label_plane > 0):
            masked = np.ma.masked_where(label_plane <= 0, label_plane)
            axes[row, 2].imshow(masked, cmap="turbo", alpha=0.38, interpolation="nearest")
            axes[row, 2].contour(label_plane > 0, levels=[0.5], linewidths=0.7)

        axes[row, 0].set_ylabel(f"{orientation}\nindex={index}", fontsize=10)
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    spacing = tuple(float(v) for v in normalized_image.GetSpacing())
    size = tuple(int(v) for v in normalized_image.GetSize())
    values = _label_values(label)
    case_id = case_dir.name
    fig.suptitle(
        f"{case_id} | size xyz={size} | spacing xyz={tuple(round(v, 3) for v in spacing)} mm | "
        f"labels={values}",
        fontsize=11,
    )
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "qc_image": str(output),
        "shape_zyx": [int(v) for v in normalized.shape],
        "spacing_xyz_mm": [float(v) for v in spacing],
        "review_indices": indices,
        "label_values": values,
        "has_bone_window": bone_path.exists(),
        "has_label": label_path.exists(),
    }


def _discover_case_dirs(root: Path) -> Iterable[Path]:
    if (root / "image_normalized.nii.gz").exists():
        yield root
        return
    for candidate in sorted(root.iterdir()):
        if candidate.is_dir() and (candidate / "image_normalized.nii.gz").exists():
            yield candidate


def generate_qc_batch(
    root: str | Path,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """为处理后病例批量生成 QC 图与人工审核 CSV 模板。"""
    root = Path(root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(root)
    if limit is not None and limit <= 0:
        raise ValueError("limit 必须 > 0")

    cases = list(_discover_case_dirs(root))
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise RuntimeError(f"没有发现标准化病例目录: {root}")

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for case_dir in cases:
        output = case_dir / "qc_contact_sheet.png"
        try:
            metadata = generate_case_qc(case_dir, output_path=output)
            rows.append(metadata)
        except Exception as exc:
            failures.append(
                {
                    "case_id": case_dir.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    review_csv = root / "manual_qc_review.csv"
    with review_csv.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "case_id",
            "qc_image",
            "auto_has_label",
            "auto_label_values",
            "orientation_ok",
            "spacing_ok",
            "label_alignment_ok",
            "bone_window_ok",
            "review_status",
            "reviewer",
            "notes",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "qc_image": row["qc_image"],
                    "auto_has_label": row["has_label"],
                    "auto_label_values": json.dumps(row["label_values"], ensure_ascii=False),
                    "orientation_ok": "",
                    "spacing_ok": "",
                    "label_alignment_ok": "",
                    "bone_window_ok": "",
                    "review_status": "",
                    "reviewer": "",
                    "notes": "",
                }
            )

    summary = {
        "root": str(root),
        "case_count": len(cases),
        "generated_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "manual_review_csv": str(review_csv),
    }
    (root / "qc_visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成骨科 CT 三视图 / 骨窗 / 标签叠加 QC 图")
    parser.add_argument("root", type=Path, help="单病例目录或包含多个标准化病例的根目录")
    parser.add_argument("--limit", type=int, default=None, help="首轮人工 QC 建议先生成 10 例")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_arg_parser().parse_args()
    summary = generate_qc_batch(args.root, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
