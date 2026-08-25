"""标准化 NIfTI 骨科 CT 训练数据集。

期望每个病例目录：

case_xxx/
├─ image_normalized.nii.gz
├─ image_bone_window.nii.gz   # 可选
├─ label.nii.gz
├─ metadata.json              # 可选
└─ qc.json                    # 可选

split JSON：
{
  "train": ["case_001", ...],
  "validation": [...],
  "test": [...]
}

训练阶段按 3D patch 采样；验证/测试返回完整体数据，由 sliding-window inference
处理。所有 split 必须在患者级别生成。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset


def _load_nifti(path: Path, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    image = nib.load(str(path))
    array = np.asarray(image.dataobj)
    return array.astype(dtype, copy=False)


def _pad_to_shape(
    image: np.ndarray,
    label: np.ndarray,
    target_dhw: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """把 (C,D,H,W)/(D,H,W) 对称 pad 到至少 target 大小。"""
    td, th, tw = [int(v) for v in target_dhw]
    _, d, h, w = image.shape
    pads_spatial = []
    for current, target in ((d, td), (h, th), (w, tw)):
        need = max(0, target - current)
        left = need // 2
        right = need - left
        pads_spatial.append((left, right))

    if any(a or b for a, b in pads_spatial):
        image = np.pad(
            image,
            ((0, 0), pads_spatial[0], pads_spatial[1], pads_spatial[2]),
            mode="constant",
            constant_values=0.0,
        )
        label = np.pad(
            label,
            (pads_spatial[0], pads_spatial[1], pads_spatial[2]),
            mode="constant",
            constant_values=0,
        )
    return image, label


def _random_crop_3d(
    image: np.ndarray,
    label: np.ndarray,
    roi_dhw: Sequence[int],
    *,
    foreground_probability: float,
    rng: random.Random,
    preferred_mask: np.ndarray | None = None,
    preferred_probability: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    image, label = _pad_to_shape(image, label, roi_dhw)
    _, d, h, w = image.shape
    rd, rh, rw = [int(v) for v in roi_dhw]

    use_preferred = (
        preferred_mask is not None
        and preferred_mask.shape == label.shape
        and np.any(preferred_mask)
        and rng.random() < preferred_probability
    )
    use_fg = rng.random() < foreground_probability and np.any(label > 0)
    if use_preferred:
        coords = np.argwhere(preferred_mask)
        center = coords[rng.randrange(len(coords))]
        starts = []
        for c, current, roi in zip(center, (d, h, w), (rd, rh, rw)):
            start = int(c) - roi // 2
            start = max(0, min(start, current - roi))
            starts.append(start)
        sd, sh, sw = starts
    elif use_fg:
        coords = np.argwhere(label > 0)
        center = coords[rng.randrange(len(coords))]
        starts = []
        for c, current, roi in zip(center, (d, h, w), (rd, rh, rw)):
            start = int(c) - roi // 2
            start = max(0, min(start, current - roi))
            starts.append(start)
        sd, sh, sw = starts
    else:
        sd = rng.randint(0, d - rd) if d > rd else 0
        sh = rng.randint(0, h - rh) if h > rh else 0
        sw = rng.randint(0, w - rw) if w > rw else 0

    image = image[:, sd : sd + rd, sh : sh + rh, sw : sw + rw]
    label = label[sd : sd + rd, sh : sh + rh, sw : sw + rw]
    return image, label


def _augment_flips(
    image: np.ndarray,
    label: np.ndarray,
    *,
    rng: random.Random,
    probability: float = 0.5,
    axes: Sequence[int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """对指定 D/H/W 轴做成对翻转。"""
    for spatial_axis in axes:
        if spatial_axis not in {0, 1, 2}:
            raise ValueError("flip axis 只能是 0/1/2（对应 D/H/W）")
        if rng.random() < probability:
            image = np.flip(image, axis=spatial_axis + 1)
            label = np.flip(label, axis=spatial_axis)
    return np.ascontiguousarray(image), np.ascontiguousarray(label)


def _boundary_mask(label: np.ndarray) -> np.ndarray:
    """生成类别边界代理，用于 baseline 前的困难 patch 采样。

    这不是基于模型错误的 hard mining；它只把标签边界作为预训练阶段的困难区域代理。
    """
    foreground = label > 0
    if not np.any(foreground):
        return np.zeros_like(foreground)
    eroded = ndimage.binary_erosion(foreground, structure=np.ones((3, 3, 3), dtype=bool))
    return np.logical_and(foreground, np.logical_not(eroded))


def _center_crop_or_pad(
    image: np.ndarray,
    label: np.ndarray,
    target_dhw: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """把几何增强后的体数据恢复到固定 D/H/W。"""
    td, th, tw = [int(v) for v in target_dhw]
    target = (td, th, tw)
    for axis, target_size in enumerate(target, start=1):
        current = image.shape[axis]
        if current > target_size:
            start = (current - target_size) // 2
            stop = start + target_size
            slicer_img = [slice(None)] * 4
            slicer_img[axis] = slice(start, stop)
            image = image[tuple(slicer_img)]
            slicer_lab = [slice(None)] * 3
            slicer_lab[axis - 1] = slice(start, stop)
            label = label[tuple(slicer_lab)]

    image, label = _pad_to_shape(image, label, target)
    return image, label


def _augment_geometry(
    image: np.ndarray,
    label: np.ndarray,
    *,
    rng: random.Random,
    rotate_deg: float = 0.0,
    scale_range: Sequence[float] = (1.0, 1.0),
    probability: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """3D 小角度旋转与各向同性缩放；image 线性插值，label 最近邻。"""
    original_shape = label.shape
    if rotate_deg > 0 and rng.random() < probability:
        angle = rng.uniform(-float(rotate_deg), float(rotate_deg))
        axis_pair = rng.choice(((0, 1), (0, 2), (1, 2)))
        image_axes = (axis_pair[0] + 1, axis_pair[1] + 1)
        image = ndimage.rotate(
            image,
            angle=angle,
            axes=image_axes,
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        label = ndimage.rotate(
            label,
            angle=angle,
            axes=axis_pair,
            reshape=False,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )

    if len(scale_range) != 2:
        raise ValueError("scale_range 必须包含 [min,max]")
    scale_min, scale_max = float(scale_range[0]), float(scale_range[1])
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("scale_range 必须满足 0 < min <= max")
    if (scale_min != 1.0 or scale_max != 1.0) and rng.random() < probability:
        scale = rng.uniform(scale_min, scale_max)
        image = ndimage.zoom(
            image,
            zoom=(1.0, scale, scale, scale),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        label = ndimage.zoom(
            label,
            zoom=(scale, scale, scale),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        image, label = _center_crop_or_pad(image, label, original_shape)

    return np.ascontiguousarray(image), np.ascontiguousarray(label)


def _augment_intensity(
    image: np.ndarray,
    *,
    rng: random.Random,
    input_channels: Sequence[str],
    gamma_range: Sequence[float] = (1.0, 1.0),
    gaussian_noise_std_range: Sequence[float] = (0.0, 0.0),
    hu_shift_range: Sequence[float] = (0.0, 0.0),
    hu_clip: Sequence[float] = (-1000.0, 2000.0),
    bone_window_width: float = 2000.0,
    ct_zscore_mean_hu: float | None = None,
    ct_zscore_std_hu: float | None = None,
    probability: float = 0.5,
) -> np.ndarray:
    """对 z-score CT 与 [0,1] 骨窗做语义一致的强度扰动。

    ``ct_normalized`` 是 HU clip 后的逐病例 z-score，因此 gamma/HU shift 会先利用
    metadata 中的 mean/std 恢复到 HU 域，再映射回 z-score。绝不把 z-score 误裁剪到
    [0,1]。``bone_window`` 始终保持在 [0,1]。
    """
    out = image.astype(np.float32, copy=True)
    if len(input_channels) != out.shape[0]:
        raise ValueError("input_channels 数量与 image channel 数不一致")
    if len(hu_clip) != 2:
        raise ValueError("hu_clip 必须包含 [min,max]")
    hu_min, hu_max = map(float, hu_clip)
    if hu_max <= hu_min:
        raise ValueError("hu_clip 必须满足 max > min")

    ct_channel_indices = [i for i, name in enumerate(input_channels) if name == "ct_normalized"]
    needs_ct_physical_params = bool(ct_channel_indices) and (
        tuple(map(float, gamma_range)) != (1.0, 1.0)
        or tuple(map(float, hu_shift_range)) != (0.0, 0.0)
    )
    if needs_ct_physical_params:
        if ct_zscore_mean_hu is None or ct_zscore_std_hu is None or ct_zscore_std_hu <= 0:
            raise ValueError(
                "ct_normalized 的 gamma/HU shift 需要 preprocessing 0.3+ metadata 中的 "
                "clipped_mean_hu/clipped_std_hu；请重新预处理旧病例"
            )

    if len(gamma_range) != 2:
        raise ValueError("gamma_range 必须包含 [min,max]")
    gamma_min, gamma_max = map(float, gamma_range)
    if gamma_min <= 0 or gamma_max < gamma_min:
        raise ValueError("gamma_range 必须满足 0 < min <= max")
    if (gamma_min != 1.0 or gamma_max != 1.0) and rng.random() < probability:
        gamma = rng.uniform(gamma_min, gamma_max)
        for channel_index, channel_name in enumerate(input_channels):
            if channel_name == "ct_normalized":
                assert ct_zscore_mean_hu is not None and ct_zscore_std_hu is not None
                hu = out[channel_index] * float(ct_zscore_std_hu) + float(ct_zscore_mean_hu)
                unit = np.clip((hu - hu_min) / (hu_max - hu_min), 0.0, 1.0)
                hu_gamma = np.power(unit, gamma) * (hu_max - hu_min) + hu_min
                out[channel_index] = (
                    hu_gamma - float(ct_zscore_mean_hu)
                ) / float(ct_zscore_std_hu)
            elif channel_name == "bone_window":
                out[channel_index] = np.power(
                    np.clip(out[channel_index], 0.0, 1.0), gamma
                )

    if len(hu_shift_range) != 2:
        raise ValueError("hu_shift_range 必须包含 [min,max]")
    shift_min, shift_max = map(float, hu_shift_range)
    if (shift_min != 0.0 or shift_max != 0.0) and rng.random() < probability:
        shift_hu = rng.uniform(shift_min, shift_max)
        for channel_index, channel_name in enumerate(input_channels):
            if channel_name == "ct_normalized":
                assert ct_zscore_std_hu is not None
                out[channel_index] += shift_hu / float(ct_zscore_std_hu)
            elif channel_name == "bone_window":
                out[channel_index] += shift_hu / max(float(bone_window_width), 1e-6)

    if len(gaussian_noise_std_range) != 2:
        raise ValueError("gaussian_noise_std_range 必须包含 [min,max]")
    noise_min, noise_max = map(float, gaussian_noise_std_range)
    if noise_min < 0 or noise_max < noise_min:
        raise ValueError("gaussian_noise_std_range 必须满足 0 <= min <= max")
    if noise_max > 0 and rng.random() < probability:
        noise_std = rng.uniform(noise_min, noise_max)
        if noise_std > 0:
            np_rng = np.random.default_rng(rng.getrandbits(64))
            out += np_rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)

    for channel_index, channel_name in enumerate(input_channels):
        if channel_name == "bone_window":
            out[channel_index] = np.clip(out[channel_index], 0.0, 1.0)
    return out.astype(np.float32, copy=False)


class ProcessedOrthopedicCTDataset(Dataset):
    def __init__(
        self,
        processed_root: str | Path,
        split_file: str | Path,
        split: str,
        *,
        input_channels: Sequence[str] = ("ct_normalized",),
        roi_size_dhw: Sequence[int] = (128, 128, 128),
        training: bool = False,
        foreground_probability: float = 0.7,
        label_mode: str = "binary",
        augmentation: dict | None = None,
        hu_clip: Sequence[float] = (-1000.0, 2000.0),
        bone_window_width: float = 2000.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.processed_root = Path(processed_root)
        self.split_file = Path(split_file)
        self.split = split
        self.input_channels = list(input_channels)
        self.roi_size_dhw = tuple(int(v) for v in roi_size_dhw)
        self.training = bool(training)
        self.foreground_probability = float(foreground_probability)
        self.label_mode = label_mode
        self.augmentation = dict(augmentation or {})
        self.hu_clip = tuple(float(v) for v in hu_clip)
        self.bone_window_width = float(bone_window_width)
        self.seed = int(seed)

        if not (0.0 <= self.foreground_probability <= 1.0):
            raise ValueError("foreground_probability 必须位于 [0,1]")
        if self.label_mode not in {"binary", "multiclass"}:
            raise ValueError("label_mode 只能为 binary 或 multiclass")
        if len(self.hu_clip) != 2 or self.hu_clip[1] <= self.hu_clip[0]:
            raise ValueError("hu_clip 必须为 [min,max] 且 max > min")
        if self.bone_window_width <= 0:
            raise ValueError("bone_window_width 必须 > 0")

        payload = json.loads(self.split_file.read_text(encoding="utf-8"))
        if split not in payload:
            raise KeyError(f"split JSON 中不存在 {split!r}")
        self.case_ids = [str(x) for x in payload[split]]
        if not self.case_ids:
            raise ValueError(f"split {split!r} 为空")

    def __len__(self) -> int:
        return len(self.case_ids)

    def _load_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray, dict]:
        case_dir = self.processed_root / case_id
        channels = []
        for name in self.input_channels:
            if name == "ct_normalized":
                path = case_dir / "image_normalized.nii.gz"
            elif name == "bone_window":
                path = case_dir / "image_bone_window.nii.gz"
            else:
                path = case_dir / f"{name}.nii.gz"
            channels.append(_load_nifti(path, np.float32))

        shapes = {arr.shape for arr in channels}
        if len(shapes) != 1:
            raise ValueError(f"{case_id}: 输入通道 shape 不一致: {sorted(shapes)}")

        image = np.stack(channels, axis=0)  # C, X, Y, Z in nibabel array axes
        # 模型内部只要求三维轴保持一致；这里统一重排到 C,D,H,W = C,Z,Y,X。
        image = np.transpose(image, (0, 3, 2, 1))

        label = _load_nifti(case_dir / "label.nii.gz", np.int64)
        label = np.transpose(label, (2, 1, 0))  # D,H,W
        if self.label_mode == "binary":
            label = (label > 0).astype(np.int64)

        if image.shape[1:] != label.shape:
            raise ValueError(
                f"{case_id}: image/label shape 不一致: {image.shape[1:]} vs {label.shape}"
            )

        metadata: dict = {}
        metadata_path = case_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return image, label, metadata

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case_id = self.case_ids[index]
        image, label, metadata = self._load_case(case_id)

        # 对每个样本/epoch worker 使用可复现但不完全相同的随机流。
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        rng = random.Random(self.seed ^ int(worker_seed) ^ index)

        if self.training:
            aug_cfg = self.augmentation
            aug_enabled = bool(aug_cfg.get("enabled", False))
            hard_cfg = aug_cfg.get("hard_sampling", {}) if aug_enabled else {}
            hard_enabled = bool(hard_cfg.get("enabled", False))
            hard_strategy = str(hard_cfg.get("strategy", "")).lower()
            preferred_mask = None
            preferred_probability = 0.0
            if hard_enabled and hard_strategy == "boundary_proxy":
                preferred_mask = _boundary_mask(label)
                preferred_probability = float(hard_cfg.get("boundary_probability", 0.35))
                if not (0.0 <= preferred_probability <= 1.0):
                    raise ValueError("boundary_probability 必须位于 [0,1]")
            elif hard_enabled and hard_strategy not in {"", "none", "tbd_after_baseline"}:
                raise ValueError(f"暂不支持 hard_sampling.strategy={hard_strategy!r}")

            image, label = _random_crop_3d(
                image,
                label,
                self.roi_size_dhw,
                foreground_probability=self.foreground_probability,
                rng=rng,
                preferred_mask=preferred_mask,
                preferred_probability=preferred_probability,
            )

            if aug_enabled:
                geometric = aug_cfg.get("geometric", {})
                if bool(geometric.get("random_flip", True)):
                    image, label = _augment_flips(
                        image,
                        label,
                        rng=rng,
                        probability=float(geometric.get("flip_probability", 0.5)),
                        axes=tuple(int(v) for v in geometric.get("flip_axes", [0, 1, 2])),
                    )
                image, label = _augment_geometry(
                    image,
                    label,
                    rng=rng,
                    rotate_deg=float(geometric.get("random_rotate_deg", 0.0)),
                    scale_range=geometric.get("random_scale_range", [1.0, 1.0]),
                    probability=float(geometric.get("transform_probability", 0.5)),
                )

                intensity = aug_cfg.get("intensity", {})
                normalization = metadata.get("processed", {}).get("normalization", {})
                image = _augment_intensity(
                    image,
                    rng=rng,
                    input_channels=self.input_channels,
                    gamma_range=intensity.get("gamma_range", [1.0, 1.0]),
                    gaussian_noise_std_range=intensity.get(
                        "gaussian_noise_std_range", [0.0, 0.0]
                    ),
                    hu_shift_range=intensity.get("hu_shift_range", [0.0, 0.0]),
                    hu_clip=self.hu_clip,
                    bone_window_width=self.bone_window_width,
                    ct_zscore_mean_hu=normalization.get("clipped_mean_hu"),
                    ct_zscore_std_hu=normalization.get("clipped_std_hu"),
                    probability=float(intensity.get("probability", 0.5)),
                )
            else:
                # 保持 baseline 既有行为：只做简单翻转。
                image, label = _augment_flips(image, label, rng=rng)

        return {
            "case_id": case_id,
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
        }
