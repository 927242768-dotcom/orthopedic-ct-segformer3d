# 03 骨科 CT 数据处理规范（DICOM → 标准化体数据）

> 目标：建立可复现、可审计、可用于训练/推理的一致数据流程。

## 1. 输入边界

支持的输入优先级：

1. DICOM series 文件夹；
2. DICOM ZIP（Web 上传后解压到临时隔离目录）；
3. NIfTI (`.nii/.nii.gz`)；
4. 已标准化缓存（仅内部使用）。

任何临床数据必须满足：

- 已脱敏；
- 有合法研究授权；
- 不在日志、Git、截图中泄露身份信息；
- 数据来源、使用范围、授权状态可追踪。

## 2. DICOM Series 识别

### 2.1 不依赖文件名排序

禁止简单按 `1.dcm, 2.dcm, 10.dcm` 文件名排序。

当前 `src/preprocessing/dicom_pipeline.py` 已实现显式切片排序：优先按几何位置；几何字段不完整时，仅允许在所有切片 `InstanceNumber` 完整且唯一时回退，否则直接拒绝自动处理。

优先按几何信息排序：

1. 读取 `ImageOrientationPatient` 得到行/列方向；
2. 计算切片法向量；
3. 使用 `ImagePositionPatient · normal` 得到切片位置；
4. 按位置排序；
5. 若几何字段缺失，再回退到 `SliceLocation` / `InstanceNumber`；
6. 若仍不能可靠排序，将病例标为 QC fail，不自动猜测。

### 2.2 Series 归并

根据至少以下信息归并：

- StudyInstanceUID；
- SeriesInstanceUID；
- Modality；
- Rows / Columns；
- orientation；
- pixel spacing。

一个文件夹中若包含 scout、localizer、不同重建核或不同相位，必须分开处理。

## 3. HU 恢复

CT 像素值通过：

```text
HU = pixel_value × RescaleSlope + RescaleIntercept
```

要求：

- slope/intercept 缺失时记录 warning；
- 不能假设所有设备固定 `slope=1, intercept=-1024`；
- 对明显异常值先保留原始统计，再做 clip。

记录：

- HU min/max；
- p0.5/p1/p50/p99/p99.5；
- air/background 比例；
- 是否存在异常极值。

## 4. 空间几何

必须保留并验证：

- spacing；
- origin；
- direction/orientation；
- shape；
- physical extent。

影像和标签必须在同一物理空间对齐。

### 4.1 方向统一

训练前统一到约定方向（例如 RAS/LPS 之一），但具体实现必须明确库的坐标约定。

不允许只对 NumPy 轴做转置而忽略 affine/direction。

## 5. 重采样

### 5.1 影像

推荐：linear / B-spline（以实验确定）。

### 5.2 标签

必须 nearest-neighbor，避免生成不存在的类别值。

### 5.3 spacing 策略

首版候选：

- 统一 1 mm 各向同性；
- 或根据数据中位 spacing 设定；
- 对极厚层 CT 额外记录插值风险。

最终 spacing 通过以下因素决定：

- GPU 显存；
- 目标骨最小结构尺度；
- 原始 z-spacing 分布；
- patch 大小；
- 重建表面质量。

不得在无统计情况下把单一 spacing 当作“最佳值”。

## 6. 强度处理

建议将“原始 CT 标准化通道”和“骨窗通道”分开配置。

### 6.1 通用 CT 通道

流程：

```text
HU → clip(low, high) → normalize
```

low/high 由训练数据统计与任务部位决定。

### 6.2 骨窗通道

采用 window center / width 生成：

```text
lower = center - width / 2
upper = center + width / 2
bone = clip(HU, lower, upper)
bone = (bone - lower) / (upper - lower)
```

骨窗参数作为 config 保存并进入消融实验，不应散落硬编码。

## 7. ROI 与 crop

原则：

- 保证目标骨完整；
- 训练 patch 不能导致测试时缺失全局位置关系；
- patient-level 数据分割在 crop 之前确定；
- 每个 patch 必须能追踪回原病例与原物理坐标。

可采用：

- body foreground crop；
- coarse localization；
- 随机正/负 patch；
- hard-example sampling。

## 8. 困难样本增强

初始策略分 4 类：

### 8.1 几何增强

- small rotation；
- scaling；
- elastic deformation（谨慎，避免不合理骨形变）；
- random crop/shift。

### 8.2 强度增强

- gamma/contrast；
- noise；
- blur/sharpen；
- HU shift/scale（模拟设备/协议差异）。

### 8.3 伪影模拟

针对金属伪影可考虑受控条纹/高亮扰动，但必须证明增强不会制造明显非真实模式。

### 8.4 hard sampling

根据上一轮模型：

- 高 loss 病例；
- 高 uncertainty 区域；
- 高 HD95 病例；

增加抽样概率。

## 9. 质量控制（QC）

### 9.1 自动 QC

每例输出 `qc.json`，至少包含：

```json
{
  "status": "pass|warning|fail",
  "warnings": [],
  "dicom": {
    "slice_count": 0,
    "series_uid_hash": "",
    "duplicate_position_count": 0
  },
  "geometry": {
    "shape": [0, 0, 0],
    "spacing_mm": [0, 0, 0],
    "orientation_valid": true
  },
  "intensity": {
    "hu_min": 0,
    "hu_max": 0,
    "p01": 0,
    "p99": 0
  }
}
```

### 9.2 QC fail 条件示例

- 无法确定 slice 顺序；
- 多个混合 series 未能区分；
- 切片尺寸不一致且无法解释；
- spacing <= 0；
- 图像像素解码失败；
- 标签与图像空间不一致；
- 存在严重缺层/重复层。

### 9.3 人工抽检

每批数据至少抽检：

- axial/coronal/sagittal；
- bone window；
- label overlay；
- 重采样前后关键结构。

## 10. 数据划分

### 10.1 患者级划分

严禁将同一患者的不同切片分到 train/val/test。

若同一患者存在多次扫描，也应根据研究问题决定是否全部归到同一 split，防止身份泄漏式过拟合。

### 10.2 多中心设置

建议两类实验：

1. 混合中心随机 patient split；
2. leave-one-center/source-out 外部泛化。

## 11. 输出规范

当前工程标准输出与训练 Dataset 统一为：

```text
case_xxx/
├─ image_normalized.nii.gz
├─ image_bone_window.nii.gz  # 启用骨窗时生成
├─ label.nii.gz              # 有监督数据必须有
├─ metadata.json
├─ qc.json
└─ preview/                  # 可选，不含身份字段
```

`metadata.json` 必须保存预处理参数与原空间恢复信息，以支持 Web 和三维重建。

### 11.1 NIfTI 公开数据入口

`src/preprocessing/nifti_pipeline.py` 已实现单病例 NIfTI 标准化：

- 检查 3D image/label 的 size、spacing、origin、direction；
- image/label 原始物理空间不一致时拒绝进入训练，不自动猜测修正；
- image 使用 linear 重采样；
- label 使用 nearest-neighbor 重采样；
- 重采样后校验 label 不出现原本不存在的类别值；
- 输出格式与 `ProcessedOrthopedicCTDataset` 完全一致。

VerSe 批量入口见 `src/preprocessing/prepare_verse.py` 与 `docs/06_public_dataset_onboarding.md`。

## 12. 可复现性

每次批处理必须记录：

- git commit；
- pipeline version；
- config；
- 数据集版本；
- 处理时间；
- 成功/失败病例；
- 环境依赖版本。

若更改 HU clip、spacing、window、orientation、crop 等任何关键参数，都应视为新 preprocessing version，不能静默覆盖旧数据。
