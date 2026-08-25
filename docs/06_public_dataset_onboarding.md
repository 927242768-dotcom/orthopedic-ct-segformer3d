# 06 公开数据集接入与首轮 baseline 操作手册

> 最近核对：2026-08-16
>
> 本文只描述公开科研数据。临床数据仍必须单独满足脱敏、授权与伦理要求。

## 1. 当前工程 baseline 默认数据集

当前先将 **VerSe complete restructured dataset** 作为工程端到端 baseline 的默认主数据集，用于尽快验证：

```text
公开 CT + vertebra mask
→ NIfTI 物理空间检查
→ 1 mm 重采样
→ HU 标准化 + bone window
→ patient/source split
→ SegFormer3D baseline
→ Dice / HD95 / ASSD
→ Web mask overlay
```

这只是工程推进默认值，**不等于组内已经最终确认论文主任务**。最终仍需确认：

- binary spine；
- multi-class vertebra；
- vertebra instance；
- 或其他骨科部位。

### 为什么优先 VerSe

- 专门面向脊柱/椎体 CT；
- 完整重构版包含 374 scans / 355 patients；
- CT 与 vertebra mask 的 NIfTI 结构明确；
- 官方维护仓库提供完整数据下载入口；
- 标签定义清楚，可直接开展 multi-label 或转换为 binary 实验；
- 相比混合来源的 CTSpine1K，更适合先验证统一的数据接入与训练链。

## 2. 已登记的数据源

机器可读登记：

```text
data/datasets.json
```

当前三类数据：

1. **VerSe complete**：工程 baseline 默认主数据；
2. **CTSpine1K**：后续扩大脊柱训练规模或做跨来源泛化；
3. **TotalSegmentator CT v2.0.1**：多骨预训练/外部泛化候选。

### 2.1 VerSe

维护仓库：

```text
https://github.com/anjany/verse
```

维护仓库描述 complete restructured data 为 **CC BY-SA 4.0**。旧 VerSe'20 challenge 页面仍可见 CC BY-SA 2.0 的旧版 data usage agreement，因此正式论文/再分发前必须再次核对当前具体发行包条款，并保留引用与 ShareAlike 要求。

标签核心定义：

```text
1-7   C1-C7
8-19  T1-T12
20-25 L1-L6
28    T13
```

### 2.2 CTSpine1K

维护仓库：

```text
https://github.com/MIRACLE-Center/CTSpine1K
```

仓库说明其 1005 CT volumes 来自 4 个开放来源，超过 11,000 个椎体标签；并明确原子数据集采用 CC-BY-NC-SA 系列许可且保持原许可。因此使用时应保留**每个组成来源的数据 provenance 与具体许可**，不要把整个混合数据集粗略写成一个未经核验的统一许可证版本。

### 2.3 TotalSegmentator CT v2.0.1

Zenodo：

```text
https://zenodo.org/records/10047292
DOI: 10.5281/zenodo.10047292
```

当前记录：1228 CT、117 structures、v2.0.1、归档约 23.6 GB、MD5 `fe250e5718e0a3b5df4c4ea9d58a62fe`。

注意：**不能把 TotalSegmentator 软件仓库的 Apache-2.0 自动当成数据集许可证。** 使用数据前必须核对 Zenodo 对应数据记录的 Rights/License。

## 3. 下载 VerSe 与当前网络状态

2026-08-16 当前工作站实测：VerSe 官方 S3 多个归档连接仍会超时，因此 **VerSe 原始数据尚未落盘**。不过 CTSpine1K 的真实小样本链已打通并完成 10 例工程验证，见 3.1。以下 VerSe 脚本继续保留为正式 benchmark 候选入口；网络恢复或切换到可达环境后可直接继续。

先只查看下载计划，不实际下载：

```powershell
cd D:\国创项目
powershell -ExecutionPolicy Bypass -File .\env\download_verse.ps1 -Edition 2020
```

明确需要下载后：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\download_verse.ps1 -Edition 2020 -Download
```

默认目录：

```text
data/raw_public/VerSe
```

原始 ZIP 与解压后的影像均已被 `.gitignore` 排除，禁止提交到 Git。

### 3.1 VerSe 不可达时的 CTSpine1K 小样本备用入口

CTSpine1K 官方维护仓库已提供 Hugging Face 镜像；当前镜像结构包含：

```text
raw_data/volumes/<sub-dataset>/*.nii.gz
raw_data/labels/<sub-dataset>/*_seg.nii.gz
```

2026-08-16 早期测试中 Hugging Face 曾出现约 20 秒连接超时/并行下载失败；随后改用 Edge 浏览器**单文件顺序下载**后成功取得 `MSD-T10` 10 例真实 CT+label。当前已落盘病例为 `liver_0`—`liver_8` 与 `liver_169`，官方 split 统计为 9 例 `trainset` + 1 例 `test_private`。命令行/并行下载稳定性仍不可靠，因此后续扩量应继续保留断点、校验和 provenance 记录。

默认先查看 3 个 MSD-T10 小病例下载计划：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\download_ctspine1k_sample.ps1
```

网络可达后显式下载：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\download_ctspine1k_sample.ps1 -Download
```

下载脚本默认仍只列出 `liver_169`、`liver_0`、`liver_1`，用于低成本 smoke test。本轮实际工程验证已扩展到 `liver_0`—`liver_8`、`liver_169` 共 10 例；它们仍只用于**真实预处理/QC 与工程 smoke test**，不能因为下载方便就直接当成正式论文 train/validation/test。尤其 `liver_169` 的官方标记为 `test_private`，不得用于训练调参。

下载完成后先 dry-run：

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_ctspine1k `
  --source-root "D:\国创项目\data\raw_public\CTSpine1K" `
  --output-root "D:\国创项目\data\processed\ctspine1k_qc_v0.1" `
  --split-file "D:\国创项目\data\raw_public\CTSpine1K\data_split.txt" `
  --dry-run
```

正式处理小样本并同步生成 QC 图：

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_ctspine1k `
  --source-root "D:\国创项目\data\raw_public\CTSpine1K" `
  --output-root "D:\国创项目\data\processed\ctspine1k_qc_v0.1" `
  --split-file "D:\国创项目\data\raw_public\CTSpine1K\data_split.txt" `
  --limit 3 `
  --qc
```

`prepare_ctspine1k` 只保留官方 `trainset / test_public / test_private` 标记，**不会擅自把 public/private test 重解释成 validation/test**。正式论文实验划分必须另行固定并记录。

## 4. 解压后的预期结构

VerSe complete 常见结构示意：

```text
<source-root>/
├─ 01_training/
│  ├─ rawdata/sub-verseXXX/*_ct.nii.gz
│  └─ derivatives/sub-verseXXX/*_seg-vert_msk.nii.gz
├─ 02_validation/
└─ 03_test/
```

批处理代码也识别：

- `training / validation / test`；
- `verse19training / verse19validation / verse19test`；
- `verse20training / verse20validation / verse20test`。

## 5. 先 dry-run，不处理体数据

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_verse `
  --source-root "D:\国创项目\data\raw_public\VerSe\<解压目录>" `
  --output-root "D:\国创项目\data\processed\verse_v0.2.0" `
  --dry-run
```

该步骤会先检查：

- CT 与 vertebra mask 是否一一匹配；
- 是否能识别官方 source split；
- 同一 patient group 是否跨 split；
- 是否存在重复 case id。

输出：

```text
verse_manifest.json
verse_official_split.json
batch_qc_summary.json
```

## 6. 首轮只处理 10 例做人工 QC

### 6.1 2026-08-16 已执行的 CTSpine1K 真实工程验证

当前已实际完成：

```text
raw:       data/raw_public/CTSpine1K/MSD-T10
processed: data/processed_ctspine1k_real
cases:     liver_0—liver_8 + liver_169
source split: 9 trainset + 1 test_private
pipeline:  0.3.0
spacing:   1.0 × 1.0 × 1.0 mm
QC:        10/10 automatic audit pass
contact sheets: 10/10 generated
manual_qc_review.csv: 已生成，人工字段仍待项目成员逐例填写/签字
```

10 例原始 CT 的 z-spacing 覆盖约 `0.8 / 1.0 / 5.0 mm`，因此当前小样本已经真实覆盖了厚层与近各向同性来源；重采样后标签类别均保持为原始类别子集，没有出现插值产生的新标签值。自动审计通过只表示**工程一致性检查通过**，不能替代人工解剖学/标注合理性审核。

VerSe 网络恢复后仍可按同一 SOP 做首轮 10 例：

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_verse `
  --source-root "D:\国创项目\data\raw_public\VerSe\<解压目录>" `
  --output-root "D:\国创项目\data\processed\verse_v0.2.0" `
  --limit 10 `
  --spacing 1 1 1 `
  --hu-min -1000 `
  --hu-max 2000 `
  --bone-center 500 `
  --bone-width 2000 `
  --qc
```

每例标准输出：

```text
case_xxx/
├─ image_normalized.nii.gz
├─ image_bone_window.nii.gz
├─ label.nii.gz
├─ metadata.json
├─ qc.json
└─ qc_contact_sheet.png          # 使用 --qc 时生成
```

其中：

- image 用 linear 重采样；
- label 用 nearest-neighbor 重采样；
- 原始 image/label 的 size、spacing、origin、direction 必须一致，否则直接拒绝进入训练；
- label 重采样后不得产生原本不存在的新类别值。

## 7. 人工 QC 最低要求

`qc_contact_sheet.png` 固定为 3×3 版式：axial/coronal/sagittal × normalized CT/bone window/label overlay。若病例存在前景标签，切片位置优先取前景区域中位位置，避免机械选择体积中心导致看不到椎体标签。

如果处理完成后需要统一重建 QC 图和人工审核表，可运行：

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.qc_visualization `
  "D:\国创项目\data\processed\<dataset_version>" `
  --limit 10
```

该命令会生成/刷新每例 `qc_contact_sheet.png`，并在根目录生成：

```text
manual_qc_review.csv
qc_visualization_summary.json
```

`manual_qc_review.csv` 预留 orientation、spacing、label alignment、bone window、review status、reviewer、notes 等人工审核字段。

首轮至少 10 例，每例检查：

1. axial；
2. coronal；
3. sagittal；
4. bone window；
5. label overlay；
6. 重采样前后椎体数量与形态；
7. label 是否出现整体平移、翻转或轴交换；
8. 部分可见椎体、骨折、异常椎体是否符合数据集标注规则。

当前 10 例已经完成自动几何/标签/spacing/normalization 审计，因而可以把 **NIfTI 预处理工程链**标记为“已完成真实数据工程验证”；但 `manual_qc_review.csv` 的人工字段尚未签字，因此不能把“10 例人工 QC 已完成”写成事实，更不能据此声称临床有效性。

## 8. 全量处理与训练前检查

10 例人工 QC 通过后再全量处理：

```powershell
.\.venv\Scripts\python.exe -m src.preprocessing.prepare_verse `
  --source-root "<VerSe 解压目录>" `
  --output-root "D:\国创项目\data\processed\verse_v0.2.0"
```

正式 baseline 前必须存在：

```text
processed cases
+ verse_official_split.json
+ batch_qc_summary.json failure_count = 0（或失败病例已逐例解释并排除）
+ GPU 环境确认
+ SegFormer3D forward 已验证
```

训练入口仍为：

```powershell
.\.venv\Scripts\python.exe -m src.modeling.train --config configs\orthopedic_ct_baseline.yaml
```

但需要先把 config 的 `processed_root` / `split_file` / 类别定义修改到本次真实数据版本，并把实际实验 config 固化到 `experiments/<run_id>/`。

## 9. 当前代码对应关系

```text
src/preprocessing/dicom_pipeline.py
  DICOM series 发现、QC、显式几何排序、重采样、HU/骨窗

src/preprocessing/nifti_pipeline.py
  单例 NIfTI image/label 几何校验、重采样、标准化、标准输出

src/preprocessing/prepare_verse.py
  VerSe 批量发现、配对、source split、防泄漏、标准化，可选 --qc

src/preprocessing/prepare_ctspine1k.py
  CTSpine1K image/label 配对、官方 split 标记、标准化，可选 --qc

src/preprocessing/qc_visualization.py
  axial/coronal/sagittal × normalized/bone-window/label-overlay QC 图与人工审核 CSV

env/download_ctspine1k_sample.ps1
  CTSpine1K 小样本下载计划/显式下载；默认不下载

data/datasets.json
  数据来源/版本/许可核验登记及本机可达性备注
```

## 10. 数据接入完成的验收定义

只有满足以下条件，`PROJECT_STATUS.md` 中的“公开数据集整理 / DICOM/CT 处理流程”才能继续上调：

- 实际下载公开数据；
- 登记版本、来源、许可、下载日期；
- [x] 至少 10 例真实 3D CT 完成标准化（CTSpine1K MSD-T10，pipeline 0.3.0）；
- [ ] 三视图 + overlay **人工**抽检完成并在 `manual_qc_review.csv` 签字（自动 contact sheet/audit 已完成）；
- [x] 批量 QC/manifest 可追踪；
- [ ] 正式论文 train/validation/test split 文件生成（当前只有 engineering smoke split，禁止当正式实验划分）；
- [x] 训练 Dataset 能读取真实病例，且真实双通道 + joint loss 单 patch forward/backward/optimizer step 已通过。
