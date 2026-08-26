# 05 中期研究材料持续汇总

> 版本：v0.1｜2026-08-15
>
> 用途：用于 2026 年 11 月中期检查前持续累积真实材料。本文只记录已经完成或明确处于进行中的工作；模型性能数字必须由正式实验产生后再填写。

## 1. 项目题目

**基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究**

## 2. 中期阶段目标对应关系

根据任务书，2026 年 5—7 月应完成文献调研、总体方案和实验环境，并建立 DICOM 解析、灰度/HU 标准化、空间重采样、骨窗增强和质量控制流程；2026 年 7—9 月应围绕 SegFormer 建立骨科 CT 多尺度分割、联合损失、困难样本增强、不确定性精修，并开展模型训练、调参与消融。

当前材料按“**已有工程证据—尚缺真实实验**”分开记录，避免把计划写成结果。

---

## 3. 已完成的阶段性工作

### 3.1 项目方案和工程目录

已建立统一技术路线：

```text
DICOM/NIfTI
→ 脱敏/数据治理
→ DICOM series 解析与 QC
→ HU/骨窗/方向/spacing 标准化
→ SegFormer3D 多尺度分割
→ Region + Boundary + Topology 联合损失
→ 困难样本学习
→ 不确定性定位与局部精修
→ mask 后处理
→ 三维表面重建
→ Web MPR / 3D / 测量
```

工程目录已按 `docs / configs / env / src / web / paper / tests / data / third_party` 分层，并以 `PROJECT_STATUS.md` 作为唯一进度台账。

### 3.2 国内外文献调研与结构化题录

已形成 `docs/08_literature_matrix.md` 的 **44 条结构化文献矩阵**，并建立 `paper/references.bib` 的 **42 条英文核心 BibTeX**。当前覆盖：

- SegFormer / SegFormer3D；
- UNETR / nnFormer / Swin UNETR；
- CTSpine1K / VerSe / TotalSegmentator；
- 2024 年脊柱 Transformer 方法 VerFormer；
- 2025 SpineMamba、2025 解剖变异感知椎体 Transformer、2026 VertebraFormer 等近期直接脊柱工作；
- 2026 开放椎体体部数据 + Residual-Encoder nnU-Net 强 baseline；
- 2025 椎体骨折 nnU-Net pipeline、真实腰椎金属植入物 deep-MAR，以及低骨密度导致椎体 fusion/split 的直接困难病例证据；
- Boundary Loss；
- clDice；
- 2024 年 Betti matching / TEDS-Net 等拓扑方法；
- EDUE / UCTNet 等不确定性方法；
- 国内肋骨 CT 三维分割与重建、椎体 CT 三维分割研究。

当前结论：本项目不应把“使用 Transformer”本身作为主要创新，而应聚焦**骨科 CT 标准化、多尺度轻量 3D 分割、表面/拓扑质量、困难病例和不确定性可靠性**。本轮同时纠正了 CTSpine1K、VerSe、VerFormer、nnFormer、SegFormer3D、TEDS-Net、EDUE 等易被二手题录写错的条目；国内文献仍需投稿前回 CNKI/万方逐条复核。

### 3.3 独立实验环境

已在 `D:\国创项目` 内完成隔离环境，不污染系统 Python：

- 项目内 Python：3.11.7；
- 虚拟环境：`.venv`；
- PyTorch：2.1.0 CPU 版；
- MONAI：1.2.0；
- pydicom / SimpleITK / nibabel；
- FastAPI；
- Lightning 2.0.9 / PyTorch Lightning 2.0.9；
- NumPy / SciPy / scikit-image 等。

当前电脑未检测到 `nvidia-smi`，因此**只完成 CPU 兼容验证，不把本机作为正式 3D CT 训练平台**。正式训练仍需确认 NVIDIA GPU/驱动或学校服务器。

### 3.4 SegFormer3D 上游代码与许可证边界

官方 `OSUPCVLab/SegFormer3D` 已克隆到：

```text
third_party/SegFormer3D
```

基线提交：`e314242`。

为兼容 PyTorch 2.1 TorchScript，对上游 `cube_root()` 做了显式 `int()` 的类型修复；补丁已在 `third_party/README.md` 记录。没有把上游 backbone 拷贝到自研目录冒充原创。

已通过 CPU smoke test：

```text
input : (1, 1, 64, 64, 64)
output: (1, 2, 64, 64, 64)
params: 4,492,066
```

该结果仅证明模型结构能在当前环境前向运行，**不是训练性能结果**。

### 3.5 DICOM / CT 标准化流程首版

已实现：

- 递归发现 DICOM；
- 按 `SeriesInstanceUID` 分组；
- `ImageOrientationPatient + ImagePositionPatient` 几何位置检查；
- PixelSpacing / z-spacing / 重复位置 QC；
- SimpleITK series 读取；
- 体素重采样；
- HU clip + 标准化；
- 骨窗通道；
- metadata/QC JSON；
- NIfTI 输出；
- patient-level split 生成器。

已使用 pydicom 自带 `CT_small.dcm` 做单切片 smoke test，成功生成：

```text
image_normalized.nii.gz
image_bone_window.nii.gz
metadata.json
qc.json
```

由于该测试样本只有 1 层，QC 正确给出 warning。下一阶段仍必须使用真实多层公开 CT series 验证切片排序、方向与 z-spacing。

### 3.6 区域—边界—拓扑联合损失首版

已实现：

- Dice + CE/BCE 区域损失；
- 基于 GT signed distance field 的 Boundary Loss；
- 3D soft skeletonization；
- soft-clDice 拓扑候选；
- 配置化 `JointOrthopedicSegLoss`。

当前限制：soft-clDice 并不天然适合所有骨结构；特别是骨折是真实断裂时，不能为了“拓扑连续”把断端错误连接。因此拓扑项必须在正常/骨折子集上分层消融。

### 3.7 不确定性首版

已实现：

- sigmoid/softmax 概率转换；
- 归一化 predictive entropy；
- Top-percent 高不确定体素选择；
- ROI 膨胀；
- uncertainty ROI 与真实错误的 overlap 统计；
- uncertainty→error AUROC / AUPRC；
- error/correct 平均 entropy；
- Top-percent error recall、ROI error rate 与 ROI fraction；
- calibration：ECE、MCE、multiclass Brier score、NLL、mean confidence、体素 accuracy 与 confidence gap。

校准指标已接入独立 checkpoint evaluation，并使用固定分箱/固定随机种子体素采样保证不同实验可复现比较。已进一步实现 `UncertaintyRefinementNet3D` 局部残差精修原型和二阶段 ROI-only 训练基线：以影像、coarse 预测和 uncertainty 为输入，只在高不确定 ROI 内修正 logits，coarse 默认冻结，loss 在 ROI 内归一化，ROI 外保持 coarse 结果不变，并记录 refinement 前后 ROI/global error delta。当前尚未在真实 checkpoint 上完成消融，因此不能声称精修已经带来性能提升。

### 3.8 训练与评价框架

已建立：

- 处理后 NIfTI dataset；
- patient-level split；
- 前景偏置 3D patch crop；
- baseline / joint-loss YAML 配置；
- SegFormer3D adapter；
- AdamW/AMP/gradient accumulation；
- MONAI sliding-window validation；
- checkpoint、scheduler state、固定 split/config、环境版本、`train.log` 与历史日志；
- Dice / IoU / Precision / Recall / HD95 / ASSD；
- connected-component count / false merge / false break；
- 独立 checkpoint evaluation，可输出逐病例 `metrics_per_case.csv`、prediction、entropy NIfTI，以及 uncertainty/calibration 指标；
- 逐病例评价设计；
- 正式/工程 `preflight`：阻止 engineering split、test_private 泄漏、人工 QC 未通过、标签范围/类别数不匹配或正式训练无 CUDA 时误启动论文实验；
- `task_lock`：只有组内明确 `task_locked=true` 且 binary/multiclass semantic 定义、类别数、数据与 split 均完整时才允许编译正式 config；当前 instance 任务因训练/评价链尚未实现而明确拒绝；
- `gpu_environment` + `formal_readiness`：只读检查 PyTorch CUDA build、可见 GPU、显存、`nvidia-smi`，并将 task/GPU/formal preflight 汇总为统一 blocker 报告；当前真实工程 split 验证返回 `ready=false`，不会误启动正式训练；
- multiclass 逐病例宏平均只统计真值或预测实际出现的前景类别，并输出 `metrics_per_class.csv`，避免双方都缺失的类别虚高 Dice。

2026-08-16 已在 CTSpine1K `MSD-T10` 真实标准化病例上完成 CT+bone-window 双通道、joint loss、backward、AdamW.step 的 36³ CPU 单 patch smoke test，证明真实数据训练链可运行。**这仍不是正式训练，随机权重 loss/梯度不属于模型性能。**

### 3.9 Web 科研辅助分析原型

已完成第一版 FastAPI + 前端：

- 系统健康检查；
- DICOM/NIfTI 多文件上传；
- 随机 case id，后端不保留原始文件名；
- series/QC 信息检查；
- axial / coronal / sagittal MPR 三视图与归一化切片位置、窗宽窗位控制；
- 真实 10 例人工 QC reviewer：contact sheet + 交互式 MPR + 真值 label overlay + 四项人工检查 + reviewer/status/notes；
- CTSpine1K/VerSe `1–25 → C1–L6` 可读标签显示，原始数值不重编码且不锁定正式任务；
- 真实真值 label 的 WebGL2 3D PLY 查看、1.5/2.0 mm 简化选项；
- binary mask→physical signed-distance field→零等值面 SDF 表面选项，并设置连通域保护；真实 `liver_0` 的 0.4 mm 参数可加载，0.8 mm 因 2→3 连通域变化被拒绝；
- evaluation results-review 页面：可发现未来正式 `evaluate.py` 输出，并查看 prediction/entropy MPR overlay；当前真实 evaluation 数为 0，因此不会伪造模型结果；
- 物理毫米坐标距离和三点夹角计算 API；
- 研究用途免责声明；
- 模型分割接口已预留。

Web `TestClient` 验证：

```text
GET /api/health → 200
GET /           → 200 text/html
```

当前 `/infer` 会明确返回 `501 not_ready`，原因是尚无真实训练 checkpoint。**这是有意设计，不使用随机权重伪造“诊断结果”。**

### 3.10 三维重建与重采样几何工程验证

已在真实 `liver_0` 标签上完成 physical-space Marching Cubes：131,983 顶点 / 264,362 面。为适配 Web，又实现 vertex-clustering 简化：1.5 mm 档约减少 60% 顶点/面，顶点近邻 HD95 约 0.707 mm；全分辨率网格仍保留，不被简化版覆盖。2026-08-26 进一步加入“法向变化加权”的曲率/关键边缘保护候选：在 2.0 mm clustering、保持相同 30,260 个简化顶点的条件下，真实 `liver_0` 高法向变化区域平均最近邻误差由约 0.679 mm 降至 0.620 mm，HD95 由约 1.068 mm 降至 1.000 mm，表面积相对变化由约 -6.47% 改善至 -5.95%。这些结果只用于真值网格工程参数筛选，尚未在 prediction surface 上验证。

进一步实现 physical-mm SDF 平滑表面基线：0.3/0.4/0.5/0.8 mm 参数已在真实 `liver_0` 上做工程 sweep。0.4 mm 保持 2→2 连通域且 Web summary/PLY 均返回 200，作为当前工程默认候选；0.8 mm 导致 2→3 连通域变化，Web 按拓扑保护返回 422 拒绝加载。该 sweep 的表面差异只用于重建参数选择，不属于分割模型性能。

同时对 10 例原始 label 与 1 mm nearest-neighbor 重采样 label 做物理表面离散化比较：整体顶点近邻 ASSD 约 0.403 mm、HD95 约 0.734 mm；原始 5 mm 厚层的 3 例扰动更明显（ASSD 约 0.514 mm、HD95 约 1.069 mm）。这些数字只描述**预处理/离散化工程误差**，不是模型性能或临床测量误差。

### 3.11 自动化测试

当前测试：

```text
94 passed
Ruff: All checks passed!
```

覆盖：

- HU clip / bone window；
- 联合损失 forward/backward；
- SegFormer3D adapter；
- 区域与表面指标；
- uncertainty entropy/ROI 与局部 refinement；
- CTSpine1K 真实/合成接入、QC 与处理后数据审计；
- 数据增强、scheduler、独立 evaluation；
- physical-space Marching Cubes / PLY、网格简化、法向变化加权特征保护候选、重采样几何误差与物理测量；
- formal/engineering preflight、task lock、GPU/formal readiness、multiclass per-class 评价、uncertainty 定量评价、calibration 评价、ROI-only refinement training；
- SimpleITK 中文项目路径兼容层；
- 椎体标签 schema；
- Web health/index/MPR/QC reviewer/真实 label 3D/SDF 表面/测量/results-review。

---

## 4. 当前尚未完成、不能在中期材料中伪写完成的内容

1. **CTSpine1K 10 例真实工程子集已下载/标准化，但正式论文主数据集及全量预处理尚未确定；**
2. **10 例自动 QC 已完成，`manual_qc_review.csv` 仍待项目成员逐例人工签字；**
3. **临床脱敏数据尚未提供；**
4. **首个具体骨科任务/标签集尚需最终确定；**
5. **GPU 正式训练未开始；**
6. **无 baseline Dice / HD95 / ASSD；**
7. **无联合损失、困难增强、uncertainty refinement 的真实消融数字；**
8. **Marching Cubes、真实 label 网格、1.5/2.0 mm 工程简化、10 例重采样几何误差以及真值 mask 的 SDF 表面工程基线已验证；但真实模型 prediction mask 的表面质量仍未验证；**
9. **Web 已接入人工 QC、交互 MPR+真值 overlay、真实 label 3D/SDF 表面、物理坐标测量和 evaluation results-review；由于尚无正式 checkpoint，当前真实 evaluation 列表为 0，prediction/uncertainty overlay 与预测网格没有可供展示的正式结果。**

任务书中的 `Dice ≥ 0.93` 是目标值，不是当前成果。

---

## 5. 中期报告“研究进展”可直接使用的表达

截至当前阶段，项目已完成总体技术路线设计、44 条结构化文献矩阵与 42 条英文核心 BibTeX，并搭建隔离实验环境。围绕骨科 CT 已建立 DICOM 序列识别、空间几何检查、HU 裁剪与 case-wise z-score、体素重采样、骨窗增强及质量控制流程。2026-08-16 已实际取得 CTSpine1K `MSD-T10` 10 例真实 CT+label（9 例官方 trainset、1 例 test_private），全部按 pipeline 0.3.0 重采样到 1 mm，自动几何/标签/normalization 审计 10/10 通过；人工审核系统已能逐例查看 contact sheet 与交互式 MPR+真值 overlay，但人工签字仍待项目成员完成。针对 SegFormer3D 与骨科 CT 的差异，项目已完成模型适配、三维 patch 数据加载、区域—边界—拓扑联合损失、困难增强、predictive entropy 定量评价、ECE/MCE/Brier/NLL 等概率校准评价、ROI-only 局部残差精修训练基线、区域/表面/连通结构评价、formal preflight、task lock、GPU/formal readiness、可复现训练日志及独立 checkpoint 评估。真实病例上的 CT+bone-window + joint-loss 单 patch forward/backward/optimizer step 已通过。Web 原型现支持上传、QC/MPR、真值 3D PLY/SDF 表面、1.5/2.0 mm 网格简化、物理距离/角度和未来 evaluation 结果复核；真实 `liver_0` 全前景网格为 131,983 顶点 / 264,362 面，0.4 mm SDF 表面通过连通域保护，0.8 mm 参数被拒绝，并已完成 10 例原始→1 mm 标签的重采样几何误差评估。当前正式任务仍未锁定，本机为 CPU PyTorch 且人工 QC 未签字，formal readiness 正确阻止正式 run，因此尚未产生可用于论文结论的 baseline DSC/HD95/ASSD 或消融数字。下一阶段重点仍是人工 QC 签字、正式任务/split/GPU baseline、真实逐病例评估与联合损失/困难增强/不确定性精修消融。

---

## 6. 中期展示建议

中期 PPT 建议只展示能被工程文件复核的内容：

1. 任务书技术路线与本项目实际模块图；
2. DICOM → HU → spacing → bone window → QC 流程图；
3. SegFormer3D backbone 与本项目新增模块边界；
4. 联合损失公式与为什么需要 HD95/ASSD；
5. Web 首页/上传/QC 真实截图；
6. 测试结果 `94 passed`、10 例真实数据自动审计、真实 patch smoke、formal readiness 与真实重采样/SDF/网格工程证据；
7. 数据集对比表；
8. baseline/消融实验表格模板（数值留空直到真实实验）；
9. 风险：GPU、临床数据、许可证、拓扑对骨折病例的适用性；
10. 下一阶段甘特图。

---

## 7. 下一阶段形成可验收“中期硬证据”的最低清单

- [ ] 明确首个任务：建议优先脊柱/椎体 CT；
- [x] 下载并登记 CTSpine1K 10 例真实工程子集；
- [x] 对 ≥10 例真实多层 CT 完成标准化和自动 QC；
- [ ] 项目成员完成 10 例三视图/overlay 人工审核并在 CSV 签字；
- [ ] 固定 patient-level split；
- [ ] GPU 跑通 SegFormer3D baseline；
- [ ] 生成第一版 `metrics_per_case.csv`；
- [ ] 报告 baseline DSC / HD95 / ASSD；
- [ ] 完成 Region vs Region+Boundary 初步消融；
- [ ] Web 接入真实 checkpoint，显示分割叠加；
- [ ] 保存可复现 config / commit / checkpoint /日志；
- [ ] 更新论文 Results 第一张真实表。
