# 骨科 CT 智能分割与三维重建项目——主进度与交接台账

> 项目目录：`D:\国创项目`
>
> 项目名称：**基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究**
>
> 台账首次建立：2026-08-15
>
> 最近更新：2026-08-28

---

## 0. 强制维护规则——任何人继续项目必须先读

**本文件是项目唯一主进度台账。任何代码、文档、配置、数据流程、模型实验、网站功能、论文内容发生实质性更改后，都必须同步更新本文件。**

每次修改项目后，最后必须完成以下检查：

1. 更新“当前总体状态”中对应模块的状态/完成度；
2. 在“最近工作记录”中追加本次实际完成内容、测试结果和遗留问题；
3. 如新增/删除/移动重要文件，更新“关键文件索引”；
4. 如任务优先级发生变化，更新“下一步任务”；
5. 如出现新的依赖、数据、硬件、许可证、医学合规风险，更新“阻塞与风险”；
6. **不得把计划、预期指标、随机权重输出、未验证功能写成“已完成”或论文结果。**

状态统一使用：

- `✅ 已完成`：实现已经存在并通过当前阶段所需的基本验证；
- `🟡 进行中`：已开始，有可检查成果，但尚未达到阶段验收条件；
- `🟠 待真实验证`：代码/方案存在，但缺少真实数据、正式实验或外部验证；
- `🔴 阻塞`：受数据、GPU、授权、伦理或其他外部条件限制；
- `⚪ 未开始`：尚未开展。

> **特别强调：任务书中的 Dice ≥ 0.93 是项目目标，不是当前实验结果。当前已产生真实 engineering / validation 实验，但尚未完成最终锁参与正式 independent test；validation 指标不得冒充 independent-test final Results。**

---

## 1. 项目任务基线

项目总体技术路线：

```text
数据标准化处理
→ 骨结构精准分割
→ 连续几何/三维重建
→ 轻量化 Web 系统集成
→ 综合测试验证
```

任务书前两阶段核心要求：

### 2026 年 5—7 月：前期准备与数据处理

- 国内外相关文献调研；
- 项目总体方案设计；
- 实验环境搭建；
- 公开数据集和临床脱敏数据整理；
- DICOM 解析；
- HU/灰度标准化；
- 空间重采样；
- 骨窗增强；
- 质量控制；
- 建立规范化骨科 CT 数据处理流程。

### 2026 年 7—9 月：骨骼智能分割模型

- 基于 SegFormer 构建骨科 CT 多尺度分割模型；
- 区域重叠、边界约束、拓扑保持联合损失；
- 困难样本增强；
- 不确定性精修；
- 模型训练、调参与消融；
- 形成高精度骨骼分割模型及中期材料。

组会材料同时要求 7—8 月完成“系统提升 + 个人读懂系统代码/学习新技能”，9 月汇总，11 月准备中期检查，12 月进入论文集中撰写阶段。因此当前工作按“**先把系统和实验链做实，再用真实结果补论文**”推进。

---

## 2. 当前总体状态（2026-08-29）

| 模块 | 状态 | 完成度 | 当前真实状态 |
|---|---|---:|---|
| 任务书/组会材料梳理 | ✅ 已完成 | 100% | 已提取研究目标、时间轴、系统功能、论文/中期要求 |
| SegFormer3D 上游调研 | ✅ 已完成 | 100% | 已读 README、核心架构、loss、依赖与许可证；官方仓库已克隆到 `third_party/SegFormer3D` |
| 项目目录与交接机制 | ✅ 已完成 | 100% | 已建立工程目录和本主台账；明确“每次实质修改必须更新本文件” |
| 总体方案设计 | ✅ 已完成 | 100% | 已形成数据层、模型层、三维层、Web 层和实验追踪设计 |
| 国内外文献调研 | 🟡 进行中 | 97% | 已形成 44 条结构化文献矩阵与 42 条英文核心 BibTeX；已核验现代强 baseline、骨折、真实金属植入物以及低骨密度椎体 fusion/split 直接分割失败证据。近期直接脊柱链覆盖 Residual-Encoder nnU-Net、SpineMamba、解剖变异 Transformer、VertebraFormer 等；后续重点收敛到国内 CNKI/万方正式题录复核和根据正式任务筛选实际可跑 baseline |
| 实验环境 | ✅ 已完成（CPU 可训练环境） | 96% | 项目内 Python 3.11.7 + `.venv` 已完成；当前实测 Ryzen 7 8745H（8C/16T）、约 20 GB RAM、PyTorch `2.1.0+cpu`。真实 36³ patch 与 3-epoch binary engineering pilot 均已在本机 CPU 跑通；`train.py`/`formal_readiness.py` 新增显式 `--allow-cpu`，无 NVIDIA 不再是方法学硬 blocker。GPU 仅作为后续提速选项 |
| DICOM/CT 处理流程 | 🟡 进行中 | 96% | NIfTI pipeline 0.3.0 已在 10 例真实 CTSpine1K CT+label 上完成 1 mm 重采样、HU clip→case-wise z-score、骨窗、label nearest-neighbor、自动/交互 QC；10/10 自动审计通过。2026-08-26 复核 `manual_qc_review.csv`：10/10 四项人工检查均 `yes`、10/10 `pass`、reviewer 已填写，人工 QC P0 已解除；真实多层 DICOM series 仍待后续数据来源验证 |
| patient-level 数据划分 | ✅ 已完成（10例 formal pilot） | 96% | 已固定 `ctspine1k_msd_t10_binary_formal_pilot_v1.json`：7 train / 2 validation / 1 test，patient-level 互斥；官方 `test_private liver_169` 只进入 test、不参与训练/调参；`formal_experiment=true`。最终论文仍需扩大病例规模 |
| 公开数据集整理 | 🟡 进行中 | 94% | CTSpine1K `MSD-T10` 10 个真实 CT+label 已落盘：`liver_0`—`liver_8` + `liver_169`，官方 split 为 9 `trainset` + 1 `test_private`；真实文件接管执行 SHA-256 校验，10 例全部标准化/QC。该子集仍是工程验证，不替代正式论文主数据集/split |
| 临床脱敏数据 | 🔴 阻塞 | 0% | 当前项目目录无临床数据；必须等待合法授权、脱敏与伦理/使用范围确认 |
| SegFormer3D 骨科适配 | 🟡 进行中 | 96% | adapter、配置、dataset、训练骨架已完成；首个任务已锁定为 `binary_semantic`。v9 证明冻结 BN running stats 可显著缓解 v6 foreground explosion，但不能消除退化；v10 在 encoder 与 BN 全冻结后仍发生 mean Dice≈`1.65e-11` 的 catastrophic background collapse，否定 encoder parameter update 为必要条件。v11 从 epoch2 起同时冻结 encoder、BN running stats 与 decoder feature（`linear_c1..c4` + `linear_fuse`），仅允许 `linear_pred` 更新，并已完成 3 epoch：mean val Dice=`0.0540700072 → 0.0543761681 → 0.0546575740`，连续三轮无 catastrophic collapse。epoch3 `liver_7/liver_8` detailed Dice≈`0.04514/0.06417`、prediction/GT ratio≈`4.22/3.78`；新的 epoch1 exact-anchor→epoch3 dynamics 证明 encoder/BN/decoder-feature delta=`0`、fixed-patch encoder/fuse/head-input activation exact equal，仅 `linear_pred` 与 final logits 改变。与已保存的 epoch1→epoch2 dynamics 交叉验证后，正式判定 stable baseline=`YES`（engineering/validation）。绝对分割精度仍低，lock parameters=`NO`、formal independent test ready=`NO`；当前证据支持 decoder feature update 是 v10 collapse 的关键机制之一，但不写成唯一根因 |
| 区域损失 | ✅ 已完成（代码） | 90% | Dice + CE/BCE 可运行并有 backward 测试 |
| Boundary Loss | 🟡 进行中 | 85% | v13 已完成 3-epoch validation 消融；相对 Region 的 HD95/ASSD 仅约改善 0.1002/0.0355 mm，收益极弱，暂保留为 sampling baseline 候选但不宣称明确优势 |
| Topology Loss | 🟡 进行中 | 80% | v14/v15 已完成 validation 消融；结构/表面指标有改善信号，但 foreground overprediction 与 calibration 代价明显，当前不选作后续 baseline；骨折/非管状骨结构适用性仍待独立检查 |
| 困难样本增强 | 🟠 待真实验证 | 68% | 已落地可配置 3D flip/小角度旋转/各向同性缩放、gamma、Gaussian noise、HU shift 与 boundary-proxy hard sampling；强度增强已兼容 z-score CT 并使用 metadata 精确回到 HU 域。金属伪影与基于真实模型误差/uncertainty 的 hard mining 仍待实验 |
| 不确定性机制 | 🟡 进行中 | 90% | v13 validation `liver_7/liver_8` 已有真实 prediction + predictive entropy，并完成 uncertainty/calibration 两例稳定性分析：AUROC=`0.92949/0.94466`、AUPRC=`0.33354/0.33014`、Top-10% error recall=`0.72891/0.79450`；ECE=`0.01418/0.00767`、Brier=`0.05465/0.03969`、NLL=`0.12079/0.08571`。error voxel entropy 约为 correct voxel 的 `10.4×/12.4×`，支持 uncertainty 作为 error indicator/QC/refinement trigger 的 validation 信号；但仅 2 例，不能声称总体稳定或已完成 calibration。ROI refinement 仍待真实 validation |
| 训练/验证框架 | 🟡 进行中 | 99% | DataLoader/AdamW/AMP/gradient accumulation/sliding-window、scheduler、完整 run 追踪已接入；`train.py` 支持 `--allow-cpu`、可靠 `--resume` 与 `training.patches_per_case`。balanced v3 已实际使用 `validation.patch_mode=false`，逐 epoch 直接以 `liver_7/liver_8` full-volume Dice 选 checkpoint；epoch 1/2 已完成，说明 full-volume-aware selector 已进入真实训练闭环，不再依赖固定 foreground patch proxy |
| 评价指标 | ✅ 已完成（代码+首个真实 pilot test） | 100% | Dice、IoU、Precision、Recall、HD95、ASSD、component count/error、false merge/break 已接入；包含 uncertainty→error 与 ECE/MCE/Brier/NLL 等指标；`evaluate.py` 可统一写入逐病例 CSV 与 summary，并新增安全 `--case-id` 以支持 CPU 分病例 full-volume 执行，且强制病例必须属于当前 validation/test split。5-epoch formal-pilot 已对独立 `liver_169` 完成 full-volume CPU evaluation；该旧结果仅为 10 例流程 pilot，禁止作为论文正式结果 |
| Web 科研辅助分析原型 | 🟡 进行中 | 85% | 首页/上传/健康检查、MPR、10 例人工 QC reviewer、C1–L6 可读标签、真值 PLY WebGL2 3D、简化/物理测量均已完成；QC reviewer 已修复全站 `.card` grid-column 与 QC 网格冲突，病例选择后使用 `hidden + display:none!important` 彻底关闭病例层并进入主审核区，“上一例 / 下一例”保持审核区，悬浮按钮可随时重新打开病例列表；已在本机 Edge 对真实 `liver_0` 完成点击关闭/重新展开实机验证。另有 SDF surface 选择与 evaluation results-review，可读取未来 prediction/entropy MPR。当前 `/api/research/evaluations` 实测 200 且 total=0，真实 checkpoint/prediction 仍不存在，系统没有伪造结果 |
| 三维重建 | 🟡 进行中 | 86% | 已实现 physical-space Marching Cubes、PLY/JSON、vertex-clustering、SDF surface、WebGL2 与物理测量；新增相邻法向变化驱动的特征保护 vertex-clustering 候选，真实 `liver_0` 在 2.0 mm/同 30,260 顶点下将高特征区域 mean-NN 约 0.679→0.620 mm、HD95 约 1.068→1.000 mm，作为真值网格工程证据；0.4 mm SDF 保持 2→2 连通域，0.8 mm 因 2→3 被保护机制拒绝。仍缺真实 prediction surface 上的正式验证 |
| 论文 | 🟡 进行中 | 60% | 中文技术初稿已补 formal preflight、uncertainty/ROI refinement、calibration、物理表面/特征保护重建；Related Work 已加入 SpineMamba、解剖变异 Transformer、VertebraFormer、2026 Residual-Encoder nnU-Net、2025 骨折 pipeline、金属植入物与低骨密度 fusion/split 困难病例证据；42 条英文 BibTeX / 44 条矩阵已同步。Results 继续保持 TBD，禁止提前填结果 |
| 中期材料 | 🟡 进行中 | 83% | 已同步 10 例真实数据、97 项测试、10/10 人工 QC、正式 binary task lock、7/2/1 formal-pilot split、`formal_readiness ready=true`，并新增 CPU CT-only 5-epoch formal-pilot checkpoint；仍缺独立 full-volume test、扩大样本规模后的正式主实验与可写入论文的稳定指标 |
| 自动化测试/代码质量 | ✅ 已完成（当前阶段） | 100% | `pytest: 133 passed`；`ruff: All checks passed`；新增 decoder-feature freeze policy、仅保留 `linear_pred` trainable、恢复 trainability 与 v11/v10 单变量 config diff 回归测试；focused freeze tests=`15 passed`。同时保留 BatchNorm running-stat freeze、encoder freeze、fixed-per-case sampling、`region_dice_ce` 权重、`patches_per_case` 多 patch 随机流、foreground-fraction evaluation、checkpoint resume、分病例 full-volume evaluation、CPU 非 AMP autocast、epoch-aware sampling 与 `allow_cpu` readiness 测试；42 条 BibTeX 结构正常 |

---

## 3. 已验证的关键技术事实

### 3.1 上游 SegFormer3D

上游：`OSUPCVLab/SegFormer3D`，本地目录：

```text
third_party/SegFormer3D
```

当前上游基线提交：

```text
e314242
```

确认：

- 面向 3D volumetric segmentation；
- 四阶段分层、多尺度 Transformer encoder；
- 3D patch embedding；
- spatial reduction attention；
- all-MLP decoder；
- 官方示例主要为 BraTS、Synapse、ACDC；
- 官方 loss 主要为 CE/BCE/Dice/DiceCE；
- 本项目所需 DICOM/HU/骨窗、Boundary、Topology、不确定性精修、三维/Web 均属于需要自行适配/扩展的部分；
- 上游许可证为 GPL-3.0，第三方代码必须保留来源和许可边界。

### 3.2 上游本地兼容补丁

PyTorch 2.1 TorchScript 导入上游 `cube_root()` 时，会把 `round(float)` 视为 float，而函数返回标注为 `int`，导致 import 失败。

本地仅做以下语义不变兼容修复：

```python
return int(round(n ** (1.0 / 3.0)))
```

补丁位置：

```text
third_party/SegFormer3D/architectures/segformer3d.py
```

补丁说明：`third_party/README.md`。

### 3.3 SegFormer3D smoke test

在项目 `.venv` 的 PyTorch 2.1.0 CPU 环境完成前向：

```text
input_shape  = (1, 1, 64, 64, 64)
output_shape = (1, 2, 64, 64, 64)
params       = 4,492,066
```

**该测试只说明网络结构、适配器和当前依赖可以前向运行，不代表任何分割精度。**

### 3.4 当前开发环境

```text
项目内 Python      3.11.7
PyTorch            2.1.0+cpu
MONAI              1.2.0
pydicom            2.4.4
SimpleITK          2.3.1
nibabel            5.1.0
FastAPI            0.115.0
Lightning          2.0.9
PyTorch Lightning  2.0.9
setuptools          80.9.0（为 Lightning/pkg_resources 兼容固定 <81）
```

`uv pip ... --dry-run` 当前显示依赖一致、无需变更。

当前 Windows 系统侧没有检测到 `nvidia-smi`，因此不能据此确认存在可用于训练的 NVIDIA CUDA 环境；当前 `.venv` 明确为 CPU PyTorch。正式 3D CT 训练应迁移到已确认的 NVIDIA GPU/服务器环境，或重新运行环境脚本安装 CUDA 11.8 对应 PyTorch。

### 3.5 自动化测试

最终本轮验证：

```text
ruff check src web tests
→ All checks passed!

pytest tests -q
→ 94 passed

JSON / BibTeX / frontend structural checks
→ data/datasets.json OK
→ configs/label_schemas/ctspine1k_verse.json OK
→ configs/task_specs/vertebra_task_template.json OK
→ paper/references.bib: 42 entries, brace balanced, duplicate key none
→ app.js / qc_review.js / research_3d.js / results_review.js: node --check OK

PowerShell parser
→ 当前 CI/本地纳入检查的 PowerShell 脚本语法均通过
```

测试覆盖：

- HU clip / normalize；
- bone window；
- DICOM 按物理几何位置排序；
- NIfTI image/label 物理空间一致性、nearest-neighbor 标签重采样与标准输出；
- VerSe CT/mask 自动配对与官方 split 识别；
- joint loss forward/backward；
- SegFormer3D adapter；
- Dice/IoU/HD95/ASSD；
- predictive entropy / uncertainty ROI / error AUROC/AUPRC / Top-percent 定量指标；
- calibration ECE/MCE/Brier/NLL/mean confidence/accuracy/confidence gap 与固定 seed 体素采样；
- ROI-only refinement training / coarse freeze / error delta；
- formal/engineering preflight 与 train/evaluate 默认保护；
- multiclass per-class 输出与空类别宏平均防虚高；
- Web health/index/MPR/QC reviewer/真值 overlay/真值 3D/测量；
- 三视图 / 骨窗 / label overlay QC contact sheet；
- QC 人工审核 CSV 模板；
- CTSpine1K 官方 split 解析、image/label 配对、标准化与可选 QC；
- VerSe 批处理 `--qc` 集成；
- 合成标准病例 Dataset→SegFormer3D→loss→backward→AdamW.step；
- 真实 CTSpine1K 双通道前景 patch→SegFormer3D→joint loss→backward→AdamW.step；
- 数据增强、scheduler、独立 checkpoint evaluation；
- connected component/false merge/false break；
- mask→physical-space mesh、NIfTI→PLY、vertex-clustering 简化；
- 10 例 raw→1 mm label 重采样 physical-surface 几何误差；
- SimpleITK Windows 中文项目路径兼容；
- CTSpine1K/VerSe 椎体标签 schema 与真实 Web/QC 可读显示。

### 3.6 DICOM smoke test

使用 pydicom 自带 `CT_small.dcm` 做单切片处理，成功输出：

```text
image_normalized.nii.gz
image_bone_window.nii.gz
metadata.json
qc.json
```

该样本仅有 1 个切片，pipeline 正确给出 warning；因此这只能验证读取/输出链路，**不能替代真实多层 DICOM series 的几何排序和 spacing 验证**。

### 3.7 Web smoke test

FastAPI `TestClient`：

```text
GET /api/health → 200
GET /            → 200 text/html
```

健康接口当前返回：

```text
research_only = true
inference_ready = false
model_checkpoint_count = 0
```

`/infer` 在没有真实 checkpoint 时故意返回 `501 not_ready`，避免用随机权重伪造“诊断/分割结果”。

### 3.8 公开 NIfTI / VerSe 接入验证

2026-08-16 新增：

```text
src/preprocessing/nifti_pipeline.py
src/preprocessing/prepare_verse.py
data/datasets.json
env/download_verse.ps1
docs/06_public_dataset_onboarding.md
```

已验证：

- 合成 3D CT + 多类别 label 可完成 1 mm 重采样；
- image 用 linear、label 用 nearest-neighbor；
- 原始 image/label 的 size/spacing/origin/direction 不一致时直接拒绝；
- label 重采样后不会生成原本不存在的类别；
- VerSe 命名的 CT 与 `seg-vert` mask 能自动一一配对；
- split 识别只按目录段精确匹配，避免路径中偶然包含 `test` 等字符串造成误判；
- 同 patient group 跨 source split 时会拒绝继续；
- `download_verse.ps1` 默认仅展示下载计划，必须显式加 `-Download` 才会下载大型归档。

VerSe/TotalSegmentator 仍属于工程链路验证；CTSpine1K 已于 2026-08-16 实际落盘 `MSD-T10` 10 个 CT+label：`liver_0`—`liver_8`、`liver_169`。官方 split 为 9 例 `trainset` + 1 例 `test_private`；10 例已完成 pipeline 0.3.0 标准化、contact sheet 和自动审计。该方便子集用于真实工程/QC，不等同于正式论文 split。

### 3.9 CTSpine1K 备用接入与人工 QC 工具

2026-08-16 进一步补齐当前网络阻塞下的备用链路：

```text
env/download_ctspine1k_sample.ps1
src/preprocessing/prepare_ctspine1k.py
src/preprocessing/qc_visualization.py
```

已验证：

- CTSpine1K Hugging Face 镜像的 `raw_data/volumes/<sub-dataset>` 与 `raw_data/labels/<sub-dataset>` 结构可用于按病例配对；
- 默认下载计划仅列出 `MSD-T10` 的 `liver_169`、`liver_0`、`liver_1` 三个小样本及对应 label，不显式 `-Download` 不会下载；
- `prepare_ctspine1k` 能解析官方 `data_split.txt` 中 `trainset / test_public / test_private` 标记，但不会擅自重解释为 validation/test；
- VerSe 与 CTSpine1K 批处理均可使用 `--qc` 生成逐例 `qc_contact_sheet.png`；
- `qc_visualization` 可批量生成/刷新 QC 图，并输出 `manual_qc_review.csv` 与 `qc_visualization_summary.json`；
- 合成 NIfTI 已验证前景标签驱动的三视图选层、骨窗显示、label overlay 与批量审核清单生成；
- 真实 10 例已生成 contact sheet，`manual_qc_review.csv` 共 10 行，人工字段留空待签字；
- `audit_processed` 对 10 例 pipeline/spacing/geometry/label/normalization 自动审计：10/10 pass；
- 真实病例原始 z-spacing 覆盖约 0.8 / 1.0 / 5.0 mm，重采样后均为 1 mm。

网络状态已发生变化：VerSe S3 仍未验证恢复；CTSpine1K Hugging Face 则通过浏览器**单文件顺序下载**成功完成 10 例。并行请求曾产生 `无法下载`，`liver_3/5/8` 顺序重试后成功。接管到项目的文件执行 SHA-256 源/目标一致性检查；因此当前阻塞已从“无真实数据”转为“正式任务/split/GPU baseline 未完成”。

---

## 4. 第一版技术方案（当前执行基线）

### 4.1 数据流程

```text
DICOM/NIfTI
→ 数据授权/脱敏检查
→ Study/Series 识别
→ IOP/IPP 几何排序与 QC
→ HU 恢复/强度统计
→ orientation 统一
→ spacing 重采样
→ HU clip + normalize
→ bone-window 通道
→ label 同步 nearest-neighbor 重采样
→ metadata.json + qc.json + NIfTI cache
→ patient-level split
```

当前重采样 `1.0×1.0×1.0 mm`、HU clip `[-1000, 2000]`、bone window center/width `500/2000` 仅为**实验初始配置**，不是预先认定的最佳临床参数，必须通过具体部位数据统计和消融确定。

### 4.2 模型实验顺序

必须逐项验证，禁止一次堆满模块后无法解释收益：

1. `B1`：SegFormer3D + 单标准化 CT + Region loss；
2. `B2`：CT + bone-window 双通道；
3. `L1`：Region + Boundary；
4. `L2`：Region + Topology；
5. `L3`：Region + Boundary + Topology；
6. `H`：加入困难样本增强/hard sampling；
7. `U`：加入 uncertainty map 与 ROI refinement；
8. 外部数据/来源泛化；
9. 失败病例和困难子集分析。

联合损失：

```text
L_total = λ_region·L_DiceCE
        + λ_boundary·L_boundary
        + λ_topology·L_topology
```

权重只在 validation 上选择，不用 test 反复调参。

### 4.3 拓扑约束的医学例外

soft-clDice 只是首个候选，并非“骨结构必然有效”。尤其对于骨折：

- 真值中的断裂可能是真实病理状态；
- 如果 topology loss 强行把断端连接，反而会产生临床错误；
- 因此需要普通病例/骨折病例分层评价；
- 如开展多椎体/多骨任务，可进一步评估 Betti matching / persistent-homology 类方法。

### 4.4 Web 阶段路线

当前已实现：

```text
上传 → 基础识别/QC → axial/coronal/sagittal MPR
```

下一阶段：

```text
真实 checkpoint 推理
→ mask overlay
→ uncertainty overlay
→ mask→mesh
→ 3D 旋转/透明/多骨开关
→ 距离/角度测量
→ 人工校核
```

网站定位始终为**科研/辅助分析原型**，在没有医疗器械合规与临床验证前，不写“自动诊断疾病”或替代医生结论。

---

## 5. 文献调研当前结论

`docs/02_literature_survey.md` 已整理首批国内外工作，主要包括：

- SegFormer / SegFormer3D；
- UNETR / nnFormer / Swin UNETR；
- CTSpine1K；
- VerSe 2019/2020；
- TotalSegmentator；
- VerFormer 2024；
- Boundary Loss；
- clDice；
- 2024 多类别 Betti matching；
- TEDS-Net 2024；
- EDUE 2024；
- UCTNet 2024；
- 国内肋骨 CT 三维分割与三维重组研究；
- 国内椎体转移瘤 CT 2D/3D U-Net/ResUNet 对比研究。

当前论文切入点建议保持为：

**“面向骨科 CT 的轻量 3D SegFormer：标准化/骨窗多尺度输入 + 区域—边界—拓扑联合约束 + 困难样本学习 + 不确定性驱动局部精修，并验证其对三维重建所关心的表面质量和结构错误的影响。”**

不能把“首次使用深度学习/首次使用 Transformer 做骨分割”作为创新表述；现有文献已经存在大量相关工作。

---

## 6. 论文当前状态

论文文件：

```text
paper/outline.md
paper/manuscript_zh_v0.1.md
paper/references.bib
```

已完成：

- 论文研究问题；
- Introduction 初稿；
- Related Work 初稿；
- 数据标准化方法描述；
- SegFormer3D 方法描述；
- 联合损失方案；
- hard sample 方案；
- uncertainty ROI + uncertainty→error 定量评价 + ROI-only refinement 方案；
- physical-space mesh / 重采样几何控制方法描述；
- formal preflight 与 multi-class 评价安全规则；
- 37 条结构化文献矩阵 + 35 条英文核心 BibTeX；
- 实验/消融设计；
- 结果表格模板；
- limitation 预留。

**Results 目前保持 TBD。**

只有当 `experiments/<run_id>/` 内存在可追溯的：

```text
config.yaml
split.json
history.csv / train.log
checkpoint
metrics_per_case.csv
summary.json
```

并完成独立测试后，才允许把数字写进论文。

---

## 7. 关键文件索引

| 文件/目录 | 用途 | 当前状态 |
|---|---|---|
| `PROJECT_STATUS.md` | 唯一主进度台账/交接入口 | ✅ |
| `README.md` | 项目入口、目录、环境、协作与公开仓库运行原则 | ✅ |
| `TASKS.md` | 多人协作总任务看板：已完成/待完成/阻塞/推荐分工/DoD | ✅ 新增 |
| `CONTRIBUTING.md` | 分支、提交、PR、测试、正式实验与医学数据协作规范 | ✅ 新增 |
| `SECURITY.md` | 公开仓库隐私、安全、科研结果边界与第三方依赖说明 | ✅ 新增 |
| `.github/` | Task/Bug Issue 模板与 Pull Request 检查模板 | ✅ 新增 |
| `docs/01_overall_design.md` | 总体架构、模块、接口与质量保证 | ✅ |
| `docs/02_literature_survey.md` | 国内外文献、数据集、研究空白 | 🟡 v0.2，已与结构化矩阵/BibTeX 同步 |
| `docs/08_literature_matrix.md` | 44 条 3D 分割/脊柱/困难病例/损失/uncertainty/重建结构化文献矩阵 | ✅ 已更新强 baseline/骨折/金属植入物/低骨密度工作 |
| `docs/03_data_pipeline_spec.md` | DICOM/HU/spacing/bone-window/QC SOP | ✅ 首版 |
| `docs/04_experiment_plan.md` | baseline、联合损失、困难样本、uncertainty 消融矩阵 | ✅ 首版 |
| `docs/05_midterm_materials.md` | 中期研究材料、已有证据、缺项、展示建议 | ✅ 首版 |
| `docs/06_public_dataset_onboarding.md` | VerSe/CTSpine1K/TotalSegmentator 登记、下载、10 例 QC 与 baseline 接入 SOP | ✅ 已更新真实状态 |
| `docs/07_real_data_validation_20260816.md` | CTSpine1K 10 例真实落盘、pipeline 0.3.0、审计、patch smoke、真实 mesh 证据 | ✅ |
| `docs/09_public_repository_manifest.md` | 公开 GitHub 仓库纳入/排除文件、医学数据隐私与提交前检查清单 | ✅ 新增 |
| `paper/outline.md` | 论文持续写作框架 | ✅ |
| `paper/manuscript_zh_v0.1.md` | 中文论文技术初稿，Methods/Experiment Design 持续补强，Results 保持 TBD | 🟡 |
| `paper/references.bib` | 42 条英文核心机器可用 BibTeX，已做 key/括号结构检查 | ✅ |
| `env/requirements.txt` | 固定项目依赖 | ✅ |
| `env/setup_env.ps1` | 项目内 Python 3.11/.venv 环境搭建；优先 uv | ✅ |
| `env/fetch_segformer3d.ps1` | 获取官方 SegFormer3D | ✅ |
| `env/download_verse.ps1` | VerSe 2019/2020 下载计划与显式下载辅助；默认不下载 | ✅ |
| `env/download_ctspine1k_sample.ps1` | CTSpine1K MSD-T10 小样本 CT+label 下载计划与显式下载；默认不下载 | ✅ |
| `env/check_gpu.ps1` | 项目 GPU/CUDA/PyTorch 只读验收入口 | ✅ 本机正确报告 CPU/no CUDA |
| `env/check_formal_readiness.ps1` | task + GPU + formal preflight 一站式验收入口 | ✅ 当前阻塞正确返回 exit 2 |
| `third_party/README.md` | 上游许可证边界和本地兼容补丁说明 | ✅ |
| `third_party/SegFormer3D/` | 官方上游代码，基线 `e314242` + 1 个本地兼容补丁 | ✅ |
| `configs/orthopedic_ct_baseline.yaml` | 单 CT + Region baseline | ✅ 配置完成 |
| `configs/orthopedic_ct_joint.yaml` | CT+bone-window + joint loss + uncertainty 实验配置 | ✅ 配置首版 |
| `configs/label_schemas/ctspine1k_verse.json` | CTSpine1K/VerSe `1–25 → C1–L6` 工程显示 schema；不锁定正式任务 | ✅ |
| `configs/task_specs/vertebra_task_template.json` | 正式任务锁定模板；默认 `task_locked=false` | ✅ 保护模板 |
| `src/label_schema.py` | 标签 schema 读取、人类可读名称与 Web/QC 输出 | ✅ real display pass |
| `src/sitk_compat.py` | SimpleITK 项目内相对路径兼容层，规避 Windows 中文绝对路径 I/O 问题 | ✅ real pass |
| `src/preprocessing/dicom_pipeline.py` | DICOM series/QC/显式几何排序/重采样/标准化/骨窗/输出 | 🟠 待真实 series |
| `src/preprocessing/nifti_pipeline.py` | 公开 NIfTI image/label 几何校验、重采样、标准化与训练标准输出 | ✅ 10 例真实 CTSpine1K 验证 |
| `src/preprocessing/prepare_verse.py` | VerSe CT/mask 配对、source split、防患者泄漏、批量预处理，可选 `--qc` | ✅ 工程验证，🟠 待真实数据 |
| `src/preprocessing/prepare_ctspine1k.py` | CTSpine1K CT/mask 配对、官方 split 标记、批量标准化，可选 `--qc` | ✅ 10 例真实处理通过 |
| `src/preprocessing/qc_visualization.py` | 三视图 × normalized/bone-window/label-overlay QC 图与人工审核 CSV | ✅ 10 例真实数据已生成，人工字段待签字 |
| `src/preprocessing/audit_processed.py` | 标准化病例 pipeline/geometry/spacing/label/normalization 自动审计 | ✅ 10/10 real pass |
| `src/preprocessing/create_split.py` | patient-level split、防重复患者泄漏 | ✅ |
| `src/modeling/segformer3d_adapter.py` | 官方 SegFormer3D 与本项目配置适配 | ✅ |
| `src/modeling/dataset.py` | 标准化 NIfTI、多通道、3D patch dataset | ✅ 首版 |
| `src/modeling/joint_loss.py` | Region + Boundary + soft-clDice | 🟠 待消融 |
| `src/modeling/metrics.py` | Dice/IoU/Precision/Recall/HD95/ASSD | ✅ |
| `src/modeling/uncertainty.py` | entropy、ROI、uncertainty-error overlap/AUROC/AUPRC/Top-percent 定量指标 | ✅ 工程实现，🟠 待真实模型验证 |
| `src/modeling/refinement.py` | uncertainty ROI 局部残差 3D 精修网络与 ROI 融合 | ✅ 工程实现，🟠 待真实消融 |
| `src/modeling/refinement_training.py` | coarse 冻结 + ROI-normalized 二阶段精修 loss/step/error delta | ✅ 工程实现，🟠 待真实 checkpoint |
| `src/modeling/preflight.py` | formal/engineering 实验前置验收与泄漏/人工QC/GPU/标签配置保护 | ✅ real engineering/formal 拦截验证 |
| `src/modeling/task_lock.py` | 锁定 binary/multiclass semantic 任务并编译带 SHA-256 指纹的正式 config；拒绝未实现 instance | ✅ targeted + full regression pass |
| `src/modeling/gpu_environment.py` | PyTorch CUDA/device/显存/`nvidia-smi` 只读验收 | ✅ 本机 CPU blocker 已实测 |
| `src/modeling/formal_readiness.py` | 汇总 task/GPU/formal preflight/config binding 的一站式正式实验验收 | ✅ 当前 real smoke override 返回 9 blockers |
| `src/modeling/train.py` | 训练、验证、scheduler、checkpoint、固定 split/config/环境/train.log | 🟠 待 GPU 正式训练 |
| `src/modeling/evaluate.py` | checkpoint sliding-window 独立评估、per-case/per-class、uncertainty 定量指标/prediction/entropy 输出 | ✅ 工程实现，🟠 待真实 checkpoint |
| `src/modeling/real_patch_smoke.py` | 真实标准化病例双通道 joint-loss 单 patch forward/backward 工程验收 | ✅ real pass |
| `src/reconstruction/mesh.py` | mask→物理空间 Marching Cubes + vertex-clustering + 法向变化加权特征保护候选 | ✅ real-label engineering pass |
| `src/reconstruction/export_mesh.py` | NIfTI label/prediction→全分辨率/简化 PLY+JSON 可追溯导出 | ✅ real-label pass |
| `src/reconstruction/resampling_error.py` | 原始 label vs 1 mm label physical-surface 重采样几何误差 | ✅ 10/10 real pass |
| `src/reconstruction/sdf_surface.py` | physical-mm signed-distance smoothing + zero-level MC + 连通域保护 | ✅ real `liver_0` sweep/Web pass |
| `src/reconstruction/measurement.py` | 物理坐标距离/三点夹角与 voxel→physical 工具 | ✅ |
| `web/backend/app.py` | FastAPI 本地科研服务：上传/MPR/QC reviewer/真值3D+SDF/results-review/测量/推理占位 | 🟡 |
| `web/frontend/` | 上传/QC、交互 MPR+overlay、QC 病例栏折叠/审核区聚焦、WebGL2 真值 3D/SDF、results-review、测量前端 | 🟡 |
| `web/run_web.ps1` | localhost Web 启动脚本 | ✅ |
| `tests/` | 自动化测试 | ✅ 94 passed |
| `data/README.md` | 数据治理与隐私规则、当前真实 CTSpine1K 子集说明 | ✅ |
| `data/datasets.json` | 公开数据集来源/版本/许可/本地状态；已登记 10 例 CTSpine1K 工程子集 | ✅ |
| `data/splits/ctspine1k_msd_t10_engineering_smoke.json` | 1/1/1 真实数据工程 smoke split，明确 `formal_experiment=false` | ✅ 非正式实验 |
| `.gitignore` | 排除医学数据、处理后数据、环境、模型、runtime、第三方 checkout、DevSpace 缓存和大型生成 PPT | ✅ 已加强公开安全规则 |

---

## 8. 当前阻塞与风险

### R1｜真实工程数据已落盘，但正式论文数据方案尚未固定

CTSpine1K `MSD-T10` 已有 10 例真实 CT+label 完成落盘和 pipeline 0.3.0 标准化，自动审计 10/10 pass；其中 9 例官方 `trainset`、1 例 `test_private`。这已经解除“完全没有真实数据”的工程阻塞，但**不能直接把该方便子集当正式论文 split**。当前仍需：

- 组内确定 binary / multi-class / instance 等正式标签任务；
- 确定主数据集/官方 split 或预注册内部 split；
- 完成 10 例人工 QC 签字；
- 在 NVIDIA GPU 上跑 baseline；
- 之后才能产生可写入论文的 DSC/HD95/ASSD、调参/消融和 Web 真实推理结果。

### R2｜临床数据授权/伦理

临床数据必须满足：

- 已脱敏；
- 合法授权；
- 研究范围明确；
- 不把患者姓名、身份证号、联系方式写入日志/文件名/截图；
- 如学校/医院要求伦理审批，先完成审批再用于研究。

### R3｜GPU 训练条件

当前项目环境为 PyTorch CPU。`src.modeling.gpu_environment` 已实测：PyTorch `2.1.0+cpu`、`torch.version.cuda=None`、`cuda_available=false`、0 个 CUDA device、无 `nvidia-smi`。3D CT 正式训练需要确认：

- NVIDIA GPU；
- 驱动；
- CUDA/PyTorch 匹配；
- 可用显存；
- 训练存储空间。

当前 `nvidia-smi` 未检测到只能说明本机当前没有可见 NVIDIA 工具链，**不能据此推断学校服务器或其他设备也没有 GPU**。

### R4｜主任务/标签范围尚未最终定稿

建议首篇论文优先：

```text
脊柱/椎体 CT
```

理由：CTSpine1K/VerSe 数据、骨边界、相邻骨粘连/断裂、跨来源泛化与三维重建问题都较契合当前创新设计。

但最终仍需组内确认是：

- 整体脊柱 binary；
- 单椎体/多椎体 semantic；
- vertebra instance；
- 或其他骨科部位。

标签定义一旦确定，loss、topology、指标、数据集和 Web 测量都会随之固定。当前已建立 `configs/label_schemas/ctspine1k_verse.json`，仅把 `1–25` 工程显示为 `C1–L6`，用于 QC/3D 可读性；另已建立 `configs/task_specs/vertebra_task_template.json` + `task_lock.py`，模板默认 `task_locked=false`，未锁定时拒绝编译正式 config，instance 任务因当前链未实现而明确拒绝。**不能把显示 schema 或模板当成正式任务已经定稿**。

### R5｜Topology 与真实骨折的冲突

不能预设“连通越好越正确”。骨折可能真实断裂。必须：

- 分层病例；
- 保留病理形态；
- 不让 topology loss 把断端错误粘连；
- 用真实标签与失败案例验证。

### R6｜开源代码与软著原创性

SegFormer3D 为 GPL-3.0 上游。不能：

- 批量复制后换变量名冒充原创；
- 在软著材料中隐瞒来源；
- 把上游 backbone 写成“本项目自主提出”。

本项目可明确自研的部分包括：骨科 DICOM/CT 数据流程、QC、骨窗/多通道适配、联合损失实现与验证、uncertainty/refinement、困难样本策略、三维/Web 系统及工程集成。

### R7｜“诊断网站”表述风险

当前系统应称：

**骨科 CT 智能辅助分析研究平台 / 科研原型**。

在无合规验证前，不应宣传为能够独立给出疾病诊断结论的临床诊断产品。

### R8｜公开数据下载网络/客户端稳定性

2026-08-16 当前本机访问 VerSe 2020 官方 S3 归档仍存在连接超时，因此 VerSe 尚未实际落盘。

CTSpine1K Hugging Face 在早期也出现超时和并行下载失败；但改为 Edge 浏览器**单文件顺序下载**后，已成功取得 `MSD-T10` 10 个 CT+label。`liver_3/5/8` 等失败病例顺序重试后成功，复制到项目时执行 SHA-256 源/目标一致性检查。结论是：CTSpine1K 当前可通过浏览器顺序方式继续扩量，但命令行/并行下载稳定性仍不可假定；后续大规模下载必须保留 provenance、校验和与断点恢复策略。

---

## 9. 下一步任务——必须按优先级执行

### P0｜当前第一优先：修复 full-volume checkpoint selection 与 baseline

- [x] 首个正式任务已锁定：`vertebra_binary_ctspine1k_msd_t10_v1`，binary semantic，2 类；
- [x] 10/10 人工 QC 已完成，7 train / 2 validation / 1 test patient-level split 已固定；`liver_169` 仅允许最终独立 test；
- [x] 64³ CT-only long-v2 已完成并 early-stop：`best.pt=epoch 1`、`last.pt=epoch 9`；
- [x] 已对 `liver_7/liver_8` 完成 `best.pt` 与 `last.pt` 四次 full-volume validation，输出 `metrics_per_case.csv` / `summary.json` 均已核对；
- [x] `best.pt` 两例平均 Dice≈0.03698；`last.pt`≈0.04953。`last.pt` 平均 ASSD≈50.78 mm、component count error≈1084，优于 `best.pt` 的≈56.77 mm / 1617；
- [x] 已确认固定单 patch validation 严重高估/误判 full-volume 泛化：epoch 1 patch-val≈0.3613，但 full-volume 平均仅≈0.037；
- [x] 已确认首要根因是 foreground/background sampling prior 严重失配：long-v2 真实训练 patch 平均前景≈21.2%，7 个 train 全卷平均≈0.68%；两例 validation prediction 前景≈14.5%–17.1%，是真值≈0.57%–0.70% 的约 24–27 倍；
- [x] 已实现 `training.patches_per_case`：单病例每 epoch 可抽多个独立可复现 patch；并在 evaluation CSV/summary 增加 prediction/target foreground fraction 与 ratio；
- [x] 已新建 `configs/orthopedic_ct_cpu_binary_balanced_fullval_v3.yaml`：foreground_probability=0.25、patches_per_case=4、64³ CT-only、Region Dice+CE 保持不变，`validation.patch_mode=false`；`formal_readiness --allow-cpu` 实测 ready=true / blocker_count=0；
- [x] balanced v3 已真实完成 epoch 1/2：epoch 1 full-volume val Dice≈0.05407、epoch 2≈0.04084，当前 `best.pt=epoch 1`；
- [x] 已对 v3 `best.pt` 分病例 detailed validation：`liver_7/liver_8` Dice≈0.04323/0.06491，Precision≈0.02753/0.04267，prediction/target foreground ratio≈3.65/3.18；相对 long-v2 约 24–27 倍已显著改善；
- [x] v3 epoch 3 已续训并明确失败：train loss≈1.63162，但两例 full-volume val Dice≈1.3e-11；detailed validation 两例 Dice/Precision/Recall=0，prediction/GT foreground ratio≈0.47/0.26，已停止继续 epoch 4；
- [x] 根因检查发现 RegionDiceCELoss3D 当前为 foreground Dice + 未加权全体素 CE 默认 1:1，且 `train.build_criterion()` 未读取 YAML 的内部 `dice_weight/ce_weight`；该工程缺口已修复并形成 v4 单变量实验；
- [x] v4 将 CE 权重降至 0.25 后两例平均 Dice≈0.04762、foreground ratio≈5.97、component error≈1993，整体劣于 v3 epoch 1，已否定“继续降低 CE 权重”方向；
- [x] v5 将 peak lr 降至 5e-5 但保留 2-epoch warmup，epoch 1/2 分别 Dice≈0.03185/0.03269，detailed validation 约 55× foreground explosion，已停止；
- [x] v6 仅将 warmup 2→1：epoch 1 train loss=`2.5537127597`、val Dice=`0.0540700072`、lr=`5e-5`，几乎精确复现 v3 epoch 1；epoch 2 train loss=`1.9332212380`、val Dice=`0.0323937293`、lr≈`4.8923e-5`，即使未升到 1e-4 仍明显恶化；
- [x] v6 epoch 2 两例 detailed validation：`liver_7/liver_8` Dice≈0.03210/0.03268、Precision≈0.01632/0.01661、Recall≈0.98562/0.99919、prediction/GT foreground ratio≈60.40/60.14、component error=87/65；这是大范围背景被预测成前景造成的严重 foreground explosion，不是 component 数下降带来的正确改善；
- [x] 已使用 Dataset 真实 sampling 逻辑与固定 seed=42 复现 v3/v6 epoch 1/2、v3 epoch 3 的 28 个 training patch：epoch 1/2/3 mean foreground fraction≈7.91%/8.84%/5.68%，median 均为 0，纯背景 patch=18/18/20；病例级暴露明显不稳定，例如 epoch 1 `liver_2/liver_6` 均 4/4 patch 纯背景，epoch 2 各病例又重新分配。说明当前独立 Bernoulli sampling 存在真实 epoch/case 波动，但 v6 epoch 1→2 的总体差异并不足以单独解释约 3.4×→60× foreground explosion，因此 sampling 只能视为已证实的稳定性问题/候选诱因，不是已证实唯一根因；
- [x] `train.py` 已新增 `sampling_stats.csv`，直接从模型实际收到的 training label 每 epoch 记录 patch_count、foreground fraction mean/median/std/min/max、q10/q25/q75/q90、foreground/background patch count，并新增回归测试；
- [x] v7/v8/v9/v10 已按单变量稳定性路线完成并形成机制证据；其中 v9 证明 BN running-stat drift 是 foreground explosion 的重要放大机制但不是唯一根因，v10 证明 encoder parameter update 不是 epoch2 degradation 的必要条件；
- [x] v11 工程已完成：新增 `training.freeze_decoder_feature_parameters_from_epoch=2`；epoch2 起冻结 decoder `linear_c1..c4` + `linear_fuse`，仅保留 `linear_pred` head 可训练；相对 v10 除 experiment name 与这一新增 freeze 配置外完全一致；focused freeze tests=`15 passed`、全量 `pytest=133 passed`、Ruff clean、formal readiness=`ready=true / blocker_count=0`；
- [x] v11 epoch1 已完成并与 v10 epoch1 exact equal：run=`experiments/20260827_180730_cpu_binary_decoder_feature_freeze_after_e1_v11_roi64`，train loss=`2.5537127597`、mean val Dice=`0.0540700072`、std=`0.0108403799`、lr=`5e-5`，sampling 28 patch、foreground/background=`10/18`；v10e1↔v11e1 的 232 个 model-state tensor 逐项 `torch.equal`、diff=`0`，因此确认工程未污染 epoch1。为避免重复昂贵 CPU evaluation，不复跑与 exact-equal checkpoint 等价的 detailed validation/diagnostics，沿用 v10e1 锚点；
- [x] v11 已从同一 run resume 到总 epoch2：train loss=`2.3053811001`、mean val Dice=`0.0543761681`、std=`0.0101640915`、lr≈`4.8923e-5`；sampling 28 patch、foreground/background=`10/18`、foreground fraction mean≈`0.08840765`；checkpoint 证明 encoder delta=`0`、BN running buffer delta=`0`、decoder feature delta=`0`，仅 `linear_pred` weight+bias 发生更新；
- [x] v11 epoch2 `liver_7/liver_8` detailed validation、diagnostics 与 v11e1→v11e2 dynamics 已完成：两例 Dice≈`0.04421/0.06454`、foreground ratio≈`3.96/3.50`；GT foreground mean P(fg)≈`0.13263/0.16114`，GT background mean P(fg)≈`0.03482/0.02670`；固定 `liver_7` 上 encoder、decoder fuse 与 final-head input activation 统计完全一致，仅 final logits 随 final head 更新而变化。当前允许继续 epoch3；
- [x] v11 epoch3 已真实完成且不重跑：train loss=`1.8300107228`、mean full-volume val Dice=`0.0546575740`、std=`0.0095167619`，三轮 Dice=`0.05407001 → 0.05437617 → 0.05465757`；`liver_7/liver_8` detailed Dice≈`0.04514/0.06417`、foreground ratio≈`4.22/3.78`。使用 v11e1 exact anchor→v11e3 新 dynamics + 已保存 v11e1→v11e2 dynamics 交叉验证，encoder/BN/decoder-feature 持续冻结，仅 final head 更新；stable baseline=`YES`（engineering/validation），lock parameters=`NO`、formal independent test ready=`NO`；
- [ ] 继续核对 Region Dice+CE 背景抑制、label mapping、normalization、sliding-window stitching/logits resize/threshold；当前没有发现 label mapping 或 resize 的直接错误证据；
- [x] stable CT-only baseline 已锁定为 v11 机制基线；最小可信 reproducibility、CT-only vs CT+bone-window 输入消融及 v11/v13/v14/v15 loss ablation 均已完成；loss 阶段选择 v13 Region+Boundary 作为后续 sampling baseline，所有选择仍只使用 train+validation；
- [ ] 在 ROI/epoch/lr/scheduler/sampling/augmentation/input/loss/checkpoint 全部只依据 train+validation 锁定前，禁止重新运行 test `liver_169`；
- [ ] 更可靠 baseline 锁定后再生成 prediction mesh / SDF / Web overlay / entropy overlay，并继续论文工程验证材料。

### P1｜联合损失与困难样本消融

- [x] Region：v11；
- [x] Region + Boundary：v13；
- [x] Region + Topology：v14；
- [x] Region + Boundary + Topology：v15；
- [ ] loss 权重 validation grid（当前最小四组消融已完成，后续是否继续 grid 以 validation 证据与 CPU 成本决定）；
- [ ] normal vs difficult subset；
- [ ] fracture/metal/low-density/thick-slice 子集（数据存在时）；
- [ ] 记录 false merge / false break。

### P2｜不确定性精修

- [x] 已实现 entropy→error AUROC/AUPRC、错误/正确平均 entropy、Top-percent error recall、ROI error rate/fraction 的定量评价代码；
- [ ] 在真实 baseline checkpoint 上验证 entropy 与真实错误空间相关性；
- [ ] 用 validation 确定 Top-percent/threshold；
- [x] 已实现 `UncertaintyRefinementNet3D` 局部残差 refinement head/network（工程代码）；
- [x] 已实现 coarse 冻结、ROI-normalized loss、ROI/global error delta 的二阶段 refinement 训练基线；
- [ ] 对比无精修 vs 全图二次推理 vs uncertainty ROI；
- [ ] 报告额外时间/显存/ROI 比例；
- [ ] 评估 uncertainty 是否可用于 QC。

### P3｜三维重建与 Web

- [x] mask → physical-space surface；
- [x] Marching Cubes baseline + PLY/JSON 导出；真实 `liver_0` label 已验证；
- [x] SDF physical-surface engineering baseline：真实 `liver_0` 0.3/0.4/0.5/0.8 mm sweep；0.4 mm 当前默认候选，0.8 mm 因改变连通域被拒绝；
- [x] 已完成 10 例原始 label→1 mm label 的重采样/各向异性 physical-surface 工程误差评估；
- [x] 曲率/关键边缘保护候选：相邻顶点法向变化加权 vertex-clustering；真实 `liver_0` 真值网格工程验证显示高特征区域误差下降，仍待 prediction 验证；
- [x] vertex-clustering mesh 简化；真实 `liver_0` 1.5 mm 档约减 60% 顶点/面，保留全分辨率基准；
- [x] MPR 三视图（axial/coronal/sagittal + 切片位置/窗宽窗位）；
- [x] WebGL2 真值 label 3D 渲染与全分辨率/1.5/2.0 mm 选择；
- [x] 椎体类别显示：`1–25 → C1–L6` 工程 schema，不改原标签值；
- [x] 物理 XYZ 距离/三点夹角计算 API；
- [x] 10 例人工 QC reviewer + 交互 MPR/真值 overlay 接口；病例列表支持选择后自动收起、随时展开、自动进入主审核区，上一例/下一例保持审核区，宽屏/窄窗口均有对应布局；
- [x] evaluation results-review 页面与 prediction/entropy MPR 接口已准备；当前真实 evaluation=0；
- [ ] 取得真实 checkpoint 后生成 prediction / uncertainty / prediction mesh 并在 Web 展示正式结果。

### P4｜论文/中期/软著

- [x] 已建立 44 条结构化文献矩阵；已补 SpineMamba、2025 解剖变异 Transformer、2026 VertebraFormer、2026 Residual-Encoder nnU-Net、2025 骨折 pipeline、真实金属植入物 deep-MAR 与 2024 低骨密度 fusion/split 直接分割证据；
- [x] 已建立 42 条英文核心 `paper/references.bib`，并纠正多条易错题录；
- [ ] 用真实实验更新论文 Results；
- [ ] 生成主结果表和消融表；
- [ ] 做失败案例图；
- [ ] 做 Web/三维真实截图；
- [ ] 中期 PPT 使用 `docs/05_midterm_materials.md`；
- [ ] 软著材料严格区分上游和自研代码；
- [ ] 每次材料更新同步回写本台账。

---

## 10. 继续项目时的推荐检查命令

### 10.1 激活环境

```powershell
cd D:\国创项目
.\.venv\Scripts\Activate.ps1
```

如环境损坏/需重建：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
```

### 10.2 测试

```powershell
python -m pytest tests -q
python -m ruff check src web tests
```

当前基准应为：

```text
103 passed
All checks passed!
```

附加结构检查：`data/datasets.json`、`configs/label_schemas/ctspine1k_verse.json`、`configs/task_specs/vertebra_task_template.json` 可解析；`paper/references.bib` 当前 42 entries、括号平衡且无重复 key；前端 `app.js / qc_review.js / research_3d.js / results_review.js` 均通过 `node --check`；当前纳入检查的 PowerShell 脚本语法 parser 通过。

### 10.3 获取上游（仅当目录不存在）

```powershell
powershell -ExecutionPolicy Bypass -File .\env\fetch_segformer3d.ps1
```

### 10.4 启动 Web

```powershell
powershell -ExecutionPolicy Bypass -File .\web\run_web.ps1
```

浏览器：

```text
http://127.0.0.1:8000
```

### 10.5 正式训练前

必须先满足：

```text
真实处理后数据存在
+ 正式 split_file 存在且 formal_experiment 允许
+ 标签定义固定
+ 人工 QC 达到正式 run 要求
+ GPU 环境确认
+ 上游 SegFormer3D 可加载
+ formal preflight ready=true
```

训练入口：

```powershell
python -m src.modeling.train --config configs\orthopedic_ct_baseline.yaml
```

当前不要执行正式训练命令，除非上述条件已经满足。

---

## 11. 最近工作记录

### 2026-08-15｜阶段 A：资料、方案与交接机制初始化

完成：

- 打开并检查 `D:\国创项目`，确认开始时为空；
- 阅读任务书与 7.21 组会材料；
- 阅读 SegFormer3D 官方 README、架构入口、loss、requirements/config；
- 建立项目目录；
- 建立 `PROJECT_STATUS.md` 主台账；
- 建立总体方案、文献、数据 SOP、实验计划和论文 outline；
- 明确 Dice≥0.93 为目标而非当前结果；
- 明确系统为科研/辅助分析原型。

### 2026-08-15｜阶段 B：第一轮工程落地

完成：

- 使用 `uv` 在项目内安装 Python 3.11.7；
- 建立独立 `.venv`；
- 固定 PyTorch 2.1.0 CPU 与医学影像/Web/Lightning 依赖；
- 发现并解决 `lightning 2.0.9` 与新 setuptools 移除 `pkg_resources` 的兼容问题，固定 `setuptools==80.9.0`；
- 克隆官方 SegFormer3D，基线提交 `e314242`；
- 发现 PyTorch 2.1 TorchScript 的 `cube_root()` 返回类型问题，做 `int(round(...))` 最小兼容补丁并记录 provenance；
- SegFormer3D 完成 `(1,1,64,64,64) → (1,2,64,64,64)` CPU 前向验证，参数量 4,492,066；
- 建立 DICOM series/QC/HU/重采样/骨窗/metadata 处理代码；
- 使用 pydicom `CT_small.dcm` 完成预处理 smoke test；
- 建立 patient-level split 工具；
- 建立 Region / Boundary / soft-clDice 联合损失；
- 建立 segmentation metrics；
- 建立 predictive entropy 与 uncertainty ROI；
- 建立处理后 NIfTI dataset；
- 建立 baseline/joint YAML；
- 建立训练/validation/checkpoint/history 框架；
- 建立 FastAPI + 前端科研 Web 原型，健康检查通过；
- 建立中文论文技术初稿；
- 扩展文献调研至国内骨 CT、VerFormer、UCTNet、Betti matching、TEDS-Net 等；
- 建立 `docs/05_midterm_materials.md`；
- 增加自动化测试和代码质量检查；
- 最终验证：`12 passed`，Ruff 全部通过，PowerShell 脚本解析通过，依赖 dry-run 无变更。

**本阶段仍未完成：**

- 真实公开主数据下载/全量预处理；
- 临床数据；
- GPU 正式训练；
- baseline 测试指标；
- 联合损失真实消融；
- uncertainty refinement network；
- 三维重建；
- Web 真实分割/MPR/3D/测量；
- 论文 Results。

### 2026-08-16｜阶段 C：P0 公开数据接入链补强

完成：

- 复核主台账、README、数据 SOP 和现有测试，确认真实数据仍是当前首要阻塞；
- 发现并修复 DICOM 读取链没有显式保证物理切片顺序的问题：新增 `sort_dicom_files_by_geometry()`，优先按 IOP/IPP 位置排序，必要时只允许唯一 `InstanceNumber` 回退；
- 将 DICOM pipeline version 更新为 `0.2.0`；
- 新增 `src/preprocessing/nifti_pipeline.py`，补齐原设计承诺但此前缺失的 NIfTI 标准化入口；
- NIfTI 入口实现 image/label size、spacing、origin、direction 一致性检查，错位标签直接拒绝；
- image 使用 linear 重采样，label 使用 nearest-neighbor，并验证输出标签类别不被插值污染；
- 新增 `src/preprocessing/prepare_verse.py`：自动发现 VerSe CT/mask、识别 source split、检查 patient group 防泄漏、生成 manifest/split/batch QC；
- 自动化测试期间发现 split 识别会被 pytest 临时路径中的 `test` 字符串干扰，已改为按路径目录段精确匹配并加入回归测试；
- 建立 `data/datasets.json`，于 2026-08-16 核对并登记 VerSe complete、CTSpine1K、TotalSegmentator CT v2.0.1 的来源/规模/许可注意事项；
- 工程 baseline 暂定优先 VerSe complete；该决定仅用于先跑通工程链，不代替组内最终论文主任务确认；
- 新增 `env/download_verse.ps1`，默认 dry-plan，只有显式 `-Download` 才下载大型数据；
- 新增 `docs/06_public_dataset_onboarding.md`，给出下载、dry-run、10 例 QC、全量处理和 baseline 前验收流程；
- README、数据 README、数据 SOP 已同步新入口和输出规范；
- 最终验证：`16 passed`、Ruff 全部通过、`datasets.json` 可解析，`setup_env.ps1` / `fetch_segformer3d.ps1` / `download_verse.ps1` / `run_web.ps1` PowerShell parser 全通过；
- 验证 `download_verse.ps1 -Edition 2020` 默认只展示 3 个 2020 数据包下载计划，没有触发真实下载；
- 继续检查真实下载可达性：当前本机访问 VerSe 2020 training/validation/test 三个 S3 归档均约 20 秒超时；
- 同步确认 CTSpine1K Hugging Face 镜像可正常浏览，完整数据约 162 GB，但支持按单病例 NIfTI 浏览；MSD-T10 目录存在几十 MB 级 CT 文件及对应较小 label，可作为当前网络条件下的真实 NIfTI 小样本备用入口。

**本阶段明确未完成：**

- VerSe/CTSpine1K/TotalSegmentator 尚未实际下载；
- 没有真实 10 例 3D CT 的三视图/overlay 人工 QC；
- 没有真实公开数据 split 的最终本地实例；
- GPU baseline、真实 DSC/HD95/ASSD 仍未产生；
- 因此论文 Results 仍必须保持 TBD。

**下一次继续时第一优先级：**

```text
优先重试 VerSe 可用下载源
→ 若 S3 仍超时，则用 CTSpine1K Hugging Face 镜像按病例获取少量真实 CT + label
→ 先完成 1 例真实 NIfTI pipeline smoke test
→ 扩展到 ≥10 例标准化
→ 人工检查三视图 + bone window + label overlay
→ QC 通过后再全量处理与训练
```

### 2026-08-16｜阶段 D：真实数据备用入口与 QC 自动化

完成：

- 在继续 P0 前重新执行全项目基线检查，原有 `16 passed` 与 Ruff clean 均确认无回归；
- 通过公开资料再次核对 CTSpine1K 官方维护仓库、Hugging Face 镜像结构及官方 `data_split.txt`；
- 本机对 Hugging Face 镜像执行实际连接检查，约 20 秒超时，确认真实数据下载仍是当前工作站外部网络阻塞，而不是代码链问题；
- 新增 `env/download_ctspine1k_sample.ps1`：默认只展示 `MSD-T10` 的 `liver_169`、`liver_0`、`liver_1` CT+label 下载计划，只有显式 `-Download` 才落盘，同时准备官方 split 元数据下载与 provenance manifest；
- 新增 `src/preprocessing/prepare_ctspine1k.py`：支持 Hugging Face 原始布局和项目小样本布局，严格配对 image/`*_seg` label，解析 `trainset / test_public / test_private`，不擅自重解释 benchmark split；
- 新增 `src/preprocessing/qc_visualization.py`：统一生成 axial/coronal/sagittal × normalized CT/bone window/label overlay 的 3×3 QC contact sheet；存在前景时按前景中位位置选层，降低“体积中心没有目标”的无效审核概率；
- 新增批量 `manual_qc_review.csv` 与 `qc_visualization_summary.json`，固定 orientation、spacing、label alignment、bone window、review status、reviewer、notes 等人工审核字段；
- 将统一 QC 接入 `prepare_verse --qc` 与 `prepare_ctspine1k --qc`，两条真实数据路径使用同一审核标准；
- 更新 `README.md`、`docs/06_public_dataset_onboarding.md`、`data/datasets.json`，记录当前网络阻塞、备用数据入口和实际命令；
- 新增 `tests/test_qc_visualization.py`、`tests/test_prepare_ctspine1k.py`，并扩展 `tests/test_prepare_verse.py` 的 `--qc` 集成测试；
- 新增 `tests/test_training_smoke.py`，用合成标准病例验证 Dataset → SegFormer3D → DiceCE → backward → AdamW.step 完整单步训练链；首次使用 `32³` ROI 时发现上游最后一层被压到 `1×1×1`，batch size=1 的 BatchNorm 无法训练，这是过小 smoke-test ROI 的结构约束而非正式 `128³` baseline 回归；将测试 ROI 调整为 `36³` 后端到端训练更新通过；
- 最终验证：`pytest tests -q → 22 passed`；Ruff 全部通过；`datasets.json` 解析通过；`setup_env.ps1`、`fetch_segformer3d.ps1`、`download_verse.ps1`、`download_ctspine1k_sample.ps1`、`run_web.ps1` 共 5 个 PowerShell 脚本语法检查通过；
- `download_ctspine1k_sample.ps1` 默认 dry-plan 已实跑，未触发真实数据下载。

**本阶段仍未完成：**

- 真实公开 CT/label 仍未落盘；
- ≥10 例真实人工 QC 尚未完成；
- GPU 环境、baseline 训练、真实 DSC/HD95/ASSD 尚未产生；
- Web 仍没有真实 checkpoint 推理；
- 论文 Results 仍必须保持 TBD。

**下一次继续时第一优先级更新为：**

```text
切换到能够访问 VerSe S3 / CTSpine1K 维护镜像的网络环境
→ 实际下载 1 个 CTSpine1K 小样本或 1 个 VerSe 病例
→ 使用 prepare_* --qc 完成首个真实病例标准化 + qc_contact_sheet.png
→ 核验 metadata/qc/label alignment
→ 扩展到 ≥10 例并填写 manual_qc_review.csv
→ 真实 QC 通过后固定训练 split 与 GPU baseline
```

### 2026-08-16｜阶段 E：真实 10 例数据验收 + 模型/评估/Web/三维工程补全

本阶段目标是**最大程度解除“无真实数据”和“配置有但代码未兑现”的阻塞**，同时继续严格区分工程 smoke 与正式科研结果。

完成：

- 先重新跑基线：原状态 `22 passed` + Ruff clean，无回归；
- 将 `configs/orthopedic_ct_joint.yaml` 中此前只写在配置里的增强真正接入 Dataset：3D flip、小角度 rotate、各向同性 scale、gamma、Gaussian noise、HU shift；
- 新增 `boundary_proxy` hard patch sampling，作为 baseline 前困难区域代理；真实模型建立后再替换/补充 high-loss/high-HD95/high-uncertainty mining；
- 真实数据发现 `ct_normalized` 为 case-wise z-score 而非 `[0,1]`，修复原强度增强语义：pipeline 升级到 `0.3.0`，metadata 保存 clipped HU mean/std，gamma/HU shift 可回 HU 域执行；
- 新增 `src/modeling/refinement.py`：`UncertaintyRefinementNet3D` 预测局部 residual logits，只在 uncertainty ROI 内修正 coarse logits；
- Web preview 从中央轴位升级为 axial/coronal/sagittal MPR，可设置归一化切片位置、窗宽窗位；
- 新增物理空间 mesh 基础：`src/reconstruction/mesh.py` + `export_mesh.py`，显式应用 spacing/origin/direction，并支持 PLY + JSON；
- 新增 `src/modeling/evaluate.py`：checkpoint 独立 sliding-window evaluation，输出 Dice/IoU/Precision/Recall/HD95/ASSD、component count、false merge/false break、逐病例推理时间，可保存 prediction/entropy NIfTI；
- 训练框架补齐 linear warmup + cosine warm restarts，checkpoint 保存 scheduler state；run 固定保存 `config.yaml`、`split.json`、`run_metadata.json`、`history.csv`、`train.log`；
- 新增 `src/preprocessing/audit_processed.py`，用于审计 pipeline version、spacing、image/label geometry、label values、bone-window、normalization metadata；
- 新增 `src/modeling/real_patch_smoke.py`，明确 `formal_metric=false`，用于真实病例 forward/backward 工程验收；
- 修复 CTSpine1K 小样本布局真实数据触发的 image/label 配对 bug，并补回归测试；
- 修复 `prepare_ctspine1k`、`qc_visualization` 等 Windows cp1252 中文 CLI 输出问题；
- Hugging Face 早期并行下载失败后，改为 Edge 浏览器**单文件顺序下载**，最终实际取得 `MSD-T10`：`liver_0`—`liver_8` + `liver_169` 共 10 个 CT+label；
- 官方 `data_split.txt`：9 例 `trainset` + `liver_169` 1 例 `test_private`；`test_private` 明确禁止进入训练调参；
- 原始数据位置：`data/raw_public/CTSpine1K/MSD-T10`；标准化位置：`data/processed_ctspine1k_real`；当前约 1.4 GiB raw + 3.3 GiB processed；
- 浏览器下载→项目接管执行 SHA-256 校验；`liver_3/5/8` 等顺序重试病例已确认源/目标哈希一致；
- 10 例均按 pipeline 0.3.0 完成 1 mm 重采样、HU clip、case-wise z-score、bone window、nearest-neighbor label、contact sheet；处理失败 0；
- 原始 z-spacing 实际覆盖约 `0.8 / 1.0 / 5.0 mm`，因此真实工程子集覆盖厚层与近各向同性 CT；
- `audit_processed` 最终：`10/10 pass`、全部 pipeline `0.3.0`；
- 批量生成 `manual_qc_review.csv` 10 行；**orientation/spacing/label alignment/bone window/reviewer/review_status 人工字段保持空白，待项目成员真正逐例签字**；
- 真实 `liver_0` 前景 patch smoke：输入 `(1,2,36,36,36)`，前景比例约 `0.5313`，joint loss→backward→AdamW.step 成功，205 组梯度有限；该 loss/梯度是随机权重工程输出，禁止作为性能；
- 真实 `liver_0` label 导出 PLY：约 9.5 MiB，131,983 顶点、264,362 面，证明真实 mask→physical-space mesh 链可运行；表面积/包围盒不作临床测量结论；
- 建立 `docs/07_real_data_validation_20260816.md`，集中记录真实数据证据、限制和下一步；
- 更新 README、数据登记、公开数据 SOP、中期材料和论文 Methods；论文 Results 继续保持 TBD；
- **最终回归：`pytest tests -q → 38 passed`；`ruff check src web tests → All checks passed!`；`data/datasets.json` JSON 解析通过。**

**本阶段明确仍未完成：**

- 10 例 contact sheet 的人工逐例签字审核；
- 首篇论文最终 binary/multiclass/instance 标签定义；
- 正式论文 train/validation/test split；
- NVIDIA GPU/服务器确认和正式 baseline 训练；
- 可写入论文的真实 DSC/HD95/ASSD；
- Boundary/Topology/hard augmentation/uncertainty refinement 真实消融；
- 真实 prediction mask 的三维表面误差/简化/高保真重建；
- Web 真实 checkpoint overlay、uncertainty、3D 渲染和测量；
- 临床脱敏数据与伦理/授权。

**下一次继续时第一优先级：**

```text
项目成员完成 10 例 manual_qc_review.csv 人工审核签字
→ 组内固定首个任务/标签定义与正式数据 split
→ 确认 NVIDIA GPU/CUDA/PyTorch 训练环境
→ 跑 SegFormer3D CT-only baseline
→ 使用 src.modeling.evaluate 生成第一份正式 metrics_per_case.csv
→ 再按 CT+bone-window / Boundary / Topology / hard augmentation / uncertainty refinement 顺序做消融
→ 将真实 checkpoint prediction/uncertainty/mesh 接入 Web
```

### 2026-08-16｜阶段 F：组会阶段进展 PPT

完成：

- 以现有 `7.19组会.pptx` 的白底、深蓝主色、浅蓝信息卡与底部蓝色条带为视觉参考，生成 4 页组会汇报材料：`8.16组会_项目阶段进展汇报.pptx`；
- PPT 聚焦当前真实状态：10 例 CTSpine1K 真实 CT+label、10/10 自动审计、真实单 patch train-step、38 passed、Web MPR 与真实 label 物理空间 mesh；
- 单独列出尚未完成事项：人工 QC 签字、正式任务/标签、patient-level split、NVIDIA GPU baseline、DSC/HD95/ASSD、临床脱敏数据；
- 明确 PPT 中未把随机权重工程输出、真值标签网格或任务书目标值写成模型性能，论文 Results 继续保持 TBD；
- 同步保留 `make_group_meeting_ppt_20260816.ps1` 与 `group_meeting_ppt_content_20260816.json`，便于后续组会快速更新同风格材料；
- 新增与 4 页 PPT 一一对应的口语化照念稿 `8.16组会_照念稿.md`，按封面/总体进度/工程证据/下一步四段组织，建议汇报时长约 4–6 分钟，并保持“工程 smoke ≠ 正式性能”的真实口径。

### 2026-08-16｜阶段 G：正式实验保护 + 交互 QC/3D + 几何误差 + uncertainty/refinement + 文献库补全

本阶段继续在**不具备正式 GPU baseline 的前提下，最大化完成所有可先验验收的工程与论文准备工作**。

完成：

- 新增 `src/modeling/preflight.py`，并接入 `train.py` / `evaluate.py`：formal 模式默认检查 `formal_experiment`、病例级 split 互斥、`test_private` 泄漏、人工 QC、pipeline/输入、标签范围/`num_classes`，正式训练额外要求 CUDA；工程 smoke 必须显式选择 engineering 模式；
- 真实 engineering split preflight：`ready=true`、0 error/0 warning；相同 split 的 formal preflight 正确返回 `ready=false`，明确拦截 engineering split、未签字人工 QC 与本机无 CUDA；
- 修复 multiclass evaluation 的空类别虚高风险：逐病例 macro 只统计真值或预测实际出现的前景类别，并输出 `metrics_per_class.csv`；
- 扩展 uncertainty 定量评价：error AUROC/AUPRC、error/correct 平均 entropy、Top-percent error recall、ROI error rate/fraction；大体积使用固定 seed 抽样避免评价内存失控；
- 新增 `src/modeling/refinement_training.py`，形成 coarse 默认冻结、ROI-normalized loss、ROI/global error delta 的二阶段精修训练基线；ROI 外 prediction 保持不变；
- 发现并解决 SimpleITK 2.3.1 在当前 Windows 中文项目绝对路径下的真实 I/O 兼容问题，建立 `src/sitk_compat.py`，接入 NIfTI/DICOM/QC/审计/mesh 等关键路径；真实 Web `liver_0 class 24` mesh 生成验证通过；
- 建立真实人工 QC Web reviewer：10 例 contact sheet + axial/coronal/sagittal 交互 MPR + 真值 label overlay + orientation/spacing/alignment/bone-window 四项人工检查 + reviewer/status/notes；系统不自动代签，`manual_qc_review.csv` 当前人工字段仍保持空白；
- 新增真值 3D 研究页与 WebGL2 PLY viewer；真实 `liver_0` 全前景 PLY HTTP 约 9.95 MB、131,983 顶点/264,362 面；真实 `class 24` Web API 现场生成 23,088 顶点/46,196 面 PLY；
- 新增 physical-space 距离/三点夹角 API 与 voxel→physical 工具；这些只是几何测量基础，不构成临床结论；
- 新增 vertex-clustering mesh 简化：真实 `liver_0` 1.5 mm 档 52,726 顶点/106,329 面，约减 60%，vertex-nearest HD95 约 0.707 mm；2.0 mm 更轻但误差更大，因此 Web 默认优先 1.5 mm，且全分辨率网格不覆盖；
- 新增 `src/reconstruction/resampling_error.py` 并在 10 例真实 label 上完成 raw→1 mm nearest-neighbor physical-surface 误差评估：10/10 成功，整体 vertex-nearest ASSD 约 0.403 mm、HD95 约 0.734 mm；5 mm 原始层厚组约 ASSD 0.514 mm / HD95 1.069 mm。上述数字只表示预处理离散化，不是模型或临床精度；
- 建立 `configs/label_schemas/ctspine1k_verse.json` + `src/label_schema.py`，按 VerSe-compatible `1–25 → C1–L6` 仅做工程显示；真实 `liver_0` 已显示为 `T11/T12/L1–L5`，原始 18–24 不重编码，`formal_task_locked=false`；
- 文献工作从首批列表扩展为 `docs/08_literature_matrix.md` 的 37 条结构化矩阵，并建立 `paper/references.bib` 35 条英文核心条目；本轮纠正 CTSpine1K、VerSe、VerFormer、nnFormer、SegFormer3D、TEDS-Net、EDUE 等易错题录，`docs/02_literature_survey.md` 更新到 v0.2；
- 论文 Methods/Experiment Design 已同步 formal preflight、uncertainty 定量评价、ROI-only refinement、physical-space surface/重采样几何控制；Results 继续保持 TBD；
- README、中期材料、真实数据验证文档与实验计划均已同步当前证据，仍明确区分真值网格/预处理几何数字与模型性能；
- **最终全量回归：`pytest tests -q → 71 passed`；`ruff check src web tests → All checks passed!`；`data/datasets.json` 与标签 schema JSON 可解析；`paper/references.bib` 35 entries、括号平衡、无重复 key；3 个前端 JS 均通过 `node --check`。**

**当前仍不能完成/不能伪写完成：**

- 10 例人工 QC 的真实逐例签字；
- binary / multi-class semantic / instance 的正式任务锁定；
- 正式 patient-level train/validation/test split；
- NVIDIA GPU/CUDA 正式训练环境；
- SegFormer3D baseline checkpoint 与独立测试 DSC/HD95/ASSD；
- Boundary/Topology/hard augmentation/uncertainty refinement 的真实消融收益；
- prediction/uncertainty/prediction-mesh 的 Web 接入；
- SDF/曲率保护等高保真表面重建；
- 临床脱敏数据、授权与伦理。

**下一次继续时严格按以下优先级：**

```text
1. 项目成员在 /qc-review 逐例完成人工 QC 并签字
2. 组内锁定正式任务/标签定义，填写 task spec，并生成正式 patient-level split/config
3. 在目标 GPU 机器运行 formal_readiness，必须确认 ready=true
4. 先跑 CT-only SegFormer3D baseline，固定第一份 checkpoint
5. 用 evaluate 生成正式 metrics_per_case.csv / metrics_per_class.csv（如适用）
6. 依次做 bone-window → Boundary → Topology → hard augmentation → uncertainty/refinement 消融
7. 用现有 results-review 复核真实 prediction/entropy，再生成 prediction mesh 并更新论文 Results
```

### 2026-08-16｜阶段 H：正式 readiness 汇总 + results-review + SDF 高保真表面 + 2025–2026 文献补强

本阶段目标是在仍缺正式 GPU checkpoint 的前提下，把“能否启动正式实验”“正式结果如何复核”“三维表面如何做拓扑保护”三条链补齐，并完成全项目回归。

完成：

- Web 新增 evaluation results-review：可发现 `experiments` 下规范 evaluation、读取逐病例指标，并把 prediction/entropy 与标准化 CT 做 MPR overlay；真实项目接口 `/api/research/evaluations` 实测 `200`、`total=0`，正确表示**当前没有真实正式 evaluation**；
- 新增 `configs/task_specs/vertebra_task_template.json` 与 `src/modeling/task_lock.py`：只有 `task_locked=true` 且 task id/type/labels/num_classes/data/split 完整时才能编译正式 config，并保存 task-spec SHA-256；当前只支持 binary/multiclass semantic，instance 因训练/评价链未实现而明确拒绝；
- 新增 `src/modeling/gpu_environment.py` + `env/check_gpu.ps1`：只读检查项目 venv、PyTorch CUDA build、可见设备、显存和 `nvidia-smi`；本机实测 Python 3.11.7、torch `2.1.0+cpu`、CUDA=false、0 device、无 `nvidia-smi`，因此不能启动正式 3D baseline；
- 新增 `src/modeling/formal_readiness.py` + `env/check_formal_readiness.ps1`：统一汇总 task lock、GPU、formal preflight 以及正式 config/task 指纹绑定；在当前真实 processed root + 1/1/1 engineering smoke split 上实测 `ready=false`、exit 2、9 个 blocker，包含任务未锁定、split 非 formal、3 例人工 QC 未批准和 GPU 环境未就绪；该失败是保护机制正确工作，不是程序故障；
- 新增 `src/reconstruction/sdf_surface.py`：binary mask→physical-mm signed distance field→可选 Gaussian smoothing→零等值面 Marching Cubes，并默认要求 smoothing 前后 connected-component count 一致；
- 真实 `liver_0` SDF sweep：0.3/0.4/0.5 mm 均保持 2→2 连通域；0.4 mm 当前作为工程默认候选，顶点 131,950、面 264,056、相对原始 MC 面积变化约 -1.81%、vertex-nearest ASSD 约 0.0209 mm、HD95 约 0.0644 mm；0.8 mm 造成 2→3 连通域变化，因此标记拒绝。上述均为**真值 mask 表面工程差异，不是模型或临床精度**；
- SDF 已接入研究 3D Web：真实 `ctspine1k-msd-t10-liver_0` 的 0.4 mm summary/PLY 路由均 `200`；0.8 mm summary 返回 `422`，证明已生成但拓扑不合格的表面也不会被前端加载；
- 文献矩阵从 37→40 条，英文 BibTeX 从 35→38 条；正式加入 SpineMamba（2025）、Transformer-enhanced vertebrae segmentation/anatomical variation（2025）和 VertebraFormer（2026），并同步 `docs/02_literature_survey.md`、论文相关工作、README 与中期材料；
- 新增 `tests/test_formal_readiness.py`，并保留 task lock/GPU/results-review/SDF Web 等专项测试；
- **最终全量回归：`pytest tests -q → 88 passed`；`ruff check src web tests → All checks passed!`；4 个前端 JS `node --check` 通过；3 个关键 JSON 可解析；`paper/references.bib` 38 entries、括号平衡、无重复 key；7 个 PowerShell 脚本语法 parser 全部通过。**

**当前仍不能完成/不能伪写完成：**

- 10 例人工 QC 的真实逐例签字；
- 正式 binary/multiclass 任务锁定和正式 patient-level split；
- NVIDIA GPU/CUDA 正式训练环境与 `formal_readiness ready=true`；
- SegFormer3D baseline checkpoint 与独立测试 DSC/HD95/ASSD；
- Boundary/Topology/hard augmentation/uncertainty refinement 的真实消融收益；
- 真实 prediction/entropy/prediction mesh 结果；当前 results-review 为空是正确状态；
- 临床脱敏数据、授权与伦理。

**下一轮优先级不变，但入口更明确：**人工 QC 签字 → task spec 锁定 + formal split/config → 目标 GPU `formal_readiness` 通过 → baseline → evaluate/results-review → 消融 → prediction mesh/论文 Results。

### 2026-08-25｜阶段 I：公开 GitHub 多人协作仓库整理

本阶段目标是把当前工程从“本机科研项目目录”整理为可以安全公开、供多人分工协作的 Git 仓库，同时避免把医学影像、临床数据、模型权重、运行缓存和第三方 checkout 误公开。

完成：

- 在 `D:\国创项目` 初始化本地 Git 仓库，默认分支为 `main`；
- 新增 `TASKS.md`：把项目拆成总体方案、数据/QC、模型、联合损失、困难样本、不确定性、三维/Web、正式实验、临床验证和论文结题等模块，并明确 P0–P5 优先级、推荐负责人和完成定义；
- 新增 `CONTRIBUTING.md`：规定 feature/docs 分支、中文 commit、Pull Request、pytest/Ruff/JS 检查、正式实验产物追踪和 `PROJECT_STATUS.md` 强制回写；
- 新增 `SECURITY.md`：明确禁止公开患者 DICOM/NIfTI、未脱敏临床数据、身份信息、token/key、checkpoint，并再次区分 random-weight smoke / GT mesh / preprocessing geometry 与正式模型性能；
- 新增 `.github/ISSUE_TEMPLATE/task.md`、`bug.md` 与 `.github/PULL_REQUEST_TEMPLATE.md`，方便团队直接按任务/bug/PR 流程协作；
- 新增 `docs/09_public_repository_manifest.md`：列出公开仓库应包含的代码、配置、测试、论文、数据元信息与脚本，以及必须排除的医学数据、处理后数据、模型权重、runtime、虚拟环境、第三方 checkout 和大型 PPT；
- 加强 `.gitignore`：新增 `data/processed_*/`、`.devspace-computer/`、`*.pptx`、`*.log` 等公开安全规则，同时保留 `data/README.md`、`data/datasets.json` 和匿名 split JSON；
- 用 `git check-ignore` 实际确认真实 `label.nii.gz`、QC 图片、DevSpace 截图、大型组会 PPT、SegFormer3D checkout、Web runtime 均被排除；首批可提交文件只包含源码/配置/测试/论文/文档/匿名数据元信息；
- 公开前扫描未发现 GitHub token/private key 等真实凭据；命中的 `token` 仅为代码变量，处理后数据中的数字模式均处于已忽略目录；
- 再次全量回归：`pytest tests -q → 88 passed`；`ruff check src web tests → All checks passed!`；
- 已完成本地首个公开协作提交 `5576a82b1025a0c3060d461f0d84cef0efdfea24`（`chore: initialize public collaboration repository`）；仓库级 Git 邮箱使用 `noreply@users.noreply.github.com`，避免把电脑全局个人邮箱写入公开 commit；
- 已在 GitHub 账号 `927242768-dotcom` 下创建 **Public** 仓库 `orthopedic-ct-segformer3d`，未让 GitHub额外初始化 README/.gitignore/license，避免与本地历史冲突；
- 已添加远程 `origin=https://github.com/927242768-dotcom/orthopedic-ct-segformer3d.git`，并成功执行 `main → origin/main` 首次 push，当前本地 `main` 已跟踪 `origin/main`；
- GitHub 页面已实机确认仓库为 Public，首个 commit、源码、配置、测试、论文、文献、任务清单、协作文档和 Issue/PR 模板均已可见；真实 CT/处理后数据/checkpoint/runtime/第三方 checkout/大型 PPT 未进入公开历史。

**当前 GitHub 发布状态：✅ 已完成。** 公开仓库：`https://github.com/927242768-dotcom/orthopedic-ct-segformer3d`。团队成员可以直接浏览/clone/fork/提 Issue/PR；如需要朋友直接 push 到主仓库，需再按其 GitHub 用户名添加 collaborator。后续所有协作以 `TASKS.md` + Issues/PR + `PROJECT_STATUS.md` 为主线。

---

## 12. 交接规则

下一位成员或下一次 AI 会话继续时，必须按以下顺序：

1. **先读本文件 `PROJECT_STATUS.md`；**
2. 再读 `README.md`；
3. 根据任务读 `docs/01_overall_design.md`、`docs/03_data_pipeline_spec.md`、`docs/04_experiment_plan.md`；
4. 如准备中期，读 `docs/05_midterm_materials.md`；
5. 如写论文，读 `paper/manuscript_zh_v0.1.md`；
6. 运行 `pytest` 和 `ruff` 确认工程状态；
7. 只处理“下一步任务”最高优先级未完成事项；
8. 不覆盖第三方仓库本地补丁；
9. 不把临床数据提交到 Git；
10. **完成任何实质修改后，最后一个项目文档动作必须是更新本 `PROJECT_STATUS.md`。**


### 2026-08-25｜阶段 J：GitHub CI 与协作质量门禁补强

本阶段目标是补齐公开协作仓库缺失的自动化质量门禁，让后续多人提交 Pull Request 时能自动发现基础工程回归，同时不把 CI 误当作正式医学实验验收。

完成：

- 新增 `.github/workflows/ci.yml`，对 `main` 的 push 与 Pull Request 自动触发；
- Python CI 固定 Python 3.11，并显式安装 PyTorch 2.1.0 CPU 栈与 `env/requirements.txt`，自动执行 `ruff check src web tests` 与 `pytest tests -q`；
- 新增前端 JavaScript 语法检查，覆盖 `app.js / qc_review.js / research_3d.js / results_review.js`；
- 新增关键 JSON 可解析性检查，覆盖数据集登记、标签 schema 与 task spec 模板；
- 新增 PowerShell 语法 parser 检查，覆盖 `env/`、`web/` 与仓库根目录中已跟踪的 `.ps1`；
- 为 CI 增加最小 `contents: read` 权限、并发取消与 job timeout，降低重复运行和权限过宽风险；
- 更新 `CONTRIBUTING.md`，明确自动 CI 的覆盖范围以及它不能替代人工 QC、task/split 锁定、GPU readiness 和真实模型实验；
- 更新 `TASKS.md`，把 GitHub Actions CI 纳入“文档与质量”已完成项；
- 修正本台账旧检查命令区仍写成 `71 passed / 35 BibTeX / 3 JS` 的历史残留，统一为当前 `88 passed / 38 BibTeX / 4 JS / 3 个关键 JSON / 7 个 PowerShell 脚本`；
- 本地重新验证：`pytest tests -q → 88 passed`、`ruff check src web tests → All checks passed!`；workflow YAML 可由 PyYAML 正常解析；4 个前端 JS `node --check` 通过；当前 PowerShell 脚本语法 parser 通过。

当前边界：

- GitHub Actions 的“云端首次真实运行”只能在 workflow 推送到 GitHub 后由 Actions runner 触发，因此在首次 CI run 成功前，本阶段只能声明“CI 配置与本地等价检查已完成”，不能声明“云端 CI 已通过”；
- 正式实验 P0 阻塞仍不变：人工 QC、正式任务锁定、正式 patient-level split、NVIDIA GPU/CUDA 环境；
- 本次没有生成任何模型性能数字，也没有改动真实医学数据、正式 split 或论文 Results。


### 2026-08-25｜阶段 K：人工 QC 病例栏遮挡/折叠交互修复

本阶段直接修复 `/qc-review` 在人工逐例审核时病例侧栏持续占据页面、窄窗口下遮挡/挤压主审核区的问题，不改变人工审核判定规则、CSV 保存逻辑、医学数据或模型结果。

完成：

- `web/frontend/qc_review.html`：为 QC 主布局、病例侧栏和主审核区补充稳定 DOM id；新增“显示病例列表 / 收起病例列表”切换按钮及 ARIA 关联；
- `web/frontend/qc_review.js`：新增病例栏折叠状态、`setCaseListCollapsed()`、`showCaseList()`、`enterReviewArea()` 与统一 `selectCase()`；点击任意病例后自动收起列表并滚动进入主审核区；“上一例 / 下一例”复用同一选择逻辑，继续保持审核区；重新展开列表时自动回到病例列表位置；
- `web/frontend/qc_review.css`：折叠后主审核区自动占满可用宽度；为 sticky 顶栏设置审核区滚动留白；窄窗口下病例栏切换按钮固定在右下角，确保长页面审核过程中仍能随时重新展开；保留 980 px / 900 px / 650 px 响应式适配；
- `tests/test_web_qc_review.py`：新增前端回归测试，验证折叠相关 DOM、切换按钮、`scrollIntoView` 进入审核区逻辑、病例点击统一选择逻辑以及上一例/下一例保持审核区逻辑存在；
- 未修改人工审核四项规则、`pass` 校验条件、`manual_qc_review.csv` 写入流程、QC 医学数据、标签、模型输出或推理结果。

本地验证：

```text
node --check web/frontend/qc_review.js
→ 通过（无语法错误）

.\.venv\Scripts\python.exe -m pytest tests/test_web_qc_review.py -q
→ 3 passed

.\.venv\Scripts\python.exe -m ruff check web tests/test_web_qc_review.py
→ All checks passed!

.\.venv\Scripts\python.exe -m pytest tests -q
→ 89 passed, 153 warnings
```

其中 153 条 warning 来自现有 MONAI / pkg_resources / Matplotlib / SciPy 等第三方依赖弃用提示，本次没有新增测试失败或功能回归。

当前状态：QC 前端病例栏遮挡问题已完成代码修复并通过定向与全量自动化回归；人工审核数据和正式实验阻塞状态不变。


### 2026-08-25｜阶段 L：人工 QC 病例层第二次修复与 Edge 实机验收

阶段 K 的 DOM/折叠逻辑虽然通过自动化测试，但用户实机截图仍显示病例任务层铺满并盖住审核 CT，说明仅验证“折叠代码存在”不足以证明浏览器实际交互可用。本阶段根据实机现象继续定位并完成结构级修复。

根因与修复：

- 根因 1：全站 `styles.css` 中 `.card { grid-column: span 6; }`（窄屏还会变为 span 12）继续作用于 `qc-layout` 内的 `.qc-sidebar/.qc-main`，与 QC 自己的 1～2 列 Grid 冲突，导致特定窗口宽度/缩放比例下病例区域异常跨列铺满并覆盖/挤压审核区域；
- `web/frontend/qc_review.css` 新增 `.qc-layout > .card { grid-column: auto; }`，并明确病例栏位于第 1 列、审核区位于第 2 列；折叠后审核区 `grid-column: 1 / -1` 独占可用宽度；窄窗口统一回到单列；
- 病例选择不再只依赖 CSS class：`setCaseListCollapsed()` 同时设置 `caseSidebar.hidden = true`，并配套 `.qc-sidebar[hidden] { display: none !important; }`，确保病例任务层真正退出布局和点击层，不再透明覆盖审核内容；
- `selectCase()` 调整为先关闭病例层，再加载当前病例并 `scrollIntoView()` 进入审核区；上一例/下一例继续复用 `selectCase()`；
- “显示病例列表 / 收起病例列表”按钮改为固定悬浮按钮，避免顶部操作区在高缩放/特殊窗口宽度下被挤出可视区；
- `web/backend/app.py` 的 `/qc-review` 增加 `Cache-Control: no-store, max-age=0`；QC 页面 CSS/JS 使用 `?v=20260825-3` 版本参数，避免 Edge 继续加载旧前端资源；
- `tests/test_web_qc_review.py` 回归测试同步验证 no-store、版本化静态资源、`hidden` 强制折叠、QC Grid 覆盖规则、病例点击与上一例/下一例统一选择逻辑。

最终自动化验证：

```text
node --check web/frontend/qc_review.js
→ 通过

.\.venv\Scripts\python.exe -m pytest tests/test_web_qc_review.py -q
→ 3 passed

.\.venv\Scripts\python.exe -m ruff check web tests/test_web_qc_review.py
→ All checks passed!

.\.venv\Scripts\python.exe -m pytest tests -q
→ 89 passed, 153 warnings
```

实机 Edge 验收：

- 对 `http://127.0.0.1:8000/qc-review` 执行 `Ctrl+F5` 后确认新版布局生效；
- 实际聚焦并选择 `ctspine1k-msd-t10-liver_0`，病例列表区域完全消失，只保留可操作的审核 CT 区域；
- UI Automation 可找到状态已切换为“显示病例列表”的 `caseListToggleBtn`；实际触发该按钮后病例列表成功重新展开；
- 因此本次已不仅通过结构测试，还完成了当前机器/当前 Edge 的真实点击交互验收。

边界保持不变：未修改人工审核四项规则、`pass` 校验、`manual_qc_review.csv` 保存字段/写入逻辑、任何医学数据、标签值、模型输出或论文结果。


### 2026-08-26｜阶段 M：概率校准指标工程链补全

本阶段在不依赖 NVIDIA GPU、正式 checkpoint 或人工 QC 签字的前提下，补齐此前任务表中尚未实现的 segmentation calibration 工程链，使后续真实 baseline/消融能够直接输出概率可信度指标；本阶段只完成代码与合成/工程测试，不产生任何论文模型成绩。

完成：

- `src/modeling/uncertainty.py` 新增 `SegmentationCalibrationMetrics` 与 `segmentation_calibration_metrics()`；
- 支持 binary/multiclass logits，统一报告 Expected Calibration Error（ECE）、Maximum Calibration Error（MCE）、multiclass Brier score、negative log-likelihood（NLL）、mean confidence、体素 accuracy 与 confidence gap；
- 对大体积采用固定随机种子的体素下采样，记录 total/sample voxel 与 sampling fraction，避免全量概率复制造成额外内存压力，并保证不同实验在固定参数下可复现比较；
- 增加类别范围、shape、空输入、采样参数等保护，避免无效标签静默进入校准结果；
- `src/modeling/evaluate.py` 接入 calibration 配置，可将逐病例 calibration 指标写入 `metrics_per_case.csv`，并在 `summary.json` 中聚合；
- `configs/orthopedic_ct_baseline.yaml` 与 `configs/orthopedic_ct_joint.yaml` 默认启用 calibration，固定 `n_bins=15`、`metric_max_samples=500000`；
- `tests/test_metrics_uncertainty.py` 新增高置信正确、过度自信错误、固定 seed 采样确定性 3 类校准测试；
- `tests/test_evaluate_smoke.py` 验证 evaluation CSV/summary 实际包含 calibration 字段；
- `docs/04_experiment_plan.md`、`docs/05_midterm_materials.md`、`paper/manuscript_zh_v0.1.md` 与 `TASKS.md` 同步加入 calibration 评价设计，并明确真实 calibration 结论仍必须等待 validation/test checkpoint；
- 保留 2026-08-25 QC reviewer 的所有未提交修改，不覆盖其前端、后端、测试或 CI 工作树内容。

验证：

```text
./.venv/Scripts/python.exe -m ruff check src/modeling/uncertainty.py src/modeling/evaluate.py tests/test_metrics_uncertainty.py tests/test_evaluate_smoke.py
→ All checks passed!

./.venv/Scripts/python.exe -m pytest tests/test_metrics_uncertainty.py tests/test_evaluate_smoke.py -q
→ 13 passed

./.venv/Scripts/python.exe -m ruff check src web tests
→ All checks passed!

./.venv/Scripts/python.exe -m pytest tests -q
→ 92 passed, 153 warnings

git diff --check
→ 通过

baseline/joint YAML
→ PyYAML 解析通过
```

153 条 warning 仍来自现有 MONAI / pkg_resources / Matplotlib / SciPy 等第三方依赖弃用提示，本阶段没有新增测试失败。

当前边界与下一步：

- calibration **代码与评估输出链已完成**，但没有真实模型 checkpoint，因此不能报告真实 ECE/Brier/NLL 或作“模型已校准”结论；
- P0 阻塞保持不变：10 例人工 QC 签字、正式 binary/multiclass semantic 任务锁定、正式 patient-level split、NVIDIA GPU/CUDA 环境；
- 一旦 baseline checkpoint 产生，`evaluate.py` 将直接输出区域/表面/结构、uncertainty 与 calibration 指标，可用于正式主结果、可信度分析和后续 uncertainty ROI refinement 消融。


### 2026-08-26｜阶段 N：曲率/关键边缘保护网格简化候选

本阶段继续处理 P3 中不依赖正式模型 checkpoint 的“三维曲率/关键边缘保护候选”。目标不是宣称已经得到高保真临床重建，而是在现有 vertex-clustering 基线上增加一个默认关闭、可量化、可回退的特征保护工程候选。

完成：

- `src/reconstruction/mesh.py` 新增 `vertex_normal_variation_scores()`：根据三角网格相邻顶点法向夹角差异构造轻量曲率/尖锐特征代理；
- `simplify_mesh_vertex_clustering()` 新增 `feature_preservation_strength`，默认 `0.0`，因此原有 Web/导出行为保持不变；大于 0 时，在每个空间聚类代表点计算中提高高法向变化顶点的权重，减少聚类平均对尖锐结构的过度平滑；
- `src/reconstruction/export_mesh.py` 增加 `--feature-preservation-strength`，JSON summary 会记录简化方法和强度，便于实验追踪；
- `tests/test_reconstruction_mesh.py` 增加特征分数有限性及“高特征顶点近邻误差不劣于普通聚类”的回归测试；
- `tests/test_export_mesh.py` 验证特征保护候选的 CLI/summary 追踪字段；
- `TASKS.md` 将“曲率/关键边缘保护候选”标记为工程代码已完成、待 prediction surface 验证；
- `docs/05_midterm_materials.md` 与论文 Methods 同步记录该候选，但明确不得把真值网格工程对照写成模型性能。

真实 `liver_0` 真值 label 工程对照（2.0 mm vertex clustering）：

```text
full vertices                         131,983
baseline simplified vertices          30,260
feature-weighted simplified vertices  30,260
高法向变化区域顶点数                  13,311

高特征区域 mean nearest-neighbor
baseline                             ≈ 0.679 mm
feature-weighted                     ≈ 0.620 mm

高特征区域 HD95
baseline                             ≈ 1.068 mm
feature-weighted                     = 1.000 mm

surface area relative change
baseline                             ≈ -6.47%
feature-weighted                     ≈ -5.95%
```

该对照说明：在顶点数/面拓扑映射规模相同的情况下，法向变化加权候选对当前真值网格的尖锐区域具有正向工程信号；但它仍然只是单个真实 GT 病例上的参数筛选，不能推断到模型 prediction、临床测量或总体数据集。

验证：

```text
./.venv/Scripts/python.exe -m ruff check src/reconstruction/mesh.py src/reconstruction/export_mesh.py tests/test_reconstruction_mesh.py tests/test_export_mesh.py
→ All checks passed!

./.venv/Scripts/python.exe -m pytest tests/test_reconstruction_mesh.py tests/test_export_mesh.py -q
→ 7 passed

./.venv/Scripts/python.exe -m ruff check src web tests
→ All checks passed!

./.venv/Scripts/python.exe -m pytest tests -q
→ 94 passed, 153 warnings
```

当前边界：

- 特征保护默认关闭，不改变既有 1.5/2.0 mm Web 简化结果；
- 尚无真实 prediction surface，因此不能完成 prediction mesh vs GT、prediction SDF surface 或正式高保真重建消融；
- 下一阶段 P3 仍需等待正式 checkpoint，再比较普通 vertex-clustering、feature-weighted clustering、SDF surface 与全分辨率 prediction mesh 的 HD95/ASSD、拓扑和计算成本。


### 2026-08-26｜阶段 O：现代强 baseline 与困难病例文献证据补强

本阶段补齐此前文献任务中最影响正式实验设计的三个缺口：现代椎体 CT 强 CNN baseline、真实骨折/断裂困难病例、真实腰椎金属植入物/金属伪影。所有新增题录均优先依据正式出版页面、PubMed/机构出版记录与 DOI 核验，不以二手博客作为题录来源。

新增核验：

1. **Hofmann et al., 2026, European Journal of Radiology 204:113118**，DOI `10.1016/j.ejrad.2026.113118`：公开 1,460 例 CT 的胸/腰椎体部标签；两套 residual-encoder nnU-Net 在 1,216 例上训练、244 例测试，并另用 300 例肿瘤 CT 做 L3 外部定位验证。该工作将 Residual-Encoder nnU-Net 明确提升为本项目正式论文必须认真考虑的强 CNN baseline，而不能只和传统 3D U-Net 比较。
2. **Glessgen et al., 2025, Clinical Radiology 83:106827**，DOI `10.1016/j.crad.2025.106827`：452 例胸腰椎 CT 的骨折报告 pipeline，最终椎体 segmentation 使用 nnU-Net，独立测试中正确分割 330/339 个椎体。该工作直接支撑“骨折/真实断裂病例必须单独评价”，尤其提醒 topology loss 不能把真实骨折断端机械地当作 false break 修复。
3. **Ye et al., 2025, Clinical Radiology 90:107076**，DOI `10.1016/j.crad.2025.107076`：93 例真实腰椎植入物患者的多能 CT deep-MAR 研究。该工作补足真实金属植入物困难成像证据，支持本项目继续坚持“先有真实 metal-artifact 病例校验，再决定是否启用人工伪影模拟/MAR 前处理”。

同步修改：

- `docs/08_literature_matrix.md`：40→43 条，新增 S12–S14，并更新强 baseline/困难病例结论；
- `paper/references.bib`：38→41 条英文核心 BibTeX，新增 `hofmann2026_vertebral_bodies`、`glessgen2025_vertebral_fracture`、`ye2025_lumbar_metal_artifact`；
- `docs/02_literature_survey.md`：新增 3.8–3.10 三个专题小节，数据集建议顺延至 3.11；金属伪影文献任务从待补改为已补，低骨密度/骨质疏松直接文献仍保留待办；
- `paper/manuscript_zh_v0.1.md`：Related Work 增加 Residual-Encoder nnU-Net、骨折 pipeline 和真实金属植入物证据，并将首版参考文献列表同步；
- `README.md`、`TASKS.md`、`docs/05_midterm_materials.md`：统一更新为 43 条矩阵 / 41 条英文 BibTeX。

结构验证：

```text
paper/references.bib
entries        = 41
unique_keys    = 41
duplicate_keys = []
brace_balance  = 0

git diff --check
→ 通过
```

边界与下一步：

- 新增论文中的性能数字只用于理解文献和选择 baseline，不是本项目自身实验结果；
- 当前文献缺口进一步收敛到：低骨密度/骨质疏松 CT 分割直接研究、国内 CNKI/万方题录正式复核，以及最终根据 GPU/任务定义筛选实际可跑的 baseline；
- 正式模型 Results 仍必须等待人工 QC、task lock、formal split 与 NVIDIA GPU baseline，不能把文献结果或真值网格工程数字替代为本项目模型指标。


### 2026-08-26｜阶段 P：低骨密度椎体分割困难病例直接证据补齐

在阶段 O 已补现代强 baseline、骨折和真实金属植入物后，本阶段继续检索“低骨密度/骨质疏松是否会直接造成椎体 segmentation 失败”的证据，避免只凭经验把 low-density 写入困难病例设计。

新增核验：

- **Xiong et al., 2024, Tomography 10(5):738–760**，DOI `10.3390/tomography10050057`，题目 *Lumbar and Thoracic Vertebrae Segmentation in CT Scans Using a 3D Multi-Object Localization and Segmentation CNN*；
- 该研究采用 3D multi-object localization + segmentation，对放疗 CT 与 VerSe2020 进行腰/胸椎分割，并在失败案例分析中明确指出：当骨密度较低时，相邻椎体可能发生错误融合，一个椎体也可能被错误分裂为多个部分；
- 该现象与本项目已经实现的 `false_merge_count`、`false_break_count`、component count/error 直接对应，因此 low-density subset 不再只是经验性假设，而有直接 CT segmentation 文献支持；
- 该文献只证明“低骨密度是值得单独分析的失败条件”，不代表本项目当前模型已经在低骨密度病例上验证，也不应把其论文性能数字当成本项目结果。

同步修改：

- `docs/08_literature_matrix.md`：43→44 条，新增 S15；
- `paper/references.bib`：41→42 条，新增 `xiong2024_low_density_vertebrae`；
- `docs/02_literature_survey.md`：新增 3.11 低骨密度专题，原数据集建议顺延到 3.12，并把“补低骨密度直接分割文献”改为已完成；
- `paper/manuscript_zh_v0.1.md`：Related Work 明确 low-density fusion/split 失败与结构指标的关系；
- `README.md`、`TASKS.md`、`docs/05_midterm_materials.md`：统一更新为 **44 条结构化矩阵 / 42 条英文核心 BibTeX**。

当前文献侧剩余工作主要是国内 CNKI/万方题录的最终数据库级复核、正式投稿格式统一，以及在首个任务/split/GPU 资源确定后从已有强 baseline 中选择真正可运行的对照，不再以机械增加文献数量为目标。


### 2026-08-26｜阶段 Q：人工 QC 解锁 + 笔记本 CPU binary 真实训练跑通

本阶段根据项目成员最新确认，重新核对真实人工 QC 状态，并将“必须 NVIDIA GPU”从方法学硬限制调整为可选效率升级项；随后直接在当前笔记本 CPU 上完成真实 CTSpine1K binary semantic 工程训练 pilot。

完成：

- 复核 `data/processed_ctspine1k_real/manual_qc_review.csv`：10/10 病例 `orientation_ok / spacing_ok / label_alignment_ok / bone_window_ok = yes`，10/10 `review_status=pass`，reviewer 均已填写，因此人工 QC P0 正式解除；
- 实测当前电脑：AMD Ryzen 7 8745H，8 核 16 线程，约 19.8 GB RAM；PyTorch 仍为 `2.1.0+cpu`；
- 真实 `36³` CT-only patch 在完整 SegFormer3D baseline 上执行 forward + backward + optimizer step 成功，整次进程约 12.43 s；
- 新建 `data/splits/ctspine1k_msd_t10_cpu_binary_engineering.json`：仅使用 9 个官方 `trainset`，固定 7 train / 2 validation；`liver_169 test_private` 完全不进入训练或调参；
- 新建 `configs/orthopedic_ct_cpu_binary_engineering.yaml`：CT-only、binary semantic、36³ ROI、batch size 1、CPU、工程 patch-validation；该配置明确 `formal_experiment=false`，不得把 pilot 指标写入论文 Results；
- 工程 preflight 实测 `ready=true`：checked=9、train=7、validation=2、test=0、pipeline 0.3.0=9、0 error / 0 warning；
- 第一次真正进入 `train.py` 时发现 PyTorch 2.1 兼容问题并修复：`torch.amp.GradScaler` 不存在，改为兼容的 `torch.cuda.amp.GradScaler(enabled=...)`；CPU 非 AMP 路径改用 `nullcontext()`，避免 CPU float16 autocast 报错；
- `train.py` 新增工程 `validation.patch_mode`：笔记本训练时验证集取固定大小前景 patch，避免每个 epoch 对 300–600 层整卷 CT 运行高成本 sliding-window；正式论文验证/测试仍必须使用 full-volume evaluation；
- `train.py` 新增 `--allow-cpu`，`formal_readiness.py` 同步新增 `--allow-cpu`；无 CUDA 不再是方法学绝对 blocker，但 CPU 模式不会放宽 task/split/QC/config binding 等其它 formal 保护；
- 1-epoch CPU 真实 run：约 53.2 s，`train_loss=5.5221`，engineering patch `val_dice=0.1847`；
- 3-epoch CPU pilot：总耗时约 139.9 s，train loss `5.5221 → 4.7494 → 3.6347`，engineering patch val Dice `0.1847 → 0.1518 → 0.2480`；checkpoint/run 已落在 `experiments/20260826_115411_cpu_binary_engineering_ct_only`。这些数字只证明 CPU 工程训练能够学习并收敛起步，不属于论文正式性能；
- 新增 CPU 非 AMP autocast、disabled GradScaler 与 `allow_cpu` readiness 回归测试；
- 最终验证：`ruff check src tests → All checks passed!`；定向 6 passed；全项目 `pytest tests -q → 97 passed, 153 warnings`。

当前任务类型解释与建议：

- `binary_semantic`：只区分“椎骨前景 vs 背景”，所有 C/T/L 椎体标签合并为一个前景类别；适合当前样本量较小、首篇先验证骨结构分割/边界/三维重建的路线；
- `multiclass_semantic`：同时区分每节椎体类别，例如 T12、L1、L2 等；信息更丰富，但类别数多、不同 CT 覆盖范围不同、训练难度和数据需求明显更高；
- 当前工程 pilot 选择 binary semantic 仅作为低风险 CPU baseline，不等于替项目组锁定正式 task spec。正式任务仍需最终确认后再把 `task_locked=true` 并生成 formal split/config。

当前真正剩余的 P0 已从“人工 QC + GPU + task + split”收敛为：**正式 task 类型锁定 + 正式 patient-level split**。GPU 仅在后续希望大幅缩短 full-volume 训练/评估耗时时再考虑。


### 2026-08-26｜阶段 Q：锁定 binary semantic 任务与 CPU formal-pilot split

本阶段根据项目负责人明确选择，将首个椎体分割任务正式锁定为 `binary_semantic`，并把已完成的 10/10 人工 QC、当前笔记本 CPU 训练能力和 10 例 CTSpine1K 数据整合成一条可通过正式保护检查的流程 pilot。当前 10 例样本量仍不足以代表最终论文主实验规模，因此本阶段只声明“正式流程已锁定并可运行”，不把后续 pilot 指标直接等同于最终论文结论。

完成：

- 新建锁定任务规格 `configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json`：`task_id=vertebra_binary_ctspine1k_msd_t10_v1`、`task_type=binary_semantic`、`task_locked=true`、`num_classes=2`，CTSpine1K/VerSe 原始 `1..25` 椎体标签在训练时统一视为前景；
- 保留 `configs/task_specs/vertebra_task_template.json` 作为未锁定模板，不覆盖模板语义；
- 新建固定 patient-level split `data/splits/ctspine1k_msd_t10_binary_formal_pilot_v1.json`：7 train / 2 validation / 1 test，`formal_experiment=true`；
- 官方 `test_private` 病例 `ctspine1k-msd-t10-liver_169` 只进入 test，严格不参与训练与调参；
- 新建 `configs/orthopedic_ct_cpu_binary_formal_pilot_v1.yaml`，绑定 locked task spec 的 SHA-256 指纹，并保留当前 Ryzen 7 8745H CPU 可运行的 36³ patch 配置；
- `src.modeling.task_lock` 实测：`ready=true`、0 error / 0 warning；
- `src.modeling.formal_readiness --allow-cpu` 实测：task ready、formal preflight ready、7/2/1 split、10 例 pipeline 0.3.0、10/10 人工 QC 均通过，最终 `ready=true`、`blocker_count=0`；GPU report 仍如实显示 CPU-only，但在负责人明确允许 CPU 时不再构成 blocker；
- 全项目回归：`pytest tests -q → 97 passed, 153 warnings`；`ruff check src web tests → All checks passed!`；`git diff --check` 通过。

当前边界：

- 当前 10 例是正式流程 pilot，不足以支撑最终论文主实验统计强度；后续仍应扩大 CTSpine1K/同任务数据规模；
- 训练期间可使用 patch validation 进行 CPU 低成本模型选择，但最终可写入论文的 test 结果必须通过独立 full-volume evaluation 生成 Dice/HD95/ASSD/结构/uncertainty/calibration 等指标；
- 下一步进入 CT-only binary baseline 的更长训练，并在每个实质任务完成后立即同步 GitHub。


### 2026-08-26｜阶段 R：CPU binary CT-only 5-epoch formal-pilot baseline

在阶段 Q 已通过 `formal_readiness --allow-cpu` 的前提下，本阶段直接使用锁定的 `binary_semantic` task 与 7/2/1 formal-pilot split，在当前 Ryzen 7 8745H CPU 笔记本上运行 CT-only SegFormer3D baseline 5 epochs。训练只使用 7 个 train 病例，2 个 validation 病例用于 patch validation，官方 `test_private liver_169` 未参与训练或调参。

真实运行目录：`experiments/20260826_151810_cpu_binary_formal_pilot_ct_only`。

运行结果：

- epoch 1：train loss≈5.5221，patch-val Dice≈0.1847；
- epoch 2：train loss≈4.7494，patch-val Dice≈0.1518；
- epoch 3：train loss≈3.6347，patch-val Dice≈0.2480；
- epoch 4：train loss≈2.8162，patch-val Dice≈0.2719，为当前最佳；
- epoch 5：train loss≈2.6524，patch-val Dice≈0.2456；
- 最佳 checkpoint：`experiments/20260826_151810_cpu_binary_formal_pilot_ct_only/checkpoint/best.pt`；
- `config.yaml / split.json / run_metadata.json / history.csv / train.log / summary.json` 均已保存。

重要边界：

- 上述 Dice 是 36³ patch-validation proxy，只用于证明当前 CPU baseline 学习趋势和 checkpoint 选择链可运行，**不得作为论文正式 Dice**；
- train loss 总体持续下降，说明当前 CPU 环境和训练链确实在学习；validation 在极小样本下存在明显波动，符合 2 例 validation + 5 epoch 的 pilot 性质；
- 下一步必须使用当前最佳 checkpoint 对独立 test `liver_169` 做 full-volume evaluation，生成 Dice/HD95/ASSD/结构、uncertainty 与 calibration 等首批真实 pilot 指标；
- 完成本阶段后按项目负责人要求立即同步 GitHub，再进入下一项任务。


### 2026-08-26｜阶段 S：5-epoch checkpoint 独立 full-volume pilot test（GitHub 同步点 #3）

本阶段严格使用阶段 R 已固定的最佳 checkpoint（epoch 4）对独立 test `ctspine1k-msd-t10-liver_169` 运行 full-volume evaluation。该病例来自官方 `test_private`，此前从未进入 train 或 validation，因此本次测试仅用于在训练方案已固定后验证完整科研评估链，不用于继续调参。

真实输出目录：`experiments/evaluation_20260826_152657_test`。

真实 full-volume CPU test：

- Dice≈`0.0220767`；
- IoU≈`0.0111616`；
- Precision≈`0.0115527`；
- Recall≈`0.2479101`；
- HD95≈`190.931 mm`；
- ASSD≈`53.420 mm`；
- component_count_error=`157`；
- false_merge_count=`0`；
- false_break_count=`15`；
- 单病例 full-volume inference≈`9.2866 s`。

真实 uncertainty / calibration pilot：

- uncertainty→error AUROC≈`0.6137`；
- uncertainty→error AUPRC≈`0.4395`；
- Top-10% error recall≈`0.1385`；
- Top-10% ROI error rate≈`0.4909`；
- ECE≈`0.2990`；
- MCE≈`0.3378`；
- Brier≈`0.6347`；
- NLL≈`2.1329`；
- mean confidence≈`0.9445`；
- sampled calibration accuracy≈`0.6455`；
- confidence gap≈`0.2990`。

真实生成文件包括：

- `experiments/evaluation_20260826_152657_test/metrics_per_case.csv`；
- `experiments/evaluation_20260826_152657_test/summary.json`；
- `experiments/evaluation_20260826_152657_test/predictions/ctspine1k-msd-t10-liver_169/prediction.nii.gz`；
- `experiments/evaluation_20260826_152657_test/uncertainty/ctspine1k-msd-t10-liver_169/predictive_entropy.nii.gz`。

本阶段结论与边界：

- **5 epoch 模型严重欠训练。** 极低 Dice、很大的 HD95/ASSD 与大量 component error 说明当前模型尚不可用；
- 上述数字是 **10 例 formal-pilot 中唯一独立 test 病例**的工程/科研流程验证结果，不能宣传为足够规模的论文主实验，更不能写成最终论文性能；
- uncertainty/calibration 链已证明可以对真实 checkpoint 输出，但 segmentation 本身很差，因此这些数字也只作为 pipeline 证据，不做“模型已可靠/已校准”结论；
- test `liver_169` 不再用于下一阶段训练方案调参；后续所有训练改动只依据 train + validation 决策，待新方案完全固定后才允许再次独立 test；
- 下一阶段优先进入更长的 CPU binary CT-only baseline：先验证更合理的 patch ROI / foreground sampling / augmentation / scheduler 与 early stopping，再运行 10–20 epoch 阶段训练并继续逐任务同步 GitHub。


### 2026-08-26｜阶段 T：修复 CPU 跨 epoch 重复训练同一 patch 的采样缺陷

在同步点 #3 完成后，没有直接把 epochs 从 5 粗暴增加到 20/50，而是先检查 `ProcessedOrthopedicCTDataset` 与训练循环。确认当前 CPU 配置 `num_workers=0` 时，`__getitem__()` 的随机流只混入固定 `seed / torch.initial_seed() / index`，训练循环又没有向 Dataset 传入 epoch，因此同一病例在不同 epoch 会重复使用同一随机裁剪/增强随机流。这会显著降低 7 个 train 病例在长训练中的有效 patch 多样性，是 5-epoch pilot 之后必须先修复的训练机制问题。

完成：

- `src/modeling/dataset.py` 新增 `epoch` 状态与 `set_epoch()`；
- 训练 patch 的随机种子显式混入 epoch，使同一病例在不同 epoch 采样不同、同一 epoch 仍可复现；
- `src/modeling/train.py` 在每个训练 epoch 开始时调用 `train_ds.set_epoch(epoch)`；
- patch-validation Dataset 不调用 `set_epoch()`，继续固定在同一验证随机流，避免每个 epoch 因验证 patch 漂移而污染 checkpoint 比较；
- `run_metadata.json` 新增 `training_patch_sampling_epoch_aware=true` 与 `validation_patch_sampling_fixed_across_epochs` 追踪字段；
- 新增 `tests/test_dataset_epoch_sampling.py`，验证同一 epoch 重复读取完全一致、不同 epoch patch 改变、负 epoch 被拒绝；
- 本修复只改变 train/validation patch 采样机制，没有读取或利用独立 test `liver_169` 的结果做参数选择。

验证：

```text
pytest tests/test_dataset_epoch_sampling.py tests/test_training_smoke.py -q --disable-warnings
→ 5 passed

pytest tests -q --disable-warnings
→ 99 passed, 153 warnings

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

结论与下一步：

- 之前 5-epoch run 的工程链仍然有效，但其 CPU `num_workers=0` 训练 patch 多样性受该缺陷限制，因此不能仅通过“继续原配置多跑 epoch”来判断模型上限；
- 下一步只使用 train/validation 做 CPU 资源与 ROI 选择：先对 `36³ / 48³ / 64³` 训练 patch 做单步耗时与可运行性检查，再固定新的 10–20 epoch CT-only baseline 配置；
- 新方案完全固定前不再次运行 test，避免把 `liver_169` 用作调参集。


### 2026-08-26｜阶段 U：CPU ROI 单步 benchmark（GitHub 同步点 #5）

本阶段严格只使用 formal-pilot 的 `train` split，复用 `src.modeling.real_patch_smoke.run_real_patch_smoke()` 对 `36³ / 48³ / 64³` 做真实单 patch 训练链检查；每次均实际执行 Dataset 取样、SegFormer3D forward、Region Dice+CE loss、backward 与 AdamW optimizer step。没有读取 validation 指标做 ROI 选择，更没有访问独立 test `liver_169`。

真实结果：

- `36³`：3 次单步进程 wall-time 约 `1.929 / 1.840 / 1.821 s`，中位数约 `1.840 s`；单次结束 RSS≈`505 MB`，进程 peak working set≈`1529.8 MB`；
- `48³`：3 次约 `1.972 / 1.894 / 1.970 s`，中位数约 `1.970 s`；单次结束 RSS≈`523 MB`，peak working set≈`1530.2 MB`；
- `64³`：3 次约 `2.083 / 1.939 / 2.038 s`，中位数约 `2.038 s`；单次结束 RSS≈`565 MB`，peak working set≈`1530.7 MB`；
- 三档均产生有限 loss、有限梯度并成功完成 optimizer step；本阶段的 loss 为随机初始化下工程 smoke 输出，不属于模型性能。

ROI 决策：

- `48³` 明确可运行；
- `64³` 相对 `48³` 的进程 wall-time 中位数仅增加约 3.5%，结束 RSS 约增加 42 MB，当前约 20 GB RAM 机器可轻松承受；同时 64³ 提供明显更大的 3D 解剖上下文；
- 因此下一版 CT-only 长 baseline 固定使用 `64³`，不再沿用旧 `36³`；若后续完整 epoch 训练出现明显吞吐/内存异常，再回退到 `48³`，但不能使用 test 指标决定是否回退。

验证与边界：

- benchmark 复用了现有真实单 patch smoke 训练链，没有新增模型性能声明；
- ROI 选择只依据 train-side 工程资源可运行性，不使用 `liver_169`；
- 下一步立即新建不覆盖历史配置的 20-epoch CT-only baseline 配置，继续采用 epoch-aware training patch、固定 patch-validation、AdamW 与合理 scheduler/early stopping，然后启动真实训练。


### 2026-08-26｜阶段 V：锁定 64³ CPU CT-only long-v2 配置（GitHub 同步点 #6）

完成新的长训练 baseline 配置：`configs/orthopedic_ct_cpu_binary_long_v2.yaml`，不覆盖旧 36³ formal-pilot 配置和历史实验。

配置关键点：

- task 继续绑定已锁定 `vertebra_binary_ctspine1k_msd_t10_v1`，binary semantic，2 类；
- 数据 split 继续使用固定 7 train / 2 validation / 1 test 的 `ctspine1k_msd_t10_binary_formal_pilot_v1.json`；
- 输入保持 CT-only，不提前加入 bone-window；loss 保持 Region Dice + CE，不提前加入 Boundary/Topology；
- train ROI 从旧 `36³` 提升为 `64³`；foreground sampling=`0.8`；
- epochs=`20`，batch size=`1`，`num_workers=0`，继续使用已修复的 epoch-aware training patch；
- validation 继续固定 `64³` foreground patch，不随 epoch 漂移，只作为小样本 checkpoint proxy；
- optimizer=`AdamW(lr=1e-4, weight_decay=0.01)`；scheduler 保持 `2 epoch warmup + cosine annealing warm restarts (T0=10, min_lr=6e-6)`；
- early stopping patience 从旧 5 提高为 `8`，避免 2 个 validation 病例的短期波动过早终止训练；
- full-volume inference/evaluation ROI 仍保持 `128³`，最终 checkpoint 仍必须由 validation full-volume 复核后再允许 test。

验收：

- `formal_readiness` 显式传入已锁定 task spec 并使用 `--allow-cpu`：`ready=true`、`blocker_count=0`；
- formal preflight：10 例检查通过，split=7/2/1，pipeline 0.3.0=10，0 error / 0 warning；
- 当前 CPU-only GPU report 仍如实显示无 CUDA，但在显式 `--allow-cpu` 下不构成 blocker；
- 下一步直接启动该配置的真实 20-epoch 训练；训练期间不得读取 `liver_169` 做任何调参。


### 2026-08-26｜阶段 W：64³ CT-only long-v2 真实训练完成 early-stop（GitHub 同步点 #8）

真实 run：`experiments/20260826_162919_cpu_binary_long_v2_ct_only_roi64`。

本阶段使用已经锁定并通过 formal preflight 的 long-v2 配置，训练仅使用 7 个 train 病例，2 个 validation 病例使用固定 64³ foreground patch 做低成本 checkpoint proxy；独立 test `liver_169` 没有参与训练、scheduler、early stopping 或任何参数决策。

真实训练轨迹：epoch 1 `loss≈4.22295 / val≈0.36133`（当前最佳）；epoch 2 `3.99598 / 0.28158`；epoch 3 `3.19031 / 0.27159`；epoch 4 `3.32249 / 0.24556`；epoch 5 `2.68145 / 0.00103`；epoch 6 `2.68893 / 0.20272`；epoch 7 `1.85129 / 0.13539`；epoch 8 `2.11683 / 0.14045`；epoch 9 `2.36157 / 0.20656`。随后达到预先固定的 `early_stopping_patience=8`，正常停止。

产物完整：`config.yaml / split.json / run_metadata.json / history.csv / train.log / checkpoint/best.pt / checkpoint/last.pt / summary.json` 均存在；`best.pt` 对应 epoch 1，`last.pt` 对应 epoch 9。

解释与边界：

- 训练不是命令超时或异常中断，而是既定 early stopping 正常触发；因此不为了凑“20 epochs”绕过 validation 规则强行继续；
- train loss 明显低于早期 epoch，但固定 patch-val 未再超过 epoch 1，提示小样本下可能存在泛化下降或 validation proxy 偏差，不能仅凭 patch proxy 选择最终 baseline；
- 下一步只对 `liver_7/liver_8` 做 full-volume validation，比较 long-v2 `best.pt` 与必要候选 checkpoint，使用 Dice/IoU/Precision/Recall/HD95/ASSD/结构/uncertainty/calibration 与 inference time 锁定 baseline；
- 在 validation 完成并固定 ROI/训练设置/checkpoint 前，禁止再次访问 `liver_169` 做选择。


### 2026-08-26｜阶段 X：支持 CPU 分病例 full-volume evaluation（GitHub 同步点 #9）

在 long-v2 `best.pt` 对整个 validation split 一次性 full-volume 评估时，单条命令超过 300 秒工具执行上限；超时前已真实生成 `liver_7` prediction/entropy，说明评估本身可执行，但两例连续运行不适合当前工具时间窗。

为保持评估方法不变且避免重复浪费算力：

- `src/modeling/evaluate.py` 新增 `case_id` 参数与 CLI `--case-id`；
- 指定病例必须已经属于所选 `validation` 或 `test` split，否则直接 `ValueError`，不能借该参数跨 split 绕过数据隔离；
- formal preflight 仍在 case filter 前执行，配置/task/split/QC 保护不降低；
- 单病例仍使用完全相同的 full-volume sliding-window、区域/表面/结构、uncertainty、calibration 指标与 prediction/entropy 输出；
- `summary.json` 增加 `case_filter` 追踪字段；
- `tests/test_evaluate_smoke.py` 增加合法 case filter 与 split 越界拒绝回归测试；定向测试 `4 passed`、Ruff clean、`git diff --check` 通过。

下一步分别运行 `liver_7`、`liver_8` 的 long-v2 `best.pt` validation，再以同样方式评估必要候选 checkpoint，并只依据两例 validation 汇总锁定 baseline。


### 2026-08-26｜阶段 Y：完成 long-v2 full-volume validation 闭环并确认 patch selector 失真（GitHub 同步点 #10）

已核对以下四个已有 evaluation 的 `metrics_per_case.csv` 与 `summary.json`，没有重复运行：

- `evaluation_20260826_long_v2_best_val_liver7`：Dice≈0.02765，ASSD≈58.50 mm，component error=1843；
- `evaluation_20260826_long_v2_best_val_liver8`：Dice≈0.04632，ASSD≈55.04 mm，component error=1391；
- `evaluation_20260826_long_v2_last_val_liver7`：Dice≈0.05431，ASSD≈51.83 mm，component error=1211；
- `evaluation_20260826_long_v2_last_val_liver8`：Dice≈0.04475，ASSD≈49.74 mm，component error=957。

两例汇总：`best.pt`（epoch 1）平均 Dice≈0.03698、平均 ASSD≈56.77 mm、平均 component error≈1617；`last.pt`（epoch 9）平均 Dice≈0.04953、平均 ASSD≈50.78 mm、平均 component error≈1084。虽然 `last.pt` 仍然非常差，但在 full-volume validation 上总体优于由固定 patch validation 选出的 `best.pt`。

关键结论：epoch 1 固定 64³ patch-val Dice≈0.3613，而同 checkpoint 两例 full-volume 平均 Dice仅≈0.037，差异近一个数量级；因此当前固定 foreground patch validation 严重高估/误判全卷泛化，不能继续作为可靠 checkpoint selector。long-v2 当前也不能称为满意 baseline，更不能据此重新访问独立 test `liver_169`。

下一步已经提升为 P0：实现 full-volume-aware checkpoint selection，并优先排查 foreground/background sampling、Region Dice+CE 背景抑制、label mapping、train/inference preprocessing、sliding-window stitching/logits resize/threshold 与 class imbalance；validation 诊断需增加 prediction/GT foreground fraction、概率分布、connected components 与 false-positive 空间分布。

本同步点回归：

```text
pytest tests -q
→ 101 passed, 153 warnings

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```


### 2026-08-26｜阶段 Z：定位全卷假阳性主因并扩展多 patch 训练/诊断（GitHub 同步点 #11）

对 long-v2 的实际采样分布与四个已有 validation prediction 做了不访问 test 的诊断。7 个 train 全卷平均前景体素占比仅≈0.68%；但按 long-v2 原配置 `foreground_probability=0.8`、每病例每 epoch 仅 1 patch 复现 epoch 1–9 的 63 个真实随机流后，训练 patch 平均前景≈21.2%，中位数≈24.3%，仅 14/63 为全背景 patch。也就是说训练数据分布中的前景先验约被放大 30 倍。

利用已有 full-volume prediction 直接统计/由 precision-recall 交叉核对：`liver_7/liver_8` GT 前景约 0.70% / 0.57%，而 long-v2 prediction 前景约 14.5%–17.1%，约为 GT 的 24–27 倍。该数量级与训练 patch prior 失配高度一致，因此当前最有证据的首要根因是 foreground-biased sampling 过强 + 每 epoch patch 数过少，而不是简单“再多训几个 epoch”。当前检查未发现 binary label mapping、train/evaluate 轴变换、sliding-window predictor resize 的直接实现错误。

工程修复：

- `ProcessedOrthopedicCTDataset` 新增 `patches_per_case`，训练时数据集长度扩展为 `case_count × patches_per_case`；
- 同病例不同 patch slot 与不同 epoch 使用独立、可复现随机流，保留原 epoch-aware sampling；
- `train.py` 接入 `training.patches_per_case`，并写入 `run_metadata.json` / `summary.json`；
- `evaluate.py` 新增 `prediction_foreground_fraction`、`target_foreground_fraction`、`prediction_to_target_foreground_ratio` 到逐病例 CSV 与 summary，后续无需再手工反推假阳性膨胀；
- 新增多 patch 随机流/参数校验与 evaluation foreground-fraction 回归测试。

本同步点回归：

```text
pytest tests -q
→ 103 passed, 153 warnings

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

下一步直接新建 CT-only v3，不覆盖 long-v2：降低 foreground sampling、提高 patches_per_case，并把 `validation.patch_mode=false`，让 checkpoint/early stopping 直接依据 `liver_7/liver_8` full-volume Dice；仍禁止访问 `liver_169`。


### 2026-08-26｜阶段 AA：锁定 balanced + full-volume validation v3 配置（GitHub 同步点 #12）

新建 `configs/orthopedic_ct_cpu_binary_balanced_fullval_v3.yaml`，只针对已定位的 sampling/checkpoint 问题做最小可归因修改：

- CT-only、64³ ROI、SegFormer3D 结构、Region Dice+CE、AdamW lr=1e-4、weight_decay=0.01 保持不变；
- `foreground_probability: 0.8 → 0.25`，显著增加纯背景/低前景 patch；
- `training.patches_per_case: 4`，7 个 train 病例每 epoch 从 7 个 patch 提升到 28 个 patch；
- `validation.patch_mode=false`，不再用固定 foreground patch proxy，checkpoint 与 early stopping 直接依据两例 validation full-volume Dice；
- 上限 12 epoch、patience=4；如果 full-volume Dice/假阳性比例明显不改善，允许提前停止，不为凑轮数继续浪费 CPU。

readiness：

```text
formal_readiness --allow-cpu
→ ready=true
→ blocker_count=0
→ preflight 10 cases / split 7-2-1 / 0 error / 0 warning
```

该配置仍严格不访问 `liver_169`；下一步直接启动 v3 真实训练并逐 epoch full-volume validation。


### 2026-08-26｜阶段 AB：balanced v3 epoch 1/2 + detailed full-volume validation（GitHub 同步点 #13）

v3 run：`experiments/20260826_173511_cpu_binary_balanced_fullval_v3_roi64`。当前真实训练历史：

- epoch 1：train loss≈`2.55371`，两例 full-volume val Dice≈`0.054070`，std≈`0.010840`，validation inference total≈`127.26 s`，lr=`5e-5`；
- epoch 2：train loss≈`1.94022`，两例 full-volume val Dice≈`0.040839`，std≈`0.001075`，validation inference total≈`130.42 s`，lr=`1e-4`；
- 当前最佳 checkpoint：`checkpoint/best.pt`，对应 epoch 1。

对 epoch 1 `best.pt` 只在 validation split 分病例做 detailed full-volume evaluation，未访问 `liver_169`：

- `liver_7`：Dice≈`0.04323`，IoU≈`0.02209`，Precision≈`0.02753`，Recall≈`0.10055`，HD95≈`199.91 mm`，ASSD≈`56.25 mm`；prediction foreground≈`2.555%`，GT≈`0.700%`，ratio≈`3.65`；pred/target components=`1581/3`，component error=`1578`，false merge=`1`，false break=`63`，inference≈`52.74 s`；
- `liver_8`：Dice≈`0.06491`，IoU≈`0.03354`，Precision≈`0.04267`，Recall≈`0.13561`，HD95≈`175.46 mm`，ASSD≈`48.26 mm`；prediction foreground≈`1.799%`，GT≈`0.566%`，ratio≈`3.18`；pred/target components=`1599/2`，component error=`1597`，false merge=`0`，false break=`68`，inference≈`83.14 s`。

两例平均 Dice≈`0.05407`、Precision≈`0.03510`、ASSD≈`52.25 mm`、foreground ratio≈`3.42`。相较 long-v2 `last.pt` 平均 Dice≈`0.04953`、Precision≈`0.02577`，且 prediction/GT foreground ratio 约 `24–27`，balanced sampling 已显著压低整卷前景过预测并带来小幅 Dice/Precision 改善，说明方向有效。

但结构错误没有同步改善：v3 两例 component count error 平均≈`1587.5`，反而高于 long-v2 `last.pt` 的≈`1084`；Recall 也从 long-v2 的高过预测状态明显下降。当前结论是“前景先验失配已大幅缓解，但预测仍高度碎片化，baseline 仍不可靠”，不能把 v3 视为完成，更不能访问独立 test。

本阶段回归：`pytest tests -q → 103 passed`；`ruff check src web tests → All checks passed`；`git diff --check → 通过`。

下一步先继续 epoch 3；若 full-volume Dice 不回升、foreground ratio/Precision 恶化或 fragmentation 继续严重，则停止机械续训并优先检查 Region Dice+CE 权重、背景分类约束与 sampling 参数。

### 2026-08-26｜阶段 AC：v3 epoch 3 背景塌缩诊断（GitHub 同步点 #14）

从同一 v3 run 的 `last.pt` resume 到总 epoch 3，真实结果：train loss≈`1.63162`，full-volume val Dice≈`1.30e-11`，说明训练损失继续下降但全卷泛化发生灾难性塌缩，因此没有继续 epoch 4。

对 epoch 3 `last.pt` 的 validation detailed evaluation：`liver_7` Dice/Precision/Recall=`0/0/0`，prediction foreground≈`0.326%`、GT≈`0.700%`、ratio≈`0.466`，pred components=`351`；`liver_8` Dice/Precision/Recall=`0/0/0`，prediction foreground≈`0.150%`、GT≈`0.566%`、ratio≈`0.265`，pred components=`341`。这些预测与真实前景完全不重叠，属于背景塌缩/错误位置少量碎片，而不是继续训练可自然恢复的普通波动。

代码检查显示 `RegionDiceCELoss3D` 当前把 foreground Dice loss 与未加权全体素 CrossEntropy 以默认 `1:1` 相加；`train.build_criterion()` 对 `region_dice_ce` 只传 `include_background`，尚未读取 YAML 的 `dice_weight/ce_weight`。在 v3 已大幅增加背景 patch 的条件下，这一实现会让 CE 的海量背景体素更容易主导优化，和 epoch 3 collapse 的方向一致。下一步采用最小可归因修复：保持 sampling 不变，仅让 YAML 可配置 Dice/CE 内部权重并降低 CE 权重做新 run validation。


### 2026-08-26｜阶段 W：实现并实测可靠 checkpoint resume（GitHub 同步点 #7）

为了避免 CPU 20/50 epoch 长训练因单次命令执行上限中断后只能重头开始，本阶段把 checkpoint resume 提前实现并完成真实端到端验证。

代码完成：

- `train.py` 新增 `--resume <checkpoint>`；`--max-epochs` 在 resume 模式下表示该 run 的总目标 epoch，而不是“再训练多少轮”；
- 每个完成的 epoch 除最佳 `best.pt` 外，固定写入 `checkpoint/last.pt`；
- checkpoint 现在同时保存/恢复：model state、optimizer state、scheduler state、当前 epoch、best validation Dice、early-stopping 连续未改善计数；
- 额外保存/恢复 Python / NumPy / Torch / CUDA RNG 状态，避免续训时 DataLoader shuffle 和其它随机流无条件重置；
- `WarmupCosineRestarts` 新增 `load_state_dict()`；
- resume 时复用原 run 目录，不覆盖 `config.yaml / split.json`，`history.csv` 改为追加；`run_metadata.json` 记录 resume event；
- checkpoint 中 config 与当前 config 不一致时拒绝 resume，防止把不同实验错误拼接在一起；
- `summary.json` 新增 `last_epoch / target_max_epochs / epochs_without_improvement / resumed`。

真实 64³ resume smoke：

- run：`experiments/20260826_162450_cpu_binary_long_v2_ct_only_roi64`；
- 第一次运行 `--max-epochs 1`：epoch 1 train loss≈`4.22295`，固定 patch-val Dice≈`0.36133`；
- 随后从同一 run 的 `checkpoint/last.pt` 执行 `--max-epochs 2 --resume ...`：成功直接进入 epoch 2，train loss≈`3.99598`，patch-val Dice≈`0.28158`；
- `history.csv` 保持 epoch `1 → 2` 连续，没有新建第二个 run；
- `best.pt` 与 `last.pt` 均存在，resume metadata 记录 checkpoint epoch=1、target=2；
- 这些 1–2 epoch 数值只是 resume 工程验证，不作为 20-epoch baseline 结果，也不访问 test `liver_169`。

回归：

```text
pytest tests -q --disable-warnings
→ 100 passed, 153 warnings

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

下一步：

- 先提交并同步本 resume 机制；
- 再从干净 Git 状态新建真正的 20-epoch long-v2 run，并利用 `last.pt` 分段续训到总目标 20 epoch；
- 20 epoch 完成后只根据 train + validation 判断是否继续 50 epoch；在 full-volume validation 完成、最终 checkpoint 固定之前，不再次访问 test。


### 2026-08-26｜阶段 AD：v4 Region Dice+CE 权重工程闭环（GitHub 同步点 #15）

在 v3 epoch 3 已确认背景塌缩后，本阶段严格按“一次只改一个主要变量”的原则，仅修复 Region Dice+CE 内部权重的配置链，不改变 v3 的数据 split、CT-only 输入、64³ ROI、`foreground_probability=0.25`、`patches_per_case=4`、full-volume validation 或 optimizer/scheduler。

完成内容：

- `src/modeling/train.py`：`region_dice_ce` 现在真实从 YAML 读取 `loss.dice_weight` 与 `loss.ce_weight`；
- `src/modeling/joint_loss.py`：新增 Dice/CE 权重非负且不能同时为 0 的合法性检查；
- `tests/test_training_smoke.py`：新增回归测试，确认 `build_criterion()` 能正确读取 `dice_weight=1.0 / ce_weight=0.25`；
- 新建 `configs/orthopedic_ct_cpu_binary_balanced_loss_v4.yaml`：保持 v3 其它条件不变，仅把 Region CE 相对权重从 1.0 降到 0.25；
- 未访问独立 test `liver_169`，本次所有检查只使用锁定 task、formal-pilot split 与 train/validation 规则。

真实验收：

```text
formal_readiness --task-spec configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json \
  --config configs/orthopedic_ct_cpu_binary_balanced_loss_v4.yaml --allow-cpu
→ ready=true
→ blocker_count=0
→ preflight 10 cases / split 7-2-1 / 0 error / 0 warning

pytest tests -q
→ 104 passed, 153 warnings

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

下一步直接启动新的 v4 run，先只跑 epoch 1；随后仅对 `liver_7/liver_8` 做 full-volume validation 与 detailed evaluation，联合检查 Dice、Precision、Recall、HD95、ASSD、prediction/target foreground ratio、component fragmentation、uncertainty/calibration 与 inference time。若 epoch 1 明显改善再续训 epoch 2/3；若仍出现“epoch 1 尚可、后续塌缩”，下一优先变量是把学习率从 `1e-4` 降到 `5e-5`，而不是继续机械增加 epoch。


### 2026-08-26｜阶段 AE：v4 epoch 1 真实验证——CE=0.25 导致前景约束过弱（GitHub 同步点 #16）

v4 run：`experiments/20260826_213818_cpu_binary_balanced_loss_v4_roi64`。本阶段只训练到 epoch 1，不访问独立 test `liver_169`。

真实训练/full-volume validation：

- epoch 1 train loss≈`1.350736`；
- 两例 full-volume validation mean Dice≈`0.047625`，std≈`0.001878`；
- validation inference total≈`134.19 s`；
- epoch 1 lr=`5e-5`；
- 当前 v4 `best.pt` 即 epoch 1：`experiments/20260826_213818_cpu_binary_balanced_loss_v4_roi64/checkpoint/best.pt`。

分病例 detailed validation：

- `liver_7`：Dice≈`0.04950`，Precision≈`0.02888`，Recall≈`0.17306`，HD95≈`218.02 mm`，ASSD≈`55.76 mm`，prediction/GT foreground ratio≈`5.99`，pred/target components=`2128/3`，component error=`2125`，false merge=`1`，false break=`89`，inference≈`54.62 s`；
- `liver_8`：Dice≈`0.04575`，Precision≈`0.02672`，Recall≈`0.15878`，HD95≈`208.74 mm`，ASSD≈`54.91 mm`，prediction/GT foreground ratio≈`5.94`，pred/target components=`1863/2`，component error=`1861`，false merge=`0`，false break=`98`，inference≈`87.58 s`。

两例平均：

- Dice≈`0.04762`；
- Precision≈`0.02780`；
- Recall≈`0.16592`；
- HD95≈`213.38 mm`；
- ASSD≈`55.33 mm`；
- prediction/GT foreground ratio≈`5.97`；
- component count error≈`1993`。

与 v3 epoch 1 对比：v3 mean Dice≈`0.05407`、Precision≈`0.03510`、foreground ratio≈`3.42`、component count error≈`1587.5`。因此 v4 并未改善，而是把全卷前景过预测与碎片化再次放大。当前证据更支持“CE=0.25 过低，背景分类约束不足”，而不是“继续降低 CE 可以修复背景塌缩”。

决策：

- 不继续机械运行 v4 epoch 2；
- 恢复 Region Dice/CE=`1:1`；
- 保持 v3 的 CT-only、64³ ROI、foreground_probability=0.25、patches_per_case=4、full-volume validation 不变；
- 下一单变量实验优先把 optimizer peak lr 从 `1e-4` 降到 `5e-5`，验证 v3 在 lr 从 5e-5 升到 1e-4 后恶化、随后塌缩是否主要由学习率导致；
- baseline 完全锁定前继续禁止访问 `liver_169`。

说明：第一次串行运行两例 detailed evaluation 时单命令超过 300 秒工具上限；`liver_7` 已完整落盘，`liver_8` 输出目录仅被创建但未写结果。为遵守“不删除/覆盖已有实验”规则，没有删除该空目录，而是使用 `experiments/evaluation_20260826_v4_e1_val_liver8_retry1` 完成 `liver_8` 评估。


### 2026-08-26｜阶段 AF：锁定 balanced-lr v5 单变量配置（GitHub 同步点 #17）

基于 v4 CE=0.25 的负结果，本阶段不继续降低 CE，也不同时修改 sampling/ROI/augmentation，而是回到 v3 的 Region Dice+CE=`1:1`，只测试学习率因素。

新建：`configs/orthopedic_ct_cpu_binary_balanced_lr_v5.yaml`。

与 v3 的方法学差异：

- optimizer peak lr：`1e-4 → 5e-5`；
- `dice_weight=1.0 / ce_weight=1.0` 仅把 v3 原本的默认 1:1 显式写入 YAML，不构成方法变化；
- CT-only、64³ ROI、foreground_probability=0.25、patches_per_case=4、validation.patch_mode=false、split、seed、模型结构、weight decay、scheduler 类型与 sliding-window inference 均保持不变。

动机：v3 epoch 1 的实际 lr=`5e-5` 时两例 full-volume Dice≈0.05407，为当前最佳；epoch 2 lr 升到 `1e-4` 后 Dice 降到≈0.04084，epoch 3 在≈`9.77e-5` 时完全背景塌缩。v4 又显示单纯把 CE 降到 0.25 会导致前景过预测/碎片化加重，因此当前更有证据的下一变量是较低 peak lr。

readiness：

```text
formal_readiness --task-spec configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json \
  --config configs/orthopedic_ct_cpu_binary_balanced_lr_v5.yaml --allow-cpu
→ ready=true
→ blocker_count=0
→ preflight 10 cases / split 7-2-1 / 0 error / 0 warning
```

下一步直接启动 v5 epoch 1；若 full-volume 指标方向合理则分段续训 epoch 2/3，并继续只使用 `liver_7/liver_8` 做所有参数决策，禁止提前访问 `liver_169`。


### 2026-08-26｜阶段 AG：GitHub Actions Python CI 新鲜 runner 可复现性修复（GitHub 同步点 #18）

收到 GitHub Actions `Python tests and lint` 失败通知后，直接读取失败 job 日志定位到 5 个 pytest 失败：其中 4 个是 GitHub 新鲜 runner 没有被 `.gitignore` 排除的 `third_party/SegFormer3D` checkout，导致 `SegFormer3DUpstreamNotFound`；另 1 个是 `tests/test_gpu_environment.py` 把“当前 Python 必须位于项目 `.venv`”硬编码为 `True`，与 GitHub Actions 的 `setup-python` runner 环境不兼容。Ruff 与前端/仓库静态检查本身均已通过。

本阶段修复：

- `.github/workflows/ci.yml`：Python job 在测试前调用 `env/fetch_segformer3d.ps1`，确保新鲜 runner 自动准备 SegFormer3D 上游；
- `env/fetch_segformer3d.ps1`：全新获取时固定检出上游 `e314242f14b6731458130809945a0ee27f4298bd`，避免 CI 随上游 `main` 漂移；随后自动应用本项目受版本控制的 PyTorch 2.1 TorchScript 兼容补丁；已有本地 `third_party/SegFormer3D` 目录仍保持“不覆盖”原则，因此不会破坏当前机器已有第三方本地 patch；
- 新增 `env/patches/segformer3d_torch21_cube_root.patch`：把此前仅存在于本地第三方 working tree 的 `cube_root()` `int(round(...))` 兼容修复变成可复制、可审计的补丁文件；
- `tests/test_gpu_environment.py`：改为验证 `project_venv` 是 machine-readable bool；若环境确实不在项目 `.venv`，则要求 report 的 `issues` 明确包含对应提示，不再错误假设所有合法测试环境都必须是 Windows 项目 `.venv`；
- `third_party/README.md` 与 `TASKS.md`：同步记录固定上游提交、受版本控制补丁与 CI 新鲜 runner 获取链。

本地回归：

```text
.venv/Scripts/python.exe -m pytest tests -q
→ 104 passed, 153 warnings

.venv/Scripts/python.exe -m ruff check src web tests
→ All checks passed!

PyYAML parse .github/workflows/ci.yml
→ OK

PowerShell parser: env/fetch_segformer3d.ps1
→ OK

git -C third_party/SegFormer3D apply --reverse --check ../../env/patches/segformer3d_torch21_cube_root.patch
→ 通过（确认受版本控制 patch 与当前本地兼容 diff 一致）

git diff --check
→ 通过
```

云端最终验收：修复提交 `6c5eac1` 推送到 `origin/main` 后触发 GitHub Actions CI run `32978966080`（run #18）。`Fetch pinned SegFormer3D upstream`、`Run Ruff`、`Run pytest` 全部 success；`Python tests and lint` 与 `Frontend and repository static checks` 两个 job 均最终为 `success`。因此可以正式确认：此前邮件中的 GitHub Actions Python CI 失败已修复，新鲜 Ubuntu runner 能按公开仓库内容自动获取固定上游、应用兼容补丁并完成全套 CI。


### 2026-08-26｜阶段 AG：v5 低峰值学习率实验确认前景泛滥（GitHub 同步点 #18）

v5 run：`experiments/20260826_221337_cpu_binary_balanced_lr_v5_roi64`。该配置保持 v3 的 CT-only、64³ ROI、`foreground_probability=0.25`、`patches_per_case=4`、Region Dice/CE=1:1 与 full-volume validation 不变，仅将 optimizer peak lr 从 `1e-4` 降到 `5e-5`；但仍保留 `warmup_epochs=2`。

真实训练：

- epoch 1：train loss≈`2.29635`，两例 full-volume val Dice≈`0.0318517`，lr=`2.5e-5`；
- epoch 2：train loss≈`2.08801`，两例 full-volume val Dice≈`0.0326909`，lr=`5e-5`；
- `best.pt` 对应 epoch 2。

对 v5 `best.pt` 的 validation detailed evaluation：

- `liver_7`：Dice≈`0.03185`，Precision≈`0.01621`，Recall≈`0.89485`，HD95≈`210.68 mm`，ASSD≈`60.61 mm`，prediction/GT foreground ratio≈`55.19`，pred/target components=`207/3`，component error=`204`；
- `liver_8`：Dice≈`0.03353`，Precision≈`0.01707`，Recall≈`0.93585`，HD95≈`202.54 mm`，ASSD≈`61.76 mm`，prediction/GT foreground ratio≈`54.82`，pred/target components=`187/2`，component error=`185`。

结论：v5 不是背景塌缩，而是严重前景泛滥。降低 peak lr 同时保留 2-epoch warmup，使 epoch 1 实际 lr 只有 `2.5e-5`，模型到 epoch 2 仍保持极高 Recall、极低 Precision 和约 55× 的全卷前景膨胀；这明显劣于 v3 epoch 1 的两例平均 Dice≈`0.05407`、Precision≈`0.03510`、foreground ratio≈`3.42`。因此不继续 v5 epoch 3。

下一实验不再把“更低 lr”与“更长 warmup”混在一起：保留 v3 其它条件，第一轮直接达到 `5e-5`，之后学习率始终不超过 `5e-5`。这样更接近 v3 epoch 1 的有效起点，同时避免 v3 epoch 2 升到 `1e-4` 后性能下降。整个 v5 训练和评估只使用 train + `liver_7/liver_8` validation，未访问 `liver_169`。


### 2026-08-26｜阶段 AH：锁定 v6 单 warmup 低学习率配置（GitHub 同步点 #19）

新建 `configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml`。v6 继承 v5 的 CT-only、64³ ROI、`foreground_probability=0.25`、`patches_per_case=4`、Region Dice/CE=1:1、optimizer peak lr=`5e-5`、full-volume validation、task/split/seed 等全部条件，仅将 `scheduler.warmup_epochs` 从 2 改为 1。

代码核对 `WarmupCosineRestarts.step()`：warmup epoch 内学习率为 `base_lr * epoch / warmup_epochs`，因此 v6 epoch 1 会直接使用 `5e-5`；之后进入 cosine schedule，学习率不会超过 base lr=`5e-5`。这与 v5 epoch 1 只有 `2.5e-5` 的情况不同，更接近 v3 epoch 1 的有效起点，同时避免 v3 epoch 2 升至 `1e-4`。

真实 readiness：

```text
formal_readiness --task-spec configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json \
  --config configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml --allow-cpu
→ ready=true
→ blocker_count=0
→ preflight 10 cases / split 7-2-1 / 0 error / 0 warning
```

下一步只跑 v6 epoch 1，并仅使用 `liver_7/liver_8` full-volume validation 判断；若能复现或超过 v3 epoch 1 的 Dice/Precision/foreground ratio 再继续 epoch 2/3，否则立即停止。独立 test `liver_169` 继续禁止访问。


### 2026-08-26｜阶段 AI：v6 epoch 1/2 与 epoch 2 detailed validation 结果回填

核对真实 run：`experiments/20260826_224150_cpu_binary_balanced_lr_v6_roi64`。`history.csv`、`best.pt`、`last.pt` 与 `summary.json` 均存在；`summary.json` 记录 `best_val_dice=0.05407000716611769`、`last_epoch=2`，因此 `best.pt` 保持 epoch 1。

真实训练/validation：

```text
epoch 1: train_loss=2.5537127596991405, val Dice=0.05407000716611769, lr=5e-5
epoch 2: train_loss=1.9332212380000524, val Dice=0.0323937293334203, lr=4.892324335849338e-5
```

v6 epoch 1 几乎精确复现 v3 epoch 1，说明第一轮直接使用 `5e-5` 是可行的，也反证 v5 第一轮 `2.5e-5` 的长 warmup 是错误方向。但 v6 epoch 2 在学习率始终未超过 `5e-5` 的情况下仍明显恶化，因此“v3 epoch 2 升至 `1e-4` 是主要根因”的假设基本被否定。

对 v6 epoch 2 `last.pt` 的 validation detailed evaluation：

- `liver_7`：Dice≈`0.0321042`，Precision≈`0.0163178`，Recall≈`0.985616`，HD95≈`222.03 mm`，ASSD≈`76.73 mm`，prediction foreground≈`42.257%`、GT≈`0.6996%`、ratio≈`60.40`，pred/target components=`90/3`，component error=`87`；
- `liver_8`：Dice≈`0.0326833`，Precision≈`0.0166134`，Recall≈`0.999194`，HD95≈`211.83 mm`，ASSD≈`81.00 mm`，prediction foreground≈`34.039%`、GT≈`0.5660%`、ratio≈`60.14`，pred/target components=`67/2`，component error=`65`。

component 数量相对早期碎片化结果虽下降，但这是因为模型把大范围背景连成巨大前景区域，不是正确结构改善。v6 epoch 2 明确属于严重 foreground explosion，因此不继续机械 epoch 3。

下一步优先级转为真实 sampling 复现：使用 `ProcessedOrthopedicCTDataset` 当前实际 epoch-aware sampling 和固定 seed，复现 v3/v6 epoch 1/2、v3 epoch 3 的 28 个 training patch foreground fraction 分布，先证实或否定 epoch-to-epoch sampling prior 漂移，再决定 v7 的唯一变量。独立 test `liver_169` 继续禁止访问。


### 2026-08-26｜阶段 AJ：复现真实 epoch patch prior，并把 sampling statistics 接入训练

使用 `configs/orthopedic_ct_cpu_binary_balanced_fullval_v3.yaml` 的真实 `ProcessedOrthopedicCTDataset`、`seed=42`、`foreground_probability=0.25`、`patches_per_case=4`，按训练代码相同的 `set_epoch(epoch)` 与随机种子组合复现 7 个 train 病例 × 4 patch 的实际 label foreground fraction。为避免重复解压大体积 NIfTI，同一病例只加载一次，但每个 patch 仍由 Dataset 自己的 `__getitem__`、真实 crop/flip/random stream 生成。

结果：

```text
epoch 1: mean=0.079073, median=0, std=0.121046, min=0, max=0.329468,
         q25=0, q75=0.162591, q90=0.283113, background=18/28, foreground=10/28
epoch 2: mean=0.088408, median=0, std=0.129709, min=0, max=0.388599,
         q25=0, q75=0.190886, q90=0.283283, background=18/28, foreground=10/28
epoch 3: mean=0.056800, median=0, std=0.105252, min=0, max=0.356922,
         q25=0, q75=0.068274, q90=0.239979, background=20/28, foreground=8/28
```

病例级波动更明显：epoch 1 的 `liver_2` 与 `liver_6` 4/4 patch 均为纯背景；epoch 2 则重新分配 foreground exposure。说明当前独立 Bernoulli foreground-aware sampling 会造成真实的 case/epoch prior 波动。不过 v6 epoch 1→2 的 overall mean 仅约 7.91%→8.84%、纯背景 patch 同为 18/28，因此该波动不能单独解释 validation 从约 3.4× foreground ratio 突然恶化到约 60×；当前结论是 sampling 稳定性确有问题，但不是已证实唯一根因。

为让后续每个 run 都能直接验证“模型变化是否对应 sampling prior 漂移”，`src/modeling/train.py` 新增 `sampling_stats.csv`，从模型实际收到的 training label 逐 epoch 记录：patch_count、foreground fraction mean/median/std/min/max、q10/q25/q75/q90、foreground/background patch count；metadata 同时标记 `training_sampling_stats_logged=true`。新增 `tests/test_training_sampling_stats.py`， focused dataset/sampling tests 共 6 passed，Ruff clean。

下一步只改一个变量做 v7：把每病例每 epoch 的 foreground-aware/random patch 配额从独立 Bernoulli 改为固定配额，保持 v6 的 CT-only、64³、Dice/CE=1:1、peak lr=5e-5、warmup=1、cosine、full-volume validation 等全部不变。独立 test `liver_169` 继续禁止访问。


### 2026-08-26｜阶段 AK：完成 v7 fixed-per-case sampling 工程闭环准备

核对上一轮未提交修改后确认工作区只包含 `src/modeling/dataset.py`、`src/modeling/train.py`、`tests/test_dataset_epoch_sampling.py` 与新配置 `configs/orthopedic_ct_cpu_binary_stable_sampling_v7.yaml`，远程基线仍为 `0fa2dab2c7eeb9fd0e010d4eabca87ed8856a117`，未执行 reset/restore 覆盖历史修改。

v7 新增 `foreground_sampling_mode=fixed_per_case`：在当前 `patches_per_case=4`、`foreground_probability=0.25` 下，每病例每 epoch 固定 1 个 foreground-aware slot 与 3 个 random slot。`train.py` 已把 sampling mode 传入 Dataset 并写入 run metadata。配置与 v6 逐项对比确认，除 `experiment_name` 外仅新增该 sampling mode；CT-only、64³ ROI、Region Dice+CE=1:1、peak lr=5e-5、warmup=1、cosine scheduler、seed、split 与 full-volume validation 均保持不变，因此 v7 仍是单主要变量实验。

工程验证重新实跑：

```text
pytest tests -q
→ 108 passed

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

下一步在完成本次 commit/push 且确认 `HEAD == origin/main` 后，立即执行 v7 `formal_readiness --allow-cpu`；必须 `ready=true / blocker_count=0` 才启动 epoch 1。epoch 1 完成后必须核对 `history.csv`、`sampling_stats.csv`、`best.pt`、`last.pt`、`summary.json`、`train.log`，随后对 `liver_7/liver_8` 做 full-volume/detailed validation。独立 test `liver_169` 继续禁止访问。


### 2026-08-27｜阶段 AL：完成 v7 epoch 1 detailed validation 并停止 fixed-per-case sampling 实验

已从真实 run `experiments/20260827_000843_cpu_binary_stable_sampling_v7_roi64` 核验 v7 epoch 1：`train_loss=2.398116941962923`，两例 full-volume validation mean Dice=`0.04561579108399831`，std=`0.03207655051989757`，validation inference total≈`123.24 s`，lr=`5e-5`；`sampling_stats.csv` 显示 28 个 training patch，foreground fraction mean≈`0.06460`、median=`0`、std≈`0.11167`、max≈`0.36113`，label-positive patches=`10/28`、pure-background patches=`18/28`。这与 fixed-per-case 设计不矛盾：固定的是每病例 1 个 foreground-aware slot + 3 个 random slot，random patch 仍可能采到前景。

对 v7 epoch 1 `best.pt` 只在 validation split 做 detailed full-volume evaluation，未访问独立 test `liver_169`：

- `ctspine1k-msd-t10-liver_7`：Dice=`0.01353924`，IoU=`0.00681576`，Precision=`0.00790974`，Recall=`0.04696506`，HD95≈`184.59 mm`，ASSD≈`56.61 mm`；prediction foreground≈`4.154%`，GT≈`0.700%`，ratio≈`5.94`；pred/target components=`1363/3`，component error=`1360`，false merge=`1`，false break=`103`，inference≈`75.89 s`；uncertainty AUROC/AUPRC≈`0.91614/0.33726`，ECE≈`0.03153`，MCE≈`0.19165`，Brier≈`0.07959`，NLL≈`0.19557`。
- `ctspine1k-msd-t10-liver_8`：Dice=`0.07769234`，IoU=`0.04041618`，Precision=`0.04419672`，Recall=`0.32087773`，HD95≈`178.33 mm`，ASSD≈`52.88 mm`；prediction foreground≈`4.109%`，GT≈`0.566%`，ratio≈`7.26`；pred/target components=`1265/2`，component error=`1263`，false merge=`0`，false break=`53`，inference≈`121.12 s`；uncertainty AUROC/AUPRC≈`0.93387/0.34531`，ECE≈`0.02835`，MCE≈`0.19835`，Brier≈`0.07196`，NLL≈`0.16423`。

两例平均：Dice≈`0.04562`、Precision≈`0.02605`、Recall≈`0.18392`、HD95≈`181.46 mm`、ASSD≈`54.74 mm`、prediction/GT foreground ratio≈`6.60`、component error≈`1311.5`。对比 v3/v6 epoch 1（mean Dice≈`0.05407`；v3 detailed mean Precision≈`0.03510`、foreground ratio≈`3.42`、component error≈`1587.5`），v7 虽未出现 v6 epoch 2 约 `60×` foreground explosion，但 Dice 与 Precision 更差、foreground ratio 更高；结构碎片数量虽略低，但没有形成足以抵消区域指标退化的整体改善。`liver_7` 还明显劣于 v3 epoch 1；`liver_8` 单例较好，但不足以使两例整体接近或超过 v3/v6 epoch 1。

因此按预先锁定的判定规则停止 v7，不机械运行 epoch 2。结论限定为：**在当前 v7 单变量实验条件下，fixed-per-case quota 没有改善 epoch 1 full-volume validation，因此 sampling instability 不是足以解决当前 baseline instability 的主要干预。** 不能据此写成“sampling 完全无影响”或“sampling 与问题无关”。

下一步进入更直接的优化动力学 diagnostics：优先记录 validation logits/probability 分布、Dice/CE 分项与前景/背景 CE contribution、最终 segmentation head 的 bias/weight/gradient norm，并用历史 v3/v6 validation checkpoint 检查 epoch 1→2 是否存在整体 foreground probability/bias 漂移；同时针对 full-volume inference 与 patch training 的 normalization、logits resize、softmax/argmax、padding/cropping 等一致性增加回归测试。独立 test `liver_169` 继续禁止访问。


### 2026-08-27｜阶段 AM：完成 checkpoint diagnostics 工程闭环并锁定 v8 BN-running-stat 单变量方向

从远程同步点 `353197296fb0a55fab706e272801e7de9f929d13` 恢复断点后，确认工作树只有 3 个未跟踪 diagnostics 文件：`src/modeling/diagnostics.py`、`src/modeling/diagnose_checkpoint.py`、`tests/test_checkpoint_diagnostics.py`，未发现其它待保留修改，也未执行 reset/restore。

本轮对 diagnostics 实现逐文件复核并重新实跑验证。该工具严格限定 validation split，不暴露 test split 参数，不执行 `optimizer.step`；支持 binary logits/probability 分布、GT foreground/background 上的 P(fg)、probability histogram、prediction/target foreground fraction、Region Dice+CE 分项及 foreground/background CE contribution、最终 segmentation head 参数范数/偏置/gradient、全部 BatchNorm3d running statistics，以及 `--bn-mode running|batch` 对照。固定 foreground-centered patch gradient diagnostic 只 backward 一次并在结束后清空 gradient；full-volume predictor 与 training 继续复用同一 `resize_logits_to_target(..., trilinear, align_corners=False)` helper。

重新读取真实文件 `experiments/diagnostics_20260827_v6_e2_liver7_bn_batch_v2/diagnostics.json` 后确认：v6 epoch2 / `liver_7` 在临时 batch-stat inference 下 prediction foreground fraction=`0.2798134`，GT foreground fraction=`0.00699608`，CE=`1.2322532`，其中 foreground/background weighted CE contribution≈`0.02247/1.20978`。标准 running-stat inference 同 checkpoint 的 prediction foreground≈`42.26%`；batch-stat 可明显缓解到≈`27.98%`，但仍远高于 GT≈`0.70%`。结合 epoch1→epoch2 final head weight norm/bias 几乎不变，而 9 个 BatchNorm3d 的 running statistics 明显漂移，当前结论限定为：**BatchNorm running-statistics / train-eval normalization mismatch 是 foreground explosion 的重要机制之一，但不是唯一机制；不能写成唯一根因已被完全证明。**

工程验证重新实跑：

```text
.venv/Scripts/python.exe -m pytest tests/test_checkpoint_diagnostics.py -q
→ 8 passed

.venv/Scripts/python.exe -m pytest tests -q
→ 116 passed

.venv/Scripts/python.exe -m ruff check src web tests
→ All checks passed!

git diff --check
→ 通过
```

下一步锁定 v8：严格基于 `configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml`，唯一实验变量是训练阶段冻结 `BatchNorm3d` running statistics；优先采用每次进入训练态后将 BN module 设为 eval、但保留 affine weight/bias `requires_grad=True` 的最小实现。必须增加回归测试证明 running_mean/running_var/num_batches_tracked 不更新、BN affine 仍有 gradient、其它模块仍为 training、默认旧配置行为不变、validation/inference 不受意外影响。完成代码/config/测试/readiness 并 GitHub 闭环后直接跑 v8 epoch1；若无明显灾难继续 epoch2，稳定则 epoch3。独立 test `ctspine1k-msd-t10-liver_169` 在 validation 参数与 checkpoint 选择规则完全锁定前继续禁止访问。


### 2026-08-27｜阶段 AN：完成 v8 BN-running-stat 单变量工程与 readiness

v8 严格基于 `configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml`，新增 `configs/orthopedic_ct_cpu_binary_bn_frozen_v8.yaml`。实际 config diff 只有两处：`experiment_name` 改为 `cpu_binary_bn_frozen_v8_roi64`，以及 `training.freeze_batchnorm_running_stats: true`；CT-only、split/seed、ROI64、patches_per_case=4、foreground_probability=0.25、Bernoulli sampling、Region Dice+CE=1:1、lr=5e-5、warmup=1、full-volume validation ROI128 / overlap=0.25 等均保持不变。

`src/modeling/train.py` 新增 `configure_batchnorm_training_mode()`：每个 epoch 仍先执行完整 `model.train()`，仅当新配置启用时把所有 `BatchNorm3d` 子模块切到 eval，使 forward 使用固定 running_mean/running_var 且不再增加 `num_batches_tracked`；函数不修改任何参数 `requires_grad`，因此 BN affine weight/bias 仍可参与反向传播。默认未配置时直接 no-op，旧配置训练行为不变；validation/inference 仍由原 `model.eval()` 路径控制。

新增 `tests/test_batchnorm_freeze_training.py`，回归覆盖：running_mean/running_var 不更新、num_batches_tracked 不增加、BN affine weight/bias 均获得非零 gradient、其它模块保持 training、默认关闭时 BN 继续正常更新 running stats、eval/inference 状态不被 helper 意外改变，以及 v8/v6 配置除实验名和 BN 新选项外完全相同。

真实工程验证：

```text
pytest tests -q
→ 120 passed

ruff check src web tests
→ All checks passed!

v8 YAML parse
→ OK

formal_readiness --task-spec configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json \
  --config configs/orthopedic_ct_cpu_binary_bn_frozen_v8.yaml --allow-cpu
→ ready=true
→ blocker_count=0
→ preflight checked_case_count=10
→ split train/validation/test=7/2/1
→ 0 error / 0 warning
```

readiness 中 GPU 子检查仍如预期报告本机 PyTorch 为 CPU build、无 CUDA；由于本轮明确使用 `--allow-cpu`，这不构成 blocker。独立 test `ctspine1k-msd-t10-liver_169` 未被访问，继续保持锁定。

下一步在本阶段 commit/push 并确认 `HEAD == origin/main` 后，直接启动 v8 epoch1。epoch1 完成后必须先核对 run artifacts 与 full-volume validation，再对 `liver_7/liver_8` 做 detailed evaluation 和 checkpoint diagnostics；若没有明显 foreground/background 灾难，则续训 epoch2，epoch2 是主要判定点。


### 2026-08-27｜阶段 AO：v8 epoch1 发生严重背景塌缩，按规则停止

v8 run：`experiments/20260827_125357_cpu_binary_bn_frozen_v8_roi64`。本轮只使用 train 7 例与 validation 2 例，未对独立 test `ctspine1k-msd-t10-liver_169` 做 tuning/evaluation。

真实训练 / full-volume validation：

```text
epoch 1: train_loss=6.0181837635
         mean val Dice=0.0001247640
         val Dice std=0.0001247640
         validation inference total≈138.29 s
         lr=5e-5
```

训练 sampling 与 v3/v6 epoch1 完全一致：28 patches，foreground fraction mean≈`7.9073%`、median=`0`、pure-background=`18/28`、foreground-positive=`10/28`。因此本轮灾难不能归因于 sampling 发生了新的变化。

对 `best.pt`（即 epoch1）做两例 validation detailed full-volume evaluation：

- `liver_7`：Dice=`0.00024953`，Precision=`0.00163274`，Recall=`0.00013509`，prediction foreground=`0.05788%`，GT foreground=`0.69961%`，prediction/GT ratio≈`0.08274`，component error=`230`；
- `liver_8`：Dice=`0`，Precision=`0`，Recall=`0`，prediction foreground=`0.07086%`，GT foreground=`0.56596%`，prediction/GT ratio≈`0.12520`，component error=`218`。

这与 v6 epoch2 的 foreground explosion 相反，属于严重 **background collapse / foreground under-prediction**。因此 v8 明显未达到“epoch1 无灾难”的继续条件，不执行 epoch2。

checkpoint diagnostics 进一步确认 BN freeze 按设计真实生效：v8 epoch1 两例都显示 9 个 `BatchNorm3d`，第一层 `num_batches_tracked=0`、running mean channel std=`0`、running var mean=`1`；即 running stats 从初始化开始完全未更新。`liver_7` 的 prediction foreground fraction≈`0.0005788`，GT≈`0.0069961`，GT foreground mean P(fg)≈`0.0001326`，GT background mean P(fg)≈`0.0006001`；`liver_8` 的 GT foreground mean P(fg) 甚至约 `4.1e-12`。这说明“从随机初始化开始把 BN running stats 永久固定为 0/1”会造成严重 train/inference representation mismatch，不能作为稳定 baseline 方案。

当前更精确的机制判断是：**BN running-statistics 确实参与了 v6 epoch1→epoch2 的不稳定，但其处理方式不能简单改为从初始化起完全冻结。** v6 epoch1 的 BN stats（例如首层 running mean std≈`0.01449`、running var mean≈`0.06225`、num_batches=28）反而与当时较好的 validation 表现相伴；v6 epoch2 漂移到 running mean std≈`0.06107`、running var mean≈`0.01357`、num_batches=56 后出现 foreground explosion。下一步必须先量化 encoder/decoder/head-input activation 漂移与 checkpoint 参数 delta，再选择一个可解释的 v9 单变量，而不是直接同时改 normalization/lr/loss/sampling。


### 2026-08-27｜阶段 AP：增加多 checkpoint activation / parameter-delta 诊断工具

为避免在 v8 失败后直接猜测 v9，本轮新增 `src/modeling/compare_checkpoint_dynamics.py`，专门对 validation 病例的固定 foreground-centered 64³ patch 比较多个 checkpoint。该入口不提供 test split，不执行 optimizer.step。

默认记录以下关键激活：encoder 四级 patch embedding、每级最后一个 transformer block、decoder `linear_fuse` Conv/BatchNorm、`linear_pred` head input 与最终 logits。每个激活记录 shape、L2 norm、mean/std/min/max 和 q01/q05/q10/q25/q50/q75/q90/q95/q99。另对 checkpoint state_dict 计算按 encoder/decoder 子模块聚合的 parameter relative delta、变化最大的单参数，以及 BatchNorm running_mean/running_var buffer delta。

新增 `tests/test_compare_checkpoint_dynamics.py`，验证 head input/output hook 能正确抓取，以及参数组变化与 BN running buffer 变化能被识别。focused tests=`2 passed`，Ruff clean。

下一步在完成本轮工程 commit/push 后，直接用同一 `liver_7` validation foreground-centered patch 比较 v6 epoch1、v6 epoch2、v8 epoch1，依据真实 activation / parameter delta 选择 v9 的唯一变量。独立 test `ctspine1k-msd-t10-liver_169` 继续禁止访问。


### 2026-08-27｜阶段 AQ：核验 checkpoint dynamics 并完成 v9 epoch1→epoch2 BN 锚定工程/readiness

从 GitHub 闭环点 `2f6389db57c0e9531227c2f61503cf9750ac3171` 恢复真实断点，工作树初始干净，`HEAD == origin/main`。重新读取 `experiments/checkpoint_dynamics_20260827_v6e1_v6e2_v8e1_liver7/checkpoint_dynamics.json`，确认该结果已真实比较 v6 epoch1、v6 epoch2、v8 epoch1，固定病例为 validation `liver_7` 的 foreground-centered 64³ patch，未访问 test。

v6 epoch1→epoch2 的 state delta 显示：普通可训练参数按组聚合的最大相对变化约为 encoder `embed_4=0.6635%`、`embed_3=0.4723%`、`embed_2=0.3114%`、`embed_1=0.2362%`，decoder `linear_fuse≈0.1908%`；相比之下 BN running buffer 明显更剧烈，多处 running_mean relative delta≈`98.9%–331.2%`，首层 running_var≈`76.6%–78.6%`，第二层约 `62.4%–66.3%`。固定 patch 上 decoder `linear_fuse` BN output std 从 v6 epoch1≈`1.4235` 降到 epoch2≈`1.1247`，head input mean/std 从≈`0.5549/0.8030` 降到≈`0.4399/0.6440`，final logits L2 norm 从≈`1480.8` 降到≈`1204.3`。这些证据进一步支持“epoch1→epoch2 BN running-stat drift / normalization mismatch 是重要机制之一”，但仍不能声称是唯一根因。

据此锁定 v9 单变量：严格基于 `configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml`，新增 `configs/orthopedic_ct_cpu_binary_bn_freeze_after_e1_v9.yaml`；唯一实验性变化是 `training.freeze_batchnorm_running_stats_from_epoch: 2`，即 epoch1 完全保持 v6 原始训练/BN 更新，epoch2 起把所有 `BatchNorm3d` 切到 eval 使用 epoch1 已建立的 running_mean/running_var，并停止增加 num_batches_tracked；BN affine weight/bias 和其它模型参数继续训练。旧 `freeze_batchnorm_running_stats: true` 继续兼容 v8，默认旧配置不冻结。

`src/modeling/train.py` 新增 epoch-aware `should_freeze_batchnorm_running_stats()`，训练循环每个 epoch 在 `model.train()` 后按当前 epoch 决定是否冻结 BN；因此从 `last.pt` resume 到 epoch2 时会直接进入冻结状态，不依赖前一进程内存状态。run metadata 同步记录 freeze-from-epoch 配置。

`tests/test_batchnorm_freeze_training.py` 新增回归覆盖：默认配置不冻结；v8 legacy always-freeze 仍生效；v9 epoch1 running_mean/running_var 会真实变化且 num_batches_tracked 增加；模拟 epoch1 validation 后 epoch2 再次 `model.train()` 时 BN 正确冻结；epoch2 running_mean/running_var/num_batches_tracked 保持 epoch1 锚点；BN affine weight/bias 仍获得非零 gradient；其它模块保持 training；resume/start_epoch=2 判定会冻结；v9/v6 配置归一化比较除 experiment_name 与 freeze-from-epoch 外完全一致。

真实工程验证：

```text
focused pytest (BN + scheduler)
→ 9 passed

pytest tests -q
→ 125 passed

ruff check src web tests
→ All checks passed!

git diff --check
→ 通过

v9 vs v6 normalized config comparison
→ normalized_equal=true
→ v9 extra training key only: freeze_batchnorm_running_stats_from_epoch

formal_readiness --allow-cpu
→ ready=true
→ blocker_count=0
→ checked_case_count=10
→ split train/validation/test=7/2/1
→ task/preflight 0 error / 0 warning
```

GPU 子检查仍如预期报告本机为 CPU build/无 CUDA；本轮显式 `--allow-cpu`，不构成 blocker。独立 test `ctspine1k-msd-t10-liver_169` 未访问。

下一步完成本阶段 commit/push 并再次确认 `HEAD == origin/main` 后，立即运行 v9 epoch1。epoch1 理论行为应与 v6 epoch1 接近；若 mean Dice 与 v6 epoch1≈`0.05407` 差异显著，则先停止检查实现，不进入 epoch2。若复现正常，则直接续训 epoch2，并验证所有 BN running_mean/running_var/num_batches_tracked 与 epoch1 checkpoint 完全保持锚定，再做两例 detailed validation + checkpoint diagnostics。


### 2026-08-27｜阶段 AR：v9 epoch1 精确复现 v6 epoch1，进入关键 epoch2

v9 工程提交 `daafe2c1d9debc96831dc141fd5f305d01932900` 已推送并确认 `HEAD == origin/main` 后，使用 `configs/orthopedic_ct_cpu_binary_bn_freeze_after_e1_v9.yaml` 启动 CPU formal validation experiment，只训练到 epoch1。preflight 再次 `ready=true`，split=7/2/1，未访问独立 test。

真实 run：`experiments/20260827_132502_cpu_binary_bn_freeze_after_e1_v9_roi64`。

```text
epoch 1
train_loss=2.5537127596991405
mean validation Dice=0.05407000716611769
validation Dice std=0.010840379918928316
validation inference total≈137.32 s
lr=5e-5
```

该结果与 v6 epoch1 数值精确一致，证明新增 `freeze_batchnorm_running_stats_from_epoch=2` 没有影响 epoch1 数据流、scheduler、optimizer、normalization 或 full-volume validation 行为，因此满足进入 epoch2 的预设条件。

对 epoch1 `best.pt` 分别做 validation-only detailed full-volume evaluation：

- `liver_7`：Dice≈`0.04322963`，IoU≈`0.02209234`，Precision≈`0.02753337`，Recall≈`0.10055295`，HD95≈`199.91 mm`，ASSD≈`56.25 mm`；prediction foreground≈`2.5550%`，GT≈`0.6996%`，ratio≈`3.6520`，component error=`1578`。
- `liver_8`：Dice≈`0.06491039`，IoU≈`0.03354387`，Precision≈`0.04266633`，Recall≈`0.13561118`，HD95≈`175.46 mm`，ASSD≈`48.26 mm`；prediction foreground≈`1.7988%`，GT≈`0.5660%`，ratio≈`3.1784`，component error=`1597`。

checkpoint diagnostics 对两例都确认同一 checkpoint 的 9 个 BatchNorm3d 状态一致：`num_batches_tracked=28`；首层 BN running mean std≈`0.0144922826`，running var mean≈`0.0622548461`，即与此前 v6 epoch1 记录的锚点一致。`liver_7` 的 GT foreground/background mean P(fg)≈`0.12394/0.03265`，foreground/background weighted CE contribution≈`0.04322/0.06985`；`liver_8` 对应≈`0.15221/0.02478` 与≈`0.03320/0.04651`。这些只作为 validation mechanism diagnostics，不是最终 test 结果。

结论：**v9 epoch1 已通过“必须接近 v6 epoch1”的门槛，而且是精确复现。** 下一步直接从本 run 的 `checkpoint/last.pt` resume 到总 epoch2；epoch2 训练后首先比较 checkpoint state_dict，要求所有 BN running_mean/running_var/num_batches_tracked 与 epoch1 checkpoint 完全一致，再运行两例 detailed validation 和 diagnostics。若 foreground explosion 显著缓解且 validation 不灾难性下降，继续 epoch3；若仍出现几十倍前景泛滥，则停止并进入 v10 机制诊断。


### 2026-08-27｜阶段 AS：完成 v9 epoch2 detailed validation / diagnostics / dynamics，停止 v9

从 Git 闭环点 `981714b17713a9b762cbbacb76817318378aca8c` 恢复真实断点后，重新确认工作树初始干净且 `HEAD == origin/main`。真实 v9 run=`experiments/20260827_132502_cpu_binary_bn_freeze_after_e1_v9_roi64` 已存在 epoch2 训练产物：`history.csv` 显示 epoch2 train loss=`2.6975343355110715`、mean validation Dice=`0.026778433853422875`、std=`0.002211471671448084`、validation inference total≈`131.09 s`、lr=`4.892324335849338e-05`；`summary.json` 明确 `last_epoch=2`、`best_val_dice=0.05407000716611769`，因此 `best.pt` 仍为 epoch1，`last.pt` 为 epoch2。`sampling_stats.csv` 显示 epoch2 仍为 28 patches，mean foreground fraction≈`8.8408%`、median=`0`、foreground-positive/background patches=`10/18`，与历史复现实采样统计一致。

对 v9 epoch2 `last.pt` 仅在 validation split 分病例运行 full-volume detailed evaluation，独立 test `liver_169` 未访问。结果目录：

- `experiments/evaluation_20260827_v9e2_liver7`
- `experiments/evaluation_20260827_v9e2_liver8`

真实逐例结果：

- `liver_7`：Dice=`0.0289899055`，IoU=`0.0147081467`，Precision=`0.0163801553`，Recall=`0.1259438040`，HD95=`187.3259 mm`，ASSD=`64.2019 mm`；prediction foreground=`5.37915%`，GT=`0.69961%`，ratio=`7.6888`；pred/GT components=`517/3`，component error=`514`，false merge=`1`，false break=`32`；inference≈`44.75 s`；uncertainty AUROC/AUPRC≈`0.86919/0.30592`；ECE/MCE/Brier/NLL≈`0.04598/0.21958/0.10427/0.35269`。
- `liver_8`：Dice=`0.0245669622`，IoU=`0.0124362414`，Precision=`0.0136715130`，Recall=`0.1209869693`，HD95=`187.8004 mm`，ASSD=`64.0517 mm`；prediction foreground=`5.00849%`，GT=`0.56596%`，ratio=`8.8496`；pred/GT components=`517/2`，component error=`515`，false merge=`0`，false break=`31`；inference≈`73.10 s`；uncertainty AUROC/AUPRC≈`0.85851/0.26238`；ECE/MCE/Brier/NLL≈`0.04316/0.17657/0.09777/0.40703`。

两例平均约：Dice=`0.02677843`、IoU=`0.01357219`、Precision=`0.01502583`、Recall=`0.12346539`、HD95=`187.56 mm`、ASSD=`64.13 mm`、prediction foreground≈`5.19%`、prediction/GT ratio≈`8.27`、component error≈`514.5`。对比 v6 epoch2：`liver_7/liver_8` prediction foreground≈`42.26%/34.04%`、ratio≈`60.40/60.14`、Recall≈`0.9856/0.9992`。因此 v9 明显压低了 v6 的 foreground explosion，但并未恢复 segmentation quality：Dice 继续下降、Precision 极低且预测碎片仍非常多。

随后对同一 v9 epoch2 `last.pt` 运行 validation-only checkpoint diagnostics：

- `experiments/diagnostics_20260827_v9e2_liver7`
- `experiments/diagnostics_20260827_v9e2_liver8`

两例 `head_parameters` 相同：segmentation head weight norm≈`22.33960`、bias norm≈`0.0008889`。`liver_7` 的 GT foreground/background mean P(fg)≈`0.13278/0.05583`，Dice loss≈`0.97081`、CE≈`0.35051`，foreground/background weighted CE contribution≈`0.04732/0.30318`；`liver_8` 对应 mean P(fg)≈`0.12832/0.05242`、Dice loss≈`0.97521`、CE≈`0.40268`、weighted CE contribution≈`0.03511/0.36757`。这说明在全卷上仍存在明显背景假阳性，同时绝大多数 GT foreground voxel 的 P(fg) 很低；不是简单的“全局前景阈值偏高”单一问题。

BN 锚定再次用真实 checkpoint state_dict 逐项核对：比较 v9 epoch1 `best.pt` 与 epoch2 `last.pt`，所有 key 名包含 `running_mean`、`running_var`、`num_batches_tracked` 的 buffer 共 `27` 个，`changed=0`。diagnostics 同时显示 9 个 BatchNorm3d 的第一层仍为 `num_batches_tracked=28`、running mean std≈`0.0144922826`、running var mean≈`0.0622548461`，即 epoch2 确实继续使用 epoch1 锚点。

由于 v9 满足“foreground explosion 明显缓解但 Dice 仍显著退化”的预设情况 A，本轮没有运行 epoch3，而是立即执行四 checkpoint dynamics：`experiments/checkpoint_dynamics_20260827_v6e1_v6e2_v9e1_v9e2_liver7`，固定 `liver_7` foreground-centered 64³ patch，未访问 test。

关键 dynamics 证据：

1. `v6e1 → v9e1`：所有普通参数组 delta=`0`，所有 BN running buffer delta=`0`，证明 v9 epoch1 checkpoint 与 v6 epoch1 不只是指标相同，而是 checkpoint state 在比较范围内精确一致。
2. `v6e1 → v6e2`：普通参数聚合最大 relative delta 仍只有 encoder embed4≈`0.663%`、embed3≈`0.472%`、embed2≈`0.311%` 等；同时 BN running_mean relative delta 最高≈`3.31×`，running_var 多处≈`66%–79%`，与此前结论一致。
3. `v6e1 → v9e2`：所有 BN running buffers relative delta=`0`；但普通 trainable 参数仍发生小幅更新，encoder embed4≈`0.658%`、embed3≈`0.436%`、embed2≈`0.275%`、decoder linear_fuse≈`0.148%`、final head≈`0.0179%`。
4. 固定 patch 上，epoch1 decoder `linear_fuse` BN output mean/std≈`-0.0715/1.4235`、head input≈`0.5549/0.8030`、final logits mean/std≈`-11.3113/11.8207`；v6e2 变为≈`-0.0500/1.1247`、`0.4399/0.6440`、`-8.7949/9.9843`；v9e2 在 BN running stats 不变时反而为≈`-0.1501/1.4962`、`0.5474/0.8370`、`-4.5614/10.5338`。也就是说 v9 抑制了 running-stat drift 后，head-input 边缘统计接近 epoch1，但 final logits 仍发生大幅位置漂移，且 final head 本身参数聚合变化极小。

当前科学判断必须严格限定为：**BN running-statistics drift / train-eval normalization mismatch 已被证明是 v6 epoch2 foreground explosion 的重要放大机制；它不是必要且充分的唯一根因。** v9 证明把 BN stats 固定在 epoch1 可以把约 `60×` 的前景爆炸显著压低到约 `8×`，但仍不能阻止 epoch2 Dice、Precision 和结构质量退化。剩余证据更指向“BN running stats 之外的 trainable normalization / upstream feature parameter update 造成的 logit dynamics”，而不是继续把问题归因于 BN buffer。

因此本阶段决策：**停止 v9，不跑 epoch3；stable baseline=NO；lock parameters=NO；formal locked test ready=NO。** 当前最佳 checkpoint 仍为 v9/v6 epoch1 的 `experiments/20260827_132502_cpu_binary_bn_freeze_after_e1_v9_roi64/checkpoint/best.pt`，mean validation Dice=`0.05407000716611769`。下一步必须只选一个 v10 主要变量，优先隔离 BN running stats 已固定后仍可训练的 normalization / feature parameters；不得同时改变 lr、loss、sampling、ROI、augmentation，也继续禁止访问独立 test。


### 2026-08-27｜阶段 AT：v10 单变量决策与 encoder-freeze 工程闭环准备

在 v9 已停止且 BN running buffers 已被严格锚定的前提下，进一步直接比较 v6e1→v6e2 与 v9e1→v9e2 checkpoint state。按参数类型聚合 L2 relative delta 后，v9e1→v9e2：patch embeddings≈`0.7842%`、encoder attention≈`0.7075%`、encoder MLP≈`0.1906%`、decoder projections≈`0.1614%`、decoder linear_fuse≈`0.1568%`；相比之下 LayerNorm affine≈`0.0208%`、BN affine≈`0.0194%`、segmentation head≈`0.0179%`。虽然个别 normalization bias 因初始绝对值接近 0 而呈现较大的逐参数相对百分比，但其聚合绝对变化和整体归一化变化均远小于 encoder patch/attention 权重组。结合 fixed foreground patch 上 final head 自身变化极小、final logits mean 却从 epoch1≈`-11.31` 漂移到 v9e2≈`-4.56`，当前最强证据更指向上游 encoder representation update，而不是优先把剩余问题归结为 BN affine。

因此 v10 只新增一个主要实验变量：`training.freeze_encoder_parameters_from_epoch: 2`。配置为 `configs/orthopedic_ct_cpu_binary_encoder_freeze_after_e1_v10.yaml`。它完整继承 v9：epoch1 正常训练；epoch2 起继续保持 `freeze_batchnorm_running_stats_from_epoch=2`，并额外冻结 `segformer_encoder` 的 trainable parameters；decoder / segmentation head 继续训练。lr、loss、sampling、ROI、augmentation、input channels、scheduler、full-volume validation 均不改变。相对 v9 的归一化 config diff 只有 experiment name 和这一条 encoder-freeze 配置。

工程实现新增 `should_freeze_encoder_parameters()` 与 `configure_encoder_parameter_training()`；resume 到 epoch2 时仍按当前 epoch 自动触发冻结。回归测试覆盖：epoch1/epoch2 policy、resume 判定、只关闭 encoder gradient 而不关闭 decoder gradient、恢复 trainability、v10/v9 单变量 config diff。focused tests=`11 passed`；全量 `pytest tests -q`=`129 passed`；Ruff=`All checks passed`；`git diff --check` 通过。`formal_readiness --allow-cpu` 对 v10 实测 `ready=true`、`blocker_count=0`，task spec、7/2/1 split、10 例 preprocessing/QC 均通过；CPU-only 状态仍如实记录但因显式 `--allow-cpu` 不构成 blocker。

当前决策：v10 工程条件已满足，下一步立即只跑 epoch1。epoch1 必须复现或接近 v9/v6 epoch1；通过后再从同一 run resume 到 epoch2，以直接检验“encoder 参数更新是否是 BN running stats 之外造成 logit/segmentation degradation 的主要剩余机制”。独立 test `liver_169` 继续禁止访问。


### 2026-08-27｜阶段 AU：v10 epoch1 精确复现、双病例诊断与进入 epoch2 门槛确认

v10 epoch1 已在既有 run `experiments/20260827_170359_cpu_binary_encoder_freeze_after_e1_v10_roi64` 真实完成，本阶段未重跑 epoch1。`history.csv` 记录：train loss=`2.5537127596991405`、mean full-volume validation Dice=`0.05407000716611769`、val Dice std=`0.010840379918928316`、validation inference total≈`143.4137 s`、lr=`5e-5`。`sampling_stats.csv` 记录 28 个训练 patch，foreground fraction mean≈`0.07907336`，foreground/background patch=`10/18`。这些数值与 v6/v9 epoch1 精确一致。

两例 detailed validation 均使用 v10 epoch1 `best.pt` 且只访问 validation split：`liver_7` Dice=`0.0432296272`、IoU=`0.0220923353`、Precision=`0.0275333742`、Recall=`0.1005529539`、HD95=`199.91498 mm`、ASSD=`56.24754 mm`、prediction/GT foreground ratio=`3.6520×`、pred components=`1581`、component error=`1578`、false merge=`1`、false break=`63`；`liver_8` Dice=`0.0649103871`、IoU=`0.0335438662`、Precision=`0.0426663277`、Recall=`0.1356111771`、HD95=`175.45655 mm`、ASSD=`48.25676 mm`、prediction/GT foreground ratio=`3.1784×`、pred components=`1599`、component error=`1597`、false merge=`0`、false break=`68`。两例结果与 v6/v9 epoch1 锚点一致。

两例 validation-only diagnostics 均已完成。`liver_7` 的 GT foreground/background mean P(fg)≈`0.12394/0.03265`，Dice loss≈`0.95683`、CE loss≈`0.11307`，foreground/background weighted CE contribution≈`0.04322/0.06985`，fixed foreground patch head weight gradient norm≈`6.75449`；`liver_8` 的对应数值≈`0.15221/0.02478`、Dice loss≈`0.94430`、CE loss≈`0.07972`、CE contribution≈`0.03320/0.04651`、head weight gradient norm≈`6.52452`。两例均显示 9 个 BatchNorm3d 已按 epoch1 正常建立 running statistics，首层 `num_batches_tracked=28`、running mean std≈`0.0144923`、running var mean≈`0.0622548`，与 v6/v9 epoch1 锚点一致。diagnostics 全程未执行 optimizer.step。

随后对 v6 epoch1、v9 epoch1、v10 epoch1 三个 `best.pt` 做逐 tensor 精确比较：三者 checkpoint epoch 均为 1、val Dice 均为 `0.05407000716611769`；`model_state_dict` 共 232 个 tensor，key 完全一致。`v6↔v9`、`v6↔v10`、`v9↔v10` 三组逐 tensor `torch.equal` 均全部成立，diff tensor=`0`；所有 `running_mean/running_var/num_batches_tracked` 的 BN buffer diff=`0`。因此可以严格写成：**v6e1 == v9e1 == v10e1（model state exact equal）**，v10 的延迟 encoder-freeze 工程没有污染 epoch1。

本阶段判断：v10 epoch1 已通过进入 epoch2 的全部门槛。下一步从同一 run 的 `checkpoint/last.pt` resume 到总 epoch2；epoch2 起必须验证 encoder trainable parameters 冻结、BN running statistics 冻结、decoder/head 继续训练，并在训练后检查 `v10e1→v10e2`：encoder parameter delta=`0`、BN running-buffer delta=`0`、decoder/head delta 以及 fixed-patch activation/logit drift。随后只对 `liver_7/liver_8` 做 detailed validation 与 diagnostics。独立 test `liver_169` 继续严格禁止访问；stable baseline 仍为 NO、lock parameters=NO、formal test ready=NO，直到至少 epoch1/2/3 连续稳定。


### 2026-08-27｜阶段 AV：v10 epoch2 灾难性背景塌缩，encoder/BN 已排除为必要条件

v10 已从同一 run `experiments/20260827_170359_cpu_binary_encoder_freeze_after_e1_v10_roi64/checkpoint/last.pt` resume 到总 epoch2。formal preflight 再次 `ready=true / 0 blocker`。epoch2 真实训练结果：train loss=`2.405222313744681`、mean full-volume validation Dice=`1.6538643217978726e-11`、std=`2.4799193429642677e-13`、validation inference total≈`128.1903 s`、lr≈`4.892324335849338e-5`。sampling_stats 仍为 28 patch、foreground/background patch=`10/18`，foreground fraction mean≈`0.08840765`，与 epoch1 的≈`0.07907336` 同量级，不能解释接近 0 的 Dice。

冻结机制经 checkpoint 逐项核对已严格生效：v10e1→v10e2 的 `segformer_encoder.*` 共 208 个 state tensor，changed=`0`、aggregate relative L2 delta=`0`；27 个 BN `running_mean/running_var/num_batches_tracked` changed=`0`。与此同时 decoder 24 个 state tensor 中 21 个发生变化，aggregate relative L2≈`0.1167%`；其中 `linear_c4≈0.2529%`、`linear_c3≈0.1811%`、`linear_fuse≈0.1681%`、`linear_c2≈0.1206%`、`linear_c1≈0.0893%`，最终 `linear_pred≈0.02255%`。final head weight/bias 均确实更新，因此不是“整个模型没有训练”。

两例 detailed validation 使用 epoch2 `last.pt` 且只访问 validation split。`liver_7`：Dice/IoU/Precision/Recall=`0/0/0/0`，HD95≈`169.059 mm`、ASSD≈`92.743 mm`，prediction foreground fraction≈`0.00051049`、GT≈`0.00699608`、prediction/GT≈`0.07297×`，pred components=`354`、component error=`351`、false merge/break=`0/0`。`liver_8`：Dice/IoU/Precision/Recall=`0/0/0/0`，HD95≈`174.311 mm`、ASSD≈`98.331 mm`，prediction foreground fraction≈`0.00036441`、GT≈`0.00565958`、prediction/GT≈`0.06439×`，pred components=`322`、component error=`320`、false merge/break=`0/0`。这不是正常改善，而是明确 background collapse。

validation-only diagnostics 与 detailed validation 一致。`liver_7` full-volume foreground/background logit mean≈`-15.0796/7.94395`，GT foreground/background mean P(fg)≈`1.34e-5/7.43e-4`，Dice loss≈`0.999973`、CE loss≈`0.111025`，foreground/background weighted CE contribution≈`0.10928/0.00175`；`liver_8` foreground/background logit mean≈`-15.0863/9.20556`，GT foreground/background mean P(fg)≈`1.82e-5/4.95e-4`，Dice loss≈`0.999964`、CE loss≈`0.087960`，CE contribution≈`0.08691/0.00105`。两例 head gradient norm 仍≈`6.94/6.91`，说明 final head 仍收到显著梯度；首层 BN 仍严格保持 epoch1 锚点 `num_batches_tracked=28`、running mean std≈`0.0144923`、running var mean≈`0.0622548`。

固定 `liver_7` foreground-centered 64³ patch 的 checkpoint dynamics 进一步隔离了机制：v10e1 与 v10e2 的所有 encoder hook activation mean/std 完全逐值一致；decoder `linear_fuse.0` mean 从≈`-0.0740` 漂到≈`-0.1246`，`linear_fuse.1` mean 从≈`-0.07155` 漂到≈`-0.10212`，final head input mean/std≈`0.55495/0.80298 → 0.54336/0.80388`，final logits mean/std≈`-11.3113/11.8207 → -9.81895/14.5471`。因此 encoder representation 本身在固定输入上完全不变，而 decoder feature transformation 与最终 logits 仍发生明显 dynamics。

科学判断必须更新为：**v10 否定了“encoder parameter update 是 epoch2 degradation 的必要条件”这一更强假设。** encoder 与 BN running stats 都被冻结后仍可出现更严重的 background collapse，说明剩余不稳定性至少可以由 decoder/head training dynamics 产生。由于 decoder feature groups 的聚合变化显著大于 final `linear_pred`，且 encoder activation 完全不变而 decoder fuse/final logits 漂移，下一单变量优先隔离 decoder representation update，而不是再改 lr/loss/sampling/ROI/augmentation。

因此 v10 立即 STOP，不跑 epoch3。stable baseline=NO；lock parameters=NO；formal test ready=NO；独立 test `liver_169` 继续禁止访问。v11 计划严格单变量：继承 v10 的 epoch2 encoder freeze + BN-running-stat freeze，并从 epoch2 起额外冻结 decoder feature parameters（`linear_c1..c4` 与 `linear_fuse`），仅保留最终 `linear_pred` segmentation head 可训练；其它 lr/loss/sampling/ROI/augmentation/input/scheduler/full-volume validation 全部不变。该实验只用于判断 decoder representation update 是否是 v10 剩余 collapse 的主要来源。


### 2026-08-27｜阶段 AW：v11 decoder-feature-freeze 工程完成并通过 readiness

基于 v10 的真实证据，本阶段没有重新开发已存在的 v11，而是先接管本地未提交修改并核对单变量设计。v11 配置为 `configs/orthopedic_ct_cpu_binary_decoder_feature_freeze_after_e1_v11.yaml`，完整继承 v10 的 CT-only、64³ training ROI、Region Dice+CE=1:1、foreground sampling、AdamW peak lr=`5e-5`、scheduler、full-volume validation、`freeze_batchnorm_running_stats_from_epoch=2` 与 `freeze_encoder_parameters_from_epoch=2`，唯一新增主要实验变量是 `freeze_decoder_feature_parameters_from_epoch=2`。

工程实现位于 `src/modeling/train.py`：新增 `should_freeze_decoder_feature_parameters()` 与 `configure_decoder_feature_parameter_training()`。epoch1 decoder 全部正常训练；epoch2 起 `segformer_decoder` 中除 `linear_pred.*` 外的参数全部 `requires_grad=False`，即冻结 `linear_c1/linear_c2/linear_c3/linear_c4/linear_fuse` 等 decoder feature parameters，同时始终保留最终 `linear_pred` segmentation head 可训练。缺少 `segformer_decoder` 或在启用冻结时缺少 `linear_pred` 会显式报错。新增 `tests/test_decoder_feature_freeze_training.py` 覆盖 epoch policy/resume、仅 `linear_pred` 保留 gradient、trainability 恢复以及 v11/v10 normalized config diff。

本阶段重新执行真实验证：focused freeze tests=`15 passed`；全量 `pytest tests -q`=`133 passed`；`ruff check src web tests`=`All checks passed!`。使用 locked task spec 执行 `python -m src.modeling.formal_readiness --task-spec configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json --config configs/orthopedic_ct_cpu_binary_decoder_feature_freeze_after_e1_v11.yaml --allow-cpu`，结果 `ready=true`、`blocker_count=0`，10 例 preprocessing/QC、7/2/1 split 与 task lock 均通过。CPU-only / CUDA unavailable 仍如实存在，但在显式 `--allow-cpu` 下不构成当前 engineering validation blocker。

当前科学目标保持严格单变量：验证“当 encoder parameters、BN running stats、decoder feature parameters 均从 epoch2 起固定，只允许 final segmentation head 更新时，epoch2 是否仍发生 collapse”。因此下一步必须先运行 v11 epoch1，并将 v11e1 `best.pt` 与 v10e1 `best.pt` 做逐 tensor `torch.equal`；只有 exact equal 才允许从同一 run resume 到 epoch2。独立 test `liver_169` 继续禁止访问；stable baseline=NO，lock parameters=NO，formal locked test ready=NO。


### 2026-08-27｜阶段 AX：v11 epoch1 精确复现并通过进入 epoch2 门槛

v11 epoch1 已在新 run `experiments/20260827_180730_cpu_binary_decoder_feature_freeze_after_e1_v11_roi64` 真实完成。formal preflight=`ready=true`。`history.csv`：train loss=`2.5537127596991405`、mean full-volume validation Dice=`0.05407000716611769`、std=`0.010840379918928316`、validation inference total≈`145.990662 s`、lr=`5e-5`。`sampling_stats.csv`：28 个 training patch，foreground fraction mean=`0.07907336098807198`、foreground/background patch=`10/18`；与 v10/v9/v6 epoch1 锚点一致。

随后直接比较 v10 epoch1 `experiments/20260827_170359_cpu_binary_encoder_freeze_after_e1_v10_roi64/checkpoint/best.pt` 与 v11 epoch1 `checkpoint/best.pt`。两者 checkpoint epoch 均为 1、val Dice 均为 `0.05407000716611769`；`model_state_dict` key 完全一致，共 232 个 tensor，逐 tensor `torch.equal` 全部成立，`diff_tensor_count=0`。因此可以严格确认：**v11e1 == v10e1 == v9e1 == v6e1（model state exact equal）**，v11 新增的延迟 decoder-feature freeze 没有污染 epoch1。

由于 v11e1 与 v10e1 checkpoint 完全相同，重复运行 `liver_7/liver_8` 的 full-volume detailed evaluation 与 validation-only diagnostics 不会产生新的科学信息，只会重复消耗 CPU；因此本阶段不为形式复跑昂贵 evaluation，直接沿用 v10e1 已完成且与该 checkpoint 等价的 detailed validation/diagnostics 锚点。下一步允许从 v11 同一 run 的 `checkpoint/last.pt` resume 到总 epoch2。epoch2 后必须真实验证 encoder parameter delta=`0`、27 个 BN running buffer delta=`0`、decoder feature delta=`0`，以及 `linear_pred` final head 是否发生非零更新；随后仅对 `liver_7/liver_8` 做 detailed validation、diagnostics 与 checkpoint dynamics。独立 test `liver_169` 继续禁止访问；stable baseline 仍为 NO。


### 2026-08-27｜阶段 AY：v11 epoch2 稳定结果、diagnostics 与 checkpoint dynamics 闭环

v11 epoch2 训练产物已在接管时完整存在，因此没有重跑 training，也没有重复 `liver_7/liver_8` detailed evaluation。`history.csv` 记录 epoch2 train loss=`2.305381100092615`、mean full-volume validation Dice=`0.05437616811727176`、std=`0.010164091466349327`、lr=`4.892324335849338e-05`；相对 epoch1 mean Dice=`0.05407000716611769` 没有下降。`sampling_stats.csv` 记录 28 个 patch、foreground/background=`10/18`、foreground fraction mean=`0.08840765271868024`。

已复核现有 detailed validation：`liver_7` Dice=`0.04421207664744834`、Precision=`0.02769497548718257`、Recall=`0.10954250720461095`、prediction/GT foreground ratio=`3.9553206051873198`；`liver_8` Dice=`0.06454025958001404`、Precision=`0.04149838474144706`、Recall=`0.14511500481173542`、ratio=`3.496883209210788`。两例均没有复现 v10 的 background collapse，也没有回到 v6 的约 `60×` foreground explosion。

本阶段新运行 validation-only diagnostics：`experiments/diagnostics_20260827_v11e2_liver7` 与 `...liver8`。`liver_7` GT foreground/background mean P(fg)=`0.1326324/0.0348158`，Dice loss=`0.9561937`、CE loss=`0.1170483`，foreground/background weighted CE contribution=`0.0422318/0.0748165`，final-head weight norm=`22.33962`、gradient norm=`6.74654`；`liver_8` 对应 mean P(fg)=`0.1611401/0.0267009`，Dice loss=`0.9445096`、CE loss=`0.08285695`，CE contribution=`0.0323833/0.0504737`，head gradient norm=`6.51029`。两例 9 个 BatchNorm3d 均保持 epoch1 running-stat 锚点（首层 `num_batches_tracked=28`、running mean std≈`0.0144923`、running var mean≈`0.0622548`）。

随后运行 `experiments/checkpoint_dynamics_20260827_v11e1_v11e2_liver7`。由于 v11 epoch1 checkpoint 已被 epoch2 的 `best.pt/last.pt` 覆盖，本次使用此前逐 tensor 已证明与 v11e1 232 个 model-state tensor exact equal 的 v10e1 `best.pt` 作为 `v11e1_exact_anchor`。固定 `liver_7` foreground-centered 64³ patch 上，8 个 encoder hook、`linear_fuse.0/.1` 与 `linear_pred.input` 的完整统计字典均 exact equal；仅 `linear_pred` final logits 发生变化。state delta 进一步显示所有 encoder groups 与 decoder `linear_c1..c4/linear_fuse` delta=`0`，18 个浮点 BN running_mean/running_var buffer delta=`0`，仅 `linear_pred` weight+bias 非零（group relative delta≈`4.491e-4`）。结合此前逐 tensor freeze 检查，v11 epoch2 的隔离条件成立：encoder、BN、decoder feature 固定，仅 final head 更新。

科学判断更新为：在冻结 decoder feature 后，v10 catastrophic background collapse 消失，且固定 patch 的 decoder feature/head-input activation 完全稳定；因此当前真实证据进一步支持 **decoder feature update 是 v10 collapse 的关键机制之一**。但由于还缺 epoch3 跨 epoch 稳定性，暂不写成唯一根因，也暂不宣布 stable baseline。当前允许直接 resume v11 到总 epoch3；独立 test `liver_169` 本阶段未访问。

质量检查重新执行：`pytest tests -q`=`133 passed`，`ruff check src web tests`=`All checks passed!`，`git diff --check` 通过。下一步先完成本阶段独立 commit/push 并确认 `HEAD == origin/main`，然后直接进入 v11 epoch3。


### 2026-08-27｜阶段 AZ：v11 epoch3 最终 freeze 验证与 stable baseline 判定

接管时确认 v11 epoch3 training 与 `liver_7/liver_8` detailed validation 已真实完成，因此本阶段严格不重跑 epoch1/2/3 training，也不重复两例 epoch3 evaluation。run=`experiments/20260827_180730_cpu_binary_decoder_feature_freeze_after_e1_v11_roi64` 的 `history.csv` 最终记录：epoch1 train loss=`2.5537127596991405`、mean val Dice=`0.05407000716611769`；epoch2 train loss=`2.305381100092615`、mean val Dice=`0.05437616811727176`；epoch3 train loss=`1.8300107227904456`、mean val Dice=`0.054657574013790594`、std=`0.009516761915114875`、lr=`4.5798373876248846e-05`。三轮 full-volume validation Dice 为 `0.05407001 → 0.05437617 → 0.05465757`，没有 catastrophic drop。

epoch3 sampling 共 28 个 training patch，foreground/background=`8/20`，foreground fraction mean=`0.05680016108921596`。现有 epoch3 detailed validation：`liver_7` Dice=`0.04514081209537846`、IoU=`0.023091592670551418`、Precision=`0.027923934385888763`、Recall=`0.11772694524495678`、HD95=`197.5005063098456 mm`、ASSD=`55.38768805983404 mm`、prediction/GT foreground ratio=`4.215987031700288`；`liver_8` Dice=`0.06417433592551017`、IoU=`0.03315088601028112`、Precision=`0.04057761156434401`、Recall=`0.15335130870533972`、HD95=`174.5995418092499 mm`、ASSD=`47.656337177566336 mm`、ratio=`3.779209835013828`。相对 epoch2 两例指标没有 foreground explosion、background collapse 或 Precision/Recall 极端突变。

由于 v11 epoch2 checkpoint 已被 epoch3 的 `best.pt/last.pt` 覆盖，本阶段没有伪造或声称恢复不存在的 epoch2 checkpoint，而是按预设纪律做交叉验证。新运行 validation-only `experiments/checkpoint_dynamics_20260827_v11e1anchor_v11e3_liver7`，baseline 使用此前已逐 tensor 证明与 v11e1 232 个 model-state tensor exact equal 的 v10e1 `best.pt`。从该 exact anchor 到 v11e3：除 `segformer_decoder.linear_pred` 外所有 parameter group delta=`0`；18 个浮点 BN `running_mean/running_var` buffer delta=`0`；固定 `liver_7` foreground-centered 64³ patch 上 8 个 encoder hook、`linear_fuse.0/.1` 与 `linear_pred.input` 的完整 activation 统计字典 exact equal，仅 `linear_pred` final logits 改变。`linear_pred` group relative delta=`0.0008488773006462217`、delta norm=`0.018964542390850672`。

再与已保存的 `experiments/checkpoint_dynamics_20260827_v11e1_v11e2_liver7` 交叉验证：epoch2 对同一 exact anchor 的 `linear_pred` group delta norm=`0.010033771648873226`，而 epoch3 为 `0.018964542390850672`；若 epoch2 与 epoch3 final head 完全相同，则它们相对同一 anchor 的差分向量及其 norm 必须相同，因此该差异严格排除“epoch2→epoch3 final head 未更新”。与此同时 frozen groups 在 epoch2 与 epoch3 都对同一 anchor 保持 exact-zero delta，可据此严谨确认 encoder parameters、BN running buffers 与 decoder feature parameters 在 epoch2→epoch3 期间继续保持冻结，仅 final head 持续更新。

基于连续三轮 full-volume validation、稳定的 foreground ratio/Precision/Recall、冻结状态交叉验证、fixed-patch activation 稳定性和可解释 sampling，本阶段正式判定：**stable baseline=YES（engineering/validation）**。这里的 stable 仅表示当前 CT-only 训练机制不再发生 v6/v10 那类灾难性漂移，并不表示绝对分割性能已经达到论文目标；当前 mean validation Dice 仍仅约 `0.05466`，结构碎片化和表面距离仍很差，因此 **lock parameters=NO、formal independent test ready=NO**。独立 `ctspine1k-msd-t10-liver_169` 本阶段未访问，旧 formal-pilot test 结果仍只作为历史工程链证据。

科学结论保持谨慎：v10 中 encoder 与 BN 已冻结仍 collapse，而 v11 进一步冻结 decoder feature 后连续 epoch2/3 稳定，且 decoder feature/head-input activation 对 exact anchor 完全不漂移；因此真实证据支持 **decoder feature update 是 v10 catastrophic collapse 的关键机制之一**，但不能据此写成唯一根因。

本阶段质量检查：`pytest tests -q`=`133 passed`；`ruff check src web tests`=`All checks passed!`；`git diff --check` 通过，仅提示 Windows checkout 的 LF→CRLF 行尾转换警告，没有 whitespace error。下一步按任务纪律直接进入最小可信 baseline reproducibility，然后再做 CT-only vs CT+bone-window 输入消融；在所有 validation 决策锁定前继续禁止重新访问 `liver_169`。


### 2026-08-27｜阶段 BA：v11 stable baseline 最小可信 reproducibility

stable baseline 确认后，本阶段没有立即重复完整 3-epoch CPU 重训，因为该操作耗时较高且不会优先解决当前最关键的输入/loss 消融问题；按照既定规则，采用“最小但可信”的复现方案验证固定 checkpoint + config 的 full-volume validation 可重复性。先尝试一次性重放整个 validation split，单次工具超时且进程退出、输出目录为空；按 timeout 纪律没有直接启动第二个相同任务，而是改为只补跑缺失的两例 validation case，且全程未访问 test split。

固定 config=`configs/orthopedic_ct_cpu_binary_decoder_feature_freeze_after_e1_v11.yaml`，SHA-256=`6898924e3b1dbf9d60d501b252ebc44fe5411d5ec1f967efda06f11355548ae9`；固定 checkpoint=`experiments/20260827_180730_cpu_binary_decoder_feature_freeze_after_e1_v11_roi64/checkpoint/best.pt`，SHA-256=`9a805bc9c97b96128ba0b63d84dc30e113bade227f0a3a1cbd524231da896d67`。分别重新运行 `liver_7` 与 `liver_8` 的 formal preflight + full-volume evaluation，输出到 `experiments/repro_20260827_v11e3_liver7` 与 `...liver8`；两次 preflight 均 `ready=true`、error/warning=`0/0`。

与原 epoch3 detailed evaluation 逐项比较 summary metrics：`liver_7` Dice 仍为 `0.04514081209537846`，`liver_8` Dice 仍为 `0.06417433592551017`；除墙钟 `inference_seconds` 外，Dice、IoU、Precision、Recall、HD95、ASSD、prediction/GT foreground ratio、component error、false merge/break、uncertainty→error、ECE/MCE、Brier、NLL、confidence gap 等所有 summary metric mean 都 exact equal。仅运行时间因系统调度变化：`liver_7` 约 `59.00→55.68 s`，`liver_8` 约 `86.75→89.49 s`。

因此本阶段可以严谨写：**v11 epoch3 checkpoint 的 full-volume inference/evaluation reproducibility=PASS**。该结论不等同于“从随机初始化重新训练 3 epoch 后得到完全相同 checkpoint/轨迹”；完整 training reproducibility 仍未执行，不能夸大。独立 test `liver_169` 本阶段未访问。下一步按计划进入 CT-only vs CT+bone-window 输入消融，并尽量保持 lr、loss、sampling、ROI、augmentation、scheduler、seed 与 validation 不变。


### 2026-08-28｜阶段 BB：v12 CT+bone-window 输入消融完成，CT-only 胜出

从远程闭环点 `96719876ff4b776d4e955afa37a86e2ff131b0c2` 恢复项目后，确认 `HEAD == origin/main`，工作树仅有尚未提交的 `configs/orthopedic_ct_cpu_binary_ct_bone_window_v12.yaml`。有效 v12 run=`experiments/20260827_233142_cpu_binary_ct_bone_window_v12_roi64` 已真实完成 3 epoch，未重跑训练；另一个 `experiments/20260827_232539_cpu_binary_ct_bone_window_v12_roi64` 是工具 timeout 后留下的不完整 run（无有效 history/checkpoint），继续保留且不作为实验结果。

v12 相对 v11 的 config diff 只有实验名、输入表示与与之匹配的输入通道数/说明：`data.input_channels=[ct_normalized] → [ct_normalized,bone_window]`、bone window=`center=500,width=2000`、`model.in_channels=1→2`；loss、optimizer、lr、scheduler、sampling、ROI、augmentation、seed、freeze policy、validation 和 inference 均保持不变。formal preflight 对两次 detailed validation 均再次 `ready=true / 0 error / 0 warning`，7/2/1 split 不变，本阶段未访问独立 test `liver_169`。

v12 三轮真实训练轨迹：epoch1 train loss=`4.263589756829398`、mean val Dice=`0.027270740458087465`；epoch2 train loss=`8.248797429459435`、mean val Dice=`0.027674864114707952`；epoch3 train loss=`6.610349318810871`、mean val Dice=`0.028027748189159436`。sampling 分别为 28 patches，foreground/background=`10/18、10/18、8/20`，foreground fraction mean=`0.07907336098807198 / 0.08840765271868024 / 0.05680016108921596`，与 v11 对应 epoch 采样一致。

本阶段新完成 epoch3 `best.pt` 的 validation-only detailed evaluation：

- `liver_7`：Dice=`0.0284302355`，IoU=`0.0144201012`，Precision=`0.0144321372`，Recall=`0.9453278098`，HD95=`248.8152 mm`，ASSD=`80.5652 mm`；prediction/GT foreground=`45.8254% / 0.69961%`，ratio=`65.5016×`；pred/GT components=`398/3`，component error=`395`，false merge/break=`1/1`；uncertainty AUROC/AUPRC=`0.65563/0.54434`，Top-10% error recall=`0.11556`；ECE/MCE/Brier/NLL=`0.42213/0.44697/0.85822/3.65555`，confidence gap=`0.42213`；CPU inference≈`76.16 s`。
- `liver_8`：Dice=`0.0276252609`，IoU=`0.0140060914`，Precision=`0.0140087668`，Recall=`0.9865479483`，HD95=`264.0114 mm`，ASSD=`86.9848 mm`；prediction/GT foreground=`39.8568% / 0.56596%`，ratio=`70.4236×`；pred/GT components=`466/2`，component error=`464`，false merge/break=`1/0`；uncertainty AUROC/AUPRC=`0.68646/0.51508`，Top-10% error recall=`0.13092`；ECE/MCE/Brier/NLL=`0.36425/0.42877/0.74279/3.15755`，confidence gap=`0.36425`；CPU inference≈`124.27 s`。

两例平均 v12：Dice=`0.0280277482`、IoU=`0.0142130963`、Precision=`0.0142204520`、Recall=`0.9659378790`、HD95=`256.4133 mm`、ASSD=`83.7750 mm`、prediction/GT foreground ratio=`67.9626×`、component error=`429.5`、uncertainty AUROC/AUPRC=`0.67105/0.52971`、Top-10% error recall=`0.12324`、ECE/MCE/Brier/NLL=`0.39319/0.43787/0.80051/3.40655`、confidence gap=`0.39319`、CPU inference≈`100.22 s`。

与 v11 CT-only epoch3 同两例严格对照：v11 平均 Dice=`0.0546575740`、IoU=`0.0281212393`、Precision=`0.0342507730`、Recall=`0.1355391270`、HD95=`186.0500 mm`、ASSD=`51.5220 mm`、prediction/GT foreground ratio=`3.9976×`、component error=`1548.0`、uncertainty AUROC/AUPRC=`0.93691/0.32938`、Top-10% error recall=`0.76162`、ECE/MCE/Brier/NLL=`0.01084/0.05635/0.04693/0.10288`、confidence gap=`0.01078`、CPU inference≈`72.88 s`。v12 虽然 Recall 大幅升高且 component count error 数值更低，但这是因为模型把约 40%–46% 的全卷都预测成前景，形成约 `68×` 的前景泛滥；区域重叠、Precision、表面距离和 calibration 均显著恶化，因此不能把高 Recall 或较低 component error 单独解释为结构改善。

v12 epoch1 checkpoint 已被后续 `best.pt/last.pt` 覆盖，本阶段没有伪造不存在的 epoch1/epoch2 checkpoint。改用当前 epoch3 checkpoint 内的 AdamW 历史 step 计数 + BN buffer 进行 freeze verification：optimizer 共 205 个 parameter state，step 值只有 `28` 和 `84`；按模型参数顺序对齐后，`segformer_encoder` 184 个参数全部 step=`28`，decoder `linear_c1..c4` 16 个参数与 `linear_fuse` 3 个参数也全部 step=`28`，仅 `linear_pred` weight+bias 两个参数 step=`84`。由于每 epoch 真实为 28 optimizer steps，这与“epoch1 全模型训练，epoch2/3 冻结 encoder + decoder feature，仅 final head 持续训练”完全一致。9 个 BatchNorm3d 的 `num_batches_tracked` 全部为 `28`，而非 84，证明 BN running statistics 也只在 epoch1 更新。`best.pt` 与 `last.pt` 的 model state 在 epoch3 exact equal。该证据足以验证 freeze policy 的历史执行，但不冒充不存在的 epoch2 checkpoint state delta 文件。

输入消融最终判定：**CT-only（v11）better，作为后续 loss ablation baseline。** v12 CT+bone-window 在当前 normalization/architecture 下产生严重 foreground overprediction，STOP，不继续扩展该输入方向。stable baseline=`YES`（仍指 v11 engineering/validation stable baseline）；lock parameters=`NO`；formal independent test ready=`NO`；本轮 `liver_169=未访问`。

下一步立即以 v11 CT-only 为固定 input baseline 进入 loss ablation：Region、Region+Boundary、Region+Topology、Region+Boundary+Topology；每次只改变 loss 这一主要变量，继续保持 ROI/sampling/lr/scheduler/augmentation/seed/freeze policy/validation 不变。若 3-epoch minimal comparison 已明显灾难性失败，则按 STOP 规则记录并进入下一项，避免浪费 CPU。


### 2026-08-28｜阶段 BC：v13 Region+Boundary loss ablation 完成

以输入消融胜出的 v11 CT-only stable baseline 为唯一基线，v13=`configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml` 只改变 loss composition：由 Region Dice+CE 改为 `joint_orthopedic`，保持 region=`1.0`、新增 boundary=`0.1`、topology=`0.0`；input、ROI、sampling、lr、scheduler、seed、freeze policy、validation 与 inference 均不变。config diff 未发现其它主要实验变量。

真实 run=`experiments/20260828_002035_cpu_binary_loss_region_boundary_v13_roi64`。最初两次工具 timeout 留下 `20260828_001444...` 与 `20260828_002420...` 空 history run，均未作为结果；有效 run 未重复训练。三轮结果：epoch1 train loss=`2.5547913696084703`、mean val Dice=`0.05414399340794464`；epoch2 train loss=`2.305156431027821`、mean val Dice=`0.05443117660509808`；epoch3 train loss=`1.830046398299081`、mean val Dice=`0.054709440953703406`、std=`0.009402906878641024`。三轮 sampling 与 v11 对应 epoch 完全一致：28 patches/epoch，foreground/background=`10/18、10/18、8/20`，foreground fraction mean=`0.07907336/0.08840765/0.05680016`。

v13 epoch3 `best.pt` validation-only detailed evaluation：`liver_7` Dice=`0.0453065341`、IoU=`0.0231783320`、Precision=`0.0279913112`、Recall=`0.1187878242`、HD95=`197.3914 mm`、ASSD=`55.3591 mm`、prediction/GT foreground ratio=`4.2437×`、pred/GT components=`1564/3`、component error=`1561`、false merge/break=`1/69`；`liver_8` Dice=`0.0641123478`、IoU=`0.0331178040`、Precision=`0.0404694855`、Recall=`0.1541957466`、HD95=`174.5083 mm`、ASSD=`47.6139 mm`、ratio=`3.8102×`、pred/GT components=`1528/2`、component error=`1526`、false merge/break=`0/60`。两例平均 Dice=`0.05470944095`、IoU=`0.0281480680`、Precision=`0.0342303983`、Recall=`0.1364917854`、HD95=`185.9498359 mm`、ASSD=`51.4864689 mm`、foreground ratio=`4.026956×`、component error=`1543.5`。

与 v11 Region 同两例严格对照：v11 平均 Dice=`0.0546575740`、HD95=`186.0500241 mm`、ASSD=`51.5220126 mm`、Precision=`0.0342507730`、Recall=`0.1355391270`、foreground ratio=`3.997598×`、component error=`1548.0`。因此 Boundary 使 Dice 仅提升约 `5.19e-5`，HD95 仅改善约 `0.1002 mm`、ASSD 约 `0.0355 mm`、component error 约 `4.5`；同时 Precision 略降、foreground ratio 略升、false break 平均 `64.0→64.5`，ECE/Brier/NLL 也有极小恶化。科学判断：**Region+Boundary 在当前 0.1 权重下只显示轻微且非常弱的表面改善证据，不能写成明确收益；不因该微小差异锁定 Boundary。**

freeze verification 继续通过：v13 epoch3 checkpoint optimizer state 中 203 个 frozen-group 参数 step=`28`，仅 final `linear_pred` 2 个参数 step=`84`；9 个 BatchNorm3d `num_batches_tracked=28`。这与 epoch2 起 encoder + decoder feature + BN-running-stat freeze、仅 final head 继续训练的策略一致。独立 test `liver_169` 本阶段未访问。stable baseline 仍为 engineering/validation 级；lock parameters=`NO`、formal independent test ready=`NO`。

下一步直接进入 v14 Region+Topology，重点看 component count / false merge / false break 是否相对 Region 有真实改善；随后 v15 Region+Boundary+Topology。


### 2026-08-28｜阶段 BD：v14 Region+Topology loss ablation 完成

v14=`configs/orthopedic_ct_cpu_binary_loss_region_topology_v14.yaml` 继续以 v11 CT-only Region stable baseline 为唯一对照，只改变 loss composition：`region=1.0`、`boundary=0.0`、`topology=0.1`、`topology_iterations=10`；input、ROI、sampling、lr、scheduler、seed、freeze policy、validation 与 inference 均保持不变。正式 preflight 已再次通过：`ready=true / blocker_count=0`。独立 test `ctspine1k-msd-t10-liver_169` 本阶段未访问。

最初 run=`experiments/20260828_121524_cpu_binary_loss_region_topology_v14_roi64` 在 epoch1 训练 28 patches 后因工具 300 秒上限停在 full-volume validation，未产生 history/checkpoint，因此明确不作为结果。有效 run=`experiments/20260828_122048_cpu_binary_loss_region_topology_v14_roi64`：epoch1 train loss=`2.64983754498618`、mean val Dice=`0.0545078970525526`；epoch2 由同一 run `last.pt` resume 后 train loss=`2.388446888753346`、mean val Dice=`0.05450463747160135`；epoch3 再由同一 run resume 后 train loss=`1.9175574907234736`、mean val Dice=`0.05450932691321425`、std=`0.0061590328625196755`。三轮 sampling 仍为 28 patches/epoch，foreground/background=`10/18、10/18、8/20`，foreground fraction mean=`0.07907336/0.08840765/0.05680016`，与 v11 对应 epoch 一致。

v14 epoch3 `best.pt` validation-only detailed evaluation 采用分病例执行，避免 CPU full-volume evaluation 再次触发工具超时：

- `liver_7`：Dice=`0.0483502940`、IoU=`0.0247740637`、Precision=`0.0288167426`、Recall=`0.1500882565`、HD95=`194.2215 mm`、ASSD=`54.2587 mm`；prediction/GT foreground=`3.6438% / 0.69961%`，ratio=`5.20837×`；pred/GT components=`1545/3`，component error=`1542`，false merge/break=`1/65`；uncertainty AUROC/AUPRC=`0.93203/0.35444`，Top-10% error recall=`0.72025`；ECE/MCE/Brier/NLL=`0.01776/0.09953/0.06347/0.13382`，confidence gap=`0.01773`；CPU inference≈`59.68 s`。
- `liver_8`：Dice=`0.0606683598`、IoU=`0.0312831279`、Precision=`0.0365756981`、Recall=`0.1777602455`、HD95=`173.7613 mm`、ASSD=`46.9011 mm`；prediction/GT foreground=`2.7506% / 0.56596%`，ratio=`4.86006×`；pred/GT components=`1540/2`，component error=`1538`，false merge/break=`0/59`；uncertainty AUROC/AUPRC=`0.94693/0.34346`，Top-10% error recall=`0.79514`；ECE/MCE/Brier/NLL=`0.01051/0.06257/0.04737/0.09661`，confidence gap=`0.01038`；CPU inference≈`90.61 s`。

两例平均 v14：Dice=`0.0545093269`、IoU=`0.0280285958`、Precision=`0.0326962203`、Recall=`0.1639242510`、HD95=`183.9914268 mm`、ASSD=`50.5799253 mm`、prediction/GT foreground ratio=`5.034217×`、component error=`1540.0`、false merge=`0.5`、false break=`62.0`；uncertainty AUROC/AUPRC=`0.93948/0.34895`、Top-10% error recall=`0.75770`；ECE/MCE/Brier/NLL=`0.01413/0.08105/0.05542/0.11522`、confidence gap=`0.01406`；CPU inference≈`75.15 s`。

与 v11 Region 同两例严格对照：v11 平均 Dice=`0.0546575740`、HD95=`186.0500241 mm`、ASSD=`51.5220126 mm`、Precision=`0.0342507730`、Recall=`0.1355391270`、prediction/GT foreground ratio=`3.997598×`、component error=`1548.0`、false merge=`0.5`、false break=`64.0`。因此 Topology 使 component error `1548→1540`、false break `64→62`、HD95 改善约 `2.0586 mm`、ASSD 改善约 `0.9421 mm`，且 uncertainty AUROC/AUPRC 分别约 `+0.00257/+0.01957`；但 Dice 下降约 `1.48e-4`、Precision 下降约 `0.00155`、prediction/GT foreground ratio 增加约 `1.04×`，ECE/Brier/NLL/confidence gap 均变差，Top-10% error recall 也略降。科学判断：**Region+Topology 对结构/表面指标出现一定改善信号，但代价是更强 foreground overprediction、较差 calibration 与轻微 Dice/Precision 下降；当前证据不足以判定 Region+Topology 整体优于 Region，正式结论为 evidence inconclusive。** 因此不锁定 Topology，继续 v15 Region+Boundary+Topology 完成 loss ablation。

freeze/checkpoint verification 继续通过：v14 epoch3 `best.pt` 的 AdamW state 共 205 个参数，其中 step=`28` 有 203 个、step=`84` 有 2 个；9 个 BatchNorm `num_batches_tracked` 均为 `28`。这与 epoch1 全模型训练、epoch2/3 冻结 encoder + decoder feature + BN running stats、仅 final `linear_pred` 两个参数继续更新的策略一致。stable baseline 仍为 engineering/validation 级；lock parameters=`NO`；formal independent test ready=`NO`；`liver_169=未访问`。

下一步直接执行 v15 Region+Boundary+Topology；完成后对 Region / Region+Boundary / Region+Topology / Region+Boundary+Topology 做统一 loss ablation 判断，再进入 sampling ablation。


### 2026-08-28｜阶段 BE：v15 联合损失完成，loss ablation 正式闭环

v15=`configs/orthopedic_ct_cpu_binary_loss_region_boundary_topology_v15.yaml`，以 v11 CT-only stable mechanism 为固定基础，仅将 loss composition 设为 `region=1.0 / boundary=0.1 / topology=0.1`（`topology_iterations=10`）；ROI、input、sampling、lr、scheduler、seed、freeze policy、validation 与 inference 保持不变。formal readiness 已真实通过：`ready=true / blocker_count=0`。有效 run=`experiments/20260828_142401_cpu_binary_loss_region_boundary_topology_v15_roi64`，3 epoch 已完成且不重跑：epoch1 train loss=`2.6507615830`、mean val Dice=`0.0545193722`、FG/BG patches=`10/18`；epoch2 train loss=`2.3858454398`、mean val Dice=`0.0545209539`、FG/BG=`10/18`；epoch3 train loss=`1.9158078730`、mean val Dice=`0.0544770773`、FG/BG=`8/20`。best Dice=`0.0545209539`，因此 `best.pt` 对应 epoch2。

v15 `best.pt` validation-only detailed evaluation 已按病例完成，未访问独立 test：`liver_7` Dice=`0.0480255358`、IoU=`0.0246035677`、Precision=`0.0288159219`、Recall=`0.1440615994`、HD95=`195.1489 mm`、ASSD=`54.4414 mm`、prediction/GT foreground ratio=`4.999375×`、pred/GT components=`1539/3`、component error=`1536`、false merge/break=`1/64`、uncertainty AUROC/AUPRC=`0.93160/0.35201`、Top-10% error recall=`0.72230`、ECE/MCE/Brier/NLL=`0.01692/0.09481/0.06154/0.13101`、confidence gap=`0.01692`、CPU inference≈`61.28 s`；`liver_8` Dice=`0.0610163721`、IoU=`0.0314682245`、Precision=`0.0371083808`、Recall=`0.1715266640`、HD95=`174.0029 mm`、ASSD=`47.0610 mm`、prediction/GT foreground ratio=`4.622316×`、pred/GT components=`1546/2`、component error=`1544`、false merge/break=`0/56`、uncertainty AUROC/AUPRC=`0.94663/0.34604`、Top-10% error recall=`0.79573`、ECE/MCE/Brier/NLL=`0.00975/0.05533/0.04562/0.09419`、confidence gap=`0.00975`、CPU inference≈`80.42 s`。两例平均 v15：Dice=`0.0545209539`、IoU=`0.0280358961`、Precision=`0.0329621513`、Recall=`0.1577941317`、HD95=`184.5758955 mm`、ASSD=`50.7511907 mm`、prediction foreground fraction=`0.03056820`、GT foreground fraction=`0.00632783`、foreground ratio=`4.810846×`、pred/GT components=`1542.5/2.5`、component error=`1540.0`、false merge=`0.5`、false break=`60.0`、uncertainty AUROC/AUPRC=`0.93911/0.34902`、Top-10% error recall=`0.75901`、ECE/MCE/Brier/NLL=`0.01334/0.07507/0.05358/0.11260`、confidence gap=`0.01334`、CPU inference≈`70.85 s`。

四组同两例 loss ablation 统一比较（全部为 validation，不是 independent test）：

| Loss | Dice | IoU | Precision | Recall | HD95 mm | ASSD mm | Pred/GT FG | Comp. error | False merge | False break | AUROC | AUPRC | Top-10% err recall | ECE | MCE | Brier | NLL | Conf. gap | Infer s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v11 Region | 0.054658 | 0.028121 | 0.034251 | 0.135539 | 186.0500 | 51.5220 | 3.9976× | 1548.0 | 0.5 | 64.0 | 0.93691 | 0.32938 | 0.76162 | 0.01084 | 0.05635 | 0.04693 | 0.10288 | 0.01078 | 72.88 |
| v13 Region+Boundary | **0.054709** | **0.028148** | 0.034230 | 0.136492 | 185.9498 | 51.4865 | 4.0270× | 1543.5 | 0.5 | 64.5 | 0.93707 | 0.33184 | **0.76170** | 0.01093 | 0.05790 | 0.04717 | 0.10325 | 0.01089 | 75.01 |
| v14 Region+Topology | 0.054509 | 0.028029 | 0.032696 | **0.163924** | **183.9914** | **50.5799** | 5.0342× | **1540.0** | 0.5 | 62.0 | **0.93948** | 0.34895 | 0.75770 | 0.01413 | 0.08105 | 0.05542 | 0.11522 | 0.01406 | 75.15 |
| v15 Region+Boundary+Topology | 0.054521 | 0.028036 | 0.032962 | 0.157794 | 184.5759 | 50.7512 | 4.8108× | **1540.0** | 0.5 | **60.0** | 0.93911 | **0.34902** | 0.75901 | 0.01334 | 0.07507 | 0.05358 | 0.11260 | 0.01334 | **70.85** |

科学判断：Boundary 单独加入时仅带来极弱的 HD95/ASSD 与 Dice 改善，不能称为明确收益；Topology（v14/v15）对 HD95/ASSD、component error、false break 和 uncertainty AUPRC 出现更明显的改善信号，但同时提高 prediction/GT foreground ratio、降低 Precision，并使 ECE/MCE/Brier/NLL/confidence gap 相对 v11/v13 变差。v15 相对 v14 能将 false break `62→60`、foreground ratio `5.034→4.811` 并略改善 calibration，但仍未消除 Topology 带来的前景过预测代价，且 Dice 仍低于 v11/v13。综合区域、表面、结构、前景、uncertainty、calibration 与速度，当前证据不足以证明 Topology 组合整体优于无 Topology 方案。

因此 loss ablation 的后续工程决策为：**选择 v13 Region+Boundary 作为 sampling ablation baseline**。理由不是“Boundary 已被证明显著有效”，而是它在四组中取得最高两例平均 Dice/IoU，同时 foreground overprediction 与 calibration 基本维持 v11 水平，表面指标也未恶化；这是当前小样本 validation 下最保守、风险最低的工作基线。Topology 保留为后续可复查候选，但当前不进入 sampling baseline。`lock parameters=NO`、`formal independent test ready=NO`，`ctspine1k-msd-t10-liver_169=未访问`。

下一步固定 v13 的 input/loss/lr/scheduler/ROI/freeze policy，只做 sampling 单变量消融：current Bernoulli baseline → fixed-per-case → boundary hard sampling；每个阶段继续记录 patches/case、FG/BG、foreground fraction mean/std、区域/表面/结构、uncertainty、calibration 与 inference time，再选择 sampling baseline。

### 2026-08-28｜阶段 BF：v16/v17 sampling ablation 完成，保留 v13 Bernoulli baseline

本阶段严格固定 v13 的 CT-only、Region+Boundary loss、64³ training ROI、AdamW peak lr=`5e-5`、warmup/cosine scheduler、seed、epoch2 起 encoder+decoder-feature+BN-running-stat freeze、full-volume validation 与 inference，仅改变 sampling 策略。独立 test `ctspine1k-msd-t10-liver_169` 继续未访问；`lock parameters=NO`、`formal independent test ready=NO`。

v16=`configs/orthopedic_ct_cpu_binary_sampling_fixed_per_case_v16.yaml`，有效 run=`experiments/20260828_150343_cpu_binary_sampling_fixed_per_case_v16_roi64`。三轮 mean validation Dice=`0.04557463 → 0.04564599 → 0.04575062`。sampling foreground-fraction mean=`0.0645966/0.0681651/0.0734618`，跨 epoch mean 的 std≈`0.003642`、range≈`0.008865`，明显比 v13 Bernoulli 的 std≈`0.013259`、range≈`0.031607` 更稳定。但 best.pt 两例 detailed validation 平均 Dice=`0.04575062`、IoU=`0.02369179`、Precision=`0.02616929`、Recall=`0.18255175`、HD95=`181.2707 mm`、ASSD=`54.7019 mm`、prediction/GT foreground ratio=`6.51485×`、component error=`1302.0`、false merge=`0.5`、false break=`76.5`、uncertainty AUROC/AUPRC=`0.92582/0.34370`、Top-10% error recall=`0.66407`、ECE/MCE/Brier/NLL=`0.02964/0.19434/0.07493/0.17802`、confidence gap=`0.02964`。因此 fixed-per-case 虽改善 sampling stability，但区域分割、foreground overprediction、ASSD、false break 与 calibration 均整体劣于 v13，**v16 不选**。

v17=`configs/orthopedic_ct_cpu_binary_sampling_boundary_hard_v17.yaml`，有效 run=`experiments/20260828_152712_cpu_binary_sampling_boundary_hard_v17_roi64`。接管时确认训练已真实完成，未重复启动：epoch1/2/3 train loss=`2.52750253/2.27737889/2.45061852`，mean validation Dice=`0.03701411 → 0.03715640 → 0.03730737`，best.pt=epoch3。sampling foreground-fraction mean=`0.1123220/0.1468705/0.1491953`，FG/BG patch=`15/13、18/10、14/14`，跨 epoch mean 的 std≈`0.016861`、range≈`0.036873`，并未比 v13 更稳定。

随后复用同一 v17 best.pt 对 validation `liver_7/liver_8` 完成 detailed evaluation。两例平均：Dice=`0.03730737`、IoU=`0.01900976`、Precision=`0.01917430`、Recall=`0.69148852`、HD95=`206.5001 mm`、ASSD=`58.8920 mm`、prediction/GT foreground ratio=`36.2590×`、component error=`568.0`、false merge=`0.5`、false break=`33.0`、uncertainty AUROC/AUPRC=`0.87484/0.54115`、Top-10% error recall=`0.24014`、ECE/MCE/Brier/NLL=`0.19569/0.46998/0.40404/1.20141`、confidence gap=`0.19569`、CPU inference≈`72.85 s`。虽然 component error/false break 与 uncertainty AUPRC 表面上改善，但这是伴随极端 foreground overprediction、极高 Recall、低 Precision、恶化 surface distance 与严重 calibration 崩坏出现的，不能视为整体结构质量提升。**v17 明显失败，STOP，不继续浪费 CPU。**

三组 sampling 统一 comparison（validation 两例平均；不是 independent test）：

| Sampling | Dice | IoU | Precision | Recall | HD95 mm | ASSD mm | Pred/GT FG | Comp. error | False break | AUROC | AUPRC | Top-10% err recall | ECE | Brier | NLL | Sampling mean std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v13 Bernoulli | **0.054709** | **0.028148** | **0.034230** | 0.136492 | 185.9498 | **51.4865** | **4.0270×** | 1543.5 | 64.5 | **0.93707** | 0.33184 | **0.76170** | **0.01093** | **0.04717** | **0.10325** | 0.013259 |
| v16 fixed-per-case | 0.045751 | 0.023692 | 0.026169 | 0.182552 | **181.2707** | 54.7019 | 6.5148× | 1302.0 | 76.5 | 0.92582 | 0.34370 | 0.66407 | 0.02964 | 0.07493 | 0.17802 | **0.003642** |
| v17 boundary-hard | 0.037307 | 0.019010 | 0.019174 | **0.691489** | 206.5001 | 58.8920 | 36.2590× | **568.0** | **33.0** | 0.87484 | **0.54115** | 0.24014 | 0.19569 | 0.40404 | 1.20141 | 0.016861 |

最终 sampling 决策：**继续使用 v13 current Bernoulli sampling（foreground_probability=`0.25`、patches_per_case=`4`）作为后续 augmentation / difficult-sample validation baseline。** 选择依据不是 sampling stability 单指标，而是区域、前景比例、表面、结构、uncertainty 与 calibration 的综合 validation 结果。v16 证明“更稳定的 sampling statistics”本身不足以带来更好的 segmentation；v17 证明当前 boundary-hard 方案会严重推高 foreground prior 并破坏 calibration。下一阶段固定 v13 input/loss/sampling/ROI/lr/scheduler/freeze policy，优先做 standard augmentation 与 intensity/HU augmentation 的最小单变量 validation；当前数据不足以真实构造 metal artifact / fracture / low-density / thick-slice difficult subset 时必须明确记为“数据不足 / 未完成”。


### 2026-08-29｜阶段 BG：v18 standard augmentation 完成并判定不选

接管时重新核验本地真实断点：`HEAD=origin/main=9fbf35d23e1dc829c8e3bf5287089efe00bba27f`，工作树仅有未跟踪的 `configs/orthopedic_ct_cpu_binary_aug_standard_v18.yaml`；Windows 当前没有项目训练 Python 进程。有效 run=`experiments/20260828_163229_cpu_binary_aug_standard_v18_roi64` 已于 2026-08-28 16:48:59 完成 3 epoch，未重跑。`history.csv`：epoch1/2/3 train loss=`2.41104026/2.16369422/1.65625836`，mean full-volume validation Dice=`0.05534875 → 0.05466724 → 0.05402823`，best=`epoch1`。sampling foreground-fraction mean=`0.0786631/0.0863857/0.0552617`，FG/BG patch=`10/18、10/18、8/20`。

随后复用同一 `best.pt`，分别完成 validation `ctspine1k-msd-t10-liver_7` 与 `...liver_8` detailed evaluation，formal preflight 均 `ready=true / 0 error / 0 warning`，未访问 `liver_169`。两例平均：Dice=`0.05534875`、IoU=`0.02846219`、Precision=`0.03134055`、Recall=`0.23691342`、HD95=`176.8493 mm`、ASSD=`51.3895 mm`、prediction/GT foreground ratio=`7.55673×`、component error=`1271.0`、false merge=`0.5`、false break=`44.5`、uncertainty AUROC/AUPRC=`0.94844/0.42240`、Top-10% error recall=`0.77338`、ECE/MCE/Brier/NLL=`0.02716/0.22515/0.07557/0.14512`、confidence gap=`0.02714`、CPU inference≈`67.46 s`。

与 v13 Bernoulli + Region+Boundary baseline（Dice=`0.05470944`、Precision=`0.03423040`、HD95/ASSD=`185.9498/51.4865 mm`、foreground ratio=`4.02696×`、component error=`1543.5`、AUROC/AUPRC=`0.93707/0.33184`、Top-10% error recall=`0.76170`、ECE/MCE/Brier/NLL=`0.01093/0.05790/0.04717/0.10325`）比较：v18 的 Dice 仅提高约 `0.000639`，HD95、component error、uncertainty 指标有改善信号，但 Precision 下降、foreground overprediction 从约 `4.03×` 增至 `7.56×`，并且 ECE/MCE/Brier/NLL 全面恶化。该小样本 validation 下，standard rotation/scale 的极弱 Dice 收益不足以抵消前景先验与 calibration 风险。

最终 standard augmentation 决策：**v18 不选；后续 geometric augmentation 状态回到 v13 的既有 flip-only baseline。** 下一步只改变 intensity augmentation，分别验证 gamma、Gaussian noise、HU shift；继续固定 CT-only、Region+Boundary、Bernoulli sampling（foreground_probability=`0.25`、patches_per_case=`4`）、ROI、lr、scheduler、seed、freeze policy、full-volume validation 与 inference。`lock parameters=NO`、`formal independent test ready=NO`，`ctspine1k-msd-t10-liver_169=未访问`。


### 2026-08-29｜阶段 BH：v19–v21 intensity augmentation 完成，最终保留 v13

本轮重新核验后确认 `HEAD=origin/main=9fbf35d23e1dc829c8e3bf5287089efe00bba27f`，上一阶段 v18 文档与 v18–v21 config 尚未提交；独立 test `ctspine1k-msd-t10-liver_169` 本轮仍未访问。

**v19 gamma augmentation（0.9–1.1）**：唯一真实 run=`experiments/20260829_021301_cpu_binary_aug_gamma_v19_roi64`。`history.csv` 只有两轮：epoch1 train loss=`2.55083960`、mean full-volume validation Dice=`0.04723427`；epoch2 train loss=`2.06645347`、Dice=`0.04693522`。连续两轮明显低于 v13 baseline `0.05470944` 且没有恢复趋势，因此已按 CPU STOP 规则终止，不跑 epoch3，不重新启动。结论：**v19 FAIL / 不选**。

**v20 Gaussian noise（std 0–0.02）**：唯一真实 run=`experiments/20260829_022332_cpu_binary_aug_gaussian_v20_roi64` 已完整完成 3 epoch，未重跑。epoch1/2/3 train loss=`2.51865584/2.33694819/1.86678295`，mean full-volume validation Dice=`0.05281761 → 0.05426826 → 0.05530480`，best=epoch3。sampling foreground-fraction mean=`0.07907336/0.08840765/0.05680016`，FG/BG patch=`10/18、10/18、8/20`。随后复用同一 `best.pt` 对 validation `liver_7/liver_8` 完成 detailed evaluation；两例 Dice=`0.03964278/0.07096681`，平均 Dice=`0.05530480`、IoU=`0.02850551`、Precision=`0.03774130`、Recall=`0.10365574`、HD95=`194.1709 mm`、ASSD=`54.1179 mm`、prediction/GT foreground ratio=`2.79203×`、component error=`1635.5`、false merge=`0.5`、false break=`68.0`、uncertainty AUROC/AUPRC=`0.92906/0.29070`、Top-10% error recall=`0.74789`、ECE/MCE/Brier/NLL=`0.00827/0.03347/0.03679/0.09091`、confidence gap=`0.00820`、CPU inference≈`68.26 s`。

与 v13 对比，v20 mean Dice 仅提高 `0.00059536`，Precision、foreground overprediction、ECE/MCE/Brier/NLL 与 CPU inference 有改善；但 Recall 下降约 `0.03284`，HD95 恶化约 `8.22 mm`，ASSD 恶化约 `2.63 mm`，component error 增加 `92`，false break 增加 `3.5`，uncertainty AUROC/AUPRC/Top-10% error recall 分别下降约 `0.00802/0.04114/0.01381`。因此小样本 validation 下证据高度混合，极弱 Dice 增益不足以抵消表面、结构与 uncertainty ranking 的退化。结论：**v20 有局部改善信号，但综合不取代 v13**。

**v21 HU shift（-50～+50 HU）**：已确认 Dataset 的 `ct_normalized` 强度增强使用 preprocessing metadata 的 `clipped_mean_hu/clipped_std_hu`，以 HU-domain 等价方式实施 shift，不把 z-score 错误裁剪到 `[0,1]`。唯一真实 run=`experiments/20260829_025018_cpu_binary_aug_hu_shift_v21_roi64`。epoch1 train loss=`2.55658871`、Dice=`0.03671569`；按规则允许 epoch2 复核，epoch2 train loss=`2.43075695`、Dice=`0.03871272`。虽然略有回升，但仍比 v13 低约 `0.0160`，远未恢复到可竞争区间，因此已主动停止父/子训练进程，不跑 epoch3。该 STOP run 没有伪造 `summary.json`；保留真实 `history.csv`、`sampling_stats.csv`、`best.pt/last.pt`。结论：**v21 FAIL / 不选**。

最终 augmentation 决策：**v18 standard geometric 不选、v19 gamma 不选、v20 Gaussian 不取代 baseline、v21 HU shift 不选；正式保留 v13 原有 flip-only augmentation。** 当前固定 validation baseline 继续为：CT-only + Region+Boundary + Bernoulli sampling（foreground_probability=`0.25`、patches_per_case=`4`）+ flip-only geometric augmentation + ROI=`64³` training + AdamW peak lr=`5e-5` + 既定 scheduler/seed + epoch2 起 encoder/decoder-feature/BN-running-stat freeze + full-volume validation。当前仍为 `lock parameters=NO`、`formal independent test ready=NO`，下一阶段进入 uncertainty/calibration、真实 difficult-sample 分析与 refinement；所有阈值/策略仍只能由 `liver_7/liver_8` validation 决定。

### 2026-08-29｜阶段 BI：v13 uncertainty / calibration 两例稳定性分析完成

重新接管后真实核验：`HEAD=origin/main=c090514c6611c295d5ca5932f937e10713ea5bf7`，工作树 clean；没有项目训练 Python 进程。v13 `best.pt` 存在，且 `experiments/evaluation_20260828_v13e3_liver7` / `...liver8` 中的 `prediction.nii.gz`、`predictive_entropy.nii.gz`、`metrics_per_case.csv`、`summary.json` 均完整，因此本阶段**直接复用已有 validation inference，不重复推理**。独立 test `ctspine1k-msd-t10-liver_169` 未访问。

逐病例真实 uncertainty / calibration：

| 指标 | liver_7 | liver_8 | 两例均值 | 两例绝对差 |
|---|---:|---:|---:|---:|
| uncertainty→error AUROC | 0.929493 | 0.944655 | 0.937074 | 0.015163 |
| uncertainty→error AUPRC | 0.333540 | 0.330136 | 0.331838 | 0.003403 |
| Top-10% error recall | 0.728909 | 0.794498 | 0.761704 | 0.065589 |
| mean entropy on error | 0.562366 | 0.593707 | 0.578037 | 0.031341 |
| mean entropy on correct | 0.054027 | 0.047809 | 0.050918 | 0.006218 |
| ECE | 0.014179 | 0.007673 | 0.010926 | 0.006506 |
| MCE | 0.075926 | 0.039879 | 0.057903 | 0.036047 |
| Brier | 0.054654 | 0.039693 | 0.047174 | 0.014961 |
| NLL | 0.120785 | 0.085713 | 0.103249 | 0.035072 |
| confidence gap | 0.014109 | 0.007673 | 0.010891 | 0.006436 |

两例 error voxel 的平均 entropy 分别约为 correct voxel 的 `10.41×` 与 `12.42×`；同时 AUROC 两例均 >`0.92`、Top-10% uncertainty 区域可覆盖约 `72.9%/79.4%` 的真实错误。**因此在当前两例 validation 上，uncertainty 可以作为较强的 error indicator，并具备 QC signal 与 ROI refinement trigger 的直接定量依据。** AUPRC 两例约 `0.33` 且非常接近，说明 ranking 在两例之间有一定一致性；但 error prevalence 仅约 `3.50%/2.55%`，AUPRC 不应与 AUROC脱离基线单独夸大。

calibration 方面 ECE / confidence gap 都较低，`liver_8` 好于 `liver_7`，但 MCE、Brier、NLL 仍存在明显病例差异；更重要的是当前 segmentation Dice 仍仅 `0.04531/0.06411`，低 ECE 主要反映绝大多数背景体素上的整体置信行为，**不能据此宣称模型在骨结构前景上“已校准”或临床可靠**。当前严谨结论为：uncertainty ranking 的两例稳定性较好；calibration 的总体数值较低但病例间仍有差异，且受严重类别不平衡影响，只能作为 validation 工程证据。

科研问题结论：1) error indicator=`YES（validation evidence）`；2) QC signal=`YES（可用于高风险区域提示，但需更多病例验证）`；3) refinement trigger=`YES（优先使用 percentile/top-percent ROI，阈值只能由 liver_7/liver_8 决定）`；4) 两例 stability=`部分支持`，AUROC/AUPRC 和 error-vs-correct entropy 一致性较好，Top-10% recall 与 calibration 指标存在病例差；5) 支持指标为 AUROC、AUPRC、Top-10% error recall、error/correct entropy gap，限制指标/证据为仅 2 个 validation 病例、低 Dice、类别不平衡及 MCE/Brier/NLL 的病例差异。

本阶段不改变 v13 baseline，不锁参：`lock parameters=NO`、`formal independent test ready=NO`。下一步进入 high-loss / high-uncertainty difficult mining 与真实 thick-slice/data-evidence 核验，然后再做 ROI refinement。
