# 骨科 CT 智能分割与三维重建项目——主进度与交接台账

> 项目目录：`D:\国创项目`
>
> 项目名称：**基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究**
>
> 台账首次建立：2026-08-15
>
> 最近更新：2026-08-16

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

> **特别强调：任务书中的 Dice ≥ 0.93 是项目目标，不是当前实验结果。当前尚未进行真实数据正式训练，因此没有可写入论文 Results 的 Dice/HD95/ASSD。**

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

## 2. 当前总体状态（2026-08-16）

| 模块 | 状态 | 完成度 | 当前真实状态 |
|---|---|---:|---|
| 任务书/组会材料梳理 | ✅ 已完成 | 100% | 已提取研究目标、时间轴、系统功能、论文/中期要求 |
| SegFormer3D 上游调研 | ✅ 已完成 | 100% | 已读 README、核心架构、loss、依赖与许可证；官方仓库已克隆到 `third_party/SegFormer3D` |
| 项目目录与交接机制 | ✅ 已完成 | 100% | 已建立工程目录和本主台账；明确“每次实质修改必须更新本文件” |
| 总体方案设计 | ✅ 已完成 | 100% | 已形成数据层、模型层、三维层、Web 层和实验追踪设计 |
| 国内外文献调研 | 🟡 进行中 | 94% | 已形成 40 条结构化文献矩阵与 38 条英文核心 BibTeX；新增核验 2025 SpineMamba、2025 椎体分割/解剖变异 Transformer、2026 VertebraFormer，已覆盖现代 3D spine、shape prior、domain generalization。后续重点转向金属伪影/低骨密度困难病例、现代强 baseline 与国内 CNKI/万方正式核验 |
| 实验环境 | ✅ 已完成（CPU 开发环境） | 92% | 项目内 Python 3.11.7 + `.venv` 已完成；PyTorch 2.1.0 CPU、MONAI、DICOM/Web/Lightning 依赖均可导入；新增 GPU/CUDA 只读检查。当前实测 torch=`2.1.0+cpu`、CUDA=false、0 device、无 `nvidia-smi`，因此正式 GPU 训练仍阻塞 |
| DICOM/CT 处理流程 | 🟡 进行中 | 93% | NIfTI pipeline 0.3.0 已在 10 例真实 CTSpine1K CT+label 上完成 1 mm 重采样、HU clip→case-wise z-score、骨窗、label nearest-neighbor、自动/交互 QC；10/10 自动审计通过。已统一修复 SimpleITK 在 Windows 中文项目绝对路径下的 I/O 兼容问题；真实多层 DICOM series 与 10 例人工 QC 签字仍待完成 |
| patient-level 数据划分 | ✅ 已完成（工具） | 90% | 已实现可复现 split 与 patient group 防泄漏检查；等待真实数据生成正式 split |
| 公开数据集整理 | 🟡 进行中 | 94% | CTSpine1K `MSD-T10` 10 个真实 CT+label 已落盘：`liver_0`—`liver_8` + `liver_169`，官方 split 为 9 `trainset` + 1 `test_private`；真实文件接管执行 SHA-256 校验，10 例全部标准化/QC。该子集仍是工程验证，不替代正式论文主数据集/split |
| 临床脱敏数据 | 🔴 阻塞 | 0% | 当前项目目录无临床数据；必须等待合法授权、脱敏与伦理/使用范围确认 |
| SegFormer3D 骨科适配 | 🟡 进行中 | 55% | adapter、配置、dataset、训练骨架已完成；除 64³ CPU forward 外，已在真实 `liver_0` 上完成 CT+bone-window、36³ 前景 patch、joint loss、backward、AdamW.step 工程 smoke。尚无 GPU 正式训练/checkpoint |
| 区域损失 | ✅ 已完成（代码） | 90% | Dice + CE/BCE 可运行并有 backward 测试 |
| Boundary Loss | 🟠 待真实验证 | 55% | SDF 边界损失首版已实现；需真实训练、表面指标与效率验证 |
| Topology Loss | 🟠 待真实验证 | 45% | 3D soft-clDice 候选已实现；骨折/非管状骨结构适用性必须单独验证 |
| 困难样本增强 | 🟠 待真实验证 | 68% | 已落地可配置 3D flip/小角度旋转/各向同性缩放、gamma、Gaussian noise、HU shift 与 boundary-proxy hard sampling；强度增强已兼容 z-score CT 并使用 metadata 精确回到 HU 域。金属伪影与基于真实模型误差/uncertainty 的 hard mining 仍待实验 |
| 不确定性机制 | 🟠 待真实验证 | 78% | predictive entropy、Top-percent ROI、膨胀、uncertainty→error AUROC/AUPRC、错误/正确平均熵、Top-percent error recall/ROI error rate 已实现；局部残差 refinement 已补 ROI-only 二阶段训练闭环。尚无真实 baseline checkpoint 条件下的定量收益与消融 |
| 训练/验证框架 | 🟡 进行中 | 94% | DataLoader/AdamW/AMP/gradient accumulation/sliding-window、scheduler、完整 run 追踪已接入；`train/evaluate` 默认 formal preflight。新增 `task_lock` 与一站式 `formal_readiness`，统一检查任务规格、config 指纹、GPU、split、人工 QC 与数据。当前真实工程 split 检查 `ready=false`/exit 2，正确阻止未锁定任务、engineering split、未签字 QC 和 CPU 环境误启动正式 run |
| 评价指标 | ✅ 已完成（代码） | 98% | Dice、IoU、Precision、Recall、HD95、ASSD、component count/error、false merge/break 已接入；新增 multiclass per-class/macro 安全策略与 uncertainty→error AUROC/AUPRC、Top-percent error recall 等可信度指标。真实模型指标仍待 checkpoint |
| Web 科研辅助分析原型 | 🟡 进行中 | 84% | 首页/上传/健康检查、MPR、10 例人工 QC reviewer、C1–L6 可读标签、真值 PLY WebGL2 3D、简化/物理测量均已完成；新增 SDF surface 选择与 evaluation results-review，可读取未来 prediction/entropy MPR。当前 `/api/research/evaluations` 实测 200 且 total=0，真实 checkpoint/prediction 仍不存在，系统没有伪造结果 |
| 三维重建 | 🟡 进行中 | 82% | 已实现 physical-space Marching Cubes、PLY/JSON、vertex-clustering、SDF surface、WebGL2 与物理测量；真实 `liver_0` 0.3/0.4/0.5/0.8 mm SDF sweep 已完成，0.4 mm 保持 2→2 连通域并作为工程默认候选，0.8 mm 因 2→3 被保护机制拒绝。真实 Web 0.4 summary/PLY=200，0.8 summary=422；仍缺 prediction surface 与曲率/关键边缘保护 |
| 论文 | 🟡 进行中 | 56% | 中文技术初稿已补 formal preflight、uncertainty/ROI refinement、物理表面重建，并将 SpineMamba、2025 解剖变异 Transformer、2026 VertebraFormer 写入相关工作；38 条英文 BibTeX / 40 条矩阵已同步。Results 继续保持 TBD，禁止提前填结果 |
| 中期材料 | 🟡 进行中 | 70% | 已同步 10 例真实数据、88 项测试、task lock/formal readiness、交互 QC/3D/SDF、results-review、重采样几何误差与 40/38 文献证据；仍缺 GPU baseline/正式指标与人工 QC 签字 |
| 自动化测试/代码质量 | ✅ 已完成（当前阶段） | 100% | `pytest: 88 passed`；`ruff: All checks passed`；3 个关键 JSON 可解析；38 条 BibTeX 括号平衡、无重复 key；4 个前端 JS `node --check` 通过；7 个 PowerShell 脚本 parser 通过。覆盖 task lock/GPU/formal readiness、results-review、SDF topology guard 及既有真实数据/QC/训练/评价/Web 链 |

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
→ 71 passed

JSON / BibTeX / frontend structural checks
→ data/datasets.json OK
→ configs/label_schemas/ctspine1k_verse.json OK
→ paper/references.bib: 35 entries, brace balanced, duplicate key none
→ app.js / qc_review.js / research_3d.js: node --check OK

PowerShell parser
→ env/setup_env.ps1 OK
→ env/fetch_segformer3d.ps1 OK
→ env/download_verse.ps1 OK
→ env/download_ctspine1k_sample.ps1 OK
→ web/run_web.ps1 OK
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
| `docs/08_literature_matrix.md` | 40 条 3D 分割/脊柱/损失/uncertainty/重建结构化文献矩阵 | ✅ 已更新 2025–2026 直接脊柱工作 |
| `docs/03_data_pipeline_spec.md` | DICOM/HU/spacing/bone-window/QC SOP | ✅ 首版 |
| `docs/04_experiment_plan.md` | baseline、联合损失、困难样本、uncertainty 消融矩阵 | ✅ 首版 |
| `docs/05_midterm_materials.md` | 中期研究材料、已有证据、缺项、展示建议 | ✅ 首版 |
| `docs/06_public_dataset_onboarding.md` | VerSe/CTSpine1K/TotalSegmentator 登记、下载、10 例 QC 与 baseline 接入 SOP | ✅ 已更新真实状态 |
| `docs/07_real_data_validation_20260816.md` | CTSpine1K 10 例真实落盘、pipeline 0.3.0、审计、patch smoke、真实 mesh 证据 | ✅ |
| `docs/09_public_repository_manifest.md` | 公开 GitHub 仓库纳入/排除文件、医学数据隐私与提交前检查清单 | ✅ 新增 |
| `paper/outline.md` | 论文持续写作框架 | ✅ |
| `paper/manuscript_zh_v0.1.md` | 中文论文技术初稿，Methods/Experiment Design 持续补强，Results 保持 TBD | 🟡 |
| `paper/references.bib` | 38 条英文核心机器可用 BibTeX，已做 key/括号结构检查 | ✅ |
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
| `src/reconstruction/mesh.py` | mask→物理空间 Marching Cubes + vertex-clustering 简化与误差摘要 | ✅ real-label pass |
| `src/reconstruction/export_mesh.py` | NIfTI label/prediction→全分辨率/简化 PLY+JSON 可追溯导出 | ✅ real-label pass |
| `src/reconstruction/resampling_error.py` | 原始 label vs 1 mm label physical-surface 重采样几何误差 | ✅ 10/10 real pass |
| `src/reconstruction/sdf_surface.py` | physical-mm signed-distance smoothing + zero-level MC + 连通域保护 | ✅ real `liver_0` sweep/Web pass |
| `src/reconstruction/measurement.py` | 物理坐标距离/三点夹角与 voxel→physical 工具 | ✅ |
| `web/backend/app.py` | FastAPI 本地科研服务：上传/MPR/QC reviewer/真值3D+SDF/results-review/测量/推理占位 | 🟡 |
| `web/frontend/` | 上传/QC、交互 MPR+overlay、WebGL2 真值 3D/SDF、results-review、测量前端 | 🟡 |
| `web/run_web.ps1` | localhost Web 启动脚本 | ✅ |
| `tests/` | 自动化测试 | ✅ 88 passed |
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

### P0｜下一轮必须先完成：真实数据与 baseline

- [ ] 组内确认首个主任务/标签集，工程默认先按“脊柱/椎体 CT”推进；确认后填写 task spec、设 `task_locked=true`，用 `task_lock` 编译正式 config；
- [x] 工程 baseline 已暂定并登记 VerSe complete；`data/datasets.json` 已记录来源、规模、标签、许可核验备注（最终论文主数据集仍需组内确认）；
- [x] 已解决首批真实数据落盘：CTSpine1K `MSD-T10` 10 例通过浏览器顺序下载成功；VerSe S3 仍待后续网络条件解决；
- [x] 已完成 CTSpine1K 小样本下载计划、批量配对/标准化、官方 split 标记解析与统一 QC 工具；
- [x] 已实际落盘 `liver_0`—`liver_8` + `liver_169` CT+label，并在 `data/datasets.json` 登记本地状态、日期、路径、pipeline 与 split 统计；
- [x] CTSpine1K 10 例已通过 `prepare_ctspine1k` 批量配对/标准化，manifest 和 batch QC 可追踪；VerSe 路径待数据可达后再执行；
- [x] 已对 10 例真实 CT 跑 pipeline 0.3.0 标准化、三视图/bone-window/label-overlay contact sheet 与自动审计；
- [x] 自动检查 orientation/spacing/label geometry/label values/normalization metadata：10/10 pass；
- [ ] 项目成员逐例填写 `manual_qc_review.csv` 并完成人工审核签字；
- [ ] 固定 patient-level train/validation/test split；
- [x] 已在真实 `liver_0` 上跑通 CT+bone-window + SegFormer3D + joint loss + backward + AdamW.step 单 patch 工程 smoke；
- [ ] 确认可用 NVIDIA GPU/服务器；在目标机器运行 `env/check_formal_readiness.ps1`，必须 `ready=true` 后才启动正式训练；
- [ ] 跑通正式 SegFormer3D baseline；
- [ ] 生成第一份真实 `metrics_per_case.csv`；
- [ ] 记录 DSC、HD95、ASSD 和单病例推理时间；
- [ ] 将真实 baseline mask 接入 Web overlay。

### P1｜联合损失与困难样本消融

- [ ] Region；
- [ ] Region + Boundary；
- [ ] Region + Topology；
- [ ] Region + Boundary + Topology；
- [ ] loss 权重 validation grid；
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
- [ ] 曲率/关键边缘保护；
- [x] vertex-clustering mesh 简化；真实 `liver_0` 1.5 mm 档约减 60% 顶点/面，保留全分辨率基准；
- [x] MPR 三视图（axial/coronal/sagittal + 切片位置/窗宽窗位）；
- [x] WebGL2 真值 label 3D 渲染与全分辨率/1.5/2.0 mm 选择；
- [x] 椎体类别显示：`1–25 → C1–L6` 工程 schema，不改原标签值；
- [x] 物理 XYZ 距离/三点夹角计算 API；
- [x] 10 例人工 QC reviewer + 交互 MPR/真值 overlay 接口；
- [x] evaluation results-review 页面与 prediction/entropy MPR 接口已准备；当前真实 evaluation=0；
- [ ] 取得真实 checkpoint 后生成 prediction / uncertainty / prediction mesh 并在 Web 展示正式结果。

### P4｜论文/中期/软著

- [x] 已建立 40 条结构化文献矩阵；已补 SpineMamba、2025 解剖变异 Transformer、2026 VertebraFormer，后续继续补困难病例/强 baseline；
- [x] 已建立 38 条英文核心 `paper/references.bib`，并纠正多条易错题录；
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
71 passed
All checks passed!
```

附加结构检查：`data/datasets.json`、`configs/label_schemas/ctspine1k_verse.json` 可解析；`paper/references.bib` 当前 35 entries、括号平衡且无重复 key；前端 `app.js / qc_review.js / research_3d.js` 均通过 `node --check`。

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
- GitHub 浏览器创建页已打开，但当前浏览器处于 GitHub **未登录**状态，因此远程公开仓库创建和 push 仍需账户本人先完成登录。计划仓库名：`orthopedic-ct-segformer3d`。

**当前 GitHub 发布状态：**本地公开安全版本已经整理完成并可提交；远程仓库尚未创建，唯一外部阻塞是 GitHub 账号登录。登录后应创建 public repository `orthopedic-ct-segformer3d`，添加 `origin`，推送 `main`，再把最终仓库 URL 回写本台账。

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
