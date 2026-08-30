# 基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究

本目录用于武汉理工大学大学生创新创业训练计划项目的工程实现、实验记录、论文材料和科研型 Web 辅助分析原型开发。

## 先读

继续项目之前必须先阅读：

1. [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) —— 唯一主进度台账；
2. [`TASKS.md`](./TASKS.md) —— 已完成/待完成/阻塞任务和推荐分工；
3. [`CONTRIBUTING.md`](./CONTRIBUTING.md) —— 分支、PR、测试和正式实验协作规范；
4. [`docs/01_overall_design.md`](./docs/01_overall_design.md) —— 总体设计；
5. [`docs/03_data_pipeline_spec.md`](./docs/03_data_pipeline_spec.md) —— CT/DICOM 处理规范；
6. [`docs/04_experiment_plan.md`](./docs/04_experiment_plan.md) —— 模型实验与消融计划；
7. [`docs/06_public_dataset_onboarding.md`](./docs/06_public_dataset_onboarding.md) —— 公开数据接入与首轮 baseline 操作手册；
8. [`docs/09_public_repository_manifest.md`](./docs/09_public_repository_manifest.md) —— 公开 GitHub 仓库允许/禁止提交内容；
9. [`docs/10_final_parameter_lock.md`](./docs/10_final_parameter_lock.md) —— 最终 v13 参数锁定记录；
10. [`docs/11_final_independent_test.md`](./docs/11_final_independent_test.md) —— 唯一一次正式独立测试记录；
11. [`docs/12_final_presentation_outline.md`](./docs/12_final_presentation_outline.md) —— v0.3.0 中期/结题展示提纲；
12. [`paper/outline.md`](./paper/outline.md) —— 论文持续写作框架。

**所有实质性修改完成后必须同步更新 `PROJECT_STATUS.md`。**

## 项目定位

系统面向骨科 CT 科研与辅助分析，计划覆盖：

- DICOM 解析与质量控制；
- HU 恢复、灰度标准化、骨窗增强、空间重采样；
- SegFormer3D 骨骼分割；
- 区域 + 边界 + 拓扑联合损失；
- 困难样本增强；
- 不确定性驱动精修；
- 高保真三维重建；
- Web 端 CT/MPR/分割/三维可视化与测量；
- 论文、软著、中期与结项材料。

> 本系统当前是科研原型，不是医疗器械，不应被用于独立临床诊断或替代医生判断。

## 目录结构

```text
D:\国创项目
├─ PROJECT_STATUS.md            # 主进度台账（每次修改必须更新）
├─ README.md
├─ TASKS.md                     # 多人协作总任务看板
├─ CONTRIBUTING.md              # Git/PR/实验协作规则
├─ SECURITY.md                  # 医学数据隐私与结果边界
├─ .github/                     # Issue / Pull Request 模板
├─ docs/
│  ├─ 01_overall_design.md      # 总体架构与模块设计
│  ├─ 02_literature_survey.md   # 文献与公开数据集调研
│  ├─ 03_data_pipeline_spec.md  # DICOM/CT 预处理 SOP
│  ├─ 04_experiment_plan.md     # 训练、调参、消融、指标规范
│  ├─ 05_midterm_materials.md   # 中期研究材料持续汇总
│  ├─ 06_public_dataset_onboarding.md # 公开数据接入与 baseline 手册
│  ├─ 07_real_data_validation_20260816.md # CTSpine1K 10 例真实工程验证记录
│  ├─ 08_literature_matrix.md   # 44 条结构化文献矩阵
│  ├─ 09_public_repository_manifest.md # GitHub 公开内容/排除清单
│  ├─ 10_final_parameter_lock.md # 最终 v13 参数锁定记录
│  ├─ 11_final_independent_test.md # 唯一一次正式独立测试记录
│  └─ 12_final_presentation_outline.md # 中期/结题展示源材料
├─ configs/                     # baseline / joint-loss 实验配置与标签 schema
├─ env/
│  ├─ requirements.txt
│  ├─ setup_env.ps1             # 只在本项目内创建 .python/.venv
│  ├─ fetch_segformer3d.ps1     # 获取官方 SegFormer3D 到 third_party
│  ├─ download_verse.ps1        # VerSe 下载计划/显式下载辅助脚本
│  ├─ download_ctspine1k_sample.ps1 # CTSpine1K 小样本 CT+label 下载；原子写入并校验 gzip 完整性
│  ├─ check_gpu.ps1             # GPU/CUDA 只读验收
│  └─ check_formal_readiness.ps1 # 任务+GPU+数据/QC 一站式正式实验验收
├─ src/
│  ├─ preprocessing/            # DICOM/NIfTI、公开数据批处理、三视图 QC、处理后审计
│  ├─ modeling/                 # SegFormer3D、task lock/formal readiness、preflight、loss、增强、不确定性精修、训练/评估
│  ├─ reconstruction/           # physical-space mesh、SDF 表面、简化、重采样几何误差、测量
│  └─ label_schema.py           # CTSpine1K/VerSe 椎体标签可读映射（不锁定正式任务）
├─ web/
│  ├─ backend/
│  └─ frontend/
├─ paper/
│  ├─ outline.md
│  ├─ references.bib            # 44 条机器可用 BibTeX（42 英文核心 + 2 已核验中文）
│  └─ manuscript_zh_v0.1.md     # 中文论文技术稿，已同步 validation/independent Results
├─ third_party/
│  └─ SegFormer3D/              # 官方上游仓库，保留 Git/许可证/本地兼容补丁
├─ tests/
└─ data/
   ├─ README.md                 # 数据治理说明，默认不提交真实影像
   └─ datasets.json             # 公开数据集版本/许可/来源登记
```

## 环境原则

上游 SegFormer3D 官方示例使用 Python 3.11.7、PyTorch 2.1.0、CUDA 11.8。当前机器系统侧没有检测到 `nvidia-smi`，本项目使用 `uv` 在目录内安装 **Python 3.11.7 (`.python`) + 独立虚拟环境 (`.venv`)**，当前为 **PyTorch 2.1.0 CPU 版**。当前 10 例 formal-pipeline pilot 的训练、validation 与唯一一次 independent test 已在显式 `--allow-cpu` 条件下完成；NVIDIA GPU 仅作为后续扩大数据规模和缩短训练时间的加速条件。

重新搭建或修复环境时运行：

```powershell
cd D:\国创项目
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
```

该脚本仅管理 `D:\国创项目\.python` 与 `D:\国创项目\.venv`；若系统没有 Python 3.11 且存在 `uv`，会把 Python 3.11.7 下载到项目目录，不注册系统 Python，也不删除或修改其他 Python/Conda 环境。

## 当前工程阶段

> **状态：研究型工程闭环已完成，当前进入扩大数据与提升模型性能阶段。**

| 项目 | 当前状态 |
|---|---|
| 数据与流程 | ✅ CTSpine1K 10 例真实 CT 已完成标准化、QC 与 7/2/1 patient-level split |
| 最终模型 | ✅ 锁定 v13：SegFormer3D + CT-only + Region/Boundary + Bernoulli sampling |
| 正式独立测试 | ✅ 已按锁参协议完成一次 `liver_169` FINAL FORMAL INDEPENDENT TEST |
| 不确定性与精修 | ✅ uncertainty/calibration 已完成；ROI refinement 综合判定 **FAIL**，正式流程禁用 |
| 三维与 Web | ✅ prediction / uncertainty / mesh / SDF / MPR / WebGL2 科研复核链已打通 |
| 工程质量 | ✅ `138 passed`，Ruff / JS syntax / `git diff --check` 全部通过 |

当前项目已经完成从 **CT 数据处理 → SegFormer3D 分割 → 不确定性分析 → 三维重建 → Web 科研复核 → 正式独立测试** 的完整可追溯流程。

> 当前模型绝对分割精度仍较低，因此项目定位是 **科研工程原型与方法验证平台**，不是临床医疗器械。详细实验过程、v11～v23 消融和历史记录请查看 [`PROJECT_STATUS.md`](./PROJECT_STATUS.md)。

## 当前最终实验结果（formal pipeline pilot）

当前公开 pilot 共 10 例 CTSpine1K `MSD-T10`：7 train / 2 validation / 1 官方 `test_private` independent test。所有模型/阈值/后处理选择都在 `liver_7/liver_8` validation 完成；最终参数通过 `docs/10_final_parameter_lock.md` 锁定并提交为 `eb0a824`，确认远端一致后才首次按最终锁定协议执行 `liver_169` 的 FINAL FORMAL INDEPENDENT TEST。仓库中更早的 5-epoch pilot test 仅保留为历史工程链证据，不属于本次最终锁定测试，也未用于 v13 参数选择。最终锁定协议下的正式 independent test 只运行一次，测试后没有继续调参。

| 阶段 | 病例 | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | FG ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | `liver_7/liver_8` mean | 0.05471 | 0.02815 | 0.03423 | 0.13649 | 185.9498 | 51.4865 | 4.0270× |
| Independent Test | `liver_169` | **0.02878** | 0.01460 | 0.02090 | 0.04622 | 136.8722 | 43.9720 | 2.2118× |

正式 independent test 的 prediction/GT components=`236/1`、component error=`235`、false break=`29`，说明当前模型仍存在严重 fragmentation。Uncertainty error AUROC/AUPRC=`0.86424/0.29665`、Top-10% error recall=`0.54993`；ECE=`0.02740`，但低 Dice 与高 background 占比意味着不能把该校准数字解释为临床可靠性。

最终 pipeline 保持 **v13 coarse**：CT-only、Region+Boundary=`1.0/0.1`、Bernoulli foreground probability=`0.25`、patches/case=`4`、flip-only、64³ training ROI、softmax/argmax；uncertainty ROI refinement 在 validation 中综合判定 **REFINEMENT=FAIL**，因此正式推理关闭 refinement。

独立 prediction 3D 工程链也已完成：原始 mesh=`365,247` 顶点 / `724,694` 面；2.0 mm + feature strength=8 后=`81,353` 顶点 / `160,384` 面，顶点减少约 `77.73%`，简化工程 ASSD/HD95≈`0.56490/1.07159 mm`；0.4 mm SDF 保持 components `236→236`，SDF-vs-original 工程 ASSD/HD95≈`0.02536/0.06367 mm`。上述 vertex-nearest 指标仅用于三维工程质量，不替代 segmentation HD95/ASSD。

Web `results-review` 已识别 validation 与 independent evaluation，可显示真实 prediction / uncertainty MPR；`research-3d` 已在 Edge WebGL2 实机加载 validation 与 independent prediction 的 2.0 mm mesh 和 0.4 mm SDF surface。完整正式测试记录见 [`docs/11_final_independent_test.md`](./docs/11_final_independent_test.md)。

### 科研限制

当前结果来自极小样本 formal-pipeline pilot，绝对分割精度较低，不能描述为高精度临床模型，也没有合法授权的临床脱敏数据或前瞻性验证。项目当前更适合作为**可追溯科研工程闭环、方法探索、uncertainty/QC、physical-space 3D 与 Web prototype**。后续若继续科研，应扩大病例规模并重新预注册 train/validation/test，加入真实强 baseline 与多中心/临床验证；不得利用本次 `liver_169` 再做参数选择。

## 上游项目

主要参考：

- OSUPCVLab/SegFormer3D
- SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation (2024)
- SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers (2021)

SegFormer3D 上游仓库使用 GPL-3.0 许可证。若后续直接复用或修改其代码，必须保留来源和许可证要求，并在软件成果材料中清楚区分“上游开源模块”和“本项目自研模块”。

## 代码与数据基本规范

- 患者数据按 patient-level 划分 train/val/test，禁止 slice 泄漏；
- 临床数据必须已脱敏且有授权；
- 原始 DICOM、NIfTI、模型权重、大型缓存默认不提交 Git；
- 每个实验必须有固定随机种子、配置文件、日志、checkpoint 与结果摘要；
- 论文结果必须能回溯到实验配置与输出；
- Web 日志不得记录患者姓名、身份证号、手机号等敏感信息。

## 多人协作

公开 GitHub 仓库：`https://github.com/927242768-dotcom/orthopedic-ct-segformer3d`

建议所有协作者从 `TASKS.md` 领取任务，在独立 feature/docs 分支开发，通过 Pull Request 合并到 `main`。代码修改至少执行 `pytest tests -q` 与 `ruff check src web tests`；正式论文实验必须先通过 task lock、人工 QC、正式 patient-level split 与 `formal_readiness ready=true`。当前 CPU 路径可在显式 `--allow-cpu` 下通过 formal gate，GPU/CUDA 仅作为扩大实验规模时的加速条件。

公开仓库只保存**代码、配置、测试、文档、数据来源/匿名 split 和可复现脚本**。真实医学影像、处理后体数据、checkpoint、runtime、虚拟环境、第三方 checkout 和大型生成 PPT 默认不进入 Git，完整规则见 [`docs/09_public_repository_manifest.md`](./docs/09_public_repository_manifest.md)。

## 下一步

快速任务看板见 [`TASKS.md`](./TASKS.md)；更详细的证据、历史与严格优先级请以 [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) 为准。
