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
9. [`paper/outline.md`](./paper/outline.md) —— 论文持续写作框架。

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
│  ├─ 08_literature_matrix.md   # 40 条结构化文献矩阵
│  └─ 09_public_repository_manifest.md # GitHub 公开内容/排除清单
├─ configs/                     # baseline / joint-loss 实验配置与标签 schema
├─ env/
│  ├─ requirements.txt
│  ├─ setup_env.ps1             # 只在本项目内创建 .python/.venv
│  ├─ fetch_segformer3d.ps1     # 获取官方 SegFormer3D 到 third_party
│  ├─ download_verse.ps1        # VerSe 下载计划/显式下载辅助脚本
│  ├─ download_ctspine1k_sample.ps1 # CTSpine1K 小样本 CT+label 下载计划/显式下载
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
│  ├─ references.bib            # 38 条已核验英文核心 BibTeX
│  └─ manuscript_zh_v0.1.md     # 中文论文技术初稿，结果区保持 TBD
├─ third_party/
│  └─ SegFormer3D/              # 官方上游仓库，保留 Git/许可证/本地兼容补丁
├─ tests/
└─ data/
   ├─ README.md                 # 数据治理说明，默认不提交真实影像
   └─ datasets.json             # 公开数据集版本/许可/来源登记
```

## 环境原则

上游 SegFormer3D 官方示例使用 Python 3.11.7、PyTorch 2.1.0、CUDA 11.8。当前机器系统侧可见 Python 3.13，但没有检测到 Conda，也没有检测到 `nvidia-smi`。本项目已使用 `uv` 在目录内安装 **Python 3.11.7 (`.python`) + 独立虚拟环境 (`.venv`)**，当前环境已通过依赖导入与单元测试；现阶段安装的是 **PyTorch 2.1.0 CPU 版**，GPU 训练环境仍需在具有 NVIDIA GPU/驱动的设备或服务器上确认。

重新搭建或修复环境时运行：

```powershell
cd D:\国创项目
powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1
```

该脚本仅管理 `D:\国创项目\.python` 与 `D:\国创项目\.venv`；若系统没有 Python 3.11 且存在 `uv`，会把 Python 3.11.7 下载到项目目录，不注册系统 Python，也不删除或修改其他 Python/Conda 环境。

## 当前工程阶段

当前处于：

**40 条结构化文献矩阵 + 38 条英文核心 BibTeX 已建立 → 项目内 CPU 开发环境可运行 → DICOM + NIfTI 数据入口/联合损失/训练评估/formal preflight 已实现 → task lock + GPU checker + formal readiness 可阻止未锁定任务或不合格环境误启动正式实验 → CTSpine1K 10 例真实 CT+label 已按 pipeline 0.3.0 完成 1 mm 标准化 → 10/10 自动数据审计通过 → 真实双通道联合损失单 patch forward/backward 已通过 → Web 已支持人工 QC、交互式 MPR+label overlay、真值 3D/SDF 表面、物理距离/角度以及未来正式 evaluate 输出的 results-review 页面 → 真实各向异性重采样与 SDF/mesh 几何误差已有工程证据 → 当前仍等待人工 QC 签字、正式任务/split 确认与 NVIDIA GPU baseline 训练**

模型训练指标尚未产生。任何 DSC、HD95、ASSD 等结果都必须由真实实验生成并进入实验记录后才能写入论文。

2026-08-16 本机对 VerSe 官方 S3 仍存在连接超时；CTSpine1K Hugging Face 则在改用浏览器**单文件顺序下载**后成功取得 `MSD-T10` 的 10 个真实 CT+label（`liver_0`—`liver_8`、`liver_169`）。其中官方 split 为 9 例 `trainset`、1 例 `test_private`。原始文件位于 `data/raw_public/CTSpine1K/MSD-T10`，标准化结果位于 `data/processed_ctspine1k_real`；10 例均完成 1 mm 重采样、HU clip→case-wise z-score、骨窗、nearest-neighbor label 重采样、QC contact sheet 与自动审计。`manual_qc_review.csv` 已生成但人工字段保持空白，必须由项目成员实际复核后签字。工程 Web 使用 VerSe-compatible `1–25 → C1–L6` 标签 schema 仅做可读显示，不锁定正式 binary/multiclass/instance 任务。当前模板执行 formal readiness 会正确返回未就绪：任务未锁定、engineering split、人工 QC 未签字、本机 PyTorch 为 CPU build/无 CUDA；这属于保护机制正常工作。真实 `liver_0` 的 SDF 0.4 mm 表面在 Web 可读取，而会改变连通域的 0.8 mm 参数会被拓扑保护拒绝。Web 的 results-review 已具备读取未来 `evaluate.py` 结果的能力，但项目当前真实 evaluation 列表仍为空，因此没有可写入论文 Results 的模型指标。全套回归的准确数量以 `PROJECT_STATUS.md` 最新值为准。

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

建议所有协作者从 `TASKS.md` 领取任务，在独立 feature/docs 分支开发，通过 Pull Request 合并到 `main`。代码修改至少执行 `pytest tests -q` 与 `ruff check src web tests`；正式论文实验必须先通过 task lock、人工 QC、正式 patient-level split、GPU/CUDA 检查和 `formal_readiness ready=true`。

公开仓库只保存**代码、配置、测试、文档、数据来源/匿名 split 和可复现脚本**。真实医学影像、处理后体数据、checkpoint、runtime、虚拟环境、第三方 checkout 和大型生成 PPT 默认不进入 Git，完整规则见 [`docs/09_public_repository_manifest.md`](./docs/09_public_repository_manifest.md)。

## 下一步

快速任务看板见 [`TASKS.md`](./TASKS.md)；更详细的证据、历史与严格优先级请以 [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) 为准。
