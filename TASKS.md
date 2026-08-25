# 项目任务总清单

> 项目：基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究
>
> 用途：供多人协作时快速查看“已完成 / 待完成 / 阻塞 / 可分工”任务。
>
> 更详细的技术证据、测试结果和历史记录请以 `PROJECT_STATUS.md` 为准。

---

## 0. 当前里程碑概览

| 模块 | 状态 | 说明 |
|---|---|---|
| 总体方案 | ✅ 已完成 | 数据→分割→不确定性→三维→Web 技术路线已建立 |
| 文献调研 | ✅ 主体完成 | 40 条结构化矩阵，38 条英文核心 BibTeX |
| CPU 开发环境 | ✅ 已完成 | Python 3.11.7 + `.venv`，PyTorch 2.1.0 CPU |
| GPU 正式环境 | 🔴 待完成 | 当前无 CUDA / 无可见 NVIDIA GPU |
| 公开数据接入 | ✅ 已完成工程子集 | CTSpine1K MSD-T10 真实 10 例 CT+label 已处理 |
| CT 标准化流程 | ✅ 基本完成 | 1 mm 重采样、HU、z-score、骨窗、QC、label NN 重采样 |
| 自动 QC | ✅ 已完成 | 10/10 自动审计通过 |
| 人工 QC | 🔴 待完成 | `manual_qc_review.csv` 需真人逐例签字 |
| SegFormer3D 工程适配 | ✅ 已完成工程链 | 真实 patch forward/backward/optimizer step 已通过 |
| 联合损失代码 | ✅ 已完成代码 | Region + Boundary + soft-clDice |
| 困难样本策略代码 | ✅ 已完成首版 | 3D 几何/强度增强 + boundary hard sampling |
| 不确定性与精修代码 | ✅ 已完成工程链 | entropy、ROI、AUROC/AUPRC、ROI-only refinement |
| 正式任务定义 | 🔴 待锁定 | binary / multiclass semantic 尚未最终确认 |
| 正式 patient-level split | 🔴 待完成 | 当前仅 engineering smoke split |
| 正式 baseline | 🔴 未开始 | 等 GPU + task + QC + formal split |
| 正式模型指标 | 🔴 未产生 | 当前没有可写入论文 Results 的 Dice/HD95/ASSD |
| 三维重建工程链 | ✅ 基本完成 | physical MC、SDF、简化、WebGL2、物理测量 |
| Web 科研原型 | ✅ 主体完成 | MPR/QC/3D/results-review 已具备 |
| 临床脱敏数据 | 🔴 外部阻塞 | 等授权/脱敏/伦理 |
| 论文 Methods | ✅ 主体完成 | Results 仍保持 TBD |
| 自动化测试 | ✅ 当前通过 | 88 passed + Ruff clean |

---

## 1. 已完成任务

### A. 项目与工程基础

- [x] 项目总体架构设计
- [x] 工程目录结构
- [x] `PROJECT_STATUS.md` 主进度台账
- [x] `README.md` 项目入口
- [x] 数据 SOP、实验计划、中期材料、论文初稿
- [x] SegFormer3D 上游仓库接入方案与 GPL-3.0 边界说明
- [x] Python 3.11.7 项目级隔离环境
- [x] 依赖安装与 CPU 可运行性验证

### B. 文献调研

- [x] 3D U-Net / V-Net / nnU-Net
- [x] SegFormer / SegFormer3D
- [x] UNETR / nnFormer / Swin UNETR / MedNeXt
- [x] CTSpine1K / VerSe / TotalSegmentator
- [x] VerFormer 2024
- [x] SpineMamba 2025
- [x] 2025 椎体 Transformer + 解剖变异工作
- [x] VertebraFormer 2026
- [x] Boundary / Hausdorff / topology 相关文献
- [x] uncertainty / calibration 相关文献
- [x] Marching Cubes / mesh / SDF 相关文献
- [x] 40 条结构化文献矩阵
- [x] 38 条英文核心 BibTeX

### C. 数据接入与预处理

- [x] CTSpine1K / VerSe / TotalSegmentator 数据源调研
- [x] CTSpine1K MSD-T10 10 个真实 CT+label 工程子集落盘
- [x] 数据来源与 split 信息登记
- [x] SHA-256 provenance 校验机制
- [x] DICOM series 发现
- [x] DICOM 几何切片排序
- [x] orientation / spacing / origin / direction 检查
- [x] HU clip
- [x] case-wise z-score
- [x] bone-window 通道
- [x] 1 mm 空间重采样
- [x] label nearest-neighbor 重采样
- [x] image/label geometry 一致性检查
- [x] label value 完整性检查
- [x] 三视图 QC contact sheet
- [x] 自动处理后审计
- [x] 10/10 真实病例自动审计通过
- [x] Windows 中文路径 SimpleITK 兼容层

### D. 模型工程

- [x] SegFormer3D adapter
- [x] 3D CT Dataset
- [x] 单通道 CT 输入
- [x] CT + bone-window 双通道输入
- [x] 3D patch crop
- [x] foreground-biased sampling
- [x] MONAI sliding-window inference
- [x] AdamW
- [x] AMP 接口
- [x] gradient accumulation
- [x] warmup + cosine restart scheduler
- [x] checkpoint / config / split / log / history 追踪
- [x] 真实 `liver_0` 36³ patch forward/backward/AdamW.step 工程 smoke

### E. Loss / 困难样本 / 不确定性

- [x] Dice + CE/BCE Region Loss
- [x] Boundary Loss
- [x] signed distance field
- [x] 3D soft-clDice
- [x] `JointOrthopedicSegLoss`
- [x] flip / rotation / scale
- [x] gamma / Gaussian noise / HU shift
- [x] boundary-proxy hard patch sampling
- [x] predictive entropy
- [x] uncertainty ROI / Top-percent ROI / dilation
- [x] uncertainty→error AUROC/AUPRC
- [x] error/correct mean entropy
- [x] Top-percent error recall
- [x] ROI error rate / ROI fraction
- [x] `UncertaintyRefinementNet3D`
- [x] ROI-only residual refinement
- [x] ROI-normalized refinement loss

### F. 评价与正式实验保护

- [x] Dice / IoU / Precision / Recall
- [x] HD95 / ASSD
- [x] connected component count
- [x] false merge / false break
- [x] multiclass per-class metrics
- [x] 防空类别虚高 macro 策略
- [x] `metrics_per_case.csv` 输出框架
- [x] `metrics_per_class.csv` 输出框架
- [x] formal / engineering preflight
- [x] test_private 泄漏阻止
- [x] task lock 编译器
- [x] GPU/CUDA 只读检查器
- [x] formal readiness 汇总检查器
- [x] 当前环境能正确返回 `ready=false` 并列出 blocker

### G. 三维重建

- [x] physical-space Marching Cubes
- [x] PLY + JSON summary
- [x] spacing/origin/direction → physical XYZ
- [x] vertex-clustering 简化
- [x] 1.5 mm / 2.0 mm Web 简化档
- [x] distance / angle 物理测量
- [x] 10 例 raw label→1 mm label 表面几何误差评估
- [x] physical-mm SDF surface
- [x] SDF smoothing 参数 sweep
- [x] connected-component topology guard
- [x] 0.4 mm SDF Web 可加载
- [x] 0.8 mm 拓扑改变时 Web 主动拒绝

### H. Web 科研原型

- [x] FastAPI 后端
- [x] 首页 / health API
- [x] DICOM/NIfTI 上传
- [x] axial/coronal/sagittal MPR
- [x] window/level
- [x] label overlay
- [x] 10 例人工 QC reviewer
- [x] C1–L6 可读标签 schema
- [x] WebGL2 3D viewer
- [x] full / simplified / SDF mesh
- [x] distance / angle API
- [x] results-review 页面
- [x] future prediction / entropy overlay 接口
- [x] 当前无真实 evaluation 时正确显示 total=0

### I. 文档与质量

- [x] 论文 Introduction / Related Work / Methods 主体
- [x] 实验设计
- [x] 中期材料
- [x] 组会汇报源材料
- [x] 88 个 pytest 全部通过
- [x] Ruff clean
- [x] 4 个前端 JS 语法检查通过
- [x] 38 条 BibTeX 结构检查通过
- [x] PowerShell 脚本语法检查通过

---

## 2. 最高优先级待完成任务（P0）

### P0-1 人工 QC

建议负责人：数据/QC 负责人

- [ ] 打开 `/qc-review`
- [ ] 逐例检查 10 个病例的 orientation
- [ ] 检查 spacing
- [ ] 检查 label alignment
- [ ] 检查 bone-window
- [ ] 填 reviewer
- [ ] 填 notes（如有）
- [ ] 将合格病例设为 pass
- [ ] 完成 `manual_qc_review.csv`

> 这一步必须由真人完成，不允许 AI 或脚本自动代签。

### P0-2 正式任务锁定

建议负责人：模型负责人 + 项目负责人共同确认

- [ ] 最终选择 `binary_semantic` 或 `multiclass_semantic`
- [ ] 明确 task_id
- [ ] 明确 foreground labels
- [ ] 明确 num_classes
- [ ] 明确主数据集
- [ ] 更新 `configs/task_specs/vertebra_task_template.json`
- [ ] 将 `task_locked=true`
- [ ] 通过 `src.modeling.task_lock` 编译正式 config

> 当前训练链不支持真正的 instance segmentation，不能用 semantic segmentation 冒充。

### P0-3 正式 patient-level split

建议负责人：数据负责人

- [ ] 确定正式主数据集和病例数量
- [ ] 确定 train / validation / test
- [ ] patient-level 防泄漏
- [ ] 保持官方 test_private 不参与训练/调参
- [ ] 将正式 split 标记 `formal_experiment=true`
- [ ] 固定唯一 split JSON

### P0-4 GPU 正式环境

建议负责人：训练/算力负责人

- [ ] 找到 NVIDIA GPU 机器/服务器
- [ ] 确认驱动
- [ ] 安装匹配的 CUDA PyTorch
- [ ] 确认显存
- [ ] 运行 `env/check_gpu.ps1`
- [ ] 运行 `env/check_formal_readiness.ps1`
- [ ] 最终达到 `ready=true`

---

## 3. 正式实验任务（P1）

### P1-1 CT-only baseline

- [ ] 正式 SegFormer3D CT-only 训练
- [ ] 保存最佳 checkpoint
- [ ] 记录 config / split / seed / environment
- [ ] validation 选择模型
- [ ] 独立 test
- [ ] 输出 `metrics_per_case.csv`
- [ ] multiclass 时输出 `metrics_per_class.csv`
- [ ] 报告 Dice / HD95 / ASSD / IoU / Precision / Recall
- [ ] 记录单病例推理时间

### P1-2 输入消融

- [ ] CT only
- [ ] CT + bone-window
- [ ] 比较区域指标
- [ ] 比较表面指标
- [ ] 比较速度/显存

### P1-3 联合损失消融

- [ ] Region
- [ ] Region + Boundary
- [ ] Region + Topology
- [ ] Region + Boundary + Topology
- [ ] loss 权重 validation grid
- [ ] Boundary 对 HD95/ASSD 的影响
- [ ] Topology 对 false merge / false break 的影响
- [ ] 骨折/真实断裂病例单独检查

### P1-4 困难样本实验

- [ ] standard augmentation baseline
- [ ] + boundary hard sampling
- [ ] + intensity/HU augmentation
- [ ] high-loss hard mining
- [ ] high-uncertainty mining
- [ ] thick-slice subset
- [ ] metal artifact subset（数据存在时）
- [ ] fracture subset（数据存在时）
- [ ] low-density subset（数据存在时）

---

## 4. 不确定性与精修任务（P2）

- [ ] 用真实 baseline checkpoint 生成 entropy
- [ ] uncertainty→error AUROC/AUPRC
- [ ] Top-X% error recall
- [ ] ROI threshold / percentile validation
- [ ] calibration 分析
- [ ] coarse baseline
- [ ] full-volume second pass 对照
- [ ] uncertainty ROI refinement
- [ ] 比较 Dice / HD95 / ASSD
- [ ] 比较 ROI error
- [ ] 比较推理时间
- [ ] 比较显存
- [ ] 评估 uncertainty 是否能作为 QC 信号

---

## 5. 三维与 Web 正式结果任务（P3）

- [ ] prediction mask → physical mesh
- [ ] prediction mesh vs GT surface
- [ ] prediction surface HD95 / ASSD
- [ ] SDF prediction surface 验证
- [ ] 简化误差评估
- [ ] 曲率/关键边缘保护候选
- [ ] Web 接正式 inference
- [ ] prediction overlay
- [ ] entropy overlay
- [ ] results-review 载入真实 evaluation
- [ ] prediction 3D mesh
- [ ] GT/prediction 对比
- [ ] uncertainty QC 提示
- [ ] 生成中期/结题/论文截图

---

## 6. 临床/外部验证任务（P4，外部条件）

- [ ] 获得合法授权的临床脱敏 CT
- [ ] 确认伦理/数据使用范围
- [ ] 临床数据标准化
- [ ] 临床人工 QC
- [ ] 外部测试
- [ ] domain shift / domain generalization 分析
- [ ] 若无法取得临床数据，与指导老师确认公开多中心数据替代方案

---

## 7. 论文/结题任务（P5）

- [ ] 正式主结果表
- [ ] baseline comparison
- [ ] 输入消融表
- [ ] loss 消融表
- [ ] uncertainty/refinement 表
- [ ] difficult subset 表
- [ ] 3D surface 表
- [ ] failure cases 图
- [ ] prediction overlay 图
- [ ] 3D mesh 图
- [ ] Discussion
- [ ] Conclusion
- [ ] 中文 CNKI/万方题录最终核验
- [ ] 全文引用/格式统一
- [ ] 最终语言润色
- [ ] 中期/结题 PPT 更新
- [ ] 软著材料（如需要）

---

## 8. 推荐多人分工

| 角色 | 建议负责内容 |
|---|---|
| 项目负责人 | task lock、实验总表、里程碑、论文整合 |
| 数据/QC | 数据下载、manifest、人工 QC、正式 split |
| 模型训练 | GPU 环境、baseline、输入/损失消融 |
| 不确定性 | entropy、calibration、refinement、困难样本 |
| 三维/Web | prediction mesh、SDF、viewer、results-review |
| 文献/论文 | 最新文献、图表、Results/Discussion、格式核验 |

人数较少时可以一人兼任多个角色，但每个正式实验必须保留唯一配置、split、seed、checkpoint 和结果摘要。

---

## 9. 完成定义（Definition of Done）

一项任务只有同时满足以下条件才允许勾选“完成”：

1. 代码/文档真实存在；
2. 有对应测试或真实工程验证；
3. 不把 smoke/random-weight/GT mesh 写成模型结果；
4. 正式实验必须能追溯 config + split + seed + checkpoint + metrics；
5. 涉及医学数据时遵守隐私、授权、伦理要求；
6. 实质修改完成后更新 `PROJECT_STATUS.md`。
