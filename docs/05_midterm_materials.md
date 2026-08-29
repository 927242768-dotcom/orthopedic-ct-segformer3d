# 05 中期研究材料持续汇总

> 版本：v0.3｜2026-08-29
>
> 用途：用于 2026 年 11 月中期检查前持续累积真实材料。本文只记录已经完成或明确处于进行中的工作；模型性能数字必须由正式实验产生后再填写。

## 1. 项目题目

**基于 SegFormer 的骨科 CT 影像智能分割与三维重建研究**

## 2. 中期阶段目标对应关系

根据任务书，2026 年 5—7 月应完成文献调研、总体方案和实验环境，并建立 DICOM 解析、灰度/HU 标准化、空间重采样、骨窗增强和质量控制流程；2026 年 7—9 月应围绕 SegFormer 建立骨科 CT 多尺度分割、联合损失、困难样本增强、不确定性精修，并开展模型训练、调参与消融。

当前材料按“**已完成的真实工程/实验结果—仍缺的规模化与外部验证**”分开记录，避免把计划、目标值或外部阻塞写成已完成结果。

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

已形成 `docs/08_literature_matrix.md` 的 **44 条结构化文献矩阵**，并建立 `paper/references.bib` 的 **44 条机器可用 BibTeX（42 条英文核心 + 2 条已核验中文文献）**。当前覆盖：

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

当前结论：本项目不应把“使用 Transformer”本身作为主要创新，而应聚焦**骨科 CT 标准化、多尺度轻量 3D 分割、表面/拓扑质量、困难病例和不确定性可靠性**。本轮同时纠正了 CTSpine1K、VerSe、VerFormer、nnFormer、SegFormer3D、TEDS-Net、EDUE 等易被二手题录写错的条目；两条国内文献已分别通过万方医学网与《中国医学装备》期刊官网/CNKI 期刊页完成一手题录复核。

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

当前电脑未检测到 `nvidia-smi`，但项目已通过显式 `--allow-cpu` 的 formal readiness，并在本机 CPU 完成当前 7/2/1 formal-pipeline pilot 的真实训练、validation 与一次性 independent test。GPU 在当前阶段仅是扩大实验规模和缩短墙钟时间的加速条件，不应再把“无 GPU”写成方法学未完成。

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

校准指标已接入 checkpoint evaluation，并使用固定分箱/固定随机种子体素采样保证不同实验可复现比较。`UncertaintyRefinementNet3D` 已完成真实二阶段验证：refinement 仅使用 7 个 train cases 训练，`liver_7/liver_8` 仅用于 validation；复用 v13 已保存 prediction + predictive entropy，通过 canonical binary logits 重建避免重复 coarse full-volume inference。两例 reconstruction prediction mismatch 均为 0、entropy max abs error≈`9.86e-7`；Top-5/10/20% × dilation 0/1/2 的 3×3 grid 与 full-volume second-pass 已全部完成，所有 ROI-only candidate 的 `outside_roi_changed_fraction=0`。数值最强候选 Top-20%+dilation2 虽把两例 mean Dice 从 `0.05471` 提到 `0.07407`、foreground ratio 从 `4.03×` 拉近到 `0.965×`，但 mean Recall 从 `0.13649` 降至 `0.06965`、component error 从 `1543.5` 恶化到 `2397.5`、false break 从 `64.5` 恶化到 `138`，`liver_7` HD95/ASSD 还出现反向恶化，CPU pipeline time 从约 `75.01 s` 增至 `99.95 s`。因此综合判定 **REFINEMENT=FAIL**，最终 validation pipeline 保留 v13 coarse，不把局部精修包装为成功结果。

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
- 正式/工程 `preflight`：阻止 engineering split、test_private 泄漏、人工 QC 未通过、标签范围/类别数不匹配等不合格正式实验；CPU 正式路径必须显式 `--allow-cpu`，避免把算力降级静默混入实验；
- `task_lock`：只有组内明确 `task_locked=true` 且 binary/multiclass semantic 定义、类别数、数据与 split 均完整时才允许编译正式 config；当前 instance 任务因训练/评价链尚未实现而明确拒绝；
- `gpu_environment` + `formal_readiness`：只读检查 PyTorch CUDA build、可见 GPU、显存、`nvidia-smi`，并将 task/GPU/formal preflight 汇总为统一报告；不合格 split/config 会被正确阻断，当前锁定 10 例 pilot 在显式 `--allow-cpu` 下已验证 `ready=true / blocker_count=0`；
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
- evaluation results-review 页面：已发现并读取真实 `evaluate.py` 输出；Edge 实机已完成 v13 validation prediction/uncertainty MPR，并在最终锁参后识别唯一一次正式 `liver_169` independent evaluation；正式 test prediction/uncertainty MPR API 均返回 200；
- `research-3d` 已接真实 evaluation prediction：validation `liver_8` 与 independent `liver_169` 的 2.0 mm feature-weighted mesh、0.4 mm SDF 均已在 Edge WebGL2 实机加载；
- 物理毫米坐标距离和三点夹角计算 API；
- 研究用途免责声明；
- 模型分割接口已预留。

Web `TestClient` 验证：

```text
GET /api/health → 200
GET /           → 200 text/html
```

科研展示主路径不通过临时 `/infer` 重新生成结果，而是直接读取可追溯的 `evaluate.py` evaluation 产物；validation 与 independent test 均保存 prediction/entropy NIfTI，并由 results-review / research-3d 复核。**Web 不使用随机权重或临时重跑结果伪造“诊断结果”。**

### 3.10 三维重建与重采样几何工程验证

已在真实 `liver_0` 标签上完成 physical-space Marching Cubes：131,983 顶点 / 264,362 面。为适配 Web，又实现 vertex-clustering 简化：1.5 mm 档约减少 60% 顶点/面，顶点近邻 HD95 约 0.707 mm；全分辨率网格仍保留，不被简化版覆盖。2026-08-26 进一步加入“法向变化加权”的曲率/关键边缘保护候选：在 2.0 mm clustering、保持相同 30,260 个简化顶点的条件下，真实 `liver_0` 高法向变化区域平均最近邻误差由约 0.679 mm 降至 0.620 mm，HD95 由约 1.068 mm 降至 1.000 mm，表面积相对变化由约 -6.47% 改善至 -5.95%。这些早期结果用于真值网格工程参数筛选；随后相同工程链已扩展到 v13 validation 与正式 independent prediction surface。

进一步实现 physical-mm SDF 平滑表面基线：0.3/0.4/0.5/0.8 mm 参数已在真实 `liver_0` 上做工程 sweep。0.4 mm 保持 2→2 连通域且 Web summary/PLY 均返回 200，作为当前工程默认候选；0.8 mm 导致 2→3 连通域变化，Web 按拓扑保护返回 422 拒绝加载。随后在 v13 validation 的真实 prediction surface 上完成正式工程复核：`liver_7/liver_8` 原始 prediction mesh 顶点约 `1,369,586/1,340,666`；2.0 mm + feature strength=8 后约 `298,840/296,483`，顶点缩减约 `78.18%/77.89%`，简化工程 ASSD≈`0.55845/0.54806 mm`、HD95≈`1.09434/1.08294 mm`。0.4 mm SDF 分别保持 components `1564→1564`、`1528→1528`，相对原 prediction surface 的工程 ASSD≈`0.02919/0.02929 mm`、HD95≈`0.06790/0.06671 mm`。这些 vertex-nearest surface 数字只用于三维重建工程控制，不属于临床 segmentation HD95/ASSD。

同时对 10 例原始 label 与 1 mm nearest-neighbor 重采样 label 做物理表面离散化比较：整体顶点近邻 ASSD 约 0.403 mm、HD95 约 0.734 mm；原始 5 mm 厚层的 3 例扰动更明显（ASSD 约 0.514 mm、HD95 约 1.069 mm）。这些数字只描述**预处理/离散化工程误差**，不是模型性能或临床测量误差。

### 3.11 自动化测试

当前测试：

```text
138 passed
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

## 4. 当前边界、已完成门禁与仍未完成内容

1. **CTSpine1K 10 例真实工程子集已下载/标准化，但最终论文主数据规模仍偏小，后续仍需扩大病例；**
2. **10/10 自动 QC 与 10/10 人工 QC 已完成并通过；**
3. **临床脱敏数据尚未提供；**
4. **首个任务已锁定为 `vertebra_binary_ctspine1k_msd_t10_v1`，binary semantic，7/2/1 patient-level split 已固定；**
5. **当前真实训练/validation 在 CPU 跑通，GPU 仅是提速项，不应把“无 GPU”写成方法学未完成；**
6. **v13 validation 与锁参后的唯一一次正式 independent test 已严格分开报告：validation mean Dice≈0.05471；independent `liver_169` Dice≈0.02878、HD95≈136.87 mm、ASSD≈43.97 mm；正式 test 后没有再次调参；**
7. **输入、loss、sampling、augmentation、困难样本、uncertainty/calibration 与 ROI refinement 均已有真实 validation 消融；其中 refinement 综合判定 FAIL，不能只凭 Dice 上升写成成功；**
8. **Marching Cubes、真实 label 网格、1.5/2.0 mm 工程简化、10 例重采样几何误差、validation prediction 与正式 independent prediction 的 mesh/SDF/simplification 均已验证；所有 vertex-nearest surface 数字仅作为工程误差；**
9. **Web validation 阶段实机验收已完成：人工 QC、交互 MPR、真值 3D/SDF、物理坐标测量、v13 prediction/entropy overlay、真实 prediction 2.0 mm WebGL、0.4 mm SDF WebGL、GT/prediction 双来源切换均已验证；这仍是科研原型，不代表临床系统。**

任务书中的 `Dice ≥ 0.93` 是目标值，不是当前成果。

---

## 5. 中期报告“研究进展”可直接使用的表达

截至当前阶段，项目已完成总体技术路线设计、44 条结构化文献矩阵与 44 条机器可用 BibTeX（42 条英文核心 + 2 条已核验中文文献），并搭建项目内 Python 3.11.7 / PyTorch 2.1.0 CPU 环境。CTSpine1K `MSD-T10` 10 例真实 CT+label 已全部完成 1 mm 标准化、自动审计与人工 QC（10/10 pass），首个任务固定为 `vertebra_binary_ctspine1k_msd_t10_v1`，patient-level split 固定为 7 train / 2 validation / 1 test。围绕 SegFormer3D 已完成训练稳定性诊断、输入、Region/Boundary/Topology loss、sampling、augmentation、模型驱动困难样本，以及 uncertainty/calibration 消融；最终 validation baseline 选择 v13：CT-only + Region+Boundary + Bernoulli sampling（foreground_probability=0.25、patches_per_case=4）+ flip-only，两例 mean Dice≈`0.05471`、HD95/ASSD≈`185.95/51.49 mm`。7-train-case uncertainty refinement 虽有候选 Dice 上升，但 Recall、fragmentation、病例稳定性与耗时恶化，因此综合判定 **REFINEMENT=FAIL**。validation/3D/Web 阶段提交 `2f333ba` push 后，最终参数通过 `docs/10_final_parameter_lock.md` 锁定并提交为 `eb0a824`；确认 `HEAD == origin/main` 后，才首次按最终锁定协议对官方 `test_private liver_169` 执行 FINAL FORMAL INDEPENDENT TEST。仓库中更早的 5-epoch pilot test 仅作为历史工程链证据，不属于本次最终锁定测试，也未用于 v13 参数选择。最终锁定协议下的正式 independent test 得到 Dice≈`0.02878`、HD95≈`136.87 mm`、ASSD≈`43.97 mm`、foreground ratio≈`2.21×`、prediction/GT components=`236/1`；uncertainty AUROC≈`0.86424`。测试后没有重新调 threshold、refinement 或模型参数。独立 prediction 的 2.0 mm simplification 顶点减少约 `77.73%`，0.4 mm SDF components `236→236`，并已在 Edge 完成 independent results-review MPR 与 3D/SDF WebGL 实机验收。当前结论必须保持：工程闭环已完整，但模型绝对分割精度仍低，不属于高精度临床系统。

---

## 6. 中期展示建议

统一的 v0.3.0 中期/结题展示源材料见 `docs/12_final_presentation_outline.md`。正式 PPT 建议只展示能被工程文件复核的内容：

1. 任务书技术路线与本项目实际模块图；
2. DICOM → HU → spacing → bone window → QC 流程图；
3. SegFormer3D backbone 与本项目新增模块边界；
4. 联合损失公式与为什么需要 HD95/ASSD；
5. Web 首页/上传/QC 真实截图；
6. 测试结果 `138 passed`、10 例真实数据自动审计、真实 patch smoke、formal readiness、正式 independent test 与真实重采样/SDF/网格工程证据；
7. 数据集对比表；
8. 已完成的输入/loss/sampling/augmentation/difficult-sample/refinement 消融表，以及明确标注“尚未真实运行”的跨架构强 baseline 计划；
9. 风险：样本规模、临床数据、许可证、跨中心泛化、拓扑约束对骨折病例的适用性；
10. 下一阶段甘特图。

---

## 7. 下一阶段形成可验收“中期硬证据”的最低清单

- [x] 明确首个任务：`vertebra_binary_ctspine1k_msd_t10_v1`，binary semantic；
- [x] 下载并登记 CTSpine1K 10 例真实工程子集；
- [x] 对 ≥10 例真实多层 CT 完成标准化和自动 QC；
- [x] 项目成员完成 10 例三视图/overlay 人工审核并在 CSV 签字；
- [x] 固定 patient-level split：7 train / 2 validation / 1 test；
- [x] CPU 跑通 SegFormer3D engineering/validation baseline；GPU 仅作为后续提速项；
- [x] 生成真实 validation `metrics_per_case.csv`；
- [x] 报告 engineering/validation baseline DSC / HD95 / ASSD，并严格与最终 independent test 区分；
- [x] 完成 Region vs Region+Boundary/Topology 初步消融；
- [x] Web 接入真实 evaluation prediction/entropy，Edge 实机显示分割与不确定性叠加；
- [x] Web `research-3d` 实机加载 v13 validation prediction 2.0 mm mesh 与 0.4 mm SDF，并验证 GT/prediction 双来源切换；
- [x] 保存可复现 config / lock commit / checkpoint / evaluation 记录；
- [x] 更新论文真实 validation 与 independent-test Results。
