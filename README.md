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
│  ├─ 08_literature_matrix.md   # 44 条结构化文献矩阵
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
│  ├─ references.bib            # 42 条已核验英文核心 BibTeX
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

**44 条结构化文献矩阵 + 42 条英文核心 BibTeX 已建立（已覆盖 2026 Residual-Encoder nnU-Net 强基线、2025 骨折 pipeline、真实腰椎金属植入物以及低骨密度椎体融合/分裂困难病例）→ 项目内 CPU 开发环境可运行 → DICOM + NIfTI 数据入口/联合损失/训练评估/formal preflight 已实现 → task lock + GPU checker + formal readiness 可阻止未锁定任务或不合格环境误启动正式实验 → CTSpine1K 10 例真实 CT+label 已按 pipeline 0.3.0 完成 1 mm 标准化 → 10/10 自动数据审计通过 → 真实双通道联合损失单 patch forward/backward 已通过 → evaluation 已支持区域/表面/结构、uncertainty 与 calibration 指标 → Web 已支持人工 QC、交互式 MPR+label overlay、真值 3D/SDF 表面、物理距离/角度以及未来正式 evaluate 输出的 results-review 页面 → 真实各向异性重采样、SDF/mesh 与法向变化特征保护简化已有工程证据 → 10/10 人工 QC、binary task lock 与 7/2/1 formal-pilot split 已完成，v11 已形成连续三轮稳定的 CT-only engineering/validation baseline；v12 CT+bone-window 输入消融已完成并因严重 foreground overprediction 判定 CT-only 胜出；v13 Region+Boundary、v14 Region+Topology、v15 Region+Boundary+Topology 的 3-epoch loss 消融已全部完成；Topology 组合对 HD95/ASSD、component error、false break 和 uncertainty AUPRC 有改善信号，但伴随更强 foreground overprediction、较低 Precision 与较差 calibration；四组统一比较后选择 v13 Region+Boundary 作为 sampling ablation baseline（当前两例平均 Dice≈0.05471、HD95/ASSD≈185.95/51.49 mm、foreground ratio≈4.03×），该选择是小样本 validation 下的保守工程决策，不把 Boundary 的极弱收益夸大为显著效果；sampling ablation 已进一步完成：v16 fixed-per-case 虽让跨 epoch foreground-fraction mean 更稳定，但 mean Dice≈0.04575、foreground ratio≈6.51×、ECE≈0.02964，整体劣于 v13；v17 boundary-hard mean Dice≈0.03731、foreground ratio≈36.26×、HD95≈206.50 mm、ECE≈0.19569，出现严重 foreground overprediction 与 calibration 崩坏，因此两者均不选，最终保留 v13 Bernoulli sampling 进入 augmentation / difficult-sample validation；v18 standard geometric augmentation（±10° rotation、scale 0.9–1.1）已完成 3 epoch + 两例 detailed validation，mean Dice≈0.05535 仅略升，但 foreground ratio≈7.56×、Precision≈0.03134、ECE≈0.02716 明显劣化，因此不选；随后 v19 gamma 在 epoch2 后持续明显低于 v13 并 STOP，v20 Gaussian noise 完成 3 epoch + 两例 detailed validation，mean Dice≈0.05530 虽略高于 v13 且 Precision/foreground ratio/calibration 有改善，但 Recall、HD95/ASSD、component error 与 uncertainty ranking 均恶化，综合不取代 v13；v21 ±50 HU shift 在 epoch1/2 Dice≈0.03672/0.03871 后按规则 STOP。因此 augmentation 最终正式保留 v13 flip-only baseline；随后复用 v13 已有 `liver_7/liver_8` prediction + entropy 完成 uncertainty/calibration 两例稳定性分析：AUROC≈`0.92949/0.94466`、AUPRC≈`0.33354/0.33014`、Top-10% error recall≈`0.72891/0.79450`，error voxel entropy 约为 correct voxel 的 `10.41×/12.42×`，支持其作为 validation error indicator、QC signal 与 refinement trigger；ECE≈`0.01418/0.00767`，但当前 Dice 仍很低且仅 2 例，不能宣称前景已充分校准或具备临床可靠性；模型驱动 difficult mining 也已完成真实消融：v22 high-loss epoch1/2 mean Dice=`0.04728839/0.04728457`，v23 high-uncertainty epoch1/2=`0.00963475/0.01010232`，两者均按 STOP 规则停在 epoch2 且不取代 v13；train 中两例 5 mm thick-slice case 的冻结 v13 candidate loss/uncertainty 分别高约 `12.9%/13.0%`，但 validation 无 thick-slice case，metal/fracture/low-density 也缺乏可靠病例标签，因此不伪造 subgroup 指标；最终 difficult-sample 决策仍保留 v13 Bernoulli sampling；随后已完成真实 uncertainty ROI refinement：仅用 7 个 train cases 训练 refinement，`liver_7/liver_8` 只做 validation，完成 Top-5/10/20% × dilation 0/1/2 的 3×3 grid 与 full-volume second-pass。canonical reconstruction 两例 prediction mismatch=`0`、entropy max abs error≈`9.86e-7`，所有 ROI-only 候选 `outside_roi_changed_fraction=0`。Top-20%+dilation2 虽将 mean Dice 从 v13 coarse `0.05471` 提到 `0.07407`、foreground ratio 从 `4.03×` 拉近到 `0.965×`，但 mean Recall 从 `0.13649` 降到 `0.06965`、component error 从 `1543.5` 恶化到 `2397.5`、false break 从 `64.5` 恶化到 `138`，且 `liver_7` HD95/ASSD 反而变差、CPU pipeline time 也从约 `75.01 s` 增到 `99.95 s`。因此按区域+表面+拓扑+两例稳定性+耗时综合判定 **REFINEMENT=FAIL**，最终 validation pipeline 继续保留 v13 coarse `best.pt`，不为论文强行选择 refinement。随后已完成真实 validation prediction 的三维工程闭环：`liver_7/liver_8` 均完成原始 physical mesh、2.0 mm + feature strength=8 的 vertex-clustering 简化与 0.4 mm SDF；2.0 mm 简化顶点缩减约 `78.18%/77.89%`，简化工程 ASSD≈`0.55845/0.54806 mm`、HD95≈`1.09434/1.08294 mm`；0.4 mm SDF 连通域分别保持 `1564→1564`、`1528→1528`，SDF-vs-original 工程 ASSD≈`0.02919/0.02929 mm`、HD95≈`0.06790/0.06671 mm`。这些 mesh vertex-nearest 指标只用于重建工程验证，绝不冒充临床 segmentation HD95/ASSD。Web 端 `results-review` 已在 Edge 实机显示 v13 prediction MPR 与 predictive-entropy/uncertainty overlay；`research-3d` 已实机加载 `liver_8` 2.0 mm prediction（`296,483` 顶点 / `591,833` 三角面）和 SDF σ=0.4 mm（`1,340,319` 顶点 / `2,672,566` 三角面），并完成 GT/prediction 双来源切换。`/api/research/cases` 当前只暴露 `liver_0`—`liver_8` 共 9 例，锁参前继续隔离 `test_private liver_169`；CPU 已可正式 pilot，GPU 仅作为后续提速项**

当前已经产生多轮 **engineering / validation experiment** 指标；v11 已在连续 3 个 epoch 上形成 **engineering / validation stable baseline**。这里的 stable baseline 只表示当前训练机制在既定 validation 条件下连续稳定，不代表绝对分割精度已经足够，也不代表已经锁参进入正式独立 test。2026-08-27 的稳定性实验表明：v6 epoch1 两例 full-volume validation mean Dice≈`0.05407`，epoch2 出现严重 foreground explosion；v8 从初始化开始冻结 BatchNorm3d running statistics 后，epoch1 mean Dice≈`0.0001248` 并转为严重 background collapse，因此 v8 已停止。随后 checkpoint dynamics 显示 v6 epoch1→epoch2 普通参数组变化总体很小，而多处 BN running statistics 与 decoder/head-input activation 明显漂移。v9 采用“epoch1 正常建立 BN stats、epoch2 起冻结”的单变量策略：epoch1 精确复现 v6 epoch1，epoch2 训练后 27/27 个 `running_mean/running_var/num_batches_tracked` 与 epoch1 完全一致，但 mean validation Dice 仍降至`0.0267784`。两例 detailed validation 的 prediction foreground 约为`5.38%/5.01%`、prediction/GT ratio≈`7.69/8.85`，已比 v6 epoch2 的≈`42.26%/34.04%`、≈`60×` 前景泛滥大幅缓解；同时 Precision 仅≈`0.01638/0.01367`，两例 prediction component 均为`517`，说明整体 segmentation degradation 仍未解决。新的四 checkpoint dynamics 进一步确认 v9e1 与 v6e1 参数/BN buffer 完全一致，v9e2 的 BN running buffers 仍严格锚定，但普通 trainable 参数仍发生小幅更新，固定前景 patch 的 final logits mean 从≈`-11.31` 漂移到≈`-4.56`。因此当前证据支持：**BN running-stat drift 是 v6 foreground explosion 的重要放大机制，但不是 epoch2 性能退化的唯一根因。** v9 已按规则停止，不进入 epoch3。进一步拆分 v9e1→v9e2 参数变化后，BN affine≈`0.0194%`、LayerNorm affine≈`0.0208%`，而 patch embeddings≈`0.7842%`、encoder attention≈`0.7075%`；因此 v10 选择更有证据的单变量“epoch2 起冻结 encoder trainable parameters”，同时保持 v9 的 BN-running-stat 锚定以及 lr/loss/sampling/ROI/augmentation/decoder/head 不变。v10 epoch1 已真实完成：train loss=`2.5537127597`、两例 full-volume mean Dice=`0.0540700072`、std=`0.0108403799`、lr=`5e-5`，与 v6/v9 epoch1 精确一致；`liver_7/liver_8` detailed Dice≈`0.04323/0.06491`、prediction/GT foreground ratio≈`3.65/3.18`。两例 validation-only diagnostics 已完成；进一步逐 tensor 比较 v6e1/v9e1/v10e1 的 232 个 model-state tensor，三者 `torch.equal` 全部成立，普通参数 diff=`0`、BN running-buffer diff=`0`。随后 v10 resume 到 epoch2：encoder 208 个 state tensor 与 27 个 BN running buffer 全部保持不变，但 decoder 24 个 tensor 中 21 个更新；mean validation Dice 却直接降到`1.65e-11`，`liver_7/liver_8` Dice/Precision/Recall 均为 0，prediction/GT foreground ratio≈`0.073/0.064`，转为灾难性 background collapse。固定前景 patch 上所有 encoder activation 与 epoch1 完全一致，而 decoder fuse 和 final logits 明显漂移。因此 v10 证明“encoder 参数更新”不是 epoch2 degradation 的必要条件，v10 已停止、不跑 epoch3。v11 工程继承 v10 的 epoch2 encoder freeze + BN-running-stat freeze，并新增唯一主要变量 `freeze_decoder_feature_parameters_from_epoch=2`，即冻结 `linear_c1..c4` 与 `linear_fuse`，仅保留最终 `linear_pred` segmentation head 可训练；其余 lr/loss/sampling/ROI/augmentation/input/scheduler/full-volume validation 均不变。v11 epoch1 已精确复现 v10e1：train loss=`2.5537127597`、mean validation Dice=`0.0540700072`，232 个 model-state tensor 逐项 exact equal。随后同一 run resume 到 epoch2：train loss=`2.3053811001`、mean validation Dice=`0.0543761681`，没有出现 v10 的 background collapse；`liver_7/liver_8` detailed Dice≈`0.04421/0.06454`、prediction/GT foreground ratio≈`3.96/3.50`。checkpoint 严格验证 encoder、BN running buffers、decoder feature 全部 delta=`0`，只有 `linear_pred` weight+bias 更新。validation-only diagnostics 中两例 GT foreground mean P(fg)≈`0.13263/0.16114`、GT background mean P(fg)≈`0.03482/0.02670`，未见 foreground explosion/background collapse；固定 `liver_7` checkpoint dynamics 显示 encoder activation、decoder fuse activation 与 final-head input 完全一致，仅 final logits 随 final head 更新而变化。v11 epoch3 已真实完成且没有重跑：train loss=`1.8300107228`、mean full-volume validation Dice=`0.0546575740`、std=`0.0095167619`，形成 `0.05407001 → 0.05437617 → 0.05465757` 的连续三轮稳定轨迹。`liver_7/liver_8` epoch3 detailed Dice≈`0.04514/0.06417`、Precision≈`0.02792/0.04058`、Recall≈`0.11773/0.15335`、prediction/GT foreground ratio≈`4.22/3.78`，相对 epoch2 没有 foreground explosion、background collapse 或 Dice catastrophic drop。由于 epoch2 checkpoint 已被后续 `best.pt/last.pt` 覆盖，本轮没有伪造 epoch2 checkpoint，而是用 v10e1 `best.pt` 这一已逐 tensor 证明与 v11e1 exact equal 的 anchor 对 v11e3 做新的 validation-only dynamics：encoder、BN running buffers、decoder `linear_c1..c4/linear_fuse` 全部 delta=`0`，仅 `linear_pred` weight+bias 非零；固定 `liver_7` patch 上 encoder、decoder fuse 与 final-head input 的统计字典 exact equal，仅 final logits 改变。再与已保存的 v11e1→v11e2 dynamics 交叉验证，`linear_pred` 对同一 anchor 的 group delta norm 从 epoch2 的`0.0100338`增至 epoch3 的`0.0189645`，可排除 epoch2→epoch3 final head 未更新。基于连续三轮 full-volume validation、freeze 证据和可解释 sampling，当前正式判定 **stable baseline=YES（engineering/validation）**。随后输入、loss、sampling、augmentation、difficult-sample 与 refinement validation 均已闭环；refinement 最终判定 FAIL，因此最终候选仍为 v13 coarse。v13 validation prediction 的 3D/SDF/Web 真实验收现已完成；validation 阶段提交 `2f333ba` 已 push 并确认远端一致；现已形成 `docs/10_final_parameter_lock.md`，当前 **lock parameters=YES、formal independent test ready=YES**。只有锁参提交本身 push 并再次确认 `HEAD == origin/main` 后，才允许第一次且唯一一次正式访问独立 `liver_169`。本阶段证据进一步支持 **decoder feature update 是 v10 catastrophic collapse 的关键机制之一**，但仍不能写成唯一根因，也不能作为论文最终性能结果。随后完成最小可信 baseline reproducibility：固定 v11 epoch3 `best.pt` 与同一 config 分病例重新执行两例 full-volume validation，除墙钟 `inference_seconds` 外所有 summary metric 均逐项 exact equal；因此 checkpoint/inference/evaluation 链可复现。完整 3-epoch CPU 重训尚未重复，不能把这一结果表述为完整训练轨迹复现。输入消融 v12 在保持 loss、optimizer、scheduler、sampling、ROI、augmentation、seed、freeze policy 与 full-volume validation 不变的情况下，仅将输入由 CT-only 改为 CT+bone-window（500/2000）并令 `in_channels=2`。三轮 mean validation Dice=`0.02727074 → 0.02767486 → 0.02802775`。epoch3 对 `liver_7/liver_8` 的 detailed validation 显示平均 Dice/IoU/Precision≈`0.02803/0.01421/0.01422`、Recall≈`0.96594`、HD95/ASSD≈`256.41/83.77 mm`、prediction/GT foreground ratio≈`67.96×`；对应 v11 CT-only 为 Dice≈`0.05466`、HD95/ASSD≈`186.05/51.52 mm`、foreground ratio≈`4.00×`。v12 的高 Recall 来自严重 foreground overprediction，而非结构质量改善；ECE/Brier/NLL 也从 v11≈`0.01084/0.04693/0.10288` 恶化到 v12≈`0.39319/0.80051/3.40655`。因此输入消融正式判定 **CT-only（v11）胜出**，v12 停止，不进入独立 test。v12 checkpoint 的 optimizer step 计数同时验证 epoch2 起 freeze：encoder 184 个参数与 decoder-feature 19 个参数均停在 28 step，`linear_pred` 两个参数到 84 step，9 个 BN 的 `num_batches_tracked` 全部为 28。

2026-08-16 本机对 VerSe 官方 S3 仍存在连接超时；CTSpine1K Hugging Face 则在改用浏览器**单文件顺序下载**后成功取得 `MSD-T10` 的 10 个真实 CT+label（`liver_0`—`liver_8`、`liver_169`）。其中官方 split 为 9 例 `trainset`、1 例 `test_private`。原始文件位于 `data/raw_public/CTSpine1K/MSD-T10`，标准化结果位于 `data/processed_ctspine1k_real`；10 例均完成 1 mm 重采样、HU clip→case-wise z-score、骨窗、nearest-neighbor label 重采样、QC contact sheet、自动审计和人工 QC，10/10 已 `pass`。首个任务已锁定为 `binary_semantic`，formal-pilot split 固定为 7 train / 2 validation / 1 test；当前 CPU 路径在显式 `--allow-cpu` 时 formal readiness 可达到 `ready=true / blocker_count=0`。当前已完成 validation 远端同步并进入最终锁参；在锁参提交 push 完成并再次确认远端一致前，仍禁止访问独立 `test_private liver_169`。除早期 `liver_0` 真值工程 sweep 外，v13 validation 的真实 prediction 也已完成 2.0 mm 简化、0.4 mm SDF 与 Edge WebGL2 验收；会改变连通域的 SDF 参数仍由拓扑保护拒绝。Web 的 results-review 已读取真实 v13 validation `evaluate.py` 输出并实机显示 prediction/entropy overlay；历史 pilot evaluation 仍只作为工程链证据，不能写成正式 independent Results。全套回归的准确数量以 `PROJECT_STATUS.md` 最新值为准。

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

建议所有协作者从 `TASKS.md` 领取任务，在独立 feature/docs 分支开发，通过 Pull Request 合并到 `main`。代码修改至少执行 `pytest tests -q` 与 `ruff check src web tests`；正式论文实验必须先通过 task lock、人工 QC、正式 patient-level split、GPU/CUDA 检查和 `formal_readiness ready=true`。

公开仓库只保存**代码、配置、测试、文档、数据来源/匿名 split 和可复现脚本**。真实医学影像、处理后体数据、checkpoint、runtime、虚拟环境、第三方 checkout 和大型生成 PPT 默认不进入 Git，完整规则见 [`docs/09_public_repository_manifest.md`](./docs/09_public_repository_manifest.md)。

## 下一步

快速任务看板见 [`TASKS.md`](./TASKS.md)；更详细的证据、历史与严格优先级请以 [`PROJECT_STATUS.md`](./PROJECT_STATUS.md) 为准。
