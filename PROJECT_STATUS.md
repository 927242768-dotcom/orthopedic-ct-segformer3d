# 骨科 CT 智能分割与三维重建项目——主进度与交接台账

> 项目目录：`D:\国创项目`
>
> 项目名称：**基于 SegFormer3D 的骨科 CT 分割、三维重建与科研 Web 辅助分析协作项目**
>
> 台账首次建立：2026-08-15
>
> 最近更新：2026-09-22

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

> **特别强调：任务书中的 Dice ≥ 0.93 是项目目标，不是当前实验结果。当前已完成 engineering / validation、最终锁参与唯一一次正式 independent test；validation 与 independent-test 指标必须严格分列，当前正式 test Dice 也仍很低，绝不能美化。**

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

## 2. 当前总体状态（2026-09-22）

> **最终训练冻结快照：SegFormer3D 当前唯一工程主模型为 v7。v7.1 为 residual-FP refinement 负结果，已归档但不替换 v7；TRAINING LINE CLOSED。以下历史 v13 / 10-case pilot 记录保留用于可追溯性，不再代表当前主模型。**

| 模块 | 状态 | 完成度 | 当前真实状态 |
|---|---|---:|---|
| 任务书/组会材料梳理 | ✅ 已完成 | 100% | 已提取研究目标、时间轴、系统功能、论文/中期要求 |
| SegFormer3D 上游调研 | ✅ 已完成 | 100% | 已读 README、核心架构、loss、依赖与许可证；官方仓库已克隆到 `third_party/SegFormer3D` |
| 项目目录与交接机制 | ✅ 已完成 | 100% | 已建立工程目录和本主台账；明确“每次实质修改必须更新本文件” |
| 总体方案设计 | ✅ 已完成 | 100% | 已形成数据层、模型层、三维层、Web 层和实验追踪设计 |
| 国内外文献调研 | ✅ 当前阶段完成 | 100% | 已形成 44 条结构化文献矩阵；`paper/references.bib` 共 44 条机器可用题录（42 条英文核心 + 2 条已核验中文文献）。现代强 baseline、骨折、真实金属植入物、低骨密度椎体 fusion/split 证据均已核验；两条国内文献已分别通过万方医学网与《中国医学装备》期刊官网/CNKI 期刊页完成一手题录复核。后续只在扩大正式实验时按任务补必要文献，不再机械凑数量 |
| 实验环境 | ✅ 已完成（CPU 可训练环境） | 96% | 项目内 Python 3.11.7 + `.venv` 已完成；当前实测 Ryzen 7 8745H（8C/16T）、约 20 GB RAM、PyTorch `2.1.0+cpu`。真实 36³ patch 与 3-epoch binary engineering pilot 均已在本机 CPU 跑通；`train.py`/`formal_readiness.py` 新增显式 `--allow-cpu`，无 NVIDIA 不再是方法学硬 blocker。GPU 仅作为后续提速选项 |
| DICOM/CT 处理流程 | 🟡 进行中 | 96% | NIfTI pipeline 0.3.0 已在 10 例真实 CTSpine1K CT+label 上完成 1 mm 重采样、HU clip→case-wise z-score、骨窗、label nearest-neighbor、自动/交互 QC；10/10 自动审计通过。2026-08-26 复核 `manual_qc_review.csv`：10/10 四项人工检查均 `yes`、10/10 `pass`、reviewer 已填写，人工 QC P0 已解除；真实多层 DICOM series 仍待后续数据来源验证 |
| patient-level 数据划分 | ✅ 已完成（10例 formal pilot） | 96% | 已固定 `ctspine1k_msd_t10_binary_formal_pilot_v1.json`：7 train / 2 validation / 1 test，patient-level 互斥；官方 `test_private liver_169` 只进入 test、不参与训练/调参；`formal_experiment=true`。最终论文仍需扩大病例规模 |
| 公开数据集整理 | 🟡 进行中 | 96% | CTSpine1K `MSD-T10` 10 个真实 CT+label 已落盘：`liver_0`—`liver_8` + `liver_169`，官方 split 为 9 `trainset` + 1 `test_private`；真实文件接管执行 SHA-256 校验，10 例全部标准化/QC，并已按 7/2/1 完成当前 formal-pipeline pilot。该规模仍远小于最终论文/临床验证要求，后续需扩大病例并建立新的预注册 split |
| 临床脱敏数据 | 🔴 阻塞 | 0% | 当前项目目录无临床数据；必须等待合法授权、脱敏与伦理/使用范围确认 |
| SegFormer3D 骨科适配 | 🟡 进行中 | 96% | adapter、配置、dataset、训练骨架已完成；首个任务已锁定为 `binary_semantic`。v9 证明冻结 BN running stats 可显著缓解 v6 foreground explosion，但不能消除退化；v10 在 encoder 与 BN 全冻结后仍发生 mean Dice≈`1.65e-11` 的 catastrophic background collapse，否定 encoder parameter update 为必要条件。v11 从 epoch2 起同时冻结 encoder、BN running stats 与 decoder feature（`linear_c1..c4` + `linear_fuse`），仅允许 `linear_pred` 更新，并已完成 3 epoch：mean val Dice=`0.0540700072 → 0.0543761681 → 0.0546575740`，连续三轮无 catastrophic collapse。epoch3 `liver_7/liver_8` detailed Dice≈`0.04514/0.06417`、prediction/GT ratio≈`4.22/3.78`；新的 epoch1 exact-anchor→epoch3 dynamics 证明 encoder/BN/decoder-feature delta=`0`、fixed-patch encoder/fuse/head-input activation exact equal，仅 `linear_pred` 与 final logits 改变。与已保存的 epoch1→epoch2 dynamics 交叉验证后，正式判定 stable baseline=`YES`（engineering/validation）。绝对分割精度仍低，但 validation/3D/Web 闭环已完成并在 `2f333ba` 推送后正式锁参：lock parameters=`YES`、formal independent test ready=`YES`；锁参记录见 `docs/10_final_parameter_lock.md`。从锁参提交开始不得依据 independent test 调整 threshold、refinement 或其他参数；当前证据支持 decoder feature update 是 v10 collapse 的关键机制之一，但不写成唯一根因 |
| 区域损失 | ✅ 已完成（代码） | 90% | Dice + CE/BCE 可运行并有 backward 测试 |
| Boundary Loss | 🟡 进行中 | 85% | v13 已完成 3-epoch validation 消融；相对 Region 的 HD95/ASSD 仅约改善 0.1002/0.0355 mm，收益极弱，暂保留为 sampling baseline 候选但不宣称明确优势 |
| Topology Loss | 🟡 进行中 | 80% | v14/v15 已完成 validation 消融；结构/表面指标有改善信号，但 foreground overprediction 与 calibration 代价明显，当前不选作后续 baseline；骨折/非管状骨结构适用性仍待独立检查 |
| 困难样本增强 | ✅ 当前 validation 阶段闭环 | 94% | boundary-proxy、冻结 v13 模型驱动 high-loss / high-uncertainty mining 均已完成真实 validation 消融；v22/v23 均失败并按 STOP 规则停在 epoch2，最终保留 v13 Bernoulli sampling。thick-slice 仅获得 train patch-level difficulty signal；metal/fracture/low-density 因 metadata 证据不足，不伪造 subgroup 结果 |
| 不确定性机制 | ✅ 当前 validation 阶段闭环 | 97% | v13 validation `liver_7/liver_8` uncertainty/calibration 已完成；随后真实完成 7 个 train cases 训练的 uncertainty ROI refinement 两例 `3×3` validation grid（Top-5/10/20% × dilation 0/1/2）及 full-volume second-pass。canonical reconstruction 两例 mismatch=`0`，entropy max abs error≈`9.86e-7`，所有 ROI-only `outside_roi_changed_fraction=0`。最佳均值候选 Top-20%+dilation2 的 Dice=`0.07407`、HD95/ASSD=`175.96/47.13 mm`、foreground ratio=`0.965×`，但 Recall 从 coarse `0.13649` 降至 `0.06965`、component error 从 `1543.5` 恶化到 `2397.5`、false break 从 `64.5` 恶化到 `138`，且 `liver_7` HD95/ASSD 反而恶化；mean pipeline time 从 `75.01 s` 增至 `99.95 s`。因此按综合区域/表面/拓扑/稳定性/耗时判定 **REFINEMENT=FAIL**，最终 pipeline 保留 v13 coarse `best.pt`；锁参后唯一一次正式 independent test 已完成，不再依据 test 调整 refinement |
| 训练/验证框架 | 🟡 进行中 | 99% | DataLoader/AdamW/AMP/gradient accumulation/sliding-window、scheduler、完整 run 追踪已接入；`train.py` 支持 `--allow-cpu`、可靠 `--resume` 与 `training.patches_per_case`。balanced v3 已实际使用 `validation.patch_mode=false`，逐 epoch 直接以 `liver_7/liver_8` full-volume Dice 选 checkpoint；epoch 1/2 已完成，说明 full-volume-aware selector 已进入真实训练闭环，不再依赖固定 foreground patch proxy |
| 评价指标 | ✅ 已完成（含正式 independent test） | 100% | Dice、IoU、Precision、Recall、HD95、ASSD、component count/error、false merge/break、uncertainty 与 ECE/MCE/Brier/NLL 已接入。锁参提交 `eb0a824` push 并确认远端一致后，已对 `test_private liver_169` 执行唯一一次 FINAL FORMAL INDEPENDENT TEST：Dice=`0.02878288`、IoU=`0.01460158`、Precision=`0.02089816`、Recall=`0.04622219`、HD95=`136.8722 mm`、ASSD=`43.97199 mm`、foreground ratio=`2.21178×`、pred/target components=`236/1`、component error=`235`、false merge=`0`、false break=`29`、inference=`9.4128 s`；uncertainty AUROC/AUPRC=`0.86424/0.29665`、Top-10% error recall=`0.54993`，ECE/MCE/Brier/NLL=`0.02740/0.08782/0.08328/0.23559`。历史 pilot 结果仍只作为工程链证据 |
| Web 科研辅助分析原型 | ✅ validation + independent test 闭环 | 99% | 首页/上传/健康检查、MPR、10 例人工 QC reviewer、C1–L6 可读标签、真值 PLY WebGL2 3D、简化/物理测量均已完成；QC reviewer 已修复全站 `.card` grid-column 与 QC 网格冲突，病例选择后使用 `hidden + display:none!important` 彻底关闭病例层并进入主审核区，“上一例 / 下一例”保持审核区，悬浮按钮可随时重新打开病例列表；已在本机 Edge 对真实 `liver_0` 完成点击关闭/重新展开实机验证。SDF surface 与 evaluation results-review 已读取真实 prediction/entropy MPR；`results-review` 已实机读取真实 v13 validation evaluation，Edge 已显示 prediction MPR overlay 与 predictive-entropy/uncertainty overlay；`research-3d` 已加载 liver_8 的 2.0 mm feature-weighted prediction mesh 与 SDF σ=0.4 mm surface，并完成 GT/prediction 双来源切换。锁参后正式 independent evaluation 已在 `results-review` 被真实识别，prediction/uncertainty MPR API 均返回 200；`research-3d` 已在 Edge 实机加载 independent liver_169 的 2.0 mm prediction mesh 与 SDF σ=0.4 mm surface |
| 三维重建 | ✅ validation + independent test 闭环 | 99% | 已实现 physical-space Marching Cubes、PLY/JSON、vertex-clustering、SDF surface、WebGL2 与物理测量；新增相邻法向变化驱动的特征保护 vertex-clustering 候选，真实 `liver_0` 在 2.0 mm/同 30,260 顶点下将高特征区域 mean-NN 约 0.679→0.620 mm、HD95 约 1.068→1.000 mm，作为真值网格工程证据；0.4 mm SDF 保持 2→2 连通域，0.8 mm 因 2→3 被保护机制拒绝。已在 v13 validation 真实 prediction surface 上完成工程验证：liver_7/liver_8 的 2.0 mm、feature strength=8 简化分别保留约 298,840/296,483 顶点，顶点缩减约 78.18%/77.89%，简化工程 ASSD/HD95≈0.55845/1.09434 mm 与 0.54806/1.08294 mm；0.4 mm SDF 分别保持 1564→1564、1528→1528 连通域，SDF-vs-original 工程 ASSD/HD95≈0.02919/0.06790 mm 与 0.02929/0.06671 mm。独立 test prediction 也已完成：原始 mesh `365,247` 顶点 / `724,694` 面；2.0 mm + feature strength=8 后 `81,353` 顶点 / `160,384` 面，顶点缩减约 `77.73%`，简化工程 ASSD/HD95≈`0.56490/1.07159 mm`；0.4 mm SDF 保持 `236→236` components，SDF-vs-original 工程 ASSD/HD95≈`0.02536/0.06367 mm`；prediction-vs-GT vertex-nearest engineering ASSD/HD95≈`41.1398/131.8726 mm`，且 size/spacing/origin/direction 全部一致。以上均为 engineering surface 指标，不冒充分割临床 HD95/ASSD |
| 论文 | 🟡 进行中 | 98% | 中文技术稿已完整同步当前 formal-pipeline pilot：validation 主结果、输入/loss/sampling/augmentation/difficult-sample 消融、uncertainty/calibration、`REFINEMENT=FAIL`、prediction 3D/SDF、Web prototype，以及最终锁定 v13 协议下的正式 `liver_169` independent test（Dice=`0.02878`）均已进入 Results；Discussion、Failure Cases、Limitations、Conclusion 已同步并明确低精度/非临床定位。两条国内中文题录已完成一手数据库/期刊页核验；仍未完成的是跨架构强 baseline、扩大数据规模，以及目标期刊/学校模板确定后的最终格式定稿 |
| 中期材料 | 🟡 进行中 | 98% | 已同步 10 例真实数据、`pytest=138 passed`、10/10 人工 QC、正式 binary task lock、7/2/1 formal-pilot split、`formal_readiness ready=true`；validation 消融、`REFINEMENT=FAIL`、prediction 3D/SDF、Web 实机验收，以及最终锁定协议下的正式 independent full-volume test / uncertainty-calibration / independent 3D/SDF/Web 均已写入。旧 5-epoch pilot test 仅标为历史工程链证据；后续主要外部缺口是扩大样本规模后的主实验、合法临床数据与外部验证 |
| 自动化测试/代码质量 | ✅ 已完成（当前阶段） | 100% | `pytest: 138 passed`；`ruff: All checks passed`；新增 decoder-feature freeze policy、仅保留 `linear_pred` trainable、恢复 trainability 与 v11/v10 单变量 config diff 回归测试；focused freeze tests=`15 passed`。同时保留 BatchNorm running-stat freeze、encoder freeze、fixed-per-case sampling、`region_dice_ce` 权重、`patches_per_case` 多 patch 随机流、foreground-fraction evaluation、checkpoint resume、分病例 full-volume evaluation、CPU 非 AMP autocast、epoch-aware sampling 与 `allow_cpu` readiness 测试；文献库现为 44 条 BibTeX，结构检查确认 44 entries / 0 duplicate key / brace balance=0 |

### 2.1 2026-08-29 validation 阶段最终门禁

- 最终 validation 主模型固定候选：`configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml` + `experiments/20260828_002035_cpu_binary_loss_region_boundary_v13_roi64/checkpoint/best.pt`；CT-only、Region+Boundary（1.0/0.1，topology=0）、Bernoulli `foreground_probability=0.25`、`patches_per_case=4`、flip-only、64³ ROI、epoch2 起冻结 encoder / decoder feature / BN stats，仅更新 `linear_pred`。
- v13 `liver_7/liver_8` mean validation Dice=`0.05470944095`；低精度、foreground overprediction、较大 surface distance 与明显 component fragmentation 均如实保留，不美化。
- uncertainty ROI refinement 已完成 3×3 grid 与 full-volume second pass；数值最强 Top-20%+dilation2 mean Dice≈`0.07407384`，但 Recall、component topology、两例稳定性与耗时综合不达标，因此 **REFINEMENT=FAIL**，最终 inference 禁用 refinement。
- 真实 prediction 3D 工程链已完成：liver_7/liver_8 原始 prediction mesh、2.0 mm feature-weighted simplification、0.4 mm SDF、physical spacing/origin/direction 复核均完成；mesh vertex-nearest error 只作为工程误差，不冒充分割临床指标。
- Edge 实机已完成：`results-review` prediction MPR + uncertainty MPR；`research-3d` v13 liver_8 2.0 mm prediction WebGL（296,483 顶点 / 591,833 三角面）；SDF σ=0.4 mm WebGL（1,340,319 顶点 / 2,672,566 三角面）；GT/prediction 双来源切换正常。
- test 隔离保持有效：`/api/research/cases` live API 仅暴露 liver_0～liver_8 共 9 例；截至本门禁记录，`liver_169` 在本轮锁参前没有被读取影像、prediction、entropy、metrics、3D 或 diagnostics。
- 质量门禁：`pytest tests -q`=`138 passed`；`ruff check src web tests`=`All checks passed!`；`git diff --check` 无 error；`node --check web/frontend/research_3d.js` 通过。
- validation 阶段提交 `2f333ba` 已 push 并确认 `HEAD == origin/main`；当前进入最终参数锁定：`lock parameters=YES`、`formal independent test ready=YES`，唯一锁参记录为 `docs/10_final_parameter_lock.md`。本锁参提交 push 并再次确认远端一致后，才允许首次正式访问 `liver_169`。

### 2.2 2026-08-29 FINAL FORMAL INDEPENDENT TEST

- 锁参提交：`eb0a824c34af4f7d900432e169759115f99a2687`（`experiment: 锁定最终独立测试参数`），Author/Committer 均为 `927242768-dotcom <927242768@qq.com>`；测试前已 push 并确认 `HEAD == origin/main`。
- 正式 test 只运行一次：`ctspine1k-msd-t10-liver_169`，输出目录 `experiments/final_independent_test_20260829_v13_locked_liver169`；provenance 文件明确标记不得重跑或据 test 调参。
- 使用完全锁定的 v13 config / `best.pt` / CT-only / Region+Boundary / softmax+argmax / refinement disabled；formal preflight=`ready=true`、0 error/0 warning。
- 区域/表面指标：Dice=`0.02878288`，IoU=`0.01460158`，Precision=`0.02089816`，Recall=`0.04622219`，HD95=`136.8722 mm`，ASSD=`43.97199 mm`。
- 结构指标：prediction/GT foreground ratio=`2.21178×`，pred/target components=`236/1`，component count error=`235`，false merge=`0`，false break=`29`。
- uncertainty：AUROC=`0.86424490`，AUPRC=`0.29665454`，Top-10% error recall=`0.54993443`；calibration：ECE=`0.02739661`，MCE=`0.08781999`，Brier=`0.08328483`，NLL=`0.23558760`，confidence gap=`0.02739661`；inference=`9.41277 s`。
- independent 3D：原始 prediction mesh=`365,247` 顶点 / `724,694` 面；2.0 mm + feature strength=8=`81,353` 顶点 / `160,384` 面，vertex reduction=`77.73%`，简化工程 ASSD/HD95=`0.56490/1.07159 mm`。
- SDF σ=0.4 mm：components=`236→236`、preserved=true，SDF-vs-original 工程 ASSD/HD95=`0.02536/0.06367 mm`。
- prediction-vs-GT vertex-nearest engineering ASSD/HD95=`41.1398/131.8726 mm`；prediction 与 GT 的 size/spacing/origin/direction 全部一致。该工程 surface 数字不替代正式 segmentation HD95/ASSD。
- Web：`results-review` 已识别该 test evaluation；prediction/uncertainty MPR API 均返回 200；Edge `research-3d` 已真实加载 independent 2.0 mm mesh 与 SDF σ=0.4 mm WebGL。
- 科研判断：正式 test 继续证明当前模型绝对精度低、fragmentation 明显；无论 test 比 validation 某些表面指标好或坏，均不允许再次调 threshold、refinement、sampling、augmentation、loss 或模型参数。

### 2.3 2026-08-29 最终收尾门禁

- 正式 independent test 未重复运行；本轮只复用已存在的 metrics、prediction/entropy 与 3D/Web 产物完成文档和工程验收。
- 实验事实提交：`20311d8`（`experiment: 完成正式独立测试`）；论文/中期材料提交：`6f69d80`（`docs: 同步正式测试论文与中期材料`）。
- 当前文档已明确区分更早 5-epoch pilot evaluation 与最终锁定 v13 的 FINAL FORMAL INDEPENDENT TEST；旧 pilot 只作为历史工程链证据，不用于 v13 参数选择。
- 最终全量门禁重新执行：`pytest tests -q`=`138 passed`；`ruff check src web tests`=`All checks passed!`；`git diff --check` 无 error；`node --check web/frontend/research_3d.js` 通过。
- 公开仓库 tracked-file 检查未发现 DICOM/NIfTI、checkpoint、`experiments/`、`.venv/`、Web runtime 或 `third_party/SegFormer3D/` 被跟踪；常见 private-key/token 模式文件扫描无命中；当前最大 tracked 文件约 248 KB。
- 当前论文与中期材料均如实保留 independent Dice=`0.02878` 的低性能结论；未填写未真实运行的 nnU-Net/Residual-Encoder nnU-Net 等跨架构指标，也未宣称临床性能或统计显著性。

---

### 2.4 2026-09-16 CTSpine1K v6 全规模开发阶段

- 正式主线保持 `SegFormer3D + HR 3D Decoder + CT shallow spatial branch`，没有把 SegFormer3D 从主方案中移除；`direct_plus_coarse` 已作为较弱对照保留。
- v6 紧凑缓存已经完成 807/807：train610 + validation197 + test0，failure=0；`test_private` 198 例仍未读取、未缓存、未参与调参。
- 807 例全量 audit：missing=0、QC bad=0、geometry bad=0、suspicious HU=0；cache CRC 标志和 image/label affine 对齐检查通过。
- 已修复 Windows 压缩 NIfTI 随机 CRC / 异常数值读取问题：预处理端加入稳定读取与 zlib sequential fallback，训练 dataset 使用 SimpleITK reader，并保持原 XYZ→DHW 语义。
- full-scale engineering preflight=`ready=true`，checked=807，error=0，warning=0。
- 当前 run：`experiments/20260916_163356_ctspine1k_v6_segformer3d_hr_full610_val197`；只从 stabilized pilot `best.pt` 加载模型权重，optimizer/scheduler fresh。
- 截至本次 GitHub 阶段快照，epoch1—6 validation patch Dice=`0.81320 → 0.82648 → 0.82718 → 0.84610 → 0.84539 → 0.85848`；当前 best=`0.8584831380`，已超过 pilot best=`0.8215395933`。该数值是开发期 fixed-patch selector，不是 full-volume/最终 test 结果。
- checkpoint / resume 链已核验：`RUNNING.lock`、`last.pt`、`best.pt`、optimizer、scheduler、Python/NumPy/Torch RNG 均存在；auto-continue 将在训练完成后执行 validation197 full-volume sliding-window evaluation。
- 下一门禁：训练/early-stop 完成 → 锁定 best.pt → validation197 full-volume → 汇总区域/表面/结构/per-case/worst-case 指标；完成前禁止触碰 test198。
- 详细记录：`docs/14_ctspine1k_v6_fullscale_stage_20260916.md`。

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

当前 Windows 系统侧没有检测到 `nvidia-smi`，当前 `.venv` 明确为 CPU PyTorch。现有 10 例 formal-pipeline pilot 已在显式 `--allow-cpu` 条件下完成训练、validation 与唯一一次正式 independent test，因此无 NVIDIA GPU 已不再是当前工程闭环的方法学 blocker；GPU 仅作为后续扩大病例规模、运行跨架构强 baseline 和缩短三维训练墙钟时间的效率升级项，GPU 显存数据在没有真实设备前继续保持未报告。

### 3.5 自动化测试

最终本轮验证：

```text
ruff check src web tests
→ All checks passed!

pytest tests -q
→ 138 passed

JSON / BibTeX / frontend structural checks
→ data/datasets.json OK
→ configs/label_schemas/ctspine1k_verse.json OK
→ configs/task_specs/vertebra_task_template.json OK
→ paper/references.bib: 44 entries, brace balanced, duplicate key none
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
- 真实 10 例已生成 contact sheet；2026-08-26 已复核 `manual_qc_review.csv`，10/10 orientation/spacing/label alignment/bone-window 均为 `yes`，10/10 `review_status=pass`，reviewer 已填写；
- `audit_processed` 对 10 例 pipeline/spacing/geometry/label/normalization 自动审计：10/10 pass；
- 真实病例原始 z-spacing 覆盖约 0.8 / 1.0 / 5.0 mm，重采样后均为 1 mm。

网络状态已发生变化：VerSe S3 仍未验证恢复；CTSpine1K Hugging Face 则通过浏览器**单文件顺序下载**成功完成 10 例。并行请求曾产生 `无法下载`，`liver_3/5/8` 顺序重试后成功。接管到项目的文件执行 SHA-256 源/目标一致性检查。当前 10 例 pilot 的任务锁定、人工 QC、7/2/1 split、validation 消融、参数锁定与唯一一次 independent test 均已完成；真正剩余的科研缺口是扩大病例规模、强 baseline、困难病例正式分层、合法临床/多中心验证等外部或新增实验条件。

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

当前已完成科研复核主链：

```text
上传 / 真实病例 QC
→ axial/coronal/sagittal MPR + label/prediction overlay
→ predictive-entropy / uncertainty overlay
→ evaluation results-review
→ prediction / GT physical-space mesh
→ 2.0 mm feature-weighted simplification / 0.4 mm SDF
→ WebGL2 3D 切换与距离/角度测量
```

validation 与正式 independent evaluation 的已保存 prediction / entropy / mesh / SDF 均已被 Web 实机读取；科研展示不通过临时 `/infer` 重跑最终模型。后续 Web 工作只属于易用性、更多病例与外部验证场景的扩展，不再把已完成的 prediction/uncertainty/3D 接入列为当前缺项。

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

- 研究问题、Introduction、Related Work、Methods 与 Experiment Design 主体；
- 数据标准化、SegFormer3D、联合损失、困难样本、uncertainty/calibration、ROI refinement、physical-space mesh / SDF 与 Web 科研复核方法描述；
- 44 条结构化文献矩阵 + 44 条机器可用 BibTeX（42 条英文核心 + 2 条已核验中文文献）；
- v11～v23 validation 消融的真实结果整理；
- v13 最终 validation 主结果、uncertainty/calibration、`REFINEMENT=FAIL`、3D/SDF/Web 工程结果；
- 最终锁定协议下唯一一次 `liver_169` independent test 的 Results；
- Discussion、Failure Cases、Limitations 与 Conclusion；
- 正文顺序引用已开始统一，并与当前参考文献编号对应。

当前 Results 已有可追溯真实数字，不再是 TBD。国内中文题录已完成当前阶段一手数据库/期刊页核验；仍未完成且不得伪造的部分包括：扩大样本后的主实验、真实跨架构强 baseline、metal/fracture/low-density 正式 subgroup、合法临床/多中心验证、统计显著性，以及目标期刊/学校模板确定后的最终格式定稿。

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
| `docs/05_midterm_materials.md` | 中期研究材料、真实结果、边界与展示建议 | ✅ v0.3 已同步最终 validation/independent 状态 |
| `docs/06_public_dataset_onboarding.md` | VerSe/CTSpine1K/TotalSegmentator 登记、下载、10 例 QC 与 baseline 接入 SOP | ✅ 已更新真实状态 |
| `docs/07_real_data_validation_20260816.md` | CTSpine1K 10 例真实落盘、pipeline 0.3.0、审计、patch smoke、真实 mesh 证据 | ✅ |
| `docs/09_public_repository_manifest.md` | 公开 GitHub 仓库纳入/排除文件、医学数据隐私与提交前检查清单 | ✅ |
| `docs/10_final_parameter_lock.md` | 最终 v13 参数锁定、独立测试访问纪律 | ✅ 已冻结 |
| `docs/11_final_independent_test.md` | 唯一一次 FINAL FORMAL INDEPENDENT TEST 与 3D/Web 记录 | ✅ 已完成 |
| `docs/12_final_presentation_outline.md` | v0.3.0 中期/结题展示统一源材料 | ✅ 2026-08-29 更新 |
| `paper/outline.md` | 论文持续写作框架 | ✅ |
| `paper/manuscript_zh_v0.1.md` | 中文论文技术稿；validation/independent Results、Discussion、Failure Cases、Limitations、Conclusion 已同步，继续做引用/格式/语言收尾 | 🟡 |
| `paper/references.bib` | 44 条机器可用 BibTeX（42 英文核心 + 2 已核验中文）；提交前执行 key/括号结构检查 | ✅ |
| `env/requirements.txt` | 固定项目依赖 | ✅ |
| `env/setup_env.ps1` | 项目内 Python 3.11/.venv 环境搭建；优先 uv | ✅ |
| `env/fetch_segformer3d.ps1` | 获取官方 SegFormer3D | ✅ |
| `env/download_verse.ps1` | VerSe 2019/2020 下载计划与显式下载辅助；默认不下载 | ✅ |
| `env/download_ctspine1k_sample.ps1` | CTSpine1K MSD-T10 小样本 CT+label 下载计划与显式下载；默认不下载 | ✅ |
| `env/check_gpu.ps1` | 项目 GPU/CUDA/PyTorch 只读验收入口 | ✅ 本机正确报告 CPU/no CUDA |
| `env/check_formal_readiness.ps1` | task + 算力 + formal preflight 一站式验收入口 | ✅ 不合格配置可阻断；锁定 pilot `--allow-cpu` 已 ready=true |
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
| `src/preprocessing/qc_visualization.py` | 三视图 × normalized/bone-window/label-overlay QC 图与人工审核 CSV | ✅ 10 例真实数据已生成并完成人工审核，10/10 pass |
| `src/preprocessing/audit_processed.py` | 标准化病例 pipeline/geometry/spacing/label/normalization 自动审计 | ✅ 10/10 real pass |
| `src/preprocessing/create_split.py` | patient-level split、防重复患者泄漏 | ✅ |
| `src/modeling/segformer3d_adapter.py` | 官方 SegFormer3D 与本项目配置适配 | ✅ |
| `src/modeling/dataset.py` | 标准化 NIfTI、多通道、3D patch dataset | ✅ 首版 |
| `src/modeling/joint_loss.py` | Region + Boundary + soft-clDice | ✅ v11/v13/v14/v15 validation 消融已完成；最终 v13 使用 Region+Boundary |
| `src/modeling/metrics.py` | Dice/IoU/Precision/Recall/HD95/ASSD | ✅ |
| `src/modeling/uncertainty.py` | entropy、ROI、uncertainty-error overlap/AUROC/AUPRC/Top-percent 与 calibration | ✅ validation + independent test 已真实验证 |
| `src/modeling/refinement.py` | uncertainty ROI 局部残差 3D 精修网络与 ROI 融合 | ✅ validation 3×3 grid + full-volume 对照已完成，最终判定 REFINEMENT=FAIL |
| `src/modeling/refinement_training.py` | coarse 冻结 + ROI-normalized 二阶段精修 loss/step/error delta | ✅ 7 个 train cases 真实 refinement 训练与两例 validation 已完成 |
| `src/modeling/preflight.py` | formal/engineering 实验前置验收与泄漏/人工QC/GPU/标签配置保护 | ✅ real engineering/formal 拦截验证 |
| `src/modeling/task_lock.py` | 锁定 binary/multiclass semantic 任务并编译带 SHA-256 指纹的正式 config；拒绝未实现 instance | ✅ targeted + full regression pass |
| `src/modeling/gpu_environment.py` | PyTorch CUDA/device/显存/`nvidia-smi` 只读验收 | ✅ 本机 CPU blocker 已实测 |
| `src/modeling/formal_readiness.py` | 汇总 task/GPU/formal preflight/config binding 的一站式正式实验验收 | ✅ 不合格配置可正确阻断；当前锁定 pilot 在 `--allow-cpu` 下 ready=true / 0 blocker |
| `src/modeling/train.py` | 训练、验证、scheduler、checkpoint、固定 split/config/环境/train.log | ✅ 当前 CPU formal-pipeline validation 训练链已完成；扩样本/GPU 训练属后续研究 |
| `src/modeling/evaluate.py` | checkpoint sliding-window 独立评估、per-case/per-class、uncertainty 定量指标/prediction/entropy 输出 | ✅ validation + locked formal independent test 已真实运行 |
| `src/modeling/real_patch_smoke.py` | 真实标准化病例双通道 joint-loss 单 patch forward/backward 工程验收 | ✅ real pass |
| `src/reconstruction/mesh.py` | mask→物理空间 Marching Cubes + vertex-clustering + 法向变化加权特征保护 | ✅ GT + validation/independent prediction engineering pass |
| `src/reconstruction/export_mesh.py` | NIfTI label/prediction→全分辨率/简化 PLY+JSON 可追溯导出 | ✅ GT + validation/independent prediction pass |
| `src/reconstruction/resampling_error.py` | 原始 label vs 1 mm label physical-surface 重采样几何误差 | ✅ 10/10 real pass |
| `src/reconstruction/sdf_surface.py` | physical-mm signed-distance smoothing + zero-level MC + 连通域保护 | ✅ GT sweep + validation/independent prediction + Web pass |
| `src/reconstruction/measurement.py` | 物理坐标距离/三点夹角与 voxel→physical 工具 | ✅ |
| `web/backend/app.py` | FastAPI 本地科研服务：上传/MPR/QC/results-review/prediction+uncertainty/3D+SDF/测量 | ✅ 当前科研原型闭环 |
| `web/frontend/` | 上传/QC、交互 MPR+prediction/entropy overlay、results-review、WebGL2 GT/prediction/SDF、测量前端 | ✅ Edge 实机验收闭环 |
| `web/run_web.ps1` | localhost Web 启动脚本 | ✅ |
| `tests/` | 自动化测试 | ✅ 138 passed |
| `data/README.md` | 数据治理与隐私规则、当前真实 CTSpine1K 子集说明 | ✅ |
| `data/datasets.json` | 公开数据集来源/版本/许可/本地状态；已登记 10 例 CTSpine1K 工程子集 | ✅ |
| `data/splits/ctspine1k_msd_t10_engineering_smoke.json` | 1/1/1 真实数据工程 smoke split，明确 `formal_experiment=false` | ✅ 非正式实验 |
| `.gitignore` | 排除医学数据、处理后数据、环境、模型、runtime、第三方 checkout、DevSpace 缓存和大型生成 PPT | ✅ 已加强公开安全规则 |

---

## 8. 当前阻塞与风险

### R1｜当前 pilot 已闭环，但最终论文样本规模不足

CTSpine1K `MSD-T10` 10 例真实 CT+label 已完成 pipeline 0.3.0 标准化、10/10 自动审计与人工 QC；当前 pilot 已锁定 `vertebra_binary_ctspine1k_msd_t10_v1`，固定 7 train / 2 validation / 1 `test_private` patient-level split，并完成 validation 消融、最终参数锁定和唯一一次 independent test。该 10 例结果可以作为**正式流程 pilot**写入当前技术稿，但样本量过小，不能代表最终论文或临床泛化结论。后续真正需要的是：

- 扩大病例规模并建立新的预注册 patient-level split；
- 在新 split 上运行强跨架构 baseline，并重新形成主实验结果；
- 有可靠 metadata 后再做 metal/fracture/low-density/thick-slice 正式 subgroup；
- 引入外部/多中心或合法临床数据后再讨论泛化与统计结论。

本次 `liver_169` 结果已经冻结，不得因扩样本前的任何文档工作重新用于调参。

### R2｜临床数据授权/伦理

临床数据必须满足：

- 已脱敏；
- 合法授权；
- 研究范围明确；
- 不把患者姓名、身份证号、联系方式写入日志/文件名/截图；
- 如学校/医院要求伦理审批，先完成审批再用于研究。

### R3｜GPU/算力仅是后续规模化效率风险

当前项目环境为 PyTorch `2.1.0+cpu`；`src.modeling.gpu_environment` 实测 `torch.version.cuda=None`、`cuda_available=false`、0 个 CUDA device、无 `nvidia-smi`。现有 10 例 formal-pipeline pilot 已在显式 `--allow-cpu` 下完成，因此 GPU 不再是当前阶段的完成阻塞。

若扩大病例、运行 nnU-Net / Residual-Encoder nnU-Net 等强 baseline 或开展更大 ROI/更长训练，建议再确认 NVIDIA GPU、驱动、CUDA/PyTorch、显存与存储空间。没有真实 GPU 设备前不填写峰值显存或 GPU 加速数字。当前本机无 `nvidia-smi` 也不能推断学校服务器或其他设备没有 GPU。

### R4｜当前 pilot 任务已锁定；未来扩任务必须重新预注册

当前 formal-pipeline pilot 的任务已经锁定为 `vertebra_binary_ctspine1k_msd_t10_v1`：binary semantic，2 类，原始前景标签 `1..25` 在训练/评价中统一映射为前景 1。`configs/label_schemas/ctspine1k_verse.json` 的 `1–25 → C1–L6` 仍只用于 QC/Web 可读显示，不改变源标签。

若未来扩大研究到 multi-class semantic、vertebra instance 或其他骨科部位，必须新建 task spec、新 split、新 config 和新的 validation/test 方案，不能把当前 v13 / `liver_169` 的锁定结果反向改写为另一任务。当前 instance 训练/评价链仍未实现，因此不能用 semantic segmentation 冒充 instance 结果。

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

## 9. validation 阶段历史任务清单（归档）

> 本节保留 v3～v15 等阶段当时的勾选状态与待办，用于追溯实验决策，不再代表 2026-08-29 v0.3.0 收尾后的当前任务。**当前真实待办以本节末尾“9.1 当前下一步”为准。**

### P0｜历史：修复 full-volume checkpoint selection 与 baseline

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
- [x] 已在 v13 `liver_7/liver_8` 真实 baseline checkpoint 上验证 entropy 与真实错误空间相关性；
- [x] 已完成 validation Top-percent/threshold 小网格：Top-5/10/20% × dilation 0/1/2；不继续无限扫参；
- [x] 已实现 `UncertaintyRefinementNet3D` 局部残差 refinement head/network（工程代码）；
- [x] 已实现 coarse 冻结、ROI-normalized loss、ROI/global error delta 的二阶段 refinement 训练基线；
- [x] 已完成 coarse vs ROI-only vs full-volume second-pass 两例对照；
- [x] 已报告额外时间与 ROI 比例；当前 CPU validation 不报告/伪造 GPU 显存结果；
- [x] ROI-only 外部 prediction 逐体素保持不变：全部 candidate `outside_roi_changed_fraction=0`；
- [x] canonical prediction+entropy reconstruction 两例 mismatch=`0`，entropy max abs error≈`9.86e-7`；
- [x] 最终综合判定：**REFINEMENT=FAIL**。虽然 mean Dice/IoU/Precision、foreground ratio、global error 与部分 surface 指标改善，但 mean Recall≈`0.13649→0.06965`，component error≈`1543.5→2397.5`，false break≈`64.5→138`，`liver_7` surface 退化且耗时增加，不满足两例稳定综合改善；最终 validation pipeline 保留 v13 coarse；
- [x] uncertainty 可用于当前 validation 的 QC 高风险区域提示，但仅 2 例，仍需扩大病例验证。

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
- [x] evaluation results-review 页面与 prediction/entropy MPR 接口已准备；v13 已存在真实 validation prediction + entropy，下一步进行 Web 实机接入验收；
- [ ] 使用最终 v13 validation prediction 完成 prediction mesh / SDF / simplification 真实验证并接入 Web；
- [ ] Edge 实机完成 prediction overlay / entropy overlay / results-review / prediction 3D 验收。

### P4｜论文/中期/软著

- [x] 已建立 44 条结构化文献矩阵；已补 SpineMamba、2025 解剖变异 Transformer、2026 VertebraFormer、2026 Residual-Encoder nnU-Net、2025 骨折 pipeline、真实金属植入物 deep-MAR 与 2024 低骨密度 fusion/split 直接分割证据；
- [x] 已建立 44 条机器可用 `paper/references.bib`（42 条英文核心 + 2 条已核验中文文献），并纠正多条易错题录；
- [ ] 用真实实验更新论文 Results；
- [ ] 生成主结果表和消融表；
- [ ] 做失败案例图；
- [ ] 做 Web/三维真实截图；
- [ ] 中期 PPT 使用 `docs/05_midterm_materials.md`；
- [ ] 软著材料严格区分上游和自研代码；
- [ ] 每次材料更新同步回写本台账。

### 9.1 当前下一步（v0.3.0 收尾后）

**本机可继续完成：**

- [x] 对 README / PROJECT_STATUS / TASKS / 中期材料 / 中文论文做当前状态一致性复核；
- [x] 论文已写入 validation 与唯一一次 independent test 真实结果，并保留低性能、非临床定位；
- [x] 论文正文顺序引用开始统一，已补主要方法/数据集/相关工作的引用编号；
- [x] GitHub 首页保持简洁，详细 v11～v23 过程只放 `PROJECT_STATUS.md`；
- [x] v0.3.0 Release 已存在且内容与正式收尾状态一致，不因纯文档同步重复发版本；
- [x] 国内中文题录最终核验：伍志发等 2022 已由万方医学网一手页面核验；艾念等 2026 已由《中国医学装备》期刊官网/CNKI 期刊页核验；未提供可确认 DOI 的条目保持空缺，不补造；
- [ ] 目标期刊/学校最终格式模板确定后，再做最后一轮格式定稿；
- [ ] GitHub `LICENSE` 需项目负责人结合自研代码与 SegFormer3D GPL-3.0 边界明确选择；当前仓库未声明许可证，不自动代选。

**需要新增数据/算力/外部条件后再做：**

- [ ] 扩大真实病例规模，并建立新的预注册 patient-level split；
- [ ] 在新 split 上真实运行 nnU-Net / Residual-Encoder nnU-Net 等强 baseline；
- [ ] 有可靠病例标记后报告 metal / fracture / low-density / thick-slice 正式 subgroup；
- [ ] 有真实 NVIDIA 设备后再报告 GPU 显存/加速数据；
- [ ] 获得合法授权临床脱敏数据后开展外部/多中心验证；
- [ ] 样本量足够后再进行置信区间、效应量和统计显著性分析。

**永久约束：**最终 `liver_169` 正式 test 已冻结，禁止再次运行 `evaluate.py` 对其做最终模型推理，也禁止依据其结果重新选择 threshold、loss、sampling、augmentation、refinement、checkpoint 或模型参数。

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
138 passed
All checks passed!
```

附加结构检查：`data/datasets.json`、`configs/label_schemas/ctspine1k_verse.json`、`configs/task_specs/vertebra_task_template.json` 可解析；`paper/references.bib` 当前 44 entries、括号平衡且无重复 key；前端 `app.js / qc_review.js / research_3d.js / results_review.js` 均通过 `node --check`；当前纳入检查的 PowerShell 脚本语法 parser 通过。

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

### 10.5 后续新增正式实验前

当前 v13 / `liver_169` formal-pipeline 已冻结，不再重复训练或最终测试。未来**新增数据规模或新任务**的正式实验必须使用新的 task spec / split / config，并至少满足：

```text
真实处理后数据存在
+ 新的预注册 patient-level split 存在且 formal_experiment 允许
+ 标签定义固定
+ 对应病例人工 QC 达到正式 run 要求
+ 上游 SegFormer3D / baseline 实现可加载
+ formal preflight ready=true
+ 若使用 CPU，必须显式 --allow-cpu；若报告 GPU 指标，必须有真实 GPU 证据
```

不得把旧 `liver_169` 重新纳入新的 validation/调参过程。

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