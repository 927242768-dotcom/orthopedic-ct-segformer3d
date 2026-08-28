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
| 文献调研 | ✅ 主体完成 | 44 条结构化矩阵，42 条英文核心 BibTeX |
| CPU 开发环境 | ✅ 已完成 | Python 3.11.7 + `.venv`，PyTorch 2.1.0 CPU |
| 训练算力环境 | ✅ CPU formal-pilot 可用 | Ryzen 7 8745H / PyTorch 2.1.0+cpu；显式 `--allow-cpu` 后 readiness 可通过，GPU 仅为后续提速选项 |
| 公开数据接入 | ✅ 已完成工程子集 | CTSpine1K MSD-T10 真实 10 例 CT+label 已处理 |
| CT 标准化流程 | ✅ 基本完成 | 1 mm 重采样、HU、z-score、骨窗、QC、label NN 重采样 |
| 自动 QC | ✅ 已完成 | 10/10 自动审计通过 |
| 人工 QC | ✅ 已完成 | 10/10 orientation/spacing/alignment/bone-window 均通过，reviewer 已填写 |
| SegFormer3D 工程适配 | ✅ 已完成工程链 | 真实 patch forward/backward/optimizer step 已通过；CPU 训练兼容路径已回归测试 |
| 联合损失代码 | ✅ 已完成代码 | Region + Boundary + soft-clDice |
| 困难样本策略代码 | ✅ 已完成首版 | 3D 几何/强度增强 + boundary hard sampling |
| 不确定性与精修代码 | ✅ 已完成工程链 | entropy、ROI、AUROC/AUPRC、calibration、ROI-only refinement |
| 正式任务定义 | ✅ 已锁定 | `vertebra_binary_ctspine1k_msd_t10_v1`，`binary_semantic`，2 类 |
| formal-pilot patient-level split | ✅ 已固定 | 7 train / 2 validation / 1 test；`liver_169` 仅 test |
| CPU CT-only formal-pilot baseline | 🟡 已跑通但严重欠训练 | 5 epoch 完成，最佳 patch-val Dice≈0.2719（epoch 4） |
| 独立 full-volume pilot test | ✅ 工程链完成 | `liver_169` Dice≈0.0221，inference≈9.29 s；仅 10 例 pilot 证据，禁止作为论文正式结果 |
| 三维重建工程链 | ✅ 基本完成 | physical MC、SDF、简化、WebGL2、物理测量 |
| Web 科研原型 | ✅ 主体完成 | MPR/QC/3D/results-review 已具备 |
| 临床脱敏数据 | 🔴 外部阻塞 | 等授权/脱敏/伦理 |
| 论文 Methods | ✅ 主体完成 | Results 仍保持 TBD |
| 自动化测试 | ✅ 当前通过 | 125 passed + Ruff clean |

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
- [x] Hofmann et al. 2026 开放椎体体部数据 + Residual-Encoder nnU-Net 强 baseline
- [x] Glessgen et al. 2025 椎体骨折 nnU-Net pipeline
- [x] Ye et al. 2025 真实腰椎金属植入物 deep-MAR
- [x] Xiong et al. 2024 低骨密度椎体 fusion/split 直接分割失败证据
- [x] Boundary / Hausdorff / topology 相关文献
- [x] uncertainty / calibration 相关文献
- [x] Marching Cubes / mesh / SDF 相关文献
- [x] 44 条结构化文献矩阵
- [x] 42 条英文核心 BibTeX

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
- [x] calibration 指标：ECE / MCE / Brier score / NLL / mean confidence / confidence gap
- [x] calibration 指标接入 `metrics_per_case.csv` 与 `summary.json`
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
- [x] 108 个 pytest 全部通过
- [x] Ruff clean
- [x] 4 个前端 JS 语法检查通过
- [x] 42 条 BibTeX 结构检查通过
- [x] PowerShell 脚本语法检查通过
- [x] GitHub Actions CI：PR/push 自动执行 pytest、Ruff、JS、JSON 与 PowerShell 静态检查
- [x] CI 新鲜 runner 可复现获取固定 SegFormer3D `e314242` 并应用受版本控制的 PyTorch 2.1 兼容补丁

---

## 2. 最高优先级待完成任务（P0）

### P0-1 人工 QC

建议负责人：数据/QC 负责人

- [x] 打开 `/qc-review`
- [x] 逐例检查 10 个病例的 orientation
- [x] 检查 spacing
- [x] 检查 label alignment
- [x] 检查 bone-window
- [x] 填 reviewer
- [x] 填 notes（如有）
- [x] 将合格病例设为 pass
- [x] 完成 `manual_qc_review.csv`

> 2026-08-26 已复核 CSV：10/10 orientation/spacing/label alignment/bone-window 均为 `yes`，10/10 `review_status=pass`，reviewer 已填写。人工 QC P0 已解除。

### P0-2 正式任务锁定

建议负责人：模型负责人 + 项目负责人共同确认

- [x] 首个任务确定为 `binary_semantic`
- [x] task_id=`vertebra_binary_ctspine1k_msd_t10_v1`
- [x] foreground labels=`1..25`，训练时统一映射为前景 1
- [x] num_classes=2
- [x] 当前流程 pilot 主数据集=`CTSpine1K/MSD-T10`
- [x] 新建已锁定规格 `configs/task_specs/vertebra_binary_ctspine1k_msd_t10_v1.json`
- [x] `task_locked=true`
- [x] `src.modeling.task_lock` 实测 `ready=true`、0 error/0 warning
- [x] 固定 CPU formal-pilot config：`configs/orthopedic_ct_cpu_binary_formal_pilot_v1.yaml`

> 模板 `vertebra_task_template.json` 继续保留为未锁定模板；真正实验引用新的 locked spec。当前训练链不支持真正的 instance segmentation，不能用 semantic segmentation 冒充。

### P0-3 当前 10 例 formal-pilot patient-level split

建议负责人：数据负责人

- [x] 当前 pilot 固定 10 例：9 个官方 trainset + 1 个官方 test_private
- [x] 固定 train / validation / test = 7 / 2 / 1
- [x] patient-level 防泄漏
- [x] 官方 test_private `liver_169` 只进入 test，不参与训练/调参
- [x] split 标记 `formal_experiment=true`
- [x] 固定 split：`data/splits/ctspine1k_msd_t10_binary_formal_pilot_v1.json`
- [x] `formal_readiness --allow-cpu` 实测 `ready=true`、`blocker_count=0`
- [ ] 最终论文主实验仍需扩大病例规模；当前 10 例只能作为正式流程 pilot，不能代表最终论文样本量

### P0-4 训练算力环境

建议负责人：训练/算力负责人

- [x] 当前笔记本 CPU 工程训练已跑通：Ryzen 7 8745H，8C/16T，约 20 GB RAM
- [x] 真实 36³ patch forward/backward/optimizer step 已验证，约 12.4 s（含进程启动）
- [x] CPU binary engineering 3-epoch pilot 已跑通
- [x] `train.py` 与 `formal_readiness.py` 增加显式 `--allow-cpu`，CPU 不再是绝对 blocker
- [ ] 若后续希望显著缩短完整 3D full-volume 训练/评估时间，再迁移 NVIDIA GPU/服务器

> GPU 现在是效率升级项，不再是方法学硬要求；CPU 正式训练仍必须通过 task/split/QC 等其它 formal 检查。

---

## 3. 正式实验任务（P1）

### P1-1 CT-only baseline

- [x] 10 例 formal-pilot：CPU CT-only 5-epoch 训练完成（7 train / 2 validation）
- [x] formal-pilot 最佳 checkpoint 已保存：epoch 4，patch-val Dice≈0.2719
- [x] formal-pilot 已固定 config / split / seed / environment / run_metadata
- [x] formal-pilot patch validation 已完成；train loss 约 5.5221→2.6524
- [x] 修复 CPU `num_workers=0` 时跨 epoch 反复抽取同一训练 patch：Dataset 新增 `set_epoch()`，训练循环每 epoch 显式更新采样随机流
- [x] validation patch 保持固定采样随机流，不随 epoch 漂移，确保 checkpoint 选择可比较
- [x] CPU ROI 单步 benchmark：36³ / 48³ / 64³ 均完成真实 forward + loss + backward + AdamW.step；3 次单步进程 wall-time 中位数约 1.84 / 1.97 / 2.04 s
- [x] 64³ 在当前 Ryzen 7 + 20 GB RAM 上可承受：单次结束 RSS≈565 MB、进程 peak working set≈1.53 GB；相对 48³ 的额外 wall-time 很小，因此下一版 CT-only baseline 选 64³
- [x] 新建不覆盖历史实验的 `configs/orthopedic_ct_cpu_binary_long_v2.yaml`：CT-only、64³、20 epochs、AdamW、warmup+cosine restart、early stopping patience=8
- [x] `train.py` 新增可靠 `--resume`：每 epoch 保存 `last.pt`，恢复 model/optimizer/scheduler/epoch/best metric/early-stopping/RNG，并在原 run 追加 history
- [x] 真实 64³ resume smoke：同一 run 完成 epoch 1 后从 `last.pt` 续训到 epoch 2，history 保持 1→2 连续
- [x] 基于修复后的 epoch-aware sampling 与 64³ ROI 启动 long-v2 CT-only CPU baseline；按预设 early stopping patience=8 于 epoch 9 正常停止，最佳固定 patch-val Dice≈0.3613（epoch 1），run=`experiments/20260826_162919_cpu_binary_long_v2_ct_only_roi64`
- [ ] 扩大数据规模后的正式 SegFormer3D CT-only 主实验训练
- [x] `evaluate.py` 新增安全 `--case-id`：只能评估当前 validation/test split 内指定病例，用于 CPU 分病例 full-volume 执行；越界病例直接拒绝
- [x] 使用 `liver_7 / liver_8` 分病例 full-volume validation 复核 long-v2 `best.pt` 与 `last.pt`：`best.pt` 平均 Dice≈0.03698，`last.pt` 平均 Dice≈0.04953；`last.pt` 平均 ASSD≈50.78 mm、component count error≈1084，也优于 `best.pt` 的≈56.77 mm / 1617
- [x] 明确发现固定 64³ foreground patch validation 与 full-volume validation 严重不一致：epoch 1 patch-val Dice≈0.3613，但其 full-volume 平均 Dice≈0.037；当前 patch proxy 不能继续作为可靠 checkpoint selector
- [ ] P1 优先修复 checkpoint selection：训练期按可控频率执行 full-volume validation，或保留多个候选 checkpoint 后在 `liver_7/liver_8` 上统一 full-volume 比较；在此之前禁止重新 test `liver_169`
- [x] 已定位首要根因：long-v2 每 epoch 仅 7 个训练 patch，9 epoch 共 63 个；复现实采样后训练 patch 平均前景≈21.2%，而 7 个 train 全卷平均前景仅≈0.68%。两例 validation 预测前景≈14.5%–17.1%，是真值≈0.57%–0.70% 的约 24–27 倍，属于严重 foreground/background sampling prior 失配
- [x] `ProcessedOrthopedicCTDataset` 新增 `patches_per_case`，同病例同 epoch 可产生多个独立可复现 patch；`train.py` 支持 `training.patches_per_case` 并写入 metadata/summary，解决 7 例数据每 epoch 只有 7 次训练 step 的欠采样问题
- [x] `evaluate.py` 新增 `prediction_foreground_fraction`、`target_foreground_fraction`、`prediction_to_target_foreground_ratio`，以后 full-volume evaluation 可直接量化全卷假阳性膨胀
- [x] 新建 `configs/orthopedic_ct_cpu_binary_balanced_fullval_v3.yaml`：foreground_probability=0.25、patches_per_case=4、64³ CT-only、Region Dice+CE 不变，`validation.patch_mode=false`，checkpoint/early stopping 直接依据两例 full-volume validation；`formal_readiness --allow-cpu` 实测 ready=true / blocker_count=0
- [x] balanced fullval v3 已真实完成 epoch 1/2：run=`experiments/20260826_173511_cpu_binary_balanced_fullval_v3_roi64`；epoch 1 train loss≈2.5537、两例 full-volume val Dice≈0.05407，epoch 2 train loss≈1.9402、val Dice≈0.04084；当前 `best.pt=epoch 1`
- [x] 已对 v3 `best.pt` 分别完成 `liver_7/liver_8` detailed full-volume validation：Dice≈0.04323/0.06491，Precision≈0.02753/0.04267，prediction/target foreground ratio≈3.65/3.18；相比 long-v2 的约 24–27 倍前景膨胀已大幅压低，证明 balanced sampling 方向有效
- [x] v3 epoch 3 已按同一 run 续训并触发明确失败：train loss≈1.6316，但两例 full-volume val Dice≈1.3e-11；detailed validation 两例 Dice/Precision/Recall 均为 0，prediction/GT foreground ratio≈0.47/0.26，说明模型已从过预测转为背景塌缩/真实前景无重叠，因此停止机械继续 epoch 4
- [x] 已检查 Region Dice+CE 实现：foreground Dice 与未加权全体素 CrossEntropy 默认 1:1；当前 `train.build_criterion()` 未从 YAML 读取 `dice_weight/ce_weight`，balanced sampling 增加背景后 CE 背景主导与 epoch 3 collapse 现象一致
- [x] 已完成最小 loss-weight 工程修复：`region_dice_ce` 现在从 YAML 读取 `dice_weight/ce_weight`，新增合法性检查与回归测试；v4 配置保持 v3 sampling/ROI/输入/full-volume validation 不变，仅将 CE 权重设为 0.25
- [x] `configs/orthopedic_ct_cpu_binary_balanced_loss_v4.yaml` 已通过 `formal_readiness --allow-cpu`：ready=true / blocker_count=0
- [x] v4 epoch 1 + `liver_7/liver_8` detailed validation 已完成：两例平均 Dice≈0.04762、Precision≈0.02780、foreground ratio≈5.97、component error≈1993；相比 v3 epoch 1（Dice≈0.05407、Precision≈0.03510、ratio≈3.42、component error≈1587.5）整体更差，说明 CE=0.25 过度削弱背景约束，不继续机械跑 v4 epoch 2
- [x] 已新建 `configs/orthopedic_ct_cpu_binary_balanced_lr_v5.yaml`：恢复 Region Dice/CE=1:1，保持 v3 sampling/ROI/full-volume validation 不变，仅将 optimizer peak lr 从 `1e-4` 降到 `5e-5`；`formal_readiness --allow-cpu` 已通过 ready=true / blocker_count=0
- [x] v5 已真实完成 epoch 1/2：run=`experiments/20260826_221337_cpu_binary_balanced_lr_v5_roi64`；epoch 1 train loss≈2.29635 / val Dice≈0.03185 / lr=2.5e-5，epoch 2 train loss≈2.08801 / val Dice≈0.03269 / lr=5e-5；两例 detailed validation 显示 Precision≈0.01621/0.01707、Recall≈0.89485/0.93585、prediction/GT foreground ratio≈55.19/54.82、component error=204/185，属于严重前景泛滥，明显劣于 v3 epoch 1，因此停止继续 v5 epoch 3
- [x] 已新建 `configs/orthopedic_ct_cpu_binary_balanced_lr_v6.yaml`：相对 v5 仅将 `warmup_epochs=2→1`，因此 epoch 1 直接达到 `5e-5`，后续 cosine 学习率始终不超过 `5e-5`；其它 v3 sampling/ROI/loss/full-volume validation 保持不变，`formal_readiness --allow-cpu` 已通过 ready=true / blocker_count=0
- [x] v6 已真实完成 epoch 1/2：run=`experiments/20260826_224150_cpu_binary_balanced_lr_v6_roi64`；epoch 1 train loss=`2.5537127597` / val Dice=`0.0540700072` / lr=`5e-5`，几乎精确复现 v3 epoch 1；epoch 2 train loss=`1.9332212380` / val Dice=`0.0323937293` / lr≈`4.8923e-5`，即使学习率未升到 `1e-4` 仍明显恶化，`best.pt` 保持 epoch 1
- [x] v6 epoch 2 `liver_7/liver_8` detailed validation 已完成：Dice≈`0.03210/0.03268`、Precision≈`0.01632/0.01661`、Recall≈`0.98562/0.99919`、prediction/GT foreground ratio≈`60.40/60.14`，component error=`87/65`；这属于严重全卷前景泛滥，不是正常结构改善，因此停止机械继续 v6 epoch 3
- [x] 已用 Dataset 真实采样逻辑 + 固定 seed=42 复现 v3/v6 epoch 1/2 与 v3 epoch 3 的 28 个 training patch：epoch 1/2/3 mean foreground fraction≈`7.91%/8.84%/5.68%`，纯背景 patch=`18/18/20`；病例级暴露明显不稳定，例如 epoch 1 的 `liver_2/liver_6` 均 4/4 patch 为纯背景。sampling prior 存在真实波动，但 v6 epoch 1→2 总体统计差异不足以单独解释约 3.4×→60× foreground explosion，因此不能把 sampling 写成唯一根因
- [x] `train.py` 已新增真实 `sampling_stats.csv`：每 epoch 记录 patch_count、foreground fraction mean/median/std/min/max、q10/q25/q75/q90、foreground/background patch count；统计来自模型实际收到的训练 label，并新增回归测试
- [x] v7 stable sampling 工程改动已完成：新增 `foreground_sampling_mode=fixed_per_case`，在 4 patches/case、foreground_probability=0.25 下固定每病例每 epoch 1 个 foreground-aware + 3 个 random patch；配置与 v6 对比除实验名外仅新增这一主要实验变量；108 tests + Ruff + `git diff --check` 已通过
- [x] v7 fixed-per-case sampling 已完成 epoch 1 + full-volume validation + `liver_7/liver_8` detailed evaluation；run=`experiments/20260827_000843_cpu_binary_stable_sampling_v7_roi64`，mean val Dice≈`0.04562`，两例平均 Precision≈`0.02605`、foreground ratio≈`6.60`、component error≈`1311.5`，整体劣于 v3/v6 epoch 1，因此按预设规则停止 v7，不机械继续 epoch 2；未访问 `liver_169`
- [x] 已完成 validation-only checkpoint diagnostics 工程：支持 logits/probability 分布、Dice/CE 分项、foreground/background CE contribution、最终 segmentation head 参数/gradient、9 个 BatchNorm3d running statistics、`--bn-mode running|batch` 与固定 foreground-centered 64³ patch backward；full-volume predictor resize 与 training resize 共用同一 helper，diagnostics 不提供 test split 且不执行 optimizer.step；全量 116 tests + Ruff + `git diff --check` 通过
- [x] v6 epoch 1→2 diagnostics 已确认 final head weight/bias 几乎不变，但上游 feature/logit 状态与 BatchNorm running statistics 明显漂移；v6 epoch2 `liver_7` 标准 running-stat inference prediction foreground≈42.26%，临时 batch-stat≈27.98%，GT≈0.70%，因此 BN train/eval normalization mismatch 是 foreground explosion 的重要机制之一但不是唯一机制
- [x] v8 BN-running-stat 单变量工程已完成：`train.py` 新增 `freeze_batchnorm_running_stats`，每 epoch 正常 `model.train()` 后仅把 BatchNorm3d 切到 eval；running_mean/running_var/num_batches_tracked 不更新，BN affine weight/bias 仍保留 gradient，其它模块保持 training，默认旧配置行为不变；v8 相对 v6 仅实验名 + 该选项两处差异；120 tests + Ruff 通过，formal readiness `ready=true / blocker_count=0`，split=7/2/1
- [x] v8 epoch1 + full-volume validation + `liver_7/liver_8` detailed evaluation + running-stat diagnostics 已完成并按规则停止：run=`experiments/20260827_125357_cpu_binary_bn_frozen_v8_roi64`，train loss≈`6.01818`，mean val Dice≈`0.0001248`；`liver_7/8` prediction foreground≈`0.0579%/0.0709%`，GT≈`0.6996%/0.5660%`，Dice≈`0.0002495/0`，属于严重背景塌缩；9 个 BN 的 `num_batches_tracked=0`、首层 running mean std=`0`、running var mean=`1`，证明冻结确实生效，但从初始化就固定 BN stats 不是稳定 baseline 方案，因此不跑 epoch2
- [x] 已新增 validation-only `compare_checkpoint_dynamics.py`：固定前景中心 64³ patch，记录 encoder 四级 embedding/block、decoder fuse、head input/output 的 mean/std/min/max/quantiles/L2 norm，并比较 checkpoint parameter-group delta、top parameter delta 与 BN running-buffer delta；不提供 test split，不执行 optimizer.step；focused 2 tests + Ruff 通过
- [x] 已运行 v6 epoch1 / v6 epoch2 / v8 epoch1 checkpoint dynamics：固定 `liver_7` foreground patch 上，v6 epoch1→epoch2 普通参数组相对变化整体很小（最大聚合组约 `0.66%`），但多处 BN running_mean 相对变化约 `1.0×–3.3×`、running_var 约 `66%–79%`，decoder/head-input activation 同时明显漂移；证据支持把 epoch1 BN stats 作为锚点验证，而不是继续从初始化冻结
- [x] v9 工程完成：新增 `configs/orthopedic_ct_cpu_binary_bn_freeze_after_e1_v9.yaml` 与 epoch-aware BN freeze；epoch1 保持 v6 原行为，epoch2 起冻结 running_mean/running_var/num_batches_tracked，BN affine 与其它模型参数继续训练；resume 到 epoch2 会按 epoch 自动冻结。v9/v6 config 归一化对比仅实验名 + `freeze_batchnorm_running_stats_from_epoch=2` 不同；125 tests + Ruff + `git diff --check` 通过，formal readiness `ready=true / blocker_count=0`，split=7/2/1
- [x] v9 epoch1 已精确复现 v6 epoch1：run=`experiments/20260827_132502_cpu_binary_bn_freeze_after_e1_v9_roi64`，train loss=`2.5537127597`、mean val Dice=`0.0540700072`、lr=`5e-5`；detailed validation `liver_7/8` Dice≈`0.04323/0.06491`、foreground ratio≈`3.65/3.18`；diagnostics 显示 9 个 BN 均 `num_batches_tracked=28`，首层 running mean std≈`0.0144923`、running var mean≈`0.0622548`，与 v6 epoch1 锚点一致，说明 v9 实现没有污染 epoch1
- [x] v9 已 resume 到 epoch2：train loss=`2.6975343355`、mean val Dice=`0.0267784339`、lr≈`4.8923e-5`；`best.pt` 仍为 epoch1，`last.pt` 为 epoch2；checkpoint 逐项复核 27/27 个 BN `running_mean/running_var/num_batches_tracked` 从 epoch1→epoch2 完全不变（changed=0）
- [x] v9 epoch2 `liver_7/liver_8` detailed validation 已完成：Dice≈`0.02899/0.02457`、Precision≈`0.01638/0.01367`、Recall≈`0.12594/0.12099`、prediction/GT foreground ratio≈`7.69/8.85`，prediction foreground≈`5.38%/5.01%`；相比 v6 epoch2 的≈`42.26%/34.04%` 与≈`60×`，foreground explosion 被明显缓解，但 Dice、Precision 与碎片化仍明显不稳定（两例 pred components 均为 `517`）
- [x] v9 epoch2 diagnostics 已完成：两例 GT foreground mean P(fg)≈`0.13278/0.12832`、GT background mean P(fg)≈`0.05583/0.05242`；foreground/background weighted CE contribution≈`0.04732/0.30318` 与 `0.03511/0.36757`；首层 BN 仍为 epoch1 锚点 `num_batches_tracked=28`、running mean std≈`0.0144923`、running var mean≈`0.0622548`
- [x] 已运行 v6e1/v6e2/v9e1/v9e2 checkpoint dynamics：v9e1 与 v6e1 参数和 BN running buffers 全部精确一致；v9e2 的 BN running buffers 对 v6e1 仍全为 relative delta=`0`，但普通 trainable 参数仍发生小幅更新（最大聚合组 encoder embed4≈`0.658%`，head≈`0.0179%`），固定 patch final logits 从 v6/v9 epoch1 mean≈`-11.31` 漂移到 v9e2≈`-4.56`。因此 BN drift 是 v6 foreground explosion 的重要放大机制，但不是 epoch2 segmentation degradation 的唯一根因
- [x] 按预设规则停止 v9，不跑 epoch3；stable baseline 仍为 NO，锁参/正式 test 条件仍未满足，独立 test `liver_169` 继续禁止访问
- [x] 已基于 v9 dynamics 进一步拆分 parameter delta：v9e1→v9e2 的 BN affine≈`0.0194%`、LayerNorm affine≈`0.0208%`，而 patch embeddings≈`0.7842%`、encoder attention≈`0.7075%`、encoder MLP≈`0.1906%`；final head 仅≈`0.0179%`，但固定 patch final logits mean 仍从≈`-11.31` 漂移到≈`-4.56`。因此 v10 不优先冻结 BN affine，而选择更有证据的单变量“epoch2 起冻结 encoder trainable parameters”
- [x] v10 工程完成：新增 `configs/orthopedic_ct_cpu_binary_encoder_freeze_after_e1_v10.yaml` 与 epoch-aware encoder freeze；相对 v9 仅实验名 + `freeze_encoder_parameters_from_epoch=2` 不同，保留 v9 的 `freeze_batchnorm_running_stats_from_epoch=2`、lr/loss/sampling/ROI/augmentation/decoder/head 全部不变；新增 policy、encoder-only gradient、恢复 trainability、v10/v9 config diff 回归测试；129 tests + Ruff + `git diff --check` 通过，formal readiness `ready=true / blocker_count=0`
- [x] v10 epoch1 已真实完成并精确复现 v6/v9 epoch1：run=`experiments/20260827_170359_cpu_binary_encoder_freeze_after_e1_v10_roi64`，train loss=`2.5537127597`、mean val Dice=`0.0540700072`、std=`0.0108403799`、lr=`5e-5`；`liver_7/liver_8` detailed Dice≈`0.04323/0.06491`、foreground ratio≈`3.65/3.18`。两例 diagnostics 已完成；v6e1/v9e1/v10e1 的 232 个 model state tensor 逐项 `torch.equal`，diff tensor=`0`、BN buffer diff=`0`，因此允许从 v10 `last.pt` resume 到 epoch2
- [x] v10 已 resume 到 epoch2 并完成完整验证：train loss=`2.4052223137`、mean val Dice=`1.6539e-11`，两例 detailed Dice/Precision/Recall 均为 `0`，prediction/GT foreground ratio≈`0.073/0.064`，属于灾难性 background collapse。checkpoint 证明 encoder 208 个 state tensor changed=`0`、27 个 BN running buffer changed=`0`，而 decoder 24 个 tensor 中 21 个更新，final head 也更新；fixed patch 所有 encoder activation 与 epoch1 完全一致，而 decoder fuse/final logits 明显漂移。因此 v10 STOP，不跑 epoch3
- [x] v11 工程完成：新增 `freeze_decoder_feature_parameters_from_epoch=2`，epoch2 起冻结 decoder feature parameters（`linear_c1..c4` + `linear_fuse`），仅保留最终 `linear_pred` segmentation head 可训练；继续保持 encoder 与 BN running stats 冻结。v11 相对 v10 仅实验名 + 这一新增主要变量不同；focused freeze tests=`15 passed`，全量 `pytest=133 passed`、Ruff clean、`formal_readiness --task-spec ... --allow-cpu`=`ready=true / blocker_count=0`
- [x] v11 epoch1 已真实完成并精确复现 v10/v9/v6 epoch1：run=`experiments/20260827_180730_cpu_binary_decoder_feature_freeze_after_e1_v11_roi64`，train loss=`2.5537127597`、mean val Dice=`0.0540700072`、std=`0.0108403799`、lr=`5e-5`；sampling 28 patch、foreground/background=`10/18`、foreground fraction mean≈`0.07907336`。v10e1↔v11e1 共 232 个 model-state tensor 逐项 `torch.equal`，diff tensor=`0`，确认 v11 延迟 freeze 工程未污染 epoch1；不重复昂贵 detailed evaluation，直接复用 exact-equal 的 v10e1 validation/diagnostics 锚点
- [x] v11 已从同一 run resume 到总 epoch2：train loss=`2.3053811001`、mean full-volume val Dice=`0.0543761681`（epoch1=`0.0540700072`，未下降），sampling 28 patch、foreground/background=`10/18`、foreground fraction mean≈`0.08840765`。两例 detailed validation Dice≈`0.04421/0.06454`、prediction/GT foreground ratio≈`3.96/3.50`，没有复现 v10 background collapse。checkpoint 严格验证 encoder state delta=`0`、BN running buffer delta=`0`、decoder feature delta=`0`，仅 `linear_pred` weight+bias 更新。validation-only diagnostics 显示 `liver_7/8` GT foreground mean P(fg)≈`0.13263/0.16114`、GT background mean P(fg)≈`0.03482/0.02670`，未见 foreground explosion/background collapse；固定 `liver_7` dynamics 中全部 encoder activation、`linear_fuse` 与 final-head input 统计完全一致，仅 final logits 随 head 更新而变化。该证据进一步支持 decoder feature update 是 v10 catastrophic collapse 的关键机制之一，但不能写成唯一根因
- [x] v11 epoch3 已真实完成且未重跑：train loss=`1.8300107228`、mean full-volume val Dice=`0.0546575740`、std=`0.0095167619`、lr≈`4.57984e-5`；三轮 Dice=`0.05407001 → 0.05437617 → 0.05465757`。epoch3 sampling 28 patch、foreground/background=`8/20`、foreground fraction mean≈`0.05680016`。`liver_7/liver_8` detailed Dice≈`0.04514/0.06417`、Precision≈`0.02792/0.04058`、Recall≈`0.11773/0.15335`、prediction/GT foreground ratio≈`4.22/3.78`，没有 foreground explosion、background collapse 或 catastrophic Dice drop。由于 epoch2 checkpoint 已被 epoch3 覆盖，使用已证明与 v11e1 exact equal 的 v10e1 anchor 对 v11e3 新做 validation-only checkpoint dynamics，并与现有 v11e1→v11e2 dynamics 交叉验证：encoder、BN running buffers、decoder feature 从 anchor 到 epoch3 全部 delta=`0`，固定 patch encoder/fuse/head-input activation exact equal，仅 `linear_pred` 与 final logits 改变；`linear_pred` group delta norm 对同一 anchor 从 epoch2=`0.0100338` 增至 epoch3=`0.0189645`，证明 epoch2→epoch3 final head 继续更新。正式判定 stable baseline=`YES`（engineering/validation），但 lock parameters=`NO`、formal independent test ready=`NO`，本轮未访问 `liver_169`
- [x] v11 baseline reproducibility 已完成最小可信验证：固定同一 v11 epoch3 `best.pt` + 同一 config，分别重新 full-volume evaluation `liver_7/liver_8`；除 `inference_seconds` 外所有 summary metric 的 mean 逐项 exact equal，Dice 仍为 `0.04514081209537846/0.06417433592551017`。config SHA-256=`6898924e3b1dbf9d60d501b252ebc44fe5411d5ec1f967efda06f11355548ae9`，checkpoint SHA-256=`9a805bc9c97b96128ba0b63d84dc30e113bade227f0a3a1cbd524231da896d67`。该结果证明 checkpoint/full-volume inference+evaluation 可复现；由于完整 3-epoch CPU 重训成本较高，本阶段没有声称完成随机初始化后的完整训练轨迹复现
- [x] formal-pilot 独立 full-volume test：`liver_169`，从未参与训练/调参
- [x] formal-pilot 输出 `metrics_per_case.csv` + prediction NIfTI + entropy NIfTI
- [x] formal-pilot 记录 Dice / HD95 / ASSD / IoU / Precision / Recall / 结构指标
- [x] formal-pilot 单病例 CPU full-volume inference≈9.29 s
- [x] formal-pilot test 真实指标：Dice≈0.0221、IoU≈0.0112、Precision≈0.0116、Recall≈0.2479、HD95≈190.93 mm、ASSD≈53.42 mm、component_count_error=157、false_merge=0、false_break=15
- [x] formal-pilot test uncertainty/calibration 已生成：AUROC≈0.6137、AUPRC≈0.4395、ECE≈0.2990、MCE≈0.3378、Brier≈0.6347、NLL≈2.1329
- [ ] **5 epoch 模型严重欠训练；上述单病例 test 只证明完整科研评估链真实跑通，不能作为论文正式结果**
- [ ] 扩大数据规模后重新报告正式主实验 test 指标

### P1-2 输入消融

- [x] CT only：v11 stable baseline，3 epoch mean full-volume validation Dice=`0.05407001 → 0.05437617 → 0.05465757`
- [x] CT + bone-window：v12，唯一主要变量为 `input_channels: [ct_normalized] → [ct_normalized, bone_window]` 且 `model.in_channels: 1 → 2`；bone window=`500/2000`
- [x] 比较区域指标：v11 两例平均 Dice/IoU/Precision≈`0.05466/0.02812/0.03425`，v12≈`0.02803/0.01421/0.01422`；v12 Recall 虽升至≈`0.96594`，但属于严重前景泛滥
- [x] 比较表面指标：v11 两例平均 HD95/ASSD≈`186.05/51.52 mm`，v12≈`256.41/83.77 mm`，CT-only 更好
- [x] 比较结构/前景/校准：v11 prediction/GT foreground ratio≈`4.00×`，v12≈`67.96×`；v11 ECE/Brier/NLL≈`0.01084/0.04693/0.10288`，v12≈`0.39319/0.80051/3.40655`
- [x] 比较速度：当前 CPU 两例 wall-clock 平均 v11≈`72.88 s`，v12≈`100.22 s`；仅作当前机器参考，不写成跨硬件结论
- [x] v12 freeze/checkpoint verification：epoch3 checkpoint AdamW step 计数显示 encoder 184 个参数与 decoder feature 19 个参数均停在 `28`，仅 `linear_pred` 2 个参数到 `84`；9 个 BN `num_batches_tracked=28`，与 epoch2 起 encoder/BN/decoder-feature freeze 策略一致
- [x] 输入消融最终判定：**CT-only（v11）胜出**；CT+bone-window 在当前 normalization/architecture 下导致约 `68×` foreground overprediction，停止该方向，不访问独立 test `liver_169`

### P1-3 联合损失消融

- [x] Region：复用 v11 CT-only stable baseline，epoch3 两例平均 Dice=`0.05465757`、HD95/ASSD=`186.0500/51.5220 mm`
- [x] Region + Boundary：v13，boundary weight=`0.1`，3 epoch mean val Dice=`0.05414399 → 0.05443118 → 0.05470944`；epoch3 两例平均 HD95/ASSD=`185.9498/51.4865 mm`
- [x] Region + Topology：v14，topology weight=`0.1`、iterations=`10`，3 epoch mean val Dice=`0.05450790 → 0.05450464 → 0.05450933`；epoch3 两例平均 HD95/ASSD=`183.9914/50.5799 mm`
- [x] Region + Boundary + Topology：v15，region/boundary/topology=`1.0/0.1/0.1`，3 epoch mean val Dice=`0.05451937 → 0.05452095 → 0.05447708`；best=`epoch2`，两例平均 HD95/ASSD=`184.5759/50.7512 mm`、foreground ratio=`4.8108×`、component error=`1540`、false break=`60`
- [ ] loss 权重 validation grid（最小四组 loss ablation 已闭环；是否继续细网格以后续 validation 证据与 CPU 成本决定）
- [x] Boundary 对 HD95/ASSD 的影响：相对 Region，HD95 仅改善约 `0.1002 mm`、ASSD 仅改善约 `0.0355 mm`，Dice 提升约 `5.19e-5`；Precision 略降、false break 略增、calibration 略差，因此判定为**轻微但很弱的改善证据，尚不足以称为明确收益**
- [x] Topology 对 false merge / false break 的影响：v14 相对 Region，平均 component error=`1548→1540`、false break=`64→62`、false merge=`0.5→0.5`，HD95/ASSD 改善约 `2.0586/0.9421 mm`；v15 false break 进一步到 `60`，但 v14/v15 都伴随更强 foreground overprediction 与更差 calibration，因此仍判定为**结构/表面有一定改善，但总体证据不充分（evidence inconclusive）**
- [x] loss ablation 最终决策：选择 **v13 Region+Boundary** 作为 sampling baseline；它在四组里两例平均 Dice/IoU 最高，foreground ratio≈`4.027×`、calibration 基本维持 Region 水平，且表面指标未恶化；不把极弱 Boundary 增益夸大为显著效果
- [ ] 骨折/真实断裂病例单独检查

### P1-4 Sampling / 困难样本实验

- [x] current sampling baseline：v13 Bernoulli，foreground_probability=`0.25`、patches_per_case=`4`
- [x] fixed-per-case sampling：v16 已完成 3 epoch + liver_7/liver_8 detailed validation；sampling 跨 epoch 更稳定（foreground-fraction mean std≈`0.003642`），但 mean Dice≈`0.04575`、foreground ratio≈`6.51×`、ECE≈`0.02964`，整体劣于 v13，因此不选
- [x] boundary hard sampling：v17 已完成 3 epoch + liver_7/liver_8 detailed validation；mean Dice≈`0.03731`、foreground ratio≈`36.26×`、HD95≈`206.50 mm`、ECE≈`0.19569`，虽 component error/false break 更低但伴随严重 foreground overprediction 与 calibration 崩坏，STOP，不选
- [x] sampling ablation 最终决策：保留 **v13 Bernoulli**（foreground_probability=`0.25`、patches_per_case=`4`）作为 augmentation / difficult-sample baseline；不能只按 sampling stability 选型
- [x] standard augmentation baseline：v18（±10° rotation、scale 0.9–1.1、p=0.5）完成 3 epoch + liver_7/liver_8 detailed validation；mean Dice=`0.05535` 虽略高于 v13，但 foreground ratio=`7.56×`、Precision=`0.03134`、ECE=`0.02716` 明显劣化，综合判定不选，后续保留 v13 原始 flip-only geometric baseline
- [x] + boundary hard sampling：v17 已作为 sampling 单变量验证并失败，不在 augmentation 阶段重复
- [x] + intensity/HU augmentation：v19 gamma 失败后 STOP；v20 Gaussian noise 完成 3 epoch 与两例 detailed validation，但综合指标不取代 v13；v21 HU shift 在 epoch2 后明显失败并 STOP；augmentation 最终保留 **v13 flip-only baseline**
- [x] high-loss hard mining：v22 唯一有效 run 完成 epoch1/2，mean val Dice=`0.04728839 → 0.04728457`，持续低于 v13=`0.05470944`；FAIL / STOP，不跑 epoch3
- [x] high-uncertainty mining：v23 有效 run 完成 epoch1/2，mean val Dice=`0.00963475 → 0.01010232`，远低于 v13；FAIL / STOP，不跑 epoch3
- [x] thick-slice data-evidence：train `liver_0/liver_1` z-spacing≈`5 mm`，其余 train 多为≈`0.8–1.0 mm`；冻结 v13 guidance 显示两例 thick-slice case 的 candidate loss/uncertainty 分别高约 `12.9%/13.0%`。validation `liver_7/liver_8` 均为 1 mm，因此 validation thick-slice subgroup 数据不足，禁止伪造 subgroup Dice
- [ ] metal artifact subset：当前 metadata / 病例记录不足以可靠标记具体病例，数据不足
- [ ] fracture subset：当前 metadata / 病例记录不足以可靠标记具体病例，数据不足
- [ ] low-density subset：当前 metadata / 病例记录不足以可靠标记具体病例，数据不足
- [x] difficult-sample 最终决策：boundary-hard、high-loss、high-uncertainty 均不取代 baseline；正式继续保留 **v13 Bernoulli**（foreground_probability=`0.25`、patches_per_case=`4`），下一阶段进入 ROI refinement

---

## 4. 不确定性与精修任务（P2）

- [x] formal-pilot 真实 checkpoint 已生成 entropy NIfTI
- [x] formal-pilot uncertainty→error AUROC≈0.6137 / AUPRC≈0.4395
- [x] formal-pilot Top-10% error recall≈0.1385，ROI error rate≈0.4909
- [ ] ROI threshold / percentile validation
- [x] calibration 指标工程实现与评估输出接入
- [x] formal-pilot test 已计算 ECE≈0.2990 / Brier≈0.6347 / NLL≈2.1329 / confidence gap≈0.2990
- [x] v13 validation `liver_7/liver_8` uncertainty/calibration 两例稳定性分析：AUROC=`0.92949/0.94466`、AUPRC=`0.33354/0.33014`、Top-10% error recall=`0.72891/0.79450`；ECE=`0.01418/0.00767`、Brier=`0.05465/0.03969`、NLL=`0.12079/0.08571`；error entropy 约为 correct entropy 的 `10.41×/12.42×`。支持 uncertainty 作为 validation error indicator/QC/refinement trigger，但仅 2 例且 Dice 很低，不能宣称总体稳定或前景校准完成
- [ ] 扩大 validation 病例后复核 reliability / calibration 稳定性
- [ ] coarse baseline
- [ ] full-volume second pass 对照
- [ ] uncertainty ROI refinement
- [ ] 比较 Dice / HD95 / ASSD
- [ ] 比较 ROI error
- [ ] 比较推理时间
- [ ] 比较显存
- [x] 评估 uncertainty 是否能作为 QC 信号：当前两例 validation 均支持作为高风险区域提示信号（AUROC>0.92、Top-10% 覆盖约 73%–79% 错误），但仍需更多病例验证

---

## 5. 三维与 Web 正式结果任务（P3）

- [ ] prediction mask → physical mesh
- [ ] prediction mesh vs GT surface
- [ ] prediction surface HD95 / ASSD
- [ ] SDF prediction surface 验证
- [ ] 简化误差评估
- [x] 曲率/关键边缘保护候选：法向变化加权 vertex-clustering（真值网格工程验证，待 prediction 验证）
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
