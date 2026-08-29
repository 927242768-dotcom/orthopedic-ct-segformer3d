# 最终独立测试参数锁定记录

锁定日期：2026-08-29

## 锁定状态

- lock parameters = YES
- formal independent test ready = YES
- validation 阶段远端基线提交：`2f333ba39c155be11e277591c5d99eb40882feb3`
- 该提交已 push，并在锁参前确认 `HEAD == origin/main`。
- 从本记录提交并 push 后开始，**不得依据 independent test 结果调整任何模型参数、threshold、sampling、augmentation、refinement 或 inference decision**。

## 最终固定 pipeline

- architecture：SegFormer3D v13
- config：`configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml`
- checkpoint：`experiments/20260828_002035_cpu_binary_loss_region_boundary_v13_roi64/checkpoint/best.pt`
- task：binary semantic vertebra segmentation
- input：CT-only，`ct_normalized`
- normalization：HU clip `[-1000, 2000]` 后逐病例 z-score；目标体素间距 `1.0 × 1.0 × 1.0 mm`
- model input channels：1
- loss：Region + Boundary
  - region weight = `1.0`
  - boundary weight = `0.1`
  - topology weight = `0.0`
- training ROI：`64 × 64 × 64`
- optimizer：AdamW
  - lr = `5e-5`
  - weight decay = `0.01`
- scheduler：warmup 1 epoch + cosine annealing warm restarts
- sampling：Bernoulli foreground-biased sampling
  - foreground probability = `0.25`
  - patches/case = `4`
- augmentation：仅保留 v13 原有 flip-only augmentation；不启用 v18/v19/v20/v21 的额外增强方案
- freeze policy：epoch 2 起冻结 encoder、decoder feature 和 BatchNorm running statistics，仅允许最终 `linear_pred` 更新
- validation/inference：full-volume sliding-window
  - inference ROI = `128 × 128 × 128`
  - overlap = `0.25`
  - sw batch size = `1`
- prediction decision：保持 `evaluate.py` 当前 softmax + argmax class decision；**不做 test 后 threshold 调参**
- refinement：DISABLED
  - 原因：validation 综合判定 `REFINEMENT=FAIL`

## 锁参依据

最终选择完全由 `liver_7/liver_8` validation 与既有工程消融确定，包括输入、loss、sampling、augmentation、difficult-sample、uncertainty/calibration、refinement、prediction 3D/SDF 与 Web 实机验收。validation mean Dice 仍约 `0.05470944`，必须如实保留低性能事实。

在本锁参记录产生前，正式独立病例 `ctspine1k-msd-t10-liver_169` 仍保持隔离；历史 pilot 结果不作为本次正式 independent test。

## 独立测试规则

锁参提交成功 push 且再次确认 `HEAD == origin/main` 后，才允许第一次正式访问 `ctspine1k-msd-t10-liver_169`，并且只运行一次 FINAL FORMAL INDEPENDENT TEST。必须保存区域、表面、结构、不确定性、校准、推理时间、prediction/entropy NIfTI，以及 independent prediction mesh / 2 mm simplification / 0.4 mm SDF 工程结果。

无论 independent test 结果好坏，都不得重新调参或重复测试，只能进入 Results / Discussion / Failure Cases / Limitations / Conclusion 和最终材料收尾。
