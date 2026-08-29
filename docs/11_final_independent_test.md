# 最终正式独立测试记录

日期：2026-08-29

## 1. 测试门禁

- validation 阶段提交：`2f333ba39c155be11e277591c5d99eb40882feb3`
- 最终锁参提交：`eb0a824c34af4f7d900432e169759115f99a2687`
- 锁参记录：`docs/10_final_parameter_lock.md`
- 正式测试开始前已确认：`HEAD == origin/main == eb0a824c34af4f7d900432e169759115f99a2687`
- Git Author / Committer：`927242768-dotcom <927242768@qq.com>`
- formal preflight：`ready=true`，0 error，0 warning
- 正式病例：`ctspine1k-msd-t10-liver_169`
- source split：`test_private`
- 该病例在锁参前未用于训练、validation、消融或参数选择。
- 本次为最终锁定 v13 协议下第一次且唯一一次 FINAL FORMAL INDEPENDENT TEST；仓库中更早的 5-epoch pilot evaluation 仅为历史工程链证据，不属于本次最终正式测试，也未用于 v13 参数选择；测试后禁止依据结果重新调参或重复最终正式测试。

## 2. 固定 pipeline

- architecture：SegFormer3D v13
- config：`configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml`
- checkpoint：`experiments/20260828_002035_cpu_binary_loss_region_boundary_v13_roi64/checkpoint/best.pt`
- input：CT-only，HU clip `[-1000,2000]` 后逐病例 z-score
- spacing：1 mm isotropic
- loss：Region + Boundary = `1.0 / 0.1`，Topology=`0`
- sampling：Bernoulli，foreground probability=`0.25`，patches/case=`4`
- augmentation：flip-only
- training ROI：`64×64×64`
- inference ROI：`128×128×128`，overlap=`0.25`
- decision：`evaluate.py` softmax + argmax
- refinement：disabled（validation 综合判定 `REFINEMENT=FAIL`）

## 3. 正式 independent-test 分割结果

输出目录：`experiments/final_independent_test_20260829_v13_locked_liver169`

| 指标 | 结果 |
|---|---:|
| Dice | 0.02878288 |
| IoU | 0.01460158 |
| Precision | 0.02089816 |
| Recall | 0.04622219 |
| HD95 | 136.8722 mm |
| ASSD | 43.97199 mm |
| Prediction foreground fraction | 0.03567316 |
| Target foreground fraction | 0.01612869 |
| Prediction / target foreground ratio | 2.21178× |
| Prediction components | 236 |
| Target components | 1 |
| Component count error | 235 |
| False merge | 0 |
| False break | 29 |
| CPU inference time | 9.41277 s |

该结果必须如实解释为当前锁定模型绝对分割精度低，且存在明显前景过预测与结构碎片化；不得因为部分 surface 指标数值低于 validation 均值就宣称独立测试成功。

## 4. Uncertainty 与 calibration

| 指标 | 结果 |
|---|---:|
| Error AUROC | 0.86424490 |
| Error AUPRC | 0.29665454 |
| Top-10% uncertainty error recall | 0.54993443 |
| ECE | 0.02739661 |
| MCE | 0.08781999 |
| Brier | 0.08328483 |
| NLL | 0.23558760 |
| Confidence gap | 0.02739661 |

Predictive entropy 在独立病例上仍具有一定错误排序能力，但弱于两例 validation 的 AUROC > 0.92；Top-10% error recall 也下降到约 55%。较低 ECE 不能脱离低 Dice 和高 background 占比解释，因此不视为临床可靠性证据。

## 5. Independent prediction 三维工程结果

### 原始 physical-space mesh

- vertices：365,247
- faces：724,694
- surface area：278,464.75 mm²
- prediction / GT 的 size、spacing、origin、direction 全部一致

### 2.0 mm feature-weighted vertex clustering

- feature preservation strength：8
- vertices：81,353
- faces：160,384
- vertex reduction：77.73%
- face reduction：77.87%
- surface-area relative change：-8.92%
- full-vs-simplified vertex-nearest engineering ASSD：0.56490 mm
- full-vs-simplified vertex-nearest engineering HD95：1.07159 mm

### SDF σ=0.4 mm

- components：236 → 236
- component preserved：true
- vertices：365,200
- faces：724,322
- SDF-vs-original engineering ASSD：0.02536 mm
- SDF-vs-original engineering HD95：0.06367 mm

### Prediction vs GT surface 工程对照

- prediction vertices：365,247
- GT vertices：110,619
- vertex-nearest engineering ASSD：41.1398 mm
- vertex-nearest engineering HD95：131.8726 mm

以上 vertex-nearest 指标只用于三维工程误差与表面链路检查，**不替代 `evaluate.py` 输出的正式 segmentation HD95 / ASSD**。

## 6. Web 实机验收

- `/api/research/evaluations` 已识别 `final_independent_test_20260829_v13_locked_liver169`。
- `results-review`：independent prediction MPR API 返回 200。
- `results-review`：independent uncertainty MPR API 返回 200。
- `research-3d`：Edge WebGL2 已真实加载 2.0 mm independent prediction mesh（81,353 顶点 / 160,384 面）。
- `research-3d`：Edge WebGL2 已真实加载 SDF σ=0.4 mm independent surface（365,200 顶点 / 724,322 面）。
- Web 只读取磁盘中本次正式 evaluation 的真实 prediction / entropy / mesh，不重新执行模型推理。

## 7. 最终科研判断

正式 independent test 未改变参数锁定结论，也不触发任何新的调参。当前项目应定位为：

**骨科 CT 分割方法探索 + uncertainty/calibration + physical-space 3D + Web 科研复核的完整可追溯工程闭环。**

当前模型不能描述为高精度临床模型。后续若继续研究，应扩大数据规模、引入强 baseline 和合法临床/多中心验证，并建立新的预注册 validation/test 方案；不得重复使用本次 `liver_169` 结果进行参数选择。
