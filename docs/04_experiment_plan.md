# 04 模型训练、调参与消融实验计划

> 原则：先得到可靠 baseline，再逐项加入创新模块。任何论文结论必须来自可复现实验，不允许一次把所有模块叠加后只报一个最终数字。

## 1. 实验目标

核心问题：

1. SegFormer3D 在骨科 CT 上能否形成稳定 baseline？
2. 骨窗/多通道输入是否提升骨边界与泛化？
3. 区域 + 边界 + 拓扑联合损失是否优于区域损失？
4. 困难样本增强是否真正改善困难病例，而非只改善平均指标？
5. 不确定性精修是否在可接受的额外计算量下提升边界/连通性？

## 2. 数据划分

必须按患者划分：

```text
train / validation / test
```

推荐策略：

- 若数据规模足够：70/15/15 或 80/10/10；
- 若数据较小：5-fold cross validation + 独立外部测试；
- 如果存在多个数据来源：额外设计 source-out 验证。

最终比例以数据集官方划分优先，避免为了方便破坏官方 benchmark。

### 2.1 当前真实工程验证（非论文实验）

2026-08-16 已取得 CTSpine1K `MSD-T10` 10 例真实 CT+label，并按 pipeline 0.3.0 完成 1 mm 标准化与自动 QC：9 例官方 `trainset`、1 例 `test_private`。10/10 自动审计通过，人工审核表已生成但尚未签字。

同时已用真实 `liver_0` 完成双通道 CT+bone-window、困难 patch 采样、SegFormer3D、joint loss、backward、AdamW.step 的 36³ CPU smoke test。该测试仅证明真实数据训练链可运行，**随机权重 loss、梯度或由 engineering smoke split 得到的任何数字均不得进入论文 Results**。

当前工程 smoke split：

```text
data/splits/ctspine1k_msd_t10_engineering_smoke.json
formal_experiment = false
```

正式 baseline 前仍必须由组内确定标签定义和正式 split，并迁移到可用 NVIDIA GPU 环境。当前 `configs/label_schemas/ctspine1k_verse.json` 只把 `1–25` 显示为 `C1–L6` 以方便 QC/Web，不代表已经决定正式任务是 binary、multi-class semantic 或 instance segmentation。

### 2.2 正式运行 preflight

`src/modeling/preflight.py` 已作为保护层接入 `train.py` / `evaluate.py`。正式模式必须在真正加载模型和长时间训练前检查：

- split 是否明确允许 formal experiment；
- train/validation/test 是否病例级互斥；
- 官方 `test_private` 是否误进入 train/validation；
- 所有 split 病例是否完成规定的人工 QC；
- pipeline version / 输入通道 / 标签值 / `label_mode` / `num_classes` 是否一致；
- 正式训练是否存在可用 CUDA GPU。

工程调试只能显式选择 `--preflight-mode engineering`；`--skip-preflight` 仅允许定位代码问题，不得用于论文正式 run。

## 3. Baseline 定义

### B0：数据处理基线

- 统一 orientation；
- 固定 spacing；
- HU clip + normalize；
- 无额外困难样本增强。

### B1：SegFormer3D 基线

- 原始 SegFormer3D 主结构；
- 单 CT 通道；
- DiceCE；
- 标准 augmentation；
- sliding-window inference。

### B2：骨窗输入

- 输入：标准 CT + bone-window channel；
- 其他配置与 B1 完全相同。

目的：隔离“骨窗输入”效果。

## 4. 联合损失消融

为了能解释贡献，至少完成：

| 实验 | Region | Boundary | Topology | 目的 |
|---|---|---|---|---|
| L0 | ✅ | ❌ | ❌ | 基础区域损失 |
| L1 | ✅ | ✅ | ❌ | 验证边界项 |
| L2 | ✅ | ❌ | ✅ | 验证拓扑项 |
| L3 | ✅ | ✅ | ✅ | 完整联合损失 |

如果 L2/L3 拓扑项在目标骨结构上无收益，应如实报告并更换更合适的拓扑定义，不能为满足任务书强行保留无效模块。

### 4.1 Loss 权重

第一阶段粗网格：

```text
λ_region = 1.0
λ_boundary ∈ {0.05, 0.1, 0.2, 0.5}
λ_topology ∈ {0.05, 0.1, 0.2, 0.5}
```

只在 validation 上确定，不使用 test 选权重。

## 5. 困难样本增强消融

### H0

普通几何/强度增强。

### H1

加入 CT 强度域增强：

- contrast/gamma；
- noise；
- blur；
- HU shift/scale。

### H2

加入困难病例采样：

- 训练历史高 loss；
- 边界误差大；
- 高 uncertainty。

### H3

若有足够真实金属内固定病例，再评估 metal artifact simulation/augmentation。

结果必须单独统计：

- normal cases；
- fracture；
- low-density；
- metal artifact；
- thick-slice。

如果公开数据不提供病例属性，至少通过人工/规则建立困难子集，记录定义。

## 6. 不确定性精修实验

### U0

普通单次推理。

### U1：Entropy-only

从 softmax/sigmoid 生成 uncertainty map，并做错误相关性定量分析。当前评价代码已实现：

- uncertainty→error AUROC；
- uncertainty→error AUPRC；
- error / correct 体素平均 entropy；
- Top-percent 高不确定区域对错误的 recall；
- ROI error rate 与 ROI fraction。

这些指标必须先证明 entropy 确实能定位错误，再进入 U2；否则不应仅凭热图宣称 uncertainty 有效。

### U2：Uncertainty ROI refinement

流程：

```text
coarse prediction
→ uncertainty map
→ 选取高不确定区域
→ ROI 扩张
→ ROI-only residual refinement
→ 融合
```

工程基线已经实现二阶段训练闭环：coarse logits 默认冻结，refinement loss 只在 ROI 内归一化，ROI 外 logits/prediction 保持不变；训练 step 同时记录 refinement 前后 ROI/global error delta。该实现目前只有单元/工程验证，必须等待真实 baseline checkpoint 后才能判断是否带来性能收益。

### U3：全图二次推理对照

目的：证明收益来自“定位困难区域”，而不是单纯多做一次计算。

必须报告：

- Dice/HD95/ASSD；
- 精修前后高不确定区域误差；
- 额外推理耗时；
- ROI 占总体素比例；
- error AUROC / AUPRC；
- Top-percent error recall / ROI error rate；
- refinement 前后 ROI/global error delta。

## 7. 模型结构消融（可选）

若时间允许：

- input channels；
- encoder embed dims；
- decoder embedding dim；
- patch size/stride；
- crop size；
- spacing；
- pretrained vs random init。

避免同时改变多个变量。

## 8. 评价指标

### 8.1 主指标

- DSC；
- HD95；
- ASSD；
- Precision；
- Recall。

### 8.2 工程指标

- Params；
- FLOPs；
- peak GPU memory；
- preprocessing time / case；
- inference time / case；
- reconstruction time / case。

### 8.3 拓扑/结构指标

候选：

- clDice；
- connected component error；
- false merge count；
- false break count。

最终选择需匹配具体骨结构。

## 9. 统计规范

每例输出指标，最终报告：

```text
mean ± std
median [IQR]
```

若比较两个模型：

- 同一测试病例上做 paired analysis；
- 根据数据分布选择配对 t-test 或 Wilcoxon signed-rank；
- 报告 effect size / 置信区间优于只报 p 值。

论文写作中预先指定主指标，避免多重比较后只挑显著结果。

## 10. 实验命名规范

```text
YYYYMMDD_dataset_model_input_loss_aug_uncert_seed
```

示例：

```text
20260901_ctspine1k_segformer3d_ctbone_dicece_boundary_h1_u0_s42
```

每次实验目录必须有：

```text
config.yaml
split.json
train.log
metrics_per_case.csv
summary.json
checkpoint/
figures/
```

## 11. 复现实验要求

每个主结果至少：

- 固定数据划分；
- 固定预处理版本；
- 3 个随机种子（资源不足时至少对最终主方法和 baseline 做多 seed）；
- 保存 best checkpoint；
- 保存环境版本；
- 保存 commit hash。

## 12. 首篇论文最小完整实验矩阵

### 主对比

- nnU-Net（如资源允许）；
- SegFormer3D baseline；
- 本项目最终方法。

### 消融

1. CT only vs CT + bone window；
2. Region vs Region+Boundary；
3. Region vs Region+Topology；
4. Region+Boundary+Topology；
5. + hard augmentation；
6. + uncertainty refinement。

### 泛化

至少一个：

- 外部数据集；
- 不同数据来源；
- 困难病例子集。

### 可视化

至少展示：

- 普通病例；
- 边界困难病例；
- 断裂/粘连病例；
- 金属/低骨密度（若数据存在）；
- baseline 与 proposed 的差异；
- uncertainty map 与错误区域对应。

## 13. 失败分析

论文必须保留失败案例，不只展示最好样本。

每类失败标注原因：

- 数据质量；
- 超出训练分布；
- 边界模糊；
- 金属伪影；
- 目标截断；
- 标签噪声；
- 模型连通性错误。

这部分可反向指导困难样本采样和后续改进。
