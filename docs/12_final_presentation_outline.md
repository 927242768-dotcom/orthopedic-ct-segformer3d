# 12｜v0.3.0 中期 / 结题展示提纲

> 更新日期：2026-08-30
>
> 用途：中期检查、结题答辩、组会汇报的统一展示源材料。
>
> 原则：只使用可追溯真实结果；不把任务书目标值、随机权重、真值网格工程误差或未运行 baseline 写成模型性能。

## 1. 封面

**题目：基于 SegFormer3D 的骨科 CT 分割、不确定性评估与三维科研分析系统**

副标题建议：

> 从 CT 标准化、分割与不确定性，到 physical-space 3D 与 Web 科研复核的可追溯工程闭环

页脚建议注明：科研原型，不是医疗器械。

## 2. 为什么做这个项目

一页只讲三个问题：

1. 骨科 CT 三维分析依赖可靠分割，但小样本、厚层扫描、低密度、金属伪影和真实断裂会增加难度；
2. 只看 Dice 不足以解释边界、碎片化和结构错误；
3. 科研结果需要能够追溯到数据、split、config、checkpoint、prediction、uncertainty 与三维表面，而不是只展示一张“看起来不错”的图。

建议配一张流程示意：

```text
CT → 标准化/QC → SegFormer3D → uncertainty/calibration
   → prediction → physical-space mesh/SDF → Web 科研复核
```

## 3. 数据与实验纪律

当前 formal-pipeline pilot：

- 数据：CTSpine1K `MSD-T10` 10 例真实 CT+label；
- 任务：`vertebra_binary_ctspine1k_msd_t10_v1`，binary semantic；
- split：7 train / 2 validation / 1 `test_private`；
- 10/10 自动审计通过；
- 10/10 人工 QC 通过；
- `liver_169` 只作为最终 independent test；
- 所有输入、loss、sampling、augmentation、difficult-sample、refinement 选择只使用 train + validation；
- 最终参数先提交锁定，再对 `liver_169` 执行一次正式测试；测试后未调参、未重复最终测试。

这页重点强调“实验纪律”，不要把 10 例包装成大样本研究。

## 4. 最终锁定 pipeline

最终 v13：

| 项目 | 锁定设置 |
|---|---|
| Backbone | SegFormer3D |
| Input | CT-only |
| Loss | Region + Boundary = `1.0 / 0.1` |
| Topology | `0` |
| Sampling | Bernoulli，`foreground_probability=0.25` |
| Patches / case | `4` |
| Augmentation | flip-only |
| Training ROI | `64×64×64` |
| Inference | full-volume sliding-window |
| Decision | softmax + argmax |
| Refinement | disabled |

说明：refinement 不是“没做”，而是完成真实 validation 后综合判定 **REFINEMENT=FAIL**，因此主动关闭。

## 5. Validation 消融：什么有效，什么无效

建议只放最终结论，不在 PPT 堆 v11～v23 全日志。

- CT-only 优于 CT + bone-window；后者出现严重 foreground overprediction；
- Region + Boundary（v13）作为最终 loss baseline，但 Boundary 增益很弱，不能写成显著提升；
- Topology 在部分表面/结构指标上有正信号，但伴随更强 overprediction 与 calibration 代价，因此未采用；
- fixed-per-case、boundary-hard、high-loss、high-uncertainty sampling 均未取代 v13 Bernoulli；
- rotation/scale、gamma、Gaussian noise、HU shift 均未形成足够稳定的综合收益；
- uncertainty ROI refinement 虽有候选 Dice 上升，但 Recall、fragmentation、病例稳定性和耗时恶化，最终 **FAIL**。

最终 validation mean（`liver_7/liver_8`）：

| Dice | IoU | Precision | Recall | HD95 | ASSD | Pred/GT FG |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05471 | 0.02815 | 0.03423 | 0.13649 | 185.95 mm | 51.49 mm | 4.027× |

结论必须直接写：**绝对分割精度仍低。**

## 6. 最终正式 independent test

病例：`ctspine1k-msd-t10-liver_169`

| 指标 | 结果 |
|---|---:|
| Dice | **0.02878288** |
| IoU | 0.01460158 |
| Precision | 0.02089816 |
| Recall | 0.04622219 |
| HD95 | 136.8722 mm |
| ASSD | 43.97199 mm |
| Prediction / GT foreground | 2.21178× |
| Prediction / GT components | 236 / 1 |
| Component error | 235 |
| False break | 29 |

讲解重点：

- independent Dice 比 validation mean 更低；
- 当前核心失败模式是前景过预测 + 大量 fragmentation；
- HD95 数值不能脱离病例范围单独比较后宣称“泛化更好”；
- 这是低性能但真实、冻结、未回调参数的结果。

## 7. Uncertainty / calibration：有价值，但不能过度解释

最终 independent test：

- Error AUROC = `0.86424490`；
- Error AUPRC = `0.29665454`；
- Top-10% uncertainty error recall = `0.54993443`；
- ECE = `0.02739661`；
- MCE = `0.08781999`；
- Brier = `0.08328483`；
- NLL = `0.23558760`。

建议表述：predictive entropy 仍有一定错误排序能力，可作为科研 QC 风险提示；但较低 ECE 受到大量 background 体素影响，不能解释为临床可靠性。

## 8. 三维重建与 Web 闭环

Independent prediction：

- 原始 physical mesh：365,247 顶点 / 724,694 面；
- 2.0 mm + feature strength=8：81,353 顶点 / 160,384 面；
- 顶点减少约 `77.73%`；
- 简化相对原 prediction surface 工程 ASSD/HD95≈`0.56490/1.07159 mm`；
- 0.4 mm SDF：components `236→236`，保持连通域数量；
- SDF-vs-original 工程 ASSD/HD95≈`0.02536/0.06367 mm`。

Web 已完成：

- results-review 读取真实 prediction / uncertainty MPR；
- research-3d 加载真实 prediction mesh / SDF；
- validation 与 independent 均已在 Edge WebGL2 实机验收；
- Web 复核直接读取保存的 evaluation 产物，不重新运行最终模型。

注意：上述 mesh vertex-nearest 数字是三维工程误差，不替代 segmentation HD95/ASSD。

## 9. 当前最可信成果与局限

### 已完成

- 真实 CT 数据接入、标准化与 QC；
- patient-level formal-pipeline；
- SegFormer3D 训练稳定性诊断与系统消融；
- uncertainty / calibration；
- refinement 负结果闭环；
- final parameter lock + 一次性 independent test；
- physical-space 3D / SDF / Web 科研复核；
- 138 项自动化测试与 GitHub Release v0.3.0。

### 仍未完成 / 不能伪造

- 大样本主实验；
- nnU-Net / Residual-Encoder nnU-Net 等真实强 baseline；
- metal / fracture / low-density 正式 subgroup；
- GPU 显存数据；
- 合法授权临床脱敏数据；
- 外部与多中心验证；
- 样本量足够后的统计显著性分析。

## 10. 下一阶段

建议结题/后续研究只保留四条主线：

1. 扩大病例规模，并建立新的预注册 train/validation/test；
2. 在同一协议下加入真实强 baseline；
3. 完善困难病例 metadata 与 subgroup evaluation；
4. 获得合法临床/多中心数据后验证外部泛化。

**不得重复使用当前 `liver_169` 结果进行参数选择。**

---

## 推荐答辩结束语

> 本项目当前没有得到高精度临床分割模型，但完成了一条从真实 CT 数据、训练与验证、负结果记录、独立测试，到不确定性、三维重建和 Web 复核的完整可追溯科研链。现阶段最重要的结论是明确识别了模型性能与数据规模瓶颈，并为下一阶段扩大数据、加入强 baseline 和外部验证建立了可靠工程基础。
