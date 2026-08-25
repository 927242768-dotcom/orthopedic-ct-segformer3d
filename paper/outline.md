# 论文持续写作框架

> 工作题目（暂定）：**Boundary- and Topology-aware SegFormer3D with Uncertainty-guided Refinement for Orthopedic CT Segmentation**
>
> 中文暂定：**融合边界—拓扑约束与不确定性精修的骨科 CT SegFormer3D 分割方法**
>
> 说明：这是持续写作文件。方法部分可以先写，Results 中所有数字必须等待真实实验结果，禁止提前填“预期结果”。

## Abstract（最后定稿）

结构：

1. 背景：骨科 CT 自动分割对三维重建/术前规划重要；
2. 问题：复杂骨边界、细小结构、各向异性、困难病例；
3. 方法：SegFormer3D + 标准化 CT pipeline + joint loss + hard augmentation + uncertainty refinement；
4. 数据与指标：真实数据集名称、患者数、DSC/HD95/ASSD；
5. 真实结果；
6. 结论与系统原型。

## 1. Introduction

### 1.1 临床/工程背景

- 骨科 CT 三维结构表达的重要性；
- 手工分割耗时；
- 传统阈值对骨质疏松、骨折、金属伪影不稳定；
- 高质量 mask 是后续三维表面与测量的基础。

### 1.2 现有方法问题

- CNN 局部感受野与全局关系；
- Transformer 计算成本；
- 纯区域损失可能忽视表面与拓扑；
- 普通平均指标不能完全反映困难病例。

### 1.3 为什么选择 SegFormer3D

- 分层多尺度；
- 轻量 all-MLP decoder；
- 适合做“高精度 + 低复杂度”平衡；
- 但原方法未针对骨科 CT 专门设计。

### 1.4 本文贡献（当前仅为待验证假设）

最终必须根据实验改写：

- 骨科 CT 标准化/骨窗多通道输入；
- region-boundary-topology joint objective；
- hard-case augmentation；
- uncertainty-guided local refinement；
- 端到端科研 Web 原型。

## 2. Related Work

### 2.1 CNN-based volumetric medical segmentation

- U-Net / 3D U-Net
- nnU-Net

### 2.2 Transformer-based medical segmentation

- UNETR
- nnFormer
- Swin UNETR
- SegFormer / SegFormer3D

### 2.3 Orthopedic CT / spine segmentation

- CTSpine1K
- VerSe（待补）
- TotalSegmentator bone structures
- 近 3 年骨科 CT Transformer 工作（待补）

### 2.4 Boundary/topology-aware segmentation

- Boundary Loss
- clDice
- Hausdorff/surface-aware losses（待补）

### 2.5 Uncertainty and refinement

- predictive entropy
- MC dropout
- uncertainty-guided refinement 代表工作（待补）

## 3. Materials and Methods

### 3.1 Datasets

待填真实信息：

- Dataset A：
- Dataset B：
- inclusion/exclusion；
- patient count；
- annotation classes；
- scanner/source；
- license/ethics；
- patient-level split。

### 3.2 CT Preprocessing

完整写清：

- DICOM sorting；
- HU conversion；
- orientation；
- resampling；
- intensity clipping；
- bone window；
- crop/patch；
- augmentation。

### 3.3 Baseline SegFormer3D

说明：

- 4-stage hierarchical encoder；
- 3D patch embedding；
- spatial reduction attention；
- multiscale features；
- all-MLP decoder。

需要画一张**本项目自己的结构图**，不能直接拿上游图当原创。

### 3.4 Bone-window Multi-channel Input

定义输入：

```text
X = concat(X_ct_normalized, X_bone_window)
```

说明参数选择和消融。

### 3.5 Joint Region-Boundary-Topology Objective

候选公式：

```text
L_total = λr L_region + λb L_boundary + λt L_topology
```

分别定义各项，并解释为什么对应骨结构问题。

### 3.6 Hard-case Augmentation

按实际最终实现写：

- intensity perturbation；
- geometric transforms；
- hard-case sampling；
- metal/thick-slice strategy（若实际采用）。

### 3.7 Uncertainty-guided Refinement

候选：

```text
p = model(x)
U = entropy(p)
ROI = select(U)
p_refined = refinement(x_ROI, p_ROI)
```

写清 ROI 选择、扩张、融合和额外计算量。

### 3.8 Implementation Details

真实填：

- Python；
- PyTorch；
- GPU；
- patch size；
- batch size；
- optimizer；
- LR scheduler；
- epochs；
- seeds；
- inference overlap；
- early stopping/checkpoint rule。

## 4. Experiments

### 4.1 Evaluation Metrics

- DSC；
- HD95；
- ASSD；
- Precision；
- Recall；
- topology metric；
- Params/FLOPs/time。

### 4.2 Comparison with Baselines

表格模板：

| Method | Params | DSC ↑ | HD95 ↓ | ASSD ↓ | Time ↓ |
|---|---:|---:|---:|---:|---:|
| nnU-Net | TBD | TBD | TBD | TBD | TBD |
| SegFormer3D | TBD | TBD | TBD | TBD | TBD |
| Ours | TBD | TBD | TBD | TBD | TBD |

### 4.3 Ablation Study

#### Table A：输入

| CT | Bone window | DSC | HD95 | ASSD |
|---|---|---:|---:|---:|
| ✅ | ❌ | TBD | TBD | TBD |
| ✅ | ✅ | TBD | TBD | TBD |

#### Table B：loss

| Region | Boundary | Topology | DSC | HD95 | Topology metric |
|---|---|---|---:|---:|---:|
| ✅ | ❌ | ❌ | TBD | TBD | TBD |
| ✅ | ✅ | ❌ | TBD | TBD | TBD |
| ✅ | ❌ | ✅ | TBD | TBD | TBD |
| ✅ | ✅ | ✅ | TBD | TBD | TBD |

#### Table C：困难样本与精修

| Hard aug | Uncertainty refine | DSC | HD95 | Difficult subset DSC | Time |
|---|---|---:|---:|---:|---:|
| ❌ | ❌ | TBD | TBD | TBD | TBD |
| ✅ | ❌ | TBD | TBD | TBD | TBD |
| ✅ | ✅ | TBD | TBD | TBD | TBD |

### 4.4 Generalization

- external dataset/source；
- thick-slice；
- metal artifact；
- fracture/low-density subset（取决于数据）。

### 4.5 Qualitative Results

每个图必须注明：

- case type；
- GT；
- baseline；
- ours；
- error/uncertainty；
- 不能只选最好样本。

## 5. Discussion

必须讨论：

- 哪个模块真正有效；
- 为什么对 boundary/topology 有效或无效；
- 是否牺牲推理速度；
- 训练成本；
- 数据偏倚；
- 标签噪声；
- 跨中心泛化；
- 是否适合其他骨结构；
- Web 原型距离临床产品还差什么。

## 6. Limitations

至少考虑：

- 数据集数量/中心数量；
- 部位范围；
- 不确定性是否经过 calibration；
- segmentation 与实际诊断之间的区别；
- 三维重建表面误差与 CT 层厚；
- 临床验证不足。

## 7. Conclusion

只总结真实得到的结论。

## References

首批：

1. SegFormer, 2021.
2. UNETR, 2021.
3. nnFormer, 2021.
4. Swin UNETR self-supervised pretraining, 2021/2022.
5. SegFormer3D, 2024.
6. CTSpine1K, 2021.
7. TotalSegmentator, 2022/2023.
8. Boundary Loss, 2018/2019.
9. clDice, 2020/2021.

后续统一维护 BibTeX，不手工反复复制参考文献。
