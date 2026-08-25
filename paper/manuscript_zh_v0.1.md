# 融合边界—拓扑约束与不确定性精修的骨科 CT SegFormer3D 分割方法

> 中文技术初稿 v0.1（2026-08-15）
>
> **注意：本文的实验结果、统计显著性和最终结论均必须由真实训练与测试数据补齐。当前版本只完成研究背景、相关工作、方法与实验设计，严禁把 TBD/预期值改写成已经取得的结果。**

## 摘要

骨科 CT 影像的高精度分割是三维重建、术前规划、骨折形态分析和计算机辅助测量的重要基础。传统阈值法和卷积神经网络在常规病例中能够获得较好的区域分割结果，但在复杂骨性边缘、相邻骨结构粘连、低骨密度、金属伪影以及层间分辨率较低的各向异性 CT 中仍可能出现边界偏移、细小结构漏分和拓扑断裂等问题。为此，本文拟以轻量级三维 Transformer 模型 SegFormer3D 为基础，构建面向骨科 CT 的标准化数据处理与多尺度分割框架。首先对 DICOM 影像进行序列识别、HU 恢复、空间方向校正、体素重采样、骨窗增强和质量控制，并研究标准化 CT 与骨窗双通道输入。随后在 SegFormer3D 的多尺度体特征编码和 all-MLP 解码基础上，引入区域重叠、边界约束和拓扑保持联合损失，以同时优化体素区域、骨表面和解剖连通性。针对骨折、低骨密度、金属伪影和大层厚等困难病例，进一步设计困难样本增强和基于预测熵的不确定性区域精修机制。实验计划在公开骨科/脊柱 CT 数据集及合法授权的脱敏临床数据上进行患者级划分，并采用 Dice、HD95、ASSD、Precision、Recall、结构连通性指标和推理效率进行综合评估，同时开展输入、损失、困难样本增强和不确定性精修的系统消融。最终将分割模型接入 Web 科研辅助分析原型，为后续三维重建、MPR 联动和交互测量提供统一接口。

**关键词：** 骨科 CT；SegFormer3D；三维医学图像分割；边界损失；拓扑保持；不确定性；三维重建

---

## 1 引言

CT 具有较高的空间分辨率和对骨组织良好的密度对比，是骨折、脊柱疾病、骨关节病变及术前规划中的重要影像手段。临床二维阅片能够提供丰富的断层信息，但当目标结构具有复杂三维几何形态时，仅依赖逐层观察不利于快速理解骨折断端、骨块关系、关节面形态和手术入路。将 CT 体数据转化为可交互三维模型通常需要准确的骨结构分割，而人工逐层勾画既耗时又具有观察者差异，因此自动化分割成为骨科影像智能分析的重要基础环节。

公开脊柱 CT 基准进一步说明了这一任务的复杂性。VerSe 数据包含不同视野、空间分辨率、扫描设备、解剖变异与病理情况，并特别涉及骨折、金属植入和过渡椎等困难情形。其基准研究指出，算法性能会受到罕见解剖变异和跨数据分布差异的显著影响。这意味着骨科 CT 分割不能只追求常规病例的平均 Dice，还需要关注跨中心泛化、边界质量、结构连通性与失败病例。

近年来，Transformer 通过自注意力机制增强了对长距离依赖和全局上下文的建模能力。SegFormer 使用分层 Transformer 编码器提取多尺度特征，并利用轻量 MLP 解码器完成语义分割；其无固定位置编码的设计也为跨分辨率输入提供了较好的适应性。SegFormer3D 将这一思想扩展至三维体数据，通过 3D patch embedding、空间降采样注意力和多尺度体特征聚合，在保持较低参数量和计算量的同时完成医学体分割。然而，其官方实现主要验证于 BraTS、Synapse 和 ACDC 等通用医学数据，并未针对骨科 CT 的 HU 标准化、骨窗、各向异性、骨边界和拓扑错误进行专门设计。

另一方面，单纯使用交叉熵或 Dice 类区域损失容易把优化重点放在整体体素重叠，而骨科三维重建对毫米级表面误差、细小骨性突起、结构断裂和相邻骨粘连更加敏感。Boundary Loss 等方法通过距离场把优化信号直接引入目标边界；clDice 等方法则表明，可微骨架/拓扑约束能够补充传统区域指标对连通关系不敏感的问题。需要注意的是，clDice 最初主要面向管状和网络结构，因此其在不同骨结构上的适用性必须通过消融验证，而不能直接假设为有效。

此外，困难病例往往集中体现模型的实际风险。金属内固定造成的高密度条纹、低骨密度导致的骨—软组织对比下降、骨折导致的局部拓扑改变以及厚层扫描造成的层间信息不足，都可能引起预测不稳定。医学分割中的不确定性研究表明，预测可靠性与质量控制应与分割精度同时评估。因此，在粗分割后定位高不确定区域并进行局部精修，有望在有限额外计算成本下把模型资源集中到最容易出错的区域。

基于上述分析，本文拟研究一种面向骨科 CT 的 SegFormer3D 改进框架。当前研究假设包括：（1）规范化 DICOM/HU/空间处理和骨窗多通道输入能够减少扫描协议差异；（2）区域、边界和拓扑联合目标能够改善骨表面误差和连接错误；（3）困难样本增强能够提高异常病例鲁棒性；（4）不确定性驱动的局部精修能够在较小额外计算量下改善困难区域。上述假设均将在统一患者级数据划分和严格消融实验中验证。

---

## 2 相关工作

### 2.1 三维医学图像分割

U-Net 系列通过编码器—解码器和跳跃连接成为医学分割的经典范式，nnU-Net 进一步通过自配置数据预处理、网络结构和训练策略建立了强大的通用基线。随着 Transformer 在视觉任务中的发展，UNETR 将三维体数据切分为 patch 序列并利用 Transformer 编码长距离关系；nnFormer 进一步面向体数据设计局部与全局注意力及 skip attention。这些方法提高了全局建模能力，但其计算成本和复杂解码结构也增加了大体积 CT 的训练与部署负担。

### 2.2 SegFormer 与 SegFormer3D

SegFormer 使用多阶段层次化 Transformer 编码器输出不同尺度特征，再将各尺度特征映射到统一维度后由 MLP 解码器融合。SegFormer3D 采用对应的三维卷积 patch embedding 和空间降采样注意力，使 Transformer 能在三维体数据上建立多尺度上下文，同时避免复杂 U 形解码器。其轻量化特点与本项目“模型精度—计算成本—Web 部署”三者平衡的需求较一致，因此本文选其作为基础骨干网络。

### 2.3 脊柱与骨结构 CT 分割

CTSpine1K 提供大规模椎体 CT 标注，为脊柱自动分割和跨来源评估提供数据基础。VerSe 2019/2020 则建立了多设备、多中心、含解剖变异和病理情况的椎体标注基准，强调了异常解剖和 domain shift 对算法性能的影响。2024 年的 VerFormer 从骨科/脊柱场景出发，在 Transformer 中引入 vertebrae-aware global query，以突出与椎体相关的 token，并在 VerSe 数据上验证。近期研究进一步扩大了竞争范围：2025 年 SpineMamba 将 Residual Visual Mamba 与可学习三维脊柱形状先验结合，表明长程依赖建模和显式结构先验已成为三维脊柱分割的重要方向；同年的椎体 Transformer 工作将 WNet 分割、ViT 类型分析与解剖变异感知结合，并针对切片缺失、噪声和扫描差异开展鲁棒性分析；2026 年 VertebraFormer 则把椎体分割、编号和病灶定位统一到结构感知多任务框架，并采用多域与 leave-one-domain-out 方案研究泛化。由此可见，本文不能把“采用 Transformer”本身作为主要创新，而应重点验证三维多尺度骨科适配、边界/拓扑约束、困难样本、不确定性局部精修以及从体素分割到物理空间表面的误差闭环。

### 2.4 边界与拓扑约束

区域损失直接优化预测区域与标签之间的体素重叠，但不能完全描述表面位置和解剖连通性。Boundary Loss 通过标签边界的距离信息建立可微表面约束，可与 Dice/CE 等区域损失联合。clDice 利用预测与真值软骨架之间的拓扑精确率和召回率刻画连通性，为结构保持提供了可微目标。2024 年的多类别 Betti matching 工作进一步将 persistent-homology 拓扑约束扩展到多类别医学分割；TEDS-Net 则通过对具有正确拓扑的先验形状进行连续形变生成解剖合理分割。这些进展说明“拓扑正确”可以被显式优化，但骨折等骨科病例的真实解剖本身可能发生断裂，因此固定拓扑先验并不总是成立。本文不把某一种拓扑损失预设为最终最优方案，而是在骨科 CT 目标上分别评估区域、边界、拓扑及其组合，并对骨折病例单独分析。

### 2.5 不确定性估计与分割质量控制

常见不确定性估计包括 Monte Carlo Dropout、Deep Ensemble、测试时增强和预测熵等。近期 EDUE 等工作强调医学分割的不确定性应与真实错误、标注者分歧或质量控制能力相关，而不是仅生成视觉热图；UCTNet 则进一步把不确定性作为特征学习的引导信号，表明 uncertainty 可以直接参与分割网络的信息选择。由于本项目首阶段公开数据通常只有单一标准标签，本文优先采用无需多标注者的预测熵作为低成本基线，并分析其与分割误差的空间相关性；后续如获得多专家标注，可进一步研究 aleatoric uncertainty 与标注分歧的一致性，若熵图在困难区域验证有效，再评估 uncertainty-guided feature/refinement 结构。

---

## 3 方法

### 3.1 总体框架

本文流程由数据标准化、三维多尺度分割、联合损失、困难样本策略、不确定性精修和结果重建六部分组成：

```text
DICOM / NIfTI
→ DICOM series 识别与 QC
→ HU 恢复 / orientation 校正 / spacing 重采样
→ CT 标准化通道 + bone-window 通道
→ SegFormer3D
→ Region + Boundary + Topology joint loss
→ uncertainty map
→ difficult ROI refinement
→ segmentation mask
→ 3D surface / Web visualization
```

### 3.2 DICOM 与 CT 标准化

对于原始 DICOM 序列，不使用文件名顺序作为层面依据，而优先根据 `ImageOrientationPatient` 与 `ImagePositionPatient` 计算切片法向位置，并以 `SeriesInstanceUID` 区分不同序列。CT 像素通过 DICOM rescale 参数恢复为 HU。随后统一影像方向和物理空间，并重采样至实验设定的目标 spacing。影像使用连续插值，标签使用 nearest-neighbor 插值以避免生成非法类别。

当前实现中，标准强度通道先在预设 HU 区间（工程初值为 \([-1000,2000]\) HU）裁剪，再按单病例裁剪后均值与标准差进行 z-score，并把该均值/标准差写入 metadata 以保证强度增强可追溯。骨窗通道根据窗口中心和宽度独立生成：

\[
I_{bone}=\frac{\mathrm{clip}(HU, C-W/2, C+W/2)-(C-W/2)}{W}.
\]

骨窗参数不是固定临床结论，而作为数据/模型配置保存，并通过输入消融确定其对目标任务的实际影响。

### 3.3 SegFormer3D 多尺度骨干网络

设输入体数据为 \(X\in \mathbb{R}^{C\times D\times H\times W}\)。SegFormer3D 通过四个层次化阶段逐步降低空间分辨率并提高特征维度。每个阶段首先使用三维卷积完成 overlapping patch embedding，再通过带 spatial reduction 的多头自注意力减少 key/value 序列长度，从而降低全分辨率三维 self-attention 的计算开销。四级特征 \(F_1,F_2,F_3,F_4\) 经 MLP 投影到统一通道维度、上采样到共同尺度并融合，得到分割 logits。

本项目不修改上游基础架构的所有权归属，而通过适配器调用官方 SegFormer3D，实现输入通道数、类别数、各层 embedding 维度和 decoder 维度的配置化。骨科专用创新主要位于数据管线、联合损失、困难样本学习、不确定性精修及三维/Web 集成。

### 3.4 区域—边界—拓扑联合损失

总损失定义为：

\[
L_{total}=\lambda_r L_{region}+\lambda_b L_{boundary}+\lambda_t L_{topology}.
\]

其中，\(L_{region}\) 采用 Dice 与 CE/BCE 组合，用于优化整体体素分类和类别不平衡；\(L_{boundary}\) 根据真值边界的 signed distance field 对预测概率进行表面约束；\(L_{topology}\) 首版采用 soft-clDice 形式，通过可微 skeletonization 约束结构连通性。对于非管状单块骨，拓扑项是否有效必须由实验决定，若出现负作用，应改为更匹配目标结构的连通域/持久同调约束。

### 3.5 困难样本增强

训练增强分为几何、强度和困难采样三层。当前已实现的几何增强包括三维随机翻转、小角度旋转和各向同性缩放；强度增强包括 gamma、Gaussian noise 与 HU shift。由于标准 CT 通道采用 case-wise z-score，gamma/HU shift 在实现时利用 metadata 中的 clipped HU mean/std 恢复到 HU 域后再变换，避免把 z-score 体数据错误裁剪到 \([0,1]\)。baseline 前的困难采样首先采用标签边界作为 `boundary-proxy`，后续在真实模型建立后再以高 loss、高 HD95 或高 uncertainty 的病例/patch 替换或补充该代理。金属伪影模拟只在有真实金属病例可用于校验时启用，避免生成不符合成像物理的伪模式。

### 3.6 不确定性驱动的局部精修

设模型输出类别概率为 \(p_c(v)\)，体素预测熵定义为：

\[
U(v)=-\sum_c p_c(v)\log(p_c(v)+\epsilon).
\]

根据 uncertainty map 选择 Top-k 百分位或高于阈值的体素，并对其连通区域进行适度膨胀得到候选 ROI。首先使用 uncertainty→error AUROC、AUPRC、错误/正确体素平均 entropy、Top-percent error recall、ROI error rate 与 ROI fraction 定量判断预测熵是否真的集中于错误区域，而不是仅依赖热图主观判断。当前工程原型进一步实现了三维局部残差精修网络：将 CT/骨窗等影像通道、coarse probability/logits 与 uncertainty 作为输入预测残差，只在 ROI mask 内更新 coarse logits，ROI 外严格保留原预测。二阶段训练中 coarse 分割默认冻结，精修损失只在 ROI 内归一化，并记录精修前后 ROI/global error delta。实验将比较：无精修、全图二次推理和 uncertainty ROI 精修，以区分“额外计算”与“不确定性定位”本身的贡献。上述实现目前只完成工程测试，尚无真实 checkpoint 性能结论。

### 3.7 质量控制与可追溯性

每例数据保存原始/处理后 shape、spacing、origin、direction、强度统计、DICOM series 数量和 QC warning。所有训练实验保存 config、患者级 split、随机种子、checkpoint、逐病例指标和代码版本。测试集只用于最终评估，不能反复用于选择 loss 权重或模型结构。

为避免工程 smoke 被误写成论文实验，训练/评估入口设置 formal preflight：在正式运行前检查病例级 split 泄漏、官方 test-private 误用、人工 QC、pipeline/输入通道、标签范围与 `num_classes`，正式训练还要求可用 CUDA 环境。当前 CTSpine1K/VerSe `1–25` 的 `C1–L6` 映射只用于 QC/Web 可读显示，不改变原始标签值，也不提前决定正式任务采用 binary、multi-class semantic 或 instance segmentation。

### 3.8 物理空间表面重建与工程几何控制

分割 mask 首先在 `(Z,Y,X)` 体素数组上使用 Marching Cubes 提取三角面，再依据影像 `spacing`、`origin` 与 `direction` 显式转换至物理 `(X,Y,Z)` 毫米坐标。Web 轻量展示保留全分辨率网格，同时提供可控的 vertex-clustering 简化，并记录顶点/面缩减率、表面积变化以及全分辨率—简化网格的顶点近邻误差。为区分模型误差和预处理误差，项目还单独比较原始标签与 1 mm nearest-neighbor 重采样标签的物理表面差异；该分析只用于量化重采样离散化，不作为模型分割性能或临床测量精度。

---

## 4 实验设计

### 4.1 数据集

首阶段公开数据候选：CTSpine1K、VerSe 2019/2020、TotalSegmentator 中与目标骨结构匹配的子集。2026-08-16 已使用 CTSpine1K `MSD-T10` 的 10 例真实 CT+label 完成工程预处理验证（9 例官方 `trainset`、1 例 `test_private`），10/10 自动几何/标签/QC 审计通过，并验证真实双通道单 patch 的 SegFormer3D+联合损失 forward/backward 链。Web/QC 使用与 VerSe 一致的 `1–25 → C1–L6` 可读映射，但原始标签不重编码。该 10 例子集仅作为工程证据，不作为论文正式 train/validation/test，也不产生可写入 Results 的性能数字。最终主数据集、binary/multiclass/instance 标签定义和实际病例数量仍待组内确认和正式实验方案固定。

临床数据仅在获得合法研究授权且完成脱敏后纳入；临床数据的机构来源、纳排标准、伦理/授权信息和划分策略必须在最终论文中明确说明。

### 4.2 数据划分

所有划分以患者为单位，避免同一患者不同切片进入不同集合。若采用公开 benchmark 的官方 split，则优先保持官方划分；若需要内部划分，则预先固定 train/validation/test，并保存唯一 split 文件。对于多来源数据，将额外设计 leave-one-source-out 或外部测试以评价泛化。

### 4.3 对比方法

计划至少包括：

1. nnU-Net（资源允许时）；
2. 原始/适配后的 SegFormer3D baseline；
3. 本文完整方法。

若显存和时间允许，再增加 UNETR、Swin UNETR 或近期脊柱专用方法作为对比。所有方法尽可能使用一致的数据划分、spacing 与输入分辨率。

### 4.4 消融实验

#### 输入通道消融

- 标准化 CT；
- 标准化 CT + bone-window。

#### Loss 消融

- Region；
- Region + Boundary；
- Region + Topology；
- Region + Boundary + Topology。

#### 困难样本消融

- 常规增强；
- 强度域增强；
- hard-example sampling；
- 若有真实金属病例，再加入金属相关策略。

#### 不确定性消融

- 无 uncertainty；
- entropy 仅用于 QC；
- uncertainty ROI refinement；
- 全图二次推理对照。

### 4.5 评价指标

主指标包括 Dice Similarity Coefficient、IoU、HD95、ASSD、Precision 和 Recall。对于结构连通性，候选指标包括 clDice、连通域数量差异、错误粘连与断裂数。若采用 multi-class 椎体任务，逐病例 macro 只对真值或预测实际出现的前景类别求平均，双方都缺失的类别不计入，以避免空类别产生虚高分数，并同时输出逐类别指标。对于 uncertainty，额外报告 error AUROC/AUPRC、错误/正确体素平均 entropy、Top-percent error recall、ROI error rate 与 ROI fraction。工程指标包括参数量、FLOPs、峰值显存、单病例预处理时间、推理时间和三维重建时间。

逐病例结果同时报告 mean±std 和 median[IQR]。方法比较采用配对统计检验，并根据分布选择配对 t 检验或 Wilcoxon signed-rank test；同时报告效应量或置信区间，避免只报告 p 值。

### 4.6 实现细节

当前官方 SegFormer3D 基础环境参考 Python 3.11.7、PyTorch 2.1.0 和其仓库依赖。最终论文需要根据实际运行环境补齐 GPU 型号、显存、batch size、ROI size、optimizer、学习率、epoch、AMP、随机种子和 sliding-window overlap 等参数。

---

## 5 结果（待真实实验填写）

> 本节当前不得填写“预期 Dice”作为真实结果。

### 5.1 与基线方法比较

| Method | Params | DSC ↑ | HD95 ↓ | ASSD ↓ | Time ↓ |
|---|---:|---:|---:|---:|---:|
| nnU-Net | TBD | TBD | TBD | TBD | TBD |
| SegFormer3D | TBD | TBD | TBD | TBD | TBD |
| Proposed | TBD | TBD | TBD | TBD | TBD |

### 5.2 联合损失消融

| Region | Boundary | Topology | DSC ↑ | HD95 ↓ | ASSD ↓ | Structure metric ↑ |
|---|---|---|---:|---:|---:|---:|
| ✓ |  |  | TBD | TBD | TBD | TBD |
| ✓ | ✓ |  | TBD | TBD | TBD | TBD |
| ✓ |  | ✓ | TBD | TBD | TBD | TBD |
| ✓ | ✓ | ✓ | TBD | TBD | TBD | TBD |

### 5.3 困难病例与不确定性

待分别统计常规病例、骨折、低骨密度、金属伪影和厚层病例（以实际数据属性为准），并评价 uncertainty 与错误区域的相关性、精修后的指标变化及额外计算量。

---

## 6 讨论（待结果后完成）

最终讨论至少回答：

1. 边界项改善的是 Dice 还是 surface metric，收益是否具有统计意义；
2. 拓扑项在目标骨结构上是否真的有效，是否出现过平滑/错误连接副作用；
3. 骨窗双通道的收益是否稳定，是否只对某些病例有效；
4. hard augmentation 是否提升困难子集而牺牲普通病例；
5. uncertainty 是否能发现真实错误，精修收益是否值得额外计算；
6. 多中心/多来源 domain shift 下性能下降程度；
7. 分割误差如何传递到三维网格和测量结果。

---

## 7 局限性

预期需要讨论但仍须根据真实实验修订：数据中心数量有限、部分困难病例稀缺、标签存在观察者差异、拓扑损失可能并不适用于所有骨结构、模型尚未经过前瞻性临床验证，以及科研 Web 原型与合规医疗器械之间仍存在较大工程与监管距离。

---

## 8 结论（待真实实验后填写）

当前版本不提前写结论。最终结论只能总结真实数据支持的发现。

---

## 参考文献（首版）

1. Xie E, Wang W, Yu Z, et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS, 2021.
2. Hatamizadeh A, Tang Y, Nath V, et al. UNETR: Transformers for 3D Medical Image Segmentation. WACV, 2022.
3. Zhou H-Y, Guo J, Zhang Y, et al. nnFormer: Interleaved Transformer for Volumetric Segmentation. arXiv:2109.03201.
4. Perera S, Navard P, Yilmaz A. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. CVPR Workshops, 2024.
5. Deng Y, Wang C, Hui Y, et al. CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography. Machine Learning for Biomedical Imaging, 2025, 3(Special Issue on MICCAI Open Data 2024-2025):824-832.
6. Sekuboyina A, et al. VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images. Medical Image Analysis, 2021.
7. Wasserthal J, et al. TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images. Radiology: Artificial Intelligence, 2023.
8. Li X, Hong Y, Xu Y, Hu M. VerFormer: Vertebrae-Aware Transformer for Automatic Spine Segmentation from CT Images. Diagnostics, 2024.
9. Zhang Z, Liu T, Fan G, et al. SpineMamba: Enhancing 3D spinal segmentation in clinical imaging through residual visual Mamba layers and shape priors. Computerized Medical Imaging and Graphics, 2025, 123:102531.
10. Yang C, Huang L, Sucharit W, et al. Transformer-enhanced vertebrae segmentation and anatomical variation recognition from CT images. Scientific Reports, 2025, 15:34329.
11. Du J, Ge H, Zhang R, et al. Structure-aware multi-task learning with domain generalization for robust vertebrae analysis in spinal CT. npj Digital Medicine, 2026, 9:217.
12. Kervadec H, et al. Boundary Loss for Highly Unbalanced Segmentation. Medical Image Analysis, 2021.
13. Shit S, Paetzold J C, Sekuboyina A, et al. clDice—A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR, 2021.
14. Abutalip K, Saeed N, Sobirov I, et al. EDUE: Expert Disagreement-Guided One-Pass Uncertainty Estimation for Medical Image Segmentation. arXiv:2403.16594, 2024.
15. Guo X, Lin X, Yang X, et al. UCTNet: Uncertainty-guided CNN-Transformer hybrid networks for medical image segmentation. Pattern Recognition, 2024, 152:110491.
16. Berger A H, Stucki N, Lux L, et al. Topologically Faithful Multi-class Segmentation in Medical Images. arXiv:2403.11001, 2024.
17. Wyburd M K, Dinsdale N K, Jenkinson M, Namburete A I L. Anatomically plausible segmentations: Explicitly preserving topology through prior deformations. Medical Image Analysis, 2024, 97:103222.

> 机器可用参考文献库已建立于 `paper/references.bib`（当前 38 条英文核心条目），结构化文献矩阵见 `docs/08_literature_matrix.md`（当前 40 条）。正式投稿前仍需统一期刊格式，并对国内文献与最新正式发表版本再次核验。
