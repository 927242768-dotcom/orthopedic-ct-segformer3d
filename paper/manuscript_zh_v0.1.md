# 基于 SegFormer3D 的骨科 CT 分割、不确定性评估与三维科研分析系统

> 中文技术稿 v0.1（持续更新至 2026-08-29）
>
> **结果声明：本文当前所有 validation、正式 independent test、uncertainty/calibration 与三维工程数字均来自项目真实产物；不使用随机权重、预期值或伪造指标。当前样本量极小，结论仅限本工程 pilot，不代表临床有效性。**

## 摘要

骨科 CT 自动分割是三维重建和计算机辅助测量的重要基础，但在小样本、低对比度、结构粘连/断裂及扫描差异下容易出现前景过预测、边界偏移和拓扑碎裂。本文以 SegFormer3D 为骨干，构建从 CT 标准化、患者级数据划分、训练与系统消融、预测熵不确定性/校准评价，到 physical-space 三维重建和 Web 科研复核的一体化工程流程。当前 pilot 使用 CTSpine1K `MSD-T10` 10 例真实 CT+label，固定为 7 train / 2 validation / 1 `test_private` independent test。通过输入、Region/Boundary/Topology loss、sampling、augmentation、困难样本和 uncertainty ROI refinement 消融，最终在测试前锁定 v13：CT-only、Region+Boundary、Bernoulli sampling、flip-only、64³ training ROI、softmax/argmax decision，refinement 因综合验证失败而禁用。两例 validation mean Dice 为 `0.05471`，正式独立 `liver_169` 一次性测试 Dice 为 `0.02878`、HD95 为 `136.87 mm`、ASSD 为 `43.97 mm`，显示当前模型绝对精度仍低。另一方面，predictive entropy 在 validation 上对错误区域具有较高排序能力，正式 test error AUROC 为 `0.86424`；2.0 mm 特征加权网格简化可减少约 `77.73%` 顶点，0.4 mm SDF 保持独立预测连通域 `236→236`，并已在 Edge WebGL2 实机展示 prediction/uncertainty/3D surface。结果表明，本工作完成了可追溯的科研工程闭环，但现阶段更适合作为方法探索与科研辅助原型，而非高精度临床分割模型。

**关键词：** 骨科 CT；SegFormer3D；三维医学图像分割；不确定性；校准；三维重建；Web 科研辅助分析

---

## 1 引言

CT 具有较高的空间分辨率和对骨组织良好的密度对比，是骨折、脊柱疾病、骨关节病变及术前规划中的重要影像手段。临床二维阅片能够提供丰富的断层信息，但当目标结构具有复杂三维几何形态时，仅依赖逐层观察不利于快速理解骨折断端、骨块关系、关节面形态和手术入路。将 CT 体数据转化为可交互三维模型通常需要准确的骨结构分割，而人工逐层勾画既耗时又具有观察者差异，因此自动化分割成为骨科影像智能分析的重要基础环节。

公开脊柱 CT 基准进一步说明了这一任务的复杂性。VerSe 数据包含不同视野、空间分辨率、扫描设备、解剖变异与病理情况，并特别涉及骨折、金属植入和过渡椎等困难情形。其基准研究指出，算法性能会受到罕见解剖变异和跨数据分布差异的显著影响。这意味着骨科 CT 分割不能只追求常规病例的平均 Dice，还需要关注跨中心泛化、边界质量、结构连通性与失败病例。

近年来，Transformer 通过自注意力机制增强了对长距离依赖和全局上下文的建模能力。SegFormer 使用分层 Transformer 编码器提取多尺度特征，并利用轻量 MLP 解码器完成语义分割；其无固定位置编码的设计也为跨分辨率输入提供了较好的适应性。SegFormer3D 将这一思想扩展至三维体数据，通过 3D patch embedding、空间降采样注意力和多尺度体特征聚合，在保持较低参数量和计算量的同时完成医学体分割。然而，其官方实现主要验证于 BraTS、Synapse 和 ACDC 等通用医学数据，并未针对骨科 CT 的 HU 标准化、骨窗、各向异性、骨边界和拓扑错误进行专门设计。

另一方面，单纯使用交叉熵或 Dice 类区域损失容易把优化重点放在整体体素重叠，而骨科三维重建对毫米级表面误差、细小骨性突起、结构断裂和相邻骨粘连更加敏感。Boundary Loss 等方法通过距离场把优化信号直接引入目标边界；clDice 等方法则表明，可微骨架/拓扑约束能够补充传统区域指标对连通关系不敏感的问题。需要注意的是，clDice 最初主要面向管状和网络结构，因此其在不同骨结构上的适用性必须通过消融验证，而不能直接假设为有效。

此外，困难病例往往集中体现模型的实际风险。金属内固定造成的高密度条纹、低骨密度导致的骨—软组织对比下降、骨折导致的局部拓扑改变以及厚层扫描造成的层间信息不足，都可能引起预测不稳定。医学分割中的不确定性研究表明，预测可靠性与质量控制应与分割精度同时评估。因此，在粗分割后定位高不确定区域并进行局部精修，有望在有限额外计算成本下把模型资源集中到最容易出错的区域。

基于上述分析，本文围绕四个可验证问题展开：规范化 CT 与骨窗双通道是否稳定改善分割；Boundary/Topology 约束能否在区域、表面和结构指标上形成一致收益；困难样本策略是否优于简单 Bernoulli sampling；predictive entropy 是否既能提示错误，又能通过局部 refinement 稳健改善结果。所有方法选择仅依据固定 validation，独立 test 在最终参数锁定并推送后只执行一次，从实验设计上避免依据 test 反向调参。

---

## 2 相关工作

### 2.1 三维医学图像分割

U-Net 系列通过编码器—解码器和跳跃连接成为医学分割的经典范式，nnU-Net 进一步通过自配置数据预处理、网络结构和训练策略建立了强大的通用基线。随着 Transformer 在视觉任务中的发展，UNETR 将三维体数据切分为 patch 序列并利用 Transformer 编码长距离关系；nnFormer 进一步面向体数据设计局部与全局注意力及 skip attention。这些方法提高了全局建模能力，但其计算成本和复杂解码结构也增加了大体积 CT 的训练与部署负担。

### 2.2 SegFormer 与 SegFormer3D

SegFormer 使用多阶段层次化 Transformer 编码器输出不同尺度特征，再将各尺度特征映射到统一维度后由 MLP 解码器融合。SegFormer3D 采用对应的三维卷积 patch embedding 和空间降采样注意力，使 Transformer 能在三维体数据上建立多尺度上下文，同时避免复杂 U 形解码器。其轻量化特点与本项目“模型精度—计算成本—Web 部署”三者平衡的需求较一致，因此本文选其作为基础骨干网络。

### 2.3 脊柱与骨结构 CT 分割

CTSpine1K 提供大规模椎体 CT 标注，为脊柱自动分割和跨来源评估提供数据基础。VerSe 2019/2020 则建立了多设备、多中心、含解剖变异和病理情况的椎体标注基准，强调了异常解剖和 domain shift 对算法性能的影响。2024 年的 VerFormer 从骨科/脊柱场景出发，在 Transformer 中引入 vertebrae-aware global query，以突出与椎体相关的 token，并在 VerSe 数据上验证。近期研究进一步扩大了竞争范围：2025 年 SpineMamba 将 Residual Visual Mamba 与可学习三维脊柱形状先验结合；同年的椎体 Transformer 工作将 WNet 分割、ViT 类型分析与解剖变异感知结合，并针对切片缺失、噪声和扫描差异开展鲁棒性分析；椎体骨折研究则采用 nnU-Net 分割与骨折分类组成完整 pipeline，说明真实断裂病例需要独立评价；真实腰椎金属植入物的 deep-MAR 研究进一步证明金属伪影应优先用真实病例校验；2024 年 3D 椎体分割研究还直接观察到低骨密度情况下的相邻椎体融合和单椎体分裂失败，支持把 low-density 病例与 false merge/false break 单独分析。2026 年 VertebraFormer 把椎体分割、编号和病灶定位统一到结构感知多任务框架，并采用多域与 leave-one-domain-out 方案研究泛化；同年开放的 1,460 例椎体体部数据与 Residual-Encoder nnU-Net 研究进一步抬高了通用 CNN baseline 的强度。因此本文不能把“采用 Transformer”本身作为主要创新，也不能只与弱 CNN 对照，而应重点验证三维多尺度骨科适配、边界/拓扑约束、困难样本、不确定性局部精修以及从体素分割到物理空间表面的误差闭环。

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

根据 uncertainty map 选择 Top-k 百分位或高于阈值的体素，并对其连通区域进行适度膨胀得到候选 ROI。首先使用 uncertainty→error AUROC、AUPRC、错误/正确体素平均 entropy、Top-percent error recall、ROI error rate 与 ROI fraction 定量判断预测熵是否真的集中于错误区域，而不是仅依赖热图主观判断。与此同时，对模型概率输出增加体素级 calibration 评价：采用固定分箱的 Expected Calibration Error（ECE）与 Maximum Calibration Error（MCE），并报告 multiclass Brier score、negative log-likelihood（NLL）、mean confidence、体素 accuracy 与 confidence gap。对超大三维体积采用固定随机种子的体素下采样，以保证不同实验之间可复现比较。工程中进一步实现了三维局部残差精修网络：将 CT/骨窗等影像通道、coarse probability/logits 与 uncertainty 作为输入预测残差，只在 ROI mask 内更新 coarse logits，ROI 外严格保留原预测。二阶段训练中 coarse 分割冻结，精修损失只在 ROI 内归一化，并记录精修前后 ROI/global error delta。正式 validation 消融已完成无精修、uncertainty ROI 精修及 full-volume second-pass 对照；虽然部分候选 Dice 上升，但 Recall、fragmentation、病例稳定性和耗时综合恶化，因此最终判定 **REFINEMENT=FAIL**，正式 independent test 保持 refinement disabled。

### 3.7 质量控制与可追溯性

每例数据保存原始/处理后 shape、spacing、origin、direction、强度统计、DICOM series 数量和 QC warning。所有训练实验保存 config、患者级 split、随机种子、checkpoint、逐病例指标和代码版本。测试集只用于最终评估，不能反复用于选择 loss 权重或模型结构。

为避免工程 smoke 被误写成论文实验，训练/评估入口设置 formal preflight：在正式运行前检查病例级 split 泄漏、官方 test-private 误用、人工 QC、pipeline/输入通道、标签范围与 `num_classes`。本 pilot 已在显式 `--allow-cpu` 条件下通过 formal readiness 并完成 CPU 正式流程；GPU 仅作为后续扩大实验规模时的加速条件，不再作为当前方法学完成与否的门槛。当前 CTSpine1K/VerSe `1–25` 的 `C1–L6` 映射只用于 QC/Web 可读显示，不改变原始标签值，也不提前决定正式任务采用 binary、multi-class semantic 或 instance segmentation。

### 3.8 物理空间表面重建与工程几何控制

分割 mask 首先在 `(Z,Y,X)` 体素数组上使用 Marching Cubes 提取三角面，再依据影像 `spacing`、`origin` 与 `direction` 显式转换至物理 `(X,Y,Z)` 毫米坐标。Web 轻量展示保留全分辨率网格，同时提供可控的 vertex-clustering 简化，并记录顶点/面缩减率、表面积变化以及全分辨率—简化网格的顶点近邻误差。为降低聚类平均对尖锐骨性结构的平滑，工程候选进一步以相邻顶点法向变化作为轻量曲率/关键边缘代理，在聚类代表点计算时提高高特征顶点权重；该策略默认关闭，必须在 prediction surface 上完成误差与拓扑消融后才可作为正式重建方案。为区分模型误差和预处理误差，项目还单独比较原始标签与 1 mm nearest-neighbor 重采样标签的物理表面差异；该分析只用于量化重采样离散化，不作为模型分割性能或临床测量精度。

---

## 4 实验设计

### 4.1 数据集

当前 formal pipeline pilot 使用 CTSpine1K `MSD-T10` 的 10 例真实 CT+label：9 例官方 `trainset` 与 1 例官方 `test_private`。10/10 病例均完成 1 mm 重采样、HU clip、逐病例 z-score、骨窗生成、nearest-neighbor label 重采样、自动审计和人工 QC。任务固定为 `binary_semantic`；训练/评价时将原始前景标签 `1..25` 统一映射为二值前景 1，而 Web/QC 仍保留原始标签及 `1–25 → C1–L6` 可读映射，不篡改源标签。该 10 例数据用于本研究的**正式流程 pilot**并产生本文当前 Results，但病例规模极小，不能替代扩大数据规模后的正式临床/多中心研究，也不能据此声称临床泛化能力。

临床数据仅在获得合法研究授权且完成脱敏后纳入；临床数据的机构来源、纳排标准、伦理/授权信息和划分策略必须在最终论文中明确说明。

### 4.2 数据划分

本 pilot 采用固定 patient-level split：`liver_0`—`liver_6` 为 7 个 train cases，`liver_7/liver_8` 为 2 个 validation cases，官方 `test_private liver_169` 为唯一 independent test case；split 文件为 `data/splits/ctspine1k_msd_t10_binary_formal_pilot_v1.json`。所有输入、loss、sampling、augmentation、困难样本、uncertainty/refinement 和三维工程选择均只依据 train/validation 完成。正式 test 在锁参提交 `eb0a824c34af4f7d900432e169759115f99a2687` 已推送并确认 `HEAD == origin/main` 后才首次访问，且仅运行一次；测试结果不再用于参数选择。

### 4.3 对比方法

受当前 10 例 pilot 样本规模与 CPU 计算条件限制，本阶段不伪造未运行的跨架构 baseline。本文当前 Results 的方法比较限定于同一 SegFormer3D 工程链内的 v11—v23 真实变体，包括输入、loss、sampling、augmentation、困难样本与 refinement 消融。nnU-Net、Residual-Encoder nnU-Net、UNETR、Swin UNETR 及近期脊柱专用方法只作为后续扩大数据规模后的外部强 baseline 计划；在实际完成同一 split/spacing/评价协议下的运行前，不填写其性能数字。

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

主指标包括 Dice Similarity Coefficient、IoU、HD95、ASSD、Precision 和 Recall。对于结构连通性，候选指标包括 clDice、连通域数量差异、错误粘连与断裂数。若采用 multi-class 椎体任务，逐病例 macro 只对真值或预测实际出现的前景类别求平均，双方都缺失的类别不计入，以避免空类别产生虚高分数，并同时输出逐类别指标。对于 uncertainty，额外报告 error AUROC/AUPRC、错误/正确体素平均 entropy、Top-percent error recall、ROI error rate 与 ROI fraction；对于概率校准，报告 ECE、MCE、Brier score、NLL、mean confidence、体素 accuracy 和 confidence gap。工程指标包括参数量、FLOPs、峰值显存、单病例预处理时间、推理时间和三维重建时间。

由于当前 validation 仅 2 例、formal independent test 仅 1 例，本 pilot 只报告逐病例数值与两例 validation mean，不进行无意义的显著性检验，也不报告不稳定的 median[IQR] 或 p 值。后续扩大病例规模后，再根据分布采用配对 t 检验或 Wilcoxon signed-rank test，并同时报告效应量或置信区间。

### 4.6 实现细节

当前实验环境为项目内 Python 3.11.7、PyTorch `2.1.0+cpu`、MONAI 1.2.0，在 Ryzen 7 8745H CPU 上执行。v13 使用 `ct_normalized` 单通道输入：HU clip `[-1000,2000]` 后逐病例 z-score，目标 spacing 为 1 mm。随机种子为 42，training ROI=`64×64×64`，batch size=1，patches/case=4，Bernoulli foreground probability=0.25；优化器为 AdamW（lr=`5e-5`，weight decay=`0.01`），warmup 1 epoch 后使用 cosine annealing warm restarts。augmentation 保留 flip-only。epoch 2 起冻结 encoder、decoder feature 与 BatchNorm running statistics，仅继续更新 `linear_pred`。full-volume inference 使用 sliding-window ROI=`128×128×128`、overlap=0.25、sw_batch_size=1，关闭 AMP；prediction decision 固定为 `evaluate.py` 的 softmax+argmax，refinement 关闭。

---

## 5 结果

> **结果口径。** 本节 5.1—5.8 的数值均来自当前固定 patient-level split 中 `liver_7/liver_8` 的 **engineering / validation**，用于方法选择和工程闭环验证，不是 independent test。`test_private liver_169` 在最终参数锁定前保持隔离；正式 independent-test 结果仅允许在锁参提交并推送后一次性产生，并单列于 5.9。任何 mesh vertex-nearest distance 均为三维重建工程指标，不冒充分割评价中的临床 HD95/ASSD。

### 5.1 最终 validation 主模型

当前 validation 最终候选为 v13：SegFormer3D、CT-only、Region+Boundary（权重 `1.0/0.1`，topology=`0`）、Bernoulli sampling（foreground probability=`0.25`，patches/case=`4`）、flip-only augmentation、`64×64×64` training ROI、AdamW（峰值 lr=`5e-5`）。epoch 2 起冻结 encoder、decoder feature 与 BatchNorm running statistics，仅继续更新 `linear_pred`；验证采用 full-volume inference 与当前 `evaluate.py` 的 softmax/argmax class decision。

| Validation case | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | Pred/GT FG | Component error | False break | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `liver_7` | 0.04531 | 0.02318 | 0.02799 | 0.11879 | 197.3914 | 55.3591 | 4.2437× | 1561 | 69 | 56.75 |
| `liver_8` | 0.06411 | 0.03312 | 0.04047 | 0.15420 | 174.5083 | 47.6139 | 3.8102× | 1526 | 60 | 93.28 |
| **Mean** | **0.05471** | **0.02815** | **0.03423** | **0.13649** | **185.9498** | **51.4865** | **4.0270×** | **1543.5** | **64.5** | **75.01** |

该结果说明当前模型仍处于明显低精度状态：Dice 仅约 0.055，prediction foreground 约为 GT 的 4 倍，surface distance 很大，且 prediction component 数远高于 target，存在显著 fragmentation。本文不将其包装为高精度临床模型，而把其作为完成统一数据—训练—不确定性—三维—Web 科研闭环的真实 validation 基线。

### 5.2 输入通道消融

在 loss、optimizer、scheduler、sampling、ROI、augmentation、seed、freeze policy 与 full-volume validation 保持一致的条件下，比较 CT-only 与 CT+bone-window。CT-only（v11）两例平均 Dice/IoU/Precision≈`0.05466/0.02812/0.03425`，HD95/ASSD≈`186.05/51.52 mm`，prediction/GT foreground ratio≈`4.00×`，ECE/Brier/NLL≈`0.01084/0.04693/0.10288`。加入 bone-window 的 v12 平均 Dice/IoU/Precision降至≈`0.02803/0.01421/0.01422`；Recall 虽升至≈`0.96594`，但 foreground ratio 达≈`67.96×`，HD95/ASSD 恶化至≈`256.41/83.77 mm`，ECE/Brier/NLL≈`0.39319/0.80051/3.40655`。因此当前 normalization/architecture 下双通道导致严重 foreground overprediction，输入消融选择 **CT-only**。

### 5.3 联合损失消融

| Validation setting | Mean Dice | HD95 (mm) | ASSD (mm) | 结构/校准观察 | 结论 |
|---|---:|---:|---:|---|---|
| Region（v11） | 0.05466 | 186.0500 | 51.5220 | baseline | 基线 |
| Region + Boundary（v13） | **0.05471** | 185.9498 | 51.4865 | FG ratio≈4.027×；校准基本维持 Region 水平 | **选用** |
| Region + Topology（v14） | 0.05451 | 183.9914 | 50.5799 | component/false-break 有改善信号，但 overprediction 与 calibration 代价更明显 | 不选 |
| Region + Boundary + Topology（v15，best epoch2） | 0.05452 | 184.5759 | 50.7512 | FG ratio≈4.8108×，component error≈1540，false break≈60 | 不选 |

Boundary 相对 Region 的 Dice 增益约 `5.19×10^-5`，HD95/ASSD 仅改善约 `0.1002/0.0355 mm`，属于极弱证据，不能宣称显著边界收益。Topology 组合在 surface 与部分结构指标上出现改善信号，但伴随更强前景过预测与较差 calibration，因此综合选择 v13 Region+Boundary，而不是按单一 surface 指标选型。

### 5.4 Sampling、augmentation 与困难样本消融

Sampling 方面，v16 fixed-per-case 的跨 epoch sampling 更稳定，但 mean Dice≈`0.04575`、foreground ratio≈`6.51×`、ECE≈`0.02964`，整体劣于 v13；v17 boundary-hard mean Dice≈`0.03731`、foreground ratio≈`36.26×`、HD95≈`206.50 mm`、ECE≈`0.19569`，出现严重 foreground overprediction 与 calibration 崩坏。因此保留 v13 Bernoulli sampling。

Augmentation 方面，v18 标准几何增强（±10° rotation、scale 0.9–1.1）mean Dice≈`0.05535`，虽略高于 v13，但 foreground ratio≈`7.56×`、Precision≈`0.03134`、ECE≈`0.02716` 均劣化；v19 gamma 按 STOP 规则提前终止，v20 Gaussian noise 完成三轮但综合指标不取代 v13，v21 ±50 HU shift 亦按规则停止。因此最终保留 **flip-only**。

模型驱动困难样本采样同样未带来收益：v22 high-loss epoch1/2 mean Dice=`0.04728839/0.04728457`，v23 high-uncertainty=`0.00963475/0.01010232`，均明显低于 v13 并按预设规则停止。训练集中 `liver_0/liver_1` 的 z-spacing 约 5 mm，冻结 v13 guidance 下 candidate loss/uncertainty 分别高约 `12.9%/13.0%`，说明厚层病例具有更强 patch-level difficulty signal；但 validation 的 `liver_7/liver_8` 均为 1 mm，且现有 metadata 不足以可靠标记 metal/fracture/low-density subgroup，因此本文不伪造这些 subgroup 的 Dice。

### 5.5 不确定性与校准

| Validation case | Error AUROC | Error AUPRC | Top-10% error recall | ECE | MCE | Brier | NLL | Confidence gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `liver_7` | 0.92949 | 0.33354 | 0.72891 | 0.01418 | 0.07593 | 0.05465 | 0.12079 | 0.01411 |
| `liver_8` | 0.94466 | 0.33014 | 0.79450 | 0.00767 | 0.03988 | 0.03969 | 0.08571 | 0.00767 |

两例 error voxel 的平均预测熵约为 correct voxel 的 `10.41×/12.42×`；Top-10% uncertainty 区域分别覆盖约 72.9% 与 79.4% 的错误。该结果支持 predictive entropy 作为当前 validation 条件下的 **error indicator / QC signal / refinement trigger**，但两例样本量极小且整体 segmentation Dice 很低，不能据此宣称模型已充分校准或具备临床可靠性。

### 5.6 不确定性 ROI 精修消融：FAIL

Refinement 仅使用 7 个 train cases 训练，在 `liver_7/liver_8` 上完成 Top-5/10/20% × dilation 0/1/2 的 3×3 validation grid 及 full-volume second-pass。canonical reconstruction 的 prediction mismatch 两例均为 0，entropy max absolute error≈`9.86×10^-7`，所有 ROI-only candidate 的 `outside_roi_changed_fraction=0`，因此比较不依赖重新运行 coarse model。

数值最强候选 Top-20%+dilation2 的两例均值为 Dice≈`0.07407384`、IoU≈`0.03865424`、Precision≈`0.08146804`、Recall≈`0.06965428`、HD95≈`175.9586 mm`、ASSD≈`47.1264 mm`。然而相对 coarse v13，Recall 从≈`0.13649` 降至≈`0.06965`，component error 从≈`1543.5` 恶化到≈`2397.5`，false break 从≈`64.5` 增至≈`138`；`liver_7` 的 surface 指标反向恶化，mean CPU pipeline time 也由≈`75.01 s` 增至≈`99.95 s`。因此按区域、表面、拓扑、两例稳定性与计算成本综合判定 **REFINEMENT=FAIL**，最终 pipeline 禁用 refinement，保留 v13 coarse `best.pt`。该失败结果是本文的重要负结果，而不是需要隐藏的异常。

### 5.7 真实 prediction 三维重建工程验证

对 v13 validation 的真实 prediction 完成 physical-space Marching Cubes、2.0 mm feature-preservation vertex-clustering 与 0.4 mm SDF surface。`liver_7/liver_8` 原始 prediction mesh 顶点约为 `1,369,586/1,340,666`；2.0 mm + feature strength=8 后约为 `298,840/296,483`，顶点缩减约 `78.18%/77.89%`。简化相对原 prediction surface 的 vertex-nearest engineering ASSD≈`0.55845/0.54806 mm`、HD95≈`1.09434/1.08294 mm`。

0.4 mm SDF 在两例上分别保持 connected components `1564→1564` 与 `1528→1528`；SDF 相对原 prediction surface 的 engineering ASSD≈`0.02919/0.02929 mm`、HD95≈`0.06790/0.06671 mm`。prediction-vs-GT 的 vertex-nearest engineering ASSD≈`53.4817/46.9266 mm`、HD95≈`186.1664/173.3450 mm`，与前述低分割质量一致。这里的 vertex-nearest 数字只用于三维工程误差跟踪，不能替代 segmentation evaluation 中基于体素/表面的正式 HD95/ASSD。

### 5.8 Web 科研原型实机验收

Web 原型未重新推理或生成虚构 prediction，而是从 `experiments/<evaluation>/predictions/<case>/prediction.nii.gz` 读取已保存的真实 evaluation 产物。Edge 实机中，`results-review` 已显示 v13 validation 的 prediction MPR overlay 与 predictive entropy / uncertainty overlay；`research-3d` 已加载 `liver_8` 2.0 mm prediction mesh（`296,483` 顶点 / `591,833` 三角面）以及 SDF σ=0.4 mm surface（`1,340,319` 顶点 / `2,672,566` 三角面），并验证 GT/prediction 双来源切换。锁参前 `/api/research/cases` 仅暴露 `liver_0`—`liver_8` 共 9 例，`test_private liver_169` 保持隔离。

### 5.9 正式独立测试

最终参数锁定记录为 `docs/10_final_parameter_lock.md`，对应锁参提交 `eb0a824c34af4f7d900432e169759115f99a2687`。该提交已推送并确认 `HEAD == origin/main` 后，才对官方 `test_private liver_169` 执行**第一次且唯一一次** FINAL FORMAL INDEPENDENT TEST。测试严格使用锁定的 v13 config、`best.pt`、softmax/argmax decision 和 `refinement=disabled`，formal preflight 为 `ready=true`、0 error/0 warning。

| Independent test case | Dice | IoU | Precision | Recall | HD95 (mm) | ASSD (mm) | Pred/GT FG | Pred/GT components | Component error | False merge | False break | Inference (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `liver_169` | **0.02878** | 0.01460 | 0.02090 | 0.04622 | 136.8722 | 43.9720 | 2.2118× | 236 / 1 | 235 | 0 | 29 | 9.41 |

正式 test 的 uncertainty error AUROC/AUPRC 为 `0.86424/0.29665`，Top-10% uncertainty error recall=`0.54993`；ECE/MCE/Brier/NLL=`0.02740/0.08782/0.08328/0.23559`，confidence gap=`0.02740`。尽管体素 calibration 数字并未异常失控，但 Dice、Precision、Recall 和结构连通性仍很差，因此不能把较低 ECE 解释为模型具有临床可靠性；高 background 占比会显著影响体素级 accuracy/confidence 指标。

独立 prediction 的 physical-space full mesh 为 `365,247` 顶点 / `724,694` 三角面；2.0 mm + feature strength=8 简化后为 `81,353` 顶点 / `160,384` 三角面，顶点减少约 `77.73%`，相对原 prediction surface 的工程 ASSD/HD95=`0.56490/1.07159 mm`。SDF σ=0.4 mm 保持 components `236→236`，SDF-vs-original 工程 ASSD/HD95=`0.02536/0.06367 mm`。prediction-vs-GT vertex-nearest engineering ASSD/HD95=`41.1398/131.8726 mm`，且 prediction 与 GT 的 size、spacing、origin、direction 完全一致。上述 vertex-nearest 距离仍只属于三维工程检查，不替代本表中的正式 segmentation HD95/ASSD。

Web `results-review` 已识别该独立 evaluation，prediction 与 uncertainty MPR 接口均返回 200；Edge `research-3d` 已实际加载 independent 2.0 mm prediction mesh 与 SDF σ=0.4 mm WebGL。测试完成后未再修改 threshold、loss、sampling、augmentation、refinement 或模型参数，也不重复运行该 formal test。

---

## 6 讨论

当前 validation 证据首先说明，本项目的最大限制不是“缺少复杂模块”，而是基础分割精度本身仍然很低。v13 mean Dice 仅约 0.0547，同时存在约 4 倍 foreground overprediction、超过 180 mm 的 mean HD95 以及千级 component count error。因此，Boundary、Topology、hard mining 或 refinement 带来的局部数字变化必须放在这一低基线背景下解释，不能用小幅相对改善掩盖绝对性能不足。

Boundary 项仅带来极弱的 Dice 与 surface 改善；Topology 项虽然在 HD95/ASSD、component error 和 false break 上出现一定正向信号，却伴随更明显的前景过预测和 calibration 代价。这说明对当前二值骨结构任务而言，单纯增加约束项并不能自动转化为综合性能提升。CT+bone-window 同样未验证最初假设：在当前 case-wise normalization 与 architecture 下，它把 Recall 推高的同时造成约 68 倍 foreground overprediction，因而 CT-only 更稳健。

不确定性模块提供了相对清晰的正结果和负结果。正面看，predictive entropy 在两例 validation 上对错误具有较高 AUROC，并能让 Top-10% uncertainty 覆盖大部分错误，适合作为科研复核/QC 风险提示。负面看，将其直接用于 ROI refinement 并没有得到稳健综合改善：Dice 上升伴随 Recall 和 topology fragmentation 显著恶化，且增加推理时间。因此本文保留 uncertainty 作为“发现哪里可能错”的辅助信号，而拒绝把 refinement 作为最终分割步骤。

三维工程结果进一步展示了“模型误差”和“重建误差”必须分开。2.0 mm feature-weighted 简化在减少约 78% 顶点的同时，把相对原 prediction surface 的工程 HD95 控制在约 1.1 mm；0.4 mm SDF 对原 prediction surface 的工程 HD95 约 0.067 mm 且保持 component count。这说明三维展示链本身可在可控几何误差下工作，但 prediction-vs-GT surface discrepancy 仍然很大，瓶颈仍来自 segmentation，而不是 Web mesh 简化或 SDF 表面生成。

当前 Web 原型的价值主要是可追溯科研复核：同一 evaluation 下可以检查 prediction、uncertainty、3D surface 与指标，且不重新生成模型结果。它有助于将模型失败以可视化方式暴露出来，但不构成医疗诊断系统，也不能弥补模型准确率不足。

### 6.1 失败病例与失败模式

正式 independent test `liver_169` 是当前最重要的失败病例证据。其 prediction/GT foreground ratio=`2.21×`，同时出现 `236/1` 的 component 数量差异和 `29` 个 false break，说明问题不仅是边界位置偏移，而是前景碎片化和错误前景区域大量存在。Web MPR、prediction mesh 和 SDF 复核进一步确认这些错误来自锁定模型的真实 prediction，而不是后续简化或 SDF 生成过程；prediction 与 GT 的 size、spacing、origin、direction 完全一致，也排除了明显的物理空间配准元数据不一致作为主要解释。

正式 independent test 将 validation 阶段的主要失败模式进一步暴露出来。独立 Dice=`0.02878`，低于两例 validation mean=`0.05471`；Precision=`0.02090`、Recall=`0.04622` 均很低，且 prediction/GT foreground ratio=`2.21×`、components=`236/1`，说明明显的前景过预测与 fragmentation 仍然存在。独立 HD95=`136.87 mm` 虽数值上低于 validation mean HD95≈`185.95 mm`，但单病例表面范围、目标体积与错误分布不同，不能据此认定泛化改善；区域重叠反而明确下降。

Uncertainty 在独立 test 上仍保留一定错误排序能力（AUROC=`0.86424`），但弱于 validation 的 >0.92；Top-10% uncertainty 仅覆盖约 55.0% 错误，也低于 validation 的约 72.9%–79.4%。这支持 entropy 作为风险提示信号，但同时表明其跨病例稳定性有限。独立 ECE≈`0.0274` 不能与低 Dice 脱离解释，因为二值体数据以 background 为主，较高总体 confidence/accuracy 并不等价于前景分割可靠。

因此 independent test 的作用严格限定为评价锁定 pipeline，而不是继续选方法。测试完成后没有依据结果反向调整 threshold、refinement、sampling、augmentation 或模型参数；较差结果被直接保留为当前小样本、模型欠拟合和结构碎裂风险的真实证据。

---

## 7 局限性

本研究当前存在至少以下局限。第一，validation 仅有 2 例，正式独立测试也只有 1 个 `test_private` 病例，统计稳定性非常有限，不能据此得出临床泛化结论。第二，当前 v13 的绝对分割精度低，foreground overprediction、surface distance 和 component fragmentation 均明显，项目更适合作为完整科研工程闭环与方法探索，而非高精度临床模型。第三，现有 metadata 不足以可靠构建 metal artifact、fracture、low-density 等困难病例 subgroup，厚层信号也仅在 train patch-level 得到观察，不能伪造 subgroup 结果。第四，Boundary/Topology/refinement 的结论均受当前小样本与低基线影响，扩大数据规模后需要重新验证。第五，当前主要在 CPU 环境完成实验，墙钟时间只代表本机条件，不用于跨硬件效率比较。第六，mesh vertex-nearest engineering error 并非临床表面距离或测量精度；虽然已复核 spacing/origin/direction、简化误差和 SDF topology preservation，仍需更严格的临床几何测量验证。第七，尚无合法授权的临床脱敏数据与前瞻性验证，科研 Web prototype 与合规医疗器械之间仍存在显著的工程、伦理和监管距离。

---

## 8 结论

本研究完成了从骨科 CT 标准化、患者级划分、SegFormer3D 训练与系统消融、uncertainty/calibration、失败可追溯的 ROI refinement，到 physical-space 3D reconstruction 与 Web 科研复核的完整工程链。统一 validation 消融后，最终锁定 CT-only、Region+Boundary、Bernoulli sampling、flip-only 的 v13 coarse pipeline；两例 validation mean Dice≈`0.05471`。Predictive entropy 能够提示部分错误区域，但 uncertainty ROI refinement 综合判定为 **FAIL**，因此最终 pipeline 禁用 refinement。

在锁参提交推送并确认远端一致后，唯一一次正式 independent test 得到 Dice=`0.02878`、HD95=`136.87 mm`、ASSD=`43.97 mm`，且存在 `236/1` 的 prediction/GT component 差异，进一步证明模型绝对精度与结构连续性仍远未达到临床应用要求。独立 test entropy error AUROC=`0.86424`，说明 uncertainty 仍有 QC 提示价值，但跨病例稳定性有限。与此同时，2.0 mm 特征加权简化将独立 prediction 顶点减少约 `77.73%`，0.4 mm SDF 保持 components `236→236`，并已在 Edge WebGL2 完成 prediction/uncertainty/3D 实机复核。

因此，本项目当前最可信的成果不是“高精度临床分割”，而是一个结果可追溯、失败不隐藏、严格区分 validation/test、能够将分割—不确定性—三维—Web 串联起来的科研工程闭环。后续研究应优先扩大训练与独立测试样本规模、引入强基线和合法临床/多中心验证，而不是围绕当前单病例 test 继续调参。

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
12. Hofmann F O, Auhage L A, Dexl J, et al. Vertebral body segmentation in CT: An open dataset, deep-learning models and comparison to existing models. European Journal of Radiology, 2026, 204:113118.
13. Glessgen C, Cyriac J, Yang S, et al. A deep learning pipeline for systematic and accurate vertebral fracture reporting in computed tomography. Clinical Radiology, 2025, 83:106827.
14. Ye K, Pan B, Li J, et al. Deep learning model trained using multi-energy computed tomography (CT) data shows better metal artifact reduction for lumbar CT imaging. Clinical Radiology, 2025, 90:107076.
15. Kervadec H, et al. Boundary Loss for Highly Unbalanced Segmentation. Medical Image Analysis, 2021.
16. Shit S, Paetzold J C, Sekuboyina A, et al. clDice—A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR, 2021.
17. Abutalip K, Saeed N, Sobirov I, et al. EDUE: Expert Disagreement-Guided One-Pass Uncertainty Estimation for Medical Image Segmentation. arXiv:2403.16594, 2024.
18. Guo X, Lin X, Yang X, et al. UCTNet: Uncertainty-guided CNN-Transformer hybrid networks for medical image segmentation. Pattern Recognition, 2024, 152:110491.
19. Berger A H, Stucki N, Lux L, et al. Topologically Faithful Multi-class Segmentation in Medical Images. arXiv:2403.11001, 2024.
20. Wyburd M K, Dinsdale N K, Jenkinson M, Namburete A I L. Anatomically plausible segmentations: Explicitly preserving topology through prior deformations. Medical Image Analysis, 2024, 97:103222.
21. Xiong X, Graves S A, Gross B A, et al. Lumbar and Thoracic Vertebrae Segmentation in CT Scans Using a 3D Multi-Object Localization and Segmentation CNN. Tomography, 2024, 10(5):738-760.

> 机器可用参考文献库已建立于 `paper/references.bib`（当前 42 条英文核心条目），结构化文献矩阵见 `docs/08_literature_matrix.md`（当前 44 条）。正式投稿前仍需统一期刊格式，并对国内文献与最新正式发表版本再次核验。
