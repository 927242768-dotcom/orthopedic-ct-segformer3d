# 08｜骨科 CT 分割—不确定性—三维重建结构化文献矩阵

> 建立日期：2026-08-16
>
> 用途：为 `paper/manuscript_zh_v0.1.md` 的 Introduction / Related Work / Methods / Discussion 提供可追溯的文献骨架。
>
> 核验原则：优先出版社、CVF/NeurIPS/PMLR 等会议官方页面、期刊官网或 arXiv 原始记录；不以二手博客/聚合网页作为题录依据。`paper/references.bib` 与本矩阵同步维护。
>
> **重要边界：表中“与本项目关系/可支持论点”是项目组的研究归纳，不等同于原作者声称本方法适用于骨科 CT。**

---

## 1. 3D 医学分割基础与 Transformer/现代 ConvNet

| ID | 文献 | 年份/出处 | 核心作用 | 与本项目关系 / 可支持论点 | 优先级 |
|---|---|---|---|---|---|
| M01 | Çiçek et al., **3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation** | 2016, MICCAI | 经典 3D encoder-decoder；三维卷积医学分割基础 | 说明 3D 体素级网络是骨科 CT 分割的直接技术基线之一 | A |
| M02 | Milletari et al., **V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation** | 2016, 3DV | 3D FCN + Dice-style objective | 支撑 Dice 类区域损失及体数据分割基线的历史来源 | A |
| M03 | Isensee et al., **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation** | 2021, Nature Methods | 自动配置预处理、网络、训练策略的强医学分割 baseline | 正式论文应优先把 nnU-Net 作为强 CNN 对照，而非只和弱模型比较 | A |
| M04 | Xie et al., **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** | 2021, NeurIPS | hierarchical Transformer encoder + lightweight MLP decoder | SegFormer3D 的二维设计思想来源；支持“轻量、多尺度”叙事 | A |
| M05 | Hatamizadeh et al., **UNETR: Transformers for 3D Medical Image Segmentation** | 2022, WACV | Transformer encoder + U-shaped decoder 的 3D 医学分割 | 说明 Transformer 早已进入 3D 医学分割，不能宣称“首次 Transformer 骨分割” | A |
| M06 | Zhou et al., **nnFormer: Interleaved Transformer for Volumetric Segmentation** | 2021, arXiv | 卷积与 self-attention 交织、volume attention、skip attention | 可作为 SegFormer3D 的重型/高表达 3D Transformer 对照思想；本库暂引用已核验的 arXiv 版本 | A |
| M07 | Tang et al., **Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis** | 2022, CVPR | Swin UNETR + 3D CT 自监督预训练 | 支撑层级窗口 Transformer 在 3D CT 中的有效性与预训练方向 | A |
| M08 | Wang et al., **TransBTS: Multimodal Brain Tumor Segmentation Using Transformer** | 2021, MICCAI | CNN encoder + Transformer bottleneck 的 3D 分割 | 展示“局部卷积 + 全局 Transformer”的另一设计路线 | B |
| M09 | Roy et al., **MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation** | 2023, arXiv | Transformer-inspired 3D ConvNeXt，强调大核与可扩展 ConvNet | 提醒论文不要把“Transformer”本身当成性能保证；现代 ConvNet 仍是强对照 | A |
| M10 | Gao et al., **A Data-scalable Transformer for Medical Image Segmentation: Architecture, Model Efficiency, and Benchmark (MedFormer)** | 2022, arXiv | 面向不同数据规模的 3D Transformer | 支撑小数据医学场景中 inductive bias、效率和数据规模的重要性 | B |
| M11 | Perera et al., **SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation** | 2024, arXiv / CVPR workshop 项目记录 | 3D patch embedding、多阶段 encoder、spatial-reduction attention、MLP decoder | 本项目 backbone 直接来源；创新必须放在骨科数据链、loss、hard sample、uncertainty、3D/Web，而非冒称 backbone 自研。当前 BibTeX 只保留已核验的 arXiv 记录，不保留未从一手页面核实的期刊 DOI | A |

### 本组结论

1. **强 baseline 必须至少包含 SegFormer3D 与 nnU-Net 思路**；如果算力允许，再加入 UNETR/Swin UNETR/MedNeXt 中一项。
2. 本项目更合理的创新表述是“**骨科 CT 任务适配 + 边界/结构约束 + 困难样本 + 不确定性局部精修 + 三维几何质量验证**”，而不是“首次将 Transformer 用于骨骼”。
3. 3D Transformer 通常依赖 patch/crop/sliding-window，本项目真实 CT 处理后 shape 差异很大，因此 `128³` patch + sliding-window 的工程路线与该类文献背景一致，但具体 ROI 必须通过显存和验证集消融确定。

---

## 2. 脊柱 / 椎体 CT 数据集与任务方法

| ID | 文献 | 年份/出处 | 数据/任务 | 与本项目关系 / 可支持论点 | 优先级 |
|---|---|---|---|---|---|
| S01 | Sekuboyina et al., **A computed tomography vertebral segmentation dataset with anatomical variations and multi-vendor scanner data** | 2021, Scientific Data | VerSe 系列数据，跨厂商、解剖变异、椎体标注 | 主数据候选；适合检验 multi-vendor 泛化和 vertebra segmentation | A |
| S02 | Sekuboyina et al., **VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images** | 2021, Medical Image Analysis 73:102166 | 椎体检测、编号与分割 benchmark；374 CT / 355 patients / 4505 vertebrae | 可用于正式 benchmark 设计、domain shift 与困难病例分析 | A |
| S03 | Deng et al., **CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography** | 2025, Machine Learning for Biomedical Imaging；2021 首发预印本 | 1,005 CT、11,100+ 标注椎体，多来源 | 本轮已真实接入 `MSD-T10` 10 例工程子集；正式引用优先使用 2025 MELBA 版本，后续可扩大训练规模/跨来源验证 | A |
| S04 | Lessmann et al., **Iterative fully convolutional neural networks for automatic vertebra segmentation and identification** | 2019, Medical Image Analysis | 迭代式单椎体分割/识别 | 说明 vertebra instance/identification 不等于简单 binary segmentation；有助于最终任务定义 | A |
| S05 | Wasserthal et al., **TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images** | 2023, Radiology: AI | 大规模多器官/骨结构 CT segmentation | 可做弱外部泛化/标签来源参考；不能与专用 vertebra benchmark 混为同一任务 | A |
| S06 | Li et al., **VerFormer: Vertebrae-Aware Transformer for Automatic Spine Segmentation from CT Images** | 2024, Diagnostics 14(17):1859 | Vertebrae-aware Global block / VGQ；VerSe 2019/2020 | 直接证明已有 Transformer 脊柱 CT 研究；其作者也讨论固定 token 对多尺度的限制，可作为本项目 3D 多尺度路线的直接对照 | A |
| S07 | 国内肋骨 CT 三维分割与三维重组相关研究（见 `docs/02_literature_survey.md` 原始记录） | 国内文献 | 肋骨 CT segmentation + reconstruction | 用于国内研究现状；最终投稿前应回 CNKI/万方逐条核验题录和引用格式 | B |
| S08 | 国内椎体转移瘤 CT 2D/3D U-Net/ResUNet 对比研究（见原始调研） | 国内文献 | 病变/椎体 CT segmentation | 支撑国内骨科 CT 已有 2D/3D CNN 应用；正式中文参考文献需数据库复核 | B |
| S09 | Zhang et al., **SpineMamba: Enhancing 3D spinal segmentation in clinical imaging through residual visual Mamba layers and shape priors** | 2025, Computerized Medical Imaging and Graphics 123:102531 | 3D spine segmentation；Residual Visual Mamba + learnable 3D spine shape prior | 直接说明 2025 年脊柱 3D 分割已开始用 state-space model 与显式形状先验；本项目若主打 topology/shape 约束，必须与该类现代结构先验工作区分并做强 baseline | A |
| S10 | Yang et al., **Transformer-enhanced vertebrae segmentation and anatomical variation recognition from CT images** | 2025, Scientific Reports 15:34329 | VerSe 2019/2020；WNet segmentation + ViT typing + anatomical-variation attention | 支撑“椎体分割还需考虑解剖变异与全局上下文”；其公开方法也做 slice skipping/noise/scanner variation 鲁棒性测试，可启发本项目困难病例与 domain-shift 设计 | A |
| S11 | Du et al., **Structure-aware multi-task learning with domain generalization for robust vertebrae analysis in spinal CT** | 2026, npj Digital Medicine 9:217 | VertebraFormer；segmentation + numbering + lesion localization；multi-domain / leave-one-domain-out | 直接支撑患者级划分、跨域外部验证、结构感知与多任务分析；对本项目后续“公开数据→临床脱敏数据”的 domain generalization 设计价值高 | A |
| S12 | Hofmann et al., **Vertebral body segmentation in CT: An open dataset, deep-learning models and comparison to existing models** | 2026, European Journal of Radiology 204:113118 | 1,460 例公开 CT 椎体体部标签；Residual-Encoder nnU-Net；244 例测试 + 300 例外部 L3 定位验证 | 直接证明现代 residual-encoder nnU-Net 是当前椎体 CT 的强 baseline；正式比较不能只选传统 3D U-Net。其开放标签/权重也适合后续外部复核 | A |
| S13 | Glessgen et al., **A deep learning pipeline for systematic and accurate vertebral fracture reporting in computed tomography** | 2025, Clinical Radiology 83:106827 | 452 例胸腰椎 CT；最终 segmentation nnU-Net + fracture classifier；测试中正确分割 330/339 椎体 | 直接支撑“骨折/真实断裂病例必须单独分析”；也说明困难病例实验不能只在正常椎体上评价 topology/false break | A |
| S14 | Ye et al., **Deep learning model trained using multi-energy computed tomography (CT) data shows better metal artifact reduction for lumbar CT imaging** | 2025, Clinical Radiology 90:107076 | 93 例腰椎植入物多能 CT；基于真实金属伪影训练 deep-MAR，并跨能量比较 | 直接支撑金属植入物造成真实 domain shift/图像退化；本项目若做 metal-artifact subset，应优先使用真实金属病例验证，不能只依赖人工伪影模拟 | A |
| S15 | Xiong et al., **Lumbar and Thoracic Vertebrae Segmentation in CT Scans Using a 3D Multi-Object Localization and Segmentation CNN** | 2024, Tomography 10(5):738–760 | 3D multi-object localization + segmentation；VerSe2020 与放疗 CT；明确讨论低骨密度失败模式 | 论文直接指出骨密度很低时可能出现相邻椎体融合或单椎体分裂，正好支撑本项目 low-density subset 及 false merge/false break 指标；比仅把“低骨密度”当经验风险更有证据 | A |

### 本组结论

- 工程默认继续以**脊柱/椎体 CT**推进是合理的，因为公开数据、椎体边界、相邻结构粘连/断裂、不同层厚和三维重建都与当前创新方向高度契合。
- 但必须尽快由组内固定任务：`binary whole-spine`、`multi-class vertebra semantic`、`instance` 三者的 loss、`num_classes`、topology 定义和评价指标差异很大。
- 当前 10 例 CTSpine1K 只能作为**真实预处理/QC/工程 smoke**，不能因为已下载就成为正式论文 split。
- 2025–2026 直接脊柱工作已经进一步推进到 **Residual-Encoder nnU-Net 强基线、Mamba + 形状先验、解剖变异感知、结构感知多任务与 domain generalization**；同时已有真实骨折与腰椎金属植入物研究，因此本项目不能只用“Transformer + Dice”作为创新点，必须用边界/拓扑、困难样本、不确定性精修和物理三维误差闭环形成可验证差异。

---

## 3. 区域、边界与拓扑约束

| ID | 文献 | 年份/出处 | 方法要点 | 与本项目关系 / 风险 | 优先级 |
|---|---|---|---|---|---|
| L01 | Sudre et al., **Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations** | 2017, DLMIA | 处理类别不均衡的 Dice 变体 | Region loss 理论背景；骨结构前景占比低时具有参考价值 | A |
| L02 | Kervadec et al., **Boundary loss for highly unbalanced segmentation** | 2021, Medical Image Analysis | 通过距离/边界信息补充区域损失 | 本项目 Boundary Loss 的主要理论依据；应主要看 HD95/ASSD 是否改善 | A |
| L03 | Karimi & Salcudean, **Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks** | 2020, IEEE TMI | 直接优化 Hausdorff-related surrogate | 可作为边界损失替代/补充；避免只靠 Dice 判断表面质量 | A |
| L04 | Shit et al., **clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation** | 2021, CVPR | soft-skeleton / centerline topology overlap | 当前 soft-clDice 代码来源思想；但脊柱并非典型管状结构，必须做适用性消融 | A |
| L05 | Hu et al., **Topology-Preserving Deep Image Segmentation** | 2019, NeurIPS | persistent homology/topological constraints | 说明拓扑约束可显式进入 segmentation objective | B |
| L06 | Clough et al., **A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology** | 2019, arXiv | persistent-homology topological loss | 若 soft-clDice 对多椎体结构无效，可作为更一般的拓扑方向 | B |
| L07 | Berger et al., **Topologically Faithful Multi-class Segmentation in Medical Images** | 2024, arXiv | multi-class Betti matching / topology | 对多椎体 semantic 情形比单 binary clDice 更值得后续评估 | A |
| L08 | Wyburd et al., **TEDS-Net: Enforcing Diffeomorphisms in Spatial Transformers to Guarantee Topology Preservation in Segmentations** | 2024, Medical Image Analysis | deformation/topology-preserving segmentation | 提供 topology-preserving 的另一类结构保证思路 | B |

### 本项目必须保留的医学例外

`Topology better ≠ clinically better`。真实骨折可能在标签中就是断裂；如果拓扑损失强制把断端连起来，反而造成病理形态错误。因此正式实验必须把：

```text
普通病例
骨折/断裂病例（若数据存在）
相邻骨粘连错误
false merge
false break
component count error
```

分开分析。当前代码已加入 `false merge / false break / component count` 工程指标，后续需用真实 prediction 验证。

---

## 4. 预测不确定性、校准与局部精修

| ID | 文献 | 年份/出处 | 方法要点 | 与本项目关系 | 优先级 |
|---|---|---|---|---|---|
| U01 | Gal & Ghahramani, **Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning** | 2016, ICML/PMLR | MC dropout / approximate Bayesian uncertainty | 若 entropy 单次 softmax 不足，可扩展 MC dropout | A |
| U02 | Lakshminarayanan et al., **Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles** | 2017, NeurIPS | deep ensembles | 更强但计算更贵的不确定性 baseline；可作为资源允许时的对照 | A |
| U03 | Guo et al., **On Calibration of Modern Neural Networks** | 2017, ICML/PMLR | calibration、temperature scaling、ECE | 提醒“softmax entropy 高低”与“概率已校准”不是同一个问题 | A |
| U04 | Mehrtash et al., **Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation** | 2020, IEEE TMI | 医学分割 calibration + predictive uncertainty | 直接支撑医学 segmentation 中校准/不确定性评价必要性 | A |
| U05 | Abutalip et al., **EDUE: Expert Disagreement-Guided One-Pass Uncertainty Estimation for Medical Image Segmentation** | 2024, arXiv:2403.16594 | 利用多专家标注分歧监督 uncertainty；单次前向 | 说明医学分割 uncertainty 不应只画热图，还应与真实分歧/错误相关；本项目没有多标注者数据，因此只借鉴评价思想，不直接复现 | B |
| U06 | Guo et al., **UCTNet: Uncertainty-guided CNN-Transformer hybrid networks for medical image segmentation** | 2024, Pattern Recognition 152:110491 | uncertainty-guided vision transformer / CNN-Transformer hybrid | 与本项目 uncertainty-aware segmentation 思路相关；已核验 DOI `10.1016/j.patcog.2024.110491` | B |

### 本项目的验证顺序

当前实现故意先做低成本 entropy baseline：

```text
coarse logits
→ predictive entropy
→ uncertainty→error AUROC/AUPRC
→ Top-percent error recall / ROI error rate
→ 如果 entropy 确实覆盖错误
→ 再训练 ROI refinement
```

这一顺序比“先加一个 refinement network 再解释”为何有效更严谨。当前代码已经具备 uncertainty→error 定量指标与 ROI-only refinement 训练 loss；真正论文结论必须等 baseline checkpoint 后生成。

---

## 5. 三维表面重建、网格与连续表示

| ID | 文献 | 年份/出处 | 方法要点 | 与本项目关系 | 优先级 |
|---|---|---|---|---|---|
| R01 | Lorensen & Cline, **Marching Cubes: A High Resolution 3D Surface Construction Algorithm** | 1987, SIGGRAPH | 从标量场抽取等值面三角网格 | 当前 `mask→mesh` 基础算法；必须在 physical space 处理 spacing/origin/direction | A |
| R02 | Garland & Heckbert, **Surface Simplification Using Quadric Error Metrics** | 1997, SIGGRAPH | QEM mesh decimation | 后续高质量网格简化的重要基线；当前已先实现可解释 vertex-clustering Web baseline | A |
| R03 | Park et al., **DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation** | 2019, CVPR | 神经隐式 SDF 连续表面 | 对“高保真连续几何”有启发；不是当前 baseline 必需项 | B |
| R04 | Mescheder et al., **Occupancy Networks: Learning 3D Reconstruction in Function Space** | 2019, CVPR | 隐式 occupancy field | 提供另一类连续 3D 表达；适合 Discussion/Future Work | B |

### 本项目当前三维证据

工程已从文献方案推进到真实数据：

- 真实 `liver_0` label → physical-space Marching Cubes：131,983 vertices / 264,362 faces；
- 1.5 mm vertex-clustering Web 网格约减少 60% 顶点/面，vertex-nearest HD95 ≈ 0.707 mm；
- 原始 label→1 mm label 的真实 10 例表面离散化评估显示厚层数据几何扰动更大。

这些数字是**预处理/网格工程证据**，不是模型分割结果或临床测量精度。

---

## 6. 当前建议的论文引用主链

为了避免 Related Work 变成“论文列表”，首篇论文正文建议按以下逻辑引用：

1. **3D segmentation baseline**：3D U-Net / V-Net → nnU-Net；
2. **3D Transformer**：SegFormer → UNETR / nnFormer / Swin UNETR → SegFormer3D；
3. **spine CT**：VerSe / CTSpine1K / Lessmann / TotalSegmentator / VerFormer → SpineMamba / 2025 fracture nnU-Net / 2026 Residual-Encoder nnU-Net / VertebraFormer，并用真实金属植入物文献定义困难病例；
4. **loss**：Generalised Dice → Boundary Loss / Hausdorff surrogate → clDice / topology；
5. **uncertainty**：MC Dropout / Deep Ensemble / Calibration → 医学 segmentation uncertainty；
6. **3D geometry**：Marching Cubes / QEM → DeepSDF / Occupancy Networks 作为连续表示扩展。

这样可以自然形成研究空白：

> 既有工作分别解决了 3D 分割、边界/拓扑约束、不确定性估计或三维表面表示，但面向骨科 CT 的工程链仍需要同时处理多层厚、物理空间一致性、相邻骨结构错误、困难区域定位以及分割结果到物理空间表面的可追踪误差。因此本项目不把单一网络结构作为全部创新，而是验证“标准化数据链 + 轻量 3D Transformer + 区域/边界/拓扑约束 + 困难样本 + 不确定性局部精修 + 三维几何质量”的完整闭环。

---

## 7. 参考文献库维护状态

机器可用 BibTeX：

```text
paper/references.bib
```

本矩阵当前收录 **44 个方法/数据/重建条目（含 2 个国内待数据库二次核验条目）**；`paper/references.bib` 当前有 **42 个英文核心 BibTeX 条目**。本轮新增并核验 2026 开放椎体体部数据/Residual-Encoder nnU-Net、2025 椎体骨折 nnU-Net pipeline、2025 真实腰椎金属植入物 deep-MAR，以及 2024 直接讨论低骨密度导致椎体融合/分裂失败的 3D segmentation 工作；此前已核验 SpineMamba、Scientific Reports 解剖变异 Transformer、VertebraFormer，并保留对 CTSpine1K、VerSe、VerFormer、nnFormer、SegFormer3D、TEDS-Net、EDUE 等易错题录的纠正。英文核心条目优先依据官方出版页面、PubMed、会议 Open Access、出版社页面或 arXiv 原始记录核对。

后续扩展时遵守：

- 正式投稿前逐条检查作者顺序、页码、卷期、DOI；
- arXiv 工作若已有正式发表版本，优先换为正式版本；
- 国内条目必须回 CNKI/万方核验，不凭二手网页补题录；
- 不把仅“主题相近”的论文写成直接支持骨科 CT 结论；
- 每次新增/删除主引用时同时更新 `paper/references.bib` 与本矩阵。
