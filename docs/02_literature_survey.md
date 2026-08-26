# 02 国内外相关文献与公开数据集调研

> 版本：v0.2｜2026-08-16
>
> 用途：支撑开题/中期、模型方案、论文 Related Work 与数据集选择。后续每次新增重要文献，应同时在 `PROJECT_STATUS.md` 记录进度。

## 1. 调研范围

围绕以下 6 条主线：

1. SegFormer / SegFormer3D 与 3D Transformer 医学分割；
2. 骨科 CT/脊柱 CT 公开数据集；
3. 区域重叠损失；
4. 边界约束与表面距离；
5. 拓扑保持；
6. 困难样本、不确定性与三维重建。

当前已从“首批必读集合”扩展到 **44 条结构化文献矩阵**，详见 `docs/08_literature_matrix.md`；其中 **42 条英文核心题录**已同步为机器可用 `paper/references.bib`。2024–2026 年脊柱直接工作已补入 SpineMamba、解剖变异感知 Transformer、VertebraFormer、Residual-Encoder nnU-Net 强基线、椎体骨折 nnU-Net pipeline、真实腰椎金属植入物 deep-MAR，以及直接记录低骨密度椎体融合/分裂失败模式的 3D segmentation 工作；后续重点转向外部泛化和国内 CNKI/万方正式题录。

---

## 2. SegFormer 与 3D Transformer 主线

### 2.1 SegFormer（2021）

**Xie E, Wang W, Yu Z, et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. 2021.**

核心价值：

- 分层 Transformer encoder 输出多尺度特征；
- 不依赖位置编码，对不同测试分辨率更友好；
- 使用轻量 MLP decoder 聚合多层特征；
- 为后续 SegFormer3D 的“多尺度 + 轻解码”思想提供直接基础。

与本项目关系：

- 任务书强调多尺度骨结构识别；
- 骨科 CT 存在不同扫描分辨率和各向异性问题，“不依赖固定位置编码”的思路具有适配价值。

链接：`https://arxiv.org/abs/2105.15203`

### 2.2 UNETR（2021）

**Hatamizadeh A, Tang Y, Nath V, et al. UNETR: Transformers for 3D Medical Image Segmentation. 2021.**

核心价值：

- 将 3D 医学体分割表述为 sequence-to-sequence；
- Transformer encoder 学习长距离依赖；
- 多尺度 skip connection 与 U 型 decoder 结合。

与本项目关系：

- 是 3D Transformer 医学分割的重要基线之一；
- 可用于论文 Related Work 对比“重型 U 型 Transformer”与 SegFormer3D 轻量路线。

链接：`https://arxiv.org/abs/2103.10504`

### 2.3 nnFormer（2021）

**Zhou H-Y, Guo J, Zhang Y, et al. nnFormer: Interleaved Transformer for Volumetric Segmentation. 2021.**

核心价值：

- 交错卷积和 self-attention；
- 局部/全局 volume attention；
- skip attention 替代传统简单拼接/求和。

与本项目关系：

- 上游 SegFormer3D 的 Synapse/ACDC 处理部分明确参考 nnFormer pipeline；
- 可作为高精度但相对复杂的 Transformer 对比路线。

链接：`https://arxiv.org/abs/2109.03201`

### 2.4 SegFormer3D（2024）——本项目直接基础

**Perera S, Navard P, Yilmaz A. SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation. 2024.**

核心贡献：

- 面向 3D volumetric segmentation 的轻量分层 Transformer；
- 多尺度体特征上计算 attention；
- 使用简单 all-MLP decoder；
- 官方报告相较部分 SOTA 具有显著参数量/GFLOPs 优势；
- 在 Synapse、BraTS、ACDC 上进行验证。

与本项目关系：

- 可作为骨科 CT 模型 backbone；
- 原始实现并未提供骨科专用 HU/骨窗/DICOM pipeline；
- 原始 loss 主要是 CE/BCE/Dice/DiceCE，不包含本项目所需 boundary + topology 联合约束；
- 原始数据集非骨科专用，因此本项目的主要工作不能仅是“换数据跑一遍”。

链接：`https://arxiv.org/abs/2404.10156`

官方代码：`https://github.com/OSUPCVLab/SegFormer3D`

---

## 3. 骨科/骨结构 CT 数据与代表性工作

### 3.1 CTSpine1K（国内团队代表性工作）

**Deng Y, Wang C, Hui Y, et al. CTSpine1K: A Large-Scale Dataset for Spinal Vertebrae Segmentation in Computed Tomography. 2021.**

公开信息显示：

- 1,005 个 CT volumes；
- 超过 11,100 个标注椎体；
- 来源多样，包含不同脊柱情况；
- 面向 vertebra segmentation 与后续脊柱分析；
- 数据由多个公开来源整合，并提供 benchmark。

优点：

- 与“骨科 CT + 椎骨分割”高度匹配；
- 数据规模足以支持 Transformer baseline；
- 适合研究多中心/多来源泛化。

风险/待核对：

- 不同子数据来源许可证需要逐一遵守；
- 标签规范和部分可见椎体的标注规则需在训练前统一核查；
- 需要明确最终论文是“单骨/多椎体实例标签”还是“整体骨骼语义标签”。

链接：`https://arxiv.org/abs/2105.14711`

官方资源：`https://github.com/MIRACLE-Center/CTSpine1K`

### 3.2 TotalSegmentator（国外重要公开 CT 资源）

**Wasserthal J, Breit H-C, Meyer M T, et al. TotalSegmentator: robust segmentation of 104 anatomical structures in CT images. 2022/2023.**

论文公开信息：

- 训练数据包含 1,204 例 CT；
- 原版任务标注 104 个主要解剖结构，其中包含大量骨结构；
- 公开数据集可用于多解剖结构 CT 分割研究；
- 其工具后续扩展到更多结构。

优点：

- 有骨盆、椎体、肋骨、肩胛骨、锁骨、股骨等多种骨结构；
- 可用于预训练、迁移学习或外部泛化验证；
- 多样化临床 CT 有利于验证鲁棒性。

注意：

- 不同版本/子任务标签数不同，写论文时必须明确使用的数据版本；
- 必须核对数据集许可证与使用条件，不能只按代码仓库许可证推断训练数据许可。

论文：`https://arxiv.org/abs/2208.05868`

公开数据 DOI：`https://doi.org/10.5281/zenodo.6802613`

### 3.3 VerSe 2019/2020（脊柱 CT 的重要公开基准）

**Sekuboyina A, et al. VerSe: A Vertebrae Labelling and Segmentation Benchmark for Multi-detector CT Images. Medical Image Analysis, 2021.**

核心信息：

- 两届 VerSe 数据合计 374 个 multi-detector CT scans、355 名患者、4505 个逐椎体体素级标注；
- 数据包含不同视野、扫描协议、解剖变异和病理情况；
- 官方基准明确评估跨 challenge iteration 的 domain shift / generalization；
- VerSe 公开资源对脊柱 CT 任务尤其有价值，因为相邻椎骨形态高度相关，同时又存在骨折、金属植入、过渡椎等困难场景；
- 数据仓库说明使用 CC BY-SA 4.0，实际使用前仍需保留版本和许可记录。

论文：`https://arxiv.org/abs/2001.09193`

官方资源：`https://github.com/anjany/verse`

### 3.4 VerFormer（2024，骨科/脊柱 Transformer 直接相关工作）

**Li X, Hong Y, Xu Y, Hu M. VerFormer: Vertebrae-Aware Transformer for Automatic Spine Segmentation from CT Images. Diagnostics, 2024.**

与本项目的直接关系：

- 来自上海交通大学医学院附属瑞金医院骨科团队，是较新的脊柱 CT Transformer 工作；
- 设计 Vertebrae-aware Global (VG) block / VGQ 模块，强调利用全局上下文突出椎骨相关 token；
- 在 VerSe 2019/2020 上验证，并讨论金属植入、骨折、多中心/多厂商等数据特点；
- 该工作使用 2D 配置，作者也指出固定 token 尺寸会限制多尺度特征，因此可以作为本项目选择 **3D、多尺度 SegFormer3D** 路线的直接对照与论证材料，而不是简单照搬。

论文：`https://doi.org/10.3390/diagnostics14171859`

### 3.5 SpineMamba（2025，3D Mamba + 脊柱形状先验）

**Zhang Z, Liu T, Fan G, et al. SpineMamba: Enhancing 3D spinal segmentation in clinical imaging through residual visual Mamba layers and shape priors. Computerized Medical Imaging and Graphics, 2025, 123:102531.**

与本项目的直接关系：

- 使用 Residual Visual Mamba 建模 3D 脊柱长程依赖，并设计可学习 3D spine shape prior；
- 说明近期脊柱分割的竞争基线已经从 CNN/Transformer 延伸到 state-space model 与显式结构先验；
- 对本项目的启示不是照搬 Mamba，而是必须证明 Boundary/Topology/uncertainty 等结构约束带来的独立收益，不能只依赖 backbone 新颖性；
- 正式实验若算力允许，可把 SpineMamba 作为“现代结构感知 baseline/Related Work”重点比较对象。

DOI：`10.1016/j.compmedimag.2025.102531`

### 3.6 Transformer 椎体分割与解剖变异识别（2025）

**Yang C, Huang L, Sucharit W, et al. Transformer-enhanced vertebrae segmentation and anatomical variation recognition from CT images. Scientific Reports, 2025, 15:34329.**

与本项目的直接关系：

- 使用 VerSe 2019/2020，组合 WNet 椎体分割、ViT 椎体类型分析和 anatomical-variation attention；
- 将 T13、L6 等解剖变异纳入分析，说明“平均 Dice”之外还需要关注结构异常和个体差异；
- 其鲁棒性设计包含 slice skipping、噪声和 scanner variation，可作为本项目厚层 CT、噪声/强度扰动和 domain-shift 实验设计的近期参考；
- 本项目不使用患者年龄/性别等临床元数据作为当前 baseline 输入，避免在授权范围和任务定义未固定时扩大变量。

DOI：`10.1038/s41598-025-16689-9`

### 3.7 VertebraFormer（2026，结构感知多任务 + Domain Generalization）

**Du J, Ge H, Zhang R, et al. Structure-aware multi-task learning with domain generalization for robust vertebrae analysis in spinal CT. npj Digital Medicine, 2026, 9:217.**

与本项目的直接关系：

- 统一处理 vertebra segmentation、numbering 与 lesion localization，并强调结构感知表示；
- 构建 multi-domain benchmark，并采用患者级划分与 leave-one-domain-out 跨域评估；
- 对本项目后续“公开数据训练/验证 → 合法脱敏临床数据外部测试”具有直接实验设计参考价值；
- 同时提醒本项目正式 Results 不应只报告单一随机 split，还应在数据条件允许时加入 source/domain 外部验证或至少来源分层分析。

DOI：`10.1038/s41746-025-02288-5`

### 3.8 开放椎体体部数据与 Residual-Encoder nnU-Net 强基线（2026）

**Hofmann F O, Auhage L A, Dexl J, et al. Vertebral body segmentation in CT: An open dataset, deep-learning models and comparison to existing models. European Journal of Radiology, 2026, 204:113118.**

与本项目的直接关系：

- 公开 1,460 例 CT 的椎体体部标签，并使用 Residual-Encoder nnU-Net 建立强分割基线；
- 训练集 1,216 例、测试集 244 例，并进一步在 300 例肿瘤 CT 上验证 L3 定位，说明强 baseline 与外部场景验证已经成为近期椎体研究的重要标准；
- 对本项目最直接的约束是：正式论文不能只与传统 3D U-Net 比较，应优先加入 nnU-Net/Residual-Encoder nnU-Net 级别强 CNN baseline；
- 其开放标签与权重可作为后续扩大数据、复核任务定义和外部对照的候选资源，但必须先核对标签语义是否与本项目最终任务一致。

DOI：`10.1016/j.ejrad.2026.113118`

### 3.9 椎体骨折报告 pipeline：nnU-Net 分割 + 分类（2025）

**Glessgen C, Cyriac J, Yang S, et al. A deep learning pipeline for systematic and accurate vertebral fracture reporting in computed tomography. Clinical Radiology, 2025, 83:106827.**

与本项目的直接关系：

- 使用 452 例胸腰椎 CT，最终分割阶段采用 nnU-Net，并在独立测试中正确分割 330/339 个椎体；
- 该工作把“椎体分割是否成功”与后续骨折分类串成完整 pipeline，说明骨折病例既是下游任务，也是检验分割鲁棒性的高价值困难子集；
- 对本项目 topology loss 尤其重要：骨折断裂可能是真实解剖状态，不能把所有断裂都当作 false break 强行修复；
- 正式消融应在有真实骨折标签/病例属性时，单独报告骨折子集的 Dice/HD95/ASSD、false merge/false break，而不是只看总体平均值。

DOI：`10.1016/j.crad.2025.106827`

### 3.10 真实腰椎金属植入物与 deep-MAR（2025）

**Ye K, Pan B, Li J, et al. Deep learning model trained using multi-energy computed tomography (CT) data shows better metal artifact reduction for lumbar CT imaging. Clinical Radiology, 2025, 90:107076.**

与本项目的直接关系：

- 基于 93 例真实腰椎植入物患者的多能 CT 研究金属伪影削减，而不是只依赖人工合成伪影；
- 直接证明金属内固定是腰椎 CT 中真实存在、可单独建模和评价的困难成像条件；
- 因此本项目当前“金属伪影模拟只在有真实金属病例可校验时启用”的策略是合理的；
- 后续若能取得带金属植入物的公开/授权病例，应优先建立真实 metal-artifact subset，再决定是否需要额外的伪影增强或 MAR 前处理。

DOI：`10.1016/j.crad.2025.107076`

### 3.11 低骨密度下的椎体融合/分裂失败模式（2024）

**Xiong X, Graves S A, Gross B A, et al. Lumbar and Thoracic Vertebrae Segmentation in CT Scans Using a 3D Multi-Object Localization and Segmentation CNN. Tomography, 2024, 10(5):738–760.**

与本项目的直接关系：

- 采用三维 multi-object localization + segmentation CNN，在放疗 CT 与 VerSe2020 上评价腰椎/胸椎分割；
- 作者在失败案例中明确指出：**骨密度较低时，相邻椎体可能被错误融合，一个椎体也可能被错误分裂成多个部分**；
- 这为本项目此前仅凭工程经验设置的 low-density difficult subset、`false_merge_count`、`false_break_count` 与 component error 提供了直接文献依据；
- 因此正式实验若病例 metadata/影像条件允许，应按低骨密度/骨质疏松相关病例分层，而不能只报告总体 Dice；同时需要结合 HD95/ASSD 和结构错误，判断低对比边界对几何质量的影响。

DOI：`10.3390/tomography10050057`

### 3.12 数据集选择建议

首篇论文建议优先选择一个“问题定义清楚”的主任务，不要一开始做全身几十类骨骼：

**优先级 1：脊柱/椎体 CT**

原因：

- CTSpine1K 规模较大；
- 骨边界、粘连、断裂、各向异性问题明显；
- 更容易建立 topology/boundary 的合理评价；
- 有利于后续三维脊柱重建与测量。

**优先级 2：骨盆/髋部骨结构**

可从 TotalSegmentator 或其他公开资源中提取；临床价值高，但需确认是否能形成独立可靠的训练/测试集。

---

## 4. 边界约束

### 4.1 Boundary Loss

**Kervadec H, Bouchtiba J, Desrosiers C, et al. Boundary loss for highly unbalanced segmentation. 2018/2019.**

核心思想：

- Dice/CE 等主要在区域内部积分；
- Boundary Loss 通过距离/轮廓空间对边界进行约束；
- 可以与标准区域损失联合使用；
- 尤其适合类别极不均衡场景。

与本项目关系：

- 骨皮质边缘、关节边缘和细小结构对表面误差敏感；
- 论文不能只看 Dice，应同时报告 HD95/ASSD 证明边界变化。

链接：`https://arxiv.org/abs/1812.07032`

### 4.2 本项目边界实验建议

至少比较：

- DiceCE；
- DiceCE + Boundary；
- DiceCE + Boundary + Topology。

如果 Dice 变化很小但 HD95/ASSD 明显改善，这仍可能是骨科三维重建的重要贡献，但必须通过统计和可视化支持。

---

## 5. 拓扑保持

### 5.1 clDice / soft-clDice

**Shit S, Paetzold J C, Sekuboyina A, et al. clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. 2020/2021.**

核心思想：

- 利用预测和标签的 skeleton/centerline 交集构造 clDice；
- soft-clDice 可微，可作为训练损失；
- 对连通性和拓扑错误更敏感。

链接：`https://arxiv.org/abs/2003.07311`

### 5.2 对骨结构使用时的注意

clDice 最初重点面向管状/网络结构。骨骼并非所有部位都满足“中心线拓扑最重要”的假设。因此本项目不能机械地写“clDice 一定适合所有骨骼”。

建议：

- 脊柱整体结构：先验证 soft-clDice 是否改善连接错误；
- 单块骨：可优先使用连通域惩罚、拓扑一致性或其他更匹配的约束；
- 论文中通过消融决定是否保留拓扑项。

---

## 6. 自监督与少样本方向

### 6.1 Swin UNETR 自监督预训练

**Tang Y, Yang D, Li W, et al. Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis. 2021/2022.**

核心价值：

- 在大量公开 3D CT 上自监督预训练；
- 再微调到下游医学分割；
- 说明在标注有限的医学影像场景，自监督是可行方向。

链接：`https://arxiv.org/abs/2111.14791`

本项目用途：

如果临床脱敏标注量有限，可把“公开 CT 预训练 → 临床微调”作为后续扩展，但首个中期目标先确保监督 baseline 可重复。

---

## 7. 不确定性精修：研究设计方向

当前任务书明确要求“不确定性精修”，但上游 SegFormer3D 未提供现成模块。

### 7.1 近期代表工作：EDUE（2024）

**Abutalip K, Saeed N, Sobirov I, et al. EDUE: Expert Disagreement-Guided One-Pass Uncertainty Estimation for Medical Image Segmentation. 2024.**

该工作强调医学分割除了 Dice 等预测性能，还需要可信的不确定性估计；其方案利用多标注者之间的差异监督不确定性，并用单次前向传播产生不确定性。它并不是骨科 CT 专用方法，而且需要多标注者信息，因此不直接复制到本项目，但给出两个重要启发：

1. 不确定性不仅要“画热图”，还应验证它和错误/困难区域是否相关；
2. 不确定性模块应同时评估 calibration / QC 能力与额外计算开销。

论文：`https://arxiv.org/abs/2403.16594`

### 7.2 本项目首版实现路线

首版候选方案按工程复杂度排序：

1. **预测熵（softmax entropy）**：单次推理，成本最低；
2. **MC Dropout**：多次 stochastic inference，计算成本中等；
3. **Deep Ensemble**：稳定但训练成本高；
4. **独立 refinement network**：对高不确定 ROI 二次分割。

建议论文第一版优先：

`entropy uncertainty map → threshold/Top-K difficult voxels → ROI crop → refinement head/network`

必须设计对照：

- 无精修；
- 全图二次推理；
- 仅不确定 ROI 精修。

这样才能证明“不确定性机制”而不是简单增加计算量带来的收益。

---

## 8. 当前研究空白与可形成的论文切入点

从现有主线看，本项目较有价值的切入点不是重新发明 SegFormer，而是解决骨科 CT 的具体问题：

### Gap 1：通用 3D Transformer ≠ 骨科 CT 专用 pipeline

现有 SegFormer3D 是通用 3D 分割框架，缺少 DICOM/HU/骨窗/各向异性/骨边界专用流程。

### Gap 2：Dice 好不代表三维骨边界好

骨科三维重建更关心表面质量、断裂、粘连和细节，因此有必要联合边界/拓扑约束，并报告 surface metrics。

### Gap 3：困难病例容易拉低临床可用性

金属伪影、骨折、低骨密度、截断视野等情况需要困难样本增强和不确定性机制。

### Gap 4：很多论文停在 segmentation metric

本项目计划把分割真正接入三维重建与 Web 交互，能形成“算法 → 系统”的完整成果链，但必须避免把工程集成夸大为算法创新。

---

## 9. 首版论文 Related Work 组织建议

### 9.1 3D Medical Image Segmentation

- 3D CNN / nnU-Net
- UNETR
- nnFormer
- Swin UNETR

### 9.2 Efficient Transformer Segmentation

- SegFormer
- SegFormer3D

### 9.3 Orthopedic CT / Spine Segmentation

- CTSpine1K
- TotalSegmentator 中骨结构相关工作
- 后续补充 VerSe 等脊柱数据集与骨科专用网络

### 9.4 Boundary- and Topology-aware Learning

- Boundary Loss
- clDice / soft-clDice
- 后续补充 Hausdorff-aware / topology-preserving methods

### 9.5 Uncertainty-guided Refinement

- 后续补充医学分割中的 uncertainty calibration、MC dropout、uncertainty-guided refinement 代表论文。

---

## 10. 首批必读文献清单

| # | 文献 | 年份 | 本项目用途 |
|---:|---|---:|---|
| 1 | SegFormer | 2021 | 2D 原始思想、多尺度 encoder + MLP decoder |
| 2 | UNETR | 2021 | 3D Transformer 重要基线 |
| 3 | nnFormer | 2021 | 体分割 Transformer 对比、数据 pipeline 参考 |
| 4 | Swin UNETR SSL | 2021/22 | 自监督/CT 预训练思路 |
| 5 | SegFormer3D | 2024 | 本项目直接 backbone |
| 6 | CTSpine1K | 2021 首发 / 2025 正式版 | 脊柱 CT 主候选数据集；正式引用优先 MELBA 2025 版本 |
| 7 | TotalSegmentator | 2022/23 | 多骨 CT 数据与外部验证候选 |
| 8 | Boundary Loss | 2018/19 | 边界联合损失 |
| 9 | clDice | 2020/21 | 拓扑约束候选 |
| 10 | VerSe benchmark | 2021 | 脊柱 CT 数据、跨域与困难病例基准 |
| 11 | VerFormer | 2024 | 近期脊柱 CT Transformer 对比工作 |
| 12 | SpineMamba | 2025 | 3D Mamba + 脊柱 shape prior，现代结构感知强相关工作 |
| 13 | Yang et al. vertebrae Transformer | 2025 | VerSe 椎体分割 + 解剖变异识别 + 鲁棒性设计 |
| 14 | VertebraFormer | 2026 | 结构感知多任务 + patient-level / leave-one-domain-out 泛化设计 |
| 15 | Hofmann et al. vertebral body ResEnc nnU-Net | 2026 | 1,460 CT 开放椎体体部标签 + Residual-Encoder nnU-Net 强 baseline |
| 16 | Glessgen et al. vertebral fracture pipeline | 2025 | nnU-Net 椎体分割 + 骨折分类；困难/真实断裂病例依据 |
| 17 | Ye et al. lumbar deep-MAR | 2025 | 真实腰椎金属植入物多能 CT；metal-artifact subset 依据 |
| 18 | Xiong et al. 3D vertebra segmentation | 2024 | 直接记录低骨密度下 vertebra fusion/split 失败；结构错误指标依据 |
| 19 | EDUE | 2024 | 不确定性评估/QC 方法学参考 |

---

## 11. 文献任务状态与下一步

- [x] nnU-Net 原始论文纳入结构化矩阵与 BibTeX；
- [x] VerSe benchmark 正式 Medical Image Analysis 版本纳入 BibTeX；
- [x] CTSpine1K 从 2021 预印本更新到 2025 MELBA 正式版本；
- [x] Hausdorff/surface-aware loss 纳入 Karimi & Salcudean 与 Boundary Loss 主线；
- [x] topology-preserving 主线纳入 clDice、persistent-homology、Betti matching 与 TEDS-Net；
- [x] uncertainty 主线纳入 MC Dropout、Deep Ensemble、Calibration、医学分割 calibration、EDUE、UCTNet；
- [x] 三维重建主线纳入 Marching Cubes、QEM、DeepSDF、Occupancy Networks；
- [x] 已形成 44 条结构化文献矩阵，其中 42 条英文核心题录进入 `paper/references.bib`；
- [x] 已补 2025–2026 骨/椎体 CT 直接工作：SpineMamba、解剖变异感知 Transformer、VertebraFormer、Residual-Encoder nnU-Net 强 baseline、椎体骨折 nnU-Net pipeline；
- [x] 已补真实腰椎金属植入物 deep-MAR 文献，作为 metal-artifact 困难病例与真实伪影校验依据；
- [ ] 继续补与 VerSe/CTSpine1K 同任务的现代 ConvNet/状态空间强基线，但不再机械凑数量；
- [x] 已补低骨密度 CT 椎体分割直接文献：低骨密度可出现 vertebra fusion/split，支撑 low-density subset 与 false merge/false break 分析；
- [ ] 补各向异性 CT 超分辨率/形状插值与连续表面重建医学应用；
- [ ] 国内条目回 CNKI/万方逐条核验作者、卷期、页码和 DOI/基金信息；
- [ ] 正式投稿前统一 BibTeX 作者名、页码、期刊缩写与引用格式。


---

## 12. 国内中文研究补充（2022—2026）

### 12.1 肋骨 CT 自动分割与三维重组（2022）

**伍志发，刘梦秋，吴娇艳，刘影. 基于深度学习的 CT 图像肋骨自动分割与三维重组研究. 临床放射学杂志, 2022, 41(2):351-356.**

万方摘要显示，该研究收集 130 例胸部 CT，比较 UNet 3D、VNet、DenseNet 3D 与 DenseVoxelNet，并把自动分割结果用于肋骨三维重组；还使用来自另外三台 CT 设备的数据进行独立验证。对本项目最有价值的不是直接复用其性能数字，而是其“**3D 分割 → 独立设备验证 → 三维重组**”研究闭环，可作为骨科 CT 系统设计和国产临床研究写作的直接参考。

检索页：`https://med.wanfangdata.com.cn/Paper/Detail?id=PeriodicalPaper_lcfsxzz202202029`

### 12.2 椎体转移瘤 CT 靶区三维分割（2026）

**艾念，周雪阳，薄宏宇，等. 基于残差 U-net 神经网络实现椎体转移瘤放疗靶区的自动勾画研究. 中国医学装备, 2026, 23(5):28-32.**

该研究基于 87 例匿名化 CT 影像比较 2D/3D U-Net 与 2D/3D ResUNet，并同时采用 DSC、IoU 和 Hausdorff Distance 评价。它直接支持本项目两个设计判断：

1. 骨/椎体相关 CT 任务应保留真正的 3D 体数据建模对照，而不能只做逐切片 2D；
2. 不能仅报告 Dice，至少同时报告表面距离指标，否则难以反映骨边界质量。

期刊检索页：`https://yxzb.cbpt.cnki.net/portal/journal/portal/client/paper/762c2fa4a965349ecb32181554bb4fcb`

### 12.3 国内研究现状对本项目的启示

国内公开研究已经证明深度学习在肋骨、椎体等骨结构 CT 自动分割与三维重建中具备应用基础，但常见路线仍以 3D CNN/U-Net 变体为主。结合国内团队提出的 CTSpine1K 与 2024 年 VerFormer，可把本项目的国内外研究脉络概括为：

`传统阈值/区域生长 → 3D CNN/U-Net → 大规模公开脊柱数据与 benchmark → Transformer 全局建模 → 多尺度/边界/拓扑/不确定性可信分割`

因此，本项目的论文切入点不应写成“首次使用深度学习进行骨 CT 分割”，而应聚焦**轻量 3D Transformer 在骨科 CT 场景下的专用适配、表面/拓扑质量和困难病例可靠性**。

---

## 13. 2024 年后拓扑与不确定性方向补充

### 13.1 UCTNet：不确定性引导的 CNN-Transformer（2024）

**Guo X, Lin X, Yang X, et al. UCTNet: Uncertainty-guided CNN-Transformer hybrid networks for medical image segmentation. Pattern Recognition, 2024, 152:110491.**

UCTNet 将不确定性用于引导 Transformer 特征学习，而不是仅在推理后显示热图。对本项目的启示是：如果首版“预测熵 → ROI 精修”证明 uncertainty 与真实误差有稳定相关性，可继续研究 uncertainty 是否能参与特征融合或训练采样。

DOI：`10.1016/j.patcog.2024.110491`

### 13.2 Topologically Faithful Multi-class Segmentation（2024）

**Berger A H, Stucki N, Lux L, et al. Topologically Faithful Multi-class Segmentation in Medical Images. 2024.**

该工作基于 persistence barcode / Betti matching 将拓扑约束扩展到多类别医学分割，为本项目后续“多椎体/多骨类别”提供比 soft-clDice 更一般的候选方案。它同时提示：拓扑保持需要与具体解剖结构的 Betti 数/连通关系相匹配，不能把管状结构的损失机械迁移到所有骨结构。

论文：`https://arxiv.org/abs/2403.11001`

### 13.3 TEDS-Net：显式拓扑先验形变（2024）

**Wyburd M K, Dinsdale N K, Jenkinson M, Namburete A I L. Anatomically plausible segmentations: Explicitly preserving topology through prior deformations. Medical Image Analysis, 2024, 97:103222.**

该方法通过对具有正确拓扑的先验形状进行连续形变来保证解剖合理性，并讨论了连续拓扑约束在离散域中的失效问题。它更适合拓扑相对稳定的器官/结构；对骨折断端这类**真实拓扑可能发生改变**的骨科场景，则需要谨慎使用固定拓扑先验。这个对比非常适合写入 Discussion：骨科分割不能为了“拓扑漂亮”而把真实骨折错误地连回去。

DOI：`10.1016/j.media.2024.103222`

### 13.4 本项目的拓扑策略因此调整为“分层验证”

- **第一层：** DiceCE + Boundary，先确保区域和表面质量；
- **第二层：** soft-clDice 作为低成本拓扑候选，验证是否减少断裂/粘连；
- **第三层：** 若最终任务为多椎体/多骨并且 soft-clDice 不足，再评估 Betti matching / persistent-homology 类损失；
- **骨折病例例外：** 真实断裂本身可能是正确解剖状态，拓扑规则必须结合标签语义，不能把“骨连续”作为绝对约束。

这也是为什么 `docs/04_experiment_plan.md` 中要求拓扑项必须单独消融，而不是预设为最终有效模块。
