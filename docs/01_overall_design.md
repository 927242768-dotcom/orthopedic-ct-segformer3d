# 01 项目总体方案设计

## 1. 设计目标

本项目面向骨科 CT 影像科研分析与辅助诊断场景，构建从原始 DICOM 到骨骼分割、三维重建、Web 可视化和测量的一体化原型系统。

总体链路：

```text
DICOM/NIfTI
   ↓
数据治理与匿名性检查
   ↓
DICOM series 解析 / HU 恢复 / 方向校正
   ↓
空间重采样 / 骨窗增强 / 强度标准化 / QC
   ↓
SegFormer3D 多尺度分割
   ↓
区域 + 边界 + 拓扑联合损失训练
   ↓
困难样本增强 + 不确定性驱动精修
   ↓
后处理 / 多骨分离 / 连通性修复
   ↓
各向异性校正 / 三维网格重建
   ↓
Web：MPR + 分割叠加 + 3D + 测量 + 人工校核
```

## 2. 核心研究问题

### RQ1：如何把通用 SegFormer3D 适配到骨科 CT？

SegFormer3D 原始仓库主要验证于 BraTS、Synapse、ACDC 等数据，并非针对骨科 CT 的骨皮质、松质骨、关节间隙、骨折断端或金属伪影场景。因此必须重新设计：

- CT 特有 HU 强度处理；
- 骨窗增强；
- 目标骨类别定义；
- 各向异性体素处理；
- 适合骨边界的评价指标；
- 小结构与粘连/断裂问题的损失函数。

### RQ2：区域、边界和拓扑约束是否互补？

单纯 Dice/CE 强调体素级区域重叠，但对高曲率骨缘、细小骨结构、断裂和粘连不一定敏感。因此采用联合损失进行可验证消融：

```text
L_total = λr · L_region + λb · L_boundary + λt · L_topology
```

目标不是预设“联合损失一定最好”，而是用 patient-level 验证集和测试集证明其对 Dice、HD95、ASSD、错误连通等指标的实际影响。

### RQ3：如何处理困难病例？

重点考虑：

- 骨折和移位；
- 低骨密度；
- 金属内固定产生的条纹/高亮伪影；
- 大层厚/各向异性 CT；
- 部分视野截断；
- 多中心扫描协议差异。

策略包括困难样本增强、hard-example sampling 和不确定性区域精修。

## 3. 系统分层架构

### 3.1 数据层

职责：

- 数据登记；
- 脱敏合规检查；
- DICOM series 归并；
- 元数据提取；
- 训练/验证/测试患者级划分；
- 预处理产物缓存；
- QC 结果保存。

建议目录逻辑：

```text
data/
├─ raw_public/         # 公共原始数据（不建议提交 Git）
├─ raw_clinical/       # 临床脱敏数据（必须受控，不提交 Git）
├─ interim/            # 中间格式
├─ processed/          # 模型输入
├─ splits/             # patient-level 划分
└─ qc/                 # 质控 JSON/图
```

### 3.2 预处理层

职责：

- series 识别；
- slice 排序；
- HU 恢复；
- spacing/orientation 统一；
- 重采样；
- 强度裁剪与标准化；
- 骨窗派生；
- 标签同步变换；
- QC。

输出必须包含：

- 处理后体数据；
- 标签；
- spacing/origin/direction；
- 原始与处理后尺寸；
- 参数配置；
- QC 标志；
- provenance（来源追踪）。

### 3.3 模型层

基础模型：SegFormer3D。

结构适配方向：

1. 输入通道：单通道标准化 CT vs CT+骨窗多通道；
2. patch/stride：根据目标 spacing 和 ROI 尺寸调整；
3. 类别：binary bone / multi-bone；
4. decoder：保留上游 all-MLP 解码器作为可比基线；
5. loss：在上游 Dice/DiceCE 之外添加 boundary/topology；
6. inference：sliding-window + overlap；
7. uncertainty：MC dropout、深度集成或 softmax entropy 中选择成本可控方案。

### 3.4 后处理与三维重建层

分割后处理：

- 小连通域清理；
- 目标骨连通性检查；
- 粘连检测；
- hole filling（仅在解剖合理时）；
- 不确定区域二次精修。

三维流程：

```text
mask
→ physical-space 恢复
→ 可选 SDF
→ 等值面提取 / Marching Cubes
→ 网格去噪与平滑
→ 曲率/边缘保护
→ 网格简化
→ glTF/PLY/STL 等输出
```

任何平滑都必须防止骨折断端、关节缘、骨性突起被过度抹平。

### 3.5 Web 层

#### 后端

建议 FastAPI：

- `/api/health`：健康检查；
- `/api/cases`：病例管理；
- `/api/upload`：上传研究病例；
- `/api/preprocess/{case_id}`：预处理；
- `/api/infer/{case_id}`：分割推理；
- `/api/reconstruct/{case_id}`：三维重建；
- `/api/result/{case_id}`：结果查询；
- `/api/export/{case_id}`：导出。

第一阶段不引入过重的微服务，先做本机单机可复现闭环。

#### 前端

第一阶段功能：

- 上传；
- 病例列表；
- 处理状态；
- axial/coronal/sagittal 基础预览；
- 分割 mask 叠加；
- 研究用途免责声明。

第二阶段：

- 3D 网格；
- MPR 与 3D 联动；
- 透明度/类别开关；
- 距离和角度测量；
- 人工修正入口。

## 4. 数据接口规范

### 4.1 Case ID

禁止使用患者姓名等身份信息作为目录名。统一随机/哈希化：

```text
case_20260815_xxxxxxxx
```

### 4.2 处理结果 JSON 示例

```json
{
  "case_id": "case_xxx",
  "source_type": "dicom",
  "shape_original": [512, 512, 260],
  "spacing_original_mm": [0.74, 0.74, 1.50],
  "shape_processed": [256, 256, 256],
  "spacing_processed_mm": [1.0, 1.0, 1.0],
  "hu_clip": [-1000, 2000],
  "qc": {
    "status": "pass",
    "warnings": []
  }
}
```

以上仅为接口格式示例，不代表最终参数。

## 5. 质量保证策略

### 5.1 数据质量

每例必须检查：

- 是否完整 series；
- slice 数是否合理；
- spacing 是否为正且一致；
- orientation 是否可解析；
- HU 是否在合理范围；
- 是否存在重复切片；
- 标签和影像 shape/affine 是否一致；
- 是否包含明显个人身份信息元数据。

### 5.2 模型质量

- patient-level split；
- 固定随机种子；
- config 驱动；
- checkpoint 可追踪；
- 验证集调参，测试集一次性评估；
- 不允许在测试集上反复调 loss 权重。

### 5.3 Web 质量

- 输入格式校验；
- 文件大小限制；
- 失败状态明确；
- 模型版本号可见；
- 结果包含生成时间和参数；
- 研究用途提示；
- 不记录敏感身份字段。

## 6. 论文与工程共用实验链

论文中的每张主表/消融表应能从工程实验目录追踪：

```text
experiment_id
  ├─ config.yaml
  ├─ split.json
  ├─ train.log
  ├─ best.ckpt
  ├─ metrics_per_case.csv
  ├─ summary.json
  └─ figures/
```

论文 Results 不手工“估计”数字；统一从实验输出生成。

## 7. 阶段里程碑

### M1：数据与可运行原型

验收条件：

- 一个公开 CT 数据集可稳定完成预处理；
- 生成 QC；
- Web 可上传并显示处理状态；
- 代码有基本测试。

### M2：SegFormer3D baseline

验收条件：

- 可训练；
- 可验证；
- 可推理；
- 输出真实 Dice/HD95/ASSD；
- 结果可在 Web 叠加显示。

### M3：联合损失与困难样本

验收条件：

- 完成完整消融；
- 统计结果可重复；
- 能明确各模块收益与失败案例。

### M4：三维系统

验收条件：

- mask → mesh；
- 网格与原 CT 空间一致；
- Web 3D；
- 基本测量；
- 端到端演示。

### M5：论文/软著/中期材料

验收条件：

- 论文方法、实验、结果均来自真实项目；
- 软件说明与真实功能一致；
- 代码来源和开源许可证清晰；
- 中期材料能从进度台账复核。
