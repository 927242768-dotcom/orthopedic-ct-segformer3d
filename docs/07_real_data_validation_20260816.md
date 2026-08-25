# 07｜2026-08-16 CTSpine1K 真实数据工程验证记录

> 目的：固定本轮真实公开数据落盘、标准化、QC、训练链 smoke test 的可追溯证据。
>
> **本记录不是论文 Results。当前没有正式训练得到的 DSC/HD95/ASSD，也没有临床有效性结论。**

## 1. 数据来源与范围

数据集：CTSpine1K，子数据集 `MSD-T10`。

本轮实际落盘 10 个 CT+label 配对病例：

```text
liver_0
liver_1
liver_2
liver_3
liver_4
liver_5
liver_6
liver_7
liver_8
liver_169
```

官方 `data_split.txt` 解析结果：

```text
trainset:     9
 test_private: 1  (liver_169)
```

`liver_169` 只作为工程 smoke/QC 病例，不允许进入训练调参。

本地位置：

```text
raw:       data/raw_public/CTSpine1K/MSD-T10
processed: data/processed_ctspine1k_real
```

当前磁盘规模约：

```text
raw CTSpine1K local subset: 1.4 GiB
processed 10 cases:          3.3 GiB
```

原始医学影像与标签均受 `.gitignore`/数据治理规则约束，不应提交代码仓库。

## 2. 下载与完整性事实

2026-08-16 早期访问 Hugging Face 时出现过连接超时和并行下载失败；之后改为 Edge 浏览器单文件顺序下载，10 个 CT+label 均成功落盘。

对本轮后续接管的真实 CT 文件执行了浏览器下载目录与项目目标文件的 SHA-256 对比，已验证的复制均哈希一致；例如：

```text
liver_3.nii.gz
595d01419d55985833d2de6c506ffbec6ece439c111b646298f127bc6f597f3d

liver_5.nii.gz
c9779e9c46e791448ddfa10431bad821356db0dad901847e4f89008a82fcbbbc

liver_8.nii.gz
a1d57969a8790a08c12d4fca6541c68648a8a3200cc7b58676d63c5f91da2541
```

此前 `liver_0`、`liver_1`、`liver_169` 等病例也已在接管时做源/目标哈希一致性校验。

## 3. 真实数据触发并修复的问题

真实数据验证不是形式检查，本轮实际暴露并修复了以下问题：

1. **CTSpine1K 小样本目录配对 bug**：`<sub-dataset>/volumes/*.nii.gz` 布局曾把 `volumes` 误识别为 sub-dataset，导致真实 label 匹配失败；已修复并加入回归测试。
2. **Windows cp1252 CLI 输出问题**：中文 JSON/help 在部分 Git Bash/Windows 终端触发 `UnicodeEncodeError`；`prepare_ctspine1k`、真实 patch smoke、`qc_visualization` 等入口已统一处理 UTF-8 stdout/stderr。
3. **强度增强语义错误**：`ct_normalized` 实际为 HU clip 后逐病例 z-score，而不是 `[0,1]`。原首版 gamma/HU shift 逻辑会错误裁掉负值；现已升级 pipeline 0.3.0，metadata 记录 clipped HU mean/std，增强时可恢复到 HU 域再变换；bone-window 通道仍保持 `[0,1]`。
4. **并行下载稳定性问题**：浏览器并发下载多个大 CT 时出现 `无法下载`/残留 `.crdownload`；改为顺序下载后 `liver_3/5/8` 均成功完成。该现象属于当前工作站网络/下载客户端稳定性问题，不属于医学处理代码问题。

## 4. Pipeline 0.3.0

当前真实 NIfTI 标准化链：

```text
image/label 物理空间检查
→ CT linear 重采样
→ label nearest-neighbor 重采样
→ target spacing 1.0 × 1.0 × 1.0 mm
→ HU clip [-1000, 2000]
→ case-wise z-score
→ bone window center/width = 500/2000
→ label 类别完整性检查
→ metadata.json + qc.json
→ qc_contact_sheet.png
```

metadata 中额外记录：

```text
normalization.method = clip_then_case_zscore
normalization.clipped_mean_hu
normalization.clipped_std_hu
```

10 例 clipped HU 统计范围：

```text
mean: -583.17 ~ -370.23 HU
std:   472.23 ~ 499.10 HU
```

## 5. 真实几何覆盖

10 例原始 z-spacing：

```text
5.0, 5.0, 5.0, 1.0, 1.0, 0.8, ~0.8, 1.0, 1.0, 1.0 mm
```

因此当前工程子集同时覆盖了明显厚层 CT 与接近各向同性 CT，可用于检查重采样链在不同层厚上的稳定性。

标准化后 10 例均为：

```text
spacing = 1.0 × 1.0 × 1.0 mm
```

处理后 shape 具有明显病例差异，例如：

```text
360×360×375
346×346×615
310×310×210
397×397×517
...
434×434×541
```

说明训练阶段必须继续使用 3D patch / sliding-window，而不能假定固定整幅体积尺寸。

## 6. 标签完整性

每例原始标签与重采样后标签均做整数类别检查；nearest-neighbor 重采样后没有出现原始标签集合之外的新类别。

本轮病例包含不同椎体覆盖范围，单病例前景类别数约 7—19 类。正式任务仍需组内最终确认是：

- binary spine/bone；
- multi-class vertebra semantic segmentation；
- vertebra instance；
- 或其他骨科任务。

在标签定义确定前，不建立正式论文 train/validation/test 结果。

## 7. 自动 QC 与人工审核状态

10 例均已生成：

```text
qc_contact_sheet.png
metadata.json
qc.json
```

`src.preprocessing.audit_processed` 最终结果：

```text
case_count: 10
status_counts.pass: 10
pipeline_versions.0.3.0: 10
all_pass: true
```

自动审计内容包括：

- pipeline version；
- image/label 几何一致性；
- 1 mm spacing；
- label 非空与类别集合；
- bone-window 输出存在性；
- normalization metadata 完整性。

批量 QC 还已生成：

```text
data/processed_ctspine1k_real/manual_qc_review.csv   # 10 行
data/processed_ctspine1k_real/qc_visualization_summary.json
```

其中 `orientation_ok / spacing_ok / label_alignment_ok / bone_window_ok / review_status / reviewer / notes` 等人工字段保持空白。

**结论边界：可以说“10 例真实数据自动工程审计通过”，不能说“10 例人工医学 QC 已完成”。人工逐例审核与签字仍是下一项工作。**

## 8. 真实训练链 smoke test

已使用真实标准化 `liver_0` 执行：

```text
ProcessedOrthopedicCTDataset
→ CT z-score + bone-window 双通道
→ foreground patch 采样
→ 可配置几何/强度增强
→ SegFormer3D
→ Region + Boundary + soft-clDice joint loss
→ backward
→ AdamW.step
```

36³ 工程 smoke 输出：

```text
status = pass
image shape = (1, 2, 36, 36, 36)
foreground_fraction ≈ 0.5313
gradient_tensor_count = 205
```

该 smoke 的 loss/梯度数值来自随机初始化网络，**不属于模型性能指标**。

工程 smoke split：

```text
data/splits/ctspine1k_msd_t10_engineering_smoke.json
```

该 split 明确标记：

```text
formal_experiment = false
```

不能用于论文 Results。

## 9. 当前代码能力

截至本轮，除数据链外已完成以下工程能力：

- SegFormer3D adapter；
- Region / Boundary / soft-clDice joint loss；
- 3D flip / rotate / scale / gamma / noise / HU shift；
- boundary-proxy hard patch sampling；
- predictive entropy / uncertainty ROI；
- uncertainty→error AUROC/AUPRC、Top-percent error recall、ROI error rate；
- `UncertaintyRefinementNet3D` 局部残差精修原型 + ROI-only 二阶段训练基线；
- Dice / IoU / Precision / Recall / HD95 / ASSD；
- connected component count / false merge / false break；
- 独立 checkpoint evaluation 输出 `metrics_per_case.csv` / `metrics_per_class.csv`；
- formal/engineering preflight，正式训练默认拦截 engineering split、test_private、未通过人工 QC、标签/类别配置错误与无 CUDA；
- 训练 scheduler、split/config/train.log/环境版本追踪；
- Web axial/coronal/sagittal MPR + 真值 label overlay + 人工 QC reviewer；
- `1–25 → C1–L6` 椎体标签可读 schema（只显示，不重编码/不锁定正式任务）；
- mask → physical-space Marching Cubes → PLY mesh；
- vertex-clustering 网格简化、WebGL2 真值 3D、物理距离/角度计算；
- 原始 label→1 mm label 重采样物理表面几何误差评估。

真实 `liver_0` label 已进一步完成网格导出验证：

```text
output: data/processed_ctspine1k_real/ctspine1k-msd-t10-liver_0/mesh_foreground.ply
PLY size: ~9.5 MiB
vertex_count: 131983
face_count: 264362
spacing: 1×1×1 mm
```

顶点已显式应用 NIfTI/SimpleITK 的 spacing、origin、direction 转换到物理 XYZ。1.5 mm vertex-clustering 将该真实全前景网格降至 52,726 顶点 / 106,329 面，约减少 60% 顶点/面，顶点近邻 HD95 约 0.707 mm；全分辨率 PLY 保留不覆盖。JSON 中的 surface area/bounds/简化误差只用于工程几何追踪，不作为临床测量或模型性能结果。

10 例原始 label 与 1 mm nearest-neighbor 重采样 label 的工程表面比较已 10/10 成功：整体顶点近邻 ASSD 约 0.403 mm、HD95 约 0.734 mm；原始 z-spacing=5 mm 的 3 例约为 ASSD 0.514 mm / HD95 1.069 mm，明显高于 1 mm 组。这说明厚层数据的离散化扰动需要单独报告，但仍不是模型误差。

阶段 G 当时的自动化验证：

```text
pytest tests -q
→ 71 passed

ruff check src web tests
→ All checks passed!
```

### 9.1 后续工程验收更新（同日）

在上述真实数据链基础上继续完成：

- `task_lock`：正式任务未锁定时禁止编译正式训练配置；
- GPU/CUDA 只读检查与一站式 `formal_readiness`；当前模板 + engineering split + 本机 CPU 环境会正确返回未就绪；
- evaluation results-review Web 页面；当前项目真实 evaluation 总数为 0，不把 smoke 输出冒充模型结果；
- physical-mm SDF 表面基线与连通域保护；真实 `ctspine1k-msd-t10-liver_0` 的 0.4 mm SDF summary/PLY 均可通过 Web 读取，0.8 mm 因连通域由 2 变 3 被拒绝；
- 全项目回归更新为 `pytest tests -q → 88 passed`，Ruff 全部通过；4 个前端 JS、3 个关键 JSON、38 条 BibTeX 结构检查均通过。

以上仍属于工程与重建验证，**没有新增任何正式模型 DSC/HD95/ASSD。**

## 10. 尚未完成 / 下一步

当前最关键阻塞已从“无真实数据”转为：

1. 项目成员逐例填写 10 行 `manual_qc_review.csv`；
2. 组内确认正式研究任务和标签定义；
3. 确认 NVIDIA GPU/服务器及 CUDA/PyTorch 环境；
4. 按官方/研究方案建立正式 train/validation/test split；
5. 跑 SegFormer3D baseline，生成第一份真实 `metrics_per_case.csv`；
6. 依次做 CT-only vs CT+bone-window、Boundary、Topology、hard augmentation、uncertainty refinement 消融；
7. 将真实 checkpoint mask/uncertainty/mesh 接入 Web；
8. 只有完成独立测试后才向论文 Results 写入 DSC/HD95/ASSD 等数字。
