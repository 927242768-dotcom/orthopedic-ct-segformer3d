# 数据目录说明

本目录默认**不保存可识别的临床原始数据到 Git**。

建议本地结构：

```text
data/
├─ raw_public/
├─ raw_clinical/
├─ interim/
├─ processed/
├─ splits/
└─ qc/
```

## 规则

1. 临床数据必须已脱敏且有研究授权；
2. 患者姓名、身份证号、手机号、住址等字段不得进入日志、截图、文件名；
3. train/val/test 必须按患者级划分；
4. 原始 DICOM/NIfTI、大型缓存和模型权重不要提交 Git；
5. 数据来源、版本、许可、下载日期、预处理版本应单独登记；
6. Web 上传数据默认只在本地项目环境处理，未经明确批准不得上传第三方服务。

## 公开数据登记

机器可读登记见：

```text
data/datasets.json
```

当前工程推进顺序：

- **VerSe complete**：作为首轮工程 baseline 默认主数据集，用于先跑通脊柱/椎体 NIfTI → 标准化 → split → training 全链路；这不代表组内已经最终确认论文主任务；
- **CTSpine1K**：作为扩大脊柱样本规模或跨来源泛化候选；2026-08-16 已在本地取得 `MSD-T10` 10 例工程子集（`liver_0`—`liver_8`、`liver_169`），原始文件位于 `data/raw_public/CTSpine1K/MSD-T10`，标准化结果位于 `data/processed_ctspine1k_real`。这 10 例仅用于真实预处理/QC 和工程 smoke，不是正式论文 split；其组成来源许可仍需逐项保留 provenance；
- **TotalSegmentator CT v2.0.1**：作为多骨预训练/外部泛化候选，数据许可必须针对 Zenodo 数据记录单独核验，不能由软件许可证推断。

接入步骤见 `docs/06_public_dataset_onboarding.md`。当前 10 例已生成 `qc_contact_sheet.png`、`manual_qc_review.csv` 和机器可读 manifest/QC summary；自动审计为 10/10 pass，但人工审核字段仍待项目成员逐例填写。

实际下载前必须再次核对各数据集许可证和使用条件；大型数据下载必须显式执行，不由普通测试或环境初始化脚本自动触发。
