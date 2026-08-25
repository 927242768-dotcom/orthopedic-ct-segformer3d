## 本次改动

请简要说明改了什么、为什么改。

## 所属模块

- [ ] 数据/QC
- [ ] 模型
- [ ] Loss/增强
- [ ] 不确定性/精修
- [ ] 三维/Web
- [ ] 文档/论文
- [ ] 环境/工程

## 验证

- [ ] `pytest tests -q`
- [ ] `ruff check src web tests`
- [ ] 前端改动已执行 `node --check`
- [ ] PowerShell 改动已做 parser 检查
- [ ] 真实数据验证（如适用）

请填写关键结果：

```text

```

## 科研结果边界

- [ ] 本 PR 没有把随机权重/smoke/GT mesh/预处理几何误差写成正式模型性能
- [ ] 如包含正式指标，能够追溯到固定 config + split + seed + checkpoint + test 输出

## 数据与隐私

- [ ] 未提交 DICOM/NIfTI/临床数据
- [ ] 未提交患者信息
- [ ] 未提交 token/key/.env
- [ ] 未提交 checkpoint/runtime 大文件

## 文档与台账

- [ ] 已更新必要文档
- [ ] 若为实质修改，最后已更新 `PROJECT_STATUS.md`

## 关联 Issue

Closes #
