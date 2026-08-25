# 协作贡献指南

欢迎参与本项目。这个仓库用于多人协作完成骨科 CT 分割、三维重建、Web 科研原型与论文实验。

## 1. 开始前必读

按顺序阅读：

1. `README.md`
2. `PROJECT_STATUS.md`
3. `TASKS.md`
4. 与你负责模块相关的 `docs/`

如果你准备跑正式实验，还必须先检查：

- `configs/task_specs/vertebra_task_template.json`
- `env/check_gpu.ps1`
- `env/check_formal_readiness.ps1`

## 2. 推荐协作流程

不要所有人直接在 `main` 上改。

建议：

```text
main
├─ feature/data-qc
├─ feature/baseline-training
├─ feature/uncertainty-refinement
├─ feature/web-3d
└─ docs/paper-update
```

每次工作：

```bash
git checkout -b feature/your-task
# 修改
pytest tests -q
ruff check src web tests
git add ...
git commit -m "中文说明本次改动"
git push -u origin feature/your-task
```

然后发 Pull Request 合并到 `main`。

## 3. 提交规范

推荐提交类型：

- `feat:` 新功能
- `fix:` 修复 bug
- `test:` 测试
- `docs:` 文档
- `refactor:` 重构
- `experiment:` 实验配置/记录
- `chore:` 环境/工程维护

示例：

```text
feat: 增加椎体多类别正式任务配置
fix: 修复厚层 CT 重采样标签几何偏差检查
experiment: 增加 CT-only baseline 配置
```

## 4. Pull Request 最低要求

PR 必须说明：

1. 为什么改；
2. 改了什么；
3. 如何验证；
4. 是否影响数据、split、指标或论文；
5. 是否新增依赖；
6. 是否需要更新 `PROJECT_STATUS.md`。

代码修改原则上至少通过：

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m ruff check src web tests
```

前端修改额外执行：

```powershell
node --check web/frontend/app.js
node --check web/frontend/qc_review.js
node --check web/frontend/research_3d.js
node --check web/frontend/results_review.js
```

## 5. 正式实验规则

正式论文实验禁止绕过以下条件：

- task spec 已锁定；
- patient-level split 固定；
- 人工 QC 已完成；
- `test_private` 不进入 train/validation；
- NVIDIA GPU/CUDA 环境确认；
- `formal_readiness` 返回 `ready=true`。

正式实验输出必须至少保留：

```text
config.yaml
split.json
run_metadata.json
history.csv
train.log
checkpoint
metrics_per_case.csv
summary.json
```

## 6. 医学数据与隐私

**严禁提交：**

- 临床原始 DICOM；
- 未脱敏医学影像；
- 患者姓名、身份证号、手机号、住院号等；
- 临床数据截图；
- 私有医院数据；
- 本地模型 checkpoint / 大型训练缓存。

公开数据也尽量不直接进 Git，使用下载脚本、dataset manifest 和来源说明复现。

`.gitignore` 已默认排除常见医学数据和大型实验产物，但提交前仍需人工检查。

## 7. 论文结果边界

以下内容**不能**写成模型性能：

- random-weight smoke 输出；
- GT label mesh；
- preprocessing geometry error；
- SDF smoothing engineering difference；
- 任务书目标 Dice；
- 尚未独立测试的 validation 数字。

只有可追溯到正式 checkpoint + 固定 test split 的指标才能进入论文 Results。

## 8. 第三方代码

`third_party/SegFormer3D` 默认不进本仓库 Git 历史，通过 `env/fetch_segformer3d.ps1` 获取。

不得：

- 删除原作者版权声明；
- 将上游 backbone 冒充自研；
- 忽略 GPL-3.0 许可要求。

## 9. 主进度台账

任何实质性改动后，最后一个项目文档动作必须更新：

```text
PROJECT_STATUS.md
```

包括：

- 当前状态；
- 测试结果；
- 新增文件；
- 新风险；
- 下一步任务。
