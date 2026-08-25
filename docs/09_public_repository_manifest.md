# 09｜公开 GitHub 仓库内容与排除清单

> 目的：让协作者知道“什么应该在公开仓库中、什么必须留在本地/私有存储”。

## 1. 应纳入公开仓库

### 项目入口与协作

- `README.md`
- `PROJECT_STATUS.md`
- `TASKS.md`
- `CONTRIBUTING.md`
- `.gitignore`
- `.github/` Issue / PR 模板

### 配置

- `configs/orthopedic_ct_baseline.yaml`
- `configs/orthopedic_ct_joint.yaml`
- `configs/label_schemas/`
- `configs/task_specs/`

### 代码

- `src/`
- `web/backend/`
- `web/frontend/`
- `web/run_web.ps1`

### 测试

- `tests/`

### 环境与复现脚本

- `env/requirements.txt`
- `env/setup_env.ps1`
- `env/fetch_segformer3d.ps1`
- `env/download_verse.ps1`
- `env/download_ctspine1k_sample.ps1`
- `env/check_gpu.ps1`
- `env/check_formal_readiness.ps1`

### 文档与论文材料

- `docs/`
- `paper/`
- `8.16组会_照念稿.md`
- `group_meeting_ppt_content_20260816.json`
- `make_group_meeting_ppt_20260816.ps1`
- `run_group_ppt_build.ps1`

### 数据元信息

- `data/README.md`
- `data/datasets.json`
- `data/splits/*.json`

> split 文件可以公开，但必须确认只含匿名 case_id，不含患者隐私信息。

---

## 2. 必须排除公开仓库

### 医学影像与处理后数据

- `*.dcm / *.dicom`
- `*.nii / *.nii.gz`
- `*.nrrd / *.mha / *.mhd`
- `data/raw_public/`
- `data/raw_clinical/`
- `data/processed*/`
- `data/interim/`
- `data/qc/`

原因：

1. 临床数据存在隐私/伦理风险；
2. 公开数据也可能受原始许可证约束；
3. 医学影像体积大，不适合普通 Git 历史；
4. 项目应通过下载脚本 + manifest 复现，而不是复制数据。

### 模型与实验大型产物

- `*.pt / *.pth / *.ckpt / *.onnx`
- `experiments/`
- `wandb/`
- `runs/`
- `outputs/`

正式模型后续若需要共享，应使用 GitHub Releases、对象存储或学校共享盘，并记录 checksum 和对应 commit。

### 本地开发环境

- `.python/`
- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`

### 第三方 checkout

- `third_party/SegFormer3D/`

仓库只保留 `third_party/README.md` 与获取脚本。协作者执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\fetch_segformer3d.ps1
```

重新获取上游源码，避免把第三方仓库完整嵌套进本仓库历史。

### 本地工具缓存/截图

- `.devspace-computer/`
- `web/runtime/`
- `web/backend/runtime/`
- `NUL`
- `*.log`

### 大型生成 PPT

当前 `.ppt/.pptx` 作为本地生成物默认不进入公开 Git 历史。仓库保留对应 Markdown / JSON / PowerShell 源材料，可重新生成。

如果团队后续确需版本管理 PPT，建议：

- 使用 Git LFS；或
- 放入 GitHub Releases；
- 提交前检查是否包含姓名、联系方式、临床截图或其他不宜公开信息。

---

## 3. 公开仓库不代表公开医学数据

本项目仓库公开的是：

> **代码 + 配置 + 实验方法 + 文档 + 可复现下载/处理脚本。**

不是：

> **患者数据仓库或公开 CT 镜像仓库。**

任何临床脱敏数据即使已获得研究授权，也只有在授权条款明确允许公开发布时才能进入公开网络；默认视为不可公开。

---

## 4. 提交前公开安全检查

每次 push 前检查：

- [ ] 没有 DICOM/NIfTI
- [ ] 没有患者信息
- [ ] 没有 `.env` / token / key
- [ ] 没有 checkpoint
- [ ] 没有大型 runtime 输出
- [ ] 没有误加入第三方 checkout
- [ ] `git status` 中的文件都能说明为何适合公开
- [ ] 代码测试通过
- [ ] `PROJECT_STATUS.md` 已同步

---

## 5. 当前公开仓库建议定位

仓库描述建议：

> 基于 SegFormer3D 的骨科/椎体 CT 分割、边界与拓扑约束、不确定性精修、三维重建和 Web 科研辅助分析原型。

推荐 Topics：

```text
medical-imaging
ct
medical-image-segmentation
segformer
transformer
spine
vertebrae
3d-reconstruction
fastapi
pytorch
```
