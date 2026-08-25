# 第三方开源代码管理说明

本目录用于存放项目依赖的第三方开源仓库，默认不将其源码混入本项目“自研代码”目录。

## SegFormer3D

上游仓库：

```text
https://github.com/OSUPCVLab/SegFormer3D
```

项目名称：SegFormer3D: an Efficient Transformer for 3D Medical Image Segmentation。

上游仓库当前声明 GPL-3.0 许可证。若本项目克隆、修改、分发或基于其源码形成派生作品，应遵守对应许可证条款，并保留版权与许可证信息。

## 当前本地兼容补丁

当前克隆基线提交：`e314242`。

为使官方代码在本项目固定的 PyTorch 2.1.0 环境中通过 TorchScript 导入，已对 `third_party/SegFormer3D/architectures/segformer3d.py` 的 `cube_root()` 做**一行类型兼容修复**：把 `round(...)` 的返回值显式转换为 `int`。原因是 PyTorch 2.1 TorchScript 将 `round(float)` 推断为 float，而上游函数标注返回 `int`，会在 import 阶段报类型错误。

该补丁不改变模型结构、参数或计算语义；上游原始提交和本地 diff 均通过 Git 可追踪。后续更新上游仓库时必须重新检查该补丁是否仍需要，不能静默覆盖。

## 本项目的原则

1. `third_party/SegFormer3D/` 仅用于保存上游仓库原始/适配版本；
2. 本项目自研的 DICOM 标准化、骨窗、多中心 QC、骨科数据适配、联合损失、不确定性精修、三维重建与 Web 系统放在本项目自己的 `src/`、`web/` 等目录；
3. 论文中引用 SegFormer3D 原论文和官方实现，不把其基础模型架构宣称为本项目原创；
4. 软件著作权材料必须明确区分第三方开源模块与本项目独立开发内容，不能通过改名、批量替换变量等方式掩盖来源；
5. 如果后续决定不直接分发上游源码，可以采用“用户单独克隆上游仓库 + 本项目适配器加载”的方式保持边界清晰。

## 推荐获取方式

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\env\fetch_segformer3d.ps1
```

脚本将克隆到：

```text
D:\国创项目\third_party\SegFormer3D
```

已有目录时不会强制覆盖。
