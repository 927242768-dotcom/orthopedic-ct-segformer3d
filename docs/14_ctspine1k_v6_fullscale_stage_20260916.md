# CTSpine1K v6 全规模训练阶段快照（2026-09-16）

> 状态：阶段进行中。本文记录截至 2026-09-16 18:08 的真实工程进展，不把尚未结束的训练或验证写成最终论文结果。

## 1. 本阶段主线

项目正式主线继续保持 **SegFormer3D Encoder + C1/C2/C3/C4 多尺度融合 + High-Resolution 3D Decoder + CT shallow spatial branch**，模型键为 `segformer3d_hr_decoder`。本阶段没有把 SegFormer3D 从主方案中移除，也没有回退到已验证较弱的 `direct_plus_coarse` 方案。

当前 full-scale 配置：`configs/orthopedic_ct_full_large_scale_v6_segformer3d_hr_formal610_val197.yaml`。

## 2. 数据与预处理闭环

- CTSpine1K 开发集缓存已完成：807/807，failure=0。
- split 固定为 train=610、validation=197、test=0。
- `test_private` 198 例仍未读取、未缓存、未用于调参或验证。
- v6 cache pipeline：`0.6.0-compact-u16-fixed-hu-minmax-1p5mm-nibsafe`。
- 807 例全量审计：missing=0、qc_bad=0、geometry_bad=0、suspicious_HU=0。
- cache CRC 标志与 image/label affine 对齐审计均通过。

本阶段定位并修复了 Windows 下压缩 NIfTI 读取的随机 CRC/异常数值问题。预处理端增加稳定读取与 zlib 顺序解压 fallback；训练 dataset 改用 SimpleITK 稳定读取，同时保持历史 XYZ -> DHW 轴语义不变。

## 3. 训练与恢复链

- full-scale 只从稳定化 pilot `best.pt` 加载模型权重；optimizer / scheduler 从头初始化。
- `RUNNING.lock` 防止重复训练。
- `last.pt` / `best.pt` 正常生成。
- checkpoint 已确认包含 model / optimizer / scheduler / Python RNG / NumPy RNG / Torch RNG 等恢复状态。
- `scripts/auto_continue_ctspine1k_v6.py` 负责训练完成后继续 validation197 full-volume evaluation，并明确禁止 test split。
- 新增训练期资源 guard，只在训练存活期间回收异常膨胀的 Clash Verge WebView renderer working set，不停止代理核心。
## 4. 当前 full610 真实结果

运行目录：`experiments/20260916_163356_ctspine1k_v6_segformer3d_hr_full610_val197`。

截至本快照，已完成 6 个完整 epoch，当前进入 epoch 7。固定 validation197 patch selector 的结果：

| Epoch | Train loss | Val Dice | Precision | Recall | Pred/Target FG | Component error |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.65718 | 0.81320 | 0.79404 | 0.84987 | 1.0886 | 15.83 |
| 2 | 0.55575 | 0.82648 | 0.78515 | 0.88588 | 1.1485 | 14.83 |
| 3 | 0.56591 | 0.82718 | 0.81649 | 0.85421 | 1.0556 | 26.01 |
| 4 | 0.51435 | 0.84610 | 0.84371 | 0.86336 | 1.0463 | 6.53 |
| 5 | 0.49599 | 0.84539 | 0.81741 | 0.89005 | 1.1035 | 23.38 |
| 6 | 0.50536 | **0.85848** | 0.84543 | 0.88309 | 1.0548 | 17.57 |

当前 best patch-selector Dice=`0.8584831380`，已超过此前 32-case stabilized pilot best Dice=`0.8215395933`，并接近当前 config 的 `target_val_dice=0.86`。该指标仅用于开发期 checkpoint selector，不等价于最终 full-volume validation197 指标。

## 5. 当前门禁与下一阶段

当前 training engineering preflight：ready=true，checked cases=807，error=0，warning=0。当前 CPU 环境没有 CUDA，因此这次 run 仍属于 full-scale engineering development，不冒充最终论文正式结果。

本阶段结束条件：

1. 当前 full610 训练结束或 early stopping 正常触发；
2. 使用锁定的 `best.pt` 对 validation197 执行 full-volume sliding-window evaluation；
3. 汇总 Dice / IoU / Precision / Recall / HD95 / ASSD / foreground ratio / component error / per-case CSV / worst cases；
4. 只在上述开发阶段完成并锁定方案后，才讨论是否进入唯一一次最终 test198 评估。

在 full-volume validation197 完成前，不根据 test198 做任何调参，不把 patch-selector Dice 写成最终模型性能。
