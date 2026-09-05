# 本地训练启停与自动续训

本项目已经支持按完整 epoch 保存 `checkpoint/last.pt` 与 `checkpoint/best.pt`，并支持 `--resume`。本文件新增一层 Windows 本地控制入口，目标是让训练可以随时开始、随时中断、下一次自动从最近完整 epoch 继续。

## 最简单用法

在项目根目录双击：

```text
train_control.cmd
```

菜单提供：

1. 开始 / 自动续训
2. 强制新建一次训练
3. 查看训练状态
4. 随时中断训练（保留最近完整 epoch）
5. 退出

默认训练配置：

```text
configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml
```

默认总目标 epoch：

```text
800
```

## 训练中断原则

“中断”会直接结束当前训练进程树，但不会删除已有实验目录和 checkpoint。

`train.py` 每完成一个 epoch 都会写入：

```text
experiments/<run>/checkpoint/last.pt
```

并在 validation Dice 创新高时写入：

```text
experiments/<run>/checkpoint/best.pt
```

因此如果在某个 epoch 中途停止：

- 当前尚未完成的 epoch 会被丢弃；
- 最近一个已经完整完成的 `last.pt` 会保留；
- 下一次选择“开始 / 自动续训”时，会寻找与当前 config SHA-256 完全一致的最近 run，并从该 `last.pt` 继续；
- 这样不会把半个 epoch 当作正式训练进度，实验记录更干净。

## 后台运行和日志

训练控制器会把 PID 和控制日志存放在：

```text
experiments/.control/
```

主要包括：

```text
train.pid
launch.json
train_stdout.log
train_stderr.log
```

这些属于运行时文件，不应作为论文结果或模型文件提交。

## 命令行方式

也可以不打开菜单，直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action start
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action status
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action stop
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action new
```

修改总 epoch：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action start -MaxEpochs 1200
```

修改配置：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\train_control.ps1 -Action start -Config configs\your_config.yaml
```

## CPU / GPU

控制器始终给 `train.py` 传入 `--allow-cpu`。这并不会强制使用 CPU：

- 如果 PyTorch 检测到 CUDA，`train.py` 仍会使用 CUDA；
- 如果没有 CUDA，则允许在 CPU 上训练；
- 数据、split、QC 和 task 的其它 formal preflight 保护仍然保留。

## 关于“训练到达到要求”

不要把“某一次 validation Dice 达到阈值”直接当成最终项目达标。建议按以下顺序判断：

1. 训练集和 validation 指标稳定；
2. validation Dice/HD95/ASSD 达到预设标准；
3. 参数锁定；
4. 使用从未参与模型选择的 fresh test set 做正式测试；
5. 再决定是否达到任务书/论文要求。

当前旧的 `liver_169` 已经被用于一次正式 independent test，因此未来扩大数据重新训练时，应建立新的 patient-level train/validation/fresh-test split，不能继续把 `liver_169` 当作全新的独立测试集。
