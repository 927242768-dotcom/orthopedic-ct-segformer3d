param(
    [ValidateSet("menu", "start", "new", "resume", "status", "stop")]
    [string]$Action = "menu",
    [string]$Config = "configs/orthopedic_ct_cpu_binary_loss_region_boundary_v13.yaml",
    [int]$MaxEpochs = 800,
    [ValidateSet("formal", "engineering")]
    [string]$PreflightMode = "formal"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExperimentsRoot = Join-Path $ProjectRoot "experiments"
$ControlRoot = Join-Path $ExperimentsRoot ".control"
$PidFile = Join-Path $ControlRoot "train.pid"
$StdoutLog = Join-Path $ControlRoot "train_stdout.log"
$StderrLog = Join-Path $ControlRoot "train_stderr.log"
$LaunchInfo = Join-Path $ControlRoot "launch.json"

New-Item -ItemType Directory -Force -Path $ExperimentsRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ControlRoot | Out-Null

function Get-PythonExe {
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".python\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "没有找到项目 Python。请先运行 env/setup_env.ps1 建立 .venv。"
}

function Get-ConfigPath {
    $path = $Config
    if (-not [System.IO.Path]::IsPathRooted($path)) {
        $path = Join-Path $ProjectRoot $path
    }
    if (-not (Test-Path $path)) {
        throw "训练配置不存在: $path"
    }
    return (Resolve-Path $path).Path
}

function Get-RecordedPid {
    if (-not (Test-Path $PidFile)) { return $null }
    $raw = (Get-Content $PidFile -Raw).Trim()
    $pidValue = 0
    if (-not [int]::TryParse($raw, [ref]$pidValue)) { return $null }
    return $pidValue
}

function Get-RunningTrainingProcess {
    $pidValue = Get-RecordedPid
    if ($null -eq $pidValue) { return $null }
    return Get-Process -Id $pidValue -ErrorAction SilentlyContinue
}

function Get-LatestCompatibleRun {
    param([string]$ConfigPath)

    if (-not (Test-Path $ExperimentsRoot)) { return $null }
    $sourceHash = (Get-FileHash -Algorithm SHA256 $ConfigPath).Hash
    $runs = Get-ChildItem -Path $ExperimentsRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ne ".control" } |
        Sort-Object LastWriteTime -Descending

    foreach ($run in $runs) {
        $copiedConfig = Join-Path $run.FullName "config.yaml"
        $lastCheckpoint = Join-Path $run.FullName "checkpoint\last.pt"
        if ((Test-Path $copiedConfig) -and (Test-Path $lastCheckpoint)) {
            try {
                $runHash = (Get-FileHash -Algorithm SHA256 $copiedConfig).Hash
                if ($runHash -eq $sourceHash) {
                    return [PSCustomObject]@{
                        Run = $run.FullName
                        Checkpoint = $lastCheckpoint
                    }
                }
            } catch {
                continue
            }
        }
    }
    return $null
}

function Show-RunMetrics {
    param([string]$RunDir)
    if (-not $RunDir) { return }
    $history = Join-Path $RunDir "history.csv"
    if (Test-Path $history) {
        $rows = @(Import-Csv $history)
        if ($rows.Count -gt 0) {
            $last = $rows[-1]
            Write-Host ""
            Write-Host "最近完整 epoch:" -ForegroundColor Cyan
            Write-Host ("  epoch={0}  train_loss={1}  val_dice={2}  lr={3}" -f $last.epoch, $last.train_loss, $last.val_dice, $last.lr)
        }
    }
    $best = Join-Path $RunDir "checkpoint\best.pt"
    $lastPt = Join-Path $RunDir "checkpoint\last.pt"
    Write-Host ("  last.pt: {0}" -f (Test-Path $lastPt))
    Write-Host ("  best.pt: {0}" -f (Test-Path $best))
}

function Show-Status {
    $configPath = Get-ConfigPath
    $process = Get-RunningTrainingProcess
    if ($null -ne $process) {
        Write-Host ("训练中：PID {0}，已运行 {1}" -f $process.Id, ((Get-Date) - $process.StartTime)) -ForegroundColor Green
    } else {
        Write-Host "当前没有训练进程。" -ForegroundColor Yellow
    }

    $latest = Get-LatestCompatibleRun -ConfigPath $configPath
    if ($null -ne $latest) {
        Write-Host ("最近兼容 run: {0}" -f $latest.Run)
        Show-RunMetrics -RunDir $latest.Run
    } else {
        Write-Host "还没有找到与当前 config 完全一致、且包含 last.pt 的 run。"
    }

    if (Test-Path $StdoutLog) {
        Write-Host ""
        Write-Host "最近训练输出：" -ForegroundColor Cyan
        Get-Content $StdoutLog -Tail 12 -ErrorAction SilentlyContinue
    }
    if ((Test-Path $StderrLog) -and ((Get-Item $StderrLog).Length -gt 0)) {
        Write-Host ""
        Write-Host "最近错误输出：" -ForegroundColor Red
        Get-Content $StderrLog -Tail 8 -ErrorAction SilentlyContinue
    }
}

function Start-Training {
    param([bool]$ForceNew)

    $existing = Get-RunningTrainingProcess
    if ($null -ne $existing) {
        throw "训练已经在运行，PID=$($existing.Id)。请先 status 或 stop。"
    }

    $python = Get-PythonExe
    $configPath = Get-ConfigPath
    $arguments = @(
        "-m", "src.modeling.train",
        "--config", $configPath,
        "--max-epochs", "$MaxEpochs",
        "--preflight-mode", $PreflightMode,
        "--allow-cpu"
    )

    $resumeInfo = $null
    if (-not $ForceNew) {
        $resumeInfo = Get-LatestCompatibleRun -ConfigPath $configPath
        if ($null -ne $resumeInfo) {
            $arguments += @("--resume", $resumeInfo.Checkpoint)
        }
    }

    Remove-Item $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue

    $process = Start-Process -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru

    Set-Content -Path $PidFile -Value $process.Id -Encoding ascii
    @{
        pid = $process.Id
        started_at = (Get-Date).ToString("o")
        config = $configPath
        max_epochs = $MaxEpochs
        preflight_mode = $PreflightMode
        resumed_from = if ($null -eq $resumeInfo) { $null } else { $resumeInfo.Checkpoint }
    } | ConvertTo-Json | Set-Content -Path $LaunchInfo -Encoding UTF8

    if ($null -ne $resumeInfo) {
        Write-Host "已从最近完整 checkpoint 自动续训：" -ForegroundColor Green
        Write-Host "  $($resumeInfo.Checkpoint)"
    } else {
        Write-Host "已启动一个新的训练 run。" -ForegroundColor Green
    }
    Write-Host "PID=$($process.Id)；MaxEpochs=$MaxEpochs"
    Write-Host "训练在后台运行。可随时再次运行 train_control.cmd 查看状态或中断。"
}

function Stop-Training {
    $process = Get-RunningTrainingProcess
    if ($null -eq $process) {
        Write-Host "当前没有训练进程。" -ForegroundColor Yellow
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        return
    }

    Write-Host "正在中断训练进程树……" -ForegroundColor Yellow
    # 这里故意终止当前未完成 epoch；最近一个完整 epoch 的 checkpoint/last.pt 不会被覆盖。
    & taskkill.exe /PID $process.Id /T /F | Out-Host
    Start-Sleep -Milliseconds 500
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "训练已中断。下次 Start/Resume 会从最近完整 last.pt 继续。" -ForegroundColor Green
}

function Invoke-Menu {
    while ($true) {
        Write-Host ""
        Write-Host "===============================================" -ForegroundColor Cyan
        Write-Host "  orthopedic-ct-segformer3d 本地训练控制"
        Write-Host "===============================================" -ForegroundColor Cyan
        Write-Host "1. 开始 / 自动续训"
        Write-Host "2. 强制新建一次训练"
        Write-Host "3. 查看训练状态"
        Write-Host "4. 随时中断训练（保留最近完整 epoch）"
        Write-Host "5. 退出"
        $choice = Read-Host "请选择"
        try {
            switch ($choice) {
                "1" { Start-Training -ForceNew $false }
                "2" { Start-Training -ForceNew $true }
                "3" { Show-Status }
                "4" { Stop-Training }
                "5" { return }
                default { Write-Host "请输入 1~5。" -ForegroundColor Yellow }
            }
        } catch {
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
    }
}

Set-Location $ProjectRoot
switch ($Action) {
    "menu"   { Invoke-Menu }
    "start"  { Start-Training -ForceNew $false }
    "resume" { Start-Training -ForceNew $false }
    "new"    { Start-Training -ForceNew $true }
    "status" { Show-Status }
    "stop"   { Stop-Training }
}
