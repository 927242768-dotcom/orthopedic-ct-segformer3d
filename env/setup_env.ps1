param(
    [ValidateSet('Auto', 'CPU', 'CUDA118')]
    [string]$TorchMode = 'Auto'
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot '.venv'
$LocalPythonRoot = Join-Path $ProjectRoot '.python'
$Requirements = Join-Path $PSScriptRoot 'requirements.txt'

Write-Host '=== Orthopedic CT project-local environment ===' -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"
Write-Host "Virtual env:  $VenvPath"

# Prefer an existing Python 3.11. If it is unavailable, use uv to install
# Python 3.11.7 into this project only. Do not register it system-wide.
$Python311 = $null
try {
    $Python311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
} catch {
    $Python311 = $null
}

if (-not $Python311 -and (Test-Path $LocalPythonRoot)) {
    $LocalPythonExe = Get-ChildItem -Path $LocalPythonRoot -Recurse -Filter python.exe -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($LocalPythonExe) {
        $Version = (& $LocalPythonExe.FullName -c "import sys; print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))").Trim()
        if ($Version -eq '3.11') {
            $Python311 = $LocalPythonExe.FullName
        }
    }
}

if (-not $Python311) {
    $Uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $Uv) {
        Write-Host 'Python 3.11 and uv were not found.' -ForegroundColor Yellow
        Write-Host 'Install Python 3.11.x or uv, then run this script again.'
        Write-Host 'The script will not force the upstream stack into Python 3.13.'
        exit 2
    }

    Write-Host 'Python 3.11 not found. Installing Python 3.11.7 under .python with uv ...' -ForegroundColor Cyan
    & uv python install 3.11.7 --install-dir $LocalPythonRoot --no-bin --no-registry
    if ($LASTEXITCODE -ne 0) {
        throw 'uv failed to install project-local Python 3.11.7'
    }

    $LocalPythonExe = Get-ChildItem -Path $LocalPythonRoot -Recurse -Filter python.exe -File | Select-Object -First 1
    if (-not $LocalPythonExe) {
        throw "python.exe was not found under $LocalPythonRoot after uv installation"
    }
    $Python311 = $LocalPythonExe.FullName
}

Write-Host "Using Python 3.11: $Python311" -ForegroundColor Green

if (-not (Test-Path $VenvPath)) {
    Write-Host 'Creating project-local .venv ...'
    & $Python311 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create .venv'
    }
} else {
    Write-Host '.venv already exists; reusing it.'
}

$Python = Join-Path $VenvPath 'Scripts\python.exe'
if (-not (Test-Path $Python)) {
    throw "Virtual environment Python not found: $Python"
}

Write-Host 'Upgrading pip/wheel ...'
& $Python -m pip install --upgrade pip wheel
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to upgrade pip toolchain'
}

# Prefer uv for package installation when available. It is substantially faster on
# Windows and resolves dependency conflicts before modifying the environment.
$UvCommand = Get-Command uv -ErrorAction SilentlyContinue

if ($TorchMode -eq 'Auto') {
    $HasNvidia = $false
    try {
        $null = Get-Command nvidia-smi -ErrorAction Stop
        $HasNvidia = $true
    } catch {
        $HasNvidia = $false
    }

    if ($HasNvidia) {
        $TorchMode = 'CUDA118'
    } else {
        $TorchMode = 'CPU'
    }
}

if ($TorchMode -eq 'CUDA118') {
    Write-Host 'Installing PyTorch 2.1.0 CUDA 11.8 wheels ...' -ForegroundColor Cyan
    $TorchIndex = 'https://download.pytorch.org/whl/cu118'
} else {
    Write-Host 'Installing PyTorch 2.1.0 CPU wheels ...' -ForegroundColor Cyan
    $TorchIndex = 'https://download.pytorch.org/whl/cpu'
}

if ($UvCommand) {
    & uv pip install --python $Python --reinstall torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url $TorchIndex
} else {
    & $Python -m pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url $TorchIndex
}
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install PyTorch'
}

Write-Host 'Installing project requirements ...'
if ($UvCommand) {
    & uv pip install --python $Python -r $Requirements
} else {
    & $Python -m pip install -r $Requirements
}
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install project requirements'
}

Write-Host ''
Write-Host 'Environment check:' -ForegroundColor Cyan
& $Python -c "import sys, torch; print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
& $Python -c "import pydicom, SimpleITK, nibabel, monai, fastapi, einops, scipy, lightning, pytorch_lightning; print('Medical/Web/SegFormer3D dependencies: OK')"
if ($LASTEXITCODE -ne 0) {
    throw 'Environment import check failed'
}

Write-Host ''
Write-Host 'Done. Activate with:' -ForegroundColor Green
Write-Host '.\.venv\Scripts\Activate.ps1'
Write-Host ''
Write-Host 'This script modifies only this project .python/.venv and does not remove or alter other Python/Conda environments.'
