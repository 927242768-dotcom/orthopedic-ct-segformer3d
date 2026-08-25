$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ThirdPartyRoot = Join-Path $ProjectRoot 'third_party'
$Target = Join-Path $ThirdPartyRoot 'SegFormer3D'
$RepoUrl = 'https://github.com/OSUPCVLab/SegFormer3D.git'

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host 'Git was not found. Install Git first.' -ForegroundColor Red
    exit 2
}

New-Item -ItemType Directory -Force -Path $ThirdPartyRoot | Out-Null

if (Test-Path $Target) {
    Write-Host "Target already exists; nothing was overwritten: $Target" -ForegroundColor Yellow
    Write-Host 'If an update is needed, inspect local changes and run git pull manually.'
    exit 0
}

Write-Host 'Cloning the official SegFormer3D repository ...' -ForegroundColor Cyan
git clone $RepoUrl $Target
if ($LASTEXITCODE -ne 0) {
    throw 'git clone failed'
}

if (-not (Test-Path (Join-Path $Target 'LICENSE'))) {
    Write-Host 'Warning: LICENSE was not found after clone. Verify the repository before use.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host "Done: $Target" -ForegroundColor Green
Write-Host 'Keep the upstream LICENSE and citation information. Do not claim the upstream backbone as original project code.'
