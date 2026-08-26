$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ThirdPartyRoot = Join-Path $ProjectRoot 'third_party'
$Target = Join-Path $ThirdPartyRoot 'SegFormer3D'
$RepoUrl = 'https://github.com/OSUPCVLab/SegFormer3D.git'
$ExpectedCommit = 'e314242f14b6731458130809945a0ee27f4298bd'
$CompatibilityPatch = Join-Path (Join-Path $PSScriptRoot 'patches') 'segformer3d_torch21_cube_root.patch'

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

Write-Host "Checking out pinned upstream commit: $ExpectedCommit" -ForegroundColor Cyan
git -C $Target checkout --detach $ExpectedCommit
if ($LASTEXITCODE -ne 0) {
    throw "failed to checkout pinned upstream commit: $ExpectedCommit"
}

if (-not (Test-Path $CompatibilityPatch)) {
    throw "compatibility patch was not found: $CompatibilityPatch"
}

Write-Host 'Applying the tracked PyTorch 2.1 compatibility patch ...' -ForegroundColor Cyan
git -C $Target apply --check $CompatibilityPatch
if ($LASTEXITCODE -ne 0) {
    throw 'SegFormer3D compatibility patch no longer applies cleanly; inspect upstream before continuing'
}
git -C $Target apply $CompatibilityPatch
if ($LASTEXITCODE -ne 0) {
    throw 'failed to apply SegFormer3D compatibility patch'
}

if (-not (Test-Path (Join-Path $Target 'LICENSE'))) {
    Write-Host 'Warning: LICENSE was not found after clone. Verify the repository before use.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host "Done: $Target" -ForegroundColor Green
Write-Host "Pinned upstream commit: $ExpectedCommit"
Write-Host "Applied compatibility patch: $CompatibilityPatch"
Write-Host 'Keep the upstream LICENSE and citation information. Do not claim the upstream backbone as original project code.'
