$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    Write-Host 'Project .venv was not found. Run setup_env.ps1 first.' -ForegroundColor Yellow
    exit 2
}

Set-Location $ProjectRoot
& $Python -m src.modeling.formal_readiness @args
exit $LASTEXITCODE
