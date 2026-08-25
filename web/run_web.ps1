$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    Write-Host 'Project .venv was not found. Run this first:' -ForegroundColor Yellow
    Write-Host 'powershell -ExecutionPolicy Bypass -File .\env\setup_env.ps1'
    exit 2
}

Set-Location $ProjectRoot
Write-Host 'Starting the Orthopedic CT research Web prototype at http://127.0.0.1:8000' -ForegroundColor Cyan
Write-Host 'The server binds to localhost only by default.'
& $Python -m uvicorn web.backend.app:app --host 127.0.0.1 --port 8000 --reload
