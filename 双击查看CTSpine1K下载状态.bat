@echo off
chcp 65001 >nul
title CTSpine1K 下载状态
powershell -NoProfile -Command "$root='H:\CTSpine1K\raw_data'; $status='H:\CTSpine1K\_download_status.json'; if(Test-Path $status){Write-Host '=== 状态文件 ==='; Get-Content $status -Raw}; if(Test-Path $root){$files=Get-ChildItem $root -Recurse -File -ErrorAction SilentlyContinue; $size=(($files|Measure-Object Length -Sum).Sum); Write-Host '=== 当前落盘 ==='; Write-Host ('文件数: '+$files.Count); Write-Host ('大小: '+[math]::Round($size/1GB,2)+' GB')}; $d=Get-PSDrive H; Write-Host ('H盘剩余: '+[math]::Round($d.Free/1GB,2)+' GB')"
pause
