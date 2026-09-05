@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\train_control.ps1" -Action menu
if errorlevel 1 (
  echo.
  echo 训练控制器运行失败，请查看上方错误信息。
  pause
)
