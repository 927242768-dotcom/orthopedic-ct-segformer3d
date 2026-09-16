@echo off
chcp 65001 >nul
title CTSpine1K 完整数据下载到 H 盘
cd /d "%~dp0"
".venv\Scripts\python.exe" "env\download_ctspine1k_full.py"
echo.
echo 如果显示完成，就可以关闭窗口；如果网络中断，再双击本文件即可续传。
pause
