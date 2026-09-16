@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" "training_dashboard.py"
) else (
    start "" ".venv\Scripts\python.exe" "training_dashboard.py"
)
exit /b 0
