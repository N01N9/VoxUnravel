@echo off
echo ==================================================
echo Starting VoxUnravel - MSST
echo ==================================================

if not exist ".venv_main\Scripts\python.exe" (
    echo [ERROR] Main environment not found. 
    echo Please run 'install.bat' first to set up the environments.
    pause
    exit /b 1
)

echo Launching application...
.venv_main\Scripts\python.exe main_gui.py

pause
