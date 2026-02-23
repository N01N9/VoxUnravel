@echo off
setlocal
echo ==================================================
echo VoxUnravel Automated Environment Setup
echo ==================================================

:: Check for Python 3.11 via py launcher
py -3.11 --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    :: Try fallback checking just 'python' if it is 3.11
    python --version 2>&1 | findstr /R "3\.11" >nul
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python 3.11 is not installed, or not configured properly.
        echo Please install Python 3.11 and try again.
        pause
        exit /b 1
    ) ELSE (
        set PYTHON_CMD=python
    )
) ELSE (
    set PYTHON_CMD=py -3.11
)

echo Using Python command: %PYTHON_CMD%
echo.

echo [1/5] Setting up Main Environment (.venv_main)...
if not exist .venv_main (
    %PYTHON_CMD% -m venv .venv_main
)
.venv_main\Scripts\python.exe -m pip install --upgrade pip --no-cache-dir
.venv_main\Scripts\python.exe -m pip install -r requirements_main.txt --no-cache-dir


echo.
echo [2/5] Setting up Separation Environment (.venv_sep)...
if not exist .venv_sep (
    %PYTHON_CMD% -m venv .venv_sep
)
.venv_sep\Scripts\python.exe -m pip install --upgrade pip
echo Installing PyTorch for Separation...
.venv_sep\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
.venv_sep\Scripts\python.exe -m pip install -r requirements_sep.txt --no-cache-dir


echo.
echo [3/5] Setting up Diarization Environment (.venv_dia)...
if not exist .venv_dia (
    %PYTHON_CMD% -m venv .venv_dia
)
.venv_dia\Scripts\python.exe -m pip install --upgrade pip

echo Installing PyTorch 2.1.1 for Diarization...
.venv_dia\Scripts\python.exe -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir

echo Fixing build tools for Github packages...
.venv_dia\Scripts\python.exe -m pip install "setuptools<70.0.0" wheel flit_core

echo Installing pyannote.audio and diarizen without build isolation...
.venv_dia\Scripts\python.exe -m pip install "pyannote.audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio" --no-build-isolation
.venv_dia\Scripts\python.exe -m pip install "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git@2418425e65814cdfa5fa0ec7051b20c76bf6fa05" --no-build-isolation

echo Installing remaining Diarization requirements...
.venv_dia\Scripts\python.exe -m pip install -r requirements_dia.txt --no-cache-dir

echo.
echo [4/5] Setting up ASR Environment (.venv_asr)...
if not exist .venv_asr (
    %PYTHON_CMD% -m venv .venv_asr
)
.venv_asr\Scripts\python.exe -m pip install --upgrade pip
echo Installing PyTorch for ASR...
.venv_asr\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir
.venv_asr\Scripts\python.exe -m pip install -r requirements_asr.txt --no-cache-dir


echo.
echo [5/5] Configuring data\environments.json...
if not exist data mkdir data
(
echo {
echo     "main": ".venv_main/Scripts/python.exe",
echo     "inference": ".venv_sep/Scripts/python.exe",
echo     "separation": ".venv_sep/Scripts/python.exe",
echo     "diarization": ".venv_dia/Scripts/python.exe",
echo     "asr": ".venv_asr/Scripts/python.exe",
echo     "pipeline": ".venv_main/Scripts/python.exe"
echo }
) > data\environments.json

echo.
echo ==================================================
echo    Setup Completed Successfully!
echo    You can now start the App.
echo    Command: .venv_main\Scripts\python.exe main_gui.py
echo ==================================================
pause
endlocal
