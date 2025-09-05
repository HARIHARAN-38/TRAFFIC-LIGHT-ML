@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo [INFO] Traffic Light Detection Application
echo [INFO] Script directory: %SCRIPT_DIR%

echo [INFO] Python: 
python --version

echo [INFO] Installing/Updating dependencies
python -m pip install --upgrade pip
python -m pip install -r "%SCRIPT_DIR%\requirements.txt"
echo [INFO] Ensuring OpenCV is installed with GUI support
python -m pip install opencv-python

echo [INFO] Launching Simple Tkinter UI (use Stop to end)
set "PYTHONPATH=%SCRIPT_DIR%"
python "%SCRIPT_DIR%\simple_ui.py" %*

endlocal
