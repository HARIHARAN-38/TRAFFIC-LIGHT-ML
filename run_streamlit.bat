@echo off
setlocal

echo ================================================
echo   Traffic Light Detector - Streamlit Launcher
echo ================================================

echo Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install -r requirements.txt

echo.
echo Launching Streamlit (http://localhost:8501)...
python -m streamlit run app.py

endlocal
