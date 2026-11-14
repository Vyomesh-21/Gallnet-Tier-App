@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ============================================================
echo === GallNet Triage: Windows Setup                       ===
echo ============================================================

where py >nul 2>nul
if %errorlevel%==0 (
    py -3.12 --version >nul 2>nul && set "PY=py -3.12"
    if not defined PY py -3.11 --version >nul 2>nul && set "PY=py -3.11"
    if not defined PY set "PY=py"
) else (
    set "PY=python"
)

echo Detected Python runner: %PY%
%PY% --version || (
    echo Python not available. Please install Python 3.11 or 3.12 from python.org.
    pause
    exit /b 1
)

if not exist backend\.venv (
    echo Creating virtual environment in backend\.venv ...
    %PY% -m venv backend\.venv || exit /b 1
) else (
    echo Virtual environment already exists at backend\.venv.
)

call backend\.venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip || exit /b 1

echo Installing backend dependencies from backend\requirements.txt ...
python -m pip install -r backend\requirements.txt || exit /b 1

echo.
echo Verifying pretrained model checkpoint...
if exist models\model-031-0.979685-0.992509.keras (
    echo Existing pretrained checkpoint found: models\model-031-0.979685-0.992509.keras
) else (
    python backend\download_pretrained.py --checkpoint-only || (
        echo Could not download model checkpoint. Please place model-031-0.979685-0.992509.keras into models\
    )
)

echo.
echo ============================================================
echo Setup complete.
echo - Start backend: double-click start_backend_windows.bat
echo   (Backend dashboard available at: http://127.0.0.1:8000)
echo - Start frontend: double-click start_frontend_windows.bat
echo ============================================================
pause
