@echo off
setlocal
cd /d "%~dp0backend"

:: Verified image scaling from training notebook (OpenCV BGR resized to 128x128 / 255.0)
set "MODEL_PREPROCESS=zero_one"

:: Set internal cache directories within project workspace to prevent Windows permission issues
set "KERAS_HOME=%~dp0.cache\keras"
set "MPLCONFIGDIR=%~dp0.cache\matplotlib"
set "TF_ENABLE_ONEDNN_OPTS=0"

if exist .venv\Scripts\python.exe (
    set "PY_EXE=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul && (
        py -3.12 --version >nul 2>nul && set "PY_EXE=py -3.12"
        if not defined PY_EXE py -3.11 --version >nul 2>nul && set "PY_EXE=py -3.11"
        if not defined PY_EXE set "PY_EXE=py"
    ) || (
        set "PY_EXE=python"
    )
)

echo =========================================================================
echo === GallNet Triage: Starting Backend Server                           ===
echo =========================================================================
echo Server URL: http://127.0.0.1:8000
echo Web UI:     http://127.0.0.1:8000/
echo Health:     http://127.0.0.1:8000/health
echo API Docs:   http://127.0.0.1:8000/docs
echo Using:      %PY_EXE%
echo =========================================================================

%PY_EXE% -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
pause
