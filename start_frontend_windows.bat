@echo off
setlocal
cd /d "%~dp0frontend"

echo =========================================================================
echo === GallNet Triage: Starting Vite + React Frontend Development Server ===
echo =========================================================================

where npm >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Program Files\nodejs\npm.cmd" (
        set "PATH=C:\Program Files\nodejs;%PATH%"
    )
)

where npm >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Node.js / npm not detected in PATH.
    echo The GallNet Triage web application is also directly available via
    echo the FastAPI backend server at:
    echo     http://127.0.0.1:8000/
    echo.
    echo To use the Vite + React dev server at http://localhost:5173,
    echo install Node.js 18+ from https://nodejs.org and re-run this script.
    echo =========================================================================
    pause
    exit /b 0
)

if not exist node_modules (
    echo Installing frontend dependencies via npm...
    call npm install || exit /b 1
)

echo Starting Vite development server on http://localhost:5173 ...
call npm run dev
pause
