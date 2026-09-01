@echo off
:: AERIS — Single-command launch script
:: Starts: FastAPI backend + opens dashboard in browser
:: Usage: run_aeris.bat

title AERIS Command Center

set PROJ=C:\Users\nuppe\OneDrive\Documents\AERIS_SIH
cd /d "%PROJ%"

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║       AERIS v2.4 — ADAPTIVE NOISE CANCELLATION          ║
echo  ║       Starting backend on http://127.0.0.1:8000          ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1 || (
    echo [ERROR] Python not found. Please install Python 3.9+.
    pause
    exit /b 1
)

:: Install missing deps silently
pip install fastapi uvicorn scikit-learn librosa soundfile numpy scipy --quiet 2>nul

:: Run verification first
echo [*] Running component verification...
python tests\verify_all.py
if %errorlevel% neq 0 (
    echo [ERROR] Verification failed. Fix errors before proceeding.
    pause
    exit /b 1
)

echo.
echo [*] Starting FastAPI backend (CTRL+C to stop)...
echo [*] Dashboard will open at: http://127.0.0.1:8000
echo.

:: Open browser after 2s
start /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

:: Start server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

pause
