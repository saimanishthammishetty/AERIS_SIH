@echo off
:: AERIS — Shutdown script
:: Stops the FastAPI backend server running on port 8000
:: Usage: stop_aeris.bat

title AERIS Command Center - Shutdown

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║       AERIS v2.4 — SHUTDOWN UTILITY                    ║
echo  ║       Locating and stopping service on port 8000       ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

setlocal enabledelayedexpansion
set FOUND=0

for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r /c:":8000 " ^| findstr LISTENING') do (
    set PID=%%a
    echo [*] Found active AERIS server running on PID !PID!
    taskkill /f /pid !PID! >nul 2>&1
    if !errorlevel! equ 0 (
         echo [✓] Successfully terminated process PID !PID!
         set /a FOUND+=1
    ) else (
         echo [X] Failed to terminate process PID !PID!
    )
)

if !FOUND! equ 0 (
    echo [!] No active processes found listening on port 8000.
) else (
    echo [✓] Service shutdown complete. terminated !FOUND! process(es).
)

echo.
pause
