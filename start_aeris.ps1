# AERIS — Single-command PowerShell launch script
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       AERIS v2.5 — ADAPTIVE NOISE CANCELLATION          ║" -ForegroundColor Cyan
Write-Host "║       Starting backend on http://127.0.0.1:8000          ║" -ForegroundColor Cyan
Write-Host "║       Web dashboard: http://127.0.0.1:8000               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$projDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projDir

# Run test verification
Write-Host "[*] Running component verification..." -ForegroundColor Yellow
python tests\verify_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Component verification failed." -ForegroundColor Red
    exit 1
}

Write-Host "[*] Launching browser dashboard in background..." -ForegroundColor Green
Start-Process "http://127.0.0.1:8000"

Write-Host "[*] Starting FastAPI backend on port 8000..." -ForegroundColor Green
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
