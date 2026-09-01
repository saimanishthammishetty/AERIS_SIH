@echo off
TITLE AERIS - Automated Setup & Environment Verification
COLOR 0B

echo =============================================================================
echo                PROJECT AERIS - AUTOMATED SYSTEM SETUP
echo       Adaptive Intelligent Noise Cancellation & Acoustic Defense Suite
echo =============================================================================
echo.

:: 1. Verify Python Installation
echo [*] Step 1/4: Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in system PATH.
    echo Please install Python 3.10+ from https://python.org and add to PATH.
    pause
    exit /b 1
)
python --version
echo [+] Python environment detected.
echo.

:: 2. Install Core Dependencies
echo [*] Step 2/4: Installing required Python packages from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Package installation encountered issues. Continuing check...
) else (
    echo [+] All dependencies installed successfully.
)
echo.

:: 3. Train / Verify AI Model Artifacts
echo [*] Step 3/4: Training AI Noise Classifier & Generating Artifacts...
python AERIS_AI_TRAIN.py
if %errorlevel% neq 0 (
    echo [ERROR] AI Model training failed. Check dataset paths.
    pause
    exit /b 1
)
echo [+] AI Classifier trained and artifacts verified.
echo.

:: 4. Final System Verification
echo [*] Step 4/4: Performing System Health Check...
python -c "import AERIS_AUDIO, AERIS_BACKEND; print('[+] AERIS Audio DSP and Backend modules verified!')"
if %errorlevel% neq 0 (
    echo [ERROR] Module verification failed.
    pause
    exit /b 1
)

echo.
echo =============================================================================
echo             [SUCCESS] PROJECT AERIS SETUP COMPLETE & VERIFIED!
echo =============================================================================
echo To start the complete system (Backend + Command Center Dashboard):
echo Execute: run_aeris.bat
echo =============================================================================
echo.
pause
