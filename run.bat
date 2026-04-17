@echo off
REM =============================================================================
REM Automated Setup and Training Pipeline - Windows
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================================================
echo THERMAL FACE DETECTION - AUTOMATED PIPELINE
echo ================================================================================
echo.

REM Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from https://www.python.org/
    exit /b 1
)

set PYTHON_CMD=python
echo [OK] Python found:
%PYTHON_CMD% --version
echo.

REM ============================================================
REM Check Python version is 3.8 - 3.12 (PyTorch CUDA requirement)
REM ============================================================
for /f "tokens=2 delims= " %%v in ('%PYTHON_CMD% --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! GEQ 13 (
    echo.
    echo ================================================================================
    echo [ERROR] Unsupported Python version: %PY_VER%
    echo ================================================================================
    echo   PyTorch CUDA wheels are only published for Python 3.8 - 3.12.
    echo   Python 3.13 and above are NOT yet supported by PyTorch CUDA builds.
    echo   Using Python %PY_VER% would silently install the CPU-only PyTorch,
    echo   which will fail at the "CUDA not available" check.
    echo.
    echo   Please install Python 3.10 or 3.11 from:
    echo     https://www.python.org/downloads/
    echo.
    echo   Make sure the supported Python is first in your PATH, then re-run run.bat.
    echo ================================================================================
    exit /b 1
)
echo [OK] Python version %PY_VER% is supported.
echo.


REM ============================================================
REM Check NVIDIA GPU and driver (required for CUDA / PyTorch)
REM ============================================================
nvidia-smi >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ================================================================================
    echo [WARNING] NVIDIA driver / nvidia-smi not found.
    echo ================================================================================
    echo   This project requires:
    echo     - An NVIDIA GPU (GTX 1660 Ti 6 GB or better)
    echo     - Up-to-date NVIDIA drivers  ^(installs nvidia-smi^)
    echo     - CUDA-enabled PyTorch       ^(installed automatically by setup^)
    echo.
    echo   If you do NOT have an NVIDIA GPU, this pipeline cannot run.
    echo.
    echo   To install NVIDIA drivers:
    echo     https://www.nvidia.com/download/index.aspx
    echo.
    echo   To install CUDA-enabled PyTorch manually ^(after driver install^):
    echo     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    echo   Press Ctrl+C to abort, or press any key to continue anyway...
    echo ================================================================================
    pause >nul
) else (
    echo [OK] NVIDIA driver detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo.
)

REM Create a timestamp for this run
for /f "tokens=1-3 delims=/ " %%a in ("%DATE%") do set DATESTAMP=%%c-%%a-%%b
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set TIMESTAMP=%%a-%%b-%%c
set RUN_TIMESTAMP=%DATESTAMP%_%TIMESTAMP%
set RUN_LOG_DIR=logs\%RUN_TIMESTAMP%
if not exist "%RUN_LOG_DIR%" mkdir "%RUN_LOG_DIR%"
echo [OK] Run ID: %RUN_TIMESTAMP%

REM Parse arguments
set SKIP_SETUP=0
set SKIP_EXTRACT=0
set PIPELINE_ARGS=

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--skip-setup" (
    set SKIP_SETUP=1
    shift
    goto parse_args
)
if /i "%~1"=="--skip-extract" (
    set SKIP_EXTRACT=1
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
set PIPELINE_ARGS=!PIPELINE_ARGS! %1
shift
goto parse_args
:end_parse

REM Run data extraction unless skipped
if !SKIP_EXTRACT!==0 (
    echo ================================================================================
    echo STEP 0: DATA EXTRACTION
    echo ================================================================================
    echo.

    %PYTHON_CMD% codes\extract_data.py
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Data extraction failed!
        exit /b 1
    )
) else (
    echo [INFO] Skipping data extraction (--skip-extract flag detected)
)

REM Run setup unless skipped
if !SKIP_SETUP!==0 (
    echo ================================================================================
    echo STEP 1: ENVIRONMENT SETUP
    echo ================================================================================
    echo.
    
    %PYTHON_CMD% codes\setup.py
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Setup failed! See messages above for details.
        exit /b 1
    )
) else (
    echo [INFO] Skipping setup (--skip-setup flag detected)
)

REM Run pipeline
echo.
echo ================================================================================
echo STEP 2: RUNNING TRAINING PIPELINE
echo ================================================================================
echo.

%PYTHON_CMD% codes\main_pipeline.py !PIPELINE_ARGS!
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Pipeline failed!
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] PIPELINE COMPLETED SUCCESSFULLY
echo ================================================================================
echo.
echo Run ID: %RUN_TIMESTAMP%
echo Results are available in timestamped subdirectories:
echo   - outputs\^<timestamp^>\models\   (trained model weights)
echo   - outputs\^<timestamp^>\plots\    (training curves and comparisons)
echo   - logs\^<timestamp^>\             (training logs and metrics)
echo.
echo ================================================================================

exit /b 0

:show_help
echo Usage: run.bat [OPTIONS]
echo.
echo Options:
echo   --skip-setup              Skip environment setup (use if already set up)
echo   --skip-extract            Skip data extraction from ZIP files
echo   --models MODEL [MODEL...] Train only specific models (unet, transunet, swin)
echo   --skip-benchmark          Skip benchmarking after training
echo   --epochs N                Number of training epochs (default: 100)
echo   --help, -h                Show this help message
echo.
echo Examples:
echo   run.bat                                  # Full pipeline with setup
echo   run.bat --skip-setup                     # Run without setup
echo   run.bat --skip-extract --skip-setup      # Skip extraction and setup
echo   run.bat --models unet                    # Train only U-Net
echo   run.bat --models transunet swin          # Train TransUNet and Swin-UNet++
echo   run.bat --skip-benchmark                 # Train all, skip benchmark
echo   run.bat --epochs 50                      # Train for 50 epochs
echo   run.bat --skip-setup --models unet       # Quick U-Net training
echo.
exit /b 0
