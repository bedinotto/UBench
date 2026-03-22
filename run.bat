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

REM Parse arguments
set SKIP_SETUP=0
set PIPELINE_ARGS=

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--skip-setup" (
    set SKIP_SETUP=1
    shift
    goto parse_args
)
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
set PIPELINE_ARGS=!PIPELINE_ARGS! %1
shift
goto parse_args
:end_parse

REM Run setup unless skipped
if !SKIP_SETUP!==0 (
    echo ================================================================================
    echo STEP 1: ENVIRONMENT SETUP
    echo ================================================================================
    echo.
    
    %PYTHON_CMD% codes\setup.py
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Setup failed!
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
echo Results are available in:
echo   - outputs\models\     (trained model weights)
echo   - outputs\plots\      (training curves and comparisons)
echo   - log\                (training logs and metrics)
echo.
echo ================================================================================

exit /b 0

:show_help
echo Usage: run.bat [OPTIONS]
echo.
echo Options:
echo   --skip-setup              Skip environment setup (use if already set up)
echo   --models MODEL [MODEL...] Train only specific models (unet, transunet, swin)
echo   --skip-benchmark          Skip benchmarking after training
echo   --epochs N                Number of training epochs (default: 100)
echo   --help, -h                Show this help message
echo.
echo Examples:
echo   run.bat                                  # Full pipeline with setup
echo   run.bat --skip-setup                     # Run without setup
echo   run.bat --models unet                    # Train only U-Net
echo   run.bat --models transunet swin          # Train TransUNet and Swin-UNet++
echo   run.bat --skip-benchmark                 # Train all, skip benchmark
echo   run.bat --epochs 50                      # Train for 50 epochs
echo   run.bat --skip-setup --models unet       # Quick U-Net training
echo.
exit /b 0
