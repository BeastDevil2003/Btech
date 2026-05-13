@echo off
REM AFAGNetV3 Project Launcher (Windows)
REM This script launches the AFAGNetV3 project manager

echo AFAGNetV3 Project 
echo ===========================
echo.

REM Check if we're in the right directory
if not exist "project.py" (
    echo Error: project.py not found in current directory
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Run the launcher
python launcher.py %*

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Press any key to exit...
    pause >nul
)