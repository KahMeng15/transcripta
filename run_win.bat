@echo off
:: Transcripta Launcher for Windows
title Transcripta CLI

cd /d %~dp0

:: Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

:: Run the transcriber
python transcriber.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Application crashed or failed to start.
    pause
)
