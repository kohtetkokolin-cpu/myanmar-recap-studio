@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Myanmar Recap Studio

if not exist ".venv\Scripts\python.exe" (
    echo The app is not installed yet.
    echo Please run  install.bat  first.
    echo.
    pause
    exit /b 1
)

REM ---- optional API keys (GOOGLE_API_KEY, etc.) ----
if exist "secrets.bat" call "secrets.bat"

REM ---- use bundled ffmpeg if present ----
set "PATH=%~dp0tools\ffmpeg\bin;%PATH%"

set "PORT=7860"
if defined PORT_OVERRIDE set "PORT=%PORT_OVERRIDE%"

echo ============================================================
echo   Starting Myanmar Recap Studio ...
echo   Address:  http://127.0.0.1:%PORT%
echo   Stop the app by closing this window ^(or press Ctrl+C^).
echo ============================================================
echo.

REM open the browser a few seconds after the server starts
start "" cmd /c "timeout /t 6 >nul & start "" http://127.0.0.1:%PORT%"

".venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port %PORT%

echo.
echo The app has stopped.
pause
