@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Myanmar Recap Studio - Installer

echo ============================================================
echo   Myanmar Recap Studio  -  Windows Installer
echo ============================================================
echo.

REM ---- working folders ----
for %%D in (data downloads fonts cloned_voices tools) do (
    if not exist "%%D" mkdir "%%D"
)
if not exist "data\generated_audio" mkdir "data\generated_audio"

REM ============================================================
REM 1. Python
REM ============================================================
call :find_python
if defined PYEXE goto :have_python
echo [1/5] Python not found - downloading Python 3.12 ...
call :install_python
call :find_python
:have_python
if not defined PYEXE (
    echo.
    echo [ERROR] Python could not be installed automatically.
    echo         Please install Python 3.12 from https://www.python.org/downloads/
    echo         ^(tick "Add python.exe to PATH"^) then run this installer again.
    echo.
    pause
    exit /b 1
)
echo [1/5] Python found: !PYEXE!

REM ============================================================
REM 2. ffmpeg
REM ============================================================
set "FFDIR=%~dp0tools\ffmpeg"
set "HAVE_FF="
where ffmpeg >nul 2>nul
if not errorlevel 1 set "HAVE_FF=1"
where ffprobe >nul 2>nul
if not errorlevel 1 set "HAVE_FF=1"
if defined HAVE_FF (
    echo [2/5] ffmpeg already installed on this PC.
    goto :ff_done
)
if exist "%FFDIR%\bin\ffmpeg.exe" (
    echo [2/5] ffmpeg already present in tools\ffmpeg.
    goto :ff_done
)
echo [2/5] ffmpeg not found - downloading ffmpeg ^(about 80 MB^) ...
call :install_ffmpeg
:ff_done

REM ============================================================
REM 3. virtual environment
REM ============================================================
if not exist ".venv\Scripts\python.exe" (
    echo [3/5] Creating virtual environment ^(.venv^) ...
    "!PYEXE!" -m venv ".venv"
)
set "VENVPY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENVPY%" (
    echo [ERROR] Virtual environment could not be created.
    pause
    exit /b 1
)

REM ============================================================
REM 4. Python packages
REM ============================================================
echo [4/5] Installing Python packages ... ^(this may take several minutes^)
"%VENVPY%" -m pip install --upgrade pip
"%VENVPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [WARN] Some packages failed - retrying the main list once ...
    "%VENVPY%" -m pip install -r requirements.txt
)
"%VENVPY%" -m pip install --upgrade "yt-dlp[default]"
"%VENVPY%" -m pip install --upgrade Pillow

REM ============================================================
REM 5. secrets file
REM ============================================================
if not exist "secrets.bat" (
    > "secrets.bat" echo @echo off
    >>"secrets.bat" echo REM ============================================================
    >>"secrets.bat" echo REM  Put your Google Gemini API key below, then save this file.
    >>"secrets.bat" echo REM  Get a free key at: https://aistudio.google.com/apikey
    >>"secrets.bat" echo REM ============================================================
    >>"secrets.bat" echo set "GOOGLE_API_KEY="
)

echo.
echo ============================================================
echo   INSTALL COMPLETE
echo ------------------------------------------------------------
echo   NEXT STEPS:
echo     1^) Open  secrets.bat  and paste your GOOGLE_API_KEY after  set
echo     2^) Double-click  run.bat  to start the app
echo        ^(a browser will open at http://127.0.0.1:7860^)
echo ============================================================
echo.
pause
exit /b 0

REM ============================================================
REM Subroutines
REM ============================================================
:find_python
set "PYEXE="
for /f "delims=" %%i in ('py -3.12 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if defined PYEXE exit /b
for /f "delims=" %%i in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if defined PYEXE exit /b
for /f "delims=" %%i in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PYEXE=%%i"
if defined PYEXE exit /b
exit /b

:install_python
set "PYINST=%TEMP%\myanmar-python-setup.exe"
for %%V in (3.12.10 3.12.9 3.12.8 3.12.7) do (
    if not exist "%PYINST%" (
        echo     trying Python %%V ...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "try{Invoke-WebRequest -UseBasicParsing -Uri 'https://www.python.org/ftp/python/%%V/python-%%V-amd64.exe' -OutFile '%PYINST%'}catch{}" >nul 2>nul
    )
)
if not exist "%PYINST%" (
    echo     download failed.
    exit /b
)
echo     installing Python ^(per-user, no admin needed^) ...
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1
del "%PYINST%" >nul 2>nul
exit /b

:install_ffmpeg
set "FFZIP=%TEMP%\myanmar-ffmpeg.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "try{Invoke-WebRequest -UseBasicParsing -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%FFZIP%'}catch{}" >nul 2>nul
if not exist "%FFZIP%" (
    echo     ffmpeg download failed - the app can still run, but video rendering needs ffmpeg.
    exit /b
)
echo     extracting ffmpeg ...
if exist "tools\_ff_tmp" rmdir /s /q "tools\_ff_tmp"
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%FFZIP%' -DestinationPath 'tools\_ff_tmp' -Force" >nul 2>nul
rmdir /s /q "%FFDIR%" >nul 2>nul
for /d %%d in ("tools\_ff_tmp\*") do move "%%d" "%FFDIR%" >nul
if exist "tools\_ff_tmp" rmdir /s /q "tools\_ff_tmp"
del "%FFZIP%" >nul 2>nul
if exist "%FFDIR%\bin\ffmpeg.exe" (
    echo     ffmpeg installed in tools\ffmpeg.
) else (
    echo     ffmpeg extraction failed.
)
exit /b
