@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYEXE="
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.12"
    if not defined PYEXE py -3.13 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.13"
    if not defined PYEXE py -3.11 -c "import sys" >nul 2>&1 && set "PYEXE=py -3.11"
)
if not defined PYEXE (
    where python >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo Python 3.11 - 3.13 not found.
    echo Install Python 3.12 from python.org with Python launcher enabled.
    pause
    exit /b 1
)
echo Using %PYEXE%
%PYEXE% -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Python dependencies ready. Next: install Tesseract and Japanese OCR data.
echo   winget install --id UB-Mannheim.TesseractOCR -e
echo Select Japanese ^(jpn^) and English ^(eng^) language data.
echo Then double-click run.bat
pause
exit /b 0
:failed
echo.
echo Setup FAILED. Check Python version and internet connection.
pause
exit /b 1
