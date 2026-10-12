@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "%~dp0scripts\setup.ps1" (set "PS1=%~dp0scripts\setup.ps1") else (set "PS1=%~dp0setup.ps1")
powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" -OcrOnly
echo.
pause
