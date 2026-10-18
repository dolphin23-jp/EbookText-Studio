@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install pyinstaller
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed ^
  --name EbookTextStudio ^
  --collect-submodules pynput ^
  --collect-submodules ebooktext ^
  --collect-all pytesseract ^
  app.py
if errorlevel 1 goto failed
copy /y install_ocr.bat dist\EbookTextStudio\ >nul
copy /y scripts\setup.ps1 dist\EbookTextStudio\ >nul
copy /y README.md dist\EbookTextStudio\ >nul
powershell -NoProfile -Command "Compress-Archive -Force -Path 'dist\EbookTextStudio' -DestinationPath 'dist\EbookTextStudio_Windows_exe.zip'"
echo.
echo Done: dist\EbookTextStudio\EbookTextStudio.exe
echo ZIP : dist\EbookTextStudio_Windows_exe.zip
pause
exit /b 0
:failed
echo.
echo Build FAILED.
pause
exit /b 1
