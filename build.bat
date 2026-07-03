@echo off
setlocal
cd /d "%~dp0"
echo.
echo Photo Enhancer Windows build
echo ============================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\build_windows.ps1"
if errorlevel 1 (
  echo.
  echo Build failed. See error messages above.
  echo Common fixes:
  echo   - Close PhotoEnhancer.exe and File Explorer windows on dist\
  echo   - Ensure backend\venv exists: cd backend ^&^& python -m venv venv
  echo   - OneDrive projects build via C:\Temp\PhotoEnhancerBuild automatically
  pause
  exit /b 1
)
echo.
pause
