#Requires -Version 5.1
<#
.SYNOPSIS
  Build the Photo Enhancer Windows desktop app (PyInstaller onedir).

.DESCRIPTION
  1. Builds the React frontend (npm run build)
  2. Generates app icon if missing
  3. Installs desktop build dependencies (pywebview, pyinstaller)
  4. Runs PyInstaller with photo-enhancer.spec

  Output: dist\PhotoEnhancer\PhotoEnhancer.exe

  Optional: install Inno Setup and run iscc installer.iss for an installer.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not (Test-Path (Join-Path $Root "backend"))) {
    $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
}
Set-Location $Root

Write-Host "==> Photo Enhancer Windows build" -ForegroundColor Cyan
Write-Host "Project root: $Root"

$VenvPython = Join-Path $Root "backend\venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Error @"
Virtual environment not found at backend\venv.
Create it first:
  cd backend
  python -m venv venv
  venv\Scripts\pip install -r requirements.txt
"@
}

Write-Host "`n==> Building frontend..." -ForegroundColor Cyan
Push-Location (Join-Path $Root "frontend")
if (-not (Test-Path "node_modules")) {
    npm install
}
npm run build
if ($LASTEXITCODE -ne 0) { throw "npm run build failed" }
Pop-Location

$IconPath = Join-Path $Root "assets\icon.ico"
if (-not (Test-Path $IconPath)) {
    Write-Host "`n==> Generating app icon..." -ForegroundColor Cyan
    & $VenvPython (Join-Path $Root "scripts\generate_icon.py")
}

Write-Host "`n==> Installing desktop build dependencies..." -ForegroundColor Cyan
& $VenvPython -m pip install -r (Join-Path $Root "backend\requirements-desktop.txt")

Write-Host "`n==> Running PyInstaller (onedir, may take several minutes)..." -ForegroundColor Cyan
& $VenvPython -m PyInstaller (Join-Path $Root "photo-enhancer.spec") --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

$ExePath = Join-Path $Root "dist\PhotoEnhancer\PhotoEnhancer.exe"
Write-Host ""
Write-Host "Build complete." -ForegroundColor Green
Write-Host "  Run:  $ExePath"
Write-Host "  Models download on first use to %APPDATA%\PhotoEnhancer\models\"
Write-Host ""
Write-Host "Optional installer (requires Inno Setup 6+):"
Write-Host "  iscc installer.iss"
