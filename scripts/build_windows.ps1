#Requires -Version 5.1
<#
.SYNOPSIS
  Build the Photo Enhancer Windows desktop app (PyInstaller onedir).

.DESCRIPTION
  1. Kills orphaned PyInstaller processes from prior failed builds
  2. Builds the React frontend (npm run build)
  3. Generates app icon if missing
  4. Installs desktop build dependencies (pywebview, pyinstaller)
  5. Runs PyInstaller with photo-enhancer.spec
     - When the project lives under OneDrive, builds to C:\Temp\PhotoEnhancerBuild
       then copies dist\PhotoEnhancer back to the project (avoids sync file locks)

  Output: dist\PhotoEnhancer\PhotoEnhancer.exe

  Optional: install Inno Setup and run iscc installer.iss for an installer.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-BuildError {
    param([string]$Message, [string]$Hint = "")
    Write-Host ""
    Write-Host "BUILD FAILED" -ForegroundColor Red
    Write-Host $Message -ForegroundColor Red
    if ($Hint) {
        Write-Host $Hint -ForegroundColor Yellow
    }
}

function Stop-OrphanedPyInstallerProcesses {
    Write-Host "`n==> Cleaning up orphaned PyInstaller processes..." -ForegroundColor Cyan
    $stopped = 0
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | ForEach-Object {
        $cmd = $_.CommandLine
        if ($null -eq $cmd) { return }
        if ($cmd -match "PyInstaller|photo-enhancer\.spec") {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            $stopped++
        }
    }
    if ($stopped -gt 0) {
        Write-Host "  Stopped $stopped orphaned process(es)."
        Start-Sleep -Seconds 2
    }
}

function Stop-PhotoEnhancerApp {
    $procs = Get-Process -Name "PhotoEnhancer" -ErrorAction SilentlyContinue
    if ($procs) {
        Write-Host "`n==> Stopping running PhotoEnhancer.exe..." -ForegroundColor Cyan
        $procs | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

try {
    $Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
    if (-not (Test-Path (Join-Path $Root "backend"))) {
        $Root = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    Set-Location $Root

    Write-Host "==> Photo Enhancer Windows build" -ForegroundColor Cyan
    Write-Host "Project root: $Root"

    Stop-PhotoEnhancerApp
    Stop-OrphanedPyInstallerProcesses

    $VenvPython = Join-Path $Root "backend\venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        throw @"
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
        if ($LASTEXITCODE -ne 0) { throw "npm install failed (exit $LASTEXITCODE)" }
    }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm run build failed (exit $LASTEXITCODE)" }
    Pop-Location

    $FrontendIndex = Join-Path $Root "frontend\dist\index.html"
    if (-not (Test-Path $FrontendIndex)) {
        throw "Frontend build did not produce frontend\dist\index.html"
    }

    $IconPath = Join-Path $Root "assets\icon.ico"
    if (-not (Test-Path $IconPath)) {
        Write-Host "`n==> Generating app icon..." -ForegroundColor Cyan
        & $VenvPython (Join-Path $Root "scripts\generate_icon.py")
        if ($LASTEXITCODE -ne 0) { throw "Icon generation failed (exit $LASTEXITCODE)" }
    }

    Write-Host "`n==> Installing desktop build dependencies..." -ForegroundColor Cyan
    & $VenvPython -m pip install -r (Join-Path $Root "backend\requirements-desktop.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install requirements-desktop.txt failed (exit $LASTEXITCODE)" }

    $UseTempBuild = $Root -match "OneDrive"
    $TempBuildRoot = "C:\Temp\PhotoEnhancerBuild"
    $SpecPath = Join-Path $Root "photo-enhancer.spec"

    if ($UseTempBuild) {
        Write-Host "`n==> OneDrive path detected - building to $TempBuildRoot" -ForegroundColor Yellow
        Write-Host "    (avoids OneDrive sync locks during PyInstaller)" -ForegroundColor DarkGray
        New-Item -ItemType Directory -Force -Path $TempBuildRoot | Out-Null
        $WorkPath = Join-Path $TempBuildRoot "build"
        $DistPath = Join-Path $TempBuildRoot "dist"
    } else {
        $WorkPath = Join-Path $Root "build"
        $DistPath = Join-Path $Root "dist"
    }

    Write-Host "`n==> Running PyInstaller onedir build - may take several minutes..." -ForegroundColor Cyan
    & $VenvPython -m PyInstaller $SpecPath --noconfirm --clean --workpath $WorkPath --distpath $DistPath
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller exited with code $LASTEXITCODE. Check output above for missing modules or locked files."
    }

    $TempOutput = Join-Path $DistPath "PhotoEnhancer"
    $TempExe = Join-Path $TempOutput "PhotoEnhancer.exe"
    if (-not (Test-Path $TempExe)) {
        throw "PyInstaller did not produce PhotoEnhancer.exe at:`n  $TempExe"
    }

    $FinalDist = Join-Path $Root "dist\PhotoEnhancer"
    $ExePath = Join-Path $FinalDist "PhotoEnhancer.exe"

    if ($UseTempBuild) {
        Write-Host "`n==> Copying build output to project dist..." -ForegroundColor Cyan
        $ProjectDistDir = Join-Path $Root "dist"
        New-Item -ItemType Directory -Force -Path $ProjectDistDir | Out-Null
        if (Test-Path $FinalDist) {
            Remove-Item -Recurse -Force $FinalDist
        }
        Copy-Item -Path $TempOutput -Destination $FinalDist -Recurse -Force
        if (-not (Test-Path $ExePath)) {
            throw "Copy to project dist failed - PhotoEnhancer.exe missing at:`n  $ExePath"
        }
    }

    Write-Host ""
    Write-Host "Build complete." -ForegroundColor Green
    Write-Host "  Run:  $ExePath"
    Write-Host "  Models download on first use to %APPDATA%\PhotoEnhancer\models\"
    Write-Host ""
    Write-Host "Optional installer (requires Inno Setup 6+):"
    Write-Host "  iscc installer.iss"
}
catch {
    $msg = $_.Exception.Message
    $hint = ""
    if ($msg -match "OneDrive|locked|access|denied|EBUSY") {
        $hint = "Tip: Close PhotoEnhancer.exe and any Explorer windows on dist\ before rebuilding."
    } elseif ($msg -match "venv|Virtual environment") {
        $hint = "Tip: Run scripts\setup_windows.ps1 or create backend\venv manually."
    } elseif ($msg -match "npm|frontend") {
        $hint = "Tip: Run 'cd frontend && npm install && npm run build' and fix any TypeScript errors."
    } elseif ($msg -match "PyInstaller") {
        $hint = "Tip: Kill stale python/PyInstaller processes, ensure enough disk space, and retry."
    }
    Write-BuildError -Message $msg -Hint $hint
    exit 1
}
