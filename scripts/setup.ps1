<#
.SYNOPSIS
    Quick setup script for Stoic Citadel on Windows

.DESCRIPTION
    Sets up Python virtual environment and installs all dependencies.

.EXAMPLE
    .\scripts\setup.ps1
#>

$ErrorActionPreference = "Stop"

function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }

# Banner
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           🏛️  STOIC CITADEL - SETUP                         ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Info "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    Write-Success "Found: $pythonVersion"
} catch {
    Write-Host "❌ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Info "Creating virtual environment..."
    python -m venv .venv
    Write-Success "Virtual environment created!"
} else {
    Write-Info "Virtual environment already exists"
}

# Activate venv
Write-Info "Activating virtual environment..."
& .venv\Scripts\Activate.ps1

# Upgrade pip
Write-Info "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Info "Installing dependencies..."
pip install -r requirements.txt --quiet
Write-Success "Core dependencies installed!"

# Install dev dependencies
Write-Info "Installing development dependencies..."
pip install -r requirements-dev.txt --quiet
Write-Success "Development dependencies installed!"

# Create necessary directories
$dirs = @(
    "user_data/data",
    "user_data/backtest_results",
    "user_data/hyperopt_results",
    "user_data/logs",
    "reports"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Success "Directories created!"

# Run smoke test
Write-Host ""
Write-Info "Running smoke test to verify installation..."
Write-Host ""
python scripts/smoke_test.py

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║           🎉 SETUP COMPLETE!                                ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Activate venv:  .venv\Scripts\Activate.ps1"
    Write-Host "  2. Run tests:      .\scripts\run_tests.ps1"
    Write-Host "  3. Run backtest:   .\scripts\backtest.ps1 -Action backtest"
    Write-Host "  4. Get help:       .\scripts\backtest.ps1 -Action help"
    Write-Host ""
} else {
    Write-Host "❌ Setup completed but smoke test failed!" -ForegroundColor Red
    exit 1
}
