<#
.SYNOPSIS
    Stoic Citadel - Backtest Management Script for Windows

.DESCRIPTION
    PowerShell script to manage backtesting operations.
    Replaces Makefile for Windows users.

.EXAMPLE
    .\scripts\backtest.ps1 -Action download -Pairs "BTC/USDT,ETH/USDT" -TimeRange "20240101-20240301"
    .\scripts\backtest.ps1 -Action backtest -Strategy "StoicEnsembleStrategyV2"
    .\scripts\backtest.ps1 -Action hyperopt -Epochs 100
    .\scripts\backtest.ps1 -Action report
    .\scripts\backtest.ps1 -Action full
    .\scripts\backtest.ps1 -Action smoke
    .\scripts\backtest.ps1 -Action test
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("download", "backtest", "hyperopt", "report", "full", "smoke", "test", "clean", "setup", "help")]
    [string]$Action,
    
    [string]$Pairs = "BTC/USDT,ETH/USDT,SOL/USDT",
    [string]$TimeRange = "20240101-20240601",
    [string]$Strategy = "StoicEnsembleStrategyV2",
    [int]$Epochs = 100,
    [string]$Exchange = "binance"
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }

# Banner
function Show-Banner {
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║           🏛️  STOIC CITADEL - BACKTEST MANAGER              ║" -ForegroundColor Cyan
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# Check Docker is running
function Test-Docker {
    try {
        $null = docker info 2>&1
        return $true
    } catch {
        return $false
    }
}

# Setup virtual environment and install dependencies
function Invoke-Setup {
    Write-Info "Setting up Python environment..."
    
    # Check if venv exists
    if (-not (Test-Path ".venv")) {
        Write-Info "Creating virtual environment..."
        python -m venv .venv
    }
    
    # Activate and install
    Write-Info "Installing dependencies..."
    & .venv\Scripts\python.exe -m pip install --upgrade pip
    & .venv\Scripts\pip.exe install -r requirements.txt
    & .venv\Scripts\pip.exe install pytest pytest-cov pytest-mock
    
    Write-Success "Setup complete! Activate with: .venv\Scripts\Activate.ps1"
}

# Download historical data
function Invoke-Download {
    Write-Info "Downloading data for pairs: $Pairs"
    Write-Info "Time range: $TimeRange"
    Write-Info "Exchange: $Exchange"
    
    if (-not (Test-Docker)) {
        Write-Error "Docker is not running! Please start Docker Desktop."
        exit 1
    }
    
    # Convert pairs to space-separated for freqtrade
    $pairsList = $Pairs -replace ",", " "
    
    docker-compose -f docker-compose.backtest.yml run --rm `
        -e PAIRS="$pairsList" `
        -e TIMERANGE="$TimeRange" `
        -e EXCHANGE="$Exchange" `
        data-downloader
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Data download complete!"
    } else {
        Write-Error "Data download failed!"
        exit 1
    }
}

# Run backtest
function Invoke-Backtest {
    Write-Info "Running backtest with strategy: $Strategy"
    Write-Info "Time range: $TimeRange"
    
    if (-not (Test-Docker)) {
        Write-Error "Docker is not running! Please start Docker Desktop."
        exit 1
    }
    
    docker-compose -f docker-compose.backtest.yml run --rm `
        -e STRATEGY="$Strategy" `
        -e TIMERANGE="$TimeRange" `
        freqtrade-backtest freqtrade backtesting `
            --config /freqtrade/user_data/config/config_backtest.json `
            --strategy $Strategy `
            --timerange $TimeRange `
            --export trades `
            --export-filename /freqtrade/user_data/backtest_results/backtest-result.json
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Backtest complete! Results in user_data/backtest_results/"
    } else {
        Write-Error "Backtest failed!"
        exit 1
    }
}

# Run hyperopt
function Invoke-Hyperopt {
    Write-Info "Running hyperopt with $Epochs epochs"
    Write-Info "Strategy: $Strategy"
    
    if (-not (Test-Docker)) {
        Write-Error "Docker is not running! Please start Docker Desktop."
        exit 1
    }
    
    docker-compose -f docker-compose.backtest.yml --profile optimize run --rm `
        hyperopt freqtrade hyperopt `
            --config /freqtrade/user_data/config/config_backtest.json `
            --strategy $Strategy `
            --hyperopt-loss SharpeHyperOptLoss `
            --epochs $Epochs `
            --spaces buy sell roi stoploss `
            --timerange $TimeRange
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Hyperopt complete!"
    } else {
        Write-Error "Hyperopt failed!"
        exit 1
    }
}

# Generate HTML report
function Invoke-Report {
    Write-Info "Generating backtest report..."
    
    # Check if Python script exists
    if (Test-Path "scripts/generate_report.py") {
        python scripts/generate_report.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Report generated! Open reports/backtest_report.html"
            
            # Try to open in browser
            if (Test-Path "reports/backtest_report.html") {
                Start-Process "reports/backtest_report.html"
            }
        }
    } else {
        Write-Warning "Report generator not found. Showing raw results..."
        
        $resultFile = Get-ChildItem -Path "user_data/backtest_results" -Filter "*.json" | 
                      Sort-Object LastWriteTime -Descending | 
                      Select-Object -First 1
        
        if ($resultFile) {
            Write-Info "Latest result: $($resultFile.FullName)"
            Get-Content $resultFile.FullName | ConvertFrom-Json | Format-List
        }
    }
}

# Run smoke test
function Invoke-SmokeTest {
    Write-Info "Running smoke test..."
    
    python scripts/smoke_test.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Smoke test passed!"
    } else {
        Write-Error "Smoke test failed!"
        exit 1
    }
}

# Run pytest
function Invoke-Test {
    Write-Info "Running unit tests..."
    
    # Try different ways to run pytest
    $pytestPaths = @(
        ".venv\Scripts\pytest.exe",
        "pytest",
        "python -m pytest"
    )
    
    $found = $false
    foreach ($pytest in $pytestPaths) {
        try {
            if ($pytest -eq "python -m pytest") {
                python -m pytest tests/test_utils/ tests/test_data/ -v --tb=short
            } else {
                & $pytest tests/test_utils/ tests/test_data/ -v --tb=short
            }
            $found = $true
            break
        } catch {
            continue
        }
    }
    
    if (-not $found) {
        Write-Warning "pytest not found. Installing..."
        pip install pytest pytest-cov pytest-mock
        python -m pytest tests/test_utils/ tests/test_data/ -v --tb=short
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All tests passed!"
    } else {
        Write-Error "Some tests failed!"
        exit 1
    }
}

# Full workflow
function Invoke-Full {
    Write-Info "Running full backtest workflow..."
    
    Invoke-Download
    Invoke-Backtest
    Invoke-Report
    
    Write-Success "Full workflow complete!"
}

# Clean up
function Invoke-Clean {
    Write-Info "Cleaning up..."
    
    # Remove backtest results
    if (Test-Path "user_data/backtest_results") {
        Remove-Item -Path "user_data/backtest_results/*.json" -Force -ErrorAction SilentlyContinue
        Write-Success "Cleaned backtest results"
    }
    
    # Remove hyperopt results
    if (Test-Path "user_data/hyperopt_results") {
        Remove-Item -Path "user_data/hyperopt_results/*" -Force -ErrorAction SilentlyContinue
        Write-Success "Cleaned hyperopt results"
    }
    
    # Remove reports
    if (Test-Path "reports") {
        Remove-Item -Path "reports/*.html" -Force -ErrorAction SilentlyContinue
        Write-Success "Cleaned reports"
    }
    
    # Remove pycache
    Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
    Write-Success "Cleaned __pycache__"
    
    Write-Success "Cleanup complete!"
}

# Show help
function Show-Help {
    Write-Host @"

🏛️ STOIC CITADEL - Backtest Manager
====================================

USAGE:
    .\scripts\backtest.ps1 -Action <action> [options]

ACTIONS:
    setup       Install dependencies and setup environment
    download    Download historical data
    backtest    Run backtest with strategy
    hyperopt    Run hyperparameter optimization
    report      Generate HTML report
    full        Run complete workflow (download → backtest → report)
    smoke       Run smoke test
    test        Run unit tests with pytest
    clean       Clean up generated files
    help        Show this help message

OPTIONS:
    -Pairs      Comma-separated pairs (default: BTC/USDT,ETH/USDT,SOL/USDT)
    -TimeRange  Time range YYYYMMDD-YYYYMMDD (default: 20240101-20240601)
    -Strategy   Strategy name (default: StoicEnsembleStrategyV2)
    -Epochs     Hyperopt epochs (default: 100)
    -Exchange   Exchange name (default: binance)

EXAMPLES:
    # Setup environment
    .\scripts\backtest.ps1 -Action setup

    # Download data
    .\scripts\backtest.ps1 -Action download -Pairs "BTC/USDT,ETH/USDT" -TimeRange "20240101-20240301"

    # Run backtest
    .\scripts\backtest.ps1 -Action backtest -Strategy "StoicEnsembleStrategyV2"

    # Run hyperopt
    .\scripts\backtest.ps1 -Action hyperopt -Epochs 200

    # Full workflow
    .\scripts\backtest.ps1 -Action full

    # Run tests
    .\scripts\backtest.ps1 -Action test

"@
}

# Main execution
Show-Banner

switch ($Action) {
    "setup"     { Invoke-Setup }
    "download"  { Invoke-Download }
    "backtest"  { Invoke-Backtest }
    "hyperopt"  { Invoke-Hyperopt }
    "report"    { Invoke-Report }
    "full"      { Invoke-Full }
    "smoke"     { Invoke-SmokeTest }
    "test"      { Invoke-Test }
    "clean"     { Invoke-Clean }
    "help"      { Show-Help }
}
