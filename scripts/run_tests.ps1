<#
.SYNOPSIS
    Run tests for Stoic Citadel

.DESCRIPTION
    PowerShell script to run pytest tests with various options.

.EXAMPLE
    .\scripts\run_tests.ps1                    # Run all tests
    .\scripts\run_tests.ps1 -Unit              # Run only unit tests
    .\scripts\run_tests.ps1 -Integration       # Run integration tests
    .\scripts\run_tests.ps1 -Coverage          # Run with coverage
    .\scripts\run_tests.ps1 -Smoke             # Run smoke test only
#>

param(
    [switch]$Unit,
    [switch]$Integration,
    [switch]$Strategy,
    [switch]$Coverage,
    [switch]$Smoke,
    [switch]$Verbose,
    [switch]$Install
)

$ErrorActionPreference = "Stop"

function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warn { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }

# Banner
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           🧪 STOIC CITADEL - TEST RUNNER                    ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Install pytest if needed
function Install-Pytest {
    Write-Info "Installing test dependencies..."
    pip install pytest pytest-cov pytest-mock --quiet
    Write-Success "Test dependencies installed!"
}

# Check if pytest is available
function Test-Pytest {
    try {
        $null = python -m pytest --version 2>&1
        return $true
    } catch {
        return $false
    }
}

# Run smoke test
if ($Smoke) {
    Write-Info "Running smoke test..."
    python scripts/smoke_test.py
    exit $LASTEXITCODE
}

# Install if requested or pytest not found
if ($Install -or -not (Test-Pytest)) {
    Install-Pytest
}

# Build pytest command
$pytestArgs = @()

# Test paths
if ($Unit) {
    $pytestArgs += "tests/test_utils/"
    $pytestArgs += "tests/test_data/"
    $pytestArgs += "-m"
    $pytestArgs += "not integration and not slow"
    Write-Info "Running unit tests..."
} elseif ($Integration) {
    $pytestArgs += "tests/test_integration/"
    $pytestArgs += "-m"
    $pytestArgs += "integration"
    Write-Info "Running integration tests..."
} elseif ($Strategy) {
    $pytestArgs += "tests/test_strategies/"
    Write-Info "Running strategy tests..."
} else {
    $pytestArgs += "tests/"
    Write-Info "Running all tests..."
}

# Verbosity
if ($Verbose) {
    $pytestArgs += "-v"
    $pytestArgs += "--tb=long"
} else {
    $pytestArgs += "-v"
    $pytestArgs += "--tb=short"
}

# Coverage
if ($Coverage) {
    $pytestArgs += "--cov=src"
    $pytestArgs += "--cov-report=term-missing"
    $pytestArgs += "--cov-report=html:reports/coverage"
    Write-Info "Coverage enabled. Report will be in reports/coverage/"
}

# Run pytest
Write-Info "Command: python -m pytest $($pytestArgs -join ' ')"
Write-Host ""

python -m pytest @pytestArgs

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Success "All tests passed!"
    if ($Coverage) {
        Write-Info "Open reports/coverage/index.html to view coverage report"
    }
} else {
    Write-Host "❌ Some tests failed!" -ForegroundColor Red
}

exit $exitCode
