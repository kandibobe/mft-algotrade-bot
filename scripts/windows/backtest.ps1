# Stoic Citadel - Backtesting Script for Windows
# Удобный запуск бэктестов с параметрами

param(
    [string]$Strategy = "SimpleTestStrategy",
    [string]$Timerange = "20241001-",
    [string]$Config = "/freqtrade/user_data/config/config.json",
    [switch]$EnablePositionStacking = $false,
    [int]$MaxOpenTrades = $null
)

$ErrorActionPreference = "Stop"

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stoic Citadel - Backtesting" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Strategy:    $Strategy" -ForegroundColor White
Write-Host "Timerange:   $Timerange" -ForegroundColor White
Write-Host "Config:      $Config" -ForegroundColor White
if ($MaxOpenTrades) {
    Write-Host "Max Trades:  $MaxOpenTrades" -ForegroundColor White
}
Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Build command
$command = @(
    "backtesting",
    "--config", $Config,
    "--strategy", $Strategy,
    "--timerange", $Timerange
)

if ($EnablePositionStacking) {
    $command += "--enable-position-stacking"
}

if ($MaxOpenTrades) {
    $command += "--max-open-trades", $MaxOpenTrades
}

Write-Host "⏳ Starting backtest..." -ForegroundColor Yellow
Write-Host ""

try {
    docker-compose run --rm freqtrade $command
    Write-Host ""
    Write-Host "✅ Backtest completed successfully!" -ForegroundColor Green
} catch {
    Write-Host ""
    Write-Host "❌ Backtest failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  • View detailed results in user_data/backtest_results/" -ForegroundColor Gray
Write-Host "  • Analyze with Jupyter: http://localhost:8888" -ForegroundColor Gray
Write-Host "  • Run HyperOpt for optimization" -ForegroundColor Gray
Write-Host ""
