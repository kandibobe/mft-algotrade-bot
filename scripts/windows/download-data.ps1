# Stoic Citadel - Data Download Script for Windows
# Загрузка исторических данных с Binance

param(
    [int]$Days = 90,
    [string]$Timeframe = "5m",
    [string[]]$Pairs = @("BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"),
    [string]$Exchange = "binance",
    [string]$Config = "/freqtrade/user_data/config/config.json"
)

$ErrorActionPreference = "Stop"

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stoic Citadel - Data Download" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Exchange:    $Exchange" -ForegroundColor White
Write-Host "Timeframe:   $Timeframe" -ForegroundColor White
Write-Host "Days:        $Days" -ForegroundColor White
Write-Host "Pairs:       $($Pairs -join ', ')" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$pairsString = $Pairs -join " "

Write-Host "⏳ Downloading data..." -ForegroundColor Yellow
Write-Host "   This may take a few minutes depending on the number of pairs and days" -ForegroundColor Gray
Write-Host ""

try {
    docker-compose run --rm freqtrade download-data `
        --config $Config `
        --exchange $Exchange `
        --pairs $pairsString `
        --timeframe $Timeframe `
        --days $Days
    
    Write-Host ""
    Write-Host "✅ Data downloaded successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Data location: user_data/data/$Exchange/" -ForegroundColor Gray
    Write-Host "File format: *.feather (Pandas-compatible)" -ForegroundColor Gray
} catch {
    Write-Host ""
    Write-Host "❌ Data download failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  • Run backtest: .\scripts\windows\backtest.ps1" -ForegroundColor Gray
Write-Host "  • Verify data: docker-compose exec freqtrade ls -lh /freqtrade/user_data/data/$Exchange/" -ForegroundColor Gray
Write-Host ""
