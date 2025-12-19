# Download Fresh Data - Windows PowerShell Script
# ================================================
# Удаляет старые данные и скачивает свежие 30 дней

param(
    [string]$Pairs = "BTC/USDT ETH/USDT",
    [int]$Days = 30,
    [string]$Timeframe = "5m"
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "🔄 DOWNLOADING FRESH DATA" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Pairs:     $Pairs" -ForegroundColor White
Write-Host "  Timeframe: $Timeframe" -ForegroundColor White
Write-Host "  Days:      $Days`n" -ForegroundColor White

# Step 1: Delete old data in Docker container
Write-Host "Step 1: Cleaning old data in Docker..." -ForegroundColor Yellow

docker exec stoic_freqtrade rm -rf /freqtrade/user_data/data/binance/*.json

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Old data deleted`n" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Could not delete old data (might not exist)`n" -ForegroundColor Yellow
}

# Step 2: Download fresh data
Write-Host "Step 2: Downloading $Days days of data..." -ForegroundColor Yellow
Write-Host "  This may take 2-5 minutes...`n" -ForegroundColor White

$pairsList = $Pairs -split ' '
$pairsArg = $pairsList -join ' '

# Download data with erase flag
docker exec stoic_freqtrade freqtrade download-data `
    --exchange binance `
    --timeframe $Timeframe `
    --pairs $pairsArg `
    --days $Days `
    --erase

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n  ✅ Data downloaded successfully!`n" -ForegroundColor Green
} else {
    Write-Host "`n  ❌ Download failed! Check errors above.`n" -ForegroundColor Red
    exit 1
}

# Step 3: Sync to local filesystem
Write-Host "Step 3: Syncing data to local filesystem..." -ForegroundColor Yellow

$localDataDir = "user_data\data\binance"
if (-not (Test-Path $localDataDir)) {
    New-Item -ItemType Directory -Path $localDataDir -Force | Out-Null
}

docker cp stoic_freqtrade:/freqtrade/user_data/data/binance/. $localDataDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✅ Data synced to local!`n" -ForegroundColor Green

    # Show downloaded files
    Write-Host "📦 Downloaded files:" -ForegroundColor Cyan
    Get-ChildItem $localDataDir -Filter "*.json" | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)

        # Count candles
        $content = Get-Content $_.FullName | ConvertFrom-Json
        $candles = $content.Count

        Write-Host "  ✅ $($_.Name)" -ForegroundColor Green
        Write-Host "     Size: $sizeMB MB | Candles: $candles" -ForegroundColor White
    }
} else {
    Write-Host "  ❌ Sync failed!`n" -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ SUCCESS! Data is ready for backtesting" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Inspect data:  python scripts/inspect_data.py --pair BTC/USDT" -ForegroundColor White
Write-Host "  2. Run backtest:  python scripts/run_backtest.py --profile full" -ForegroundColor White
Write-Host "`n"
