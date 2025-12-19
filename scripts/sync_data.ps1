# Sync Data from Docker to Local
# ===============================
# This script copies data from Docker container to local filesystem

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Syncing data from Docker to local..." -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Create local directory if not exists
$localDataDir = "user_data\data\binance"
if (-not (Test-Path $localDataDir)) {
    Write-Host "Creating directory: $localDataDir" -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $localDataDir -Force | Out-Null
}

# Copy data from Docker container
Write-Host "Copying data from stoic_freqtrade container..." -ForegroundColor Yellow

docker cp stoic_freqtrade:/freqtrade/user_data/data/binance/. $localDataDir

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Data synced successfully!" -ForegroundColor Green

    # Show what was copied
    Write-Host "`nDownloaded files:" -ForegroundColor Cyan
    Get-ChildItem $localDataDir -Filter "*.json" | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  - $($_.Name) ($sizeMB MB)" -ForegroundColor White
    }

    Write-Host "`nYou can now use:" -ForegroundColor Green
    Write-Host "  python scripts/inspect_data.py --pair BTC/USDT" -ForegroundColor White
    Write-Host "  python scripts/run_backtest.py --profile quick" -ForegroundColor White

} else {
    Write-Host "`n❌ Failed to copy data from Docker" -ForegroundColor Red
    Write-Host "Make sure Docker container is running:" -ForegroundColor Yellow
    Write-Host "  docker ps" -ForegroundColor White
}

Write-Host "`n========================================`n" -ForegroundColor Cyan
