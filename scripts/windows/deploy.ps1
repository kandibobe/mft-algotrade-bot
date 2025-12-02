# Stoic Citadel - Full Deployment Script for Windows
# Автоматическое развертывание всей инфраструктуры

param(
    [switch]$SkipJupyter = $false,
    [switch]$SkipData = $false,
    [switch]$SkipBacktest = $false
)

$ErrorActionPreference = "Stop"

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stoic Citadel - Deployment Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Pull latest changes
Write-Host "[1/7] Pulling latest changes from GitHub..." -ForegroundColor Yellow
try {
    git pull origin simplify-architecture
    Write-Host "✅ Git pull successful" -ForegroundColor Green
} catch {
    Write-Host "❌ Git pull failed: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Stop existing containers
Write-Host "[2/7] Stopping existing containers..." -ForegroundColor Yellow
docker-compose down
Write-Host "✅ Containers stopped" -ForegroundColor Green
Write-Host ""

# Step 3: Build Jupyter (optional)
if (-not $SkipJupyter) {
    Write-Host "[3/7] Building Jupyter Lab container..." -ForegroundColor Yellow
    Write-Host "⏳ This may take 5-10 minutes (compiling TA-Lib from source)" -ForegroundColor Gray
    try {
        docker-compose build jupyter
        Write-Host "✅ Jupyter built successfully" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Jupyter build failed (non-critical): $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "[3/7] Skipping Jupyter build (use -SkipJupyter:$false to build)" -ForegroundColor Gray
}
Write-Host ""

# Step 4: Start core services
Write-Host "[4/7] Starting Freqtrade and FreqUI..." -ForegroundColor Yellow
try {
    docker-compose up -d freqtrade frequi
    Write-Host "✅ Core services started" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to start services: $_" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Wait for health check
Write-Host "[5/7] Waiting for services to become healthy..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
$status = docker-compose ps
Write-Host $status
Write-Host "✅ Health check complete" -ForegroundColor Green
Write-Host ""

# Step 6: Download data (optional)
if (-not $SkipData) {
    Write-Host "[6/7] Downloading 90 days of 5m data..." -ForegroundColor Yellow
    try {
        docker-compose run --rm freqtrade download-data `
            --config /freqtrade/user_data/config/config.json `
            --exchange binance `
            --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT `
            --timeframe 5m `
            --days 90
        Write-Host "✅ Data downloaded" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Data download failed (non-critical): $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "[6/7] Skipping data download (use -SkipData:$false to download)" -ForegroundColor Gray
}
Write-Host ""

# Step 7: Run test backtest (optional)
if (-not $SkipBacktest) {
    Write-Host "[7/7] Running test backtest..." -ForegroundColor Yellow
    try {
        docker-compose run --rm freqtrade backtesting `
            --config /freqtrade/user_data/config/config.json `
            --strategy SimpleTestStrategy `
            --timerange 20241101-
        Write-Host "✅ Backtest completed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Backtest failed (non-critical): $_" -ForegroundColor Yellow
    }
} else {
    Write-Host "[7/7] Skipping backtest (use -SkipBacktest:$false to run)" -ForegroundColor Gray
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  🎉 Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access Points:" -ForegroundColor White
Write-Host "  • FreqUI Dashboard: http://localhost:3000" -ForegroundColor Gray
Write-Host "    Login: stoic_admin / StoicGuard2024" -ForegroundColor Gray
Write-Host ""
Write-Host "  • Jupyter Lab: http://localhost:8888" -ForegroundColor Gray
Write-Host "    Token: stoic2024" -ForegroundColor Gray
Write-Host ""
Write-Host "  • API: http://localhost:8080/api/v1/ping" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor White
Write-Host "  1. Open FreqUI to monitor bot activity" -ForegroundColor Gray
Write-Host "  2. Run more backtests: .\scripts\windows\backtest.ps1" -ForegroundColor Gray
Write-Host "  3. Check logs: docker-compose logs -f freqtrade" -ForegroundColor Gray
Write-Host ""
