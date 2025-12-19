# Overnight Setup Script
# =======================
# Полная автоматическая настройка:
# - Скачивает свежие данные (30 дней)
# - Обучает ML модели
# - Запускает бэктесты
# - Генерирует отчеты
#
# Запусти на ночь, утром получишь готовую систему!

param(
    [string[]]$Pairs = @("BTC/USDT", "ETH/USDT", "BNB/USDT"),
    [int]$Days = 30,
    [switch]$SkipData,
    [switch]$SkipTraining,
    [switch]$SkipBacktest,
    [switch]$Optimize  # Enable hyperparameter optimization (takes longer)
)

$ErrorActionPreference = "Continue"
$startTime = Get-Date

# Colors
function Write-Header {
    param([string]$Text)
    Write-Host "`n$('='*70)" -ForegroundColor Cyan
    Write-Host $Text -ForegroundColor Cyan
    Write-Host $('='*70)`n -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Text)
    Write-Host "`n🔹 $Text" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Text)
    Write-Host "✅ $Text" -ForegroundColor Green
}

function Write-Error {
    param([string]$Text)
    Write-Host "❌ $Text" -ForegroundColor Red
}

# Log file
$logFile = "overnight_setup_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logFile

Write-Header "🌙 OVERNIGHT SETUP STARTED"

Write-Host "Configuration:" -ForegroundColor White
Write-Host "  Pairs:         $($Pairs -join ', ')" -ForegroundColor White
Write-Host "  Days:          $Days" -ForegroundColor White
Write-Host "  Skip Data:     $SkipData" -ForegroundColor White
Write-Host "  Skip Training: $SkipTraining" -ForegroundColor White
Write-Host "  Skip Backtest: $SkipBacktest" -ForegroundColor White
Write-Host "  Optimize:      $Optimize" -ForegroundColor White

$results = @{
    data = $false
    training = $false
    backtest = $false
}

# ==============================================================================
# STEP 1: Download Fresh Data
# ==============================================================================

if (-not $SkipData) {
    Write-Header "📥 STEP 1: DOWNLOADING DATA"

    Write-Step "Checking Docker containers..."
    $containers = docker ps --format "{{.Names}}"
    if ($containers -notcontains "stoic_freqtrade") {
        Write-Error "Docker container 'stoic_freqtrade' not running!"
        Write-Host "Start containers: docker-compose up -d`n" -ForegroundColor Yellow
        exit 1
    }
    Write-Success "Docker containers running"

    Write-Step "Deleting old data..."
    docker exec stoic_freqtrade rm -rf /freqtrade/user_data/data/binance/*.json
    Write-Success "Old data deleted"

    Write-Step "Downloading $Days days of data for $($Pairs.Count) pairs..."
    Write-Host "  This may take 5-10 minutes...`n" -ForegroundColor White

    $pairsArg = $Pairs -join ' '

    docker exec stoic_freqtrade freqtrade download-data `
        --exchange binance `
        --timeframe 5m `
        --pairs $pairsArg `
        --days $Days `
        --erase

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Data downloaded successfully"
        $results.data = $true
    } else {
        Write-Error "Data download failed"
        exit 1
    }

    Write-Step "Syncing data to local filesystem..."
    $localDataDir = "user_data\data\binance"
    if (-not (Test-Path $localDataDir)) {
        New-Item -ItemType Directory -Path $localDataDir -Force | Out-Null
    }

    docker cp stoic_freqtrade:/freqtrade/user_data/data/binance/. $localDataDir

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Data synced to local"

        # Show files
        Write-Host "`n📦 Downloaded files:" -ForegroundColor Cyan
        Get-ChildItem $localDataDir -Filter "*.json" | ForEach-Object {
            $sizeMB = [math]::Round($_.Length / 1MB, 2)
            $content = Get-Content $_.FullName | ConvertFrom-Json
            $candles = $content.Count
            Write-Host "  ✅ $($_.Name) - $sizeMB MB - $candles candles" -ForegroundColor Green
        }
    } else {
        Write-Error "Data sync failed"
        exit 1
    }
} else {
    Write-Host "`n⏭️  Skipping data download`n" -ForegroundColor Yellow
    $results.data = $true
}

# ==============================================================================
# STEP 2: Train ML Models
# ==============================================================================

if (-not $SkipTraining) {
    Write-Header "🤖 STEP 2: TRAINING ML MODELS"

    Write-Step "Activating virtual environment..."

    # Check if venv exists
    if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
        Write-Error "Virtual environment not found!"
        Write-Host "Create it: python -m venv .venv`n" -ForegroundColor Yellow
        exit 1
    }

    # Activate venv (in the same session)
    & .\.venv\Scripts\Activate.ps1

    Write-Success "Virtual environment activated"

    Write-Step "Training models for $($Pairs.Count) pairs..."
    Write-Host "  This may take 15-60 minutes depending on data size...`n" -ForegroundColor White

    $pairsArg = $Pairs -join ' '
    $optimizeFlag = if ($Optimize) { "--optimize --trials 50" } else { "" }

    python scripts/train_models.py --pairs $pairsArg $optimizeFlag

    if ($LASTEXITCODE -eq 0) {
        Write-Success "Models trained successfully"
        $results.training = $true

        # Show trained models
        Write-Host "`n🤖 Trained models:" -ForegroundColor Cyan
        Get-ChildItem user_data\models -Filter "*.pkl" | Sort-Object LastWriteTime -Descending | Select-Object -First 5 | ForEach-Object {
            $sizeMB = [math]::Round($_.Length / 1MB, 2)
            Write-Host "  ✅ $($_.Name) - $sizeMB MB" -ForegroundColor Green
        }
    } else {
        Write-Error "Model training failed"
        $results.training = $false
    }
} else {
    Write-Host "`n⏭️  Skipping model training`n" -ForegroundColor Yellow
    $results.training = $true
}

# ==============================================================================
# STEP 3: Run Backtests
# ==============================================================================

if (-not $SkipBacktest) {
    Write-Header "📊 STEP 3: RUNNING BACKTESTS"

    Write-Step "Running backtest profiles..."
    Write-Host "  This may take 10-30 minutes...`n" -ForegroundColor White

    # Run full backtest for each pair
    foreach ($pair in $Pairs) {
        Write-Host "`n  🔹 Backtesting $pair..." -ForegroundColor Yellow

        python scripts/run_backtest.py --pair $pair --days $Days

        if ($LASTEXITCODE -eq 0) {
            Write-Success "$pair backtest completed"
        } else {
            Write-Error "$pair backtest failed"
        }
    }

    $results.backtest = $true

    Write-Success "All backtests completed"
    Write-Host "`n📊 View results at: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "   Login: stoic_admin / StoicGuard2024!ChangeMe`n" -ForegroundColor White

} else {
    Write-Host "`n⏭️  Skipping backtests`n" -ForegroundColor Yellow
    $results.backtest = $true
}

# ==============================================================================
# SUMMARY
# ==============================================================================

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Header "✅ OVERNIGHT SETUP COMPLETED"

Write-Host "Results:" -ForegroundColor White
Write-Host "  Data Download: $(if ($results.data) {'✅ Success'} else {'❌ Failed'})" -ForegroundColor $(if ($results.data) {'Green'} else {'Red'})
Write-Host "  ML Training:   $(if ($results.training) {'✅ Success'} else {'❌ Failed'})" -ForegroundColor $(if ($results.training) {'Green'} else {'Red'})
Write-Host "  Backtests:     $(if ($results.backtest) {'✅ Success'} else {'❌ Failed'})" -ForegroundColor $(if ($results.backtest) {'Green'} else {'Red'})

Write-Host "`nDuration: $($duration.Hours)h $($duration.Minutes)m $($duration.Seconds)s" -ForegroundColor White
Write-Host "Log file: $logFile`n" -ForegroundColor White

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Open FreqUI:     http://localhost:3000" -ForegroundColor White
Write-Host "  2. View backtest results in 'Backtesting' tab" -ForegroundColor White
Write-Host "  3. Inspect data:    python scripts/inspect_data.py --pair BTC/USDT" -ForegroundColor White
Write-Host "  4. Check models:    ls user_data/models/" -ForegroundColor White

if ($results.data -and $results.training -and $results.backtest) {
    Write-Host "`n🎉 YOUR TRADING SYSTEM IS READY!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Some steps failed. Check log: $logFile" -ForegroundColor Yellow
}

Write-Host "`n$('='*70)`n" -ForegroundColor Cyan

Stop-Transcript
