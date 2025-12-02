#!/usr/bin/env pwsh
# Stoic Citadel - Data Download Script for Windows
# Загрузка исторических данных с биржи

param(
    [Parameter(Mandatory=$false)]
    [int]$Days = 90,
    
    [Parameter(Mandatory=$false)]
    [string]$Timeframe = "5m",
    
    [Parameter(Mandatory=$false)]
    [string]$Exchange = "binance",
    
    [Parameter(Mandatory=$false)]
    [string]$Pairs = "BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT",
    
    [Parameter(Mandatory=$false)]
    [switch]$TradingViewFormat,
    
    [Parameter(Mandatory=$false)]
    [switch]$WithBTC1d,  # Для продвинутых стратегий с режимным фильтром
    
    [Parameter(Mandatory=$false)]
    [int]$BTC1dDays = 365
)

# Цвета
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Step {
    param([string]$Message)
    Write-ColorOutput "`n=== $Message ===" "Cyan"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" "Red"
}

# Баннер
Write-Host @"

╔════════════════════════════════════════════════════════╗
║         STOIC CITADEL - DATA DOWNLOADER                ║
╚════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

# Параметры
Write-Step "Параметры загрузки"
Write-ColorOutput "Биржа:            $Exchange" "Cyan"
Write-ColorOutput "Таймфрейм:        $Timeframe" "Cyan"
Write-ColorOutput "Дней данных:      $Days" "Cyan"
Write-ColorOutput "Пары:             $Pairs" "Cyan"

# Оценка времени и размера
$pairsArray = $Pairs -split " "
$pairsCount = $pairsArray.Count
$estimatedSize = [math]::Round(($Days * $pairsCount * 0.5), 2)  # Примерно 0.5MB на пару в день
$estimatedTime = [math]::Round(($pairsCount * 3), 0)  # Примерно 3 секунды на пару

Write-ColorOutput "`n📊 Оценка:" "Yellow"
Write-ColorOutput "   Количество пар:    $pairsCount" "Gray"
Write-ColorOutput "   Примерный размер:  ~$estimatedSize MB" "Gray"
Write-ColorOutput "   Примерное время:   ~$estimatedTime секунд" "Gray"

# Формирование команды
$command = @(
    "docker-compose", "run", "--rm", "freqtrade", "download-data",
    "--config", "/freqtrade/user_data/config/config.json",
    "--exchange", $Exchange,
    "--timeframe", $Timeframe,
    "--days", $Days.ToString()
)

# Добавляем пары
$command += "--pairs"
$command += $pairsArray

# TradingView формат (опционально)
if ($TradingViewFormat) {
    $command += "--trading-mode", "spot"
    $command += "--data-format-ohlcv", "json"
    Write-ColorOutput "   Формат:            TradingView JSON" "Yellow"
}

# Основная загрузка
Write-Step "Загрузка данных $Timeframe"
Write-ColorOutput "Запуск команды..." "Yellow"
Write-Host ""

& $command[0] $command[1..($command.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Error "Ошибка при загрузке данных!"
    exit 1
}

Write-Success "Данные $Timeframe успешно загружены"

# Дополнительная загрузка BTC 1d для продвинутых стратегий
if ($WithBTC1d) {
    Write-Step "Загрузка BTC/USDT 1d для режимного фильтра"
    Write-ColorOutput "Загрузка $BTC1dDays дней дневных данных..." "Yellow"
    
    $btcCommand = @(
        "docker-compose", "run", "--rm", "freqtrade", "download-data",
        "--config", "/freqtrade/user_data/config/config.json",
        "--exchange", $Exchange,
        "--timeframe", "1d",
        "--days", $BTC1dDays.ToString(),
        "--pairs", "BTC/USDT"
    )
    
    & $btcCommand[0] $btcCommand[1..($btcCommand.Length-1)]
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "BTC/USDT 1d данные загружены"
    } else {
        Write-Error "Ошибка при загрузке BTC/USDT 1d данных"
    }
}

# Проверка загруженных данных
Write-Step "Проверка загруженных данных"

$dataDir = ".\user_data\data\$Exchange"
if (Test-Path $dataDir) {
    $files = Get-ChildItem -Path $dataDir -Filter "*.feather" -Recurse
    $totalSize = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
    
    Write-Success "Найдено файлов: $($files.Count)"
    Write-ColorOutput "Общий размер: $([math]::Round($totalSize, 2)) MB" "Cyan"
    
    Write-Host "`nСписок файлов:" -ForegroundColor Yellow
    foreach ($file in $files) {
        $sizeKB = [math]::Round($file.Length / 1KB, 2)
        Write-ColorOutput "  📄 $($file.Name) - $sizeKB KB" "Gray"
    }
} else {
    Write-Error "Директория данных не найдена: $dataDir"
}

# Итоговая информация
Write-Host @"

╔════════════════════════════════════════════════════════╗
║                  ЗАГРУЗКА ЗАВЕРШЕНА                    ║
╚════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host @"
Следующие шаги:

1. Запустить бэктест:
   .\scripts\windows\backtest.ps1 -Strategy "SimpleTestStrategy"

2. Загрузить дополнительные данные:
   .\scripts\windows\download-data.ps1 -Days 180 -Timeframe "1h"

3. Загрузить данные для продвинутых стратегий:
   .\scripts\windows\download-data.ps1 -WithBTC1d

4. Просмотреть данные в Jupyter:
   docker-compose up -d jupyter
   Открыть: http://localhost:8888 (token: stoic2024)

"@ -ForegroundColor Gray

Write-Success "✨ Данные готовы для использования!`n"
