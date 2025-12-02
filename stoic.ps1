# ==============================================================================
# STOIC CITADEL - PowerShell Management Script (UPDATED)
# ==============================================================================
# Unified command interface for Windows users
# Based on repository: https://github.com/kandibobe/hft-algotrade-bot
# ==============================================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Strategy = "StoicStrategyV1",
    
    [Parameter(Position=2)]
    [string]$Service = "freqtrade"
)

$ErrorActionPreference = "Stop"
$PROJECT_DIR = "C:\hft-algotrade-bot"

# Color output helper
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Show-Header {
    Write-Host ""
    Write-ColorOutput Cyan "╔════════════════════════════════════════════════════════════╗"
    Write-ColorOutput Cyan "║            STOIC CITADEL - TRADING BOT                     ║"
    Write-ColorOutput Cyan "╚════════════════════════════════════════════════════════════╝"
    Write-Host ""
}

function Show-Help {
    Show-Header
    Write-ColorOutput Green "📋 ДОСТУПНЫЕ КОМАНДЫ:"
    Write-Host ""
    Write-ColorOutput Yellow "Управление:"
    Write-Host "  help              - Показать эту справку"
    Write-Host "  setup             - Первоначальная настройка"
    Write-Host "  start             - Запустить все сервисы"
    Write-Host "  stop              - Остановить все сервисы"
    Write-Host "  restart           - Перезапустить сервисы"
    Write-Host "  status            - Статус всех сервисов"
    Write-Host "  logs              - Показать логи (по умолчанию freqtrade)"
    Write-Host "  build             - Пересобрать Docker контейнеры"
    Write-Host ""
    Write-ColorOutput Yellow "Трейдинг:"
    Write-Host "  trade-dry         - Запустить paper trading (dry-run)"
    Write-Host "  trade-live        - Запустить LIVE trading (ОСТОРОЖНО!)"
    Write-Host "  backtest          - Запустить бэктест стратегии"
    Write-Host "  hyperopt          - Оптимизация параметров"
    Write-Host "  list-strategies   - Список всех стратегий"
    Write-Host "  list-pairs        - Список торговых пар"
    Write-Host ""
    Write-ColorOutput Yellow "Исследования:"
    Write-Host "  research          - Запустить Jupyter Lab"
    Write-Host "  download-data     - Скачать исторические данные"
    Write-Host "  verify-data       - Проверить качество данных"
    Write-Host ""
    Write-ColorOutput Yellow "Мониторинг:"
    Write-Host "  dashboard         - Открыть FreqUI dashboard"
    Write-Host "  monitoring        - Запустить Grafana мониторинг"
    Write-Host "  monitoring-stop   - Остановить мониторинг"
    Write-Host ""
    Write-ColorOutput Yellow "Обслуживание:"
    Write-Host "  clean             - Очистить контейнеры (данные остаются)"
    Write-Host "  clean-all         - Очистить ВСЁ включая данные"
    Write-Host "  db-backup         - Сделать backup базы данных"
    Write-Host "  validate-config   - Проверить конфигурацию"
    Write-Host ""
    Write-ColorOutput Green "📊 ПРИМЕРЫ:"
    Write-Host ""
    Write-Host "  .\stoic.ps1 setup"
    Write-Host "  .\stoic.ps1 trade-dry"
    Write-Host "  .\stoic.ps1 backtest StoicCitadelV2"
    Write-Host "  .\stoic.ps1 logs jupyter"
    Write-Host ""
}

function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-ColorOutput Yellow "⚠️  .env файл не найден. Создаю из шаблона..."
        Copy-Item ".env.example" ".env"
        Write-ColorOutput Green "✅ Создан .env файл"
        Write-ColorOutput Yellow "⚠️  ВАЖНО: Настройте .env файл перед продолжением!"
        Write-ColorOutput Yellow "   Откройте .env и заполните:"
        Write-Host "   - BINANCE_API_KEY"
        Write-Host "   - BINANCE_API_SECRET"
        Write-Host "   - FREQTRADE_API_PASSWORD"
        Write-Host ""
        $response = Read-Host "Настроили .env? (yes/no)"
        if ($response -ne "yes") {
            Write-ColorOutput Red "❌ Сначала настройте .env файл"
            exit 1
        }
    }
}

function Invoke-Setup {
    Show-Header
    Write-ColorOutput Cyan "🚀 Запуск мастера настройки Stoic Citadel..."
    
    Set-Location $PROJECT_DIR
    
    # Проверка Docker
    Write-ColorOutput Cyan "📋 Проверка Docker..."
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
        Write-ColorOutput Green "✅ Docker установлен"
    } catch {
        Write-ColorOutput Red "❌ Docker не найден. Установите Docker Desktop"
        exit 1
    }
    
    # Создание .env если нет
    Test-EnvFile
    
    # Создание директорий
    Write-ColorOutput Cyan "📁 Создание необходимых директорий..."
    $dirs = @(
        "user_data/data/binance",
        "user_data/logs", 
        "user_data/backtest_results",
        "user_data/hyperopt_results",
        "user_data/notebooks",
        "backups",
        "reports"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Write-ColorOutput Green "✅ Директории созданы"
    
    # Проверка Python скриптов
    if (Test-Path "scripts/setup_wizard.py") {
        Write-ColorOutput Cyan "🔧 Запуск Python мастера настройки..."
        try {
            python scripts/setup_wizard.py
        } catch {
            Write-ColorOutput Yellow "⚠️  Python мастер не удалось запустить (это нормально)"
        }
    }
    
    Write-ColorOutput Green "✅ Настройка завершена!"
    Write-Host ""
    Write-ColorOutput Cyan "📊 Следующие шаги:"
    Write-Host "  1. .\stoic.ps1 download-data   # Скачать данные"
    Write-Host "  2. .\stoic.ps1 trade-dry       # Запустить бота"
    Write-Host "  3. .\stoic.ps1 dashboard       # Открыть dashboard"
    Write-Host ""
}

function Invoke-Start {
    Show-Header
    Write-ColorOutput Cyan "🚀 Запуск Stoic Citadel сервисов..."
    
    Set-Location $PROJECT_DIR
    Test-EnvFile
    
    docker-compose up -d
    
    Start-Sleep -Seconds 3
    
    Write-ColorOutput Green "✅ Все сервисы запущены!"
    Write-Host ""
    Write-ColorOutput Cyan "📊 Точки доступа:"
    Write-Host "  FreqUI Dashboard:  http://localhost:3000"
    Write-Host "  Jupyter Lab:       http://localhost:8888 (token: stoic2024)"
    Write-Host "  Portainer:         http://localhost:9000"
    Write-Host ""
}

function Invoke-Stop {
    Write-ColorOutput Yellow "⏹️  Остановка всех сервисов..."
    Set-Location $PROJECT_DIR
    docker-compose down
    Write-ColorOutput Green "✅ Все сервисы остановлены"
}

function Invoke-Restart {
    Write-ColorOutput Cyan "🔄 Перезапуск сервисов..."
    Invoke-Stop
    Start-Sleep -Seconds 2
    Invoke-Start
}

function Invoke-Status {
    Show-Header
    Write-ColorOutput Cyan "📊 Статус сервисов:"
    Write-Host ""
    Set-Location $PROJECT_DIR
    docker-compose ps
}

function Invoke-Logs {
    Write-ColorOutput Cyan "📋 Логи для $Service (Ctrl+C для выхода):"
    Write-Host ""
    Set-Location $PROJECT_DIR
    docker-compose logs -f --tail=100 $Service
}

function Invoke-Build {
    Write-ColorOutput Cyan "🔨 Пересборка Docker контейнеров..."
    Set-Location $PROJECT_DIR
    docker-compose build --no-cache
    Write-ColorOutput Green "✅ Сборка завершена"
}

function Invoke-TradeDry {
    Show-Header
    Write-ColorOutput Cyan "📈 Запуск trading бота в DRY-RUN режиме..."
    
    Set-Location $PROJECT_DIR
    Test-EnvFile
    
    # Проверка что DRY_RUN=true в .env
    $envContent = Get-Content ".env" -Raw
    if ($envContent -notmatch "DRY_RUN\s*=\s*true") {
        Write-ColorOutput Yellow "⚠️  ВНИМАНИЕ: Убедитесь что DRY_RUN=true в .env файле!"
        $response = Read-Host "Продолжить? (yes/no)"
        if ($response -ne "yes") {
            Write-ColorOutput Yellow "⏹️  Отменено"
            return
        }
    }
    
    docker-compose up -d freqtrade frequi
    Start-Sleep -Seconds 3
    
    Write-ColorOutput Green "✅ Trading бот запущен (dry-run режим)"
    Write-Host ""
    Write-ColorOutput Cyan "📊 Мониторинг:"
    Write-Host "  Dashboard: http://localhost:3000"
    Write-Host "  Логи:      .\stoic.ps1 logs freqtrade"
    Write-Host ""
}

function Invoke-TradeLive {
    Show-Header
    Write-ColorOutput Red "╔════════════════════════════════════════════════════════════╗"
    Write-ColorOutput Red "║                    LIVE TRADING MODE                       ║"
    Write-ColorOutput Red "║                                                            ║"
    Write-ColorOutput Red "║  ⚠️  WARNING: THIS WILL USE REAL MONEY! ⚠️                  ║"
    Write-ColorOutput Red "║                                                            ║"
    Write-ColorOutput Red "║  Checklist:                                                ║"
    Write-ColorOutput Red "║  [ ] Протестировал в dry-run минимум 2 недели              ║"
    Write-ColorOutput Red "║  [ ] API ключи настроены с торговыми правами               ║"
    Write-ColorOutput Red "║  [ ] Лимиты рисков установлены в config                    ║"
    Write-ColorOutput Red "║  [ ] Telegram уведомления работают                         ║"
    Write-ColorOutput Red "║  [ ] Мониторинг настроен                                   ║"
    Write-ColorOutput Red "╚════════════════════════════════════════════════════════════╝"
    Write-Host ""
    
    $confirm = Read-Host "Введите 'Я ПОНИМАЮ РИСКИ' для продолжения"
    if ($confirm -ne "Я ПОНИМАЮ РИСКИ") {
        Write-ColorOutput Yellow "⚠️  Live trading отменён. Оставайся в безопасности!"
        return
    }
    
    Write-ColorOutput Red "⚠️  ВАЖНО: Установите dry_run: false в config_production.json"
    $configured = Read-Host "Конфиг настроен? (yes/no)"
    if ($configured -ne "yes") {
        Write-ColorOutput Yellow "⚠️  Сначала настройте конфиг"
        return
    }
    
    Set-Location $PROJECT_DIR
    Test-EnvFile
    
    docker-compose up -d freqtrade frequi
    
    Write-ColorOutput Green "✅ Live trading запущен!"
    Write-ColorOutput Red "⚠️  Внимательно мониторьте! Проверяйте логи регулярно!"
}

function Invoke-Backtest {
    Write-ColorOutput Cyan "🧪 Запуск бэктеста для стратегии: $Strategy"
    Set-Location $PROJECT_DIR
    Test-EnvFile
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange 20240101- `
        --enable-protections
    
    Write-ColorOutput Green "✅ Бэктест завершён!"
}

function Invoke-Hyperopt {
    Write-ColorOutput Cyan "🔍 Запуск оптимизации параметров для $Strategy"
    Write-ColorOutput Yellow "⏱️  Это займёт 2-4 часа..."
    
    Set-Location $PROJECT_DIR
    Test-EnvFile
    
    docker-compose run --rm freqtrade hyperopt `
        --strategy $Strategy `
        --hyperopt-loss SharpeHyperOptLoss `
        --epochs 500 `
        --spaces buy sell
    
    Write-ColorOutput Green "✅ Оптимизация завершена!"
}

function Invoke-ListStrategies {
    Write-ColorOutput Cyan "📋 Доступные стратегии:"
    Set-Location $PROJECT_DIR
    docker-compose run --rm freqtrade list-strategies
}

function Invoke-ListPairs {
    Write-ColorOutput Cyan "📋 Настроенные торговые пары:"
    Set-Location $PROJECT_DIR
    docker-compose run --rm freqtrade list-pairs
}

function Invoke-Research {
    Show-Header
    Write-ColorOutput Cyan "🔬 Запуск Jupyter Lab..."
    
    Set-Location $PROJECT_DIR
    docker-compose up -d jupyter
    Start-Sleep -Seconds 3
    
    Write-ColorOutput Green "✅ Jupyter Lab запущен!"
    Write-Host ""
    Write-ColorOutput Cyan "🌐 Доступ: http://localhost:8888"
    Write-ColorOutput Cyan "🔑 Token:  stoic2024"
    Write-Host ""
    
    Start-Process "http://localhost:8888"
}

function Invoke-DownloadData {
    Write-ColorOutput Cyan "📥 Скачивание исторических данных..."
    Set-Location $PROJECT_DIR
    
    if (Test-Path "scripts/download_data.sh") {
        # Используем WSL если доступен
        try {
            wsl bash scripts/download_data.sh 90 5m
        } catch {
            Write-ColorOutput Yellow "⚠️  WSL не найден, используем прямой метод..."
            # Альтернативный метод через Docker
            docker-compose run --rm freqtrade download-data `
                --exchange binance `
                --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT `
                --timeframes 5m 15m 1h `
                --days 90
        }
    } else {
        docker-compose run --rm freqtrade download-data `
            --exchange binance `
            --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT ADA/USDT `
            --timeframes 5m 15m 1h `
            --days 90
    }
    
    Write-ColorOutput Green "✅ Данные скачаны!"
}

function Invoke-VerifyData {
    Write-ColorOutput Cyan "🔍 Проверка качества данных..."
    Set-Location $PROJECT_DIR
    
    docker-compose run --rm jupyter python /home/jovyan/scripts/verify_data.py
    
    Write-ColorOutput Green "✅ Проверка завершена!"
}

function Invoke-Dashboard {
    Write-ColorOutput Cyan "📊 Открытие FreqUI Dashboard..."
    Start-Process "http://localhost:3000"
    Write-ColorOutput Green "✅ Dashboard открыт в браузере"
}

function Invoke-Monitoring {
    Show-Header
    Write-ColorOutput Cyan "📈 Запуск мониторинга (Prometheus + Grafana)..."
    
    Set-Location $PROJECT_DIR
    docker-compose -f docker-compose.monitoring.yml up -d
    Start-Sleep -Seconds 5
    
    Write-ColorOutput Green "✅ Мониторинг запущен!"
    Write-Host ""
    Write-ColorOutput Cyan "📊 Точки доступа:"
    Write-Host "  Grafana:    http://localhost:3001 (admin/admin)"
    Write-Host "  Prometheus: http://localhost:9090"
    Write-Host ""
    
    Start-Process "http://localhost:3001"
}

function Invoke-MonitoringStop {
    Write-ColorOutput Yellow "⏹️  Остановка мониторинга..."
    Set-Location $PROJECT_DIR
    docker-compose -f docker-compose.monitoring.yml down
    Write-ColorOutput Green "✅ Мониторинг остановлен"
}

function Invoke-Clean {
    Write-ColorOutput Yellow "⚠️  Это удалит все контейнеры и сети..."
    $confirm = Read-Host "Вы уверены? (yes/no)"
    
    if ($confirm -eq "yes") {
        Write-ColorOutput Cyan "🧹 Очистка..."
        Set-Location $PROJECT_DIR
        docker-compose down
        Write-ColorOutput Green "✅ Очистка завершена"
    } else {
        Write-ColorOutput Yellow "⏹️  Отменено"
    }
}

function Invoke-CleanAll {
    Write-ColorOutput Red "⚠️  ЭТО УДАЛИТ ВСЁ ВКЛЮЧАЯ ДАННЫЕ И ИСТОРИЮ СДЕЛОК!"
    $confirm = Read-Host "Введите 'DELETE EVERYTHING' для подтверждения"
    
    if ($confirm -eq "DELETE EVERYTHING") {
        Write-ColorOutput Cyan "🧹 Удаление всех контейнеров, томов и данных..."
        Set-Location $PROJECT_DIR
        
        docker-compose down -v
        docker-compose -f docker-compose.test.yml down -v 2>$null
        docker-compose -f docker-compose.monitoring.yml down -v 2>$null
        
        # Очистка локальных данных
        Remove-Item -Path "user_data/data/*" -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "user_data/logs/*" -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-ColorOutput Green "✅ Всё очищено!"
    } else {
        Write-ColorOutput Yellow "⏹️  Отменено"
    }
}

function Invoke-DbBackup {
    Write-ColorOutput Cyan "💾 Создание backup базы данных..."
    Set-Location $PROJECT_DIR
    
    $backupDir = "backups"
    if (-not (Test-Path $backupDir)) {
        New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    }
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupFile = "$backupDir\tradesv3_$timestamp.sqlite"
    
    if (Test-Path "user_data\tradesv3.sqlite") {
        Copy-Item "user_data\tradesv3.sqlite" $backupFile
        Write-ColorOutput Green "✅ Backup сохранён: $backupFile"
    } else {
        Write-ColorOutput Yellow "⚠️  База данных не найдена"
    }
}

function Invoke-ValidateConfig {
    Write-ColorOutput Cyan "🔍 Проверка конфигурации..."
    Set-Location $PROJECT_DIR
    
    if (Test-Path "scripts/validate_config.py") {
        try {
            python scripts/validate_config.py
            Write-ColorOutput Green "✅ Конфигурация валидна"
        } catch {
            Write-ColorOutput Red "❌ Ошибка валидации конфигурации"
        }
    } else {
        Write-ColorOutput Yellow "⚠️  Скрипт валидации не найден"
    }
}

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

Set-Location $PROJECT_DIR

switch ($Command.ToLower()) {
    # Управление
    "help"              { Show-Help }
    "setup"             { Invoke-Setup }
    "start"             { Invoke-Start }
    "stop"              { Invoke-Stop }
    "restart"           { Invoke-Restart }
    "status"            { Invoke-Status }
    "logs"              { Invoke-Logs }
    "build"             { Invoke-Build }
    
    # Трейдинг
    "trade-dry"         { Invoke-TradeDry }
    "trade-live"        { Invoke-TradeLive }
    "backtest"          { Invoke-Backtest }
    "hyperopt"          { Invoke-Hyperopt }
    "list-strategies"   { Invoke-ListStrategies }
    "list-pairs"        { Invoke-ListPairs }
    
    # Исследования
    "research"          { Invoke-Research }
    "download-data"     { Invoke-DownloadData }
    "verify-data"       { Invoke-VerifyData }
    
    # Мониторинг
    "dashboard"         { Invoke-Dashboard }
    "monitoring"        { Invoke-Monitoring }
    "monitoring-stop"   { Invoke-MonitoringStop }
    
    # Обслуживание
    "clean"             { Invoke-Clean }
    "clean-all"         { Invoke-CleanAll }
    "db-backup"         { Invoke-DbBackup }
    "validate-config"   { Invoke-ValidateConfig }
    
    default {
        Write-ColorOutput Red "❌ Неизвестная команда: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}

Write-Host ""
Write-ColorOutput Cyan "🏛️  Stoic Citadel - Trade with wisdom, not emotion."
Write-Host ""
