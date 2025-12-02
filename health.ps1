# ==============================================================================
# HEALTH CHECK - Проверка здоровья системы
# ==============================================================================

$PROJECT_DIR = "C:\hft-algotrade-bot"

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
    Write-ColorOutput Cyan "║            STOIC CITADEL - HEALTH CHECK                    ║"
    Write-ColorOutput Cyan "╚════════════════════════════════════════════════════════════╝"
    Write-Host ""
}

Show-Header
Set-Location $PROJECT_DIR

$allGood = $true

# Проверка Docker
Write-ColorOutput Cyan "🐳 Проверка Docker..."
try {
    $dockerVersion = docker --version
    Write-ColorOutput Green "  ✅ Docker: $dockerVersion"
} catch {
    Write-ColorOutput Red "  ❌ Docker не найден или не запущен"
    $allGood = $false
}

try {
    $composeVersion = docker-compose --version
    Write-ColorOutput Green "  ✅ Docker Compose: $composeVersion"
} catch {
    Write-ColorOutput Red "  ❌ Docker Compose не найден"
    $allGood = $false
}

# Проверка .env файла
Write-Host ""
Write-ColorOutput Cyan "📝 Проверка .env файла..."
if (Test-Path ".env") {
    Write-ColorOutput Green "  ✅ .env файл существует"
    
    $envContent = Get-Content ".env" -Raw
    
    # Проверка критичных переменных
    $required = @(
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "FREQTRADE_API_PASSWORD"
    )
    
    foreach ($var in $required) {
        if ($envContent -match "$var\s*=\s*\S+") {
            Write-ColorOutput Green "  ✅ $var настроен"
        } else {
            Write-ColorOutput Red "  ❌ $var не настроен или пустой"
            $allGood = $false
        }
    }
    
    # Проверка DRY_RUN
    if ($envContent -match "DRY_RUN\s*=\s*true") {
        Write-ColorOutput Green "  ✅ DRY_RUN=true (безопасно)"
    } elseif ($envContent -match "DRY_RUN\s*=\s*false") {
        Write-ColorOutput Yellow "  ⚠️  DRY_RUN=false (LIVE TRADING!)"
    }
    
} else {
    Write-ColorOutput Red "  ❌ .env файл не найден"
    $allGood = $false
}

# Проверка директорий
Write-Host ""
Write-ColorOutput Cyan "📁 Проверка структуры директорий..."
$requiredDirs = @(
    "user_data",
    "user_data/strategies",
    "user_data/config",
    "user_data/data",
    "scripts",
    "research"
)

foreach ($dir in $requiredDirs) {
    if (Test-Path $dir) {
        Write-ColorOutput Green "  ✅ $dir"
    } else {
        Write-ColorOutput Yellow "  ⚠️  $dir отсутствует (будет создана при setup)"
    }
}

# Проверка стратегий
Write-Host ""
Write-ColorOutput Cyan "🎯 Проверка стратегий..."
$strategies = Get-ChildItem "user_data\strategies\*.py" -ErrorAction SilentlyContinue

if ($strategies) {
    Write-ColorOutput Green "  ✅ Найдено $($strategies.Count) стратегий:"
    foreach ($strat in $strategies) {
        Write-Host "     - $($strat.BaseName)"
    }
} else {
    Write-ColorOutput Red "  ❌ Стратегии не найдены"
    $allGood = $false
}

# Проверка Docker контейнеров
Write-Host ""
Write-ColorOutput Cyan "🐳 Проверка Docker контейнеров..."
try {
    $containers = docker-compose ps --format json 2>$null | ConvertFrom-Json
    
    if ($containers) {
        Write-ColorOutput Green "  ✅ Найдено $($containers.Count) контейнеров:"
        foreach ($container in $containers) {
            $name = $container.Service
            $status = $container.State
            
            if ($status -eq "running") {
                Write-ColorOutput Green "     ✅ $name - running"
            } else {
                Write-ColorOutput Yellow "     ⚠️  $name - $status"
            }
        }
    } else {
        Write-ColorOutput Yellow "  ⚠️  Контейнеры не запущены"
    }
} catch {
    Write-ColorOutput Yellow "  ⚠️  Не удалось проверить контейнеры"
}

# Проверка данных
Write-Host ""
Write-ColorOutput Cyan "📊 Проверка данных..."
$dataFiles = Get-ChildItem "user_data\data\binance\*.json" -ErrorAction SilentlyContinue -Recurse

if ($dataFiles) {
    Write-ColorOutput Green "  ✅ Найдено $($dataFiles.Count) файлов данных"
    
    # Проверка свежести данных
    $newest = $dataFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $age = (Get-Date) - $newest.LastWriteTime
    
    if ($age.Days -lt 1) {
        Write-ColorOutput Green "  ✅ Данные свежие (обновлены сегодня)"
    } elseif ($age.Days -lt 7) {
        Write-ColorOutput Yellow "  ⚠️  Данные устарели ($($age.Days) дней)"
    } else {
        Write-ColorOutput Red "  ❌ Данные очень старые ($($age.Days) дней)"
        Write-Host "     Запусти: .\stoic.ps1 download-data"
    }
} else {
    Write-ColorOutput Yellow "  ⚠️  Данные не найдены"
    Write-Host "     Запусти: .\stoic.ps1 download-data"
}

# Проверка базы данных
Write-Host ""
Write-ColorOutput Cyan "💾 Проверка базы данных..."
if (Test-Path "user_data\tradesv3.sqlite") {
    $dbSize = (Get-Item "user_data\tradesv3.sqlite").Length / 1MB
    Write-ColorOutput Green "  ✅ База данных найдена ($('{0:N2}' -f $dbSize) MB)"
} else {
    Write-ColorOutput Yellow "  ⚠️  База данных не найдена (создастся при первом запуске)"
}

# Проверка портов
Write-Host ""
Write-ColorOutput Cyan "🌐 Проверка доступности портов..."
$ports = @{
    "3000" = "FreqUI Dashboard"
    "8080" = "Freqtrade API"
    "8888" = "Jupyter Lab"
    "9000" = "Portainer"
    "5432" = "PostgreSQL"
}

foreach ($port in $ports.Keys) {
    try {
        $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue -ErrorAction SilentlyContinue
        if ($connection.TcpTestSucceeded) {
            Write-ColorOutput Green "  ✅ Port $port ($($ports[$port])) - открыт"
        } else {
            Write-ColorOutput Yellow "  ⚠️  Port $port ($($ports[$port])) - закрыт"
        }
    } catch {
        Write-ColorOutput Yellow "  ⚠️  Port $port ($($ports[$port])) - недоступен"
    }
}

# Итоговый результат
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
if ($allGood) {
    Write-ColorOutput Green "✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!"
    Write-Host ""
    Write-ColorOutput Cyan "🚀 Готово к запуску:"
    Write-Host "   .\stoic.ps1 trade-dry    # Paper trading"
    Write-Host "   .\stoic.ps1 dashboard    # Открыть dashboard"
} else {
    Write-ColorOutput Yellow "⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОШЛИ"
    Write-Host ""
    Write-ColorOutput Cyan "🔧 Рекомендации:"
    Write-Host "   1. Проверьте Docker Desktop (должен быть запущен)"
    Write-Host "   2. Заполните .env файл"
    Write-Host "   3. Запустите: .\stoic.ps1 setup"
}
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host ""

Write-ColorOutput Cyan "🏛️  Stoic Citadel - Trade with wisdom, not emotion"
Write-Host ""
