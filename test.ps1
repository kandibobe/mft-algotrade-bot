# ==============================================================================
# TESTING AUTOMATION - Автоматизация тестирования стратегий (UPDATED)
# ==============================================================================
# Доступные стратегии:
# - StoicStrategyV1 (по умолчанию)
# - StoicCitadelV2
# - StoicEnsembleStrategy

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Strategy = "StoicStrategyV1",
    
    [Parameter(Position=2)]
    [int]$Days = 30
)

$ErrorActionPreference = "Stop"
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
    Write-ColorOutput Cyan "║         STOIC CITADEL - TESTING AUTOMATION                ║"
    Write-ColorOutput Cyan "╚════════════════════════════════════════════════════════════╝"
    Write-Host ""
}

function Show-Help {
    Show-Header
    Write-ColorOutput Green "📋 КОМАНДЫ ТЕСТИРОВАНИЯ:"
    Write-Host ""
    Write-Host "  quick              - Быстрый тест (7 дней)"
    Write-Host "  standard           - Стандартный тест (30 дней)"
    Write-Host "  full               - Полный тест (90 дней)"
    Write-Host "  compare            - Сравнить с baseline"
    Write-Host "  compare-all        - Сравнить все стратегии"
    Write-Host "  analyze            - Анализ последних результатов"
    Write-Host "  report             - Создать отчет"
    Write-Host "  walk-forward       - Walk-forward валидация"
    Write-Host ""
    Write-ColorOutput Green "📊 ПРИМЕРЫ:"
    Write-Host ""
    Write-Host "  .\test.ps1 quick StoicStrategyV1"
    Write-Host "  .\test.ps1 standard StoicCitadelV2"
    Write-Host "  .\test.ps1 full StoicEnsembleStrategy 90"
    Write-Host "  .\test.ps1 compare"
    Write-Host "  .\test.ps1 compare-all"
    Write-Host "  .\test.ps1 analyze"
    Write-Host ""
    Write-ColorOutput Yellow "Доступные стратегии:"
    Write-Host "  - StoicStrategyV1 (по умолчанию)"
    Write-Host "  - StoicCitadelV2"
    Write-Host "  - StoicEnsembleStrategy"
    Write-Host ""
}

# Быстрый тест (7 дней)
function Test-Quick {
    Show-Header
    Write-ColorOutput Cyan "🚀 Быстрый тест: $Strategy (7 дней)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-7).ToString("yyyyMMdd")
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections
    
    Write-ColorOutput Green "✅ Быстрый тест завершен!"
    Invoke-Analyze
}

# Стандартный тест (30 дней)
function Test-Standard {
    Show-Header
    Write-ColorOutput Cyan "📊 Стандартный тест: $Strategy (30 дней)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-30).ToString("yyyyMMdd")
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections `
        --breakdown day week
    
    Write-ColorOutput Green "✅ Стандартный тест завершен!"
    Invoke-Analyze
}

# Полный тест (90+ дней)
function Test-Full {
    Show-Header
    Write-ColorOutput Cyan "📈 Полный тест: $Strategy ($Days дней)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-$Days).ToString("yyyyMMdd")
    
    Write-ColorOutput Yellow "⏱️  Это займет 5-10 минут..."
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections `
        --breakdown day week month
    
    Write-ColorOutput Green "✅ Полный тест завершен!"
    Invoke-Analyze
}

# Сравнение с baseline
function Test-Compare {
    Show-Header
    Write-ColorOutput Cyan "🔄 Сравнение с baseline ($Strategy)"
    Set-Location $PROJECT_DIR
    
    # Сохраняем текущие результаты как baseline если нет
    $baselineFile = "user_data\backtest_results\baseline_$Strategy.json"
    
    if (-not (Test-Path $baselineFile)) {
        Write-ColorOutput Yellow "⚠️  Baseline для $Strategy не найден"
        $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
                  Where-Object { $_.Name -notlike "baseline_*" } |
                  Sort-Object LastWriteTime -Descending | 
                  Select-Object -First 1
        
        if ($latest) {
            Write-ColorOutput Yellow "⚠️  Создаю baseline из последнего теста..."
            Copy-Item $latest.FullName $baselineFile
            Write-ColorOutput Green "✅ Baseline создан: $($latest.Name)"
        } else {
            Write-ColorOutput Red "❌ Нет результатов для baseline. Запусти сначала тест."
            Write-Host ""
            Write-Host "Запусти: .\test.ps1 standard $Strategy"
            return
        }
    }
    
    # Запускаем новый тест
    Write-ColorOutput Cyan "🧪 Запуск нового теста для сравнения..."
    Test-Standard
    
    # Сравниваем
    Write-ColorOutput Cyan "📊 Анализ разницы..."
    
    try {
        $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
        $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
                  Where-Object { $_.Name -notlike "baseline_*" } |
                  Sort-Object LastWriteTime -Descending | 
                  Select-Object -First 1
        
        if (-not $latest) {
            Write-ColorOutput Red "❌ Новые результаты не найдены"
            return
        }
        
        $current = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        
        Write-Host ""
        Write-ColorOutput Cyan "📊 СРАВНЕНИЕ С BASELINE ($Strategy):"
        Write-Host ""
        Write-Host "Метрика              | Baseline  | Текущий   | Разница"
        Write-Host "---------------------|-----------|-----------|----------"
        
        # Сравнение прибыли
        $baseProfit = [math]::Round($baseline.strategy.$Strategy.profit_total_abs, 2)
        $currProfit = [math]::Round($current.strategy.$Strategy.profit_total_abs, 2)
        $profitDiff = [math]::Round($currProfit - $baseProfit, 2)
        $profitSign = if ($profitDiff -gt 0) { "+" } else { "" }
        Write-Host "Total Profit (USDT)  | $baseProfit    | $currProfit    | $profitSign$profitDiff"
        
        # Сравнение Win Rate
        $baseWR = [math]::Round($baseline.strategy.$Strategy.wins / $baseline.strategy.$Strategy.total_trades * 100, 1)
        $currWR = [math]::Round($current.strategy.$Strategy.wins / $current.strategy.$Strategy.total_trades * 100, 1)
        $wrDiff = [math]::Round($currWR - $baseWR, 1)
        $wrSign = if ($wrDiff -gt 0) { "+" } else { "" }
        Write-Host "Win Rate (%)         | $baseWR%     | $currWR%     | $wrSign$wrDiff%"
        
        # Сравнение Drawdown
        $baseDD = [math]::Round([math]::Abs($baseline.strategy.$Strategy.max_drawdown_abs), 2)
        $currDD = [math]::Round([math]::Abs($current.strategy.$Strategy.max_drawdown_abs), 2)
        $ddDiff = [math]::Round($currDD - $baseDD, 2)
        $ddSign = if ($ddDiff -gt 0) { "+" } else { "" }
        Write-Host "Max Drawdown (USDT)  | $baseDD    | $currDD    | $ddSign$ddDiff"
        
        Write-Host ""
        
        # Итоговая оценка
        $improvements = 0
        if ($profitDiff -gt 0) { $improvements++ }
        if ($wrDiff -gt 0) { $improvements++ }
        if ($ddDiff -lt 0) { $improvements++ }  # Меньше DD = лучше
        
        if ($improvements -ge 2) {
            Write-ColorOutput Green "✅ УЛУЧШЕНИЕ! ($improvements из 3 метрик лучше)"
        } elseif ($improvements -eq 1) {
            Write-ColorOutput Yellow "➖ СМЕШАННЫЕ РЕЗУЛЬТАТЫ (1 из 3 метрик лучше)"
        } else {
            Write-ColorOutput Red "❌ УХУДШЕНИЕ! Откатитесь к baseline"
        }
        
        Write-Host ""
        $updateBaseline = Read-Host "Обновить baseline новыми результатами? (yes/no)"
        if ($updateBaseline -eq "yes") {
            Copy-Item $latest.FullName $baselineFile -Force
            Write-ColorOutput Green "✅ Baseline обновлен для $Strategy!"
        }
        
    } catch {
        Write-ColorOutput Red "❌ Ошибка при сравнении: $_"
    }
}

# Сравнение всех стратегий
function Test-CompareAll {
    Show-Header
    Write-ColorOutput Cyan "🔄 Сравнение всех стратегий"
    Set-Location $PROJECT_DIR
    
    $strategies = @("StoicStrategyV1", "StoicCitadelV2", "StoicEnsembleStrategy")
    $results = @()
    
    foreach ($strat in $strategies) {
        Write-ColorOutput Cyan "🧪 Тестирование $strat..."
        
        $endDate = Get-Date -Format "yyyyMMdd"
        $startDate = (Get-Date).AddDays(-30).ToString("yyyyMMdd")
        
        docker-compose run --rm freqtrade backtesting `
            --strategy $strat `
            --timerange ${startDate}-${endDate} `
            --enable-protections | Out-Null
        
        # Находим последний результат
        $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
                  Where-Object { $_.Name -notlike "baseline_*" } |
                  Sort-Object LastWriteTime -Descending | 
                  Select-Object -First 1
        
        if ($latest) {
            try {
                $data = Get-Content $latest.FullName -Raw | ConvertFrom-Json
                $stratData = $data.strategy.$strat
                
                $results += [PSCustomObject]@{
                    Strategy = $strat
                    Profit = [math]::Round($stratData.profit_total_abs, 2)
                    WinRate = [math]::Round($stratData.wins / $stratData.total_trades * 100, 1)
                    Drawdown = [math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)
                    Trades = $stratData.total_trades
                }
            } catch {
                Write-ColorOutput Yellow "⚠️  Не удалось обработать результаты $strat"
            }
        }
    }
    
    Write-Host ""
    Write-ColorOutput Cyan "📊 СРАВНЕНИЕ ВСЕХ СТРАТЕГИЙ (30 дней):"
    Write-Host ""
    $results | Format-Table -AutoSize
    
    # Определяем лучшую
    $best = $results | Sort-Object -Property Profit -Descending | Select-Object -First 1
    Write-Host ""
    Write-ColorOutput Green "🏆 ЛУЧШАЯ СТРАТЕГИЯ: $($best.Strategy)"
    Write-Host "   Profit: $($best.Profit) USDT | Win Rate: $($best.WinRate)% | DD: $($best.Drawdown) USDT"
    Write-Host ""
}

# Walk-forward валидация
function Test-WalkForward {
    Show-Header
    Write-ColorOutput Cyan "🚶 Walk-forward валидация для $Strategy"
    Write-ColorOutput Yellow "⏱️  Это займет 15-30 минут..."
    Set-Location $PROJECT_DIR
    
    # Проверяем наличие скрипта
    if (Test-Path "scripts\walk_forward.py") {
        docker-compose run --rm jupyter python /home/jovyan/scripts/walk_forward.py `
            --strategy $Strategy `
            --train-period 60 `
            --test-period 15
        
        Write-ColorOutput Green "✅ Walk-forward валидация завершена!"
    } else {
        Write-ColorOutput Red "❌ Скрипт walk_forward.py не найден"
    }
}

# Анализ результатов
function Invoke-Analyze {
    Write-Host ""
    Write-ColorOutput Cyan "📊 АНАЛИЗ РЕЗУЛЬТАТОВ ($Strategy):"
    Write-Host ""
    Set-Location $PROJECT_DIR
    
    # Находим последний результат
    $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
              Where-Object { $_.Name -notlike "baseline_*" } |
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    if (-not $latest) {
        Write-ColorOutput Red "❌ Нет результатов для анализа"
        return
    }
    
    try {
        $results = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        $stratData = $results.strategy.$Strategy
        
        # Основные метрики
        $totalTrades = $stratData.total_trades
        $wins = $stratData.wins
        $losses = $stratData.total_trades - $stratData.wins
        $profitTotal = [math]::Round($stratData.profit_total_abs, 2)
        $profitPct = [math]::Round($stratData.profit_total * 100, 2)
        $winRate = [math]::Round($wins / $totalTrades * 100, 1)
        $maxDrawdown = [math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)
        $avgProfit = [math]::Round($stratData.profit_mean, 2)
        
        Write-Host "╔════════════════════════════════════════════════╗"
        Write-Host "║           ОСНОВНЫЕ МЕТРИКИ                     ║"
        Write-Host "╚════════════════════════════════════════════════╝"
        Write-Host ""
        Write-Host "  📊 Total Trades:       $totalTrades"
        Write-Host "  ✅ Wins:               $wins"
        Write-Host "  ❌ Losses:             $losses"
        Write-Host "  💰 Total Profit:       $profitTotal USDT ($profitPct%)"
        Write-Host "  📈 Avg Profit/Trade:   $avgProfit USDT"
        Write-Host "  ✅ Win Rate:           $winRate%"
        Write-Host "  📉 Max Drawdown:       $maxDrawdown USDT"
        Write-Host ""
        
        # Оценка стратегии
        $score = 0
        
        # Profit
        if ($profitPct -gt 10) { $score += 3 }
        elseif ($profitPct -gt 5) { $score += 2 }
        elseif ($profitPct -gt 0) { $score += 1 }
        else { $score -= 5 }
        
        # Win Rate
        if ($winRate -gt 60) { $score += 3 }
        elseif ($winRate -gt 55) { $score += 2 }
        elseif ($winRate -gt 50) { $score += 1 }
        else { $score -= 2 }
        
        # Drawdown
        if ($maxDrawdown -lt 50) { $score += 3 }
        elseif ($maxDrawdown -lt 100) { $score += 2 }
        elseif ($maxDrawdown -lt 150) { $score += 1 }
        else { $score -= 3 }
        
        Write-Host "╔════════════════════════════════════════════════╗"
        Write-Host "║           ОЦЕНКА СТРАТЕГИИ                     ║"
        Write-Host "╚════════════════════════════════════════════════╝"
        Write-Host ""
        Write-Host "  Общий счет: $score баллов"
        Write-Host ""
        
        if ($score -ge 8) {
            Write-ColorOutput Green "  🟢 ОТЛИЧНО! Стратегия готова к production"
        } elseif ($score -ge 5) {
            Write-ColorOutput Yellow "  🟡 ХОРОШО! Есть что улучшить"
        } elseif ($score -ge 2) {
            Write-ColorOutput Yellow "  🟠 ПОСРЕДСТВЕННО! Нужна оптимизация"
        } else {
            Write-ColorOutput Red "  🔴 ПЛОХО! Переделывай стратегию"
        }
        Write-Host ""
        
    } catch {
        Write-ColorOutput Red "❌ Ошибка при анализе: $_"
    }
}

# Создание отчета
function New-Report {
    Show-Header
    Write-ColorOutput Cyan "📝 Создание отчета для $Strategy"
    Set-Location $PROJECT_DIR
    
    $reportDir = "reports"
    if (-not (Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir | Out-Null
    }
    
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $reportFile = "$reportDir\test_report_${Strategy}_${timestamp}.txt"
    
    # Находим последний результат
    $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
              Where-Object { $_.Name -notlike "baseline_*" } |
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    if (-not $latest) {
        Write-ColorOutput Red "❌ Нет результатов для отчета"
        return
    }
    
    try {
        $results = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        $stratData = $results.strategy.$Strategy
        
        # Создаем отчет
        $report = @"
╔══════════════════════════════════════════════════════════════════╗
║         STOIC CITADEL - ОТЧЕТ О ТЕСТИРОВАНИИ                     ║
╚══════════════════════════════════════════════════════════════════╝

Дата создания: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Стратегия: $Strategy
Период тестирования: $($results.backtest_start_time) - $($results.backtest_end_time)

═══════════════════════════════════════════════════════════════════
ОСНОВНЫЕ МЕТРИКИ
═══════════════════════════════════════════════════════════════════

Total Trades:           $($stratData.total_trades)
Winning Trades:         $($stratData.wins)
Losing Trades:          $($stratData.total_trades - $stratData.wins)
Win Rate:               $([math]::Round($stratData.wins / $stratData.total_trades * 100, 2))%

Total Profit:           $([math]::Round($stratData.profit_total_abs, 2)) USDT
Total Profit %:         $([math]::Round($stratData.profit_total * 100, 2))%
Avg Profit per Trade:   $([math]::Round($stratData.profit_mean, 2)) USDT
Best Trade:             $([math]::Round($stratData.best_pair.profit_abs, 2)) USDT
Worst Trade:            $([math]::Round($stratData.worst_pair.profit_abs, 2)) USDT

Max Drawdown:           $([math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)) USDT
Max Drawdown %:         $([math]::Round($stratData.max_drawdown * 100, 2))%

═══════════════════════════════════════════════════════════════════
РЕКОМЕНДАЦИИ
═══════════════════════════════════════════════════════════════════

"@
        
        # Добавляем рекомендации
        $profitPct = [math]::Round($stratData.profit_total * 100, 2)
        $winRate = [math]::Round($stratData.wins / $stratData.total_trades * 100, 1)
        $maxDD = [math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)
        
        if ($profitPct -lt 0) {
            $report += "`n❌ КРИТИЧНО: Стратегия убыточна! Требуется полная переработка."
        } elseif ($winRate -lt 50) {
            $report += "`n⚠️  Win Rate низкий. Рекомендуется улучшить условия входа."
        } elseif ($maxDD -gt 200) {
            $report += "`n⚠️  Drawdown слишком высокий. Добавьте защитные механизмы."
        } else {
            $report += "`n✅ Стратегия показывает приемлемые результаты."
            
            if ($profitPct -gt 10 -and $winRate -gt 55) {
                $report += "`n🟢 Стратегия готова к тестированию в dry-run режиме."
            }
        }
        
        $report += "`n`nДля детального анализа см. файл: $($latest.Name)"
        
        # Сохраняем отчет
        $report | Out-File -FilePath $reportFile -Encoding UTF8
        
        Write-ColorOutput Green "✅ Отчет создан: $reportFile"
        Write-Host ""
        Write-Host "Открыть отчет? (Enter для открытия, любая клавиша для отмены)"
        $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        if ($key.VirtualKeyCode -eq 13) {  # Enter
            notepad $reportFile
        }
        
    } catch {
        Write-ColorOutput Red "❌ Ошибка при создании отчета: $_"
    }
}

# Main logic
Set-Location $PROJECT_DIR

switch ($Command.ToLower()) {
    "quick"         { Test-Quick }
    "standard"      { Test-Standard }
    "full"          { Test-Full }
    "compare"       { Test-Compare }
    "compare-all"   { Test-CompareAll }
    "walk-forward"  { Test-WalkForward }
    "analyze"       { Invoke-Analyze }
    "report"        { New-Report }
    "help"          { Show-Help }
    
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
