# ==============================================================================
# TESTING AUTOMATION - РђРІС‚РѕРјР°С‚РёР·Р°С†РёСЏ С‚РµСЃС‚РёСЂРѕРІР°РЅРёСЏ СЃС‚СЂР°С‚РµРіРёР№ (UPDATED)
# ==============================================================================
# Р”РѕСЃС‚СѓРїРЅС‹Рµ СЃС‚СЂР°С‚РµРіРёРё:
# - StoicStrategyV1 (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ)
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
    Write-ColorOutput Cyan "в•”в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•—"
    Write-ColorOutput Cyan "в•‘         STOIC CITADEL - TESTING AUTOMATION                в•‘"
    Write-ColorOutput Cyan "в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ"
    Write-Host ""
}

function Show-Help {
    Show-Header
    Write-ColorOutput Green "рџ“‹ РљРћРњРђРќР”Р« РўР•РЎРўРР РћР’РђРќРРЇ:"
    Write-Host ""
    Write-Host "  quick              - Р‘С‹СЃС‚СЂС‹Р№ С‚РµСЃС‚ (7 РґРЅРµР№)"
    Write-Host "  standard           - РЎС‚Р°РЅРґР°СЂС‚РЅС‹Р№ С‚РµСЃС‚ (30 РґРЅРµР№)"
    Write-Host "  full               - РџРѕР»РЅС‹Р№ С‚РµСЃС‚ (90 РґРЅРµР№)"
    Write-Host "  compare            - РЎСЂР°РІРЅРёС‚СЊ СЃ baseline"
    Write-Host "  compare-all        - РЎСЂР°РІРЅРёС‚СЊ РІСЃРµ СЃС‚СЂР°С‚РµРіРёРё"
    Write-Host "  analyze            - РђРЅР°Р»РёР· РїРѕСЃР»РµРґРЅРёС… СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ"
    Write-Host "  report             - РЎРѕР·РґР°С‚СЊ РѕС‚С‡РµС‚"
    Write-Host "  walk-forward       - Walk-forward РІР°Р»РёРґР°С†РёСЏ"
    Write-Host ""
    Write-ColorOutput Green "рџ“Љ РџР РРњР•Р Р«:"
    Write-Host ""
    Write-Host "  .\test.ps1 quick StoicStrategyV1"
    Write-Host "  .\test.ps1 standard StoicCitadelV2"
    Write-Host "  .\test.ps1 full StoicEnsembleStrategy 90"
    Write-Host "  .\test.ps1 compare"
    Write-Host "  .\test.ps1 compare-all"
    Write-Host "  .\test.ps1 analyze"
    Write-Host ""
    Write-ColorOutput Yellow "Р”РѕСЃС‚СѓРїРЅС‹Рµ СЃС‚СЂР°С‚РµРіРёРё:"
    Write-Host "  - StoicStrategyV1 (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ)"
    Write-Host "  - StoicCitadelV2"
    Write-Host "  - StoicEnsembleStrategy"
    Write-Host ""
}

# Р‘С‹СЃС‚СЂС‹Р№ С‚РµСЃС‚ (7 РґРЅРµР№)
function Test-Quick {
    Show-Header
    Write-ColorOutput Cyan "рџљЂ Р‘С‹СЃС‚СЂС‹Р№ С‚РµСЃС‚: $Strategy (7 РґРЅРµР№)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-7).ToString("yyyyMMdd")
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections
    
    Write-ColorOutput Green "вњ… Р‘С‹СЃС‚СЂС‹Р№ С‚РµСЃС‚ Р·Р°РІРµСЂС€РµРЅ!"
    Invoke-Analyze
}

# РЎС‚Р°РЅРґР°СЂС‚РЅС‹Р№ С‚РµСЃС‚ (30 РґРЅРµР№)
function Test-Standard {
    Show-Header
    Write-ColorOutput Cyan "рџ“Љ РЎС‚Р°РЅРґР°СЂС‚РЅС‹Р№ С‚РµСЃС‚: $Strategy (30 РґРЅРµР№)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-30).ToString("yyyyMMdd")
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections `
        --breakdown day week
    
    Write-ColorOutput Green "вњ… РЎС‚Р°РЅРґР°СЂС‚РЅС‹Р№ С‚РµСЃС‚ Р·Р°РІРµСЂС€РµРЅ!"
    Invoke-Analyze
}

# РџРѕР»РЅС‹Р№ С‚РµСЃС‚ (90+ РґРЅРµР№)
function Test-Full {
    Show-Header
    Write-ColorOutput Cyan "рџ“€ РџРѕР»РЅС‹Р№ С‚РµСЃС‚: $Strategy ($Days РґРЅРµР№)"
    Set-Location $PROJECT_DIR
    
    $endDate = Get-Date -Format "yyyyMMdd"
    $startDate = (Get-Date).AddDays(-$Days).ToString("yyyyMMdd")
    
    Write-ColorOutput Yellow "вЏ±пёЏ  Р­С‚Рѕ Р·Р°Р№РјРµС‚ 5-10 РјРёРЅСѓС‚..."
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange ${startDate}-${endDate} `
        --enable-protections `
        --breakdown day week month
    
    Write-ColorOutput Green "вњ… РџРѕР»РЅС‹Р№ С‚РµСЃС‚ Р·Р°РІРµСЂС€РµРЅ!"
    Invoke-Analyze
}

# РЎСЂР°РІРЅРµРЅРёРµ СЃ baseline
function Test-Compare {
    Show-Header
    Write-ColorOutput Cyan "рџ”„ РЎСЂР°РІРЅРµРЅРёРµ СЃ baseline ($Strategy)"
    Set-Location $PROJECT_DIR
    
    # РЎРѕС…СЂР°РЅСЏРµРј С‚РµРєСѓС‰РёРµ СЂРµР·СѓР»СЊС‚Р°С‚С‹ РєР°Рє baseline РµСЃР»Рё РЅРµС‚
    $baselineFile = "user_data\backtest_results\baseline_$Strategy.json"
    
    if (-not (Test-Path $baselineFile)) {
        Write-ColorOutput Yellow "вљ пёЏ  Baseline РґР»СЏ $Strategy РЅРµ РЅР°Р№РґРµРЅ"
        $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
                  Where-Object { $_.Name -notlike "baseline_*" } |
                  Sort-Object LastWriteTime -Descending | 
                  Select-Object -First 1
        
        if ($latest) {
            Write-ColorOutput Yellow "вљ пёЏ  РЎРѕР·РґР°СЋ baseline РёР· РїРѕСЃР»РµРґРЅРµРіРѕ С‚РµСЃС‚Р°..."
            Copy-Item $latest.FullName $baselineFile
            Write-ColorOutput Green "вњ… Baseline СЃРѕР·РґР°РЅ: $($latest.Name)"
        } else {
            Write-ColorOutput Red "вќЊ РќРµС‚ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ РґР»СЏ baseline. Р—Р°РїСѓСЃС‚Рё СЃРЅР°С‡Р°Р»Р° С‚РµСЃС‚."
            Write-Host ""
            Write-Host "Р—Р°РїСѓСЃС‚Рё: .\test.ps1 standard $Strategy"
            return
        }
    }
    
    # Р—Р°РїСѓСЃРєР°РµРј РЅРѕРІС‹Р№ С‚РµСЃС‚
    Write-ColorOutput Cyan "рџ§Є Р—Р°РїСѓСЃРє РЅРѕРІРѕРіРѕ С‚РµСЃС‚Р° РґР»СЏ СЃСЂР°РІРЅРµРЅРёСЏ..."
    Test-Standard
    
    # РЎСЂР°РІРЅРёРІР°РµРј
    Write-ColorOutput Cyan "рџ“Љ РђРЅР°Р»РёР· СЂР°Р·РЅРёС†С‹..."
    
    try {
        $baseline = Get-Content $baselineFile -Raw | ConvertFrom-Json
        $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
                  Where-Object { $_.Name -notlike "baseline_*" } |
                  Sort-Object LastWriteTime -Descending | 
                  Select-Object -First 1
        
        if (-not $latest) {
            Write-ColorOutput Red "вќЊ РќРѕРІС‹Рµ СЂРµР·СѓР»СЊС‚Р°С‚С‹ РЅРµ РЅР°Р№РґРµРЅС‹"
            return
        }
        
        $current = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        
        Write-Host ""
        Write-ColorOutput Cyan "рџ“Љ РЎР РђР’РќР•РќРР• РЎ BASELINE ($Strategy):"
        Write-Host ""
        Write-Host "РњРµС‚СЂРёРєР°              | Baseline  | РўРµРєСѓС‰РёР№   | Р Р°Р·РЅРёС†Р°"
        Write-Host "---------------------|-----------|-----------|----------"
        
        # РЎСЂР°РІРЅРµРЅРёРµ РїСЂРёР±С‹Р»Рё
        $baseProfit = [math]::Round($baseline.strategy.$Strategy.profit_total_abs, 2)
        $currProfit = [math]::Round($current.strategy.$Strategy.profit_total_abs, 2)
        $profitDiff = [math]::Round($currProfit - $baseProfit, 2)
        $profitSign = if ($profitDiff -gt 0) { "+" } else { "" }
        Write-Host "Total Profit (USDT)  | $baseProfit    | $currProfit    | $profitSign$profitDiff"
        
        # РЎСЂР°РІРЅРµРЅРёРµ Win Rate
        $baseWR = [math]::Round($baseline.strategy.$Strategy.wins / $baseline.strategy.$Strategy.total_trades * 100, 1)
        $currWR = [math]::Round($current.strategy.$Strategy.wins / $current.strategy.$Strategy.total_trades * 100, 1)
        $wrDiff = [math]::Round($currWR - $baseWR, 1)
        $wrSign = if ($wrDiff -gt 0) { "+" } else { "" }
        Write-Host "Win Rate (%)         | $baseWR%     | $currWR%     | $wrSign$wrDiff%"
        
        # РЎСЂР°РІРЅРµРЅРёРµ Drawdown
        $baseDD = [math]::Round([math]::Abs($baseline.strategy.$Strategy.max_drawdown_abs), 2)
        $currDD = [math]::Round([math]::Abs($current.strategy.$Strategy.max_drawdown_abs), 2)
        $ddDiff = [math]::Round($currDD - $baseDD, 2)
        $ddSign = if ($ddDiff -gt 0) { "+" } else { "" }
        Write-Host "Max Drawdown (USDT)  | $baseDD    | $currDD    | $ddSign$ddDiff"
        
        Write-Host ""
        
        # РС‚РѕРіРѕРІР°СЏ РѕС†РµРЅРєР°
        $improvements = 0
        if ($profitDiff -gt 0) { $improvements++ }
        if ($wrDiff -gt 0) { $improvements++ }
        if ($ddDiff -lt 0) { $improvements++ }  # РњРµРЅСЊС€Рµ DD = Р»СѓС‡С€Рµ
        
        if ($improvements -ge 2) {
            Write-ColorOutput Green "вњ… РЈР›РЈР§РЁР•РќРР•! ($improvements РёР· 3 РјРµС‚СЂРёРє Р»СѓС‡С€Рµ)"
        } elseif ($improvements -eq 1) {
            Write-ColorOutput Yellow "вћ– РЎРњР•РЁРђРќРќР«Р• Р Р•Р—РЈР›Р¬РўРђРўР« (1 РёР· 3 РјРµС‚СЂРёРє Р»СѓС‡С€Рµ)"
        } else {
            Write-ColorOutput Red "вќЊ РЈРҐРЈР”РЁР•РќРР•! РћС‚РєР°С‚РёС‚РµСЃСЊ Рє baseline"
        }
        
        Write-Host ""
        $updateBaseline = Read-Host "РћР±РЅРѕРІРёС‚СЊ baseline РЅРѕРІС‹РјРё СЂРµР·СѓР»СЊС‚Р°С‚Р°РјРё? (yes/no)"
        if ($updateBaseline -eq "yes") {
            Copy-Item $latest.FullName $baselineFile -Force
            Write-ColorOutput Green "вњ… Baseline РѕР±РЅРѕРІР»РµРЅ РґР»СЏ $Strategy!"
        }
        
    } catch {
        Write-ColorOutput Red "вќЊ РћС€РёР±РєР° РїСЂРё СЃСЂР°РІРЅРµРЅРёРё: $_"
    }
}

# РЎСЂР°РІРЅРµРЅРёРµ РІСЃРµС… СЃС‚СЂР°С‚РµРіРёР№
function Test-CompareAll {
    Show-Header
    Write-ColorOutput Cyan "рџ”„ РЎСЂР°РІРЅРµРЅРёРµ РІСЃРµС… СЃС‚СЂР°С‚РµРіРёР№"
    Set-Location $PROJECT_DIR
    
    $strategies = @("StoicStrategyV1", "StoicCitadelV2", "StoicEnsembleStrategy")
    $results = @()
    
    foreach ($strat in $strategies) {
        Write-ColorOutput Cyan "рџ§Є РўРµСЃС‚РёСЂРѕРІР°РЅРёРµ $strat..."
        
        $endDate = Get-Date -Format "yyyyMMdd"
        $startDate = (Get-Date).AddDays(-30).ToString("yyyyMMdd")
        
        docker-compose run --rm freqtrade backtesting `
            --strategy $strat `
            --timerange ${startDate}-${endDate} `
            --enable-protections | Out-Null
        
        # РќР°С…РѕРґРёРј РїРѕСЃР»РµРґРЅРёР№ СЂРµР·СѓР»СЊС‚Р°С‚
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
                Write-ColorOutput Yellow "вљ пёЏ  РќРµ СѓРґР°Р»РѕСЃСЊ РѕР±СЂР°Р±РѕС‚Р°С‚СЊ СЂРµР·СѓР»СЊС‚Р°С‚С‹ $strat"
            }
        }
    }
    
    Write-Host ""
    Write-ColorOutput Cyan "рџ“Љ РЎР РђР’РќР•РќРР• Р’РЎР•РҐ РЎРўР РђРўР•Р“РР™ (30 РґРЅРµР№):"
    Write-Host ""
    $results | Format-Table -AutoSize
    
    # РћРїСЂРµРґРµР»СЏРµРј Р»СѓС‡С€СѓСЋ
    $best = $results | Sort-Object -Property Profit -Descending | Select-Object -First 1
    Write-Host ""
    Write-ColorOutput Green "рџЏ† Р›РЈР§РЁРђРЇ РЎРўР РђРўР•Р“РРЇ: $($best.Strategy)"
    Write-Host "   Profit: $($best.Profit) USDT | Win Rate: $($best.WinRate)% | DD: $($best.Drawdown) USDT"
    Write-Host ""
}

# Walk-forward РІР°Р»РёРґР°С†РёСЏ
function Test-WalkForward {
    Show-Header
    Write-ColorOutput Cyan "рџљ¶ Walk-forward РІР°Р»РёРґР°С†РёСЏ РґР»СЏ $Strategy"
    Write-ColorOutput Yellow "вЏ±пёЏ  Р­С‚Рѕ Р·Р°Р№РјРµС‚ 15-30 РјРёРЅСѓС‚..."
    Set-Location $PROJECT_DIR
    
    # РџСЂРѕРІРµСЂСЏРµРј РЅР°Р»РёС‡РёРµ СЃРєСЂРёРїС‚Р°
    if (Test-Path "scripts\walk_forward.py") {
        docker-compose run --rm jupyter python /home/jovyan/scripts/walk_forward.py `
            --strategy $Strategy `
            --train-period 60 `
            --test-period 15
        
        Write-ColorOutput Green "вњ… Walk-forward РІР°Р»РёРґР°С†РёСЏ Р·Р°РІРµСЂС€РµРЅР°!"
    } else {
        Write-ColorOutput Red "вќЊ РЎРєСЂРёРїС‚ walk_forward.py РЅРµ РЅР°Р№РґРµРЅ"
    }
}

# РђРЅР°Р»РёР· СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ
function Invoke-Analyze {
    Write-Host ""
    Write-ColorOutput Cyan "рџ“Љ РђРќРђР›РР— Р Р•Р—РЈР›Р¬РўРђРўРћР’ ($Strategy):"
    Write-Host ""
    Set-Location $PROJECT_DIR
    
    # РќР°С…РѕРґРёРј РїРѕСЃР»РµРґРЅРёР№ СЂРµР·СѓР»СЊС‚Р°С‚
    $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
              Where-Object { $_.Name -notlike "baseline_*" } |
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    if (-not $latest) {
        Write-ColorOutput Red "вќЊ РќРµС‚ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ РґР»СЏ Р°РЅР°Р»РёР·Р°"
        return
    }
    
    try {
        $results = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        $stratData = $results.strategy.$Strategy
        
        # РћСЃРЅРѕРІРЅС‹Рµ РјРµС‚СЂРёРєРё
        $totalTrades = $stratData.total_trades
        $wins = $stratData.wins
        $losses = $stratData.total_trades - $stratData.wins
        $profitTotal = [math]::Round($stratData.profit_total_abs, 2)
        $profitPct = [math]::Round($stratData.profit_total * 100, 2)
        $winRate = [math]::Round($wins / $totalTrades * 100, 1)
        $maxDrawdown = [math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)
        $avgProfit = [math]::Round($stratData.profit_mean, 2)
        
        Write-Host "в•”в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•—"
        Write-Host "в•‘           РћРЎРќРћР’РќР«Р• РњР•РўР РРљР                     в•‘"
        Write-Host "в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ"
        Write-Host ""
        Write-Host "  рџ“Љ Total Trades:       $totalTrades"
        Write-Host "  вњ… Wins:               $wins"
        Write-Host "  вќЊ Losses:             $losses"
        Write-Host "  рџ’° Total Profit:       $profitTotal USDT ($profitPct%)"
        Write-Host "  рџ“€ Avg Profit/Trade:   $avgProfit USDT"
        Write-Host "  вњ… Win Rate:           $winRate%"
        Write-Host "  рџ“‰ Max Drawdown:       $maxDrawdown USDT"
        Write-Host ""
        
        # РћС†РµРЅРєР° СЃС‚СЂР°С‚РµРіРёРё
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
        
        Write-Host "в•”в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•—"
        Write-Host "в•‘           РћР¦Р•РќРљРђ РЎРўР РђРўР•Р“РР                     в•‘"
        Write-Host "в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ"
        Write-Host ""
        Write-Host "  РћР±С‰РёР№ СЃС‡РµС‚: $score Р±Р°Р»Р»РѕРІ"
        Write-Host ""
        
        if ($score -ge 8) {
            Write-ColorOutput Green "  рџџў РћРўР›РР§РќРћ! РЎС‚СЂР°С‚РµРіРёСЏ РіРѕС‚РѕРІР° Рє production"
        } elseif ($score -ge 5) {
            Write-ColorOutput Yellow "  рџџЎ РҐРћР РћРЁРћ! Р•СЃС‚СЊ С‡С‚Рѕ СѓР»СѓС‡С€РёС‚СЊ"
        } elseif ($score -ge 2) {
            Write-ColorOutput Yellow "  рџџ  РџРћРЎР Р•Р”РЎРўР’Р•РќРќРћ! РќСѓР¶РЅР° РѕРїС‚РёРјРёР·Р°С†РёСЏ"
        } else {
            Write-ColorOutput Red "  рџ”ґ РџР›РћРҐРћ! РџРµСЂРµРґРµР»С‹РІР°Р№ СЃС‚СЂР°С‚РµРіРёСЋ"
        }
        Write-Host ""
        
    } catch {
        Write-ColorOutput Red "вќЊ РћС€РёР±РєР° РїСЂРё Р°РЅР°Р»РёР·Рµ: $_"
    }
}

# РЎРѕР·РґР°РЅРёРµ РѕС‚С‡РµС‚Р°
function New-Report {
    Show-Header
    Write-ColorOutput Cyan "рџ“ќ РЎРѕР·РґР°РЅРёРµ РѕС‚С‡РµС‚Р° РґР»СЏ $Strategy"
    Set-Location $PROJECT_DIR
    
    $reportDir = "reports"
    if (-not (Test-Path $reportDir)) {
        New-Item -ItemType Directory -Path $reportDir | Out-Null
    }
    
    $timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $reportFile = "$reportDir\test_report_${Strategy}_${timestamp}.txt"
    
    # РќР°С…РѕРґРёРј РїРѕСЃР»РµРґРЅРёР№ СЂРµР·СѓР»СЊС‚Р°С‚
    $latest = Get-ChildItem user_data\backtest_results\*.json -ErrorAction SilentlyContinue | 
              Where-Object { $_.Name -notlike "baseline_*" } |
              Sort-Object LastWriteTime -Descending | 
              Select-Object -First 1
    
    if (-not $latest) {
        Write-ColorOutput Red "вќЊ РќРµС‚ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ РґР»СЏ РѕС‚С‡РµС‚Р°"
        return
    }
    
    try {
        $results = Get-Content $latest.FullName -Raw | ConvertFrom-Json
        $stratData = $results.strategy.$Strategy
        
        # РЎРѕР·РґР°РµРј РѕС‚С‡РµС‚
        $report = @"
в•”в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•—
в•‘         STOIC CITADEL - РћРўР§Р•Рў Рћ РўР•РЎРўРР РћР’РђРќРР                     в•‘
в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ

Р”Р°С‚Р° СЃРѕР·РґР°РЅРёСЏ: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
РЎС‚СЂР°С‚РµРіРёСЏ: $Strategy
РџРµСЂРёРѕРґ С‚РµСЃС‚РёСЂРѕРІР°РЅРёСЏ: $($results.backtest_start_time) - $($results.backtest_end_time)

в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
РћРЎРќРћР’РќР«Р• РњР•РўР РРљР
в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

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

в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ
Р Р•РљРћРњР•РќР”РђР¦РР
в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ

"@
        
        # Р”РѕР±Р°РІР»СЏРµРј СЂРµРєРѕРјРµРЅРґР°С†РёРё
        $profitPct = [math]::Round($stratData.profit_total * 100, 2)
        $winRate = [math]::Round($stratData.wins / $stratData.total_trades * 100, 1)
        $maxDD = [math]::Round([math]::Abs($stratData.max_drawdown_abs), 2)
        
        if ($profitPct -lt 0) {
            $report += "`nвќЊ РљР РРўРР§РќРћ: РЎС‚СЂР°С‚РµРіРёСЏ СѓР±С‹С‚РѕС‡РЅР°! РўСЂРµР±СѓРµС‚СЃСЏ РїРѕР»РЅР°СЏ РїРµСЂРµСЂР°Р±РѕС‚РєР°."
        } elseif ($winRate -lt 50) {
            $report += "`nвљ пёЏ  Win Rate РЅРёР·РєРёР№. Р РµРєРѕРјРµРЅРґСѓРµС‚СЃСЏ СѓР»СѓС‡С€РёС‚СЊ СѓСЃР»РѕРІРёСЏ РІС…РѕРґР°."
        } elseif ($maxDD -gt 200) {
            $report += "`nвљ пёЏ  Drawdown СЃР»РёС€РєРѕРј РІС‹СЃРѕРєРёР№. Р”РѕР±Р°РІСЊС‚Рµ Р·Р°С‰РёС‚РЅС‹Рµ РјРµС…Р°РЅРёР·РјС‹."
        } else {
            $report += "`nвњ… РЎС‚СЂР°С‚РµРіРёСЏ РїРѕРєР°Р·С‹РІР°РµС‚ РїСЂРёРµРјР»РµРјС‹Рµ СЂРµР·СѓР»СЊС‚Р°С‚С‹."
            
            if ($profitPct -gt 10 -and $winRate -gt 55) {
                $report += "`nрџџў РЎС‚СЂР°С‚РµРіРёСЏ РіРѕС‚РѕРІР° Рє С‚РµСЃС‚РёСЂРѕРІР°РЅРёСЋ РІ dry-run СЂРµР¶РёРјРµ."
            }
        }
        
        $report += "`n`nР”Р»СЏ РґРµС‚Р°Р»СЊРЅРѕРіРѕ Р°РЅР°Р»РёР·Р° СЃРј. С„Р°Р№Р»: $($latest.Name)"
        
        # РЎРѕС…СЂР°РЅСЏРµРј РѕС‚С‡РµС‚
        $report | Out-File -FilePath $reportFile -Encoding UTF8
        
        Write-ColorOutput Green "вњ… РћС‚С‡РµС‚ СЃРѕР·РґР°РЅ: $reportFile"
        Write-Host ""
        Write-Host "РћС‚РєСЂС‹С‚СЊ РѕС‚С‡РµС‚? (Enter РґР»СЏ РѕС‚РєСЂС‹С‚РёСЏ, Р»СЋР±Р°СЏ РєР»Р°РІРёС€Р° РґР»СЏ РѕС‚РјРµРЅС‹)"
        $key = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        if ($key.VirtualKeyCode -eq 13) {  # Enter
            notepad $reportFile
        }
        
    } catch {
        Write-ColorOutput Red "вќЊ РћС€РёР±РєР° РїСЂРё СЃРѕР·РґР°РЅРёРё РѕС‚С‡РµС‚Р°: $_"
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
        Write-ColorOutput Red "вќЊ РќРµРёР·РІРµСЃС‚РЅР°СЏ РєРѕРјР°РЅРґР°: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}

Write-Host ""
Write-ColorOutput Cyan "рџЏ›пёЏ  Stoic Citadel - Trade with wisdom, not emotion."
Write-Host ""
