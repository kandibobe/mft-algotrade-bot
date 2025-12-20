# ==============================================================================
# HEALTH CHECK - РџСЂРѕРІРµСЂРєР° Р·РґРѕСЂРѕРІСЊСЏ СЃРёСЃС‚РµРјС‹
# ==============================================================================

$PROJECT_DIR = "C:\mft-algotrade-bot"

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
    Write-ColorOutput Cyan "в•‘            STOIC CITADEL - HEALTH CHECK                    в•‘"
    Write-ColorOutput Cyan "в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ"
    Write-Host ""
}

Show-Header
Set-Location $PROJECT_DIR

$allGood = $true

# РџСЂРѕРІРµСЂРєР° Docker
Write-ColorOutput Cyan "рџђі РџСЂРѕРІРµСЂРєР° Docker..."
try {
    $dockerVersion = docker --version
    Write-ColorOutput Green "  вњ… Docker: $dockerVersion"
} catch {
    Write-ColorOutput Red "  вќЊ Docker РЅРµ РЅР°Р№РґРµРЅ РёР»Рё РЅРµ Р·Р°РїСѓС‰РµРЅ"
    $allGood = $false
}

try {
    $composeVersion = docker-compose --version
    Write-ColorOutput Green "  вњ… Docker Compose: $composeVersion"
} catch {
    Write-ColorOutput Red "  вќЊ Docker Compose РЅРµ РЅР°Р№РґРµРЅ"
    $allGood = $false
}

# РџСЂРѕРІРµСЂРєР° .env С„Р°Р№Р»Р°
Write-Host ""
Write-ColorOutput Cyan "рџ“ќ РџСЂРѕРІРµСЂРєР° .env С„Р°Р№Р»Р°..."
if (Test-Path ".env") {
    Write-ColorOutput Green "  вњ… .env С„Р°Р№Р» СЃСѓС‰РµСЃС‚РІСѓРµС‚"
    
    $envContent = Get-Content ".env" -Raw
    
    # РџСЂРѕРІРµСЂРєР° РєСЂРёС‚РёС‡РЅС‹С… РїРµСЂРµРјРµРЅРЅС‹С…
    $required = @(
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "FREQTRADE_API_PASSWORD"
    )
    
    foreach ($var in $required) {
        if ($envContent -match "$var\s*=\s*\S+") {
            Write-ColorOutput Green "  вњ… $var РЅР°СЃС‚СЂРѕРµРЅ"
        } else {
            Write-ColorOutput Red "  вќЊ $var РЅРµ РЅР°СЃС‚СЂРѕРµРЅ РёР»Рё РїСѓСЃС‚РѕР№"
            $allGood = $false
        }
    }
    
    # РџСЂРѕРІРµСЂРєР° DRY_RUN
    if ($envContent -match "DRY_RUN\s*=\s*true") {
        Write-ColorOutput Green "  вњ… DRY_RUN=true (Р±РµР·РѕРїР°СЃРЅРѕ)"
    } elseif ($envContent -match "DRY_RUN\s*=\s*false") {
        Write-ColorOutput Yellow "  вљ пёЏ  DRY_RUN=false (LIVE TRADING!)"
    }
    
} else {
    Write-ColorOutput Red "  вќЊ .env С„Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ"
    $allGood = $false
}

# РџСЂРѕРІРµСЂРєР° РґРёСЂРµРєС‚РѕСЂРёР№
Write-Host ""
Write-ColorOutput Cyan "рџ“Ѓ РџСЂРѕРІРµСЂРєР° СЃС‚СЂСѓРєС‚СѓСЂС‹ РґРёСЂРµРєС‚РѕСЂРёР№..."
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
        Write-ColorOutput Green "  вњ… $dir"
    } else {
        Write-ColorOutput Yellow "  вљ пёЏ  $dir РѕС‚СЃСѓС‚СЃС‚РІСѓРµС‚ (Р±СѓРґРµС‚ СЃРѕР·РґР°РЅР° РїСЂРё setup)"
    }
}

# РџСЂРѕРІРµСЂРєР° СЃС‚СЂР°С‚РµРіРёР№
Write-Host ""
Write-ColorOutput Cyan "рџЋЇ РџСЂРѕРІРµСЂРєР° СЃС‚СЂР°С‚РµРіРёР№..."
$strategies = Get-ChildItem "user_data\strategies\*.py" -ErrorAction SilentlyContinue

if ($strategies) {
    Write-ColorOutput Green "  вњ… РќР°Р№РґРµРЅРѕ $($strategies.Count) СЃС‚СЂР°С‚РµРіРёР№:"
    foreach ($strat in $strategies) {
        Write-Host "     - $($strat.BaseName)"
    }
} else {
    Write-ColorOutput Red "  вќЊ РЎС‚СЂР°С‚РµРіРёРё РЅРµ РЅР°Р№РґРµРЅС‹"
    $allGood = $false
}

# РџСЂРѕРІРµСЂРєР° Docker РєРѕРЅС‚РµР№РЅРµСЂРѕРІ
Write-Host ""
Write-ColorOutput Cyan "рџђі РџСЂРѕРІРµСЂРєР° Docker РєРѕРЅС‚РµР№РЅРµСЂРѕРІ..."
try {
    $containers = docker-compose ps --format json 2>$null | ConvertFrom-Json
    
    if ($containers) {
        Write-ColorOutput Green "  вњ… РќР°Р№РґРµРЅРѕ $($containers.Count) РєРѕРЅС‚РµР№РЅРµСЂРѕРІ:"
        foreach ($container in $containers) {
            $name = $container.Service
            $status = $container.State
            
            if ($status -eq "running") {
                Write-ColorOutput Green "     вњ… $name - running"
            } else {
                Write-ColorOutput Yellow "     вљ пёЏ  $name - $status"
            }
        }
    } else {
        Write-ColorOutput Yellow "  вљ пёЏ  РљРѕРЅС‚РµР№РЅРµСЂС‹ РЅРµ Р·Р°РїСѓС‰РµРЅС‹"
    }
} catch {
    Write-ColorOutput Yellow "  вљ пёЏ  РќРµ СѓРґР°Р»РѕСЃСЊ РїСЂРѕРІРµСЂРёС‚СЊ РєРѕРЅС‚РµР№РЅРµСЂС‹"
}

# РџСЂРѕРІРµСЂРєР° РґР°РЅРЅС‹С…
Write-Host ""
Write-ColorOutput Cyan "рџ“Љ РџСЂРѕРІРµСЂРєР° РґР°РЅРЅС‹С…..."
$dataFiles = Get-ChildItem "user_data\data\binance\*.json" -ErrorAction SilentlyContinue -Recurse

if ($dataFiles) {
    Write-ColorOutput Green "  вњ… РќР°Р№РґРµРЅРѕ $($dataFiles.Count) С„Р°Р№Р»РѕРІ РґР°РЅРЅС‹С…"
    
    # РџСЂРѕРІРµСЂРєР° СЃРІРµР¶РµСЃС‚Рё РґР°РЅРЅС‹С…
    $newest = $dataFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    $age = (Get-Date) - $newest.LastWriteTime
    
    if ($age.Days -lt 1) {
        Write-ColorOutput Green "  вњ… Р”Р°РЅРЅС‹Рµ СЃРІРµР¶РёРµ (РѕР±РЅРѕРІР»РµРЅС‹ СЃРµРіРѕРґРЅСЏ)"
    } elseif ($age.Days -lt 7) {
        Write-ColorOutput Yellow "  вљ пёЏ  Р”Р°РЅРЅС‹Рµ СѓСЃС‚Р°СЂРµР»Рё ($($age.Days) РґРЅРµР№)"
    } else {
        Write-ColorOutput Red "  вќЊ Р”Р°РЅРЅС‹Рµ РѕС‡РµРЅСЊ СЃС‚Р°СЂС‹Рµ ($($age.Days) РґРЅРµР№)"
        Write-Host "     Р—Р°РїСѓСЃС‚Рё: .\stoic.ps1 download-data"
    }
} else {
    Write-ColorOutput Yellow "  вљ пёЏ  Р”Р°РЅРЅС‹Рµ РЅРµ РЅР°Р№РґРµРЅС‹"
    Write-Host "     Р—Р°РїСѓСЃС‚Рё: .\stoic.ps1 download-data"
}

# РџСЂРѕРІРµСЂРєР° Р±Р°Р·С‹ РґР°РЅРЅС‹С…
Write-Host ""
Write-ColorOutput Cyan "рџ’ѕ РџСЂРѕРІРµСЂРєР° Р±Р°Р·С‹ РґР°РЅРЅС‹С…..."
if (Test-Path "user_data\tradesv3.sqlite") {
    $dbSize = (Get-Item "user_data\tradesv3.sqlite").Length / 1MB
    Write-ColorOutput Green "  вњ… Р‘Р°Р·Р° РґР°РЅРЅС‹С… РЅР°Р№РґРµРЅР° ($('{0:N2}' -f $dbSize) MB)"
} else {
    Write-ColorOutput Yellow "  вљ пёЏ  Р‘Р°Р·Р° РґР°РЅРЅС‹С… РЅРµ РЅР°Р№РґРµРЅР° (СЃРѕР·РґР°СЃС‚СЃСЏ РїСЂРё РїРµСЂРІРѕРј Р·Р°РїСѓСЃРєРµ)"
}

# РџСЂРѕРІРµСЂРєР° РїРѕСЂС‚РѕРІ
Write-Host ""
Write-ColorOutput Cyan "рџЊђ РџСЂРѕРІРµСЂРєР° РґРѕСЃС‚СѓРїРЅРѕСЃС‚Рё РїРѕСЂС‚РѕРІ..."
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
            Write-ColorOutput Green "  вњ… Port $port ($($ports[$port])) - РѕС‚РєСЂС‹С‚"
        } else {
            Write-ColorOutput Yellow "  вљ пёЏ  Port $port ($($ports[$port])) - Р·Р°РєСЂС‹С‚"
        }
    } catch {
        Write-ColorOutput Yellow "  вљ пёЏ  Port $port ($($ports[$port])) - РЅРµРґРѕСЃС‚СѓРїРµРЅ"
    }
}

# РС‚РѕРіРѕРІС‹Р№ СЂРµР·СѓР»СЊС‚Р°С‚
Write-Host ""
Write-Host "в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ"
if ($allGood) {
    Write-ColorOutput Green "вњ… Р’РЎР• РџР РћР’Р•Р РљР РџР РћР™Р”Р•РќР«!"
    Write-Host ""
    Write-ColorOutput Cyan "рџљЂ Р“РѕС‚РѕРІРѕ Рє Р·Р°РїСѓСЃРєСѓ:"
    Write-Host "   .\stoic.ps1 trade-dry    # Paper trading"
    Write-Host "   .\stoic.ps1 dashboard    # РћС‚РєСЂС‹С‚СЊ dashboard"
} else {
    Write-ColorOutput Yellow "вљ пёЏ  РќР•РљРћРўРћР Р«Р• РџР РћР’Р•Р РљР РќР• РџР РћРЁР›Р"
    Write-Host ""
    Write-ColorOutput Cyan "рџ”§ Р РµРєРѕРјРµРЅРґР°С†РёРё:"
    Write-Host "   1. РџСЂРѕРІРµСЂСЊС‚Рµ Docker Desktop (РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ Р·Р°РїСѓС‰РµРЅ)"
    Write-Host "   2. Р—Р°РїРѕР»РЅРёС‚Рµ .env С„Р°Р№Р»"
    Write-Host "   3. Р—Р°РїСѓСЃС‚РёС‚Рµ: .\stoic.ps1 setup"
}
Write-Host "в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђ"
Write-Host ""

Write-ColorOutput Cyan "рџЏ›пёЏ  Stoic Citadel - Trade with wisdom, not emotion"
Write-Host ""
