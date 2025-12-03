# Stoic Citadel - PowerShell Management Script
# Clean version without emoji - Windows compatible

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

function Write-ColorOutput($ForegroundColor, $Message) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    Write-Output $Message
    $host.UI.RawUI.ForegroundColor = $fc
}

function Show-Header {
    Write-Host ""
    Write-ColorOutput "Cyan" "============================================================"
    Write-ColorOutput "Cyan" "            STOIC CITADEL - TRADING BOT v2.0            "
    Write-ColorOutput "Cyan" "============================================================"
    Write-Host ""
}

function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-ColorOutput "Yellow" "[!] .env file not found. Creating from template..."
        Copy-Item ".env.example" ".env"
        Write-ColorOutput "Green" "[OK] Created .env file"
        Write-ColorOutput "Yellow" "[!] IMPORTANT: Configure .env before continuing!"
        return $false
    }
    return $true
}

function New-SecurePassword {
    param([int]$Length = 32)
    $chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    $password = ""
    for ($i = 0; $i -lt $Length; $i++) {
        $password += $chars[(Get-Random -Minimum 0 -Maximum $chars.Length)]
    }
    return $password
}

function Invoke-GenerateSecrets {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Generating secure passwords..."
    
    $freqtradePass = New-SecurePassword -Length 32
    $postgresPass = New-SecurePassword -Length 32
    
    Write-ColorOutput "Green" "[OK] Passwords generated!"
    Write-Host ""
    Write-Host "Copy these values to your .env file:"
    Write-Host ""
    Write-Host "FREQTRADE_API_PASSWORD=$freqtradePass"
    Write-Host "POSTGRES_PASSWORD=$postgresPass"
    Write-Host ""
    
    $envContent = "FREQTRADE_API_PASSWORD=$freqtradePass`nPOSTGRES_PASSWORD=$postgresPass`n"
    $envContent | Out-File -FilePath ".env.generated" -Encoding ASCII
    
    Write-ColorOutput "Green" "[OK] Passwords saved to .env.generated"
    Write-ColorOutput "Yellow" "[!] Copy them to .env manually!"
}

function Invoke-HealthCheck {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Checking container health..."
    Write-Host ""
    
    Set-Location $PROJECT_DIR
    
    try {
        docker ps | Out-Null
        Write-ColorOutput "Green" "[OK] Docker is running"
    } catch {
        Write-ColorOutput "Red" "[ERROR] Docker not running"
        return
    }
    
    $containers = @("stoic_freqtrade", "stoic_frequi", "stoic_postgres", "stoic_jupyter")
    
    foreach ($container in $containers) {
        $status = docker inspect -f '{{.State.Health.Status}}' $container 2>$null
        if ($status -eq "healthy") {
            Write-ColorOutput "Green" "[OK] $container - HEALTHY"
        } elseif ($status -eq "starting") {
            Write-ColorOutput "Yellow" "[WAIT] $container - STARTING"
        } else {
            Write-ColorOutput "Red" "[ERROR] $container - UNHEALTHY or NOT RUNNING"
        }
    }
    
    Write-Host ""
    Write-ColorOutput "Cyan" "[*] Resource usage:"
    docker stats --no-stream --format "table {{.Container}}`t{{.CPUPerc}}`t{{.MemUsage}}" $containers
}

function Invoke-WatchHealth {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Continuous monitoring (Ctrl+C to exit)..."
    
    while ($true) {
        Clear-Host
        Invoke-HealthCheck
        Start-Sleep -Seconds 10
    }
}

function Invoke-Setup {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Running Stoic Citadel setup wizard..."
    
    Set-Location $PROJECT_DIR
    
    Write-ColorOutput "Cyan" "[*] Checking Docker..."
    try {
        docker --version | Out-Null
        docker-compose --version | Out-Null
        Write-ColorOutput "Green" "[OK] Docker installed"
    } catch {
        Write-ColorOutput "Red" "[ERROR] Docker not found. Install Docker Desktop"
        exit 1
    }
    
    Test-EnvFile | Out-Null
    
    Write-ColorOutput "Cyan" "[*] Creating directories..."
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
    Write-ColorOutput "Green" "[OK] Directories created"
    
    Write-ColorOutput "Green" "[OK] Setup complete!"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1. .\stoic.ps1 generate-secrets"
    Write-Host "  2. .\stoic.ps1 download-data"
    Write-Host "  3. .\stoic.ps1 trade-dry"
    Write-Host ""
}

function Invoke-Start {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Starting Stoic Citadel services..."
    
    Set-Location $PROJECT_DIR
    if (-not (Test-EnvFile)) { return }
    
    docker-compose up -d
    Start-Sleep -Seconds 5
    
    Write-ColorOutput "Green" "[OK] All services started!"
    Write-Host ""
    Write-Host "Access points:"
    Write-Host "  FreqUI:     http://localhost:3000"
    Write-Host "  Jupyter:    http://localhost:8888 (token: stoic2024)"
    Write-Host "  Portainer:  http://localhost:9000"
    Write-Host "  PostgreSQL: localhost:5433"
    Write-Host ""
}

function Invoke-Stop {
    Write-ColorOutput "Yellow" "[*] Stopping all services..."
    Set-Location $PROJECT_DIR
    docker-compose down
    Write-ColorOutput "Green" "[OK] All services stopped"
}

function Invoke-Restart {
    Write-ColorOutput "Cyan" "[*] Restarting services..."
    Invoke-Stop
    Start-Sleep -Seconds 2
    Invoke-Start
}

function Invoke-Status {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Service status:"
    Write-Host ""
    Set-Location $PROJECT_DIR
    docker-compose ps
}

function Invoke-Logs {
    Write-ColorOutput "Cyan" "[*] Logs for $Service (Ctrl+C to exit):"
    Write-Host ""
    Set-Location $PROJECT_DIR
    docker-compose logs -f --tail=100 $Service
}

function Invoke-TradeDry {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Starting trading bot in DRY-RUN mode..."
    
    Set-Location $PROJECT_DIR
    if (-not (Test-EnvFile)) { return }
    
    docker-compose up -d freqtrade frequi postgres
    Start-Sleep -Seconds 5
    
    Write-ColorOutput "Green" "[OK] Trading bot started (dry-run mode)"
    Write-Host ""
    Write-Host "Monitoring:"
    Write-Host "  Dashboard: http://localhost:3000"
    Write-Host "  Logs:      .\stoic.ps1 logs freqtrade"
    Write-Host "  Health:    .\stoic.ps1 health"
    Write-Host ""
}

function Invoke-TradeLive {
    Show-Header
    Write-ColorOutput "Red" "============================================================"
    Write-ColorOutput "Red" "              [!] LIVE TRADING MODE [!]"
    Write-ColorOutput "Red" "                THIS WILL USE REAL MONEY!"
    Write-ColorOutput "Red" "============================================================"
    Write-Host ""
    
    $confirm = Read-Host "Type 'I UNDERSTAND THE RISKS' to continue"
    if ($confirm -ne "I UNDERSTAND THE RISKS") {
        Write-ColorOutput "Yellow" "[!] Live trading cancelled"
        return
    }
    
    Set-Location $PROJECT_DIR
    if (-not (Test-EnvFile)) { return }
    
    docker-compose up -d freqtrade frequi postgres
    
    Write-ColorOutput "Green" "[OK] Live trading started!"
    Write-ColorOutput "Red" "[!] MONITOR CONSTANTLY!"
}

function Invoke-Backtest {
    Write-ColorOutput "Cyan" "[*] Running backtest for strategy: $Strategy"
    Set-Location $PROJECT_DIR
    
    docker-compose run --rm freqtrade backtesting `
        --strategy $Strategy `
        --timerange 20240101- `
        --enable-protections
    
    Write-ColorOutput "Green" "[OK] Backtest complete!"
}

function Invoke-DownloadData {
    Write-ColorOutput "Cyan" "[*] Downloading historical data..."
    Set-Location $PROJECT_DIR
    
    docker-compose run --rm freqtrade download-data `
        --exchange binance `
        --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT ADA/USDT `
        --timeframes 5m 15m 1h `
        --days 90
    
    Write-ColorOutput "Green" "[OK] Data downloaded!"
}

function Invoke-Dashboard {
    Write-ColorOutput "Cyan" "[*] Opening FreqUI Dashboard..."
    Start-Process "http://localhost:3000"
    Write-ColorOutput "Green" "[OK] Dashboard opened"
}

function Invoke-Research {
    Show-Header
    Write-ColorOutput "Cyan" "[*] Starting Jupyter Lab..."
    
    Set-Location $PROJECT_DIR
    docker-compose up -d jupyter
    Start-Sleep -Seconds 5
    
    Write-ColorOutput "Green" "[OK] Jupyter Lab started!"
    Write-Host ""
    Write-Host "URL: http://localhost:8888"
    Write-Host "Token: stoic2024"
    
    Start-Process "http://localhost:8888"
}

function Invoke-Clean {
    Write-ColorOutput "Yellow" "[!] This will remove all containers..."
    $confirm = Read-Host "Continue? (yes/no)"
    
    if ($confirm -eq "yes") {
        Write-ColorOutput "Cyan" "[*] Cleaning..."
        Set-Location $PROJECT_DIR
        docker-compose down
        Write-ColorOutput "Green" "[OK] Cleanup complete"
    }
}

function Show-Help {
    Show-Header
    Write-ColorOutput "Green" "[*] AVAILABLE COMMANDS:"
    Write-Host ""
    Write-ColorOutput "Yellow" "Management:"
    Write-Host "  help              - Show this help"
    Write-Host "  setup             - Initial setup"
    Write-Host "  start             - Start all services"
    Write-Host "  stop              - Stop services"
    Write-Host "  restart           - Restart services"
    Write-Host "  status            - Service status"
    Write-Host "  logs [service]    - Show logs"
    Write-Host ""
    Write-ColorOutput "Yellow" "Security:"
    Write-Host "  generate-secrets  - Generate passwords"
    Write-Host ""
    Write-ColorOutput "Yellow" "Monitoring:"
    Write-Host "  health            - Health check"
    Write-Host "  health-watch      - Continuous monitoring"
    Write-Host "  dashboard         - Open dashboard"
    Write-Host ""
    Write-ColorOutput "Yellow" "Trading:"
    Write-Host "  trade-dry         - Paper trading"
    Write-Host "  trade-live        - Live trading (CAUTION!)"
    Write-Host "  backtest [strat]  - Backtest"
    Write-Host ""
    Write-ColorOutput "Yellow" "Data:"
    Write-Host "  download-data     - Download historical data"
    Write-Host "  research          - Start Jupyter"
    Write-Host ""
    Write-ColorOutput "Yellow" "Maintenance:"
    Write-Host "  clean             - Clean containers"
    Write-Host ""
}

Set-Location $PROJECT_DIR

switch ($Command.ToLower()) {
    "help" { Show-Help }
    "setup" { Invoke-Setup }
    "start" { Invoke-Start }
    "stop" { Invoke-Stop }
    "restart" { Invoke-Restart }
    "status" { Invoke-Status }
    "logs" { Invoke-Logs }
    
    "generate-secrets" { Invoke-GenerateSecrets }
    
    "health" { Invoke-HealthCheck }
    "health-watch" { Invoke-WatchHealth }
    "dashboard" { Invoke-Dashboard }
    
    "trade-dry" { Invoke-TradeDry }
    "trade-live" { Invoke-TradeLive }
    "backtest" { Invoke-Backtest }
    
    "download-data" { Invoke-DownloadData }
    "research" { Invoke-Research }
    
    "clean" { Invoke-Clean }
    
    default {
        Write-ColorOutput "Red" "[ERROR] Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}

Write-Host ""
Write-ColorOutput "Cyan" "Stoic Citadel - Trade with wisdom, not emotion."
Write-Host ""
