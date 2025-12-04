# ============================================================
# Stoic Citadel - PowerShell Management Script v2.1
# ============================================================
# 
# Usage: .\stoic.ps1 <command> [options]
#        .\stoic.ps1 help
#
# ============================================================

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(Position=1)]
    [string]$Arg1 = "",
    
    [Parameter(Position=2)]
    [string]$Arg2 = ""
)

$ErrorActionPreference = "Stop"

# Auto-detect project directory (where this script is located)
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PROJECT_DIR

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

function Write-Status {
    param(
        [string]$Status,
        [string]$Message
    )
    
    switch ($Status) {
        "OK"      { Write-Host "[OK] " -ForegroundColor Green -NoNewline; Write-Host $Message }
        "ERROR"   { Write-Host "[ERROR] " -ForegroundColor Red -NoNewline; Write-Host $Message }
        "WARN"    { Write-Host "[!] " -ForegroundColor Yellow -NoNewline; Write-Host $Message }
        "INFO"    { Write-Host "[*] " -ForegroundColor Cyan -NoNewline; Write-Host $Message }
        "WAIT"    { Write-Host "[...] " -ForegroundColor Gray -NoNewline; Write-Host $Message }
        default   { Write-Host $Message }
    }
}

function Show-Header {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "        STOIC CITADEL - Trading Bot Management v2.1        " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-DockerRunning {
    try {
        docker info 2>&1 | Out-Null
        return $true
    } catch {
        Write-Status "ERROR" "Docker is not running. Start Docker Desktop first."
        return $false
    }
}

function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Status "WARN" ".env file not found. Creating from template..."
        Copy-Item ".env.example" ".env"
        Write-Status "OK" "Created .env file from template"
        Write-Status "WARN" "IMPORTANT: Edit .env and set your passwords before continuing!"
        Write-Host ""
        Write-Host "Required settings:" -ForegroundColor Yellow
        Write-Host "  - FREQTRADE_API_PASSWORD"
        Write-Host "  - POSTGRES_PASSWORD"
        Write-Host "  - JUPYTER_TOKEN"
        Write-Host ""
        Write-Host "Run: .\stoic.ps1 generate-secrets" -ForegroundColor Cyan
        return $false
    }
    
    # Validate required env vars
    $envContent = Get-Content ".env" -Raw
    $missing = @()
    
    if ($envContent -notmatch "FREQTRADE_API_PASSWORD=.+") { $missing += "FREQTRADE_API_PASSWORD" }
    if ($envContent -notmatch "POSTGRES_PASSWORD=.+") { $missing += "POSTGRES_PASSWORD" }
    if ($envContent -notmatch "JUPYTER_TOKEN=.+") { $missing += "JUPYTER_TOKEN" }
    
    if ($missing.Count -gt 0) {
        Write-Status "ERROR" "Missing required .env variables:"
        foreach ($var in $missing) {
            Write-Host "  - $var" -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "Run: .\stoic.ps1 generate-secrets" -ForegroundColor Cyan
        return $false
    }
    
    return $true
}

function New-SecurePassword {
    param([int]$Length = 32)
    $chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    -join (1..$Length | ForEach-Object { $chars[(Get-Random -Maximum $chars.Length)] })
}

function Initialize-ConfigFile {
    $templatePath = "user_data/config/config.json.template"
    $configPath = "user_data/config/config.json"
    
    if (-not (Test-Path $configPath)) {
        if (Test-Path $templatePath) {
            # Load .env values
            $env = @{}
            if (Test-Path ".env") {
                Get-Content ".env" | ForEach-Object {
                    if ($_ -match "^([^#][^=]+)=(.*)$") {
                        $env[$matches[1].Trim()] = $matches[2].Trim()
                    }
                }
            }
            
            $content = Get-Content $templatePath -Raw
            $content = $content -replace "__JWT_SECRET_KEY__", (New-SecurePassword -Length 64)
            $content = $content -replace "__WS_TOKEN__", (New-SecurePassword -Length 32)
            $content = $content -replace "__API_USERNAME__", ($env["FREQTRADE_API_USERNAME"] ?? "stoic_admin")
            $content = $content -replace "__API_PASSWORD__", ($env["FREQTRADE_API_PASSWORD"] ?? (New-SecurePassword -Length 32))
            
            $content | Set-Content $configPath -Encoding UTF8
            Write-Status "OK" "Generated config.json from template"
        }
    }
}

# ============================================================
# COMMANDS
# ============================================================

function Invoke-GenerateSecrets {
    Show-Header
    Write-Status "INFO" "Generating secure passwords..."
    
    $secrets = @{
        "FREQTRADE_API_PASSWORD" = New-SecurePassword -Length 32
        "POSTGRES_PASSWORD" = New-SecurePassword -Length 32
        "JUPYTER_TOKEN" = New-SecurePassword -Length 24
    }
    
    Write-Host ""
    Write-Host "Generated secrets (copy to .env):" -ForegroundColor Green
    Write-Host ""
    foreach ($key in $secrets.Keys) {
        Write-Host "$key=$($secrets[$key])"
    }
    Write-Host ""
    
    # Optionally update .env directly
    $update = Read-Host "Update .env file automatically? (yes/no)"
    if ($update -eq "yes") {
        if (-not (Test-Path ".env")) {
            Copy-Item ".env.example" ".env"
        }
        
        $envContent = Get-Content ".env" -Raw
        foreach ($key in $secrets.Keys) {
            $pattern = "(?m)^$key=.*$"
            if ($envContent -match $pattern) {
                $envContent = $envContent -replace $pattern, "$key=$($secrets[$key])"
            } else {
                $envContent += "`n$key=$($secrets[$key])"
            }
        }
        $envContent | Set-Content ".env" -Encoding UTF8
        Write-Status "OK" "Updated .env with new secrets"
    }
}

function Invoke-Setup {
    Show-Header
    Write-Status "INFO" "Running initial setup..."
    
    # Check Docker
    if (-not (Test-DockerRunning)) { return }
    
    # Create directories
    $dirs = @(
        "user_data/data/binance",
        "user_data/logs",
        "user_data/backtest_results",
        "user_data/hyperopt_results",
        "user_data/notebooks",
        "research",
        "backups",
        "reports"
    )
    
    foreach ($dir in $dirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    Write-Status "OK" "Directories created"
    
    # Setup .env
    if (-not (Test-EnvFile)) {
        Write-Host ""
        Write-Status "WARN" "Complete .env setup before starting services"
        return
    }
    
    # Initialize config
    Initialize-ConfigFile
    
    Write-Status "OK" "Setup complete!"
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. .\stoic.ps1 download-data"
    Write-Host "  2. .\stoic.ps1 start"
    Write-Host ""
}

function Invoke-Start {
    Show-Header
    Write-Status "INFO" "Starting services..."
    
    if (-not (Test-DockerRunning)) { return }
    if (-not (Test-EnvFile)) { return }
    
    Initialize-ConfigFile
    
    docker compose up -d freqtrade frequi
    
    Write-Host ""
    Write-Status "OK" "Services started!"
    Write-Host ""
    Write-Host "Access:" -ForegroundColor Green
    Write-Host "  Dashboard: http://localhost:3000"
    Write-Host "  API:       http://localhost:8080"
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  .\stoic.ps1 health    - Check status"
    Write-Host "  .\stoic.ps1 logs      - View logs"
    Write-Host "  .\stoic.ps1 stop      - Stop services"
}

function Invoke-StartAll {
    Show-Header
    Write-Status "INFO" "Starting ALL services (including research tools)..."
    
    if (-not (Test-DockerRunning)) { return }
    if (-not (Test-EnvFile)) { return }
    
    Initialize-ConfigFile
    
    docker compose --profile research --profile analytics --profile management up -d
    
    Write-Host ""
    Write-Status "OK" "All services started!"
    Write-Host ""
    Write-Host "Access:" -ForegroundColor Green
    Write-Host "  Dashboard:  http://localhost:3000"
    Write-Host "  Jupyter:    http://localhost:8888"
    Write-Host "  Portainer:  http://localhost:9000"
    Write-Host "  PostgreSQL: localhost:5433"
}

function Invoke-Stop {
    Write-Status "INFO" "Stopping services..."
    docker compose --profile research --profile analytics --profile management down
    Write-Status "OK" "Services stopped"
}

function Invoke-Restart {
    Invoke-Stop
    Start-Sleep -Seconds 2
    Invoke-Start
}

function Invoke-Status {
    Show-Header
    docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
}

function Invoke-Health {
    Show-Header
    Write-Status "INFO" "Checking health..."
    Write-Host ""
    
    $containers = @("stoic_freqtrade", "stoic_frequi", "stoic_postgres", "stoic_jupyter")
    
    foreach ($container in $containers) {
        $running = docker ps --filter "name=$container" --format "{{.Names}}" 2>$null
        
        if ($running) {
            $health = docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}' $container 2>$null
            
            switch ($health) {
                "healthy"  { Write-Status "OK" "$container - HEALTHY" }
                "starting" { Write-Status "WAIT" "$container - STARTING" }
                "running"  { Write-Status "OK" "$container - RUNNING" }
                default    { Write-Status "WARN" "$container - $health" }
            }
        } else {
            Write-Host "[--] " -ForegroundColor DarkGray -NoNewline
            Write-Host "$container - NOT RUNNING" -ForegroundColor DarkGray
        }
    }
    
    Write-Host ""
    Write-Status "INFO" "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" 2>$null
}

function Invoke-Logs {
    $service = if ($Arg1) { $Arg1 } else { "freqtrade" }
    Write-Status "INFO" "Logs for $service (Ctrl+C to exit):"
    docker compose logs -f --tail=100 $service
}

function Invoke-Backtest {
    $strategy = if ($Arg1) { $Arg1 } else { "StoicStrategyV1" }
    $timerange = if ($Arg2) { $Arg2 } else { "20240101-" }
    
    Write-Status "INFO" "Backtesting $strategy..."
    
    docker compose run --rm freqtrade backtesting `
        --strategy $strategy `
        --timerange $timerange `
        --enable-protections
}

function Invoke-DownloadData {
    Write-Status "INFO" "Downloading historical data..."
    
    docker compose run --rm freqtrade download-data `
        --exchange binance `
        --pairs BTC/USDT ETH/USDT BNB/USDT SOL/USDT XRP/USDT `
        --timeframes 5m 15m 1h 4h 1d `
        --days 90
    
    Write-Status "OK" "Data downloaded!"
}

function Invoke-Research {
    Write-Status "INFO" "Starting Jupyter Lab..."
    
    if (-not (Test-DockerRunning)) { return }
    if (-not (Test-EnvFile)) { return }
    
    docker compose --profile research up -d jupyter
    Start-Sleep -Seconds 5
    
    $token = (Get-Content ".env" | Where-Object { $_ -match "^JUPYTER_TOKEN=" }) -replace "JUPYTER_TOKEN=", ""
    
    Write-Status "OK" "Jupyter Lab started!"
    Write-Host ""
    Write-Host "URL: http://localhost:8888" -ForegroundColor Green
    Write-Host "Token: $token" -ForegroundColor Green
    
    Start-Process "http://localhost:8888"
}

function Invoke-Dashboard {
    Start-Process "http://localhost:3000"
    Write-Status "OK" "Opening dashboard..."
}

function Invoke-Clean {
    Write-Status "WARN" "This will stop all containers..."
    $confirm = Read-Host "Continue? (yes/no)"
    
    if ($confirm -eq "yes") {
        docker compose --profile research --profile analytics --profile management down
        Write-Status "OK" "Cleanup complete"
    }
}

function Invoke-Reset {
    Write-Host ""
    Write-Host "!!! WARNING !!!" -ForegroundColor Red
    Write-Host "This will DELETE all data including:" -ForegroundColor Red
    Write-Host "  - Database contents"
    Write-Host "  - Docker volumes"
    Write-Host ""
    
    $confirm = Read-Host "Type 'DELETE ALL DATA' to confirm"
    
    if ($confirm -eq "DELETE ALL DATA") {
        docker compose --profile research --profile analytics --profile management down -v
        docker system prune -f
        Write-Status "OK" "Full reset complete"
        Write-Status "INFO" "Run '.\stoic.ps1 setup' to start fresh"
    }
}

function Show-Help {
    Show-Header
    
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\stoic.ps1 <command> [args]"
    Write-Host ""
    
    Write-Host "SETUP:" -ForegroundColor Yellow
    Write-Host "  setup            Initial setup wizard"
    Write-Host "  generate-secrets Generate secure passwords"
    Write-Host ""
    
    Write-Host "SERVICES:" -ForegroundColor Yellow
    Write-Host "  start            Start trading services"
    Write-Host "  start-all        Start all services (incl. research)"
    Write-Host "  stop             Stop all services"
    Write-Host "  restart          Restart services"
    Write-Host "  status           Show service status"
    Write-Host ""
    
    Write-Host "MONITORING:" -ForegroundColor Yellow
    Write-Host "  health           Health check"
    Write-Host "  logs [service]   View logs (default: freqtrade)"
    Write-Host "  dashboard        Open FreqUI"
    Write-Host ""
    
    Write-Host "TRADING:" -ForegroundColor Yellow
    Write-Host "  backtest [strat] Run backtest"
    Write-Host "  download-data    Download historical data"
    Write-Host ""
    
    Write-Host "RESEARCH:" -ForegroundColor Yellow
    Write-Host "  research         Start Jupyter Lab"
    Write-Host ""
    
    Write-Host "MAINTENANCE:" -ForegroundColor Yellow
    Write-Host "  clean            Stop and remove containers"
    Write-Host "  reset            Full reset (DELETES DATA!)"
    Write-Host ""
    
    Write-Host "EXAMPLES:" -ForegroundColor Cyan
    Write-Host "  .\stoic.ps1 setup"
    Write-Host "  .\stoic.ps1 start"
    Write-Host "  .\stoic.ps1 logs freqtrade"
    Write-Host "  .\stoic.ps1 backtest StoicStrategyV1"
    Write-Host ""
}

# ============================================================
# MAIN
# ============================================================

switch ($Command.ToLower()) {
    "help"             { Show-Help }
    "setup"            { Invoke-Setup }
    "generate-secrets" { Invoke-GenerateSecrets }
    
    "start"            { Invoke-Start }
    "start-all"        { Invoke-StartAll }
    "stop"             { Invoke-Stop }
    "restart"          { Invoke-Restart }
    "status"           { Invoke-Status }
    
    "health"           { Invoke-Health }
    "logs"             { Invoke-Logs }
    "dashboard"        { Invoke-Dashboard }
    
    "backtest"         { Invoke-Backtest }
    "download-data"    { Invoke-DownloadData }
    
    "research"         { Invoke-Research }
    
    "clean"            { Invoke-Clean }
    "reset"            { Invoke-Reset }
    
    default {
        Write-Status "ERROR" "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}

Write-Host ""
Write-Host "Stoic Citadel - Trade with wisdom." -ForegroundColor DarkGray
Write-Host ""
