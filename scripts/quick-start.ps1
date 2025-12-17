# Stoic Citadel - Quick Start Script
# Automated setup for Windows

param(
    [switch]$Clean = $false,
    [switch]$Build = $false
)

$ErrorActionPreference = "Stop"

Write-Host "рџЏ›пёЏ  STOIC CITADEL - QUICK START" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "рџђі Checking Docker..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "вњ… Docker is running" -ForegroundColor Green
} catch {
    Write-Host "вќЊ Docker is not running!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "вљ пёЏ  .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "вњ… Created .env file" -ForegroundColor Green
    Write-Host "вљ пёЏ  IMPORTANT: Edit .env and change passwords!" -ForegroundColor Red
    Write-Host ""
    Start-Sleep -Seconds 2
}

# Clean if requested
if ($Clean) {
    Write-Host "рџ§№ Cleaning up old containers..." -ForegroundColor Yellow
    docker-compose down -v 2>$null
    docker system prune -f 2>$null
    Write-Host "вњ… Cleanup complete" -ForegroundColor Green
}

# Build if requested or if images don't exist
if ($Build -or $Clean) {
    Write-Host "рџ”Ё Building Docker images (this may take 10-15 minutes)..." -ForegroundColor Yellow
    Write-Host "в• Time for coffee!" -ForegroundColor Cyan
    docker-compose build --no-cache
    Write-Host "вњ… Build complete" -ForegroundColor Green
}

# Start services
Write-Host "рџљЂ Starting services..." -ForegroundColor Yellow
docker-compose up -d

# Wait for services to be ready
Write-Host "вЏі Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check service status
Write-Host ""
Write-Host "рџ“Љ Service Status:" -ForegroundColor Cyan
docker-compose ps

Write-Host ""
Write-Host "рџЋ‰ STOIC CITADEL IS READY!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "рџ“± Access your services:" -ForegroundColor Cyan
Write-Host "  вЂў Dashboard:  http://localhost:3000" -ForegroundColor White
Write-Host "  вЂў Jupyter:    http://localhost:8888 (token: stoic2024)" -ForegroundColor White
Write-Host "  вЂў Portainer:  http://localhost:9000" -ForegroundColor White
Write-Host ""
Write-Host "рџ“љ Quick commands:" -ForegroundColor Cyan
Write-Host "  вЂў Logs:       docker-compose logs -f freqtrade" -ForegroundColor White
Write-Host "  вЂў Stop:       docker-compose down" -ForegroundColor White
Write-Host "  вЂў Restart:    docker-compose restart" -ForegroundColor White
Write-Host ""
Write-Host "рџ“– Read LAUNCH_INSTRUCTIONS.md for detailed setup" -ForegroundColor Yellow