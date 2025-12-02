# Stoic Citadel - Log Viewer Script for Windows
# Удобный просмотр логов с фильтрацией

param(
    [string]$Service = "freqtrade",
    [int]$Lines = 100,
    [switch]$Follow = $false,
    [ValidateSet("ALL", "ERROR", "WARNING", "INFO")]
    [string]$Level = "ALL"
)

$ErrorActionPreference = "Stop"

Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Stoic Citadel - Log Viewer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service:     $Service" -ForegroundColor White
Write-Host "Lines:       $Lines" -ForegroundColor White
Write-Host "Level:       $Level" -ForegroundColor White
Write-Host "Follow:      $Follow" -ForegroundColor White
Write-Host "" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($Follow) {
    Write-Host "📋 Following logs in real-time (Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host ""
    
    if ($Level -eq "ALL") {
        docker-compose logs -f --tail=$Lines $Service
    } else {
        docker-compose logs -f --tail=$Lines $Service | Select-String $Level
    }
} else {
    Write-Host "📋 Showing last $Lines lines..." -ForegroundColor Yellow
    Write-Host ""
    
    if ($Level -eq "ALL") {
        docker-compose logs --tail=$Lines $Service
    } else {
        docker-compose logs --tail=$Lines $Service | Select-String $Level
    }
}

Write-Host ""
Write-Host "Log file location: user_data/logs/freqtrade.log" -ForegroundColor Gray
Write-Host ""
