#!/usr/bin/env pwsh
# Stoic Citadel - Logs Viewer Script for Windows
# Просмотр и анализ логов

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("freqtrade", "frequi", "jupyter", "postgres", "portainer", "all")]
    [string]$Service = "freqtrade",
    
    [Parameter(Mandatory=$false)]
    [int]$Lines = 50,
    
    [Parameter(Mandatory=$false)]
    [switch]$Follow,
    
    [Parameter(Mandatory=$false)]
    [switch]$Timestamps,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("ERROR", "WARNING", "INFO", "DEBUG")]
    [string]$Level = "",
    
    [Parameter(Mandatory=$false)]
    [string]$Search = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$FileLog,
    
    [Parameter(Mandatory=$false)]
    [switch]$Export
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

# Баннер
Write-Host @"

╔════════════════════════════════════════════════════════╗
║           STOIC CITADEL - LOGS VIEWER                  ║
╚════════════════════════════════════════════════════════╝

"@ -ForegroundColor Magenta

# Просмотр файловых логов
if ($FileLog) {
    Write-Step "Просмотр файловых логов Freqtrade"
    
    $logFile = ".\user_data\logs\freqtrade.log"
    
    if (-not (Test-Path $logFile)) {
        Write-ColorOutput "❌ Лог файл не найден: $logFile" "Red"
        Write-ColorOutput "   Убедитесь что Freqtrade запущен и создал лог файл" "Yellow"
        exit 1
    }
    
    $logContent = Get-Content $logFile -Tail $Lines
    
    # Фильтр по уровню
    if ($Level) {
        $logContent = $logContent | Select-String $Level
        Write-ColorOutput "Фильтр по уровню: $Level" "Yellow"
    }
    
    # Поиск
    if ($Search) {
        $logContent = $logContent | Select-String $Search
        Write-ColorOutput "Поиск: $Search" "Yellow"
    }
    
    Write-ColorOutput "`nПоследние $Lines строк из $logFile`n" "Cyan"
    $logContent | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "ERROR") {
            Write-ColorOutput $line "Red"
        } elseif ($line -match "WARNING") {
            Write-ColorOutput $line "Yellow"
        } elseif ($line -match "INFO") {
            Write-ColorOutput $line "White"
        } else {
            Write-ColorOutput $line "Gray"
        }
    }
    
    # Экспорт
    if ($Export) {
        $exportFile = "logs_export_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        $logContent | Out-File $exportFile
        Write-ColorOutput "`n✅ Логи экспортированы в: $exportFile" "Green"
    }
    
    exit 0
}

# Docker логи
Write-Step "Просмотр Docker логов: $Service"

# Формирование команды
$command = @("docker-compose", "logs")

if ($Follow) {
    $command += "-f"
    Write-ColorOutput "Режим: Следование за логами (Ctrl+C для выхода)" "Yellow"
} else {
    $command += "--tail=$Lines"
    Write-ColorOutput "Показать последних строк: $Lines" "Yellow"
}

if ($Timestamps) {
    $command += "--timestamps"
}

if ($Service -ne "all") {
    $command += $Service
    Write-ColorOutput "Сервис: $Service" "Cyan"
} else {
    Write-ColorOutput "Сервисы: ВСЕ" "Cyan"
}

Write-Host ""

# Запуск без фильтров
if ([string]::IsNullOrEmpty($Level) -and [string]::IsNullOrEmpty($Search)) {
    & $command[0] $command[1..($command.Length-1)]
    exit 0
}

# Запуск с фильтрами
if ($Level -or $Search) {
    Write-ColorOutput "Применение фильтров..." "Yellow"
    if ($Level) { Write-ColorOutput "  Уровень: $Level" "Gray" }
    if ($Search) { Write-ColorOutput "  Поиск: $Search" "Gray" }
    Write-Host ""
    
    $logs = & $command[0] $command[1..($command.Length-1)] 2>&1
    
    # Фильтрация
    $filtered = $logs
    if ($Level) {
        $filtered = $filtered | Select-String $Level
    }
    if ($Search) {
        $filtered = $filtered | Select-String $Search
    }
    
    # Цветной вывод
    $filtered | ForEach-Object {
        $line = $_.ToString()
        if ($line -match "ERROR") {
            Write-ColorOutput $line "Red"
        } elseif ($line -match "WARNING") {
            Write-ColorOutput $line "Yellow"
        } elseif ($line -match "INFO") {
            Write-ColorOutput $line "White"
        } else {
            Write-ColorOutput $line "Gray"
        }
    }
    
    # Экспорт
    if ($Export) {
        $exportFile = "logs_${Service}_export_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
        $filtered | Out-File $exportFile
        Write-ColorOutput "`n✅ Отфильтрованные логи экспортированы в: $exportFile" "Green"
    }
}

# Справка
if (-not $Follow) {
    Write-Host @"

╔════════════════════════════════════════════════════════╗
║                  ПОЛЕЗНЫЕ КОМАНДЫ                      ║
╚════════════════════════════════════════════════════════╝

Следить за логами в реальном времени:
  .\scripts\windows\logs.ps1 -Service freqtrade -Follow

Показать только ошибки:
  .\scripts\windows\logs.ps1 -Level ERROR -Lines 100

Поиск по тексту:
  .\scripts\windows\logs.ps1 -Search "Strategy" -Lines 200

Файловые логи Freqtrade:
  .\scripts\windows\logs.ps1 -FileLog -Lines 100

Экспорт логов:
  .\scripts\windows\logs.ps1 -Level ERROR -Export

Все сервисы:
  .\scripts\windows\logs.ps1 -Service all -Lines 30

Прямые команды Docker:
  docker-compose logs -f freqtrade
  docker-compose logs --tail=100 freqtrade
  docker-compose ps

"@ -ForegroundColor Gray
}
