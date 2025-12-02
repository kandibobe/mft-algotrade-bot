# ==============================================================================
# DOCS NAVIGATOR - Быстрый доступ к документации
# ==============================================================================

param(
    [Parameter(Position=0)]
    [string]$Doc = "menu"
)

$PROJECT_DIR = "C:\hft-algotrade-bot"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Show-Menu {
    Write-Host ""
    Write-ColorOutput Cyan "╔════════════════════════════════════════════════════════════╗"
    Write-ColorOutput Cyan "║            STOIC CITADEL - ДОКУМЕНТАЦИЯ                    ║"
    Write-ColorOutput Cyan "╚════════════════════════════════════════════════════════════╝"
    Write-Host ""
    
    Write-ColorOutput Green "📚 ГЛАВНЫЕ ДОКУМЕНТЫ (читай по порядку):"
    Write-Host ""
    Write-Host "  1. start          - START_HERE.md (начни здесь!)"
    Write-Host "  2. todo           - TODO_FOR_YOU.md (что делать сейчас)"
    Write-Host "  3. how            - HOW_TO_USE.md (как пользоваться)"
    Write-Host "  4. plan           - DEVELOPMENT_PLAN.md (план на 6 недель)"
    Write-Host ""
    
    Write-ColorOutput Green "📝 РАБОЧИЕ ДОКУМЕНТЫ:"
    Write-Host ""
    Write-Host "  5. journal        - TRADING_JOURNAL.md (дневник наблюдений)"
    Write-Host "  6. checklist      - CHECKLIST.md (чеклист запуска)"
    Write-Host "  7. quick          - QUICKSTART_WINDOWS.md (детальное руководство)"
    Write-Host ""
    
    Write-ColorOutput Green "ℹ️  СПРАВОЧНЫЕ:"
    Write-Host ""
    Write-Host "  8. roadmap        - ROADMAP.txt (карта пути)"
    Write-Host "  9. summary        - SETUP_SUMMARY_FINAL.md (итоговая сводка)"
    Write-Host "  10. all           - ALL_SET.md (обзор всех ресурсов)"
    Write-Host ""
    
    Write-ColorOutput Green "🔧 ТЕХНИЧЕСКИЕ:"
    Write-Host ""
    Write-Host "  11. strategies    - Открыть папку стратегий"
    Write-Host "  12. configs       - Открыть папку конфигураций"
    Write-Host "  13. github        - Открыть репозиторий на GitHub"
    Write-Host ""
    
    Write-ColorOutput Yellow "📊 ПРИМЕРЫ:"
    Write-Host ""
    Write-Host "  .\docs.ps1 start     # Открыть START_HERE.md"
    Write-Host "  .\docs.ps1 plan      # Открыть план развития"
    Write-Host "  .\docs.ps1 journal   # Открыть дневник"
    Write-Host ""
}

Set-Location $PROJECT_DIR

switch ($Doc.ToLower()) {
    "menu"          { Show-Menu }
    "1"             { notepad "START_HERE.md" }
    "start"         { notepad "START_HERE.md" }
    "2"             { notepad "TODO_FOR_YOU.md" }
    "todo"          { notepad "TODO_FOR_YOU.md" }
    "3"             { notepad "HOW_TO_USE.md" }
    "how"           { notepad "HOW_TO_USE.md" }
    "4"             { notepad "DEVELOPMENT_PLAN.md" }
    "plan"          { notepad "DEVELOPMENT_PLAN.md" }
    "5"             { notepad "TRADING_JOURNAL.md" }
    "journal"       { notepad "TRADING_JOURNAL.md" }
    "6"             { notepad "CHECKLIST.md" }
    "checklist"     { notepad "CHECKLIST.md" }
    "7"             { notepad "QUICKSTART_WINDOWS.md" }
    "quick"         { notepad "QUICKSTART_WINDOWS.md" }
    "8"             { notepad "ROADMAP.txt" }
    "roadmap"       { notepad "ROADMAP.txt" }
    "9"             { notepad "SETUP_SUMMARY_FINAL.md" }
    "summary"       { notepad "SETUP_SUMMARY_FINAL.md" }
    "10"            { notepad "ALL_SET.md" }
    "all"           { notepad "ALL_SET.md" }
    "11"            { explorer "user_data\strategies" }
    "strategies"    { explorer "user_data\strategies" }
    "12"            { explorer "user_data\config" }
    "configs"       { explorer "user_data\config" }
    "13"            { Start-Process "https://github.com/kandibobe/hft-algotrade-bot" }
    "github"        { Start-Process "https://github.com/kandibobe/hft-algotrade-bot" }
    
    default {
        Write-ColorOutput Red "❌ Неизвестный документ: $Doc"
        Write-Host ""
        Show-Menu
    }
}
