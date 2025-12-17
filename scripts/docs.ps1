# ==============================================================================
# DOCS NAVIGATOR - Р‘С‹СЃС‚СЂС‹Р№ РґРѕСЃС‚СѓРї Рє РґРѕРєСѓРјРµРЅС‚Р°С†РёРё
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
    Write-ColorOutput Cyan "в•”в•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•—"
    Write-ColorOutput Cyan "в•‘            STOIC CITADEL - Р”РћРљРЈРњР•РќРўРђР¦РРЇ                    в•‘"
    Write-ColorOutput Cyan "в•љв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ђв•ќ"
    Write-Host ""
    
    Write-ColorOutput Green "рџ“љ Р“Р›РђР’РќР«Р• Р”РћРљРЈРњР•РќРўР« (С‡РёС‚Р°Р№ РїРѕ РїРѕСЂСЏРґРєСѓ):"
    Write-Host ""
    Write-Host "  1. start          - START_HERE.md (РЅР°С‡РЅРё Р·РґРµСЃСЊ!)"
    Write-Host "  2. todo           - TODO_FOR_YOU.md (С‡С‚Рѕ РґРµР»Р°С‚СЊ СЃРµР№С‡Р°СЃ)"
    Write-Host "  3. how            - HOW_TO_USE.md (РєР°Рє РїРѕР»СЊР·РѕРІР°С‚СЊСЃСЏ)"
    Write-Host "  4. plan           - DEVELOPMENT_PLAN.md (РїР»Р°РЅ РЅР° 6 РЅРµРґРµР»СЊ)"
    Write-Host ""
    
    Write-ColorOutput Green "рџ“ќ Р РђР‘РћР§РР• Р”РћРљРЈРњР•РќРўР«:"
    Write-Host ""
    Write-Host "  5. journal        - TRADING_JOURNAL.md (РґРЅРµРІРЅРёРє РЅР°Р±Р»СЋРґРµРЅРёР№)"
    Write-Host "  6. checklist      - CHECKLIST.md (С‡РµРєР»РёСЃС‚ Р·Р°РїСѓСЃРєР°)"
    Write-Host "  7. quick          - QUICKSTART_WINDOWS.md (РґРµС‚Р°Р»СЊРЅРѕРµ СЂСѓРєРѕРІРѕРґСЃС‚РІРѕ)"
    Write-Host ""
    
    Write-ColorOutput Green "в„№пёЏ  РЎРџР РђР’РћР§РќР«Р•:"
    Write-Host ""
    Write-Host "  8. roadmap        - ROADMAP.txt (РєР°СЂС‚Р° РїСѓС‚Рё)"
    Write-Host "  9. summary        - SETUP_SUMMARY_FINAL.md (РёС‚РѕРіРѕРІР°СЏ СЃРІРѕРґРєР°)"
    Write-Host "  10. all           - ALL_SET.md (РѕР±Р·РѕСЂ РІСЃРµС… СЂРµСЃСѓСЂСЃРѕРІ)"
    Write-Host ""
    
    Write-ColorOutput Green "рџ”§ РўР•РҐРќРР§Р•РЎРљРР•:"
    Write-Host ""
    Write-Host "  11. strategies    - РћС‚РєСЂС‹С‚СЊ РїР°РїРєСѓ СЃС‚СЂР°С‚РµРіРёР№"
    Write-Host "  12. configs       - РћС‚РєСЂС‹С‚СЊ РїР°РїРєСѓ РєРѕРЅС„РёРіСѓСЂР°С†РёР№"
    Write-Host "  13. github        - РћС‚РєСЂС‹С‚СЊ СЂРµРїРѕР·РёС‚РѕСЂРёР№ РЅР° GitHub"
    Write-Host ""
    
    Write-ColorOutput Yellow "рџ“Љ РџР РРњР•Р Р«:"
    Write-Host ""
    Write-Host "  .\docs.ps1 start     # РћС‚РєСЂС‹С‚СЊ START_HERE.md"
    Write-Host "  .\docs.ps1 plan      # РћС‚РєСЂС‹С‚СЊ РїР»Р°РЅ СЂР°Р·РІРёС‚РёСЏ"
    Write-Host "  .\docs.ps1 journal   # РћС‚РєСЂС‹С‚СЊ РґРЅРµРІРЅРёРє"
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
        Write-ColorOutput Red "вќЊ РќРµРёР·РІРµСЃС‚РЅС‹Р№ РґРѕРєСѓРјРµРЅС‚: $Doc"
        Write-Host ""
        Show-Menu
    }
}
