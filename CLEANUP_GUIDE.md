# 🗑️ Файлы для удаления (необязательно)

После успешного запуска проекта, следующие файлы можно безопасно удалить для очистки репозитория:

## Дублирующиеся документационные файлы:

```bash
# Старые setup файлы (информация теперь в START.md)
rm ALL_SET.md
rm SETUP_COMPLETE.md
rm SETUP_SUMMARY.md
rm SETUP_SUMMARY_FINAL.md
rm FINAL_UPDATE_SUMMARY.md

# Старые текстовые гайды (заменены на .md)
rm FIRST_RUN.txt
rm QUICK_REFERENCE.txt
rm ROADMAP.txt
rm WELCOME.txt

# Дублирующиеся конфиги
rm user_data/config/config_production_fixed.json
```

## Лишние Docker файлы в корне:

```bash
# Переместить в /docker или удалить
rm Dockerfile.fix
rm Dockerfile.jupyter  # уже есть в /docker
```

## Backup файлы стратегий:

```bash
# Стратегия восстановлена, backup больше не нужен
rm user_data/strategies/StoicStrategyV1.py.bak
rm user_data/strategies/signals.py  # пустой файл
```

## Автоматическая очистка:

Если используешь Git Bash или Linux:

```bash
# Удалить все вышеуказанные файлы одной командой
git rm ALL_SET.md SETUP_COMPLETE.md SETUP_SUMMARY.md \
       SETUP_SUMMARY_FINAL.md FINAL_UPDATE_SUMMARY.md \
       FIRST_RUN.txt QUICK_REFERENCE.txt ROADMAP.txt WELCOME.txt \
       Dockerfile.fix Dockerfile.jupyter \
       user_data/config/config_production_fixed.json \
       user_data/strategies/StoicStrategyV1.py.bak \
       user_data/strategies/signals.py

git commit -m "🧹 Cleanup: Removed obsolete documentation and backup files"
git push origin main
```

## PowerShell команда:

```powershell
# Для Windows PowerShell
$filesToDelete = @(
    "ALL_SET.md",
    "SETUP_COMPLETE.md",
    "SETUP_SUMMARY.md",
    "SETUP_SUMMARY_FINAL.md",
    "FINAL_UPDATE_SUMMARY.md",
    "FIRST_RUN.txt",
    "QUICK_REFERENCE.txt",
    "ROADMAP.txt",
    "WELCOME.txt",
    "Dockerfile.fix",
    "Dockerfile.jupyter",
    "user_data/config/config_production_fixed.json",
    "user_data/strategies/StoicStrategyV1.py.bak",
    "user_data/strategies/signals.py"
)

foreach ($file in $filesToDelete) {
    if (Test-Path $file) {
        git rm $file
        Write-Host "✅ Удален: $file"
    }
}

git commit -m "🧹 Cleanup: Removed obsolete files"
git push origin main
```

## ⚠️ ВАЖНО:

- Эти файлы **можно оставить** - они не мешают работе
- Удаляй **только после** успешного запуска и тестирования
- Перед удалением сделай backup или commit в Git

---

**Примечание**: После этой очистки проект станет более чистым и понятным, но это не обязательно для работы.
