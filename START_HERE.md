# 🚀 НАЧНИ ОТСЮДА - Stoic Citadel

## ⚠️ ВАЖНО: Ты в worktree!

Эта директория - **git worktree** для ветки `condescending-chaum`.

**Главный репозиторий:** `C:\hft-algotrade-bot`
**Этот worktree:** `C:\Users\Владислав\.claude-worktrees\hft-algotrade-bot\condescending-chaum`

---

## 📋 Быстрый старт

### 1. Установка зависимостей (если не установлены)

```bash
# В этой директории (worktree)
pip install -r requirements-dev.txt
```

### 2. Запуск системы

```bash
# Docker контейнеры
docker-compose up -d freqtrade frequi

# Просмотр логов
docker-compose logs -f freqtrade
```

### 3. Доступ к FreqUI

**URL:** http://localhost:3000

```
Логин:  stoic_admin
Пароль: StoicTrade2025!Secure
```

### 4. Запуск тестов

```bash
# Windows
run_tests.bat

# Linux/Mac
pytest tests/test_order_manager/ -v
```

---

## 🔐 Учетные данные

Все пароли в файле: **`CREDENTIALS.md`**

---

## 📂 Структура worktree

```
condescending-chaum/          # ← ТЫ ЗДЕСЬ
├── src/
│   ├── order_manager/        # ✅ NEW: Order Management System
│   ├── ml/                   # 🚧 IN PROGRESS: ML Pipeline
│   ├── strategies/
│   ├── data/
│   └── utils/
├── tests/
│   ├── test_order_manager/   # ✅ NEW: Order tests
│   └── test_ml/              # 📋 TODO
├── examples/
│   └── order_management_example.py  # ✅ Работающие примеры
├── docs/
│   └── ORDER_MANAGEMENT.md   # ✅ Полная документация
├── .env                      # ✅ Конфигурация с паролями
├── CREDENTIALS.md            # ✅ Все учетные данные
├── QUICKSTART.md             # ✅ Быстрый старт
└── START_HERE.md             # ✅ Этот файл
```

---

## 🎯 Что было сделано (Phase 1)

### ✅ Order Management System - ЗАВЕРШЕНО

- **Order Types** - Market, Limit, Stop-Loss, Take-Profit, Trailing Stop
- **Position Manager** - Трекинг позиций с PnL
- **Circuit Breaker** - Защита от катастрофических потерь
- **Slippage Simulator** - Реалистичные бэктесты
- **Order Executor** - Надежное исполнение
- **25 Unit Tests** - 100% pass ✅
- **Examples** - 5 рабочих примеров ✅
- **Documentation** - Полная документация ✅

---

## 🚧 В процессе (Phase 2)

### ML Training Pipeline

- [x] Feature Engineering
- [ ] Model Trainer
- [ ] Experiment Tracker (W&B)
- [ ] Model Registry
- [ ] Tests

---

## ⚠️ Распространенные ошибки

### Ошибка: "no tests ran"

**Проблема:** Запускаешь pytest из главной директории `C:\hft-algotrade-bot`

**Решение:** Запускай из **worktree**:
```bash
cd C:\Users\Владислав\.claude-worktrees\hft-algotrade-bot\condescending-chaum
pytest tests/test_order_manager/ -v
```

Или используй:
```bash
run_tests.bat
```

### Ошибка: "Could not import signals"

**Видно в логах:** `Could not import /freqtrade/user_data/strategies/StoicCitadelV2.py due to 'No module named 'signals''`

**Это нормально!** Freqtrade пробует загрузить все стратегии и пропускает те, у которых нет зависимостей. Используется `StoicEnsembleStrategyV2` - она работает ✅

### Warning: "Found orphan containers"

**Решение:** Почистить старые контейнеры:
```bash
docker-compose down --remove-orphans
```

---

## 📊 Текущий статус

```
Phase 1: Order Management   ✅ 100% Complete
Phase 2: ML Pipeline        🚧 30% In Progress
Phase 3: Monitoring         📋 Planned
Phase 4: CI/CD             📋 Planned
```

---

## 🔄 Git Workflow (worktree)

### Проверить изменения
```bash
git status
```

### Коммит изменений
```bash
git add .
git commit -m "feat: add ML pipeline"
```

### Переключиться в main
```bash
cd C:\hft-algotrade-bot
git checkout main
```

### Merge ветки (когда готово)
```bash
cd C:\hft-algotrade-bot
git checkout main
git merge condescending-chaum
git push
```

---

## 📞 Полезные команды

### Docker
```bash
# Запуск
docker-compose up -d freqtrade frequi

# Логи
docker-compose logs -f freqtrade

# Остановка
docker-compose down

# Полная очистка
docker-compose down -v
```

### Tests
```bash
# Все тесты Order Management
pytest tests/test_order_manager/ -v

# С покрытием
pytest tests/test_order_manager/ --cov=src.order_manager --cov-report=html

# Конкретный тест
pytest tests/test_order_manager/test_circuit_breaker.py -v
```

### Examples
```bash
# Запуск примеров Order Management
python examples/order_management_example.py
```

---

## 📚 Документация

1. **QUICKSTART.md** - Быстрый старт
2. **CREDENTIALS.md** - Все пароли
3. **docs/ORDER_MANAGEMENT.md** - API документация
4. **PROGRESS_SUMMARY.md** - Детальный прогресс
5. **README.md** - Основная документация

---

## 🎓 Best Practices

### Перед коммитом
1. Запусти тесты: `run_tests.bat`
2. Проверь форматирование: `black src/ tests/`
3. Проверь импорты: `isort src/ tests/`

### Перед merge в main
1. Все тесты проходят ✅
2. Документация обновлена ✅
3. CHANGELOG обновлен ✅

---

**Последнее обновление:** 2025-12-17
**Ветка:** condescending-chaum
**Статус:** Phase 1 Complete, Phase 2 In Progress
