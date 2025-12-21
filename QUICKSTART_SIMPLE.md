# Быстрый старт Stoic Citadel

## Текущее состояние проекта
✅ **Проект полностью функционален и готов к использованию**

### Проверенные компоненты:
1. ✅ Все тесты проходят (190+ тестов)
2. ✅ ML модели загружены и работают
3. ✅ Конфигурационные файлы валидны
4. ✅ Данные для backtest доступны
5. ✅ Зависимости установлены

## Быстрый запуск

### 1. Активация виртуального окружения
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Запуск простого backtest
```bash
python scripts/simple_walk_forward_test.py \
  --config user_data/config/config_backtest.json \
  --strategy StoicEnsembleStrategyV4 \
  --timeframe 5m \
  --pairs BTC/USDT \
  --days 7 \
  --dry-run
```

### 3. Запуск торговли в dry-run режиме
```bash
# Используйте конфиг для paper trading
python scripts/run_backtest.py \
  --config user_data/config/config_dryrun.json \
  --strategy StoicEnsembleStrategyV4
```

### 4. Запуск мониторинга
```bash
# Health check системы
python test_health_check_simple.py

# Или полный health check
python test_health_check.py
```

## Архитектура проекта

### Ключевые модули:
1. **src/order_manager/** - Система управления ордерами
   - Умные лимитные ордера (экономия на комиссиях)
   - Circuit breaker (защита от больших потерь)
   - Симуляция проскальзывания

2. **src/ml/** - ML пайплайн
   - Feature engineering (50+ индикаторов)
   - Triple barrier labeling (правильные ML метки)
   - Обучение моделей (Random Forest, XGBoost, LightGBM)
   - Online learning (адаптация к рынку)

3. **src/risk/** - Управление рисками
   - Position sizing (размер позиций на основе волатильности)
   - Correlation monitoring (мониторинг корреляций)
   - Pre-trade checks (проверки перед сделкой)

4. **src/strategies/** - Торговые стратегии
   - StoicEnsembleStrategyV4 (ML-enhanced)
   - Режимная адаптация (bull/bear/sideways)
   - Ансамбль сигналов (ML + технический анализ)

5. **src/monitoring/** - Мониторинг
   - Health checks (проверка всех компонентов)
   - Metrics exporter (Prometheus)
   - Structured logging (ELK stack)

## Конфигурация

### Основные конфиги:
- `user_data/config/config_backtest.json` - для backtest
- `user_data/config/config_dryrun.json` - для paper trading
- `user_data/config/config_production.json` - для реальной торговли

### Стратегии:
- `StoicEnsembleStrategyV2` - Базовая ансамблевая стратегия
- `StoicEnsembleStrategyV3` + Meta-learning
- `StoicEnsembleStrategyV4` + ML predictions (рекомендуется)

## ML Модели

Модели хранятся в `user_data/models/`:
- `random_forest_*.pkl` - Random Forest модели
- `production_model.pkl` - текущая production модель
- `online_model.pkl` - модель для online learning

## Данные

Данные хранятся в `user_data/data/`:
- `BTC_USDT_1h.parquet` - исторические данные
- `binance/` - данные с Binance

## Мониторинг и логи

- Логи: `user_data/logs/`
- Результаты backtest: `user_data/walk_forward_results/`
- Графики: `user_data/plot/`

## Docker запуск

```bash
# Запуск всех сервисов
docker-compose up -d

# Доступ к интерфейсам:
# - FreqUI: http://localhost:3000
# - Jupyter: http://localhost:8888
# - Portainer: http://localhost:9000
```

## Устранение неполадок

### 1. Если тесты не проходят:
```bash
# Запуск всех тестов
pytest tests/ -v

# Запуск тестов конкретного модуля
pytest tests/test_order_manager/ -v
pytest tests/test_ml/ -v
```

### 2. Если нет ML моделей:
```bash
# Обучение новой модели
python scripts/train_models.py
```

### 3. Если нет данных:
```bash
# Скачивание данных
python scripts/download_data.py
```

### 4. Если проблемы с зависимостями:
```bash
# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Безопасность

⚠️ **ВАЖНО**: 
- Начинайте с dry-run режима
- Тестируйте стратегии на исторических данных
- Используйте small amounts в live trading
- Мониторьте circuit breaker статус

## Контакты и поддержка

- Документация: `docs/` директория
- Примеры: `examples/` директория
- Скрипты: `scripts/` директория

---

**Stoic Citadel** - Trade with wisdom, not emotion.
**Статус**: Production Ready
**Версия**: 2.1.0
**Последнее обновление**: 2025-12-21
