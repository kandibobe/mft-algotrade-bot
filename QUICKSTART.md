# 🚀 Stoic Citadel - Быстрый старт

## Что было добавлено (Phase 1)

### ✅ Order Management System

Полноценная система управления ордерами:

- **Order Types** - Market, Limit, Stop-Loss, Take-Profit, Trailing Stop
- **Position Manager** - Трекинг позиций с real-time PnL
- **Circuit Breaker** - Защита от катастрофических потерь
- **Slippage Simulator** - Реалистичная симуляция исполнения
- **Order Executor** - Надежное исполнение с retry логикой

📖 **Документация:** `docs/ORDER_MANAGEMENT.md`

---

## 🔐 Учетные данные

### FreqUI Web Dashboard
```
URL:    http://localhost:3000
Логин:  stoic_admin
Пароль: StoicTrade2025!Secure
```

### Jupyter Lab (для исследований)
```
URL:   http://localhost:8888
Token: JupyterStoic2025!Token
```

### PostgreSQL Database
```
Host:     localhost:5433
User:     stoic_trader
Password: PostgresDB2025!Secure
Database: trading_analytics
```

> 💡 **Полный список учетных данных:** см. файл `CREDENTIALS.md`

---

## 📦 Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
```

### 2. Конфигурация уже настроена

Файл `.env` уже создан с базовыми настройками.

Для изменения параметров:
```bash
nano .env  # или любой редактор
```

### 3. Запуск системы

```bash
# Запуск Freqtrade + FreqUI
docker-compose up -d freqtrade frequi

# Просмотр логов
docker-compose logs -f freqtrade

# Остановка
docker-compose down
```

### 4. Доступ к FreqUI

1. Откройте http://localhost:3000
2. Введите учетные данные:
   - Логин: `stoic_admin`
   - Пароль: `StoicTrade2025!Secure`

---

## 🧪 Тестирование Order Management System

### Запуск тестов

```bash
# Все тесты
pytest tests/test_order_manager/ -v

# С покрытием
pytest tests/test_order_manager/ --cov=src.order_manager --cov-report=html

# Конкретный тест
pytest tests/test_order_manager/test_circuit_breaker.py -v
```

### Запуск примеров

```bash
python examples/order_management_example.py
```

Вы увидите:
- Lifecycle ордера (создание → исполнение → заполнение)
- Управление позициями с PnL
- Работу circuit breaker
- Симуляцию slippage
- Полный торговый workflow

---

## 📊 Backtesting

### Скачать данные

```bash
make -f Makefile.backtest download PAIRS="BTC/USDT ETH/USDT" TIMERANGE="20240101-20240601"
```

### Запустить бэктест

```bash
# С новым Order Management System
make -f Makefile.backtest backtest STRATEGY=StoicEnsembleStrategyV2

# Просмотреть результаты
make -f Makefile.backtest report
```

### Walk-Forward оптимизация

```bash
python scripts/walk_forward.py \
    --strategy StoicEnsembleStrategyV2 \
    --train-months 3 \
    --test-months 1 \
    --start-date 20230101 \
    --end-date 20240101
```

---

## 🏗️ Архитектура Order Management

```
src/order_manager/
├── order_types.py          # Order classes & state machine
├── position_manager.py     # Position tracking
├── circuit_breaker.py      # Risk protection
├── slippage_simulator.py   # Execution simulation
└── order_executor.py       # Order execution
```

### Использование в стратегии

```python
from freqtrade.strategy import IStrategy
from src.order_manager import CircuitBreaker, PositionManager

class MyStrategy(IStrategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.circuit_breaker = CircuitBreaker()
        self.position_manager = PositionManager(max_positions=3)

    def populate_entry_trend(self, dataframe, metadata):
        # Проверка circuit breaker перед входом
        if not self.circuit_breaker.is_operational:
            dataframe['enter_long'] = 0
            return dataframe

        # ... ваша логика входа ...

        return dataframe
```

---

## 📈 Мониторинг

### Запуск с мониторингом

```bash
docker-compose --profile analytics up -d
```

Сервисы:
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)
- **Alertmanager**: http://localhost:9093

### Метрики

Order Management System экспортирует метрики:
- `trading_trades_total` - Количество сделок
- `trading_pnl_total` - Общий PnL
- `trading_drawdown_current` - Текущая просадка
- `trading_positions_open` - Открытые позиции

---

## 🔧 Troubleshooting

### FreqUI не подключается

```bash
# Проверить статус
docker-compose ps

# Проверить API
curl http://localhost:8080/api/v1/ping

# Перезапустить
docker-compose restart freqtrade frequi
```

### Забыли пароль

1. Откройте `.env`
2. Измените `FREQTRADE_API_PASSWORD`
3. Перезапустите: `docker-compose restart freqtrade`

### Ошибки при запуске тестов

```bash
# Установить зависимости
pip install -r requirements-dev.txt

# Проверить Python версию (должна быть 3.11+)
python --version
```

---

## 📚 Документация

- **Order Management**: `docs/ORDER_MANAGEMENT.md`
- **Architecture**: `ARCHITECTURE_ANALYSIS.md`
- **Development Plan**: `DEVELOPMENT_PLAN.md`
- **Deployment**: `DEPLOYMENT.md`

---

## 🎯 Следующие шаги

### Phase 2: ML Pipeline (в разработке)

Планируется:
- ML Training Pipeline
- Experiment Tracking (W&B / MLflow)
- Model Registry
- Automated model validation

### Phase 3: Enhanced Monitoring

- Детальные метрики для Prometheus
- Custom Grafana dashboards
- Alerting через Slack/Email
- ELK Stack для логов

---

## ⚠️ Важные заметки

### Безопасность

- ✅ `.env` добавлен в `.gitignore`
- ✅ `CREDENTIALS.md` не коммитится
- ⚠️ Измените пароли перед продакшеном!
- ⚠️ Используйте `DRY_RUN=true` для тестирования

### Trading Mode

По умолчанию включен **paper trading** (виртуальные деньги):
```bash
DRY_RUN=true
DRY_RUN_WALLET=10000
```

Для live trading:
1. Получите API ключи от биржи
2. Добавьте в `.env`:
   ```
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   DRY_RUN=false
   ```
3. **Начните с малых сумм!**

---

## 🤝 Contributing

1. Fork репозиторий
2. Создайте feature branch
3. Запустите тесты: `pytest tests/ -v`
4. Commit с conventional commits: `feat: add feature`
5. Push и создайте Pull Request

---

## 📞 Поддержка

- **Issues**: https://github.com/kandibobe/hft-algotrade-bot/issues
- **Документация**: См. `docs/` директорию
- **Examples**: См. `examples/` директорию

---

**🏛️ Stoic Citadel** - Professional Algorithmic Trading System

*"The wise man accepts losses with equanimity."*
