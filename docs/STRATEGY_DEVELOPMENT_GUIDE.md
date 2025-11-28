# 📊 Strategy Development & Modification Guide

## Содержание
1. [Быстрый старт: Смена стратегии](#быстрый-старт-смена-стратегии)
2. [Создание новой стратегии](#создание-новой-стратегии)
3. [Модификация существующей стратегии](#модификация-существующей-стратегии)
4. [Research → Production Pipeline](#research-production-pipeline)
5. [Оптимизация гиперпараметров](#оптимизация-гиперпараметров)
6. [Troubleshooting](#troubleshooting)

---

## Быстрый старт: Смена стратегии

### Вариант A: Смена через конфиг (рекомендуется)

#### 1. Откройте конфиг:
```bash
nano user_data/config/config_dryrun.json
```

#### 2. Найдите строку `--strategy` в docker-compose.yml:
```bash
nano docker-compose.yml
```

Измените:
```yaml
command: >
  trade
  --logfile /freqtrade/user_data/logs/freqtrade.log
  --db-url sqlite:////freqtrade/user_data/tradesv3.sqlite
  --config /freqtrade/user_data/config/config_production.json
  --strategy StoicCitadelV2  # ⬅️ ЗДЕСЬ ИЗМЕНИТЕ
```

#### 3. Перезапустите бота:
```bash
make restart
# ИЛИ
docker-compose restart freqtrade
```

### Вариант B: Смена через CLI (для экспериментов)

```bash
# Остановите текущий бот
make stop

# Запустите с новой стратегией
docker-compose run --rm freqtrade trade \
  --strategy StoicCitadelV2 \
  --config user_data/config/config_dryrun.json
```

---

## Создание новой стратегии

### Шаг 1: Создайте файл стратегии

```bash
# Используйте shared library (РЕКОМЕНДУЕТСЯ)
cp user_data/strategies/StoicCitadelV2.py \
   user_data/strategies/MyCustomStrategy.py
```

#### Откройте и измените:
```python
# user_data/strategies/MyCustomStrategy.py

from freqtrade.strategy import IStrategy
from pandas import DataFrame

class MyCustomStrategy(IStrategy):
    """
    Моя кастомная стратегия.

    Описание логики:
    - Entry: [опишите условия входа]
    - Exit: [опишите условия выхода]
    - Risk: [опишите риск-менеджмент]
    """

    INTERFACE_VERSION = 3

    # Метаданные стратегии
    minimal_roi = {
        "0": 0.10,   # 10% немедленно
        "30": 0.05,  # 5% через 30 мин
        "60": 0.03   # 3% через час
    }

    stoploss = -0.05  # -5% hard stop

    timeframe = '5m'

    # ... остальной код
```

### Шаг 2: Реализуйте логику

#### Обязательные методы:

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """
    Расчет индикаторов.

    ⚠️ ВАЖНО: Используйте shared library для parity с research!
    """
    # Вариант 1: Используйте SignalGenerator (РЕКОМЕНДУЕТСЯ)
    from signals.indicators import SignalGenerator
    signal_gen = SignalGenerator()
    dataframe = signal_gen.populate_all_indicators(dataframe)

    # Вариант 2: Свои индикаторы
    dataframe['my_indicator'] = talib.RSI(dataframe['close'], timeperiod=14)

    return dataframe

def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Условия входа в сделку."""
    dataframe.loc[
        (
            # Ваши условия
            (dataframe['rsi'] < 30) &
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean())
        ),
        'enter_long'
    ] = 1

    return dataframe

def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    """Условия выхода из сделки."""
    dataframe.loc[
        (
            (dataframe['rsi'] > 70)
        ),
        'exit_long'
    ] = 1

    return dataframe
```

### Шаг 3: Тестирование

```bash
# 1. Unit тест
pytest tests/test_strategies/test_my_custom_strategy.py

# 2. Quick backtest
make backtest STRATEGY=MyCustomStrategy TIMERANGE=20240101-20240130

# 3. Full backtest
docker-compose run --rm freqtrade backtesting \
  --strategy MyCustomStrategy \
  --timerange 20240101- \
  --enable-protections
```

---

## Модификация существующей стратегии

### Use Case 1: Изменение параметров входа

**Задача**: Хочу открывать сделки при RSI < 25 вместо < 30

#### Шаг 1: Найдите метод `populate_entry_trend`:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = (
        (dataframe['rsi'] < 35) &  # ⬅️ БЫЛО 35
        (dataframe['slowk'] < 30) &
        # ...
    )
```

#### Шаг 2: Измените значение:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = (
        (dataframe['rsi'] < 25) &  # ⬅️ СТАЛО 25 (более строгий фильтр)
        (dataframe['slowk'] < 30) &
        # ...
    )
```

#### Шаг 3: Обновите версию стратегии:

```python
class MyCustomStrategy(IStrategy):
    """
    Version: 1.1.0  # ⬅️ ОБНОВИТЕ ВЕРСИЮ
    Changelog:
    - 1.1.0: Изменен RSI порог с 35 на 25
    - 1.0.0: Исходная версия
    """
```

#### Шаг 4: Тестирование:

```bash
# Сравните результаты с предыдущей версией
docker-compose run --rm freqtrade backtesting \
  --strategy MyCustomStrategy \
  --timerange 20240101-20240630 \
  --export trades \
  --export-filename v1_1_0_results.json

# Проанализируйте разницу
python scripts/compare_backtests.py \
  --old v1_0_0_results.json \
  --new v1_1_0_results.json
```

### Use Case 2: Добавление нового индикатора

**Задача**: Добавить ATR для фильтрации волатильности

#### Шаг 1: Добавьте расчет в `populate_indicators`:

```python
def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    # Существующие индикаторы
    dataframe['rsi'] = ta.RSI(dataframe['close'])

    # ⬇️ НОВЫЙ ИНДИКАТОР
    dataframe['atr'] = ta.ATR(
        dataframe['high'],
        dataframe['low'],
        dataframe['close'],
        timeperiod=14
    )
    dataframe['atr_pct'] = dataframe['atr'] / dataframe['close']

    return dataframe
```

#### Шаг 2: Используйте в условиях входа:

```python
def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    conditions = (
        (dataframe['rsi'] < 30) &
        # ⬇️ НОВОЕ УСЛОВИЕ: фильтр по волатильности
        (dataframe['atr_pct'] > 0.01) &  # Минимальная волатильность 1%
        (dataframe['atr_pct'] < 0.10)    # Максимальная волатильность 10%
    )
```

#### Шаг 3: Проверьте, что индикатор рассчитывается:

```bash
# Запустите бэктест с экспортом
docker-compose run --rm freqtrade backtesting \
  --strategy MyCustomStrategy \
  --timerange 20240101-20240107 \
  --export trades \
  --export-filename test.json

# Проверьте наличие индикатора
python -c "
import json
import pandas as pd

with open('user_data/backtest_results/test.json') as f:
    data = json.load(f)

# Проверка
print('ATR column exists:', 'atr' in data['columns'])
"
```

### Use Case 3: Модификация стоп-лосса

**Задача**: Использовать динамический стоп-лосс на основе ATR

#### Исходная версия (фиксированный):
```python
stoploss = -0.05  # Фиксированный -5%
```

#### Новая версия (динамический):
```python
stoploss = -0.05  # Fallback значение

def custom_stoploss(
    self,
    pair: str,
    trade,
    current_time: datetime,
    current_rate: float,
    current_profit: float,
    **kwargs
) -> float:
    """
    Динамический стоп-лосс на основе ATR.

    Логика: stoploss = current_price - (2 * ATR)
    """
    dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
    last_candle = dataframe.iloc[-1].squeeze()

    atr = last_candle['atr']
    current_price = last_candle['close']

    # Расчет стопа: 2 ATR от текущей цены
    stop_price = current_price - (2 * atr)
    stop_loss_pct = (stop_price - trade.open_rate) / trade.open_rate

    # Ограничение: не больше -10%
    return max(stop_loss_pct, -0.10)
```

---

## Research → Production Pipeline

### Философия: Code Once, Use Everywhere

Используйте **shared library** для гарантии parity между research и production.

### Pipeline:

```
┌─────────────────┐
│  1. Research    │  Jupyter Notebook
│     (VectorBT)  │  - Тестирование идей
└────────┬────────┘  - Оптимизация параметров
         │
         ▼
┌─────────────────┐
│  2. Shared Lib  │  src/signals/indicators.py
│     Creation    │  - Централизованная логика
└────────┬────────┘  - Pure functions
         │
         ▼
┌─────────────────┐
│  3. Freqtrade   │  user_data/strategies/
│     Strategy    │  - Импорт shared lib
└────────┬────────┘  - Минимальный wrapper
         │
         ▼
┌─────────────────┐
│  4. Backtest    │  - Проверка parity
│     Validation  │  - Сравнение с research
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Production  │  - Live trading
└─────────────────┘
```

### Пример: Создание стратегии из исследования

#### 1. Research (Jupyter):

```python
# research/my_research.ipynb

import pandas as pd
import vectorbt as vbt

# Загрузка данных
data = vbt.BinanceData.download(
    symbols=['BTC/USDT'],
    timeframe='5m',
    start='2024-01-01',
    end='2024-06-30'
)

# Тестирование логики
def calculate_my_signal(close, rsi_period=14):
    """Моя логика генерации сигнала."""
    rsi = vbt.RSI.run(close, window=rsi_period).rsi
    entries = (rsi < 30)
    exits = (rsi > 70)
    return entries, exits

# Бэктест
entries, exits = calculate_my_signal(data.close)

portfolio = vbt.Portfolio.from_signals(
    data.close,
    entries,
    exits,
    fees=0.001
)

print(portfolio.stats())
# Win rate: 55%, Sharpe: 1.2 ✅
```

#### 2. Перенос в Shared Library:

```python
# src/signals/my_signals.py

import talib.abstract as ta
from pandas import Series

def calculate_my_signal_entry(close: Series, rsi_period: int = 14) -> Series:
    """
    Генерация сигнала входа.

    ⚠️ ИДЕНТИЧНЫЙ КОД с research!
    """
    rsi = ta.RSI(close, timeperiod=rsi_period)
    return (rsi < 30).astype(int)

def calculate_my_signal_exit(close: Series, rsi_period: int = 14) -> Series:
    """Генерация сигнала выхода."""
    rsi = ta.RSI(close, timeperiod=rsi_period)
    return (rsi > 70).astype(int)
```

#### 3. Freqtrade Strategy:

```python
# user_data/strategies/MyResearchStrategy.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from freqtrade.strategy import IStrategy
from pandas import DataFrame
from signals.my_signals import calculate_my_signal_entry, calculate_my_signal_exit
import talib.abstract as ta

class MyResearchStrategy(IStrategy):
    """Стратегия из исследования - 100% parity."""

    INTERFACE_VERSION = 3
    timeframe = '5m'
    stoploss = -0.05

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Минимум индикаторов - всё в shared library
        dataframe['rsi'] = ta.RSI(dataframe['close'], timeperiod=14)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # ⚠️ ИДЕНТИЧНАЯ ЛОГИКА!
        dataframe['enter_long'] = calculate_my_signal_entry(
            dataframe['close'],
            rsi_period=14
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = calculate_my_signal_exit(
            dataframe['close'],
            rsi_period=14
        )
        return dataframe
```

#### 4. Валидация Parity:

```python
# tests/test_parity.py

import pytest
from my_research_strategy import MyResearchStrategy
from signals.my_signals import calculate_my_signal_entry

def test_research_production_parity(sample_dataframe):
    """Проверка идентичности research и production логики."""
    strategy = MyResearchStrategy()

    # Production сигнал
    df_prod = strategy.populate_indicators(sample_dataframe.copy(), {})
    df_prod = strategy.populate_entry_trend(df_prod, {})

    # Research сигнал
    research_signal = calculate_my_signal_entry(sample_dataframe['close'])

    # ДОЛЖНЫ БЫТЬ ИДЕНТИЧНЫ
    pd.testing.assert_series_equal(
        df_prod['enter_long'],
        research_signal,
        check_names=False
    )
```

---

## Оптимизация гиперпараметров

### Hyperopt (встроенный в Freqtrade)

#### 1. Определите пространство поиска:

```python
# user_data/strategies/MyCustomStrategy.py

from skopt.space import Integer, Real, Categorical

class MyCustomStrategy(IStrategy):
    # ... остальной код

    # Пространство гиперпараметров
    buy_rsi = IntParameter(20, 40, default=30, space='buy')
    buy_adx = IntParameter(15, 30, default=20, space='buy')
    sell_rsi = IntParameter(60, 80, default=70, space='sell')

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = (
            (dataframe['rsi'] < self.buy_rsi.value) &  # ⬅️ Использование параметра
            (dataframe['adx'] > self.buy_adx.value)
        )
        # ...
```

#### 2. Запустите оптимизацию:

```bash
# Оптимизация на 500 эпохах
make hyperopt

# Альтернатива
docker-compose run --rm freqtrade hyperopt \
  --strategy MyCustomStrategy \
  --hyperopt-loss SharpeHyperOptLoss \
  --epochs 500 \
  --spaces buy sell \
  --timerange 20240101-20240630
```

#### 3. Применение результатов:

Hyperopt выведет:
```
Best result:
    buy_rsi = 25
    buy_adx = 22
    sell_rsi = 75
```

Обновите стратегию:
```python
buy_rsi = IntParameter(20, 40, default=25, space='buy')  # ⬅️ Обновлено
buy_adx = IntParameter(15, 30, default=22, space='buy')
sell_rsi = IntParameter(60, 80, default=75, space='sell')
```

---

## Troubleshooting

### Проблема: Стратегия не загружается

**Ошибка**:
```
ImportError: cannot import name 'MyStrategy'
```

**Решение**:
```bash
# 1. Проверьте имя файла = имени класса
ls user_data/strategies/MyStrategy.py  # Должен существовать

# 2. Проверьте синтаксис
python user_data/strategies/MyStrategy.py

# 3. Проверьте список стратегий
docker-compose run --rm freqtrade list-strategies
```

### Проблема: Индикаторы не рассчитываются

**Симптомы**: No trades in backtest

**Диагностика**:
```bash
# Проверьте, что индикаторы добавляются
docker-compose run --rm freqtrade backtesting \
  --strategy MyStrategy \
  --timerange 20240101-20240107 \
  --export trades

# Посмотрите в экспортированные данные
python -c "
import json
with open('user_data/backtest_results/.../trades.json') as f:
    data = json.load(f)
    print(data['columns'])  # Должны быть ваши индикаторы
"
```

### Проблема: Research и Production дают разные результаты

**Причины**:
1. **Lookahead bias** - использование будущих данных
2. **Разные библиотеки** - pandas vs talib
3. **Разное выравнивание данных** - NaN handling

**Решение**: Используйте shared library!

---

## Best Practices

### ✅ DO:

1. **Версионируйте стратегии**:
   ```python
   class MyStrategy_v1_2_0(IStrategy):
       """Version 1.2.0 - Added ATR filter"""
   ```

2. **Документируйте изменения**:
   ```python
   """
   Changelog:
   - v1.2.0: Added ATR volatility filter
   - v1.1.0: Changed RSI threshold from 30 to 25
   - v1.0.0: Initial version
   """
   ```

3. **Используйте type hints**:
   ```python
   def custom_stoploss(
       self,
       pair: str,
       trade: Trade,
       current_time: datetime,
       current_rate: float,
       current_profit: float,
       **kwargs
   ) -> float:
   ```

4. **Тестируйте каждое изменение**:
   ```bash
   pytest && make backtest STRATEGY=MyStrategy
   ```

### ❌ DON'T:

1. **Не оптимизируйте на одном датасете**
   - Гарантированный overfitting

2. **Не модифицируйте live стратегию без тестов**
   - Сначала backtest → paper trading → live

3. **Не используйте магические числа**:
   ```python
   # ❌ Плохо
   if dataframe['rsi'] < 30:

   # ✅ Хорошо
   RSI_OVERSOLD_THRESHOLD = 30
   if dataframe['rsi'] < RSI_OVERSOLD_THRESHOLD:
   ```

---

## Итоговый Checklist

Перед деплоем новой стратегии:

- [ ] Код прошел unit тесты
- [ ] Backtest показал profit > 0
- [ ] Win rate > 50%
- [ ] Max drawdown < 15%
- [ ] Walk-forward validation пройдена
- [ ] Paper trading 2+ недели
- [ ] Документация обновлена
- [ ] Версия увеличена
- [ ] Changelog заполнен
- [ ] Code review пройден
- [ ] Shared library используется (если применимо)

---

🏛️ **Stoic Citadel** - Code once, profit repeatedly.
