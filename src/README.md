# Stoic Citadel - Shared Library

**Version**: 1.0.0
**Purpose**: Обеспечение 100% parity между Research и Production

---

## Философия

```
┌────────────────────┐         ┌────────────────────┐
│   RESEARCH         │         │   PRODUCTION       │
│   (Jupyter +       │         │   (Freqtrade)      │
│    VectorBT)       │         │                    │
│                    │         │                    │
│  ┌──────────────┐  │         │  ┌──────────────┐  │
│  │  Backtests   │  │         │  │ Live Trading │  │
│  └───────┬──────┘  │         │  └───────┬──────┘  │
│          │         │         │          │         │
│          ▼         │         │          ▼         │
│  ┌──────────────┐  │         │  ┌──────────────┐  │
│  │   Signals    │◄─┼─────────┼─►│   Signals    │  │
│  │   Library    │  │         │  │   Library    │  │
│  └──────────────┘  │  SAME   │  └──────────────┘  │
│                    │  CODE!  │                    │
└────────────────────┘         └────────────────────┘
```

**Ключевой принцип**: Code Once, Use Everywhere

---

## Структура

```
src/
├── __init__.py
│
├── signals/                   # Генерация торговых сигналов
│   ├── __init__.py
│   └── indicators.py         # Core: IndicatorLibrary, SignalGenerator
│
├── risk/                      # Risk management
│   ├── __init__.py
│   └── correlation.py        # CorrelationManager, DrawdownMonitor
│
└── ml_inference/             # ML inference (future)
    └── __init__.py
```

---

## Использование

### В Research (Jupyter):

```python
import sys
sys.path.insert(0, '../src')

from signals.indicators import SignalGenerator
import vectorbt as vbt

# Загрузка данных
data = vbt.BinanceData.download(...)

# Генерация сигналов (ИДЕНТИЧНО production!)
signal_gen = SignalGenerator()
df = signal_gen.populate_all_indicators(data.get())

entries = signal_gen.generate_entry_signal(df)
exits = signal_gen.generate_exit_signal(df)

# Бэктест
portfolio = vbt.Portfolio.from_signals(
    data.close,
    entries,
    exits,
    fees=0.001
)

print(portfolio.stats())
```

### В Production (Freqtrade):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from freqtrade.strategy import IStrategy
from signals.indicators import SignalGenerator

class MyStrategy(IStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.signal_generator = SignalGenerator()

    def populate_indicators(self, dataframe, metadata):
        # ИДЕНТИЧНО research!
        return self.signal_generator.populate_all_indicators(dataframe)

    def populate_entry_trend(self, dataframe, metadata):
        dataframe['enter_long'] = self.signal_generator.generate_entry_signal(
            dataframe
        )
        return dataframe
```

---

## Модули

### signals/indicators.py

**Classes**:
- `IndicatorLibrary`: Pure functions для расчета индикаторов
- `SignalGenerator`: Генерация entry/exit сигналов

**Методы**:

```python
# Trend indicators
calculate_ema_trio(close, fast=50, medium=100, slow=200)
calculate_adx(high, low, close, period=14)

# Oscillators
calculate_rsi(close, period=14)
calculate_stochastic(high, low, close)
calculate_macd(close, fast=12, slow=26, signal=9)

# Volatility
calculate_bollinger_bands(close, period=20)
calculate_atr(high, low, close, period=14)

# Custom
calculate_trend_score(close, ema_fast, ema_medium, ema_slow)
```

**Signal Generation**:
```python
signal_gen = SignalGenerator()

# Все индикаторы
df = signal_gen.populate_all_indicators(dataframe)

# Entry сигналы
entries = signal_gen.generate_entry_signal(df)

# Exit сигналы
exits = signal_gen.generate_exit_signal(df)
```

---

### risk/correlation.py

**Classes**:
- `CorrelationManager`: Управление корреляцией портфеля
- `DrawdownMonitor`: Circuit breaker для max drawdown

**Usage**:

```python
from risk.correlation import CorrelationManager, DrawdownMonitor

# Correlation manager
manager = CorrelationManager(
    correlation_window=24,
    max_correlation=0.7,
    max_portfolio_heat=0.15
)

# Проверка перед входом
allowed = manager.check_entry_correlation(
    new_pair='ETH/USDT',
    new_pair_data=eth_data,
    open_positions=open_trades,
    all_pairs_data=all_data
)

if not allowed:
    print("❌ Entry blocked: high correlation")

# Drawdown monitor
monitor = DrawdownMonitor(
    max_drawdown=0.15,
    stop_duration_minutes=240
)

# Проверка
trading_allowed = monitor.check_drawdown(
    current_balance=990,
    peak_balance=1000
)
```

---

## Testing

### Unit Tests

```bash
# Тесты shared library
pytest tests/test_signals/ -v

# Тесты parity
pytest tests/test_parity.py -v
```

### Test Example

```python
# tests/test_signals/test_indicators.py

def test_indicator_library():
    from signals.indicators import IndicatorLibrary

    lib = IndicatorLibrary()

    # Test RSI
    rsi = lib.calculate_rsi(sample_close_prices)
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()

def test_signal_generator():
    from signals.indicators import SignalGenerator

    signal_gen = SignalGenerator()

    # Test indicators
    df = signal_gen.populate_all_indicators(sample_dataframe)
    assert 'rsi' in df.columns
    assert 'ema_50' in df.columns

    # Test signals
    entries = signal_gen.generate_entry_signal(df)
    assert entries.isin([0, 1]).all()  # Binary signals
```

---

## Development Guidelines

### 1. Pure Functions

```python
# ✅ Good: Pure function
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """No side effects, testable."""
    return ta.RSI(close, timeperiod=period)

# ❌ Bad: Side effects
def calculate_rsi(self, dataframe):
    """Modifies dataframe directly."""
    dataframe['rsi'] = ta.RSI(dataframe['close'])
    self.last_rsi = dataframe['rsi'].iloc[-1]  # Side effect!
```

### 2. Type Hints

```python
# ✅ Always use type hints
def calculate_correlation(
    pair1_data: pd.DataFrame,
    pair2_data: pd.DataFrame,
    method: str = 'pearson'
) -> float:
    ...
```

### 3. Documentation

```python
def calculate_ema_trio(...) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate EMA trio for trend detection.

    Args:
        close: Close prices
        fast: Fast EMA period (default: 50)
        medium: Medium EMA period (default: 100)
        slow: Slow EMA period (default: 200)

    Returns:
        Tuple of (ema_fast, ema_medium, ema_slow)

    Example:
        >>> ema_50, ema_100, ema_200 = calculate_ema_trio(df['close'])
    """
```

### 4. Error Handling

```python
# ✅ Always handle errors gracefully
def calculate_correlation(...) -> float:
    try:
        return returns1.corr(returns2)
    except Exception as e:
        logger.warning(f"Correlation calc failed: {e}")
        return 0.0  # Safe fallback
```

---

## Adding New Indicators

### Step 1: Add to IndicatorLibrary

```python
# src/signals/indicators.py

class IndicatorLibrary:
    @staticmethod
    def calculate_my_indicator(
        close: pd.Series,
        param1: int = 14
    ) -> pd.Series:
        """
        Calculate my custom indicator.

        Args:
            close: Close prices
            param1: Parameter description

        Returns:
            Indicator series
        """
        # Calculation
        return result
```

### Step 2: Add to SignalGenerator

```python
# src/signals/indicators.py

class SignalGenerator:
    def populate_all_indicators(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # ...
        # Add your indicator
        dataframe['my_indicator'] = self.indicators.calculate_my_indicator(
            dataframe['close'],
            param1=14
        )
        return dataframe
```

### Step 3: Test

```python
# tests/test_signals/test_my_indicator.py

def test_my_indicator():
    from signals.indicators import IndicatorLibrary

    lib = IndicatorLibrary()
    result = lib.calculate_my_indicator(sample_close)

    # Assertions
    assert len(result) == len(sample_close)
    assert not result.isna().all()
```

---

## Roadmap

### Phase 1: ✅ DONE
- Core indicator library
- Signal generation
- Correlation management
- Drawdown monitoring

### Phase 2: 🚧 IN PROGRESS
- ML inference service (async)
- Redis integration
- WebSocket data streaming

### Phase 3: 📋 PLANNED
- Multi-timeframe analysis
- Portfolio optimization
- Advanced risk models
- A/B testing framework

---

## Best Practices

### ✅ DO:

1. **Write pure functions**
2. **Add type hints to everything**
3. **Write tests for all functions**
4. **Document with docstrings**
5. **Handle errors gracefully**
6. **Use logging for important events**

### ❌ DON'T:

1. **Don't modify input dataframes in-place**
2. **Don't use global state**
3. **Don't hardcode values**
4. **Don't skip error handling**
5. **Don't commit without tests**

---

## Support

- **Issues**: [GitHub Issues](https://github.com/kandibobe/hft-algotrade-bot/issues)
- **Docs**: [Main README](../README.md)
- **Testing**: [docs/TESTING_GUIDE.md](../docs/TESTING_GUIDE.md)

---

🏛️ **Stoic Citadel** - Shared library for shared success.
