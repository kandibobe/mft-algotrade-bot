# Strategy Development Guide

## Available Strategies

### StoicStrategyV1 (Default)

Trend-following strategy with BTC market regime filter.

**Logic:**
1. Only trade when BTC > EMA200 (daily)
2. Enter when RSI oversold + stochastic confirmation
3. Exit when RSI overbought or trend reversal

**Parameters:**
- `buy_rsi_threshold`: 20-40 (default: 30)
- `sell_rsi_threshold`: 60-80 (default: 70)
- `regime_ema_period`: 150-250 (default: 200)

### StoicCitadelV2

Advanced strategy with multi-timeframe analysis.

### StoicEnsembleStrategy

Combines multiple strategies with voting mechanism.

## Creating Custom Strategy

1. Copy template:
```bash
cp user_data/strategies/StoicStrategyV1.py user_data/strategies/MyStrategy.py
```

2. Modify class name and logic
3. Test with backtest:
```powershell
.\stoic.ps1 backtest MyStrategy
```

## Backtesting

### Basic Backtest

```powershell
.\stoic.ps1 backtest StoicStrategyV1
```

### Custom Timerange

```powershell
docker compose run --rm freqtrade backtesting \
  --strategy StoicStrategyV1 \
  --timerange 20240601-20241101 \
  --enable-protections
```

### HyperOpt Optimization

```powershell
docker compose run --rm freqtrade hyperopt \
  --strategy StoicStrategyV1 \
  --hyperopt-loss SharpeHyperOptLoss \
  --spaces buy sell \
  --epochs 100
```

## Walk-Forward Optimization

Use the walk-forward script for proper out-of-sample testing:

```powershell
python scripts/walk_forward.py --strategy StoicStrategyV1
```

## Risk Management

All strategies include:
- Hard stoploss: -5%
- ROI targets: 6%/4%/2%/1%
- Dynamic position sizing based on ATR
- Low liquidity hour avoidance (UTC 0-5)
- Emergency timeout exits

## Performance Metrics

Key metrics to watch:
- Sharpe Ratio (>1.0 good, >2.0 excellent)
- Max Drawdown (<20% preferred)
- Win Rate (>50% with good R:R)
- Profit Factor (>1.5)
