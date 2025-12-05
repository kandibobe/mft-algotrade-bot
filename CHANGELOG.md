# Changelog

All notable changes to Stoic Citadel will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-XX-XX

### Added

#### Infrastructure (Phase A)
- `docker-compose.backtest.yml` - Isolated backtest environment
- `config_backtest.json` - Backtest-only configuration without API keys
- `Makefile.backtest` - Quick commands for backtesting workflow
- Smoke test fixture data for CI

#### Data Pipeline (Phase B)
- `src/data/loader.py` - Unified OHLCV data loading (CSV, Feather, JSON)
- `src/data/downloader.py` - Historical data download wrapper
- `src/data/validator.py` - Data validation and integrity checks
- Dataset versioning with hash/metadata

#### Indicators Library (Phase C)
- `src/utils/indicators.py` - Vectorized technical indicators
  - EMA, SMA, RSI, MACD, ATR, Bollinger Bands
  - Stochastic, ADX, VWAP, OBV
  - `calculate_all_indicators()` convenience function
- All indicators use pandas/numpy vectorization (no loops)

#### Risk Management (Phase C)
- `src/utils/risk.py` - Position sizing and risk metrics
  - Fixed risk position sizing
  - Kelly criterion position sizing
  - Sharpe, Sortino, Calmar ratios
  - Max drawdown calculation

#### Strategy Refactoring (Phase D)
- `src/strategies/base_strategy.py` - Base class for all strategies
- `src/strategies/strategy_config.py` - Configuration management
  - YAML/JSON config file support
  - Environment variable overrides
  - Validation

#### Improved Strategy (Phase E)
- `StoicEnsembleStrategyV2` - Advanced ensemble strategy
  - Momentum sub-strategy (EMA crossover + ADX)
  - Mean reversion sub-strategy (RSI + BB)
  - Breakout sub-strategy (volatility expansion)
  - Ensemble voting (2/3 agreement required)
  - Regime-aware parameter adjustment
  - Dynamic position sizing based on volatility
  - Time-of-day filters

#### Regime Detection (Phase E)
- `src/utils/regime_detection.py` - Market regime analysis
  - Trend regime detection (bull/bear/ranging)
  - Volatility regime detection
  - Composite regime score (0-100)
  - Parameter adjustment recommendations

#### Walk-Forward Optimization (Phase F)
- `scripts/walk_forward_optimization.py` - WFO pipeline
  - Rolling window optimization
  - Train/Validate/Test splits
  - Result aggregation and robustness check

#### Testing & CI (Phase G)
- Comprehensive unit tests for indicators
- Integration tests for data pipeline
- Strategy validation tests
- Updated CI workflow with smoke tests
- pytest configuration

#### Documentation
- Updated README with quick start guide
- Strategy configuration documentation
- API documentation for modules

### Changed
- CI workflow now runs actual smoke tests
- Improved code organization in `src/` directory
- Standardized indicator calculations

### Fixed
- Data validation edge cases
- NaN handling in indicators
- Type hints throughout codebase

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- Basic Freqtrade setup
- StoicStrategyV1 and StoicEnsembleStrategy
- Docker configuration
- Basic CI/CD pipeline
