# Stoic Citadel - HFT Algorithmic Trading Bot

A production-ready cryptocurrency trading bot built on [Freqtrade](https://www.freqtrade.io/) with market regime detection, walk-forward optimization, and comprehensive risk management.

## Quick Start

```powershell
# Clone the repository
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot

# Run setup wizard
.\stoic.ps1 setup

# Generate secure passwords
.\stoic.ps1 generate-secrets

# Download historical data
.\stoic.ps1 download-data

# Start trading (paper mode by default)
.\stoic.ps1 start
```

## Features

**Trading Strategies**
- `StoicStrategyV1` - Trend following with BTC regime filter
- `StoicCitadelV2` - Advanced multi-timeframe analysis
- `StoicEnsembleStrategy` - Multiple strategy combination

**Risk Management**
- Market regime detection (only trade in BTC bull markets)
- Dynamic position sizing based on volatility
- Emergency exits and timeout protection
- Stoploss on exchange

**Infrastructure**
- Docker-based deployment
- FreqUI web dashboard
- Jupyter Lab for research
- PostgreSQL for analytics
- Telegram notifications

## Architecture

```
├── user_data/
│   ├── config/          # Trading configurations
│   ├── strategies/      # Trading strategies
│   ├── data/            # Historical price data
│   └── logs/            # Application logs
├── scripts/             # Automation scripts
├── src/                 # Extended modules
├── tests/               # Test suite
├── docker/              # Docker configurations
└── docs/                # Documentation
```

## Management Commands

```powershell
# Setup & Configuration
.\stoic.ps1 setup              # Initial setup
.\stoic.ps1 generate-secrets   # Generate passwords

# Service Control
.\stoic.ps1 start              # Start trading services
.\stoic.ps1 start-all          # Start all (incl. research)
.\stoic.ps1 stop               # Stop services
.\stoic.ps1 restart            # Restart services

# Monitoring
.\stoic.ps1 status             # Service status
.\stoic.ps1 health             # Health check
.\stoic.ps1 logs [service]     # View logs
.\stoic.ps1 dashboard          # Open FreqUI

# Trading Operations
.\stoic.ps1 backtest [strat]   # Run backtest
.\stoic.ps1 download-data      # Download data

# Research
.\stoic.ps1 research           # Start Jupyter Lab

# Maintenance
.\stoic.ps1 clean              # Remove containers
.\stoic.ps1 reset              # Full reset (DELETES DATA!)
```

## Configuration

1. Copy `.env.example` to `.env`
2. Set required passwords (or run `.\stoic.ps1 generate-secrets`)
3. Add your exchange API keys for live trading

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FREQTRADE_API_PASSWORD` | Yes | API access password |
| `POSTGRES_PASSWORD` | Yes | Database password |
| `JUPYTER_TOKEN` | Yes | Jupyter access token |
| `BINANCE_API_KEY` | Live only | Exchange API key |
| `BINANCE_API_SECRET` | Live only | Exchange API secret |
| `TELEGRAM_BOT_TOKEN` | No | Telegram notifications |

## Web Interfaces

| Service | URL | Description |
|---------|-----|-------------|
| FreqUI | http://localhost:3000 | Trading dashboard |
| Jupyter | http://localhost:8888 | Research environment |
| Portainer | http://localhost:9000 | Container management |
| PostgreSQL | localhost:5433 | Analytics database |

## Development

```powershell
# Run tests
python -m pytest tests/ -v

# Backtest a strategy
.\stoic.ps1 backtest StoicStrategyV1

# Open research environment
.\stoic.ps1 research
```

## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [Strategy Development](docs/strategies.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api.md)

## Requirements

- Windows 10/11 or Linux
- Docker Desktop
- PowerShell 5.1+ (Windows) or pwsh (Linux)
- 8GB RAM minimum
- 20GB free disk space

## Safety

⚠️ **WARNING**: Cryptocurrency trading involves significant risk.

- Always start with `dry_run: true` (paper trading)
- Never invest more than you can afford to lose
- Test thoroughly before live trading
- Monitor your bot regularly

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- [GitHub Issues](https://github.com/kandibobe/hft-algotrade-bot/issues)
- [Freqtrade Discord](https://discord.gg/freqtrade)
