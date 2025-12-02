# 🏛️ Stoic Citadel - Professional HFT Algorithmic Trading Bot

<div align="center">

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Freqtrade](https://img.shields.io/badge/freqtrade-2024.11-orange.svg)

**Trade with wisdom, not emotion**

[Features](#-features) • [Quick Start](#-quick-start-windows) • [Documentation](#-documentation) • [Strategies](#-trading-strategies) • [Support](#-support)

</div>

---

## 📋 Overview

Stoic Citadel is a professional-grade algorithmic trading bot built on Freqtrade with comprehensive Docker infrastructure, automated testing, and a complete 6-week learning path for beginners.

### Key Highlights

- 🤖 **3 Production-Ready Strategies** with backtesting and optimization
- 📊 **Complete Monitoring Stack** (Grafana + Prometheus)
- 🔬 **Jupyter Lab Integration** for research and development
- 🧪 **Automated Testing Framework** with walk-forward validation
- 📈 **Intelligent Evaluation System** (automated strategy scoring)
- 🛡️ **Enterprise-Grade Security** with multiple protection layers
- 📚 **6-Week Learning Path** from beginner to production

---

## ✨ Features

### Trading Infrastructure

- **Freqtrade 2024.11** - Professional trading execution engine
- **FreqUI Dashboard** - Real-time monitoring and control
- **PostgreSQL Analytics** - Trade data persistence and analysis
- **Portainer** - Container management interface
- **Health Checks** - Automated system monitoring

### Strategies (3 Production-Ready)

1. **StoicStrategyV1** - Conservative multi-indicator approach
2. **StoicCitadelV2** - Advanced technical analysis
3. **StoicEnsembleStrategy** - ML-enhanced ensemble method

### Automation & Testing

- **Automated Backtesting** with performance scoring
- **Walk-Forward Validation** for robustness testing
- **A/B Testing Framework** (baseline comparison)
- **Hyperparameter Optimization** with Hyperopt
- **Automated Reports** in TXT format

### Monitoring & Alerts

- **Grafana Dashboards** - Visual performance tracking
- **Prometheus Metrics** - System and trading metrics
- **Telegram Notifications** - Real-time trade alerts
- **Health Checks** - System status monitoring

---

## 🚀 Quick Start (Windows)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) for Windows
- Windows 10/11 with PowerShell 5.1+
- [Binance Account](https://www.binance.com/) with API access

### Installation (20 minutes)

```powershell
# 1. Clone repository
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot

# 2. Setup (interactive wizard)
.\stoic.ps1 setup

# 3. Configure API keys
# Edit .env file with your Binance API credentials

# 4. Download historical data
.\stoic.ps1 download-data

# 5. Start paper trading
.\stoic.ps1 trade-dry

# 6. Open dashboard
.\stoic.ps1 dashboard
```

**Access Points:**
- FreqUI Dashboard: http://localhost:3000 (stoic_admin / password_from_env)
- Jupyter Lab: http://localhost:8888 (token: stoic2024)
- Portainer: http://localhost:9000

---

## 📚 Documentation

### Getting Started (Read in Order)

1. **[START_HERE.md](START_HERE.md)** - Visual overview and quick start
2. **[TODO_FOR_YOU.md](TODO_FOR_YOU.md)** - What to do right now
3. **[HOW_TO_USE.md](HOW_TO_USE.md)** - Complete usage guide
4. **[DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)** - 6-week roadmap

### Working Documents

- **[TRADING_JOURNAL.md](TRADING_JOURNAL.md)** - Daily observation template
- **[CHECKLIST.md](CHECKLIST.md)** - Launch checklist
- **[QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md)** - Detailed Windows guide

### Reference

- **[ROADMAP.txt](ROADMAP.txt)** - Visual roadmap
- **[ALL_SET.md](ALL_SET.md)** - Complete resource overview
- **[SETUP_SUMMARY_FINAL.md](SETUP_SUMMARY_FINAL.md)** - Setup summary

### Quick Access

```powershell
.\docs.ps1 menu      # Show documentation menu
.\docs.ps1 start     # Open START_HERE.md
.\docs.ps1 plan      # Open development plan
.\docs.ps1 journal   # Open trading journal
```

---

## 🎯 Trading Strategies

### StoicStrategyV1 (Default)
- **Type:** Conservative multi-indicator
- **Indicators:** RSI, MACD, Bollinger Bands
- **Risk:** Low-Medium
- **Best For:** Stable markets

### StoicCitadelV2
- **Type:** Advanced technical analysis
- **Indicators:** RSI, EMA, Volume, ATR
- **Risk:** Medium
- **Best For:** Trending markets

### StoicEnsembleStrategy
- **Type:** ML-enhanced ensemble
- **Indicators:** Multiple + Machine Learning
- **Risk:** Medium-High
- **Best For:** Experienced traders

---

## 🧪 Testing & Validation

### Quick Test (7 days)
```powershell
.\test.ps1 quick StoicStrategyV1
```

### Standard Test (30 days)
```powershell
.\test.ps1 standard StoicCitadelV2
```

### Compare All Strategies
```powershell
.\test.ps1 compare-all
```

### Walk-Forward Validation
```powershell
.\test.ps1 walk-forward StoicEnsembleStrategy
```

**Automated Evaluation:**
- 🟢 8+ points = Excellent! Ready for production
- 🟡 5-9 points = Good, room for improvement  
- 🟠 0-4 points = Mediocre, needs optimization
- 🔴 <0 points = Poor, redesign needed

---

## 📊 Management Commands

### Core Operations

```powershell
.\stoic.ps1 start           # Start all services
.\stoic.ps1 stop            # Stop all services
.\stoic.ps1 status          # Show service status
.\stoic.ps1 logs            # View logs
```

### Trading

```powershell
.\stoic.ps1 trade-dry       # Paper trading (DRY_RUN)
.\stoic.ps1 trade-live      # LIVE trading (CAUTION!)
.\stoic.ps1 backtest        # Run backtest
.\stoic.ps1 hyperopt        # Optimize parameters
```

### Research

```powershell
.\stoic.ps1 research        # Jupyter Lab
.\stoic.ps1 download-data   # Download market data
.\stoic.ps1 verify-data     # Verify data quality
```

### Monitoring

```powershell
.\stoic.ps1 dashboard       # Open FreqUI
.\stoic.ps1 monitoring      # Start Grafana+Prometheus
.\health.ps1                # System health check
```

---

## 🛡️ Security Best Practices

### API Configuration

✅ **DO:**
- Use "Enable Reading" permission only for dry-run
- Enable "Spot & Margin Trading" only for live trading
- Set IP whitelist to your VPS/home IP
- Enable 2FA on Binance account
- Keep API keys in .env (never commit!)

❌ **NEVER:**
- Enable "Enable Withdrawals" permission
- Share API keys or commit .env to git
- Use same keys for testing and production
- Skip 2FA on exchange account

### Trading Safety

- Start with **DRY_RUN=true** (paper trading)
- Test minimum **2 weeks** before going live
- Begin with **small capital** ($100-200)
- Set appropriate **risk limits** in config
- Monitor **Telegram notifications**
- Review **logs regularly**

---

## 📈 6-Week Learning Path

### Week 1: Foundation
- Setup infrastructure
- Understand dashboard
- First backtest

### Week 2-3: Testing
- Daily monitoring
- Keep trading journal
- Test on different periods

### Week 4-5: Optimization
- Parameter experiments
- Hyperopt optimization
- Strategy improvements

### Week 6: Production
- Final testing (7 days dry-run)
- Security configuration
- First live trading (small capital!)

**Details:** [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)

---

## 🔧 Project Structure

```
hft-algotrade-bot/
├── user_data/
│   ├── strategies/          # 3 trading strategies
│   ├── config/              # Trading configurations
│   └── data/                # Historical market data
│
├── scripts/
│   ├── setup_wizard.py      # Interactive setup
│   ├── walk_forward.py      # Walk-forward validation
│   └── download_data.sh     # Data downloader
│
├── research/                # Jupyter notebooks
├── monitoring/              # Grafana/Prometheus configs
├── docs/                    # Additional documentation
│
├── stoic.ps1                # Main management script
├── test.ps1                 # Testing automation
├── docs.ps1                 # Documentation navigator
├── health.ps1               # Health check script
│
└── docker-compose.yml       # Main infrastructure
```

---

## 🤝 Support

### Documentation
- Start with [START_HERE.md](START_HERE.md)
- Follow [TODO_FOR_YOU.md](TODO_FOR_YOU.md)
- Read [HOW_TO_USE.md](HOW_TO_USE.md)

### Issues
- Check existing [Issues](https://github.com/kandibobe/hft-algotrade-bot/issues)
- Review [QUICKSTART_WINDOWS.md](QUICKSTART_WINDOWS.md) troubleshooting
- Run `.\health.ps1` for diagnostic

### Community
- [Freqtrade Documentation](https://www.freqtrade.io/en/stable/)
- [Freqtrade Discord](https://discord.gg/freqtrade)
- [r/algotrading](https://reddit.com/r/algotrading)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**IMPORTANT:** This software is for educational purposes. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose.

- Past performance does not guarantee future results
- Always test extensively in dry-run mode first
- Start with minimal capital when going live
- Use proper risk management
- Monitor your trades regularly

The developers are not responsible for any financial losses.

---

## 🌟 Acknowledgments

- [Freqtrade](https://www.freqtrade.io/) - Core trading engine
- [Binance](https://www.binance.com/) - Exchange integration
- [Docker](https://www.docker.com/) - Containerization
- [Grafana](https://grafana.com/) - Monitoring dashboards

---

<div align="center">

🏛️ **Stoic Citadel** - Trade with wisdom, not emotion

**[⬆ Back to Top](#-stoic-citadel---professional-hft-algorithmic-trading-bot)**

</div>
