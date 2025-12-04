# Quick Start Guide

This guide will help you get Stoic Citadel running in under 10 minutes.

## Prerequisites

- Windows 10/11 or Linux
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- PowerShell 5.1+ (Windows) or pwsh (Linux)
- Git

## Step 1: Clone Repository

```powershell
git clone https://github.com/kandibobe/hft-algotrade-bot.git
cd hft-algotrade-bot
```

## Step 2: Run Setup

```powershell
.\stoic.ps1 setup
```

This will:
- Check Docker installation
- Create necessary directories
- Generate .env from template

## Step 3: Configure Secrets

```powershell
.\stoic.ps1 generate-secrets
```

When prompted, type `yes` to automatically update your `.env` file.

## Step 4: Download Historical Data

```powershell
.\stoic.ps1 download-data
```

This downloads 90 days of price data for backtesting.

## Step 5: Start Trading (Paper Mode)

```powershell
.\stoic.ps1 start
```

## Step 6: Access Dashboard

Open http://localhost:3000 in your browser.

Login credentials:
- Username: `stoic_admin`
- Password: (from your .env file `FREQTRADE_API_PASSWORD`)

## What's Next?

1. **Run a backtest**: `.\stoic.ps1 backtest StoicStrategyV1`
2. **View logs**: `.\stoic.ps1 logs freqtrade`
3. **Check health**: `.\stoic.ps1 health`
4. **Open Jupyter**: `.\stoic.ps1 research`

## Common Issues

### Docker not starting
- Make sure Docker Desktop is running
- On Windows, enable WSL2 backend

### Port conflicts
- Edit `.env` to change port numbers
- Default ports: 3000, 8080, 8888, 5433, 9000

### Permission denied
- Run PowerShell as Administrator
- Check Docker has file sharing enabled

## Need Help?

- Run `.\stoic.ps1 help` for all commands
- Check logs: `.\stoic.ps1 logs freqtrade`
- [GitHub Issues](https://github.com/kandibobe/hft-algotrade-bot/issues)
