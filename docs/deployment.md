# Deployment Guide

## Development (Local)

### Quick Start

```powershell
.\stoic.ps1 setup
.\stoic.ps1 generate-secrets
.\stoic.ps1 start
```

### Start All Services

```powershell
.\stoic.ps1 start-all
```

This includes:
- Freqtrade (trading engine)
- FreqUI (dashboard)
- Jupyter Lab (research)
- PostgreSQL (analytics)
- Portainer (management)

## Production

### Prerequisites

- VPS with 4GB+ RAM
- Docker & Docker Compose
- Domain name (optional)

### Setup

1. Clone repository to server
2. Configure `.env` with production values
3. Set `DRY_RUN=false` for live trading
4. Add exchange API credentials

### Security Hardening

```bash
# Use strong passwords
./stoic.ps1 generate-secrets

# Restrict network access
firewall-cmd --add-port=3000/tcp  # Dashboard only

# Use reverse proxy (nginx/traefik)
# Enable HTTPS
```

### Monitoring

```powershell
# Check status
.\stoic.ps1 health

# Continuous monitoring
.\stoic.ps1 status

# View logs
.\stoic.ps1 logs freqtrade
```

### Telegram Alerts

1. Create bot via @BotFather
2. Get chat ID via @userinfobot
3. Add to `.env`:
```
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

4. Enable in config:
```json
"telegram": {
  "enabled": true
}
```

## Live Trading Checklist

- [ ] Backtest shows positive results
- [ ] Walk-forward validation passed
- [ ] Paper trading for 2+ weeks
- [ ] API keys with trade permission only
- [ ] Telegram alerts configured
- [ ] Server monitoring enabled
- [ ] Backup strategy in place
- [ ] Emergency stop procedure documented

## Backup & Recovery

### Backup

```bash
# Backup configuration
tar -czf backup-$(date +%Y%m%d).tar.gz \
  .env \
  user_data/config/ \
  user_data/strategies/

# Backup database
docker compose exec postgres pg_dump -U stoic_trader trading_analytics > backup.sql
```

### Restore

```bash
# Restore files
tar -xzf backup-YYYYMMDD.tar.gz

# Restore database
cat backup.sql | docker compose exec -T postgres psql -U stoic_trader trading_analytics
```

## Updating

```bash
git pull
docker compose pull
.\stoic.ps1 restart
```
