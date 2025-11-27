# Security Policy 🔒

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **DO NOT** create a public GitHub issue
2. **Email** security concerns to the maintainers
3. **Include** detailed description and steps to reproduce
4. **Allow** reasonable time for a fix before public disclosure

## 🔐 Security Best Practices

### API Keys & Secrets

```bash
# ✅ DO: Use environment variables
export BINANCE_API_KEY="your_key"
export BINANCE_SECRET="your_secret"

# ❌ DON'T: Hardcode in config files
{
  "exchange": {
    "key": "abc123..."  # NEVER DO THIS
  }
}
```

### Exchange API Configuration

1. **Create API keys with minimal permissions**:
   - ✅ Enable: Spot Trading
   - ❌ Disable: Withdrawals
   - ❌ Disable: Futures (unless needed)

2. **IP Whitelist** your server's IP address

3. **Enable 2FA** on your exchange account


### Docker Security

```yaml
# docker-compose.yml - Security recommendations
services:
  freqtrade:
    # Run as non-root user
    user: "1000:1000"
    # Read-only filesystem where possible
    read_only: true
    # Drop all capabilities
    cap_drop:
      - ALL
    # No privilege escalation
    security_opt:
      - no-new-privileges:true
```

### Network Security

1. **Firewall**: Only expose necessary ports
2. **VPN**: Consider running bot behind VPN
3. **SSH**: Use key-based authentication only

## 🛡️ Risk Management Security

These settings are **critical** and should never be weakened:

```python
# StoicEnsembleStrategy.py
stoploss = -0.05          # Maximum 5% loss per trade
trailing_stop = True       # Lock in profits
max_open_trades = 3        # Limit exposure

protections = [
    {
        "method": "MaxDrawdown",
        "max_allowed_drawdown": 0.15  # 15% max drawdown
    },
    {
        "method": "StoplossGuard",
        "trade_limit": 3,
        "stop_duration_candles": 24
    }
]
```

## 🔍 Security Checklist

Before deploying to production:

- [ ] API keys use environment variables
- [ ] Exchange has IP whitelist enabled
- [ ] Exchange has 2FA enabled
- [ ] Withdrawal permissions disabled on API
- [ ] `.env` file is in `.gitignore`
- [ ] No secrets in git history
- [ ] Docker runs as non-root
- [ ] Firewall configured
- [ ] Monitoring and alerts enabled
- [ ] Regular backups configured

## 📋 Audit Log

Keep track of all security-related changes:

| Date | Change | Author |
|------|--------|--------|
| 2024-01 | Initial security policy | Team |

---

**Remember**: No amount of profit is worth compromising security. When in doubt, prioritize safety.
