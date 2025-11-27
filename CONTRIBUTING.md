# Contributing to Stoic Citadel 🏛️

First off, thank you for considering contributing to Stoic Citadel! It's people like you that make this project better.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)

## 📜 Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## 🤝 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected vs actual behavior**
- **Screenshots** if applicable
- **Environment details** (OS, Docker version, Python version)

### 💡 Suggesting Features

Feature requests are welcome! Please provide:

- **Clear description** of the feature
- **Use case** - why is this needed?
- **Possible implementation** approach

### 🔧 Code Contributions


1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/hft-algotrade-bot.git
cd hft-algotrade-bot

# Run setup
make setup

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Start development environment
make research
```

## 📝 Style Guidelines

### Python Code Style

- **Formatter**: Black (line length: 88)
- **Linter**: Flake8
- **Type Checker**: MyPy
- **Docstrings**: Google style

```python
def calculate_signal(
    dataframe: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate trading signal based on RSI.

    Args:
        dataframe: OHLCV dataframe
        period: RSI period (default: 14)

    Returns:
        DataFrame with signal column added
    """
    # Implementation
    pass
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new RSI divergence indicator
fix: correct stoploss calculation
docs: update installation guide
test: add unit tests for strategy
refactor: simplify signal generation
```

## 🔄 Pull Request Process

1. **Ensure** all tests pass: `make test`
2. **Run** linters: `make lint`
3. **Update** documentation if needed
4. **Add** tests for new features
5. **Request** review from maintainers

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Lint checks pass
- [ ] No hardcoded secrets
- [ ] Risk limits unchanged (unless intentional)

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific tests
make test-unit
make test-integration

# Run with coverage
make test-coverage
```

## 📚 Documentation

- Keep README.md updated
- Add docstrings to all functions
- Update docs/ for major features
- Include examples where helpful

## ⚠️ Important Notes

### Risk Management

**NEVER** modify these without explicit approval:
- `stoploss` value
- `max_open_trades` limit
- `max_drawdown` protection
- `tradable_balance_ratio`

### Security

- Never commit API keys or secrets
- Use environment variables
- Check `.gitignore` before committing

---

Thank you for contributing! 🎉
