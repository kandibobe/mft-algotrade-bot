# Contributing to Stoic Citadel 🏛️

First, thank you for considering contributing to Stoic Citadel. Every contribution, from a typo fix to a new feature, helps make this project better.

This document provides a guide for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process.

## 📜 Code of Conduct
This project and everyone participating in it is governed by the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 💬 Community & Support
*(Placeholder: Add links to your Discord, Slack, or other community channels here.)*

---

## 🚀 How to Contribute

There are many ways to contribute, including:
- Reporting bugs
- Suggesting enhancements
- Improving documentation
- Submitting pull requests for new features or bug fixes

### 🐛 Reporting Bugs
Before creating a bug report, please check the existing [issues](https://github.com/kandibobe/mft-algotrade-bot/issues) to see if the bug has already been reported. If not, create a new issue using our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).

### ✨ Suggesting Features
We welcome new ideas! Please use our [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md) to submit your suggestions.

---

## 🛠️ Development Workflow

### 1. Project Philosophy
Before you start coding, it's important to understand the core principles of Stoic Citadel:
- **Robustness over Performance:** The system must be fail-safe. Risk management and stability are the highest priorities.
- **Decoupled Architecture:** Maintain the separation between the Macro (strategy) and Micro (execution) layers.
- **Data-Driven Decisions:** All new features and strategies should be backed by rigorous backtesting and data analysis.

### 2. Setup Your Environment
```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/mft-algotrade-bot.git
cd mft-algotrade-bot

# 3. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install pre-commit hooks
pre-commit install
```

### 3. Branching Strategy
Create a new branch for each feature or bug fix. Use the following naming convention:
- `feature/<feature-name>` (e.g., `feature/add-new-indicator`)
- `fix/<issue-number>-<short-description>` (e.g., `fix/123-resolve-orderbook-leak`)
- `docs/<topic>` (e.g., `docs/update-risk-management-guide`)

### 4. Writing Code
- Follow the existing code style.
- Write clear, concise, and well-documented code.
- Add or update unit and integration tests as necessary.

### 5. Submitting a Pull Request
1.  Ensure all tests and linting checks pass:
    ```bash
    make lint
    make test
    ```
2.  Push your branch to your fork and submit a pull request to the `main` branch of the main repository.
3.  Fill out the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md) completely.
4.  The PR will be reviewed, and you may be asked to make changes.

---

## ✅ Testing Guidelines

- **Unit Tests (`tests/unit`):** Test individual functions and classes in isolation.
- **Integration Tests (`tests/integration`):** Test how different parts of the system work together (e.g., the flow from strategy signal to order execution).
- **End-to-End (E2E) Tests (`tests/e2e`):** Test the full system in a simulated environment.

All new features should be accompanied by appropriate tests.

---

## ✍️ Style Guide

### Code
- **Formatter:** Black
- **Linter:** Flake8
- **Docstrings:** Google style

### Commit Messages
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This helps us automate changelog generation and makes the commit history more readable.

- `feat:` A new feature.
- `fix:` A bug fix.
- `docs:` Documentation only changes.
- `style:` Changes that do not affect the meaning of the code (white-space, formatting, etc).
- `refactor:` A code change that neither fixes a bug nor adds a feature.
- `test:` Adding missing tests or correcting existing tests.
- `chore:` Changes to the build process or auxiliary tools.

Example: `feat(risk): add new circuit breaker for market volatility`

---

Thank you for your contribution!
