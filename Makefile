# Stoic Citadel MFT System - Makefile
# Optimized for Python 3.10+ and Cross-Platform Compatibility

# --- Configuration & Variables ---
PYTHON := .venv/Scripts/python.exe
PIP := .venv/Scripts/pip.exe
PYTEST := .venv/Scripts/pytest.exe
BLACK := .venv/Scripts/black.exe
RUFF := .venv/Scripts/ruff.exe
MYPY := .venv/Scripts/mypy.exe
MKDOCS := .venv/Scripts/mkdocs.exe

# Fallback for non-Windows systems
ifeq ($(OS),Windows_NT)
    # Windows-specific settings are already defaults above
else
    PYTHON := .venv/bin/python
    PIP := .venv/bin/pip
    PYTEST := .venv/bin/pytest
    BLACK := .venv/bin/black
    RUFF := .venv/bin/ruff
    MYPY := .venv/bin/mypy
    MKDOCS := .venv/bin/mkdocs
endif

DOCKER_COMPOSE_BASE := docker-compose -f deploy/docker-compose.yml
DOCKER_COMPOSE_PROD := docker-compose -f deploy/docker-compose.prod.yml
DOCKER_COMPOSE_TEST := docker-compose -f deploy/docker-compose.test.yml

STRATEGY := StoicEnsembleStrategyV6
CONFIG := user_data/config/config_paper.json
TIMEFRAME := 5m

.PHONY: help setup install dev-install update lint format check-types test test-cov clean \
        docker-build docker-up docker-down docker-logs \
        backtest hyperopt docs docs-serve chaos verify

# --- General Targets ---

help: ## Show this help message
	@$(PYTHON) -c "import re; \
	print('{:<20} {:<}'.format('Target', 'Description')); \
	print('-' * 40); \
	for line in open('Makefile'): \
		match = re.search(r'^([a-zA-Z_-]+):.*?## (.*)$$', line); \
		if match: \
			print('{:<20} {:<}'.format(match.group(1), match.group(2)))"

setup: ## Create virtual environment and install dependencies
	python -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e .

dev-install: setup ## Install development dependencies
	$(PIP) install -e .[dev,research,freqtrade]
	pre-commit install

update: ## Update dependencies
	$(PIP) install --upgrade -e .[dev,research,freqtrade]

# --- Quality Assurance ---

lint: ## Run ruff and black check
	$(RUFF) check src/ tests/
	$(BLACK) --check src/ tests/

format: ## Format code with ruff and black
	$(RUFF) check --fix src/ tests/
	$(BLACK) src/ tests/

check-types: ## Run static type checking with mypy
	$(MYPY) src/

test: ## Run all tests
	$(PYTEST) tests/

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/

test-cov: ## Run tests with coverage report
	$(PYTEST) --cov=src tests/ --cov-report=html --cov-report=term

clean: ## Clean temporary files and caches
	@$(PYTHON) -c "import shutil, os; \
	[shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache', '.coverage', 'htmlcov', 'dist', 'build', '.mypy_cache', 'stoic_citadel.egg-info']]; \
	[os.remove(f) for f in os.listdir('.') if f.endswith('.pyc') or f.endswith('.pyo')]"
	@$(PYTHON) -c "import os; \
	for root, dirs, files in os.walk('.'): \
		for d in dirs: \
			if d == '__pycache__': \
				import shutil; shutil.rmtree(os.path.join(root, d), ignore_errors=True)"

# --- Docker Management ---

docker-build: ## Build docker images
	$(DOCKER_COMPOSE_BASE) build

docker-up: ## Start services in background
	$(DOCKER_COMPOSE_BASE) up -d

docker-down: ## Stop and remove containers
	$(DOCKER_COMPOSE_BASE) down

docker-logs: ## View service logs
	$(DOCKER_COMPOSE_BASE) logs -f

# --- Freqtrade Operations ---

backtest: ## Run Freqtrade backtest (use STRATEGY and CONFIG vars)
	$(PYTHON) -m freqtrade backtesting --strategy $(STRATEGY) --config $(CONFIG) --timerange 20230101-

hyperopt: ## Run Freqtrade hyperopt (use STRATEGY and CONFIG vars)
	$(PYTHON) -m freqtrade hyperopt --strategy $(STRATEGY) --config $(CONFIG) --hyperopt-loss AdvancedSortinoLoss --epochs 100

# --- Documentation ---

docs: ## Build documentation
	$(MKDOCS) build

docs-serve: ## Serve documentation locally
	$(MKDOCS) serve

# --- Project Specific ---

chaos: ## Run chaos tests
	$(PYTHON) scripts/risk/chaos_test.py

verify: ## Verify deployment
	$(PYTHON) scripts/verify_deployment.py
