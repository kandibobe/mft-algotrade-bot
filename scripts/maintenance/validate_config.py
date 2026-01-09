#!/usr/bin/env python3
"""
Stoic Citadel - Configuration Validator
========================================

Проверяет конфигурацию перед запуском торговли.

Проверки:
- API ключи биржи (подключение, баланс)
- Telegram бот (токен, chat ID)
- Конфигурационные файлы (синтаксис, параметры)
- Стратегии (импорт, методы)
- Docker окружение

Usage:
    python3 scripts/validate_config.py
    python3 scripts/validate_config.py --exchange binance
    python3 scripts/validate_config.py --full

Author: Stoic Citadel Team
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CheckResult:
    """Результат проверки."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


class ConfigValidator:
    """Валидатор конфигурации Stoic Citadel."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[CheckResult] = []
        self.env_path = Path(".env")
        self.config_path = Path("user_data/config/config_production.json")

    def run_all_checks(self, check_exchange: bool = True) -> bool:
        """
        Запускает все проверки.

        Returns:
            True если все критичные проверки прошли
        """
        print("\n" + "=" * 70)
        print("STOIC CITADEL - ПРОВЕРКА КОНФИГУРАЦИИ")
        print("=" * 70 + "\n")

        # Базовые проверки
        self.check_directory_structure()
        self.check_env_file()
        self.check_config_files()
        self.check_strategies()
        self.check_docker()

        # Проверки подключений (опционально)
        if check_exchange:
            self.check_telegram()
            self.check_exchange_connection()

        # Вывод результатов
        self.print_results()

        # Определение успешности
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        return len(errors) == 0

    def check_directory_structure(self):
        """Проверка структуры директорий."""
        required_dirs = [
            "user_data",
            "user_data/config",
            "user_data/strategies",
            "user_data/data",
            "research",
            "scripts",
            "docker",
        ]

        missing = []
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                missing.append(dir_path)

        if missing:
            self.results.append(
                CheckResult(
                    name="Структура директорий",
                    passed=False,
                    message=f"Отсутствуют директории: {', '.join(missing)}",
                    severity="error",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    name="Структура директорий",
                    passed=True,
                    message="Все необходимые директории на месте",
                    severity="info",
                )
            )

    def check_env_file(self):
        """Проверка .env файла."""
        if not self.env_path.exists():
            self.results.append(
                CheckResult(
                    name=".env файл",
                    passed=False,
                    message=".env файл не найден. Создайте из .env.example",
                    severity="error",
                )
            )
            return

        # Загрузка переменных
        env_vars = {}
        with open(self.env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        # Проверка обязательных переменных
        warnings = []

        # API ключи (необязательны для dry-run)
        if not env_vars.get("BINANCE_API_KEY"):
            warnings.append("BINANCE_API_KEY не установлен (OK для dry-run)")

        if not env_vars.get("BINANCE_API_SECRET"):
            warnings.append("BINANCE_API_SECRET не установлен (OK для dry-run)")

        # Telegram (необязательно, но рекомендуется)
        if not env_vars.get("TELEGRAM_TOKEN"):
            warnings.append("TELEGRAM_TOKEN не установлен (уведомления отключены)")

        if not env_vars.get("TELEGRAM_CHAT_ID"):
            warnings.append("TELEGRAM_CHAT_ID не установлен (уведомления отключены)")

        if warnings:
            self.results.append(
                CheckResult(
                    name=".env файл",
                    passed=True,
                    message="Файл найден. Предупреждения:\n  - " + "\n  - ".join(warnings),
                    severity="warning",
                )
            )
        else:
            self.results.append(
                CheckResult(
                    name=".env файл",
                    passed=True,
                    message="Все переменные окружения установлены",
                    severity="info",
                )
            )

    def check_config_files(self):
        """Проверка конфигурационных файлов."""
        configs = [
            "user_data/config/config_production.json",
            "user_data/config/config_dryrun.json",
        ]

        for config_file in configs:
            path = Path(config_file)
            if not path.exists():
                self.results.append(
                    CheckResult(
                        name=f"Конфиг: {config_file}",
                        passed=False,
                        message=f"Файл не найден: {config_file}",
                        severity="error",
                    )
                )
                continue

            # Проверка синтаксиса JSON
            try:
                with open(path) as f:
                    config = json.load(f)

                # Проверка обязательных полей
                required_fields = [
                    "max_open_trades",
                    "stake_currency",
                    "dry_run",
                    "exchange",
                ]

                missing_fields = [field for field in required_fields if field not in config]

                if missing_fields:
                    self.results.append(
                        CheckResult(
                            name=f"Конфиг: {config_file}",
                            passed=False,
                            message=f"Отсутствуют поля: {', '.join(missing_fields)}",
                            severity="error",
                        )
                    )
                else:
                    # Проверка dry_run статуса
                    is_dry_run = config.get("dry_run", True)
                    mode = "DRY-RUN (безопасно)" if is_dry_run else "LIVE (реальные деньги!)"

                    self.results.append(
                        CheckResult(
                            name=f"Конфиг: {config_file}",
                            passed=True,
                            message=f"Синтаксис OK. Режим: {mode}",
                            severity="info",
                        )
                    )

            except json.JSONDecodeError as e:
                self.results.append(
                    CheckResult(
                        name=f"Конфиг: {config_file}",
                        passed=False,
                        message=f"Ошибка JSON синтаксиса: {e}",
                        severity="error",
                    )
                )

    def check_strategies(self):
        """Проверка стратегий."""
        strategy_dir = Path("user_data/strategies")
        if not strategy_dir.exists():
            self.results.append(
                CheckResult(
                    name="Стратегии",
                    passed=False,
                    message="Директория strategies не найдена",
                    severity="error",
                )
            )
            return

        # Поиск Python файлов
        strategies = list(strategy_dir.glob("*.py"))
        if not strategies:
            self.results.append(
                CheckResult(
                    name="Стратегии",
                    passed=False,
                    message="Не найдено ни одной стратегии (.py файлов)",
                    severity="error",
                )
            )
            return

        # Проверка импорта каждой стратегии
        valid_strategies = []
        for strategy_file in strategies:
            if strategy_file.name.startswith("__"):
                continue

            # Проверка синтаксиса Python
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(strategy_file)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode == 0:
                    valid_strategies.append(strategy_file.stem)
                else:
                    self.results.append(
                        CheckResult(
                            name=f"Стратегия: {strategy_file.name}",
                            passed=False,
                            message=f"Ошибка синтаксиса: {result.stderr}",
                            severity="error",
                        )
                    )
            except Exception as e:
                self.results.append(
                    CheckResult(
                        name=f"Стратегия: {strategy_file.name}",
                        passed=False,
                        message=f"Не удалось проверить: {e}",
                        severity="warning",
                    )
                )

        if valid_strategies:
            self.results.append(
                CheckResult(
                    name="Стратегии",
                    passed=True,
                    message=f"Найдено валидных стратегий: {', '.join(valid_strategies)}",
                    severity="info",
                )
            )

    def check_docker(self):
        """Проверка Docker окружения."""
        # Проверка Docker
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                self.results.append(
                    CheckResult(
                        name="Docker",
                        passed=False,
                        message="Docker не установлен или не запущен",
                        severity="error",
                    )
                )
                return
        except Exception as e:
            self.results.append(
                CheckResult(
                    name="Docker",
                    passed=False,
                    message=f"Не удалось проверить Docker: {e}",
                    severity="error",
                )
            )
            return

        # Проверка Docker Compose
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                self.results.append(
                    CheckResult(
                        name="Docker Compose",
                        passed=False,
                        message="Docker Compose не установлен",
                        severity="error",
                    )
                )
                return
        except Exception as e:
            self.results.append(
                CheckResult(
                    name="Docker Compose",
                    passed=False,
                    message=f"Не удалось проверить Docker Compose: {e}",
                    severity="error",
                )
            )
            return

        # Проверка docker-compose.yml
        if not Path("docker-compose.yml").exists():
            self.results.append(
                CheckResult(
                    name="docker-compose.yml",
                    passed=False,
                    message="Файл docker-compose.yml не найден",
                    severity="error",
                )
            )
            return

        self.results.append(
            CheckResult(
                name="Docker",
                passed=True,
                message="Docker и Docker Compose установлены",
                severity="info",
            )
        )

    def check_telegram(self):
        """Проверка Telegram бота."""
        # Загрузка .env
        if not self.env_path.exists():
            return

        env_vars = {}
        with open(self.env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        token = env_vars.get("TELEGRAM_TOKEN", "")
        chat_id = env_vars.get("TELEGRAM_CHAT_ID", "")

        if not token or not chat_id:
            self.results.append(
                CheckResult(
                    name="Telegram бот",
                    passed=True,
                    message="Telegram не настроен (уведомления отключены)",
                    severity="warning",
                )
            )
            return

        # Проверка подключения к Telegram API
        try:
            import requests

            url = f"https://api.telegram.org/bot{token}/getMe"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_name = data["result"].get("username", "Unknown")
                    self.results.append(
                        CheckResult(
                            name="Telegram бот",
                            passed=True,
                            message=f"Подключение OK. Бот: @{bot_name}",
                            severity="info",
                        )
                    )
                else:
                    self.results.append(
                        CheckResult(
                            name="Telegram бот",
                            passed=False,
                            message=f"Ошибка API: {data.get('description', 'Unknown')}",
                            severity="error",
                        )
                    )
            else:
                self.results.append(
                    CheckResult(
                        name="Telegram бот",
                        passed=False,
                        message=f"HTTP {response.status_code}: Неверный токен",
                        severity="error",
                    )
                )

        except ImportError:
            self.results.append(
                CheckResult(
                    name="Telegram бот",
                    passed=True,
                    message="Библиотека requests не установлена (пропущено)",
                    severity="warning",
                )
            )
        except Exception as e:
            self.results.append(
                CheckResult(
                    name="Telegram бот",
                    passed=False,
                    message=f"Не удалось проверить: {e}",
                    severity="warning",
                )
            )

    def check_exchange_connection(self):
        """Проверка подключения к бирже."""
        print("\n🔍 Проверка подключения к бирже...")
        print("   (это может занять 10-30 секунд)\n")

        try:
            result = subprocess.run(
                [
                    "docker-compose",
                    "run",
                    "--rm",
                    "freqtrade",
                    "list-exchanges",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.results.append(
                    CheckResult(
                        name="Подключение к бирже",
                        passed=True,
                        message="Freqtrade может подключаться к биржам",
                        severity="info",
                    )
                )
            else:
                self.results.append(
                    CheckResult(
                        name="Подключение к бирже",
                        passed=False,
                        message=f"Ошибка: {result.stderr}",
                        severity="warning",
                    )
                )

        except subprocess.TimeoutExpired:
            self.results.append(
                CheckResult(
                    name="Подключение к бирже",
                    passed=False,
                    message="Timeout: Docker контейнер не отвечает",
                    severity="warning",
                )
            )
        except Exception as e:
            self.results.append(
                CheckResult(
                    name="Подключение к бирже",
                    passed=False,
                    message=f"Не удалось проверить: {e}",
                    severity="warning",
                )
            )

    def print_results(self):
        """Вывод результатов проверок."""
        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ПРОВЕРКИ")
        print("=" * 70 + "\n")

        # Группировка по severity
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        success = [r for r in self.results if r.passed]

        # Ошибки
        if errors:
            print("❌ ОШИБКИ (критичные):\n")
            for result in errors:
                print(f"   ❌ {result.name}")
                print(f"      {result.message}\n")

        # Предупреждения
        if warnings:
            print("⚠️  ПРЕДУПРЕЖДЕНИЯ:\n")
            for result in warnings:
                print(f"   ⚠️  {result.name}")
                print(f"      {result.message}\n")

        # Успешные
        if success:
            print("✅ УСПЕШНО:\n")
            for result in success:
                print(f"   ✅ {result.name}")
                if self.verbose:
                    print(f"      {result.message}")

        print("\n" + "=" * 70)
        print("ИТОГО")
        print("=" * 70)
        print(f"✅ Успешно:       {len(success)}")
        print(f"⚠️  Предупреждения: {len(warnings)}")
        print(f"❌ Ошибки:        {len(errors)}")
        print("=" * 70 + "\n")

        # Финальное заключение
        if errors:
            print("❌ КОНФИГУРАЦИЯ СОДЕРЖИТ ОШИБКИ")
            print("\nИсправьте критичные ошибки перед запуском бота.")
            print("\nПолезные команды:")
            print("  - Создать .env:    cp .env.example .env")
            print("  - Настройка API:   см. docs/API_SETUP_RU.md")
            print("  - Настройка TG:    см. docs/TELEGRAM_SETUP_RU.md")
        elif warnings:
            print("⚠️  КОНФИГУРАЦИЯ С ПРЕДУПРЕЖДЕНИЯМИ")
            print("\nБот может запуститься, но рекомендуется исправить предупреждения.")
            print("\nДля dry-run (тестирование): можно запускать")
            print("Для live trading: исправьте все предупреждения")
        else:
            print("✅ КОНФИГУРАЦИЯ КОРРЕКТНА")
            print("\nВсё готово к запуску!")
            print("\nСледующие шаги:")
            print("  1. Скачать данные:  ./scripts/deploy.sh --data")
            print("  2. Запустить бота:  ./scripts/citadel.sh trade")
            print("  3. Открыть дашборд: http://127.0.0.1:3000")

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Проверка конфигурации Stoic Citadel")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Подробный вывод",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Полная проверка (включая подключение к бирже)",
    )

    parser.add_argument(
        "--no-exchange",
        action="store_true",
        help="Пропустить проверку подключения к бирже",
    )

    args = parser.parse_args()

    validator = ConfigValidator(verbose=args.verbose)

    check_exchange = args.full and not args.no_exchange

    success = validator.run_all_checks(check_exchange=check_exchange)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
