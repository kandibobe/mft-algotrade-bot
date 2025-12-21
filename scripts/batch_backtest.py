#!/usr/bin/env python3
"""
Batch backtesting - тест нескольких стратегий параллельно.

Эффект: Тестирование нескольких стратегий параллельно
Проблема: Тестирование 5 стратегий = 5 часов
Решение: Параллельный запуск с использованием ProcessPoolExecutor

Usage:
    # Запустить тестирование всех стратегий по умолчанию
    python scripts/batch_backtest.py

    # Тестировать только определенные стратегии
    python scripts/batch_backtest.py --strategies StoicStrategyV1 StoicEnsembleStrategyV2

    # Использовать кастомный timerange
    python scripts/batch_backtest.py --timerange 20230101-20241231

    # Запустить без Docker (локальный freqtrade)
    python scripts/batch_backtest.py --no-docker

    # Использовать определенные пары
    python scripts/batch_backtest.py --pairs BTC/USDT ETH/USDT

    # Показать справку
    python scripts/batch_backtest.py --help
"""

import argparse
import concurrent.futures
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import signal
import threading
import os

# Добавляем путь к проекту для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.logger import setup_logger
    logger = setup_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Исключение для таймаута выполнения."""
    pass


def timeout_handler(signum=None, frame=None):
    """Обработчик сигнала таймаута."""
    raise TimeoutException("Execution timed out")


class BatchBacktester:
    """Класс для параллельного запуска бэктестов."""

    # Стратегии по умолчанию
    DEFAULT_STRATEGIES = [
        "StoicStrategyV1",
        "StoicEnsembleStrategyV2",
        "StoicEnsembleStrategyV3",
    ]

    # Пары по умолчанию
    DEFAULT_PAIRS = ["BTC/USDT"]

    # Таймфрейм по умолчанию
    DEFAULT_TIMEFRAME = "5m"

    # Таймаут выполнения одной стратегии (в секундах)
    DEFAULT_TIMEOUT = 7200  # 2 часа

    def __init__(
        self,
        strategies: List[str] = None,
        timerange: str = "20230101-20241231",
        pairs: List[str] = None,
        timeframe: str = None,
        use_docker: bool = True,
        max_workers: int = None,
        timeout: int = None,
        export_trades: bool = True,
        config_path: str = None
    ):
        """
        Инициализация батч-бэктестера.

        Args:
            strategies: Список стратегий для тестирования
            timerange: Диапазон времени для тестирования (формат: YYYYMMDD-YYYYMMDD)
            pairs: Список торговых пар
            timeframe: Таймфрейм
            use_docker: Использовать Docker для запуска freqtrade
            max_workers: Максимальное количество параллельных процессов
            timeout: Таймаут выполнения одной стратегии (секунды)
            export_trades: Экспортировать результаты торгов
            config_path: Путь к конфигурационному файлу freqtrade
        """
        self.strategies = strategies or self.DEFAULT_STRATEGIES
        self.timerange = timerange
        self.pairs = pairs or self.DEFAULT_PAIRS
        self.timeframe = timeframe or self.DEFAULT_TIMEFRAME
        self.use_docker = use_docker
        self.max_workers = max_workers or min(len(self.strategies), 3)  # Макс 3 процесса
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.export_trades = export_trades
        self.config_path = config_path

        # Имя Docker-контейнера
        self.container_name = "stoic_freqtrade"

        # Результаты выполнения
        self.results: List[Dict[str, Any]] = []

        # Флаг прерывания
        self._interrupted = False

    def check_docker_available(self) -> bool:
        """Проверить доступность Docker."""
        if not self.use_docker:
            return True

        try:
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("Docker не найден. Будет использован локальный freqtrade.")
            return False

    def check_freqtrade_available(self) -> bool:
        """Проверить доступность freqtrade."""
        try:
            if self.use_docker:
                cmd = ["docker", "exec", self.container_name, "freqtrade", "--version"]
            else:
                cmd = ["freqtrade", "--version"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            logger.error("freqtrade не найден. Убедитесь, что он установлен.")
            return False

    def build_freqtrade_command(self, strategy: str) -> List[str]:
        """Построить команду для запуска freqtrade."""
        if self.use_docker:
            base_cmd = ["docker", "exec", self.container_name, "freqtrade", "backtesting"]
        else:
            base_cmd = ["freqtrade", "backtesting"]

        # Базовые параметры
        cmd = base_cmd + [
            "--strategy", strategy,
            "--timerange", self.timerange,
            "--timeframe", self.timeframe,
        ]

        # Добавить пары
        for pair in self.pairs:
            cmd.extend(["--pairs", pair])

        # Экспорт результатов
        if self.export_trades:
            cmd.extend(["--export", "trades"])

        # Конфигурационный файл
        config_to_use = self.config_path
        if not config_to_use:
            # Попробовать найти конфиг по умолчанию
            default_configs = [
                "user_data/config/config_production.json",
                "user_data/config/config_backtest.json",
                "user_data/config/config.json"
            ]
            for config in default_configs:
                if Path(config).exists():
                    config_to_use = config
                    break
        
        if config_to_use:
            cmd.extend(["--config", config_to_use])

        # Дополнительные параметры для ускорения
        cmd.extend(["--cache", "none"])  # Отключить кэш для параллельного выполнения

        return cmd

    def run_single_backtest(self, strategy: str) -> Dict[str, Any]:
        """
        Запустить backtest для одной стратегии.

        Args:
            strategy: Название стратегии

        Returns:
            Словарь с результатами выполнения
        """
        if self._interrupted:
            return {
                "strategy": strategy,
                "success": False,
                "error": "Прервано пользователем",
                "duration": 0,
                "output": ""
            }

        logger.info(f"🚀 Запуск стратегии: {strategy}")
        start_time = datetime.now()

        # Построить команду
        cmd = self.build_freqtrade_command(strategy)
        logger.debug(f"Команда: {' '.join(cmd)}")

        try:
            # Запустить процесс
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Таймер для таймаута (кросс-платформенный)
            timeout_occurred = False
            timer = None
            
            if os.name == 'posix':  # Unix/Linux/Mac
                # Используем signal.alarm
                import signal
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
            else:  # Windows
                # Используем threading.Timer
                timer = threading.Timer(self.timeout, lambda: process.kill())
                timer.start()

            # Собираем вывод
            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line.strip())
                    # Логируем прогресс (каждые 10 строк)
                    if len(output_lines) % 10 == 0:
                        logger.debug(f"{strategy}: обработано {len(output_lines)} строк вывода")

            # Дождаться завершения
            return_code = process.wait()

            # Отключить таймаут
            if os.name == 'posix':
                signal.alarm(0)
            elif timer:
                timer.cancel()

            duration = (datetime.now() - start_time).total_seconds()

            if return_code == 0:
                logger.info(f"✅ Завершено: {strategy} за {duration:.1f}с")
                success = True
            else:
                # Проверить, не был ли процесс убит по таймауту (Windows)
                if os.name != 'posix' and return_code == -9:
                    raise TimeoutException(f"Таймаут ({self.timeout}с)")
                logger.error(f"❌ Ошибка в стратегии {strategy} (код: {return_code})")
                success = False

            # Получить последние 500 символов вывода
            output = "\n".join(output_lines)
            short_output = output[-500:] if len(output) > 500 else output

            return {
                "strategy": strategy,
                "success": success,
                "return_code": return_code,
                "duration": duration,
                "output": short_output,
                "full_output": output if not success else "",  # Сохраняем полный вывод только при ошибках
                "command": " ".join(cmd)
            }

        except TimeoutException as e:
            logger.error(f"⏰ Таймаут выполнения стратегии {strategy} ({self.timeout}с)")
            duration = (datetime.now() - start_time).total_seconds()

            # Попытаться убить процесс
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass

            return {
                "strategy": strategy,
                "success": False,
                "error": f"Таймаут ({self.timeout}с)",
                "duration": duration,
                "output": f"Превышено время выполнения: {self.timeout} секунд"
            }

        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка в стратегии {strategy}: {e}")
            duration = (datetime.now() - start_time).total_seconds()

            return {
                "strategy": strategy,
                "success": False,
                "error": str(e),
                "duration": duration,
                "output": f"Исключение: {e}"
            }

    def run_parallel(self) -> List[Dict[str, Any]]:
        """
        Запустить параллельное тестирование стратегий.

        Returns:
            Список результатов для каждой стратегии
        """
        logger.info(f"📊 Batch Backtest: {len(self.strategies)} стратегий")
        logger.info(f"⏱️  Timerange: {self.timerange}")
        logger.info(f"🔄 Параллельных процессов: {self.max_workers}")
        logger.info(f"⏰ Таймаут на стратегию: {self.timeout}с")

        # Проверить доступность Docker/freqtrade
        if not self.check_freqtrade_available():
            logger.error("❌ freqtrade недоступен. Проверьте установку.")
            return []

        # Запустить параллельное выполнение
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Запустить все задачи
            future_to_strategy = {
                executor.submit(self.run_single_backtest, strategy): strategy
                for strategy in self.strategies
            }

            # Обработать результаты по мере завершения
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result(timeout=self.timeout + 10)  # Небольшой запас
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Ошибка при выполнении стратегии {strategy}: {e}")
                    results.append({
                        "strategy": strategy,
                        "success": False,
                        "error": f"Ошибка future: {e}",
                        "duration": 0,
                        "output": ""
                    })

                # Проверить флаг прерывания
                if self._interrupted:
                    logger.warning("⚠️  Прерывание выполнения...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        # Сохранить результаты
        self.results = results
        return results

    def interrupt(self):
        """Прервать выполнение."""
        self._interrupted = True
        logger.warning("Получен сигнал прерывания")

    def generate_report(self, output_dir: str = "user_data/backtest_results/batch"):
        """Сгенерировать отчет о результатах."""
        if not self.results:
            logger.warning("Нет результатов для отчета")
            return

        # Создать директорию
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Сохранить результаты в JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"batch_results_{timestamp}.json"

        # Подготовить данные для JSON
        report_data = {
            "timestamp": timestamp,
            "config": {
                "strategies": self.strategies,
                "timerange": self.timerange,
                "pairs": self.pairs,
                "timeframe": self.timeframe,
                "use_docker": self.use_docker,
                "max_workers": self.max_workers,
                "timeout": self.timeout
            },
            "results": self.results,
            "summary": {
                "total": len(self.results),
                "successful": sum(1 for r in self.results if r.get("success", False)),
                "failed": sum(1 for r in self.results if not r.get("success", True)),
                "total_duration": sum(r.get("duration", 0) for r in self.results),
                "avg_duration": sum(r.get("duration", 0) for r in self.results) / len(self.results) if self.results else 0
            }
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"📄 Отчет сохранен: {json_file}")

        # Вывести сводку в консоль
        self.print_summary()

        return str(json_file)

    def print_summary(self):
        """Вывести сводку результатов."""
        if not self.results:
            print("Нет результатов для отображения")
            return

        print("\n" + "="*70)
        print("📈 СВОДКА РЕЗУЛЬТАТОВ BATCH BACKTEST")
        print("="*70)

        successful = 0
        total_duration = 0

        for result in self.results:
            strategy = result["strategy"]
            success = result.get("success", False)
            duration = result.get("duration", 0)
            error = result.get("error", "")

            status = "✅" if success else "❌"
            print(f"{status} {strategy}: {duration:.1f}с", end="")

            if error:
                print(f" - {error}")
            else:
                print()

            if success:
                successful += 1
                total_duration += duration

        print("="*70)
        print(f"Всего стратегий: {len(self.results)}")
        print(f"Успешно: {successful}")
        print(f"Неудачно: {len(self.results) - successful}")
        print(f"Общее время: {total_duration:.1f}с")
        print(f"Эффективность параллелизации: {total_duration / max(total_duration, 1) * self.max_workers:.1f}x")
        print("="*70)


def main():
    """Основная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Параллельный запуск бэктестов для нескольких стратегий"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        help=f"Стратегии для тестирования (по умолчанию: {', '.join(BatchBacktester.DEFAULT_STRATEGIES)})"
    )
    parser.add_argument(
        "--timerange",
        type=str,
        default="20230101-20241231",
        help="Диапазон времени (формат: YYYYMMDD-YYYYMMDD, по умолчанию: 20230101-20241231)"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help=f"Торговые пары (по умолчанию: {', '.join(BatchBacktester.DEFAULT_PAIRS)})"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5m",
        help="Таймфрейм (по умолчанию: 5m)"
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Использовать локальный freqtrade вместо Docker"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help=f"Максимальное количество параллельных процессов (по умолчанию: min(количество стратегий, 3))"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Таймаут выполнения одной стратегии в секундах (по умолчанию: 7200 = 2 часа)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Путь к конфигурационному файлу freqtrade"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="user_data/backtest_results/batch",
        help="Директория для сохранения результатов (по умолчанию: user_data/backtest_results/batch)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Проверить конфигурацию без запуска тестов"
    )

    args = parser.parse_args()

    # Создать экземпляр бэктестера
    backtester = BatchBacktester(
        strategies=args.strategies,
        timerange=args.timerange,
        pairs=args.pairs,
        timeframe=args.timeframe,
        use_docker=not args.no_docker,
        max_workers=args.max_workers,
        timeout=args.timeout,
        config_path=args.config
    )

    # Проверить конфигурацию
    if args.dry_run:
        print("\n" + "="*70)
        print("🔧 ПРОВЕРКА КОНФИГУРАЦИИ (DRY RUN)")
        print("="*70)
        print(f"Стратегии: {backtester.strategies}")
        print(f"Timerange: {backtester.timerange}")
        print(f"Пары: {backtester.pairs}")
        print(f"Таймфрейм: {backtester.timeframe}")
        print(f"Использовать Docker: {backtester.use_docker}")
        print(f"Макс. процессов: {backtester.max_workers}")
        print(f"Таймаут: {backtester.timeout}с")
        print(f"Конфиг: {backtester.config_path or 'по умолчанию'}")
        print("="*70)
        print("✅ Конфигурация корректна. Для запуска уберите --dry-run")
        return

    # Зарегистрировать обработчик прерывания
    import signal
    def signal_handler(sig, frame):
        print("\n⚠️  Получен сигнал прерывания (Ctrl+C)")
        backtester.interrupt()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    # Запустить параллельное тестирование
    try:
        results = backtester.run_parallel()

        if results:
            # Сгенерировать отчет
            report_file = backtester.generate_report(args.output_dir)
            print(f"\n📊 Результаты сохранены в: {report_file}")
        else:
            print("\n❌ Не удалось запустить тестирование")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Выполнение прервано пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
