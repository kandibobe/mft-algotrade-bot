#!/usr/bin/env python3
"""
Stoic Citadel Preflight Check Script
Performs comprehensive health check of environment and codebase before running expensive ML tasks.
"""

import sys
import json
import importlib
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ANSI colors for pretty output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BLUE = "\033[94m"
NC = "\033[0m"  # No Color

class PreflightCheck:
    """Main preflight check class."""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
    def check(self, name: str, condition: bool, message: str = "") -> None:
        """Record a check result."""
        self.results.append((name, condition, message))
        if not condition:
            self.errors.append(f"{name}: {message}")
        elif message and "warning" in message.lower():
            self.warnings.append(f"{name}: {message}")
    
    def print_header(self) -> None:
        """Print script header."""
        print(f"""
{CYAN}╔════════════════════════════════════════════════════════════════════╗
║                 STOIC CITADEL PREFLIGHT CHECK                       ║
║           Environment & Codebase Health Verification                ║
╚════════════════════════════════════════════════════════════════════╝{NC}
""")
    
    def print_result(self, name: str, success: bool, message: str = "") -> None:
        """Print a single check result with emoji."""
        emoji = "✅" if success else "❌"
        color = GREEN if success else RED
        if not success and "warning" in message.lower():
            emoji = "⚠️"
            color = YELLOW
        
        print(f"  {emoji} {color}{name}:{NC} {message}")
    
    def print_summary(self) -> None:
        """Print summary of all checks."""
        print(f"\n{BLUE}{'='*60}{NC}")
        print(f"{CYAN}SUMMARY{NC}")
        print(f"{BLUE}{'='*60}{NC}")
        
        total = len(self.results)
        passed = sum(1 for _, success, _ in self.results if success)
        failed = total - passed
        
        print(f"Total checks: {total}")
        print(f"{GREEN}Passed: {passed}{NC}")
        if failed > 0:
            print(f"{RED}Failed: {failed}{NC}")
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings:{NC}")
            for warning in self.warnings:
                print(f"  ⚠️  {warning}")
        
        if self.errors:
            print(f"\n{RED}Errors:{NC}")
            for error in self.errors:
                print(f"  ❌ {error}")
        
        print(f"\n{BLUE}{'='*60}{NC}")
        
        if failed == 0 and not self.errors:
            print(f"{GREEN}✅ READY TO LAUNCH - All preflight checks passed!{NC}")
            return 0
        else:
            print(f"{RED}❌ FIX REQUIRED - Please address the issues above.{NC}")
            if self.errors:
                print(f"\nMissing components:")
                for error in self.errors:
                    print(f"  • {error}")
            return 1
    
    def run_all_checks(self) -> int:
        """Run all preflight checks."""
        self.print_header()
        
        print(f"{CYAN}1. Environment & Dependencies{NC}")
        self.check_python_version()
        self.check_critical_libraries()
        
        print(f"\n{CYAN}2. Configuration Integrity{NC}")
        self.check_config_json()
        self.check_exchange_name()
        self.check_strategy_v4()
        
        print(f"\n{CYAN}3. Data Availability{NC}")
        self.check_data_directory()
        self.check_pair_files()
        self.check_models_directory()
        
        print(f"\n{CYAN}4. Project Structure{NC}")
        self.check_critical_source_files()
        
        print(f"\n{CYAN}5. Docker Readiness{NC}")
        self.check_docker_compose()
        
        # Print all results
        print(f"\n{CYAN}CHECK RESULTS:{NC}")
        for name, success, message in self.results:
            self.print_result(name, success, message)
        
        # Return summary
        return self.print_summary()
    
    def check_python_version(self) -> None:
        """Check Python version >= 3.10."""
        version = platform.python_version()
        major, minor, _ = map(int, version.split('.'))
        success = major == 3 and minor >= 10
        self.check(
            "Python Version",
            success,
            f"{version} {'>= 3.10' if success else '< 3.10'}"
        )
    
    def check_critical_libraries(self) -> None:
        """Check critical libraries can be imported."""
        libraries = [
            "freqtrade",
            "xgboost", 
            "ccxt",
            "pandas",
            "tenacity"
        ]
        
        for lib in libraries:
            try:
                importlib.import_module(lib)
                self.check(f"Library: {lib}", True, "Available")
            except ImportError as e:
                self.check(f"Library: {lib}", False, f"Missing: {e}")
    
    def check_config_json(self) -> None:
        """Check if user_data/config.json exists and is valid JSON."""
        config_paths = [
            Path("user_data/config.json"),
            Path("user_data/config/config.json"),
            Path("user_data/config/config_production.json"),
            Path("user_data/config/config_dryrun.json"),
        ]
        
        found = False
        valid_json = False
        config_file = ""
        
        for config_path in config_paths:
            if config_path.exists():
                found = True
                config_file = str(config_path)
                try:
                    with open(config_path, 'r') as f:
                        json.load(f)
                    valid_json = True
                    break
                except json.JSONDecodeError:
                    valid_json = False
                    break
        
        if found:
            self.check(
                "Config JSON",
                valid_json,
                f"Found at {config_file}" + ("" if valid_json else " (invalid JSON)")
            )
        else:
            self.check("Config JSON", False, "Not found in user_data/")
    
    def check_exchange_name(self) -> None:
        """Check if exchange.name is set in config."""
        config_paths = [
            Path("user_data/config.json"),
            Path("user_data/config/config.json"),
            Path("user_data/config/config_production.json"),
            Path("user_data/config/config_dryrun.json"),
        ]
        
        exchange_set = False
        config_file = ""
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    if config.get("exchange", {}).get("name"):
                        exchange_set = True
                        config_file = str(config_path)
                        break
                except (json.JSONDecodeError, KeyError):
                    continue
        
        self.check(
            "Exchange Name",
            exchange_set,
            f"{'Set in config' if exchange_set else 'Not set or config not found'}"
        )
    
    def check_strategy_v4(self) -> None:
        """Check if user_data/strategies/StrategyV4.py exists."""
        strategy_paths = [
            Path("user_data/strategies/StrategyV4.py"),
            Path("user_data/strategies/StoicEnsembleStrategyV4.py"),
        ]
        
        found = False
        strategy_file = ""
        
        for strategy_path in strategy_paths:
            if strategy_path.exists():
                found = True
                strategy_file = strategy_path.name
                break
        
        self.check(
            "Strategy V4",
            found,
            f"{'Found: ' + strategy_file if found else 'Not found (check StrategyV4.py or StoicEnsembleStrategyV4.py)'}"
        )
    
    def check_data_directory(self) -> None:
        """Check directory user_data/data/binance."""
        data_dir = Path("user_data/data/binance")
        exists = data_dir.exists() and data_dir.is_dir()
        self.check(
            "Data Directory",
            exists,
            f"{'Exists' if exists else 'Missing: user_data/data/binance/'}"
        )
    
    def check_pair_files(self) -> None:
        """Count how many pair files exist in user_data/data/binance."""
        data_dir = Path("user_data/data/binance")
        if data_dir.exists():
            # Look for common data file extensions
            extensions = ['.feather', '.parquet', '.json', '.csv', '.h5']
            pair_files = []
            for ext in extensions:
                pair_files.extend(list(data_dir.glob(f"*{ext}")))
            
            count = len(pair_files)
            success = count > 0
            self.check(
                "Pair Files",
                success,
                f"{count} files found" + ("" if success else " (WARNING: No data files)")
            )
            if count == 0:
                self.warnings.append("No pair files found in user_data/data/binance/")
        else:
            self.check("Pair Files", False, "Data directory not found")
    
    def check_models_directory(self) -> None:
        """Check if user_data/models directory exists (create if not)."""
        models_dir = Path("user_data/models")
        exists = models_dir.exists()
        
        if not exists:
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
                self.check(
                    "Models Directory",
                    True,
                    "Created (was missing)"
                )
            except Exception as e:
                self.check(
                    "Models Directory",
                    False,
                    f"Failed to create: {e}"
                )
        else:
            self.check(
                "Models Directory",
                True,
                "Exists"
            )
    
    def check_critical_source_files(self) -> None:
        """Verify critical source files exist."""
        critical_files = [
            ("smart_limit_executor.py", Path("src/order_manager/smart_limit_executor.py"), True),
            ("trainer.py", Path("src/ml/training/trainer.py"), False),  # Not critical if alternative exists
            ("model_trainer.py", Path("src/ml/training/model_trainer.py"), True),  # Alternative
        ]
        
        trainer_found = False
        model_trainer_found = False
        
        for file_name, file_path, is_critical in critical_files:
            exists = file_path.exists()
            
            if file_name == "trainer.py":
                if exists:
                    trainer_found = True
                continue  # We'll handle this after checking model_trainer.py
            elif file_name == "model_trainer.py":
                model_trainer_found = exists
            
            self.check(
                f"Source: {file_name}",
                exists if is_critical else True,  # Mark as passed if not critical
                f"{'Found' if exists else 'Missing'}"
            )
        
        # Handle trainer.py check
        if trainer_found:
            self.check("Source: trainer.py", True, "Found")
        elif model_trainer_found:
            self.check("Source: trainer.py", True, "Not found, but model_trainer.py exists (OK)")
        else:
            self.check("Source: trainer.py", False, "Missing (no trainer.py or model_trainer.py)")
    
    def check_docker_compose(self) -> None:
        """Check if docker-compose.yml exists."""
        docker_paths = [
            Path("docker-compose.yml"),
            Path("docker-compose.backtest.yml"),
            Path("docker-compose.monitoring.yml"),
        ]
        
        found = False
        docker_file = ""
        
        for docker_path in docker_paths:
            if docker_path.exists():
                found = True
                docker_file = docker_path.name
                break
        
        self.check(
            "Docker Compose",
            found,
            f"{'Found: ' + docker_file if found else 'Not found'}"
        )

def main() -> int:
    """Main entry point."""
    checker = PreflightCheck()
    return checker.run_all_checks()

if __name__ == "__main__":
    sys.exit(main())
