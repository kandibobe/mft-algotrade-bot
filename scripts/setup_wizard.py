#!/usr/bin/env python3
"""
Stoic Citadel - Interactive Setup Wizard
=========================================

Guides users through first-time setup and validates the environment.

Author: Stoic Citadel Team
License: MIT
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from typing import Optional, Tuple

# Color codes for terminal output
class Colors:
    CYAN = '\033[0;36m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


def print_banner():
    """Print the Stoic Citadel banner."""
    print()
    print(f"{Colors.CYAN}╔════════════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.CYAN}║                                                                    ║{Colors.NC}")
    print(f"{Colors.CYAN}║                    {Colors.MAGENTA}STOIC CITADEL{Colors.CYAN}                                 ║{Colors.NC}")
    print(f"{Colors.CYAN}║                                                                    ║{Colors.NC}")
    print(f"{Colors.CYAN}║         {Colors.YELLOW}Professional MFT Trading Infrastructure{Colors.CYAN}               ║{Colors.NC}")
    print(f"{Colors.CYAN}║                                                                    ║{Colors.NC}")
    print(f"{Colors.CYAN}║                    {Colors.GREEN}Setup Wizard v1.0{Colors.CYAN}                              ║{Colors.NC}")
    print(f"{Colors.CYAN}║                                                                    ║{Colors.NC}")
    print(f"{Colors.CYAN}╚════════════════════════════════════════════════════════════════════╝{Colors.NC}")
    print()


def print_step(step: int, total: int, message: str):
    """Print a step indicator."""
    print(f"\n{Colors.BOLD}[{step}/{total}] {message}{Colors.NC}")
    print("─" * 70)


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {message}{Colors.NC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.CYAN}ℹ️  {message}{Colors.NC}")


def check_docker() -> bool:
    """Check if Docker is installed and running."""
    try:
        result = subprocess.run(
            ['docker', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Docker found: {version}")

            # Check if Docker daemon is running
            result = subprocess.run(
                ['docker', 'ps'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print_success("Docker daemon is running")
                return True
            else:
                print_error("Docker daemon is not running")
                print_info("Please start Docker and try again")
                return False
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_error("Docker not found")
        print_info("Please install Docker: https://docs.docker.com/get-docker/")
        return False


def check_docker_compose() -> bool:
    """Check if Docker Compose is installed."""
    try:
        result = subprocess.run(
            ['docker-compose', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print_success(f"Docker Compose found: {version}")
            return True
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Try docker compose (v2 syntax)
        try:
            result = subprocess.run(
                ['docker', 'compose', 'version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print_success(f"Docker Compose v2 found: {version}")
                return True
        except:
            pass

        print_error("Docker Compose not found")
        print_info("Please install Docker Compose: https://docs.docker.com/compose/install/")
        return False


def check_disk_space() -> bool:
    """Check available disk space."""
    import shutil

    stat = shutil.disk_usage('.')
    free_gb = stat.free / (1024 ** 3)

    if free_gb >= 20:
        print_success(f"Available disk space: {free_gb:.1f} GB")
        return True
    elif free_gb >= 10:
        print_warning(f"Low disk space: {free_gb:.1f} GB (20 GB recommended)")
        return True
    else:
        print_error(f"Insufficient disk space: {free_gb:.1f} GB (minimum 10 GB required)")
        return False


def create_env_file() -> bool:
    """Create .env file from template."""
    env_example = Path('.env.example')
    env_file = Path('.env')

    if env_file.exists():
        print_warning(".env file already exists")
        response = input("Overwrite? (yes/no): ").lower().strip()
        if response != 'yes':
            print_info("Keeping existing .env file")
            return True

    if not env_example.exists():
        print_error(".env.example not found")
        return False

    # Copy template
    with open(env_example, 'r') as f:
        content = f.read()

    # Interactive configuration
    print_info("Configuring environment variables...")

    # Telegram configuration
    print("\n" + Colors.BOLD + "Telegram Configuration (optional)" + Colors.NC)
    print("To receive trading alerts via Telegram:")
    print("1. Create a bot via @BotFather on Telegram")
    print("2. Get your chat ID via @userinfobot")

    telegram_token = input("\nTelegram Bot Token (or press Enter to skip): ").strip()
    if telegram_token:
        if validate_telegram_token(telegram_token):
            content = content.replace('YOUR_TELEGRAM_BOT_TOKEN', telegram_token)
            telegram_chat_id = input("Telegram Chat ID: ").strip()
            content = content.replace('YOUR_TELEGRAM_CHAT_ID', telegram_chat_id)
            print_success("Telegram configured")
        else:
            print_warning("Invalid Telegram token format, skipping")

    # Write .env file
    with open(env_file, 'w') as f:
        f.write(content)

    print_success(".env file created")
    return True


def validate_telegram_token(token: str) -> bool:
    """Validate Telegram bot token format."""
    pattern = r'^\d+:[A-Za-z0-9_-]{35}$'
    return bool(re.match(pattern, token))


def create_directories():
    """Create necessary directories."""
    directories = [
        'user_data/config',
        'user_data/strategies',
        'user_data/data',
        'user_data/notebooks',
        'user_data/logs',
        'research',
        'backups',
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print_success("Directories created")


def build_containers() -> bool:
    """Build Docker containers."""
    print_info("Building Docker containers (this may take a few minutes)...")

    try:
        result = subprocess.run(
            ['docker-compose', 'build'],
            timeout=600  # 10 minutes max
        )
        if result.returncode == 0:
            print_success("Containers built successfully")
            return True
        else:
            print_error("Failed to build containers")
            return False
    except subprocess.TimeoutExpired:
        print_error("Build timeout (took more than 10 minutes)")
        return False
    except Exception as e:
        print_error(f"Build error: {e}")
        return False


def run_healthcheck() -> bool:
    """Run health checks on the configuration."""
    print_info("Running configuration health checks...")

    # Check config files
    config_files = [
        'user_data/config/config_dryrun.json',
        'user_data/config/config_production.json'
    ]

    all_valid = True
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
                print_success(f"{config_file} is valid")
            except json.JSONDecodeError as e:
                print_error(f"{config_file} has JSON errors: {e}")
                all_valid = False
        else:
            print_warning(f"{config_file} not found")

    return all_valid


def show_next_steps():
    """Show next steps to the user."""
    print()
    print(f"{Colors.BOLD}{'═' * 70}{Colors.NC}")
    print(f"{Colors.GREEN}{Colors.BOLD}✅ Setup Complete!{Colors.NC}")
    print(f"{Colors.BOLD}{'═' * 70}{Colors.NC}")
    print()
    print(f"{Colors.BOLD}Next Steps:{Colors.NC}")
    print()
    print("1. Download historical data:")
    print(f"   {Colors.CYAN}make download{Colors.NC}")
    print()
    print("2. Start research environment:")
    print(f"   {Colors.CYAN}make research{Colors.NC}")
    print(f"   Then open: http://localhost:8888 (token: stoic2024)")
    print()
    print("3. Run tests:")
    print(f"   {Colors.CYAN}make test{Colors.NC}")
    print()
    print("4. Start dry-run trading:")
    print(f"   {Colors.CYAN}make trade-dry{Colors.NC}")
    print(f"   Dashboard: http://localhost:3000")
    print()
    print(f"{Colors.BOLD}Documentation:{Colors.NC}")
    print(f"   {Colors.CYAN}cat README.md{Colors.NC}")
    print()
    print(f"{Colors.BOLD}Available commands:{Colors.NC}")
    print(f"   {Colors.CYAN}make help{Colors.NC}")
    print()


def main():
    """Main setup wizard."""
    print_banner()

    print(f"{Colors.BOLD}Welcome to Stoic Citadel Setup Wizard!{Colors.NC}")
    print()
    print("This wizard will guide you through the initial setup process.")
    print()
    input("Press Enter to continue...")

    # Step 1: Check Docker
    print_step(1, 6, "Checking Docker Installation")
    if not check_docker():
        print_error("Setup failed: Docker not available")
        sys.exit(1)

    if not check_docker_compose():
        print_error("Setup failed: Docker Compose not available")
        sys.exit(1)

    # Step 2: Check disk space
    print_step(2, 6, "Checking System Resources")
    if not check_disk_space():
        print_error("Setup failed: Insufficient disk space")
        sys.exit(1)

    # Step 3: Create .env file
    print_step(3, 6, "Creating Environment Configuration")
    if not create_env_file():
        print_error("Setup failed: Could not create .env file")
        sys.exit(1)

    # Step 4: Create directories
    print_step(4, 6, "Creating Project Directories")
    create_directories()

    # Step 5: Build containers
    print_step(5, 6, "Building Docker Containers")
    if not build_containers():
        print_warning("Container build failed, but you can try again later with: make build")

    # Step 6: Health check
    print_step(6, 6, "Running Health Checks")
    run_healthcheck()

    # Show next steps
    show_next_steps()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warning("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print()
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
