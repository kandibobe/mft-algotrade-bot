import logging
import os
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SetupDevEnv")


def run_command(command, description):
    logger.info(f"🚀 Starting: {description}...")
    try:
        if sys.platform == "win32":
            subprocess.check_call(command, shell=True)
        else:
            subprocess.check_call(command, shell=True)
        logger.info(f"✅ Completed: {description}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed: {description}")
        logger.error(f"Error details: {e}")
        sys.exit(1)


def main():
    logger.info("🔧 Stoic Citadel - Development Environment Setup 🔧")
    logger.info("==================================================")

    # 1. Check for .env
    if not os.path.exists(".env"):
        logger.info("⚠️  .env not found.")
        if os.path.exists(".env.example"):
            run_command(
                "cp .env.example .env" if sys.platform != "win32" else "copy .env.example .env",
                "Creating .env from template",
            )
        else:
            logger.warning("❌ .env.example not found! Please create .env manually.")

    # 2. Run Makefile Setup (Virtualenv, Dependencies, ML Init)
    # Check if 'make' is available
    try:
        subprocess.check_call(
            ["make", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        has_make = True
    except (OSError, subprocess.CalledProcessError):
        has_make = False
        logger.warning("⚠️  'make' not found. Falling back to direct commands.")

    if has_make:
        run_command("make setup", "Full Environment Setup (via Makefile)")
    else:
        # Fallback for Windows without Make
        logger.info("running fallback setup for Windows/No-Make environment...")
        # Create venv if not exists
        if not os.path.exists(".venv"):
            run_command(f"{sys.executable} -m venv .venv", "Creating Virtual Environment")

        # Determine pip path
        if sys.platform == "win32":
            pip_cmd = r".venv\Scripts\pip"
            python_cmd = r".venv\Scripts\python"
        else:
            pip_cmd = ".venv/bin/pip"
            python_cmd = ".venv/bin/python"

        run_command(f"{pip_cmd} install --upgrade pip", "Upgrading Pip")
        run_command(f'{pip_cmd} install -e ".[dev]"', "Installing Development Dependencies")
        run_command(f"{python_cmd} scripts/setup/init_ml_system.py", "Initializing ML System")

    # 3. Verify Deployment (Config generation & Strategy check)
    run_command(f"{sys.executable} scripts/verify_deployment.py", "Final System Verification")

    logger.info("==================================================")
    logger.info("🎉 SUCCESS! Development environment is ready.")
    logger.info("   To start trading (dry-run):")
    logger.info("   source .venv/bin/activate  # or .venv\\Scripts\\activate")
    logger.info(
        "   freqtrade trade --config user_data/config/config_production.json --strategy StoicEnsembleStrategyV6"
    )
    logger.info("==================================================")


if __name__ == "__main__":
    main()
