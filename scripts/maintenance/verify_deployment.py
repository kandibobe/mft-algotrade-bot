import logging
import os
import sys
import asyncio
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeploymentVerifier")


def check_dependencies():
    logger.info("Checking dependencies...")
    try:
        import scipy
        logger.info(f"✅ Scipy found: {scipy.__version__}")
    except ImportError:
        logger.error("❌ Scipy NOT found! (Critical for HRP)")
        return False

    try:
        import feast
        logger.info(f"✅ Feast found: {feast.__version__}")
    except ImportError:
        logger.warning("⚠️ Feast not found. Feature Store will run in Mock mode.")

    try:
        import redis
        logger.info(f"✅ Redis found: {redis.__version__}")
    except ImportError:
        logger.warning("⚠️ Redis driver not found.")

    return True


def check_config():
    logger.info("Checking configuration...")
    config_path = Path("user_data/config/config_production.json")

    if not config_path.exists():
        logger.warning(f"⚠️ {config_path} does not exist.")
        # Try to generate it from Unified Config
        try:
            from src.config.unified_config import TradingConfig

            logger.info("Generating config from Unified Config defaults...")
            config = TradingConfig()  # Load from .env or defaults

            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to JSON
            config.to_json(str(config_path))
            logger.info(f"✅ Generated {config_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to generate config: {e}")
            return False
    else:
        logger.info(f"✅ Config found at {config_path}")
        return True


def check_strategy():
    logger.info("Checking Strategy V6...")
    try:
        # Import Strategy V6
        # Note: In a real run, Freqtrade imports this dynamically.
        # Here we just check syntax/imports.
        from user_data.strategies.StoicEnsembleStrategyV6 import StoicEnsembleStrategyV6
        logger.info("✅ StoicEnsembleStrategyV6 importable.")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import Strategy V6: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_alternative_data():
    logger.info("Checking Alternative Data...")
    try:
        from src.data.alternative_data_fetcher import AlternativeDataFetcher
        logger.info("✅ AlternativeDataFetcher importable.")
        return True
    except ImportError:
        logger.error("❌ Failed to import AlternativeDataFetcher.")
        return False

def check_notifications():
    logger.info("Checking Notification Config...")
    try:
        from src.utils.notifications import Notifier
        n = Notifier()
        
        if n.telegram_token and n.telegram_chat_id:
            logger.info("✅ Telegram Configured")
        else:
            logger.warning("⚠️ Telegram NOT configured")
        
        if n.slack_webhook:
            logger.info("✅ Slack Configured")
        
        if n.email_host and n.email_recipient:
            logger.info("✅ Email Configured")
        else:
            logger.warning("⚠️ Email NOT configured (Optional)")
            
        return True
    except Exception as e:
        logger.error(f"❌ Notification check failed: {e}")
        return False

def check_risk_system():
    logger.info("Checking Risk System...")
    try:
        from src.risk.circuit_breaker import CircuitBreaker
        # Basic instantiation test
        cb = CircuitBreaker()
        logger.info("✅ Circuit Breaker initialized.")
        
        if Path("scripts/risk/chaos_test.py").exists():
             logger.info("✅ Chaos Test script found.")
        else:
             logger.warning("⚠️ Chaos Test script missing.")
             
        return True
    except Exception as e:
        logger.error(f"❌ Risk System check failed: {e}")
        return False

def main():
    logger.info("🚀 Starting Deployment Verification...")
    
    checks = [
        check_dependencies,
        check_config,
        check_strategy,
        check_alternative_data,
        check_notifications,
        check_risk_system
    ]
    
    failed = False
    for check in checks:
        if not check():
            failed = True
            
    if not failed:
        logger.info("\n🎉 All checks passed! System is ready for deployment.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some checks failed. Please review logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
