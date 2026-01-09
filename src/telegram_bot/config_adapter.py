# src/telegram_bot/config_adapter.py
"""
Adapter to bridge Stoic Citadel's unified config with the Telegram Bot's config structure.
"""
import os
from src.config.unified_config import load_config

# Load main application config
_main_config = load_config()

# --- Telegram Bot ---
TELEGRAM_BOT_TOKEN = _main_config.telegram.token or os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = _main_config.telegram.chat_id or os.getenv("ADMIN_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- API Keys (kept from env for now as they are not in UnifiedConfig yet) ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
ALPHA_VANTage_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
CRYPTO_PANIC_API_KEY = os.getenv("CRYPTO_PANIC_API_KEY")
NEWS_API_ORG_KEY = os.getenv("NEWS_API_ORG_KEY")
GLASSNODE_API_KEY = os.getenv("GLASSNODE_API_KEY")

# --- Bot Behavior & Timings ---
PRICE_ALERT_CHECK_INTERVAL = 60
VOLATILITY_FETCH_INTERVAL = 15 * 60
SHARED_DATA_FETCH_INTERVAL = 10 * 60
INDEX_FETCH_INTERVAL = 30 * 60
ONCHAIN_FETCH_INTERVAL = 4 * 60 * 60
DIGEST_SEND_HOUR_UTC = 7
API_COOLDOWN = 1.5
DEFAULT_REQUEST_TIMEOUT = 15
ALERT_COOLDOWN_SECONDS = 60 * 60 * 3
SIGNAL_DATA_CACHE_AGE = 60 * 60 * 2

# --- User Limits ---
WATCHLIST_LIMIT = 5
PRICE_ALERT_LIMIT = 5
WATCHLIST_LIMIT_PREMIUM = 25
PRICE_ALERT_LIMIT_PREMIUM = 25

# --- Analysis & Content ---
DEFAULT_ANALYSIS_PERIOD = 30
ANALYSIS_PERIODS = [14, 30, 90]
CRYPTO_PANIC_FILTER = "important"
NEWS_API_PAGE_SIZE = 7

# --- Localization ---
DEFAULT_LANGUAGE = "ru"
SUPPORTED_LANGUAGES = ["ru", "en"]

# --- Database ---
from src.config import config

DATABASE_URL = config().paths.db_url


PROXY_URL = os.getenv("PROXY_URL")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
