import logging
import requests
from typing import Optional
from src.config.unified_config import load_config

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for sending notifications about trades and errors.
    """
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        config = load_config()
        self.token = token or config.telegram.token
        self.chat_id = chat_id or config.telegram.chat_id
        self.enabled = config.telegram.enabled

        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Notifications disabled.")
            self.enabled = False

    def send_message(self, message: str):
        """
        Send a message to the configured Telegram chat.
        """
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
