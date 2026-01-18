import logging

import aiohttp
import requests

from src.config.unified_config import load_config

logger = logging.getLogger(__name__)


class TelegramBot:
    """
    Telegram bot for sending notifications about trades and errors.
    Supports both sync and async operations.
    """

    def __init__(self, token: str | None = None, chat_id: str | None = None):
        config = load_config()
        self.token = token or config.telegram.token
        self.chat_id = chat_id or config.telegram.chat_id
        self.enabled = config.telegram.enabled

        if not self.token or not self.chat_id:
            logger.warning("Telegram token or chat_id not configured. Notifications disabled.")
            self.enabled = False

    def send_message(self, message: str):
        """
        Send a message to the configured Telegram chat (Synchronous).
        WARNING: Blocks the thread. Use only in Sync layers.
        """
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Telegram message: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")

    async def send_message_async(self, message: str):
        """
        Send a message to the configured Telegram chat (Asynchronous).
        Use this in AsyncIO loops to avoid blocking.
        """
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send async Telegram message: {e}")
