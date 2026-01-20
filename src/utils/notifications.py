"""
Stoic Citadel - Notification System
====================================

Handles multi-channel notifications (Telegram, Slack, Email).
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)


class Notifier:
    """Central notification manager."""

    def __init__(self):
        # Telegram
        self.telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        # Slack
        self.slack_webhook = os.environ.get("SLACK_WEBHOOK_URL")

        # Email (SMTP)
        self.email_host = os.environ.get("EMAIL_HOST")
        self.email_port = int(os.environ.get("EMAIL_PORT", 587))
        self.email_user = os.environ.get("EMAIL_USER")
        self.email_password = os.environ.get("EMAIL_PASSWORD")
        self.email_recipient = os.environ.get("EMAIL_RECIPIENT")

        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram notifications not configured in environment")

    def send_notification(self, message: str, level: str = "info") -> bool:
        """
        Send notification to configured channels.

        Args:
            message: Message text
            level: info, warning, critical

        Returns:
            True if at least one channel succeeded
        """
        # Add emoji based on level
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(level, "")

        full_message = f"{emoji} {message}"

        success = False

        # 1. Telegram
        if self._send_telegram(full_message):
            success = True

        # 2. Slack
        if self._send_slack(full_message):
            success = True

        # 3. Email (Critical/Warning only to avoid spam)
        if level in ["critical", "warning"]:
            if self._send_email(full_message, level):
                success = True

        # 4. Always Log
        if level == "critical":
            logger.critical(full_message)
        elif level == "warning":
            logger.warning(full_message)
        else:
            logger.info(full_message)

        return success

    def _send_telegram(self, message: str) -> bool:
        if not (self.telegram_token and self.telegram_chat_id):
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                return True
            else:
                logger.error(f"Telegram API error: {response.text}")
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
        return False

    def _send_slack(self, message: str) -> bool:
        if not self.slack_webhook:
            return False

        try:
            payload = {"text": message}
            response = requests.post(self.slack_webhook, json=payload, timeout=10)
            if response.status_code == 200:
                return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
        return False

    def _send_email(self, message: str, level: str) -> bool:
        if not (
            self.email_host and self.email_user and self.email_password and self.email_recipient
        ):
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_user
            msg["To"] = self.email_recipient
            msg["Subject"] = f"Stoic Citadel Alert: {level.upper()}"

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP(self.email_host, self.email_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_user, self.email_recipient, text)
            server.quit()
            logger.info(f"Email notification sent to {self.email_recipient}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Email notification: {e}")
            return False


_notifier_instance = None


def get_notifier() -> Notifier:
    """Global notifier access."""
    global _notifier_instance
    if _notifier_instance is None:
        _notifier_instance = Notifier()
    return _notifier_instance
