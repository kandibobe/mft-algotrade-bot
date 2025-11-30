#!/usr/bin/env python3
"""
Telegram Bot Connector
=======================

Send alerts and notifications via Telegram.

Features:
- Trade notifications
- Performance reports
- Error alerts
- Interactive commands

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "ℹ️"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    TRADE_OPEN = "🟢"
    TRADE_CLOSE = "🔴"
    PROFIT = "💰"
    LOSS = "📉"


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    token: str
    chat_id: str
    parse_mode: str = "HTML"
    disable_notification: bool = False
    rate_limit: float = 1.0  # Min seconds between messages


class TelegramConnector:
    """
    Telegram notification connector.
    
    Usage:
        config = TelegramConfig(
            token="YOUR_BOT_TOKEN",
            chat_id="YOUR_CHAT_ID"
        )
        
        telegram = TelegramConnector(config)
        
        # Send simple message
        await telegram.send_message("🚀 Bot started!")
        
        # Send trade alert
        await telegram.send_trade_alert(
            action="BUY",
            symbol="BTC/USDT",
            price=95000,
            quantity=0.01
        )
        
        # Send performance report
        await telegram.send_performance_report(metrics)
    """
    
    BASE_URL = "https://api.telegram.org/bot"
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_message_time = 0.0
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    async def start(self):
        """Start the connector."""
        self._session = aiohttp.ClientSession()
        self._running = True
        asyncio.create_task(self._process_queue())
        logger.info("Telegram connector started")
    
    async def stop(self):
        """Stop the connector."""
        self._running = False
        if self._session:
            await self._session.close()
    
    # =========================================================================
    # Basic Messaging
    # =========================================================================
    
    async def send_message(
        self,
        text: str,
        parse_mode: Optional[str] = None,
        disable_notification: Optional[bool] = None,
        reply_markup: Optional[Dict] = None
    ) -> bool:
        """Send a text message."""
        await self._message_queue.put({
            "type": "message",
            "text": text,
            "parse_mode": parse_mode or self.config.parse_mode,
            "disable_notification": disable_notification or self.config.disable_notification,
            "reply_markup": reply_markup
        })
        return True
    
    async def _send_message_immediate(self, message: Dict) -> bool:
        """Send message immediately (internal)."""
        if not self._session:
            self._session = aiohttp.ClientSession()
        
        url = f"{self.BASE_URL}{self.config.token}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": message["text"],
            "parse_mode": message.get("parse_mode", self.config.parse_mode),
            "disable_notification": message.get("disable_notification", False)
        }
        
        if message.get("reply_markup"):
            payload["reply_markup"] = message["reply_markup"]
        
        try:
            async with self._session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                else:
                    error = await resp.text()
                    logger.error(f"Telegram API error: {error}")
                    return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    async def _process_queue(self):
        """Process message queue with rate limiting."""
        import time
        
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0
                )
                
                # Rate limiting
                now = time.time()
                elapsed = now - self._last_message_time
                if elapsed < self.config.rate_limit:
                    await asyncio.sleep(self.config.rate_limit - elapsed)
                
                await self._send_message_immediate(message)
                self._last_message_time = time.time()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
    
    # =========================================================================
    # Trade Alerts
    # =========================================================================
    
    async def send_trade_alert(
        self,
        action: str,  # BUY, SELL, CLOSE
        symbol: str,
        price: float,
        quantity: float,
        pnl: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        reason: Optional[str] = None
    ):
        """Send trade notification."""
        if action.upper() == "BUY":
            emoji = AlertLevel.TRADE_OPEN.value
            title = "LONG ENTRY"
        elif action.upper() == "SELL":
            emoji = AlertLevel.TRADE_CLOSE.value
            title = "POSITION CLOSED"
        else:
            emoji = "🔄"
            title = action.upper()
        
        text = f"{emoji} <b>{title}</b>\n\n"
        text += f"📊 <b>Symbol:</b> {symbol}\n"
        text += f"💵 <b>Price:</b> ${price:,.2f}\n"
        text += f"📦 <b>Quantity:</b> {quantity}\n"
        
        if pnl is not None:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            text += f"{pnl_emoji} <b>P&L:</b> ${pnl:,.2f}"
            if pnl_pct is not None:
                text += f" ({pnl_pct:+.2f}%)"
            text += "\n"
        
        if reason:
            text += f"📝 <b>Reason:</b> {reason}\n"
        
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    async def send_stoploss_alert(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        loss: float,
        loss_pct: float
    ):
        """Send stop loss triggered alert."""
        text = f"{AlertLevel.WARNING.value} <b>STOP LOSS TRIGGERED</b>\n\n"
        text += f"📊 <b>Symbol:</b> {symbol}\n"
        text += f"➡️ <b>Entry:</b> ${entry_price:,.2f}\n"
        text += f"⬅️ <b>Exit:</b> ${exit_price:,.2f}\n"
        text += f"🔴 <b>Loss:</b> ${abs(loss):,.2f} ({loss_pct:+.2f}%)\n"
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    async def send_takeprofit_alert(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        profit: float,
        profit_pct: float
    ):
        """Send take profit triggered alert."""
        text = f"{AlertLevel.PROFIT.value} <b>TAKE PROFIT HIT!</b>\n\n"
        text += f"📊 <b>Symbol:</b> {symbol}\n"
        text += f"➡️ <b>Entry:</b> ${entry_price:,.2f}\n"
        text += f"⬅️ <b>Exit:</b> ${exit_price:,.2f}\n"
        text += f"🟢 <b>Profit:</b> ${profit:,.2f} (+{profit_pct:.2f}%)\n"
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    # =========================================================================
    # Performance Reports
    # =========================================================================
    
    async def send_performance_report(
        self,
        total_equity: float,
        total_pnl: float,
        total_pnl_pct: float,
        open_positions: int,
        win_rate: float,
        trades_today: int = 0,
        pnl_today: float = 0
    ):
        """Send daily/periodic performance report."""
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        text = f"📊 <b>PERFORMANCE REPORT</b>\n\n"
        text += f"💰 <b>Total Equity:</b> ${total_equity:,.2f}\n"
        text += f"{pnl_emoji} <b>Total P&L:</b> ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)\n"
        text += f"📍 <b>Open Positions:</b> {open_positions}\n"
        text += f"🎯 <b>Win Rate:</b> {win_rate:.1f}%\n"
        
        if trades_today > 0:
            text += f"\n📅 <b>Today:</b>\n"
            text += f"   Trades: {trades_today}\n"
            today_emoji = "🟢" if pnl_today >= 0 else "🔴"
            text += f"   P&L: {today_emoji} ${pnl_today:,.2f}\n"
        
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    async def send_daily_summary(
        self,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        best_trade: float,
        worst_trade: float
    ):
        """Send end-of-day summary."""
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        
        text = f"🌅 <b>DAILY SUMMARY</b>\n\n"
        text += f"📈 <b>Total Trades:</b> {trades}\n"
        text += f"✅ <b>Wins:</b> {wins}\n"
        text += f"❌ <b>Losses:</b> {losses}\n"
        text += f"{pnl_emoji} <b>Net P&L:</b> ${pnl:,.2f}\n"
        text += f"🏆 <b>Best Trade:</b> ${best_trade:,.2f}\n"
        text += f"💥 <b>Worst Trade:</b> ${worst_trade:,.2f}\n"
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    # =========================================================================
    # System Alerts
    # =========================================================================
    
    async def send_error_alert(self, error: str, context: Optional[str] = None):
        """Send error notification."""
        text = f"{AlertLevel.ERROR.value} <b>ERROR</b>\n\n"
        text += f"📝 {error}\n"
        if context:
            text += f"\n<i>Context: {context}</i>\n"
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text, disable_notification=False)
    
    async def send_critical_alert(self, message: str):
        """Send critical system alert."""
        text = f"{AlertLevel.CRITICAL.value} <b>CRITICAL ALERT</b>\n\n"
        text += f"{message}\n"
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text, disable_notification=False)
    
    async def send_bot_status(
        self,
        status: str,  # started, stopped, paused
        message: Optional[str] = None
    ):
        """Send bot status change notification."""
        emojis = {
            "started": "🚀",
            "stopped": "🛑",
            "paused": "⏸️",
            "resumed": "▶️",
            "error": "❌"
        }
        
        emoji = emojis.get(status.lower(), "ℹ️")
        text = f"{emoji} <b>Bot {status.upper()}</b>\n"
        
        if message:
            text += f"\n{message}\n"
        
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text)
    
    async def send_drawdown_alert(
        self,
        current_drawdown: float,
        max_allowed: float
    ):
        """Send drawdown warning."""
        level = AlertLevel.WARNING if current_drawdown < max_allowed else AlertLevel.CRITICAL
        
        text = f"{level.value} <b>DRAWDOWN ALERT</b>\n\n"
        text += f"📉 <b>Current Drawdown:</b> {current_drawdown:.2f}%\n"
        text += f"🚧 <b>Max Allowed:</b> {max_allowed:.2f}%\n"
        
        if current_drawdown >= max_allowed:
            text += f"\n⚠️ <b>Trading paused due to drawdown limit!</b>\n"
        
        text += f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        await self.send_message(text, disable_notification=False)
