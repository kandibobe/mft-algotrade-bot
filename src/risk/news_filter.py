"""
News Filter Module
==================

Prevents trading during High Impact news events.
Fetches economic calendar from ForexFactory (via faireconomy.media JSON).

Rules:
- Block trading 10 minutes before and 10 minutes after High Impact news.
- Filter by currency (USD + Base/Quote of the pair).
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
from dateutil import parser

from src.utils.logger import log

logger = logging.getLogger(__name__)


class NewsFilter:
    """
    Economic Calendar News Filter.
    """

    def __init__(self, block_minutes_before: int = 10, block_minutes_after: int = 10):
        self.block_before = timedelta(minutes=block_minutes_before)
        self.block_after = timedelta(minutes=block_minutes_after)
        self.events = []
        self.last_update = datetime.min.replace(tzinfo=timezone.utc)
        self.update_interval = timedelta(hours=1)
        self._lock = asyncio.Lock()
        
        # URL for ForexFactory calendar JSON (This Week)
        self.url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

    async def _fetch_calendar(self):
        """Fetch calendar from source."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._process_events(data)
                        self.last_update = datetime.now(timezone.utc)
                        logger.info(f"Updated Economic Calendar: {len(self.events)} high impact events found.")
                    else:
                        logger.warning(f"Failed to fetch calendar: Status {response.status}")
        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")

    def _process_events(self, data: list):
        """Parse and filter high impact events."""
        high_impact = []
        for event in data:
            if event.get("impact") == "High":
                try:
                    # date format: "2024-01-20T14:30:00-05:00"
                    dt_str = event.get("date")
                    if dt_str:
                        # Parse and ensure timezone awareness (convert to UTC)
                        dt = parser.parse(dt_str)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)
                        
                        high_impact.append({
                            "title": event.get("title"),
                            "country": event.get("country"),
                            "time": dt,
                            "impact": "High"
                        })
                except Exception as e:
                    logger.warning(f"Failed to parse event date {event}: {e}")
        
        # Sort by time
        high_impact.sort(key=lambda x: x["time"])
        self.events = high_impact

    async def update(self):
        """Trigger update if needed."""
        now = datetime.now(timezone.utc)
        if now - self.last_update > self.update_interval:
            async with self._lock:
                # Double check inside lock
                if now - self.last_update > self.update_interval:
                    await self._fetch_calendar()

    async def check_news_impact(self, symbol: str) -> tuple[bool, str | None]:
        """
        Check if trading is allowed for the symbol based on news.
        
        Args:
            symbol: e.g. "BTC/USD", "EUR/USD"
            
        Returns:
            (allowed: bool, reason: str | None)
        """
        await self.update()
        
        if not self.events:
            return True, None
            
        now = datetime.now(timezone.utc)
        
        # Determine currencies involved
        # For Crypto, usually USD news impacts everything.
        # But we can try to filter.
        # "BTC/USDT" -> BTC, USDT (USD)
        parts = symbol.replace("/", "").replace("-", "")
        # Heuristic: match country code. 
        # "USD" matches "USD" country.
        # "EUR" matches "EUR" country.
        # "GBP" matches "GBP" country.
        
        relevant_currencies = ["USD"] # USD affects everything
        if "EUR" in symbol: relevant_currencies.append("EUR")
        if "GBP" in symbol: relevant_currencies.append("GBP")
        if "JPY" in symbol: relevant_currencies.append("JPY")
        if "CAD" in symbol: relevant_currencies.append("CAD")
        if "AUD" in symbol: relevant_currencies.append("AUD")
        if "NZD" in symbol: relevant_currencies.append("NZD")
        if "CHF" in symbol: relevant_currencies.append("CHF")
        
        for event in self.events:
            # Check currency relevance
            if event["country"] not in relevant_currencies:
                continue
                
            event_time = event["time"]
            
            # Check window
            # Block: [Time - Before, Time + After]
            block_start = event_time - self.block_before
            block_end = event_time + self.block_after
            
            if block_start <= now <= block_end:
                reason = f"High Impact News ({event['country']}): {event['title']} at {event_time.strftime('%H:%M UTC')}"
                return False, reason
                
        return True, None