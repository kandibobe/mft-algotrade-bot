"""
Alternative Data Fetcher
========================

Fetches data from alternative sources like sentiment analysis APIs, on-chain data, etc.
"""

import logging
import httpx
from typing import Any, Dict

logger = logging.getLogger(__name__)

class AlternativeDataFetcher:
    """
    A class to fetch data from various alternative sources.
    """

    def __init__(self, http_client: httpx.AsyncClient | None = None):
        self.http_client = http_client or httpx.AsyncClient()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()

    async def fetch_fear_and_greed_index(self) -> Dict[str, Any]:
        """
        Fetches the Fear and Greed Index from api.alternative.me.
        """
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            data = response.json()
            return data['data'][0]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Fear and Greed Index: {e}")
            return {}
        except Exception as e:
            logger.error(f"An error occurred while fetching Fear and Greed Index: {e}")
            return {}

    async def fetch_crypto_news_sentiment(self, symbol: str) -> float:
        """
        Fetches news sentiment for a specific symbol.
        Simulates sentiment analysis from public APIs.
        Returns a score from -1.0 (Bearish) to 1.0 (Bullish).
        """
        # In a real implementation, this would call Cryptopanic or similar API
        # and use a model like VADER or TextBlob for scoring.
        logger.info(f"Fetching news sentiment for {symbol}...")
        
        # Placeholder: returning a slightly positive neutral sentiment
        return 0.15 

async def get_fear_and_greed_index() -> Dict[str, Any]:
    """
    Convenience function to fetch the Fear and Greed Index.
    """
    async with AlternativeDataFetcher() as fetcher:
        return await fetcher.fetch_fear_and_greed_index()
