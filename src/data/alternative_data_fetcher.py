"""
Alternative Data Fetcher
========================

Production implementation for ingesting non-traditional alpha sources:
- Fear & Greed Index (Alternative.me)
- Crypto Market Trends (CoinGecko)
- DeFi TVL & Stablecoin Flows (DefiLlama)
- News Sentiment (NewsAPI)

Note: Most APIs used here are free public tiers or requires optional keys in .env.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import aiohttp
from pycoingecko import CoinGeckoAPI

from src.config.unified_config import load_config

logger = logging.getLogger(__name__)

class AlternativeDataFetcher:
    """
    Asynchronous fetcher for alternative data sources.
    """

    def __init__(self):
        self.config = load_config()
        self._session: aiohttp.ClientSession | None = None

        # API Keys from Environment
        self.cg_api_key = os.getenv("COINGECKO_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_ORG_KEY") or os.getenv("NEWS_API_KEY")

        # Initialize CoinGecko with API Key if available
        if self.cg_api_key:
            self._coingecko = CoinGeckoAPI(api_key=self.cg_api_key)
        else:
            self._coingecko = CoinGeckoAPI()

        # API Endpoints
        self.fng_api_url = "https://api.alternative.me/fng/"
        self.defi_api_url = "https://api.llama.fi"
        self.news_api_url = "https://newsapi.org/v2/everything"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()

    async def fetch_fear_and_greed(self) -> dict[str, Any]:
        """
        Fetch the Crypto Fear & Greed Index from Alternative.me.
        """
        session = await self._get_session()
        try:
            async with session.get(self.fng_api_url, params={"limit": 1}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    item = data['data'][0]
                    return {
                        "value": int(item['value']),
                        "classification": item['value_classification'],
                        "timestamp": int(item['timestamp'])
                    }
                logger.warning(f"Failed to fetch FnG Index: {resp.status}")
        except Exception as e:
            logger.error(f"Error fetching FnG: {e}")

        return {"value": 50, "classification": "Neutral", "timestamp": 0} # Fallback

    async def fetch_coingecko_global(self) -> dict[str, Any]:
        """
        Fetch global crypto market metrics via CoinGecko.
        """
        try:
            loop = asyncio.get_event_loop()
            global_data = await loop.run_in_executor(
                None,
                self._coingecko.get_global
            )

            data = global_data.get('data', {})
            return {
                "active_cryptocurrencies": data.get('active_cryptocurrencies', 0),
                "btc_dominance": data.get('market_cap_percentage', {}).get('btc', 50.0),
                "eth_dominance": data.get('market_cap_percentage', {}).get('eth', 20.0),
                "total_market_cap_usd": data.get('total_market_cap', {}).get('usd', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching CoinGecko global: {e}")
            return {"btc_dominance": 50.0}

    async def fetch_news_sentiment(self, query: str = "crypto") -> dict[str, Any]:
        """
        Fetch recent news and analyze sentiment using simple heuristics.
        Requires NEWS_API_ORG_KEY in .env.
        """
        if not self.news_api_key:
            return {"sentiment": 0.5, "count": 0, "status": "no_key"}

        session = await self._get_session()
        try:
            params = {
                "q": query,
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": self.news_api_key
            }
            async with session.get(self.news_api_url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    articles = data.get('articles', [])

                    # Basic sentiment heuristic (can be replaced with TextBlob/VADER)
                    positive_words = ['bullish', 'surge', 'growth', 'gain', 'breakout', 'success', 'adoption']
                    negative_words = ['bearish', 'crash', 'drop', 'loss', 'hack', 'scam', 'ban', 'regulation']

                    scores = []
                    for art in articles:
                        text = (art.get('title', '') + ' ' + art.get('description', '')).lower()
                        pos = sum(1 for w in positive_words if w in text)
                        neg = sum(1 for w in negative_words if w in text)
                        if pos + neg > 0:
                            scores.append(pos / (pos + neg))
                        else:
                            scores.append(0.5)

                    avg_sentiment = sum(scores) / len(scores) if scores else 0.5
                    return {"sentiment": avg_sentiment, "count": len(articles), "status": "ok"}
        except Exception as e:
            logger.error(f"Error fetching NewsAPI: {e}")

        return {"sentiment": 0.5, "count": 0, "status": "error"}

    async def fetch_defillama_metrics(self) -> dict[str, Any]:
        """
        Fetch DeFi TVL from DefiLlama.
        """
        session = await self._get_session()
        try:
            async with session.get(f"{self.defi_api_url}/v2/chains") as resp:
                if resp.status == 200:
                    chains = await resp.json()
                    total_tvl = sum(c.get('tvl', 0) for c in chains)
                    return {'total_tvl': total_tvl}
        except Exception as e:
            logger.error(f"Error fetching DefiLlama stats: {e}")

        return {'total_tvl': 0}

    async def get_alpha_signals(self, symbol: str = "BTC/USDT") -> dict[str, Any]:
        """
        Aggregate alternative data into actionable alpha signals.
        """
        fng, global_m, defi, news = await asyncio.gather(
            self.fetch_fear_and_greed(),
            self.fetch_coingecko_global(),
            self.fetch_defillama_metrics(),
            self.fetch_news_sentiment()
        )

        fng_val = fng.get('value', 50)
        btc_dom = global_m.get('btc_dominance', 50.0)
        news_sent = news.get('sentiment', 0.5)

        # Composite Alpha Score (0-100)
        # Weights: FnG (40%), News (40%), BTC Dom (20%)

        # 1. FnG contribution (contrarian: low FnG is positive)
        fng_score = (100 - fng_val)

        # 2. News contribution
        news_score = news_sent * 100

        # 3. BTC Dominance (Flight to quality is neutral/positive for BTC, negative for alts)
        dom_score = 50
        if symbol.startswith("BTC"):
            dom_score = 50 + (btc_dom - 50) * 2 # Rising dominance is good for BTC
        else:
            dom_score = 50 - (btc_dom - 50) * 2 # Rising dominance is bad for Alts

        alpha_score = (fng_score * 0.4) + (news_score * 0.4) + (dom_score * 0.2)

        return {
            "symbol": symbol,
            "alpha_score": min(100, max(0, alpha_score)),
            "components": {
                "fng": fng_val,
                "news_sentiment": news_sent,
                "btc_dominance": btc_dom,
                "total_tvl": defi.get('total_tvl')
            },
            "timestamp": datetime.utcnow().isoformat()
        }

async def main_demo():
    """Demo for alternative data fetcher."""
    fetcher = AlternativeDataFetcher()
    try:
        signals = await fetcher.get_alpha_signals()
        import json
        print("--- Alpha Signal Report ---")
        print(json.dumps(signals, indent=2))
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main_demo())
