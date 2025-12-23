#!/usr/bin/env python3
"""
Trading Data MCP Server
=======================

MCP сервер для работы с торговыми данными через биржевые API.

Предоставляет инструменты:
- fetch_ohlcv: Получение исторических данных OHLCV
- fetch_ticker: Получение текущих цен
- fetch_orderbook: Получение стакана заявок
- fetch_multiple_pairs: Параллельное получение данных по нескольким парам

Author: Stoic Citadel Team
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

# MCP SDK
try:
    from mcp.server.models import InitializationOptions
    from mcp.server import NotificationOptions, Server
    from mcp.server.stdio import stdio_server
    from mcp import types
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.async_fetcher import AsyncDataFetcher, FetcherConfig

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global fetcher instance
fetcher: AsyncDataFetcher = None


async def initialize_fetcher():
    """Инициализация fetcher с API ключами из переменных окружения."""
    global fetcher
    
    if fetcher is not None:
        return
    
    config = FetcherConfig(
        exchange="binance",
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        rate_limit=True,
        max_retries=5,
    )
    
    fetcher = AsyncDataFetcher(config)
    await fetcher.connect()
    logger.info("Trading data fetcher initialized")


async def cleanup_fetcher():
    """Закрытие соединений."""
    global fetcher
    if fetcher is not None:
        await fetcher.close()
        fetcher = None
        logger.info("Trading data fetcher closed")


# Create MCP server
server = Server("trading-data")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="fetch_ohlcv",
            description="Получить исторические OHLCV данные для торговой пары",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (например: BTC/USDT)",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Таймфрейм (1m, 5m, 15m, 1h, 4h, 1d)",
                        "default": "1h",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Количество свечей (макс. 1000)",
                        "default": 500,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="fetch_ticker",
            description="Получить текущую цену и объем торговой пары",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (например: BTC/USDT)",
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="fetch_orderbook",
            description="Получить стакан заявок (bids/asks)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (например: BTC/USDT)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Глубина стакана (по умолчанию 20)",
                        "default": 20,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="fetch_multiple_pairs",
            description="Получить данные сразу для нескольких торговых пар параллельно",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список торговых пар (например: ['BTC/USDT', 'ETH/USDT'])",
                    },
                    "timeframe": {
                        "type": "string",
                        "description": "Таймфрейм для всех пар",
                        "default": "1h",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Количество свечей",
                        "default": 500,
                    },
                },
                "required": ["symbols"],
            },
        ),
        types.Tool(
            name="get_exchange_status",
            description="Проверить статус подключения к бирже",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Обработка вызовов инструментов."""
    
    # Инициализация fetcher при первом вызове
    await initialize_fetcher()
    
    try:
        if name == "fetch_ohlcv":
            symbol = arguments.get("symbol")
            timeframe = arguments.get("timeframe", "1h")
            limit = arguments.get("limit", 500)
            
            df = await fetcher.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if df.empty:
                result = {
                    "success": False,
                    "error": "No data received",
                }
            else:
                result = {
                    "success": True,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "first_candle": df.index[0].isoformat(),
                    "last_candle": df.index[-1].isoformat(),
                    "data_preview": df.tail(5).to_dict(orient="records"),
                    "statistics": {
                        "open": {
                            "min": float(df["open"].min()),
                            "max": float(df["open"].max()),
                            "mean": float(df["open"].mean()),
                        },
                        "volume": {
                            "total": float(df["volume"].sum()),
                            "mean": float(df["volume"].mean()),
                        },
                    },
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "fetch_ticker":
            symbol = arguments.get("symbol")
            ticker = await fetcher.fetch_ticker(symbol)
            
            result = {
                "success": True,
                "symbol": symbol,
                "ticker": {
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "last": ticker.get("last"),
                    "volume": ticker.get("baseVolume"),
                    "change_24h": ticker.get("percentage"),
                    "high_24h": ticker.get("high"),
                    "low_24h": ticker.get("low"),
                },
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "fetch_orderbook":
            symbol = arguments.get("symbol")
            limit = arguments.get("limit", 20)
            
            orderbook = await fetcher.fetch_orderbook(symbol, limit)
            
            result = {
                "success": True,
                "symbol": symbol,
                "timestamp": orderbook.get("timestamp"),
                "bids": orderbook["bids"][:10],  # Top 10
                "asks": orderbook["asks"][:10],
                "bid_ask_spread": float(orderbook["asks"][0][0]) - float(orderbook["bids"][0][0]),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "fetch_multiple_pairs":
            symbols = arguments.get("symbols", [])
            timeframe = arguments.get("timeframe", "1h")
            limit = arguments.get("limit", 500)
            
            data = await fetcher.fetch_multiple(symbols, timeframe, limit)
            
            result = {
                "success": True,
                "timeframe": timeframe,
                "pairs_fetched": len(data),
                "pairs": {},
            }
            
            for symbol, df in data.items():
                if not df.empty:
                    result["pairs"][symbol] = {
                        "rows": len(df),
                        "first_candle": df.index[0].isoformat(),
                        "last_candle": df.index[-1].isoformat(),
                        "last_close": float(df["close"].iloc[-1]),
                    }
                else:
                    result["pairs"][symbol] = {"error": "No data"}
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_exchange_status":
            status = {
                "success": True,
                "connected": fetcher._connected,
                "exchange": fetcher.config.exchange,
                "rate_limit_enabled": fetcher.config.rate_limit,
            }
            
            return [types.TextContent(type="text", text=json.dumps(status, indent=2))]
        
        else:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "error": f"Unknown tool: {name}"}),
                )
            ]
    
    except Exception as e:
        logger.error(f"Error in {name}: {e}", exc_info=True)
        return [
            types.TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "tool": name,
                }),
            )
        ]


async def main():
    """Запуск MCP сервера."""
    logger.info("Starting Trading Data MCP Server...")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="trading-data",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await cleanup_fetcher()


if __name__ == "__main__":
    asyncio.run(main())
