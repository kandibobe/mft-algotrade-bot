#!/usr/bin/env python3
"""
Exchange Connector MCP Server
==============================

MCP сервер для управления подключениями к биржам.
Использует централизованную конфигурацию проекта.
"""

import asyncio
import json
import logging
import os
import sys

# MCP SDK
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.unified_config import load_config
from src.data.async_fetcher import AsyncOrderExecutor, FetcherConfig

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("mcp-exchange-connector")

# Global executor instance
executor: AsyncOrderExecutor = None

async def initialize_executor():
    """Инициализация executor с использованием unified_config."""
    global executor

    if executor is not None:
        return

    try:
        # Загружаем конфиг проекта
        # Пытаемся найти config.json или использовать дефолтный путь
        config_path = os.getenv("CONFIG_PATH", "config/config.json")
        if not os.path.exists(config_path):
            config_path = None # Будет загружено из переменных окружения

        full_config = load_config(config_path)

        # Настройки биржи
        exchange_name = full_config.exchange.name

        config = FetcherConfig(
            exchange=exchange_name,
            api_key=full_config.exchange.api_key,
            api_secret=full_config.exchange.api_secret,
            sandbox=full_config.exchange.sandbox,
            rate_limit=full_config.exchange.rate_limit,
            timeout=full_config.exchange.timeout_ms
        )

        executor = AsyncOrderExecutor(config)
        await executor.connect()
        logger.info(f"Exchange executor initialized for {exchange_name} (Sandbox: {full_config.exchange.sandbox})")
    except Exception as e:
        logger.error(f"Failed to initialize exchange executor: {e}")
        raise

async def cleanup_executor():
    """Закрытие соединений."""
    global executor
    if executor is not None:
        await executor.close()
        executor = None
        logger.info("Exchange executor closed")

# Create MCP server
server = Server("exchange-connector")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="get_balance",
            description="Получить баланс аккаунта. Можно указать конкретную валюту (например, USDT).",
            inputSchema={
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "Код валюты (опционально)",
                    },
                },
            },
        ),
        types.Tool(
            name="get_ticker",
            description="Получить текущие котировки для пары (например, BTC/USDT)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара",
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get_order_status",
            description="Проверить статус ордера по его ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "ID ордера",
                    },
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара",
                    },
                },
                "required": ["order_id", "symbol"],
            },
        ),
        types.Tool(
            name="get_connection_status",
            description="Проверить статус подключения к бирже и настройки API",
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

    try:
        await initialize_executor()
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": f"Init failed: {e}"}))]

    try:
        if name == "get_balance":
            currency = arguments.get("currency")
            balance_data = await executor.exchange.fetch_balance()

            if currency:
                info = balance_data.get(currency, {})
                result = {
                    "success": True,
                    "currency": currency,
                    "free": info.get("free", 0),
                    "used": info.get("used", 0),
                    "total": info.get("total", 0),
                }
            else:
                # Только непустые балансы
                balances = {k: v for k, v in balance_data.items() if isinstance(v, dict) and v.get("total", 0) > 0}
                result = {"success": True, "balances": balances}

            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_ticker":
            symbol = arguments.get("symbol")
            ticker = await executor.exchange.fetch_ticker(symbol)
            result = {
                "success": True,
                "symbol": symbol,
                "last": ticker.get("last"),
                "bid": ticker.get("bid"),
                "ask": ticker.get("ask"),
                "volume": ticker.get("baseVolume"),
                "timestamp": ticker.get("timestamp")
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_order_status":
            order_id = arguments.get("order_id")
            symbol = arguments.get("symbol")
            order = await executor.fetch_order(order_id, symbol)
            return [types.TextContent(type="text", text=json.dumps({"success": True, "order": order}, indent=2))]

        elif name == "get_connection_status":
            status = await executor.exchange.fetch_status()
            result = {
                "success": True,
                "status": status.get("status"),
                "exchange": executor.config.exchange,
                "sandbox": executor.config.sandbox,
                "api_key_set": bool(executor.config.api_key)
            }
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [types.TextContent(type="text", text=json.dumps({"success": False, "error": f"Unknown tool: {name}"}))]

    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

async def main():
    """Запуск MCP сервера."""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="exchange-connector",
                    server_version="1.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await cleanup_executor()

if __name__ == "__main__":
    asyncio.run(main())
