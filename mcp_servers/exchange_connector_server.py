#!/usr/bin/env python3
"""
Exchange Connector MCP Server
==============================

MCP сервер для управления подключениями к биржам и выполнения торговых операций.

Предоставляет инструменты:
- create_order: Создание ордера
- cancel_order: Отмена ордера
- get_balance: Получение баланса
- get_open_orders: Получение открытых ордеров
- get_order_status: Проверка статуса ордера

ВНИМАНИЕ: Этот сервер работает с реальными торговыми операциями!
Используйте с осторожностью в production.

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
from src.data.async_fetcher import AsyncOrderExecutor, FetcherConfig

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global executor instance
executor: AsyncOrderExecutor = None


async def initialize_executor():
    """Инициализация executor с API ключами из переменных окружения."""
    global executor
    
    if executor is not None:
        return
    
    # Определяем биржу (по умолчанию Binance)
    exchange = os.getenv("EXCHANGE_NAME", "binance")
    
    config = FetcherConfig(
        exchange=exchange,
        api_key=os.getenv(f"{exchange.upper()}_API_KEY"),
        api_secret=os.getenv(f"{exchange.upper()}_API_SECRET"),
        rate_limit=True,
        max_retries=3,
    )
    
    executor = AsyncOrderExecutor(config)
    await executor.connect()
    logger.info(f"Exchange executor initialized for {exchange}")


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
            name="create_limit_order",
            description="Создать лимитный ордер на покупку/продажу",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара (например: BTC/USDT)",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Сторона: buy или sell",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Количество базовой валюты",
                    },
                    "price": {
                        "type": "number",
                        "description": "Цена лимитного ордера",
                    },
                },
                "required": ["symbol", "side", "amount", "price"],
            },
        ),
        types.Tool(
            name="create_market_order",
            description="Создать рыночный ордер (выполняется немедленно по рыночной цене)",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Торговая пара",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["buy", "sell"],
                        "description": "Сторона: buy или sell",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Количество базовой валюты",
                    },
                },
                "required": ["symbol", "side", "amount"],
            },
        ),
        types.Tool(
            name="cancel_order",
            description="Отменить открытый ордер",
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
            name="get_order_status",
            description="Получить статус ордера",
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
            name="get_balance",
            description="Получить баланс аккаунта",
            inputSchema={
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "Валюта (опционально, например: USDT, BTC)",
                    },
                },
            },
        ),
        types.Tool(
            name="get_connection_status",
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
    
    # Инициализация executor при первом вызове
    await initialize_executor()
    
    try:
        if name == "create_limit_order":
            symbol = arguments.get("symbol")
            side = arguments.get("side")
            amount = arguments.get("amount")
            price = arguments.get("price")
            
            logger.info(f"Creating limit order: {side} {amount} {symbol} @ {price}")
            
            order = await executor.create_limit_order(
                symbol=symbol,
                side=side,
                amount=amount,
                price=price,
            )
            
            result = {
                "success": True,
                "order_id": order.get("id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "type": order.get("type"),
                "amount": order.get("amount"),
                "price": order.get("price"),
                "status": order.get("status"),
                "timestamp": order.get("timestamp"),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "create_market_order":
            symbol = arguments.get("symbol")
            side = arguments.get("side")
            amount = arguments.get("amount")
            
            logger.info(f"Creating market order: {side} {amount} {symbol}")
            
            order = await executor.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
            )
            
            result = {
                "success": True,
                "order_id": order.get("id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "type": order.get("type"),
                "amount": order.get("amount"),
                "filled": order.get("filled"),
                "average_price": order.get("average"),
                "status": order.get("status"),
                "timestamp": order.get("timestamp"),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "cancel_order":
            order_id = arguments.get("order_id")
            symbol = arguments.get("symbol")
            
            logger.info(f"Canceling order: {order_id} for {symbol}")
            
            result_data = await executor.cancel_order(order_id, symbol)
            
            result = {
                "success": True,
                "order_id": result_data.get("id"),
                "symbol": result_data.get("symbol"),
                "status": result_data.get("status"),
                "message": "Order cancelled successfully",
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_order_status":
            order_id = arguments.get("order_id")
            symbol = arguments.get("symbol")
            
            order = await executor.fetch_order(order_id, symbol)
            
            result = {
                "success": True,
                "order_id": order.get("id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "type": order.get("type"),
                "amount": order.get("amount"),
                "filled": order.get("filled"),
                "remaining": order.get("remaining"),
                "price": order.get("price"),
                "average_price": order.get("average"),
                "status": order.get("status"),
                "timestamp": order.get("timestamp"),
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_balance":
            currency = arguments.get("currency")
            
            balance_data = await executor.exchange.fetch_balance()
            
            if currency:
                # Specific currency
                balance_info = balance_data.get(currency, {})
                result = {
                    "success": True,
                    "currency": currency,
                    "free": balance_info.get("free", 0),
                    "used": balance_info.get("used", 0),
                    "total": balance_info.get("total", 0),
                }
            else:
                # All currencies with non-zero balance
                balances = {}
                for curr, info in balance_data.items():
                    if curr not in ["info", "free", "used", "total"]:
                        if info.get("total", 0) > 0:
                            balances[curr] = {
                                "free": info.get("free", 0),
                                "used": info.get("used", 0),
                                "total": info.get("total", 0),
                            }
                
                result = {
                    "success": True,
                    "balances": balances,
                    "currencies_count": len(balances),
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_connection_status":
            # Проверяем подключение через ping или простой запрос
            try:
                await executor.exchange.fetch_status()
                status = {
                    "success": True,
                    "connected": executor._connected,
                    "exchange": executor.config.exchange,
                    "message": "Connection active",
                }
            except Exception as e:
                status = {
                    "success": False,
                    "connected": False,
                    "exchange": executor.config.exchange,
                    "error": str(e),
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
    logger.info("Starting Exchange Connector MCP Server...")
    
    # Проверка наличия API ключей
    exchange = os.getenv("EXCHANGE_NAME", "binance")
    api_key = os.getenv(f"{exchange.upper()}_API_KEY")
    
    if not api_key:
        logger.warning(f"WARNING: {exchange.upper()}_API_KEY not set. Some features will be limited.")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="exchange-connector",
                    server_version="1.0.0",
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
