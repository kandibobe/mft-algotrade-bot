#!/usr/bin/env python3
"""
WebSocket Stream MCP Server
============================

MCP сервер для управления WebSocket стримами рыночных данных.

Предоставляет инструменты:
- subscribe: Подписаться на символ
- unsubscribe: Отписаться от символа
- get_status: Статус подключений
- get_stats: Статистика сообщений

Author: Stoic Citadel Team
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, List, Optional

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
from src.websocket.data_stream import WebSocketDataStream, StreamConfig, Exchange

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global stream instance
stream: Optional[WebSocketDataStream] = None


async def initialize_stream():
    """Инициализация WebSocket стрима."""
    global stream
    if stream is None:
        config = StreamConfig(
            exchange=Exchange.BINANCE,
            symbols=["BTC/USDT", "ETH/USDT"],
            channels=["ticker", "trade"]
        )
        stream = WebSocketDataStream(config)
        # Run stream in background
        asyncio.create_task(stream.start())
        logger.info("WebSocket data stream started")


async def cleanup_stream():
    """Остановка стрима."""
    global stream
    if stream is not None:
        await stream.stop()
        stream = None
        logger.info("WebSocket data stream stopped")


# Create MCP server
server = Server("websocket-stream")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="subscribe",
            description="Подписаться на рыночные данные символа",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Например: BTC/USDT"},
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="unsubscribe",
            description="Отписаться от рыночных данных символа",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get_stream_status",
            description="Получить статус и статистику стрима",
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
    
    await initialize_stream()
    
    try:
        if name == "subscribe":
            symbol = arguments.get("symbol")
            await stream.subscribe_symbol(symbol)
            return [types.TextContent(type="text", text=json.dumps({"success": True, "subscribed": symbol}))]
        
        elif name == "unsubscribe":
            symbol = arguments.get("symbol")
            await stream.unsubscribe_symbol(symbol)
            return [types.TextContent(type="text", text=json.dumps({"success": True, "unsubscribed": symbol}))]
        
        elif name == "get_stream_status":
            stats = stream.get_stats()
            health = await stream.health_check()
            
            result = {
                "success": True,
                "status": health["status"],
                "connected": stats["connected"],
                "uptime": stats["uptime_seconds"],
                "messages_received": stats["messages_received"],
                "mps": stats["messages_per_second"],
                "subscribed_symbols": stats["subscribed_symbols"],
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [types.TextContent(type="text", text=json.dumps({"success": False, "error": f"Unknown tool: {name}"}))]
            
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]


async def main():
    """Запуск MCP сервера."""
    logger.info("Starting WebSocket Stream MCP Server...")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="websocket-stream",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await cleanup_stream()


if __name__ == "__main__":
    asyncio.run(main())
