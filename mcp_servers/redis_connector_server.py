#!/usr/bin/env python3
"""
Redis Connector MCP Server
===========================

MCP сервер для работы с Redis cache и PubSub.

Предоставляет инструменты:
- set_value: Записать значение в кэш
- get_value: Получить значение из кэша
- delete_value: Удалить значение
- publish: Опубликовать сообщение в PubSub канале

Author: Stoic Citadel Team
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Optional

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
from src.ml.redis_client import RedisClient

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global redis client
redis_client: Optional[RedisClient] = None


async def initialize_redis():
    """Инициализация redis."""
    global redis_client
    if redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", 6379))
        db = int(os.getenv("REDIS_DB", 0))
        
        redis_client = RedisClient(host=host, port=port, db=db)
        await redis_client.connect()
        logger.info(f"Redis client initialized ({host}:{port})")


async def cleanup_redis():
    """Закрытие соединений."""
    global redis_client
    if redis_client is not None:
        await redis_client.disconnect()
        redis_client = None
        logger.info("Redis client closed")


# Create MCP server
server = Server("redis-connector")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="set_value",
            description="Записать значение в кэш",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "ttl": {"type": "integer", "description": "Время жизни в секундах (опционально)"},
                },
                "required": ["key", "value"],
            },
        ),
        types.Tool(
            name="get_value",
            description="Получить значение из кэша",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        ),
        types.Tool(
            name="delete_value",
            description="Удалить значение из кэша",
            inputSchema={
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                },
                "required": ["key"],
            },
        ),
        types.Tool(
            name="publish_message",
            description="Опубликовать сообщение в канал",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["channel", "message"],
            },
        ),
        types.Tool(
            name="check_health",
            description="Проверить статус Redis",
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
    
    await initialize_redis()
    
    try:
        if name == "set_value":
            key = arguments.get("key")
            value = arguments.get("value")
            ttl = arguments.get("ttl")
            
            # RedisClient expected to have set method
            # If not, we might need to access the underlying client
            success = await redis_client.client.set(key, value, ex=ttl)
            
            return [types.TextContent(type="text", text=json.dumps({"success": bool(success), "key": key}))]
        
        elif name == "get_value":
            key = arguments.get("key")
            value = await redis_client.client.get(key)
            
            if value:
                value = value.decode("utf-8") if isinstance(value, bytes) else value
            
            return [types.TextContent(type="text", text=json.dumps({"success": True, "key": key, "value": value}))]
        
        elif name == "delete_value":
            key = arguments.get("key")
            count = await redis_client.client.delete(key)
            
            return [types.TextContent(type="text", text=json.dumps({"success": True, "key": key, "deleted_count": count}))]
        
        elif name == "publish_message":
            channel = arguments.get("channel")
            message = arguments.get("message")
            
            receivers = await redis_client.client.publish(channel, message)
            
            return [types.TextContent(type="text", text=json.dumps({"success": True, "channel": channel, "receivers": receivers}))]
        
        elif name == "check_health":
            try:
                ping = await redis_client.client.ping()
                return [types.TextContent(type="text", text=json.dumps({"success": True, "status": "healthy", "ping": ping}))]
            except Exception as e:
                return [types.TextContent(type="text", text=json.dumps({"success": False, "status": "unhealthy", "error": str(e)}))]
        
        else:
            return [types.TextContent(type="text", text=json.dumps({"success": False, "error": f"Unknown tool: {name}"}))]
            
    except Exception as e:
        logger.error(f"Error in {name}: {e}")
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]


async def main():
    """Запуск MCP сервера."""
    logger.info("Starting Redis Connector MCP Server...")
    
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="redis-connector",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    finally:
        await cleanup_redis()


if __name__ == "__main__":
    asyncio.run(main())
