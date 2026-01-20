#!/usr/bin/env python3
"""
Redis Connector MCP Server
===========================

MCP сервер для работы с Redis.
Оптимизирован: быстрые таймауты для предотвращения фризов системы.
"""

import asyncio
import json
import logging
import os
import socket
import sys

# MCP SDK
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed.", file=sys.stderr)
    sys.exit(1)

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.ml.redis_client import RedisClient
except ImportError:
    RedisClient = None

# Logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)


def is_port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except:
        return False


# Global redis client
redis_client: RedisClient | None = None


async def initialize_redis():
    global redis_client
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", 6379))

    if not is_port_open(host, port):
        raise ConnectionError(f"Redis at {host}:{port} is not reachable. Is Docker running?")

    if redis_client is None:
        if RedisClient is None:
            raise ImportError("RedisClient module not found")
        redis_client = RedisClient(host=host, port=port)
        await redis_client.connect()


# Create MCP server
server = Server("redis-connector")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_value",
            description="Get value from Redis",
            inputSchema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        ),
        types.Tool(
            name="check_health",
            description="Check Redis health",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    try:
        await initialize_redis()
    except Exception as e:
        return [
            types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))
        ]

    try:
        if name == "get_value":
            key = arguments.get("key")
            value = await redis_client.client.get(key)
            if value:
                value = value.decode("utf-8") if isinstance(value, bytes) else value
            return [
                types.TextContent(
                    type="text", text=json.dumps({"success": True, "key": key, "value": value})
                )
            ]
        elif name == "check_health":
            ping = await redis_client.client.ping()
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"success": True, "status": "healthy", "ping": ping}),
                )
            ]
        else:
            return [
                types.TextContent(
                    type="text", text=json.dumps({"success": False, "error": "Unknown tool"})
                )
            ]
    except Exception as e:
        return [
            types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))
        ]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="redis-connector",
                server_version="1.2.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
