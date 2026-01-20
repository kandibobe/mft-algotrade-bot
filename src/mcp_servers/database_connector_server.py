#!/usr/bin/env python3
"""
Database Connector MCP Server
==============================

MCP сервер для работы с PostgreSQL базой данных.
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

from config.database import get_db_config

# Logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-database-connector")


def is_port_open(host, port):
    """Быстрая проверка порта, чтобы не вешать систему."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except:
        return False


# Global database config
db_config = None


def initialize_db():
    """Инициализация database с быстрой проверкой доступности."""
    global db_config

    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = int(os.getenv("POSTGRES_PORT", 5431))

    if not is_port_open(host, port):
        raise ConnectionError(f"Database at {host}:{port} is not reachable. Is Docker running?")

    if db_config is None:
        db_config = get_db_config()


# Create MCP server
server = Server("database-connector")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="execute_query",
            description="SQL query execution",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_tables",
            description="Show all tables",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    try:
        initialize_db()
    except Exception as e:
        return [
            types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))
        ]

    try:
        if name == "execute_query":
            query = arguments.get("query", "")
            with db_config.get_session() as session:
                # Ограничиваем время выполнения запроса 2 секундами
                session.execute("SET statement_timeout = 2000")
                result_proxy = session.execute(query)
                if query.strip().upper().startswith("SELECT"):
                    rows = result_proxy.fetchall()
                    result = {
                        "success": True,
                        "rows": [dict(zip(result_proxy.keys(), r, strict=False)) for r in rows],
                    }
                else:
                    result = {"success": True, "affected": result_proxy.rowcount}
            return [types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

        elif name == "list_tables":
            query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            with db_config.get_session() as session:
                rows = session.execute(query).fetchall()
                tables = [row[0] for row in rows]
            return [
                types.TextContent(type="text", text=json.dumps({"success": True, "tables": tables}))
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
                server_name="database-connector",
                server_version="1.2.0",
                capabilities=server.get_capabilities(),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
