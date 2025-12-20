#!/usr/bin/env python3
"""
Database Connector MCP Server
==============================

MCP сервер для работы с PostgreSQL базой данных.

Предоставляет инструменты:
- execute_query: Выполнение SQL запроса
- get_pool_status: Статус connection pool
- health_check: Проверка подключения

Author: Stoic Citadel Team
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any

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
from config.database import get_db_config, get_pool_status

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database config
db_config = None


def initialize_db():
    """Инициализация database."""
    global db_config
    if db_config is None:
        db_config = get_db_config(pool_size=10, max_overflow=20)
        logger.info("Database connection initialized")


# Create MCP server
server = Server("database-connector")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов."""
    return [
        types.Tool(
            name="execute_query",
            description="Выполнить SQL запрос (SELECT, INSERT, UPDATE, DELETE)",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL запрос",
                    },
                    "params": {
                        "type": "object",
                        "description": "Параметры запроса (опционально)",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_pool_status",
            description="Получить статус connection pool",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        types.Tool(
            name="health_check",
            description="Проверить подключение к базе данных",
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
    
    initialize_db()
    
    try:
        if name == "execute_query":
            query = arguments.get("query", "")
            params = arguments.get("params", {})
            
            with db_config.get_session() as session:
                result_proxy = session.execute(query, params)
                
                # Проверяем тип запроса
                if query.strip().upper().startswith("SELECT"):
                    rows = result_proxy.fetchall()
                    columns = result_proxy.keys()
                    
                    result = {
                        "success": True,
                        "rows": [dict(zip(columns, row)) for row in rows],
                        "row_count": len(rows),
                        "columns": list(columns),
                    }
                else:
                    # INSERT, UPDATE, DELETE
                    result = {
                        "success": True,
                        "affected_rows": result_proxy.rowcount,
                        "message": "Query executed successfully",
                    }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "get_pool_status":
            status = get_pool_status()
            
            result = {
                "success": True,
                "pool_status": status,
            }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "health_check":
            try:
                with db_config.get_session() as session:
                    session.execute("SELECT 1")
                
                result = {
                    "success": True,
                    "status": "healthy",
                    "message": "Database connection is working",
                }
            except Exception as e:
                result = {
                    "success": False,
                    "status": "unhealthy",
                    "error": str(e),
                }
            
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
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
    logger.info("Starting Database Connector MCP Server...")
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="database-connector",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
