#!/usr/bin/env python3
"""
Freqtrade Management MCP Server
===============================

MCP server to manage Freqtrade bot via its API.
Provides tools to start/stop the bot, check status, and force exit trades.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# MCP SDK
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed.", file=sys.stderr)
    sys.exit(1)

import httpx

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-freqtrade-manager")

# Config
API_URL = os.getenv("FREQTRADE_API_URL", "http://127.0.0.1:8080/api/v1")
CONFIG_PATH = os.getenv("FREQTRADE_CONFIG_PATH", "user_data/config/config_production.json")

server = Server("freqtrade-manager")

class FreqtradeClient:
    def __init__(self):
        self.token = None
        self.username = None
        self.password = None
        self._load_credentials()

    def _load_credentials(self):
        # Try to load from config file
        try:
            path = Path(CONFIG_PATH)
            if not path.is_absolute():
                path = Path(PROJECT_ROOT).parent / path
            
            if path.exists():
                with open(path, "r") as f:
                    config = json.load(f)
                    api_config = config.get("api_server", {})
                    self.username = api_config.get("username", "stoic_admin")
                    self.password = api_config.get("password", "")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

        # Override with env vars if present
        if os.getenv("FREQTRADE_API_USERNAME"):
            self.username = os.getenv("FREQTRADE_API_USERNAME")
        if os.getenv("FREQTRADE_API_PASSWORD"):
            self.password = os.getenv("FREQTRADE_API_PASSWORD")

    async def login(self):
        if not self.username or not self.password:
             logger.warning("No credentials found. Assuming no auth or already authenticated.")
             return

        async with httpx.AsyncClient() as client:
            try:
                # Freqtrade API login typically uses form data
                response = await client.post(
                    f"{API_URL}/token/login",
                    data={"username": self.username, "password": self.password},
                    timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    self.token = data.get("access_token")
                else:
                    logger.error(f"Login failed: {response.text}")
            except Exception as e:
                logger.error(f"Login error: {e}")

    async def _request(self, method, endpoint, **kwargs):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method, 
                    f"{API_URL}{endpoint}", 
                    headers=headers, 
                    timeout=10.0,
                    **kwargs
                )
                
                if response.status_code == 401:
                    # Token might be expired, try login again
                    await self.login()
                    if self.token:
                         headers["Authorization"] = f"Bearer {self.token}"
                         response = await client.request(
                            method, 
                            f"{API_URL}{endpoint}", 
                            headers=headers, 
                            timeout=10.0,
                            **kwargs
                        )
                
                return response
            except Exception as e:
                logger.error(f"Request error: {e}")
                raise

    async def get_status(self):
        # Common endpoint for status is /status (bot state) or /ping
        # Using /status which returns running state
        resp = await self._request("GET", "/status")
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def start(self):
        resp = await self._request("POST", "/start")
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def stop(self):
        resp = await self._request("POST", "/stop")
        return resp.json() if resp.status_code == 200 else {"error": resp.text}
        
    async def force_exit(self, trade_id="all"):
        # Freqtrade uses /forcesell usually
        payload = {}
        if trade_id != "all":
            payload["tradeid"] = trade_id
        
        resp = await self._request("POST", "/forcesell", json=payload)
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

client = FreqtradeClient()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="freqtrade_status",
            description="Get the current status of the Freqtrade bot.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="freqtrade_start",
            description="Start the trading bot.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="freqtrade_stop",
            description="Stop the trading bot.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="freqtrade_force_exit",
            description="Force exit (sell) open trades.",
            inputSchema={
                "type": "object",
                "properties": {
                    "trade_id": {
                        "type": "string",
                        "description": "ID of the trade to exit, or 'all' to exit all.",
                        "default": "all"
                    }
                }
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}
        
    try:
        if name == "freqtrade_status":
            result = await client.get_status()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "freqtrade_start":
            result = await client.start()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "freqtrade_stop":
            result = await client.stop()
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "freqtrade_force_exit":
            trade_id = arguments.get("trade_id", "all")
            result = await client.force_exit(trade_id)
            return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
            
        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

async def main():
    # Initial login
    await client.login()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="freqtrade-manager",
                server_version="1.0.0",
                capabilities=server.get_capabilities(),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())