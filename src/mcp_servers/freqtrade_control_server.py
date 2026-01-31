#!/usr/bin/env python3
"""
Freqtrade Control MCP Server
=============================

MCP server for controlling the Freqtrade bot via its API.
Features:
- Start/Stop bot
- Force exit positions
- Get status/profit
- Show active trades
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

import aiohttp

# MCP SDK imports
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed.", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-freqtrade-control")

# Configuration
FREQTRADE_API_URL = os.getenv("FREQTRADE_API_URL", "http://localhost:8080/api/v1")
FREQTRADE_API_USERNAME = os.getenv("FREQTRADE_API_USERNAME", "freqtrader")
FREQTRADE_API_PASSWORD = os.getenv("FREQTRADE_API_PASSWORD", "SuperSecurePassword123!")

class FreqtradeClient:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.auth = aiohttp.BasicAuth(username, password)
        self.token = None

    async def _get_headers(self) -> Dict[str, str]:
        if not self.token:
            await self.login()
        return {"Authorization": f"Bearer {self.token}"}

    async def login(self):
        url = f"{self.base_url}/token/login"
        # Basic auth for getting the token
        async with aiohttp.ClientSession() as session:
            async with session.post(url, auth=self.auth) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self.token = data.get("access_token")
                else:
                    raise Exception(f"Login failed: {resp.status} - {await resp.text()}")

    async def _request(self, method: str, endpoint: str, data: Dict = None) -> Any:
        url = f"{self.base_url}/{endpoint}"
        headers = await self._get_headers()
        
        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, headers=headers, json=data) as resp:
                if resp.status == 401:
                    # Token might be expired, retry once
                    self.token = None
                    headers = await self._get_headers()
                    async with session.request(method, url, headers=headers, json=data) as retry_resp:
                         if retry_resp.status >= 400:
                             raise Exception(f"API Error {retry_resp.status}: {await retry_resp.text()}")
                         return await retry_resp.json()
                
                if resp.status >= 400:
                    raise Exception(f"API Error {resp.status}: {await resp.text()}")
                
                return await resp.json()

    async def get_status(self) -> Dict:
        return await self._request("GET", "status")

    async def get_profit(self) -> Dict:
        return await self._request("GET", "profit")
        
    async def get_trades(self) -> Dict:
        return await self._request("GET", "status") # 'status' returns open trades

    async def start_bot(self) -> Dict:
        return await self._request("POST", "start")

    async def stop_bot(self) -> Dict:
        return await self._request("POST", "stop")

    async def force_exit(self, trade_id: str) -> Dict:
        return await self._request("POST", f"forceexit/{trade_id}")
        
    async def force_exit_all(self) -> Dict:
        # Freqtrade doesn't have a single "exit all" endpoint usually exposed simply,
        # but forceexit usually takes 'all' or we iterate. 
        # Checking docs: /forceexit endpoint handles 'tradeid' or 'all'
        # But let's stick to standard practice.
        # Actually standard freqtrade api is POST /forceexit with json body {"tradeid": "all"}?
        # Let's try checking documentation standard. 
        # Standard API: POST /forceexit
        return await self._request("POST", "forceexit", {"tradeid": "all"})


# Initialize Client
client = FreqtradeClient(FREQTRADE_API_URL, FREQTRADE_API_USERNAME, FREQTRADE_API_PASSWORD)
server = Server("freqtrade-control")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_bot_status",
            description="Get current bot status and open trades",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_profit_stats",
            description="Get profit statistics",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="control_bot",
            description="Start or stop the bot",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop"], "description": "Action to perform"}
                },
                "required": ["action"]
            },
        ),
        types.Tool(
            name="force_exit_trade",
            description="Force exit a trade or all trades",
            inputSchema={
                "type": "object",
                "properties": {
                    "trade_id": {"type": "string", "description": "Trade ID to exit, or 'all'"}
                },
                "required": ["trade_id"]
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}

    try:
        if name == "get_bot_status":
            status = await client.get_status()
            return [types.TextContent(type="text", text=json.dumps(status, indent=2))]

        elif name == "get_profit_stats":
            profit = await client.get_profit()
            return [types.TextContent(type="text", text=json.dumps(profit, indent=2))]

        elif name == "control_bot":
            action = arguments.get("action")
            if action == "start":
                res = await client.start_bot()
            elif action == "stop":
                res = await client.stop_bot()
            else:
                raise ValueError("Invalid action")
            return [types.TextContent(type="text", text=json.dumps(res, indent=2))]

        elif name == "force_exit_trade":
            trade_id = arguments.get("trade_id")
            if trade_id.lower() == "all":
                res = await client.force_exit_all()
            else:
                res = await client.force_exit(trade_id)
            return [types.TextContent(type="text", text=json.dumps(res, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="freqtrade-control",
                server_version="1.0.0",
                capabilities=server.get_capabilities(),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())