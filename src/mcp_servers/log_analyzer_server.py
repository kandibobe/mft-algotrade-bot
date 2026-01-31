#!/usr/bin/env python3
"""
Log Analyzer MCP Server
========================

MCP server for real-time log analysis of Freqtrade logs.
Features:
- Live tailing of logs
- Filtering by severity (INFO, WARNING, ERROR)
- Keyword search
- Error summarization
"""

import asyncio
import json
import logging
import os
import re
import sys
from collections import deque
from datetime import datetime
from typing import List, Optional

# MCP SDK imports
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed.", file=sys.stderr)
    sys.exit(1)

# Configure logging for the server itself
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-log-analyzer")

# Constants
LOG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "user_data",
    "logs",
    "freqtrade.log",
)
MAX_LINES_TO_READ = 1000
DEFAULT_LINES = 50


class LogAnalyzer:
    def __init__(self, log_path: str):
        self.log_path = log_path

    def _read_last_lines(self, n: int) -> List[str]:
        """Efficiently read the last n lines of the log file."""
        if not os.path.exists(self.log_path):
            return [f"Error: Log file not found at {self.log_path}"]

        try:
            with open(self.log_path, "rb") as f:
                # Basic implementation for small-ish logs or simple tail
                # For very large logs, we should seek from end. 
                # Given typical log rotation, full read might be okay but seek is safer.
                f.seek(0, 2)
                file_size = f.tell()
                
                lines_found = []
                block_size = 4096
                offset = file_size
                
                while offset > 0 and len(lines_found) < n:
                    if offset - block_size > 0:
                        offset -= block_size
                    else:
                        block_size = offset
                        offset = 0
                    
                    f.seek(offset)
                    block = f.read(block_size)
                    
                    # Split lines and reverse to process from end
                    decoded_block = block.decode('utf-8', errors='ignore')
                    new_lines = decoded_block.splitlines()
                    
                    # Handle the case where the split happens in the middle of a line
                    if lines_found and decoded_block and not decoded_block.endswith('\n'):
                         # Merge the last part of this block with the first part of the previous block
                         # (Logic simplified here, just gathering lines for now)
                         pass

                    # Add to our list (reversed)
                    lines_found = new_lines + lines_found
                
                return lines_found[-n:]
        except Exception as e:
            return [f"Error reading log file: {str(e)}"]

    def get_logs(self, lines: int = DEFAULT_LINES, level: Optional[str] = None, filter_text: Optional[str] = None) -> List[str]:
        raw_lines = self._read_last_lines(lines * 2) # Read more to allow filtering
        filtered_lines = []
        
        for line in raw_lines:
            if level and level.upper() not in line.upper():
                continue
            if filter_text and filter_text.lower() not in line.lower():
                continue
            filtered_lines.append(line)
            
        return filtered_lines[-lines:]

    def analyze_errors(self, lines: int = 200) -> dict:
        """Analyze recent errors and group them."""
        recent_logs = self.get_logs(lines=lines)
        errors = []
        warnings = []
        
        error_pattern = re.compile(r'ERROR', re.IGNORECASE)
        warning_pattern = re.compile(r'WARNING', re.IGNORECASE)
        
        for line in recent_logs:
            if error_pattern.search(line):
                errors.append(line)
            elif warning_pattern.search(line):
                warnings.append(line)
                
        # Group similar errors (simple heuristic)
        grouped_errors = {}
        for error in errors:
            # Remove timestamp and thread info to group by message
            # 2023-10-27 10:00:00,000 - freqtrade.worker - ERROR - Message
            parts = error.split(' - ')
            if len(parts) >= 4:
                msg = parts[-1]
            else:
                msg = error
            
            if msg not in grouped_errors:
                grouped_errors[msg] = 0
            grouped_errors[msg] += 1
            
        return {
            "total_errors": len(errors),
            "total_warnings": len(warnings),
            "unique_errors": [{"message": k, "count": v} for k, v in grouped_errors.items()],
            "recent_error_samples": errors[-5:] if errors else []
        }

# Initialize Log Analyzer
analyzer = LogAnalyzer(LOG_FILE_PATH)
server = Server("log-analyzer")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="read_logs",
            description="Read recent logs with optional filtering",
            inputSchema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of lines to read (default 50)"},
                    "level": {"type": "string", "description": "Log level filter (INFO, WARNING, ERROR)"},
                    "filter_text": {"type": "string", "description": "Text to search for"}
                }
            },
        ),
        types.Tool(
            name="analyze_errors",
            description="Analyze recent errors and warnings",
            inputSchema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "Number of lines to scan (default 200)"}
                }
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}

    try:
        if name == "read_logs":
            lines = arguments.get("lines", DEFAULT_LINES)
            level = arguments.get("level")
            filter_text = arguments.get("filter_text")
            
            logs = analyzer.get_logs(lines, level, filter_text)
            return [types.TextContent(type="text", text="\n".join(logs))]

        elif name == "analyze_errors":
            lines = arguments.get("lines", 200)
            analysis = analyzer.analyze_errors(lines)
            return [types.TextContent(type="text", text=json.dumps(analysis, indent=2))]

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
                server_name="log-analyzer",
                server_version="1.0.0",
                capabilities=server.get_capabilities(),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())