#!/usr/bin/env python3
"""
Documentation Assistant MCP Server
==================================

MCP server for indexing and searching the project documentation.
Helps answer questions like "how to add a strategy?" by finding relevant docs.
"""

import asyncio
import json
import logging
import os
import sys
import re

# MCP SDK
try:
    from mcp import types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server
except ImportError:
    print("Error: MCP SDK not installed. Please install 'mcp'.", file=sys.stderr)
    sys.exit(1)

# Project imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging
logging.basicConfig(level=logging.ERROR, stream=sys.stderr)
logger = logging.getLogger("mcp-docs-assistant")

DOCS_DIR = os.path.join(PROJECT_ROOT, "..", "docs")
DOCS_DIR = os.path.abspath(DOCS_DIR)

server = Server("docs-assistant")

def get_markdown_files(root_dir):
    """Recursively find all markdown files."""
    md_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files

def search_files(query, limit=5):
    """Simple keyword search across markdown files."""
    results = []
    query_terms = query.lower().split()
    
    files = get_markdown_files(DOCS_DIR)
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                content_lower = content.lower()
                
                # Calculate score based on term frequency
                score = 0
                matches = []
                
                for term in query_terms:
                    count = content_lower.count(term)
                    if count > 0:
                        score += count
                        # Find context
                        start_idx = content_lower.find(term)
                        start = max(0, start_idx - 50)
                        end = min(len(content), start_idx + 100)
                        matches.append(content[start:end].replace('\n', ' ').strip() + "...")
                
                if score > 0:
                    rel_path = os.path.relpath(file_path, DOCS_DIR)
                    results.append({
                        "path": rel_path,
                        "score": score,
                        "matches": matches[:3], # Top 3 snippets
                        "full_path": file_path
                    })
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_docs",
            description="Search documentation for specific topics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'risk limits', 'add strategy')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max number of results to return (default 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
        ),
        types.Tool(
            name="read_doc_file",
            description="Read the full content of a documentation file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the doc file (from search results)"
                    }
                },
                "required": ["path"]
            },
        ),
        types.Tool(
            name="list_all_docs",
            description="List all available documentation files.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
    if arguments is None:
        arguments = {}

    try:
        if name == "search_docs":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            results = search_files(query, limit)
            
            output = [f"Search results for '{query}':\n"]
            for r in results:
                output.append(f"File: {r['path']} (Score: {r['score']})")
                for m in r['matches']:
                    output.append(f"  - ...{m}")
                output.append("")
                
            if not results:
                output.append("No matches found.")
                
            return [types.TextContent(type="text", text="\n".join(output))]

        elif name == "read_doc_file":
            path = arguments.get("path", "")
            # Sanitize path to prevent traversal
            if ".." in path or path.startswith("/"):
                 return [types.TextContent(type="text", text="Error: Invalid path")]
                 
            full_path = os.path.join(DOCS_DIR, path)
            if not os.path.exists(full_path):
                return [types.TextContent(type="text", text=f"Error: File not found: {path}")]
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return [types.TextContent(type="text", text=content)]

        elif name == "list_all_docs":
            files = get_markdown_files(DOCS_DIR)
            rel_files = [os.path.relpath(f, DOCS_DIR) for f in files]
            rel_files.sort()
            return [types.TextContent(type="text", text="\n".join(rel_files))]

        else:
            return [types.TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"success": False, "error": str(e)}))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="docs-assistant",
                server_version="1.0.0",
                capabilities=server.get_capabilities(),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())