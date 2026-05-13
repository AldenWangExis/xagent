"""List MCP tools from inside sandbox using a stable Python entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

# WARNING: This file runs as a standalone script in the sandbox, not a module.
# Absolute imports only — relative imports are unavailable (no package context).
from xagent.core.tools.adapters.vibe.sandboxed_tool.runner_utils import (
    ensure_user_bin_in_path,
)
from xagent.core.tools.core.mcp.sessions import Connection
from xagent.core.tools.core.mcp.tools import load_mcp_tools


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for sandboxed MCP tool listing."""
    parser = argparse.ArgumentParser(description="List MCP tools in sandbox")
    parser.add_argument("--connection-b64", required=True)
    parser.add_argument("--result-file", required=True)
    return parser.parse_args()


def _load_connection(connection_b64: str) -> Connection:
    """Decode a base64-encoded connection config."""
    connection_json = base64.b64decode(connection_b64).decode("utf-8")
    return cast(Connection, json.loads(connection_json))


async def _list_tools(connection: Connection) -> list[dict[str, Any]]:
    """List and serialize MCP tools for JSON output."""
    tools = await load_mcp_tools(None, connection=connection)
    return [cast(dict[str, Any], tool.model_dump(mode="json")) for tool in tools]


def main() -> None:
    """CLI entrypoint for sandboxed MCP tool listing."""
    ensure_user_bin_in_path()
    try:
        parsed = _parse_args()
        connection = _load_connection(parsed.connection_b64)
    except Exception as e:
        print(f"Sandbox mcp config error: {e}")
        raise

    result = asyncio.run(_list_tools(connection))
    Path(parsed.result_file).write_text(
        json.dumps(result, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
