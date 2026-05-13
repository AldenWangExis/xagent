"""Tests for mcp_runner.py helper functions and main()."""

import base64
import json
from unittest.mock import AsyncMock, patch

from xagent.core.tools.adapters.vibe.sandboxed_tool.mcp_runner import (
    _load_connection,
    main,
)


class _FakeTool:
    """Minimal fake MCP tool for serialization tests."""

    def __init__(self, name: str) -> None:
        self.name = name

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {
            "name": self.name,
            "description": f"Tool {self.name}",
            "inputSchema": {"type": "object", "properties": {}},
        }


class TestLoadConnection:
    """Tests for _load_connection()."""

    def test_roundtrip(self):
        connection = {"transport": "stdio", "command": "npx", "args": ["demo"]}
        connection_b64 = base64.b64encode(json.dumps(connection).encode()).decode()
        assert _load_connection(connection_b64) == connection


class TestMain:
    """Tests for mcp_runner.main()."""

    def test_happy_path(self, tmp_path):
        result_file = str(tmp_path / "result.json")
        connection_b64 = base64.b64encode(
            json.dumps(
                {"transport": "stdio", "command": "npx", "args": ["demo"]}
            ).encode()
        ).decode()
        argv = [
            "--connection-b64",
            connection_b64,
            "--result-file",
            result_file,
        ]

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.mcp_runner.load_mcp_tools",
                new=AsyncMock(return_value=[_FakeTool("echo")]),
            ),
            patch("sys.argv", ["mcp_runner"] + argv),
        ):
            main()

        result = json.loads((tmp_path / "result.json").read_text())
        assert result == [
            {
                "name": "echo",
                "description": "Tool echo",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
