"""
End-to-End Subprocess Stdio Integration Test for Perron MCP Server.
Spawns the perron-mcp server process, communicates strictly via JSON-RPC 2.0 over stdio,
verifies protocol envelopes and tools, and outputs docs/mcp_transcript.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mcp_stdio_lifecycle_and_tools(tmp_path: Path):
    """Verify JSON-RPC 2.0 stdio communication, tool dispatch, and clean termination."""
    # Create minimal fixture repository with code
    fixture_dir = tmp_path / "repo"
    fixture_dir.mkdir()
    (fixture_dir / "calculator.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    '''Add two integers.'''\n"
        "    return a + b\n\n"
        "def multiply(a: int, b: int) -> int:\n"
        "    '''Multiply two integers.'''\n"
        "    return a * b\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, "-m", "perron.mcp_server", "--repo", str(fixture_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    transcript = []

    def send_and_recv(msg: dict) -> dict:
        wire = json.dumps(msg) + "\n"
        proc.stdin.write(wire)
        proc.stdin.flush()
        line = proc.stdout.readline()
        resp = json.loads(line)
        transcript.append({"direction": "sent", "payload": msg})
        transcript.append({"direction": "received", "payload": resp})
        return resp

    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "test-client", "version": "1.0.0"},
            },
        }
        init_resp = send_and_recv(init_req)
        assert init_resp.get("id") == 1
        assert "result" in init_resp
        server_info = init_resp["result"].get("serverInfo", {})
        assert server_info.get("name") == "perron-mcp"
        assert server_info.get("version") == "0.3.0"
        assert "capabilities" in init_resp["result"]

        # 2. Initialized Notification (no response expected)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        proc.stdin.write(json.dumps(notif) + "\n")
        proc.stdin.flush()
        transcript.append({"direction": "sent", "payload": notif})

        # 3. Tools List
        tools_req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        tools_resp = send_and_recv(tools_req)
        assert tools_resp.get("id") == 2
        tools = tools_resp.get("result", {}).get("tools", [])
        tool_names = {t["name"] for t in tools}
        assert "retrieve_context" in tool_names
        assert "inspect_symbol_breadcrumbs" in tool_names
        assert "build_code_graph" in tool_names

        # 4. Tool Call: retrieve_context
        call_req = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "retrieve_context",
                "arguments": {
                    "query": "add two numbers",
                    "token_budget": 1024,
                },
            },
        }
        call_resp = send_and_recv(call_req)
        assert call_resp.get("id") == 3
        content = call_resp.get("result", {}).get("content", [])
        assert len(content) > 0
        assert content[0].get("type") == "text"
        assert call_resp.get("result", {}).get("isError") is False

    finally:
        proc.stdin.close()
        proc.wait(timeout=5)

    # Save transcript
    transcript_file = REPO_ROOT / "docs" / "mcp_transcript.json"
    transcript_file.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2)
    assert transcript_file.exists()
