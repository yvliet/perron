"""
Unit and integration tests for perron/mcp_server.py (Model Context Protocol).
"""

import json
import pytest
from pathlib import Path
from perron.mcp_server import PerronMCPServer, TOOLS
from perron import __version__


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a minimal synthetic repo with function and class call relations."""
    repo = tmp_path / "toy_repo"
    repo.mkdir()
    
    pkg = repo / "mypkg"
    pkg.mkdir()
    
    init_file = pkg / "__init__.py"
    init_file.write_text("# init\n", encoding="utf-8")
    
    models_file = pkg / "models.py"
    models_file.write_text(
        "class BaseModel:\n"
        "    def save(self):\n"
        "        \"\"\"Save base model.\"\"\"\n"
        "        return True\n"
        "\n"
        "def helper_func():\n"
        "    \"\"\"Helper calculation.\"\"\"\n"
        "    m = BaseModel()\n"
        "    return m.save()\n",
        encoding="utf-8",
    )
    return repo


def test_mcp_initialize_and_tools_list():
    server = PerronMCPServer()
    try:
        # 1. Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05"},
        }
        init_resp = server.handle_request(init_req)
        assert init_resp["id"] == 1
        assert init_resp["result"]["serverInfo"]["name"] == "perron-mcp"
        assert init_resp["result"]["serverInfo"]["version"] == __version__

        # 2. Ping
        ping_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        assert ping_resp["id"] == 2
        assert ping_resp["result"] == {}

        # 3. Notification
        notif_resp = server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert notif_resp is None

        # 4. Tools list
        tools_resp = server.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
        assert tools_resp["id"] == 3
        tools = tools_resp["result"]["tools"]
        tool_names = {t["name"] for t in tools}
        assert "retrieve_context" in tool_names
        assert "inspect_symbol_breadcrumbs" in tool_names
        assert "build_code_graph" in tool_names
    finally:
        server.close()


def test_mcp_tool_execution(sample_repo: Path, tmp_path: Path):
    server = PerronMCPServer()
    cache_dir = tmp_path / "test_cache"
    try:
        # 1. Build Code Graph
        build_req = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "build_code_graph",
                "arguments": {
                    "repo_path": str(sample_repo),
                    "output_dir": str(cache_dir),
                    "force": True,
                },
            },
        }
        build_resp = server.handle_request(build_req)
        assert build_resp["id"] == 10
        assert build_resp["result"]["isError"] is False
        build_text = build_resp["result"]["content"][0]["text"]
        assert "Successfully compiled code graph" in build_text
        assert "Symbols indexed:" in build_text

        # 2. Retrieve Context
        query_req = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "retrieve_context",
                "arguments": {
                    "repo_path": str(sample_repo),
                    "cache_dir": str(cache_dir),
                    "query": "save base model helper",
                    "token_budget": 2048,
                },
            },
        }
        query_resp = server.handle_request(query_req)
        assert query_resp["id"] == 11
        assert query_resp["result"]["isError"] is False
        query_text = query_resp["result"]["content"][0]["text"]
        assert "Perron Code Graph Context" in query_text

        # 3. Inspect Symbol Breadcrumbs
        crumb_req = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "inspect_symbol_breadcrumbs",
                "arguments": {
                    "repo_path": str(sample_repo),
                    "cache_dir": str(cache_dir),
                    "symbol_identifier": "save",
                },
            },
        }
        crumb_resp = server.handle_request(crumb_req)
        assert crumb_resp["id"] == 12
        assert crumb_resp["result"]["isError"] is False
        crumb_text = crumb_resp["result"]["content"][0]["text"]
        assert "Symbol Breadcrumb:" in crumb_text
        assert "save" in crumb_text
        assert "Global PageRank" in crumb_text
    finally:
        server.close()


def test_mcp_error_handling(sample_repo: Path):
    server = PerronMCPServer()
    try:
        # Unknown tool
        req1 = {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "tools/call",
            "params": {"name": "non_existent_tool", "arguments": {}},
        }
        resp1 = server.handle_request(req1)
        assert resp1["id"] == 20
        assert "error" in resp1
        assert resp1["error"]["code"] == -32601

        # Missing required args
        req2 = {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {"name": "retrieve_context", "arguments": {}},
        }
        resp2 = server.handle_request(req2)
        assert resp2["id"] == 21
        assert resp2["result"]["isError"] is True
        assert "Error executing 'retrieve_context'" in resp2["result"]["content"][0]["text"]
    finally:
        server.close()
