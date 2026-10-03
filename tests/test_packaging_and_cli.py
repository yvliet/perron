"""
Tests for packaging manifests, version synchronization, and CLI entrypoints.
"""

import subprocess
import sys
import pytest
import perron
from perron import __version__


def test_version_string():
    assert __version__ == "0.3.0"
    assert perron.__version__ == "0.3.0"


def test_cli_version():
    res = subprocess.run(
        [sys.executable, "-m", "perron.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "perron-core 0.3.0" in res.stdout


def test_cli_help():
    res = subprocess.run(
        [sys.executable, "-m", "perron.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "High-Throughput Spectral Code Graph Retrieval" in res.stdout
    assert "index" in res.stdout
    assert "query" in res.stdout


def test_mcp_version():
    res = subprocess.run(
        [sys.executable, "-m", "perron.mcp_server", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "perron-mcp 0.3.0" in res.stdout


def test_mcp_help():
    res = subprocess.run(
        [sys.executable, "-m", "perron.mcp_server", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Model Context Protocol (MCP) Server" in res.stdout
