"""
Model Context Protocol (MCP) Server for Perron Code Graph Retrieval.

Implements standard MCP JSON-RPC 2.0 protocol over stdio, exposing:
- retrieve_context: Context-budgeted code graph subgraph retrieval
- inspect_symbol_breadcrumbs: Topological neighborhood and caller/callee inspection
- build_code_graph: Offline compilation of zero-copy mmap CSR matrices

Compatible with Claude Code, Cursor, and local agents (Gemma 4, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from perron import __version__
from perron.retriever import CodeGraph, PerronRetriever

# Direct all diagnostics to stderr so stdout remains a pristine JSON-RPC stream
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[perron-mcp] %(levelname)s: %(message)s",
)
logger = logging.getLogger("perron-mcp")


TOOLS = [
    {
        "name": "retrieve_context",
        "description": (
            "Retrieve context-bounded AST code graph subgraphs for a developer query "
            "using spectral specificity diffusion with query-level hub adaptivity."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository root directory.",
                },
                "query": {
                    "type": "string",
                    "description": "Bug description, issue text, or target symbol identifiers.",
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Maximum token budget ceiling for packed context (default: 4096).",
                    "default": 4096,
                },
                "gamma": {
                    "type": "number",
                    "description": "Hub specificity discounting exponent (default: 0.70).",
                    "default": 0.70,
                },
                "beta": {
                    "type": "number",
                    "description": "Teleport restart damping factor (default: 0.85).",
                    "default": 0.85,
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Optional custom cache directory (defaults to <repo_path>/.perron_cache).",
                },
            },
            "required": ["repo_path", "query"],
        },
    },
    {
        "name": "inspect_symbol_breadcrumbs",
        "description": (
            "Inspect the topological graph neighborhood, callers, callees, and enclosing "
            "breadcrumb hierarchy for a specific symbol or symbol ID."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository root directory.",
                },
                "symbol_identifier": {
                    "type": "string",
                    "description": "Qualified or simple symbol identifier (e.g. 'Model.save' or 'save').",
                },
                "symbol_id": {
                    "type": "integer",
                    "description": "Numerical AST symbol ID.",
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Optional custom cache directory.",
                },
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "build_code_graph",
        "description": (
            "Compile and persist zero-copy memory-mapped CSR code graph assets for a repository."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Path to the repository root directory.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Custom cache output directory (defaults to <repo_path>/.perron_cache).",
                },
                "force": {
                    "type": "boolean",
                    "description": "Force recompilation even if cache exists (default: false).",
                    "default": False,
                },
            },
            "required": ["repo_path"],
        },
    },
]


class PerronMCPServer:
    """
    Lightweight, self-contained Model Context Protocol (MCP) server.
    """

    def __init__(self, default_repo: Optional[Path] = None, default_cache: Optional[Path] = None):
        self.default_repo = default_repo
        self.default_cache = default_cache
        self.active_graphs: Dict[str, CodeGraph] = {}

    def get_or_load_graph(self, repo_path: str, cache_dir: Optional[str] = None) -> CodeGraph:
        rp = Path(repo_path).resolve()
        cp = Path(cache_dir).resolve() if cache_dir else (rp / ".perron_cache")
        key = f"{rp}:{cp}"
        if key not in self.active_graphs:
            logger.info("Opening code graph for %s (cache: %s)...", rp, cp)
            self.active_graphs[key] = CodeGraph.from_directory(rp, cp)
        return self.active_graphs[key]

    def close(self) -> None:
        for key, graph in list(self.active_graphs.items()):
            try:
                graph.close()
            except Exception as e:
                logger.warning("Error closing graph %s: %s", key, e)
        self.active_graphs.clear()

    # -------------------------------------------------------------------------
    # Tool Implementations
    # -------------------------------------------------------------------------

    def tool_retrieve_context(self, arguments: Dict[str, Any]) -> str:
        repo_path = arguments.get("repo_path") or (str(self.default_repo) if self.default_repo else None)
        query = arguments.get("query")
        if not repo_path or not query:
            raise ValueError("Arguments 'repo_path' and 'query' are required.")

        budget = int(arguments.get("token_budget", 4096))
        gamma = float(arguments.get("gamma", 0.70))
        beta = float(arguments.get("beta", 0.85))
        cache_dir = arguments.get("cache_dir")

        t0 = time.perf_counter()
        graph = self.get_or_load_graph(repo_path, cache_dir)
        retriever = PerronRetriever(graph, gamma=gamma, beta=beta)
        context = retriever.query(query, max_tokens=budget)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        header = f"# Perron Code Graph Context ({elapsed_ms:.1f} ms diffusion | budget: {budget} tokens)\n\n"
        return header + context

    def tool_inspect_symbol_breadcrumbs(self, arguments: Dict[str, Any]) -> str:
        repo_path = arguments.get("repo_path") or (str(self.default_repo) if self.default_repo else None)
        if not repo_path:
            raise ValueError("Argument 'repo_path' is required.")

        sym_id = arguments.get("symbol_id")
        sym_ident = arguments.get("symbol_identifier")
        cache_dir = arguments.get("cache_dir")

        if sym_id is None and not sym_ident:
            raise ValueError("Either 'symbol_id' or 'symbol_identifier' must be provided.")

        graph = self.get_or_load_graph(repo_path, cache_dir)
        target_idx: Optional[int] = None

        if sym_id is not None:
            if 0 <= sym_id < len(graph.symbols):
                target_idx = int(sym_id)
        elif sym_ident:
            q_clean = sym_ident.strip().lower()
            # 1. Exact match on qualified_name
            for i, sym in enumerate(graph.symbols):
                qn = getattr(sym, "qualified_name", getattr(sym, "name", "")).lower()
                if qn == q_clean:
                    target_idx = i
                    break
            # 2. Substring match fallback
            if target_idx is None:
                for i, sym in enumerate(graph.symbols):
                    qn = getattr(sym, "qualified_name", getattr(sym, "name", "")).lower()
                    if q_clean in qn:
                        target_idx = i
                        break

        if target_idx is None:
            return f"Symbol not found for identifier='{sym_ident}', id='{sym_id}' across {len(graph.symbols)} indexed symbols."

        sym = graph.symbols[target_idx]
        name = getattr(sym, "qualified_name", getattr(sym, "name", "unknown"))
        fpath = getattr(sym, "file_path", "unknown")
        start_line = getattr(sym, "start_line", getattr(sym, "line_start", 1))
        end_line = getattr(sym, "end_line", getattr(sym, "line_end", 1))
        stype = getattr(sym, "symbol_type", "function")
        doc = getattr(sym, "docstring", None) or ""

        # Analyze graph connections from CSR matrix
        callee_indices: List[int] = []
        caller_indices: List[int] = []
        if hasattr(graph.t_matrix, "indptr") and hasattr(graph.t_matrix, "indices"):
            # Outgoing edges (callees)
            c_start = graph.t_matrix.indptr[target_idx]
            c_end = graph.t_matrix.indptr[target_idx + 1]
            callee_indices = [int(idx) for idx in graph.t_matrix.indices[c_start:c_end]]

            # Incoming edges (callers) via transpose lookup
            t_trans = graph.t_matrix.tocsc()
            in_start = t_trans.indptr[target_idx]
            in_end = t_trans.indptr[target_idx + 1]
            caller_indices = [int(idx) for idx in t_trans.indices[in_start:in_end]]

        pi_val = float(graph.pi_global[target_idx]) if graph.pi_global is not None else 0.0

        callees_fmt = [
            f"- [{c_idx}] `{getattr(graph.symbols[c_idx], 'qualified_name', 'node_' + str(c_idx))}` ({getattr(graph.symbols[c_idx], 'file_path', '')})"
            for c_idx in callee_indices[:20]
        ]
        callers_fmt = [
            f"- [{c_idx}] `{getattr(graph.symbols[c_idx], 'qualified_name', 'node_' + str(c_idx))}` ({getattr(graph.symbols[c_idx], 'file_path', '')})"
            for c_idx in caller_indices[:20]
        ]

        lines = [
            f"## Symbol Breadcrumb: `{name}` (ID: {target_idx})",
            f"- **Type**: `{stype}`",
            f"- **Location**: `{fpath}#L{start_line}-L{end_line}`",
            f"- **Global PageRank (\\pi_g)**: `{pi_val:.6e}`",
            f"- **In-Degree (Callers)**: {len(caller_indices)}",
            f"- **Out-Degree (Callees)**: {len(callee_indices)}",
        ]
        if doc:
            first_line_doc = doc.strip().split("\n")[0]
            lines.append(f"- **Docstring**: {first_line_doc}")

        lines.append(f"\n### Direct Callers (Incoming Edges: {len(caller_indices)} total):")
        lines.extend(callers_fmt if callers_fmt else ["- None"])

        lines.append(f"\n### Direct Callees (Outgoing Edges: {len(callee_indices)} total):")
        lines.extend(callees_fmt if callees_fmt else ["- None"])

        return "\n".join(lines)

    def tool_build_code_graph(self, arguments: Dict[str, Any]) -> str:
        repo_path = arguments.get("repo_path") or (str(self.default_repo) if self.default_repo else None)
        if not repo_path:
            raise ValueError("Argument 'repo_path' is required.")

        output_dir = arguments.get("output_dir")
        force = bool(arguments.get("force", False))

        rp = Path(repo_path).resolve()
        cp = Path(output_dir).resolve() if output_dir else (rp / ".perron_cache")

        t0 = time.perf_counter()
        graph = CodeGraph.from_directory(rp, cp, force_recompile=force)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n_syms = len(graph.symbols)
        nnz = int(graph.t_matrix.nnz)
        shape = list(graph.t_matrix.shape)

        return (
            f"Successfully compiled code graph for `{rp}` in {elapsed_ms:.1f} ms.\n"
            f"- Symbols indexed: {n_syms}\n"
            f"- CSR shape: {shape}\n"
            f"- Edges (nnz): {nnz}\n"
            f"- Cache location: `{cp}`\n"
            f"- Memory virtualization: Zero-copy read-only mmap ready."
        )

    # -------------------------------------------------------------------------
    # JSON-RPC Message Dispatcher
    # -------------------------------------------------------------------------

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        msg_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {}) or {}

        # 1. Initialize
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "perron-mcp",
                        "version": __version__,
                    },
                },
            }

        # 2. Initialized notification (no response)
        if method == "notifications/initialized":
            return None

        # 3. Ping
        if method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {},
            }

        # 4. Tools list
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": TOOLS,
                },
            }

        # 5. Tools call
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            try:
                if tool_name == "retrieve_context":
                    output_text = self.tool_retrieve_context(arguments)
                elif tool_name == "inspect_symbol_breadcrumbs":
                    output_text = self.tool_inspect_symbol_breadcrumbs(arguments)
                elif tool_name == "build_code_graph":
                    output_text = self.tool_build_code_graph(arguments)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method/Tool '{tool_name}' not found.",
                        },
                    }

                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": output_text,
                            }
                        ],
                        "isError": False,
                    },
                }
            except Exception as e:
                logger.error("Tool execution error: %s", e, exc_info=True)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error executing '{tool_name}': {str(e)}",
                            }
                        ],
                        "isError": True,
                    },
                }

        # Unknown method
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}",
                },
            }
        return None

    def run_stdio(self) -> None:
        """
        Run the JSON-RPC stdio event loop.
        """
        logger.info("Perron MCP Server v%s starting on stdio...", __version__)
        try:
            if hasattr(sys.stdin, "reconfigure"):
                sys.stdin.reconfigure(encoding="utf-8")
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError as err:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(err)}",
                    },
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()
                continue

            response = self.handle_request(request)
            if response is not None:
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

        self.close()
        logger.info("Perron MCP Server terminated cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="perron-mcp",
        description="Perron Model Context Protocol (MCP) Server for Code Graph Retrieval",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"perron-mcp {__version__}",
    )
    parser.add_argument(
        "--repo",
        "-r",
        help="Optional default repository root path.",
    )
    parser.add_argument(
        "--cache",
        "-c",
        help="Optional default cache directory.",
    )
    args = parser.parse_args()

    default_repo = Path(args.repo).resolve() if args.repo else None
    default_cache = Path(args.cache).resolve() if args.cache else None

    server = PerronMCPServer(default_repo=default_repo, default_cache=default_cache)
    try:
        server.run_stdio()
    except KeyboardInterrupt:
        server.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
