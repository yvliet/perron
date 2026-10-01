"""
Perron Real Repository AST Graph Extractor.

Extracts real call/caller graphs from Python packages, constructs row-stochastic
CSR transition matrices, and benchmarks physical binary size, zero-copy mmap latency,
and CPU power iteration diffusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ast
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.sparse import csr_matrix

from perron.matrix import (
    build_static_transition_matrix,
    save_mmap_csr,
    load_mmap_csr,
    close_mmap_csr,
)
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import ASTContextSymbol


@dataclass
class RealGraphTelemetry:
    package_name: str
    num_symbols: int
    num_call_edges: int
    num_caller_edges: int
    total_edges: int
    max_in_degree: int
    top_5_hubs: List[Tuple[str, int]]
    csr_disk_size_bytes: int
    csr_disk_size_kb: float
    mmap_load_time_ms: float
    global_ppr_time_ms: float
    query_ppr_time_ms: float
    fits_in_l3_cache: bool


class CodebaseASTVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, base_dir: Path):
        self.file_path = file_path
        self.rel_path = str(Path(file_path).relative_to(base_dir)).replace("\\", "/")
        self.symbols: List[dict] = []
        self.calls: List[Tuple[str, str]] = []  # (caller_name, callee_name)
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None

    def visit_ClassDef(self, node: ast.ClassDef):
        prev_class = self.current_class
        self.current_class = node.name
        sym_name = f"{self.current_class}"
        self.symbols.append({
            "name": sym_name,
            "file_path": self.rel_path,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 10),
            "parent_class": prev_class,
            "is_class": True,
        })
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node)

    def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        prev_func = self.current_function
        func_name = f"{self.current_class}.{node.name}" if self.current_class else node.name
        self.current_function = func_name
        self.symbols.append({
            "name": func_name,
            "file_path": self.rel_path,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno + 10),
            "parent_class": self.current_class,
            "is_class": False,
        })
        self.generic_visit(node)
        self.current_function = prev_func

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            callee = None
            if isinstance(node.func, ast.Name):
                callee = node.func.id
            elif isinstance(node.func, ast.Attribute):
                callee = node.func.attr
            if callee:
                self.calls.append((self.current_function, callee))
        self.generic_visit(node)


def extract_real_graph_from_directory(
    directory: Path,
    package_name: str = "python_stdlib_net",
    max_files: Optional[int] = None,
) -> Tuple[List[ASTContextSymbol], List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, str]]:
    """
    Parse all Python source files under a directory and extract an AST symbol dependency graph
    using perron.graph's composite keying and verbatim code extraction.
    """
    from perron.graph import extract_repository_graph

    sym_nodes, call_edges, caller_edges, canonical_to_id, id_to_canonical = extract_repository_graph(
        repo_dir=directory,
        max_files=max_files,
    )

    symbols = [s.to_context_symbol() for s in sym_nodes]

    return symbols, call_edges, caller_edges, id_to_canonical


def profile_real_graph(
    directory: Path,
    output_dir: Path,
    package_name: str = "stdlib_core",
) -> RealGraphTelemetry:
    """
    Profile real AST extraction, build CSR, benchmark mmap and CPU diffusion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols, call_edges, caller_edges, id_to_name = extract_real_graph_from_directory(
        directory=directory,
        package_name=package_name,
    )
    num_nodes = len(symbols)

    t_matrix, dangling = build_static_transition_matrix(
        num_nodes=num_nodes,
        call_edges=call_edges,
        caller_edges=caller_edges,
        lambda_call=1.0,
        lambda_caller=0.2,
    )

    node_to_id = {name: nid for nid, name in id_to_name.items()}
    save_mmap_csr(output_dir, t_matrix, dangling, node_to_id, id_to_name)

    # Compute physical disk size
    files_to_check = ["data.npy", "indices.npy", "indptr.npy", "dangling.npy"]
    total_bytes = sum((output_dir / f).stat().st_size for f in files_to_check if (output_dir / f).exists())

    # Measure zero-copy mmap load latency
    t0 = time.perf_counter()
    loaded_t, loaded_dang, _, _ = load_mmap_csr(output_dir)
    t1 = time.perf_counter()
    mmap_ms = (t1 - t0) * 1000.0

    # Measure global PPR latency
    t2 = time.perf_counter()
    pi_global = compute_global_pagerank(loaded_t, loaded_dang, beta=0.85, max_iter=100)
    t3 = time.perf_counter()
    global_ms = (t3 - t2) * 1000.0

    # Measure query-directed PPR latency
    p_0 = np.zeros(num_nodes, dtype=np.float32)
    p_0[0] = 1.0
    t4 = time.perf_counter()
    pi_query = personalized_pagerank_power_iteration(loaded_t, loaded_dang, p_0, beta=0.85, max_iter=100)
    t5 = time.perf_counter()
    query_ms = (t5 - t4) * 1000.0

    # Compute in-degree distribution to verify power law
    in_degrees: Dict[int, int] = {}
    for _, v in call_edges:
        in_degrees[v] = in_degrees.get(v, 0) + 1

    sorted_hubs = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5 = [(id_to_name.get(nid, f"node_{nid}"), deg) for nid, deg in sorted_hubs]
    max_in_deg = sorted_hubs[0][1] if sorted_hubs else 0

    close_mmap_csr(loaded_t, loaded_dang)

    return RealGraphTelemetry(
        package_name=package_name,
        num_symbols=num_nodes,
        num_call_edges=len(call_edges),
        num_caller_edges=len(caller_edges),
        total_edges=len(call_edges) + len(caller_edges),
        max_in_degree=max_in_deg,
        top_5_hubs=top_5,
        csr_disk_size_bytes=total_bytes,
        csr_disk_size_kb=round(total_bytes / 1024.0, 2),
        mmap_load_time_ms=round(mmap_ms, 3),
        global_ppr_time_ms=round(global_ms, 2),
        query_ppr_time_ms=round(query_ms, 2),
        fits_in_l3_cache=(total_bytes < 12 * 1024 * 1024),  # 12 MB Intel Smart Cache
    )


if __name__ == "__main__":
    # Target real stdlib packages: urllib, http, email, json
    import sysconfig
    stdlib_lib = Path(sysconfig.get_path("stdlib"))
    if not (stdlib_lib / "urllib").is_dir():
        import urllib
        stdlib_lib = Path(urllib.__file__).parent.parent
    target_dirs = [stdlib_lib / "urllib", stdlib_lib / "http", stdlib_lib / "email"]

    # Also test perron package itself
    perron_dir = REPO_ROOT / "perron"
    out_dir = REPO_ROOT / "benchmarks" / "real_graphs_out"

    print("Extracting real AST call graph from perron codebase...")
    telemetry_perron = profile_real_graph(perron_dir, out_dir / "perron", "perron_core")
    print(f"[PERRON] Symbols: {telemetry_perron.num_symbols}, Edges: {telemetry_perron.total_edges}, CSR Size: {telemetry_perron.csr_disk_size_kb} KB, mmap: {telemetry_perron.mmap_load_time_ms} ms, diffusion: {telemetry_perron.query_ppr_time_ms} ms, L3 Cache: {telemetry_perron.fits_in_l3_cache}")

    print("Extracting real AST call graph from Python stdlib networking & HTTP stack...")
    telemetry_stdlib = profile_real_graph(stdlib_lib / "urllib", out_dir / "urllib", "stdlib_urllib")
    print(f"[STDLIB_URLLIB] Symbols: {telemetry_stdlib.num_symbols}, Edges: {telemetry_stdlib.total_edges}, Max In-Degree: {telemetry_stdlib.max_in_degree}, Top Hubs: {telemetry_stdlib.top_5_hubs[:3]}, CSR Size: {telemetry_stdlib.csr_disk_size_kb} KB, mmap: {telemetry_stdlib.mmap_load_time_ms} ms, diffusion: {telemetry_stdlib.query_ppr_time_ms} ms, L3 Cache: {telemetry_stdlib.fits_in_l3_cache}")
