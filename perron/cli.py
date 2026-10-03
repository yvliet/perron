"""
Perron Command-Line Interface.
Provides turnkey CLI commands: `perron index` and `perron query`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from perron import __version__
from perron.retriever import CodeGraph, PerronRetriever


def cmd_index(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo).resolve()
    cache_path = Path(args.out).resolve() if args.out else repo_path / ".perron_cache"

    print(f"Indexing repository at: {repo_path}")
    print(f"Cache target directory: {cache_path}")
    t0 = time.perf_counter()

    graph = CodeGraph.from_directory(repo_path, cache_path, force_recompile=args.force)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[SUCCESS] Indexed {graph.n_symbols} symbols in {elapsed_ms:.2f} ms")
    print(f"Matrix shape: {graph.t_matrix.shape}, Edges (nnz): {graph.t_matrix.nnz}")
    graph.close()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    repo_path = Path(args.repo).resolve()
    cache_path = Path(args.cache).resolve() if args.cache else repo_path / ".perron_cache"

    has_cache = (cache_path / "indptr.npy").exists() or (cache_path / "t_matrix_indptr.npy").exists()
    if not has_cache:
        print(f"[ERROR] Cache not found at {cache_path}. Run `perron index {repo_path}` first.")
        return 1

    graph = CodeGraph.from_directory(repo_path, cache_path)
    retriever = PerronRetriever(graph, gamma=args.gamma, beta=args.beta)

    t0 = time.perf_counter()
    context = retriever.query(args.query_text, max_tokens=args.budget)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n--- Perron Context ({elapsed_ms:.2f} ms diffusion) ---\n")
    print(context)
    graph.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from perron.bench import format_benchmark_report, run_benchmark

    try:
        res = run_benchmark(method=args.method, split=args.split)
        print("\n" + format_benchmark_report(res) + "\n")
        return 0
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {e}", file=sys.stderr)
        return 1


def cmd_eval(args: argparse.Namespace) -> int:
    """Run evaluation suite across registered baselines (e.g. table2 on dev_val)."""
    suite = getattr(args, "suite", "table2")
    split = getattr(args, "split", "dev_val")
    print(f"Running Perron Evaluation Suite: {suite} on split: {split}")
    import subprocess
    cmd = [sys.executable, "scripts/run_suite.py", "--split", split]
    ret = subprocess.run(cmd)
    return ret.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="perron",
        description="Perron: High-Throughput Spectral Code Graph Retrieval for Developer Agents",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"perron-core {__version__}",
        help="Show program's version number and exit.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: index
    p_index = subparsers.add_parser("index", help="Compile and index a repository call graph.")
    p_index.add_argument("repo", help="Path to repository root.")
    p_index.add_argument("--out", "-o", help="Custom cache output directory.")
    p_index.add_argument("--force", "-f", action="store_true", help="Force recompilation.")
    p_index.set_defaults(func=cmd_index)

    # Subcommand: query
    p_query = subparsers.add_parser("query", help="Query repository context using spectral diffusion.")
    p_query.add_argument("repo", help="Path to repository root.")
    p_query.add_argument("query_text", help="Query string or bug description.")
    p_query.add_argument("--budget", "-k", type=int, default=4096, help="Token budget ceiling.")
    p_query.add_argument("--gamma", type=float, default=0.70, help="Specificity damping exponent.")
    p_query.add_argument("--beta", type=float, default=0.85, help="Teleport damping factor.")
    p_query.add_argument("--cache", "-c", help="Custom cache path.")
    p_query.set_defaults(func=cmd_query)

    # Subcommand: bench
    p_bench = subparsers.add_parser("bench", help="Run the Hub-Gold vs Non-Hub-Gold ablation benchmark.")
    p_bench.add_argument("--method", "-m", default="perron", help="Retrieval method (bm25, std_ppr, deg_ppr, perron).")
    p_bench.add_argument("--split", "-s", default="dev_val", choices=["dev_val", "heldout", "dev", "full"], help="Dataset split.")
    p_bench.set_defaults(func=cmd_bench)

    # Subcommand: eval
    p_eval = subparsers.add_parser("eval", help="Regenerate evaluation benchmark suite.")
    p_eval.add_argument("--suite", default="table2", help="Benchmark suite name.")
    p_eval.add_argument("--split", "-s", default="dev_val", choices=["dev_val", "heldout"], help="Dataset split.")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
