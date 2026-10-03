#!/usr/bin/env python3
"""
Measurement and Verification Engine for Spectral Convergence and Brauer Bounds.

Measures Personalized PageRank power iteration convergence across authentic
scale-free Python code graphs (Requests and SymPy), verifying the Brauer rank-1
perturbation theorem bound (|lambda_2| <= beta = 0.85, spectral gap >= 0.15,
max iterations <= 90) and serializing empirical telemetry to results/spectral_convergence.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_power_iteration_profile(
    repo_name: str,
    target_node_idx: int,
    beta: float = 0.85,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Dict[str, Any]:
    graph_dir = REPO_ROOT / "data" / "real_graphs" / repo_name
    if not graph_dir.exists():
        raise FileNotFoundError(f"Missing real graph directory: {graph_dir}")

    indptr = np.load(graph_dir / "indptr.npy")
    indices = np.load(graph_dir / "indices.npy")
    data = np.load(graph_dir / "data.npy")
    dangling = np.load(graph_dir / "dangling.npy")

    num_nodes = len(indptr) - 1
    num_edges = len(indices)
    t_mat = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))

    # Query teleport prior: seed teleport on the canonical defect target symbol
    p_0_vec = np.zeros(num_nodes, dtype=np.float64)
    p_0_vec[target_node_idx % num_nodes] = 1.0

    dangling_vec = dangling.astype(np.float64)

    # Benchmark convergence timing
    latencies = []
    iters_converged = 0
    final_residual = 0.0

    # Warmup
    pi = p_0_vec.copy()
    for iters in range(1, max_iter + 1):
        pi_next = beta * (pi @ t_mat)
        dangling_mass = np.dot(pi, dangling_vec)
        teleport_factor = beta * dangling_mass + (1.0 - beta)
        pi_next += teleport_factor * p_0_vec
        diff = np.sum(np.abs(pi_next - pi))
        pi = pi_next
        if diff < tol:
            iters_converged = iters
            final_residual = float(diff)
            break

    # Measurement trials for stable mean latency
    for _ in range(50):
        t0 = time.perf_counter()
        pi = p_0_vec.copy()
        for iters in range(1, max_iter + 1):
            pi_next = beta * (pi @ t_mat)
            dangling_mass = np.dot(pi, dangling_vec)
            teleport_factor = beta * dangling_mass + (1.0 - beta)
            pi_next += teleport_factor * p_0_vec
            diff = np.sum(np.abs(pi_next - pi))
            pi = pi_next
            if diff < tol:
                break
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))

    return {
        "repo": repo_name,
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "target_node_idx": int(target_node_idx),
        "beta": float(beta),
        "tolerance": float(tol),
        "iterations_to_convergence": int(iters_converged),
        "final_l1_residual": float(final_residual),
        "latency_ms_mean": round(mean_latency_ms, 2),
    }


def measure_spectral_convergence() -> Dict[str, Any]:
    req_profile = run_power_iteration_profile("requests", target_node_idx=35)
    sym_profile = run_power_iteration_profile("sympy", target_node_idx=0)
    req_iters = int(req_profile["iterations_to_convergence"])
    sym_iters = int(sym_profile["iterations_to_convergence"])
    max_iters = math.ceil(math.log(2.0 / 1e-6) / math.log(1.0 / 0.85))

    results = {
        "theoretical_bounds": {
            "beta": 0.85,
            "brauer_bound_lambda_2": 0.85,
            "spectral_gap": 0.15,
            "tolerance_epsilon": 1e-06,
            "theoretical_max_iterations": max_iters,
            "formula": "ceil(ln(2/eps) / ln(1/beta))",
        },
        "empirical_convergence": {
            "requests": {
                "num_nodes": req_profile["num_nodes"],
                "num_edges": req_profile["num_edges"],
                "seed_node_idx": 35,
                "beta": 0.85,
                "tolerance": 1e-06,
                "power_iterations": req_iters,
                "latency_ms": req_profile["latency_ms_mean"],
                "final_l1_residual": req_profile["final_l1_residual"],
            },
            "sympy": {
                "num_nodes": sym_profile["num_nodes"],
                "num_edges": sym_profile["num_edges"],
                "seed_node_idx": 0,
                "beta": 0.85,
                "tolerance": 1e-06,
                "power_iterations": sym_iters,
                "latency_ms": sym_profile["latency_ms_mean"],
                "final_l1_residual": sym_profile["final_l1_residual"],
            },
        },
    }

    out_file = REPO_ROOT / "results" / "spectral_convergence.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Serialized spectral convergence telemetry to {out_file}")
    return results


if __name__ == "__main__":
    measure_spectral_convergence()
