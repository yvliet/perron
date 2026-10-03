"""
Benchmark: Synthetic 1M-Node CSR Profiling & Multi-Worker Shared Memory Virtualization.

Evaluates systems performance on scale-free graphs (|V| = 1M, |E| = 10M):
1. End-to-End FLOP Reality: Single-pass O(|E|) transition reweighting vs 85-iteration SpMV diffusion.
2. Multi-Worker Memory Scaling: Static mmap CSR (W -> 1 OS page cache sharing) vs Dynamic Reweighting (W x Copy-on-Write).
3. Cold-Start Loading Latency: mmap vs np.load vs pickle.

Saves benchmark results to data/csr_scaling_profile.json.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp


def generate_synthetic_powerlaw_graph(
    num_nodes: int = 1_000_000,
    num_edges: int = 10_000_000,
    alpha: float = 2.1,
    seed: int = 42,
) -> sp.csr_matrix:
    """
    Generate synthetic directed power-law multigraph using Pareto preferential attachment.
    """
    rng = np.random.default_rng(seed)
    print(f"Generating synthetic graph: |V| = {num_nodes:,}, |E| = {num_edges:,}...")
    t0 = time.time()

    # Sample destination nodes from Pareto power-law distribution
    raw_dest = (rng.pareto(alpha - 1.0, size=num_edges) * 5.0).astype(np.int64)
    cols = np.clip(raw_dest, 0, num_nodes - 1).astype(np.int32)

    # Sample source nodes uniformly across nodes
    rows = rng.integers(0, num_nodes, size=num_edges, dtype=np.int32)
    data = np.ones(num_edges, dtype=np.float32)

    # Construct CSR
    coo = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    csr = coo.tocsr()

    # Enforce row-stochasticity
    row_sums = np.asarray(csr.sum(axis=1)).flatten()
    inv_r = np.zeros_like(row_sums, dtype=np.float32)
    mask = row_sums > 0
    inv_r[mask] = 1.0 / row_sums[mask]
    csr = csr.multiply(inv_r[:, None]).tocsr()

    print(f"Graph generated in {time.time() - t0:.2f}s (nnz={csr.nnz:,}).")
    return csr


def profile_diffusion_vs_reweighting(
    t_mat: sp.csr_matrix,
    num_iterations: int = 85,
    beta: float = 0.85,
    n_runs: int = 5,
) -> Dict[str, Any]:
    """
    Measure end-to-end wall-clock time of 85 SpMV diffusion iterations vs single-pass O(|E|) reweighting.
    """
    n = t_mat.shape[0]
    nnz = t_mat.nnz
    dangling = (np.diff(t_mat.indptr) == 0).astype(np.float32)
    p_0 = np.full(n, 1.0 / n, dtype=np.float32)

    # 1. Benchmark SpMV Diffusion (Power Iteration)
    spmv_times = []
    for _ in range(n_runs):
        x = p_0.copy()
        t0 = time.perf_counter()
        for _ in range(num_iterations):
            dangling_mass = float(np.dot(x, dangling))
            x_next = beta * t_mat.transpose().dot(x) + (1.0 - beta + beta * dangling_mass) * p_0
            x = x_next
        t1 = time.perf_counter()
        spmv_times.append((t1 - t0) * 1000.0)

    mean_spmv_ms = float(np.mean(spmv_times))

    # 2. Benchmark Single-Pass O(|E|) Dynamic Reweighting
    weights = np.random.uniform(0.5, 2.0, size=n).astype(np.float32)
    reweight_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        diag_w = sp.diags(weights)
        m_rw = t_mat.dot(diag_w).tocsr()
        r_sums = np.asarray(m_rw.sum(axis=1)).flatten()
        inv_r = np.zeros_like(r_sums, dtype=np.float32)
        mask = r_sums > 0
        inv_r[mask] = 1.0 / r_sums[mask]
        t_rw = m_rw.multiply(inv_r[:, None]).tocsr()
        t1 = time.perf_counter()
        reweight_times.append((t1 - t0) * 1000.0)

    mean_reweight_ms = float(np.mean(reweight_times))
    flop_fraction = (mean_reweight_ms / mean_spmv_ms) * 100.0

    return {
        "num_nodes": n,
        "num_edges": nnz,
        "spmv_iterations": num_iterations,
        "mean_diffusion_time_ms": round(mean_spmv_ms, 2),
        "mean_reweighting_time_ms": round(mean_reweight_ms, 2),
        "reweighting_overhead_percentage": round(flop_fraction, 2),
    }


def profile_storage_formats(
    t_mat: sp.csr_matrix,
    temp_dir: Path,
) -> Dict[str, Any]:
    """
    Compare cold-start loading time and file sizes across mmap, np.load, and pickle.
    """
    mmap_path = temp_dir / "graph_mmap.bin"
    npz_path = temp_dir / "graph_np.npz"
    pkl_path = temp_dir / "graph_pkl.pkl"

    # Save to disk
    sp.save_npz(npz_path, t_mat)
    with open(pkl_path, "wb") as f:
        pickle.dump(t_mat, f)

    # Save raw CSR binary buffers for mmap
    with open(mmap_path, "wb") as f:
        f.write(t_mat.indptr.tobytes())
        f.write(t_mat.indices.tobytes())
        f.write(t_mat.data.tobytes())

    # Measure sizes
    npz_size_mb = os.path.getsize(npz_path) / (1024 * 1024)
    pkl_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    mmap_size_mb = os.path.getsize(mmap_path) / (1024 * 1024)

    # Measure Cold-Start Load Time
    # 1. Pickle load
    t0 = time.perf_counter()
    with open(pkl_path, "rb") as f:
        _ = pickle.load(f)
    pkl_load_ms = (time.perf_counter() - t0) * 1000.0

    # 2. NPZ load
    t0 = time.perf_counter()
    _ = sp.load_npz(npz_path)
    npz_load_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Memory-mapped load (zero-copy pointer mapping)
    t0 = time.perf_counter()
    offset_indptr = 0
    size_indptr = (t_mat.shape[0] + 1) * 4
    offset_indices = size_indptr
    size_indices = t_mat.nnz * 4
    offset_data = offset_indices + size_indices

    m_indptr = np.memmap(mmap_path, dtype=np.int32, mode="r", offset=offset_indptr, shape=(t_mat.shape[0] + 1,))
    m_indices = np.memmap(mmap_path, dtype=np.int32, mode="r", offset=offset_indices, shape=(t_mat.nnz,))
    m_data = np.memmap(mmap_path, dtype=np.float32, mode="r", offset=offset_data, shape=(t_mat.nnz,))
    _ = sp.csr_matrix((m_data, m_indices, m_indptr), shape=t_mat.shape)
    mmap_load_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "formats": {
            "pickle": {"size_mb": round(pkl_size_mb, 2), "load_time_ms": round(pkl_load_ms, 2)},
            "npz": {"size_mb": round(npz_size_mb, 2), "load_time_ms": round(npz_load_ms, 2)},
            "mmap_csr": {"size_mb": round(mmap_size_mb, 2), "load_time_ms": round(mmap_load_ms, 2)},
        }
    }


def compute_multiworker_memory_scaling(
    base_matrix_mb: float = 84.0,
    workers_list: Tuple[int, ...] = (1, 2, 4, 8),
) -> Dict[str, Any]:
    """
    Calculate theoretical and virtualized physical resident memory consumption
    across parallel agent workers sharing static mmap CSR vs private heap dynamic reweighting.
    """
    res = {}
    for w in workers_list:
        dynamic_cow_mb = w * base_matrix_mb
        static_mmap_mb = base_matrix_mb  # Shared physical OS page cache copy (W -> 1)
        savings_mb = dynamic_cow_mb - static_mmap_mb
        res[f"{w}_workers"] = {
            "workers": w,
            "dynamic_reweighting_rss_mb": round(dynamic_cow_mb, 1),
            "static_mmap_rss_mb": round(static_mmap_mb, 1),
            "memory_saved_mb": round(savings_mb, 1),
            "reduction_factor": f"{w}x",
        }
    return res


def run_csr_profiling() -> Dict[str, Any]:
    """Execute complete 1M-node systems profiling benchmark."""
    print("=" * 80)
    print("STARTING 1M-NODE CSR SYSTEMS PROFILING (MULTI-WORKER MEMORY VIRTUALIZATION)")
    print("=" * 80)

    csr = generate_synthetic_powerlaw_graph(num_nodes=1_000_000, num_edges=10_000_000, alpha=2.1)

    print("\n[Step 1] Profiling 85-iteration SpMV diffusion vs single-pass reweighting...")
    diffusion_profile = profile_diffusion_vs_reweighting(csr, num_iterations=85)
    print(
        f"Diffusion (85 iters): {diffusion_profile['mean_diffusion_time_ms']} ms | "
        f"Reweighting (1 pass): {diffusion_profile['mean_reweighting_time_ms']} ms "
        f"({diffusion_profile['reweighting_overhead_percentage']}% of total diffusion time)."
    )

    print("\n[Step 2] Profiling cold-start loading across storage formats...")
    with tempfile.TemporaryDirectory() as tmp_d:
        storage_profile = profile_storage_formats(csr, Path(tmp_d))
    for fmt, data in storage_profile["formats"].items():
        print(f"  {fmt:<10}: Size = {data['size_mb']:>6.1f} MB | Cold-Start Load = {data['load_time_ms']:>6.2f} ms")

    print("\n[Step 3] Modeling multi-worker shared memory virtualization (W -> 1)...")
    memory_scaling = compute_multiworker_memory_scaling(base_matrix_mb=84.0)
    for k, v in memory_scaling.items():
        print(
            f"  {v['workers']} Workers: Dynamic = {v['dynamic_reweighting_rss_mb']} MB | "
            f"Static mmap = {v['static_mmap_rss_mb']} MB | Saved = {v['memory_saved_mb']} MB ({v['reduction_factor']} reduction)"
        )

    full_results = {
        "benchmark_name": "Synthetic 1M-Node CSR Systems Profile",
        "diffusion_vs_reweighting": diffusion_profile,
        "storage_and_cold_start": storage_profile,
        "multiworker_memory_virtualization": memory_scaling,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_p = Path("data/csr_scaling_profile.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved profile results to {out_p}")
    print("=" * 80 + "\n")
    return full_results


if __name__ == "__main__":
    run_csr_profiling()
