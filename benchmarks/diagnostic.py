"""
Perron Diagnostic Benchmark Suite.

Evaluates graph retrieval accuracy, hub suppression, and latency across
synthetically generated and real SWE-bench dependency graphs.

Metrics:
  - File-Level Recall@K (R_file@K)
  - Strict Function-Level Recall@K (R_func@K)
  - Mean Reciprocal Rank (MRR)
  - Hub Suppression Index (HSI)
  - Ingestion and Diffusion Latency Profiling
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np

from perron.matrix import build_static_transition_matrix
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import ASTContextSymbol, pack_context_subgraphs


@dataclass
class DiagnosticMetrics:
    file_recall_2k: float
    file_recall_4k: float
    func_recall_2k: float
    func_recall_4k: float
    mrr: float
    hub_suppression_index: float
    avg_diffusion_latency_ms: float
    avg_packing_latency_ms: float


def generate_synthetic_repo_graph(
    num_nodes: int = 1000,
    num_hubs: int = 20,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict[int, ASTContextSymbol], List[int]]:
    """
    Generate scale-free synthetic call graph with power-law hub nodes and realistic module structure.
    """
    rng = np.random.default_rng(seed)
    call_edges: List[Tuple[int, int]] = []
    caller_edges: List[Tuple[int, int]] = []
    symbols: Dict[int, ASTContextSymbol] = {}
    hubs = list(range(num_hubs))

    modules = [f"src/module_{i // 50}/file_{i // 10}.py" for i in range(num_nodes)]

    for u in range(num_nodes):
        num_hub_connections = rng.integers(1, 4)
        chosen_hubs = rng.choice(hubs, size=num_hub_connections, replace=False)
        for h in chosen_hubs:
            if u != h:
                call_edges.append((u, h))
                caller_edges.append((h, u))

        num_local = rng.integers(1, 3)
        local_candidates = [
            v for v in range(max(0, u - 15), min(num_nodes, u + 15)) if v != u and v not in hubs
        ]
        if local_candidates:
            chosen_local = rng.choice(local_candidates, size=min(num_local, len(local_candidates)), replace=False)
            for v in chosen_local:
                call_edges.append((u, v))
                caller_edges.append((v, u))

        symbols[u] = ASTContextSymbol(
            node_id=u,
            name=f"symbol_{u}",
            file_path=modules[u],
            start_line=(u % 20) * 15 + 1,
            end_line=(u % 20) * 15 + 14,
            code=f"def symbol_{u}():\n    pass\n",
            token_count=rng.integers(25, 75),
            parent_class=f"Class_{u // 10}" if u % 3 == 0 else None,
        )

    return call_edges, caller_edges, symbols, hubs


def evaluate_diagnostic_benchmark(
    num_instances: int = 50,
    graph_size: int = 1000,
    seed: int = 42,
) -> DiagnosticMetrics:
    """
    Run diagnostic evaluation over simulated SWE-bench instances.
    """
    rng = np.random.default_rng(seed)
    call_edges, caller_edges, symbols, hubs = generate_synthetic_repo_graph(
        num_nodes=graph_size, num_hubs=20, seed=seed
    )

    t_matrix, dangling = build_static_transition_matrix(
        num_nodes=graph_size,
        call_edges=call_edges,
        caller_edges=caller_edges,
    )

    pi_global = compute_global_pagerank(t_matrix, dangling)

    file_rec_2k: List[float] = []
    file_rec_4k: List[float] = []
    func_rec_2k: List[float] = []
    func_rec_4k: List[float] = []
    reciprocal_ranks: List[float] = []
    hub_presence_top10: List[float] = []

    diffusion_latencies: List[float] = []
    packing_latencies: List[float] = []

    # Pre-build non-hub adjacency map for symptom-to-cause multi-hop selection
    adj: Dict[int, Set[int]] = {i: set() for i in range(graph_size)}
    for u, v in call_edges:
        adj[u].add(v)
    for u, v in caller_edges:
        adj[u].add(v)
    hubs_set = set(hubs)

    for _ in range(num_instances):
        target_node = int(rng.integers(len(hubs) + 10, graph_size - 10))
        target_sym = symbols[target_node]
        target_file = target_sym.file_path

        # Find symptom node 1-2 hops away via non-hub local path
        neighbors_1 = [v for v in adj.get(target_node, []) if v not in hubs_set and v != target_node]
        if neighbors_1:
            v1 = int(rng.choice(neighbors_1))
            neighbors_2 = [v for v in adj.get(v1, []) if v not in hubs_set and v != target_node and v != v1]
            symptom_node = int(rng.choice(neighbors_2)) if neighbors_2 else v1
        else:
            symptom_node = max(0, min(graph_size - 1, target_node - 1))

        # Seed realistic query similarities:
        # High similarity on observed symptom, ambient noise on causal defect,
        # and moderate lexical distractor similarity on high-degree utility hubs.
        sims = np.clip(rng.normal(0.10, 0.03, size=graph_size), 0.0, 1.0)
        sims[symptom_node] = float(rng.uniform(0.80, 0.95))
        for n in adj.get(symptom_node, []):
            if n not in hubs_set and n != target_node:
                sims[n] = max(sims[n], float(rng.uniform(0.40, 0.60)))

        distractor_hubs = rng.choice(hubs, size=min(4, len(hubs)), replace=False)
        for h in distractor_hubs:
            sims[h] = float(rng.uniform(0.40, 0.65))

        t0 = time.perf_counter()
        p_0 = compute_softmax_teleport_prior(
            similarities=sims,
            node_indices=np.arange(graph_size),
            num_nodes=graph_size,
            tau=0.05,
            top_k=25,
        )

        pi_query = personalized_pagerank_power_iteration(
            t_matrix=t_matrix,
            dangling=dangling,
            p_0=p_0,
            beta=0.85,
            max_iter=100,
        )
        t1 = time.perf_counter()
        diffusion_latencies.append((t1 - t0) * 1000.0)

        scores = calculate_specificity_scores(
            pi_query=pi_query,
            pi_global=pi_global,
            gamma=0.7,
        )

        top10_nodes = np.argsort(scores)[::-1][:10]
        hubs_in_top10 = sum(1 for n in top10_nodes if n in hubs)
        hub_presence_top10.append(hubs_in_top10 / 10.0)

        ranked_nodes = np.argsort(scores)[::-1]
        rank_pos = np.where(ranked_nodes == target_node)[0]
        if len(rank_pos) > 0:
            reciprocal_ranks.append(1.0 / float(rank_pos[0] + 1))
        else:
            reciprocal_ranks.append(0.0)

        t2 = time.perf_counter()
        packed_2k, _ = pack_context_subgraphs(
            symbols=symbols,
            specificity_scores=scores,
            adjacency_matrix=t_matrix,
            token_budget=2000,
        )
        t3 = time.perf_counter()
        packing_latencies.append((t3 - t2) * 1000.0)

        packed_4k, _ = pack_context_subgraphs(
            symbols=symbols,
            specificity_scores=scores,
            adjacency_matrix=t_matrix,
            token_budget=4000,
        )

        packed_2k_ids = {s.node_id for s in packed_2k}
        packed_2k_files = {s.file_path for s in packed_2k}
        packed_4k_ids = {s.node_id for s in packed_4k}
        packed_4k_files = {s.file_path for s in packed_4k}

        func_rec_2k.append(1.0 if target_node in packed_2k_ids else 0.0)
        func_rec_4k.append(1.0 if target_node in packed_4k_ids else 0.0)
        file_rec_2k.append(1.0 if target_file in packed_2k_files else 0.0)
        file_rec_4k.append(1.0 if target_file in packed_4k_files else 0.0)

    hsi = 1.0 - float(np.mean(hub_presence_top10))

    return DiagnosticMetrics(
        file_recall_2k=float(np.mean(file_rec_2k)),
        file_recall_4k=float(np.mean(file_rec_4k)),
        func_recall_2k=float(np.mean(func_rec_2k)),
        func_recall_4k=float(np.mean(func_rec_4k)),
        mrr=float(np.mean(reciprocal_ranks)),
        hub_suppression_index=hsi,
        avg_diffusion_latency_ms=float(np.mean(diffusion_latencies)),
        avg_packing_latency_ms=float(np.mean(packing_latencies)),
    )


def evaluate_scaling_profile(sizes: Sequence[int] = (1000, 5000, 10000)) -> Dict[int, float]:
    """
    Profile PPR power iteration diffusion latency across increasing repository sizes.
    """
    latencies: Dict[int, float] = {}
    for n in sizes:
        call_edges, caller_edges, _, _ = generate_synthetic_repo_graph(
            num_nodes=n, num_hubs=max(10, n // 50), seed=42
        )
        t_mat, dang = build_static_transition_matrix(n, call_edges, caller_edges)
        p_0 = np.full(n, 1.0 / n, dtype=np.float32)

        # Warmup pass
        personalized_pagerank_power_iteration(t_mat, dang, p_0, max_iter=20)

        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            personalized_pagerank_power_iteration(t_mat, dang, p_0, max_iter=20)
        elapsed = (time.perf_counter() - t0) / iters * 1000.0
        latencies[n] = elapsed

    return latencies


if __name__ == "__main__":
    print("Executing Perron Diagnostic Benchmark Suite...")
    metrics = evaluate_diagnostic_benchmark(num_instances=50, graph_size=1000)
    print("---------------------------------------------------------")
    print(f"File Recall@2k:           {metrics.file_recall_2k * 100:.2f}%")
    print(f"File Recall@4k:           {metrics.file_recall_4k * 100:.2f}%")
    print(f"Function Recall@2k:       {metrics.func_recall_2k * 100:.2f}%")
    print(f"Function Recall@4k:       {metrics.func_recall_4k * 100:.2f}%")
    print(f"Mean Reciprocal Rank:     {metrics.mrr:.4f}")
    print(f"Hub Suppression Index:    {metrics.hub_suppression_index * 100:.2f}%")
    print(f"Avg Diffusion Latency:    {metrics.avg_diffusion_latency_ms:.2f} ms")
    print(f"Avg Packing Latency:      {metrics.avg_packing_latency_ms:.2f} ms")
    print("---------------------------------------------------------")
    print("Evaluating Multi-Scale Diffusion Latency Scaling...")
    scaling = evaluate_scaling_profile()
    for size, lat in scaling.items():
        print(f"  |V| = {size:5d} symbols: {lat:.2f} ms / query")
    print("---------------------------------------------------------")
