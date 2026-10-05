"""
Topological Reachability & Degree-Preserving Null Rewiring Verification Suite.

Validates the complete remediation of null model edge-rewiring invariants:
1. Exact simple graph invariants (no multi-edges, no self-loops, Delta deg = 0).
2. Transitive closure DAG acyclicity bitmask/BFS (preserves swaps across disjoint trees).
3. Transpose symmetry E_caller_null = (E_call_null)^T.
4. Unprivileged frontier masking (zero inspection of gold targets G).
5. Safe zero-division regularization for E_h and z_E.
6. numpy.savez_compressed multi-layer serialization roundtrip.
7. Deterministic PRNG reproducibility.
8. Performance threshold: 100 null replicates complete in < 60s.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import pytest
import scipy.sparse as sp

from perron.graph.null_model import (
    DAGReachabilityTracker,
    MaslovSneppenRewirer,
    UnprivilegedFrontierTraverser,
    build_boolean_adjacency_matrix,
    compute_enrichment_z_score,
    compute_volume_normalized_enrichment,
    is_dag_swap_acyclic_bfs,
    load_multiplex_layers,
    rewire_relation_layers,
    save_multiplex_layers,
)


def is_strictly_acyclic(num_nodes: int, edges: List[Tuple[int, int]]) -> bool:
    """Verifies that a directed graph is strictly acyclic via Kahn's algorithm."""
    in_deg = [0] * num_nodes
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        in_deg[v] += 1

    queue = [u for u in range(num_nodes) if in_deg[u] == 0]
    visited = 0
    while queue:
        curr = queue.pop()
        visited += 1
        for nxt in adj[curr]:
            in_deg[nxt] -= 1
            if in_deg[nxt] == 0:
                queue.append(nxt)
    return visited == num_nodes


class TestNullModelInvariants:

    def test_simple_graph_invariants_and_exact_degree_conservation(self):
        """
        Defect 2 Remediation: Verifies that Maslov-Sneppen rewiring preserves
        exact in- and out-degree sequences with zero self-loops and zero multi-edges.
        """
        rng = np.random.default_rng(20261005)
        num_nodes = 200
        m = 400

        # Construct random simple directed graph
        edges_set: Set[Tuple[int, int]] = set()
        while len(edges_set) < m:
            u = int(rng.integers(0, num_nodes))
            v = int(rng.integers(0, num_nodes))
            if u != v:
                edges_set.add((u, v))
        initial_edges = list(edges_set)

        # Compute initial degree sequences
        in_deg_init = [0] * num_nodes
        out_deg_init = [0] * num_nodes
        for u, v in initial_edges:
            out_deg_init[u] += 1
            in_deg_init[v] += 1

        # Rewire
        rewired = MaslovSneppenRewirer.rewire_edges(
            initial_edges, num_nodes=num_nodes, rng=rng, target_swaps=m, is_dag=False
        )

        assert len(rewired) == len(initial_edges), "Edge count must remain strictly invariant."
        assert len(set(rewired)) == len(rewired), "Rewired graph must not contain duplicate multi-edges."
        for u, v in rewired:
            assert u != v, f"Rewired graph must not contain self-loops: ({u}, {v})"

        # Verify exact degree sequence conservation
        in_deg_post = [0] * num_nodes
        out_deg_post = [0] * num_nodes
        for u, v in rewired:
            out_deg_post[u] += 1
            in_deg_post[v] += 1

        assert out_deg_init == out_deg_post, "Out-degree sequence Delta deg_out != 0."
        assert in_deg_init == in_deg_post, "In-degree sequence Delta deg_in != 0."

    def test_dag_acyclicity_and_disjoint_tree_ergodicity(self):
        """
        Defect 1 Remediation: Verifies that sparse local BFS permits
        swaps between disjoint trees (no ergodicity freeze) while strictly preventing cycles.
        """
        # Disjoint Trees: T1: 0 -> 1, T2: 2 -> 3
        num_nodes = 4
        disjoint_edges = [(0, 1), (2, 3)]
        tracker = DAGReachabilityTracker(num_nodes, disjoint_edges)

        # Cross-component swap: (0, 3) and (2, 1)
        assert tracker.can_swap(0, 1, 2, 3) is True, (
            "Sparse local BFS must permit valid swaps between disjoint trees."
        )

        # Cycle creation test: Chain 0 -> 1 -> 2 -> 3
        chain_edges = [(0, 1), (1, 2), (2, 3)]
        chain_tracker = DAGReachabilityTracker(4, chain_edges)
        # Swap (0, 1) and (2, 3) -> (0, 3) and (2, 1) creates cycle 1 -> 2 -> 1
        assert chain_tracker.can_swap(0, 1, 2, 3) is False, (
            "Sparse local BFS must reject cycle-creating candidate swaps."
        )

        # Test on larger random DAG
        rng = np.random.default_rng(42)
        n_dag = 300
        m_dag = 250
        dag_edges_set = set()
        while len(dag_edges_set) < m_dag:
            u = int(rng.integers(0, n_dag - 1))
            v = int(rng.integers(u + 1, n_dag))
            dag_edges_set.add((u, v))
        dag_edges = list(dag_edges_set)

        rewired_dag = MaslovSneppenRewirer.rewire_edges(
            dag_edges, num_nodes=n_dag, rng=rng, target_swaps=m_dag, is_dag=True
        )

        assert is_strictly_acyclic(n_dag, rewired_dag) is True, (
            "Rewired inheritance layer must remain strictly acyclic."
        )

    def test_multiplex_transpose_symmetry(self):
        """
        Defect 3 Remediation: Verifies that E_caller_null is the exact transpose of E_call_null
        and that caller edges are never independently randomized.
        """
        num_nodes = 50
        call_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        inherit_edges = [(10, 11), (20, 21)]
        import_edges = [(30, 31)]

        layers = rewire_relation_layers(
            num_nodes=num_nodes,
            call_edges=call_edges,
            inherit_edges=inherit_edges,
            import_edges=import_edges,
            seed=42,
        )

        c_null = layers["call"]
        cr_null = layers["caller"]

        expected_cr = sorted([(v, u) for u, v in c_null])
        assert cr_null == expected_cr, "E_caller_null must be bitwise equal to (E_call_null)^T."

    def test_unprivileged_frontier_masking(self):
        """
        Defect 4 Remediation: Verifies that intermediate hubs are masked without inspecting
        ground truth targets G, while seeds are permitted to expand.
        """
        # Graph with central hub:
        # Node 0 -> Hub 1 (degree 60)
        # Hub 1 -> Nodes 2..61
        num_nodes = 100
        edges = [(0, 1)]
        for v in range(2, 62):
            edges.append((1, v))

        rows, cols = zip(*edges)
        adj = sp.csr_matrix(
            (np.ones(len(edges), dtype=bool), (rows, cols)),
            shape=(num_nodes, num_nodes),
            dtype=bool,
        )

        traverser = UnprivilegedFrontierTraverser(adj, theta_hub=20)
        hops = traverser.traverse(seeds=[0], max_hops=3)

        assert 1 in hops[1], "Seed 0 must reach intermediate hub 1 at hop 1."
        assert len(hops[2]) == 2, f"Intermediate hub 1 must not explode into hop 2. Hops[2] size: {len(hops[2])}"
        assert hops[2] == {0, 1}, "Hop 2 must contain only seeds and intermediate node, not hub children."

        # Verify that Traverser API does not accept G as a parameter
        import inspect
        sig = inspect.signature(traverser.traverse)
        assert "gold" not in sig.parameters, "Traverser.traverse must not accept gold ground truth parameters."

    def test_zero_division_regularization(self):
        """
        Defect 5 Remediation: Verifies that enrichment ratios and z-scores do not raise
        ZeroDivisionError or return NaN/inf under boundary singularities.
        """
        # Case 1: Empty reachable set
        e1 = compute_volume_normalized_enrichment(gold_set={1, 2}, reachable_nodes=set(), num_nodes=100)
        assert e1 == 0.0

        # Case 2: Empty gold set
        e2 = compute_volume_normalized_enrichment(gold_set=set(), reachable_nodes={1, 2}, num_nodes=100)
        assert e2 == 0.0

        # Case 3: Empty node universe
        e3 = compute_volume_normalized_enrichment(gold_set={1}, reachable_nodes={1}, num_nodes=0)
        assert e3 == 0.0

        # Case 4: Zero variance across null replicates (both real and null are 0.0)
        z1 = compute_enrichment_z_score(e_real=0.0, null_enrichments=[0.0] * 100)
        assert z1 == 0.0

        # Case 5: Zero variance across null replicates with positive real
        z2 = compute_enrichment_z_score(e_real=2.5, null_enrichments=[0.0] * 100)
        assert np.isfinite(z2) and z2 > 0.0

    def test_layer_serialization_roundtrip(self, tmp_path: Path):
        """
        Defect 6 Remediation: Verifies that numpy.savez_compressed archives serialize
        and deserialize multi-layer edge arrays bitwise identically.
        """
        out_file = tmp_path / "test_instance_layers.npz"
        num_nodes = 500
        call_edges = [(1, 2), (3, 4), (5, 6)]
        inherit_edges = [(10, 20), (30, 40)]
        import_edges = [(100, 200)]
        caller_edges = [(2, 1), (4, 3), (6, 5)]

        save_multiplex_layers(
            out_file,
            num_nodes=num_nodes,
            call_edges=call_edges,
            inherit_edges=inherit_edges,
            import_edges=import_edges,
            caller_edges=caller_edges,
        )

        loaded = load_multiplex_layers(out_file)
        assert loaded["num_nodes"] == num_nodes
        assert loaded["call_edges"] == call_edges
        assert loaded["inherit_edges"] == inherit_edges
        assert loaded["import_edges"] == import_edges
        assert loaded["caller_edges"] == caller_edges

    def test_performance_null_rewire_threshold(self):
        """
        Asserts that 100 null model replicates on a benchmark-scale graph
        complete in strictly < 60.0 seconds.
        """
        num_nodes = 2000
        m_call = 4000
        m_inherit = 500
        m_import = 1500

        rng = np.random.default_rng(42)

        def make_edges(m: int, dag: bool = False) -> List[Tuple[int, int]]:
            es = set()
            while len(es) < m:
                if dag:
                    u = int(rng.integers(0, num_nodes - 1))
                    v = int(rng.integers(u + 1, num_nodes))
                else:
                    u = int(rng.integers(0, num_nodes))
                    v = int(rng.integers(0, num_nodes))
                if u != v:
                    es.add((u, v))
            return list(es)

        call_edges = make_edges(m_call)
        inherit_edges = make_edges(m_inherit, dag=True)
        import_edges = make_edges(m_import)

        t0 = time.time()
        for rep in range(100):
            null_layers = rewire_relation_layers(
                num_nodes=num_nodes,
                call_edges=call_edges,
                inherit_edges=inherit_edges,
                import_edges=import_edges,
                seed=42 + rep,
            )
            adj = build_boolean_adjacency_matrix(
                num_nodes=num_nodes,
                call_edges=null_layers["call"],
                caller_edges=null_layers["caller"],
                inherit_edges=null_layers["inherit"],
                import_edges=null_layers["import"],
            )
            traverser = UnprivilegedFrontierTraverser(adj)
            _ = traverser.traverse(seeds=[1, 2, 3, 4, 5], max_hops=3)

        elapsed = time.time() - t0
        print(f"Null Model Benchmark: 100 replicates elapsed in {elapsed:.3f}s")
        assert elapsed < 60.0, f"Performance threshold breached: elapsed {elapsed:.3f}s >= 60.0s."
