"""
Toy graph verification tests across all 14 retrieval baselines.

Verifies:
8-node canonical toy graph:
- S (Seed, high BM25 matching query text)
- T (Target, 1 call-hop from S, low in-degree)
- H (Hub, in-degree 6, one edge from S, edges from fillers)
- 5 Filler nodes (Fillers with secondary connections)

Query text names S only.
Proves whether each baseline can retrieve unmentioned 1-call-hop target T in its top-3.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine


@pytest.fixture
def toy_graph_engine() -> tuple[RepositoryRetrievalEngine, str, int]:
    """
    Construct the canonical 8-node toy graph:
    0: Seed S ('process_order')
    1: Filler F1 ('alpha')
    2: Filler F2 ('beta')
    3: Filler F3 ('gamma')
    4: Filler F4 ('delta')
    5: Hub H ('logging_util', in-degree 6)
    6: Target T ('calculate_tax', in-degree 1 from S)
    7: Filler F5 ('epsilon')
    """
    num_nodes = 8
    # Edges:
    # S(0) -> T(6), H(5)
    # F1(1) -> H(5), F2(2)
    # F2(2) -> H(5), F3(3)
    # F4(4) -> H(5), F3(3)
    # F5(7) -> H(5)
    # T(6) -> H(5)
    # Total incoming to H(5): from 0, 1, 2, 4, 6, 7 = 6 edges (in-degree 6).
    # Incoming to T(6): from 0 = 1 edge (low degree).
    rows = [0, 0, 1, 1, 2, 2, 4, 4, 6, 7]
    cols = [6, 5, 5, 2, 5, 3, 5, 3, 5, 5]
    data = [1.0] * len(rows)

    adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    row_sums = np.asarray(adj.sum(axis=1)).flatten()
    inv_sums = np.zeros_like(row_sums, dtype=np.float64)
    mask = row_sums > 0
    inv_sums[mask] = 1.0 / row_sums[mask]
    t_matrix = adj.multiply(inv_sums[:, None]).tocsr()

    symbols = [
        {"id": 0, "identifier": "process_order", "file_path": "order.py"},
        {"id": 1, "identifier": "alpha", "file_path": "alpha.py"},
        {"id": 2, "identifier": "beta", "file_path": "beta.py"},
        {"id": 3, "identifier": "gamma", "file_path": "gamma.py"},
        {"id": 4, "identifier": "delta", "file_path": "delta.py"},
        {"id": 5, "identifier": "logging_util", "file_path": "logger.py"},
        {"id": 6, "identifier": "calculate_tax", "file_path": "tax.py"},
        {"id": 7, "identifier": "epsilon", "file_path": "epsilon.py"},
    ]

    engine = RepositoryRetrievalEngine(t_matrix=t_matrix, symbols=symbols)
    query = "process_order"
    target_id = 6
    return engine, query, target_id


def test_toy_graph_structural_properties(toy_graph_engine) -> None:
    engine, _, target_id = toy_graph_engine
    assert engine.num_nodes == 8
    # Hub H(5) has in-degree 6
    assert engine.in_degrees[5] == 6.0
    # Target T(6) has low in-degree (1.0)
    assert engine.in_degrees[target_id] == 1.0


@pytest.mark.parametrize(
    "method_name",
    [
        "bm25_1hop",
        "standard_ppr",
        "aider",
        "hub_blocklist",
        "deg_matrix",
        "degree_normalized",
        "hipporag",
        "query_reweighted_ppr",
        "perron_static",
        "perron",
        "oracle",
    ],
)
def test_graph_diffusion_baselines_find_target_in_top_3(
    toy_graph_engine, method_name: str
) -> None:
    """All 11 graph-aware diffusion baselines successfully retrieve T in top 3."""
    engine, query, target_id = toy_graph_engine
    ranking = engine.retrieve_by_name(
        method_name=method_name,
        query_text=query,
        gold_symbol_ids=[target_id],
    )
    top_3 = ranking[:3]
    assert target_id in top_3, (
        f"Baseline {method_name} failed to place target {target_id} in top 3: {top_3}"
    )


@pytest.mark.parametrize(
    "method_name,expected_reason",
    [
        ("bm25", "BM25 is purely lexical and cannot traverse graph edges to unmentioned symbols"),
        ("dense", "Dense retrieval is strictly text-overlap and cannot traverse graph edges"),
        ("prior_only", "Prior-only control uses p0 without diffusion, giving target zero probability mass"),
    ],
)
def test_purely_lexical_baselines_intrinsic_failure(
    toy_graph_engine, method_name: str, expected_reason: str
) -> None:
    """
    Purely lexical / prior-only baselines fail to retrieve T in top 3 when query
    names S only, because by mathematical design they lack graph edge diffusion.
    """
    engine, query, target_id = toy_graph_engine
    ranking = engine.retrieve_by_name(
        method_name=method_name,
        query_text=query,
        gold_symbol_ids=[target_id],
    )
    top_3 = ranking[:3]
    # Documented intrinsic limitation: T has 0 lexical overlap and ranks below top 3
    assert target_id not in top_3, (
        f"Unexpected hit for purely lexical baseline {method_name}; reason: {expected_reason}"
    )
