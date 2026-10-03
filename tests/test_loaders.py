"""
Unit tests for Perron Graph Loaders.
Verifies loading instance graphs, NetworkX conversions, and top-hub extractions.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from perron.loaders import from_networkx, load_instance_graph, to_networkx

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_instance_graph():
    """Verify loading an exported instance graph from data_release/graphs/."""
    graphs_dir = REPO_ROOT / "data_release" / "graphs"
    assert graphs_dir.exists(), "Requires data_release/graphs/"

    # Pick first available instance
    inst_dirs = [d for d in graphs_dir.iterdir() if d.is_dir()]
    assert len(inst_dirs) > 0, "No instance graphs found"

    sample_dir = inst_dirs[0]
    graph = load_instance_graph(sample_dir)

    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    assert len(graph.dangling) == graph.num_nodes
    assert graph.nodes_df is not None
    assert "qualified_name" in graph.nodes_df.columns
    assert "in_degree" in graph.nodes_df.columns

    # Test top_hubs
    hubs = graph.top_hubs(k=5)
    assert len(hubs) <= 5
    for name, deg in hubs:
        assert isinstance(name, str)
        assert isinstance(deg, (int, np.integer))

    # Test NetworkX conversion
    nx_graph = graph.to_networkx()
    assert nx_graph.number_of_nodes() == graph.num_nodes
    assert nx_graph.number_of_edges() == graph.num_edges


def test_networkx_roundtrip():
    """Verify from_networkx creates a valid row-stochastic transition matrix."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2.0)
    G.add_edge(0, 2, weight=2.0)
    G.add_edge(1, 2, weight=1.0)
    # node 2 is a sink / dangling node

    t_matrix, dangling = from_networkx(G)
    assert t_matrix.shape == (3, 3)

    # Row 0 sum must be 1.0 (0.5 to node 1, 0.5 to node 2)
    assert abs(t_matrix[0].sum() - 1.0) < 1e-5
    # Row 1 sum must be 1.0 (1.0 to node 2)
    assert abs(t_matrix[1].sum() - 1.0) < 1e-5
    # Row 2 is dangling
    assert t_matrix[2].sum() == 0.0
    assert dangling[2] == 1.0
    assert dangling[0] == 0.0
