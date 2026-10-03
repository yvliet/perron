"""
Perron Graph Dataset Loaders.
Ingests zero-copy CSR multigraphs and symbol tables into Python data structures,
NetworkX DiGraphs, and PyG Data objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import pyarrow.parquet as pq


class InstanceGraph:
    """Encapsulation of an exported SWE-bench Lite instance code graph."""

    def __init__(
        self,
        instance_id: str,
        t_matrix: sp.csr_matrix,
        dangling: np.ndarray,
        nodes_table: Any,
        manifest: Dict[str, Any],
    ):
        self.instance_id = instance_id
        self.t_matrix = t_matrix
        self.dangling = dangling
        self.nodes_table = nodes_table
        self.manifest = manifest
        self._df = None

    @property
    def num_nodes(self) -> int:
        return self.t_matrix.shape[0]

    @property
    def num_edges(self) -> int:
        return self.t_matrix.nnz

    @property
    def nodes_df(self):
        if self._df is None:
            self._df = self.nodes_table.to_pandas()
        return self._df

    def top_hubs(self, k: int = 10) -> List[Tuple[str, int]]:
        """Return top-k highest in-degree symbols (qualified_name, in_degree)."""
        df = self.nodes_df
        sorted_df = df.sort_values(by="in_degree", ascending=False).head(k)
        return list(zip(sorted_df["qualified_name"], sorted_df["in_degree"]))

    def to_networkx(self, directed: bool = True):
        """Convert CSR transition matrix and node metadata into a NetworkX DiGraph."""
        import networkx as nx

        G = nx.DiGraph() if directed else nx.Graph()

        # Add nodes with attributes
        df = self.nodes_df
        for row in df.itertuples():
            G.add_node(
                row.id,
                kind=row.kind,
                qualified_name=row.qualified_name,
                file=row.file,
                start_line=row.start_line,
                end_line=row.end_line,
                in_degree=row.in_degree,
                out_degree=row.out_degree,
                is_test=row.is_test,
                is_hub_top25=row.is_hub_top25,
                is_hub_p99=row.is_hub_p99,
            )

        # Add edges from CSR matrix
        coo = self.t_matrix.tocoo()
        for u, v, w in zip(coo.row, coo.col, coo.data):
            G.add_edge(int(u), int(v), weight=float(w))

        return G

    def to_pyg(self):
        """Convert to PyTorch Geometric Data object (requires torch and torch_geometric)."""
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyG export requires 'torch' and 'torch_geometric' to be installed.")

        coo = self.t_matrix.tocoo()
        edge_index = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        edge_weight = torch.tensor(coo.data, dtype=torch.float)
        in_degrees = torch.tensor(self.nodes_df["in_degree"].values, dtype=torch.float)

        return Data(edge_index=edge_index, edge_attr=edge_weight, x=in_degrees.unsqueeze(-1), num_nodes=self.num_nodes)


def load_instance_graph(instance_path: Union[str, Path]) -> InstanceGraph:
    """Load an instance graph from a directory containing csr_*.npy and nodes.parquet."""
    p = Path(instance_path)
    if not p.is_dir():
        raise NotADirectoryError(f"Instance graph directory not found: {p}")

    manifest_p = p / "manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(f"Missing manifest.json in {p}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    instance_id = manifest.get("instance_id", p.name)
    data = np.load(p / "csr_data.npy")
    indices = np.load(p / "csr_indices.npy")
    indptr = np.load(p / "csr_indptr.npy")
    dangling = np.load(p / "csr_dangling.npy")

    num_nodes = len(indptr) - 1
    t_matrix = sp.csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))

    nodes_table = pq.read_table(p / "nodes.parquet")

    return InstanceGraph(
        instance_id=instance_id,
        t_matrix=t_matrix,
        dangling=dangling,
        nodes_table=nodes_table,
        manifest=manifest,
    )


def to_networkx(t_matrix: sp.csr_matrix, node_labels: Optional[List[str]] = None):
    """Utility function to convert a raw CSR matrix to NetworkX DiGraph."""
    import networkx as nx

    coo = t_matrix.tocoo()
    G = nx.DiGraph()
    num_nodes = t_matrix.shape[0]

    for i in range(num_nodes):
        label = node_labels[i] if node_labels and i < len(node_labels) else str(i)
        G.add_node(i, label=label)

    for u, v, w in zip(coo.row, coo.col, coo.data):
        G.add_edge(int(u), int(v), weight=float(w))

    return G


def from_networkx(G) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Convert a NetworkX DiGraph to row-stochastic CSR transition matrix and dangling vector."""
    import networkx as nx

    n = G.number_of_nodes()
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}

    rows: List[int] = []
    cols: List[int] = []
    weights: List[float] = []

    out_weights = np.zeros(n, dtype=np.float64)

    for u, v, data in G.edges(data=True):
        u_idx = mapping[u]
        v_idx = mapping[v]
        w = float(data.get("weight", 1.0))
        rows.append(u_idx)
        cols.append(v_idx)
        weights.append(w)
        out_weights[u_idx] += w

    # Normalize to row-stochastic
    norm_data: List[float] = []
    for r, w in zip(rows, weights):
        norm_data.append(w / out_weights[r] if out_weights[r] > 0 else 0.0)

    t_matrix = sp.csr_matrix((norm_data, (rows, cols)), shape=(n, n), dtype=np.float32)
    dangling = (out_weights == 0).astype(np.float32)

    return t_matrix, dangling
