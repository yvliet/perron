"""
Candidate Pool Construction for Gemma 4 Evaluation Protocol.

Implements pre-registered candidate pools adhering to benchmarks/protocol/gemma4_protocol_v1.yaml:
- S: BM25 top-10 seeds over the universe
- Filter F: node_kind in {function, method}, not test node, not in H25
- Arms:
  B0: BM25 baseline order
  B1: Perron Static baseline order
  A1: Rerank seeds (S only, M_t=10)
  A2: Rerank depth-matched BM25 (S + BM25 next ranks passing F up to M_t)
  A3: Rerank same-file matched (S + same-file symbols passing F ordered by BM25 desc, filled from A2 if needed)
  A4: Rerank graph 1-hop (S + top (M_t-10) of E_t ordered by adjacency score a(u))
  A5: Rerank null graph 1-hop (degree-preserving null replicate r_t = sha256(iid) % 100, filled from BM25 if needed)
  A6: Rerank Perron diffusion (S + Perron Static ranks passing F up to M_t)
- Downstream pools D0-D3 for edit-location mini-test
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import scipy.sparse as sp

from perron.graph.null_model import build_boolean_adjacency_matrix, rewire_relation_layers


@dataclass
class PoolDiagnostics:
    instance_id: str
    m_t: int
    a3_fill_count: int
    a5_fill_count: int
    jaccard_a4_a2: float
    jaccard_a4_a3: float
    graph_unique_count: int
    oracle_recalls: Dict[str, float]


def is_filter_f_passing(
    sym: Dict[str, Any],
    node_idx: int,
    is_test_mask: np.ndarray,
    hub_set: Set[int],
) -> bool:
    """Filter F rules: node kind in {function, method}, not test, not in H25."""
    kind = sym.get("symbol_type") or sym.get("kind") or "function"
    if kind not in ("function", "method"):
        return False
    if is_test_mask is not None and node_idx < len(is_test_mask) and is_test_mask[node_idx]:
        return False
    if node_idx in hub_set:
        return False
    return True


class CandidatePoolBuilder:
    """
    Constructs and verifies candidate pools for a repository.
    """

    def __init__(
        self,
        engine: Any,
        layers: Optional[Dict[str, Any]] = None,
        null_cache_limit: int = 100,
    ):
        self.engine = engine
        self.symbols = engine.symbols
        self.num_nodes = len(self.symbols)
        self.t_matrix = engine.t_matrix
        self.is_test_mask = engine.is_test_mask
        self.hub_set = engine.hub_set
        self.layers = layers
        self._null_matrix_cache: Dict[int, sp.csr_matrix] = {}
        self.null_cache_limit = null_cache_limit

        # Index symbols by file path for fast same-file lookup
        self.symbols_by_file: Dict[str, List[int]] = {}
        for idx, sym in enumerate(self.symbols):
            fpath = sym.get("file_path", "")
            if fpath:
                self.symbols_by_file.setdefault(fpath, []).append(idx)

    def compute_seeds(self, query_text: str) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """
        Compute top-10 BM25 seeds S over sanitized universe.
        Returns:
            seeds: List of 10 node indices
            bm25_scores: full array of BM25 scores
            p0: teleport prior over S (normalized BM25 scores over S)
        """
        scores = self.engine.bm25_index.query(query_text).astype(np.float64)
        masked_scores = np.copy(scores)
        if self.is_test_mask is not None and np.any(self.is_test_mask):
            masked_scores[self.is_test_mask] = -1e9

        # Rank stably by descending score, tie-break by identifier ascending
        sorted_indices = sorted(
            range(self.num_nodes),
            key=lambda i: (-masked_scores[i], self.symbols[i].get("identifier", "")),
        )
        seeds = sorted_indices[:10]

        # Prior p0 over S
        seed_scores = np.maximum(scores[seeds], 0.0)
        s_sum = float(np.sum(seed_scores))
        p0 = np.zeros(self.num_nodes, dtype=np.float64)
        if s_sum > 1e-8:
            p0[seeds] = seed_scores / s_sum
        else:
            p0[seeds] = 1.0 / len(seeds)

        return seeds, scores, p0

    def compute_graph_expansion(
        self,
        seeds: List[int],
        p0: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """
        Computes 1-hop expansion E_t under Perron operator T:
        a(u) = sum_{s in S} p0(s) * T[s, u]
        Filtered by Filter F, minus S.
        Ordered by a(u) descending, ties by identifier ascending.
        """
        seed_set = set(seeds)
        # Vector-matrix multiplication: a = p0 * T
        p0_sparse = sp.csr_matrix(p0)
        a_vec = np.asarray(p0_sparse.dot(self.t_matrix).todense()).flatten()

        candidates: List[Tuple[int, float]] = []
        for u in range(self.num_nodes):
            if u in seed_set:
                continue
            if a_vec[u] <= 0.0:
                continue
            if not is_filter_f_passing(self.symbols[u], u, self.is_test_mask, self.hub_set):
                continue
            candidates.append((u, float(a_vec[u])))

        # Order by a(u) descending, ties by identifier ascending
        candidates.sort(
            key=lambda item: (-item[1], self.symbols[item[0]].get("identifier", ""))
        )
        return candidates

    def get_null_transition_matrix(self, rep_idx: int) -> sp.csr_matrix:
        """Fetch or build null transition matrix for replicate rep_idx."""
        if rep_idx in self._null_matrix_cache:
            return self._null_matrix_cache[rep_idx]

        if not self.layers:
            return self.t_matrix

        rep_seed = 42 + rep_idx
        null_layers = rewire_relation_layers(
            num_nodes=self.num_nodes,
            call_edges=self.layers["call_edges"],
            inherit_edges=self.layers["inherit_edges"],
            import_edges=self.layers["import_edges"],
            seed=rep_seed,
        )
        null_adj = build_boolean_adjacency_matrix(
            num_nodes=self.num_nodes,
            call_edges=null_layers["call"],
            caller_edges=null_layers["caller"],
            inherit_edges=null_layers["inherit"],
            import_edges=null_layers["import"],
        )

        # Row-normalize to transition matrix
        row_sums = np.asarray(null_adj.sum(axis=1)).flatten()
        inv_sums = np.zeros_like(row_sums, dtype=np.float64)
        pos_mask = row_sums > 0
        inv_sums[pos_mask] = 1.0 / row_sums[pos_mask]
        t_null = null_adj.multiply(inv_sums[:, None]).tocsr()

        if len(self._null_matrix_cache) < self.null_cache_limit:
            self._null_matrix_cache[rep_idx] = t_null
        return t_null

    def compute_null_graph_expansion(
        self,
        seeds: List[int],
        p0: np.ndarray,
        rep_idx: int,
    ) -> List[Tuple[int, float]]:
        """Computes 1-hop expansion on degree-preserving null replicate."""
        t_null = self.get_null_transition_matrix(rep_idx)
        seed_set = set(seeds)
        p0_sparse = sp.csr_matrix(p0)
        a_vec = np.asarray(p0_sparse.dot(t_null).todense()).flatten()

        candidates: List[Tuple[int, float]] = []
        for u in range(self.num_nodes):
            if u in seed_set:
                continue
            if a_vec[u] <= 0.0:
                continue
            if not is_filter_f_passing(self.symbols[u], u, self.is_test_mask, self.hub_set):
                continue
            candidates.append((u, float(a_vec[u])))

        candidates.sort(
            key=lambda item: (-item[1], self.symbols[item[0]].get("identifier", ""))
        )
        return candidates

    def build_all_pools(
        self,
        query_text: str,
        instance_id: str,
        perron_scores: Optional[np.ndarray] = None,
        gold_symbol_ids: Optional[Set[int]] = None,
    ) -> Tuple[Dict[str, List[int]], PoolDiagnostics]:
        """
        Constructs all candidate pools A1-A6 and baselines B0, B1 for the task.
        """
        seeds, bm25_scores, p0 = self.compute_seeds(query_text)
        seed_set = set(seeds)

        # 1-hop graph expansion E_t
        e_graph = self.compute_graph_expansion(seeds, p0)
        e_graph_capped = e_graph[:40]
        m_t = 10 + min(len(e_graph), 40)
        m_t = max(10, min(50, m_t))

        # A1: S only (10)
        pool_a1 = list(seeds)

        # BM25 sorted candidate stream for controls (skipping S, filter_F required)
        bm25_stream = [
            i for i in sorted(
                range(self.num_nodes),
                key=lambda idx: (-bm25_scores[idx], self.symbols[idx].get("identifier", "")),
            )
            if i not in seed_set and is_filter_f_passing(self.symbols[i], i, self.is_test_mask, self.hub_set)
        ]

        # A2: S, then continue down BM25 ranking until M_t
        pool_a2 = list(seeds)
        needed_a2 = m_t - 10
        pool_a2.extend(bm25_stream[:needed_a2])

        # A3: S, then symbols in files containing any seed ordered by BM25 desc, up to M_t
        seed_files = {self.symbols[s].get("file_path", "") for s in seeds if self.symbols[s].get("file_path")}
        same_file_candidates = []
        for fpath in seed_files:
            for u in self.symbols_by_file.get(fpath, []):
                if u not in seed_set and is_filter_f_passing(self.symbols[u], u, self.is_test_mask, self.hub_set):
                    same_file_candidates.append(u)

        # Unique & sort by BM25 descending
        same_file_candidates = sorted(
            list(set(same_file_candidates)),
            key=lambda idx: (-bm25_scores[idx], self.symbols[idx].get("identifier", "")),
        )

        pool_a3 = list(seeds)
        needed_a3 = m_t - 10
        a3_from_same_file = same_file_candidates[:needed_a3]
        pool_a3.extend(a3_from_same_file)
        a3_fill_count = 0
        if len(pool_a3) < m_t:
            # Fill with next BM25 ranks
            a3_current_set = set(pool_a3)
            for u in bm25_stream:
                if u not in a3_current_set:
                    pool_a3.append(u)
                    a3_current_set.add(u)
                    a3_fill_count += 1
                    if len(pool_a3) >= m_t:
                        break

        # A4: S, then the first (M_t - 10) of E_t by adjacency score
        pool_a4 = list(seeds)
        pool_a4.extend([u for u, _ in e_graph_capped[:needed_a2]])

        # A5: As A4 on null replicate r_t = sha256(instance_id) mod 100
        r_t = int(hashlib.sha256(instance_id.encode("utf-8")).hexdigest(), 16) % 100
        e_null = self.compute_null_graph_expansion(seeds, p0, rep_idx=r_t)
        pool_a5 = list(seeds)
        pool_a5.extend([u for u, _ in e_null[:needed_a2]])
        a5_fill_count = 0
        if len(pool_a5) < m_t:
            a5_current_set = set(pool_a5)
            for u in bm25_stream:
                if u not in a5_current_set:
                    pool_a5.append(u)
                    a5_current_set.add(u)
                    a5_fill_count += 1
                    if len(pool_a5) >= m_t:
                        break

        # A6: S, then Perron Static ranking, skipping S and filter_F failures, until M_t
        pool_a6 = list(seeds)
        if perron_scores is not None:
            perron_stream = [
                i for i in np.argsort(-perron_scores, kind="stable")
                if i not in seed_set and is_filter_f_passing(self.symbols[i], i, self.is_test_mask, self.hub_set)
            ]
            pool_a6.extend(perron_stream[:needed_a2])
        else:
            pool_a6.extend(bm25_stream[:needed_a2])

        # B0: BM25 baseline (top-50)
        b0_ranked = sorted(
            range(self.num_nodes),
            key=lambda idx: (-bm25_scores[idx], self.symbols[idx].get("identifier", "")),
        )
        if self.is_test_mask is not None and np.any(self.is_test_mask):
            b0_ranked = [i for i in b0_ranked if not self.is_test_mask[i]]
        pool_b0 = b0_ranked[:50]

        # B1: Perron Static baseline (top-50)
        if perron_scores is not None:
            b1_scores = np.copy(perron_scores)
            if self.is_test_mask is not None and np.any(self.is_test_mask):
                b1_scores[self.is_test_mask] = -1e9
            pool_b1 = list(np.argsort(-b1_scores, kind="stable")[:50])
        else:
            pool_b1 = pool_b0

        pools = {
            "B0": pool_b0,
            "B1": pool_b1,
            "A1": pool_a1,
            "A2": pool_a2,
            "A3": pool_a3,
            "A4": pool_a4,
            "A5": pool_a5,
            "A6": pool_a6,
        }

        # Diagnostics
        set_a2 = set(pool_a2)
        set_a3 = set(pool_a3)
        set_a4 = set(pool_a4)

        jaccard_a4_a2 = len(set_a4.intersection(set_a2)) / max(1, len(set_a4.union(set_a2)))
        jaccard_a4_a3 = len(set_a4.intersection(set_a3)) / max(1, len(set_a4.union(set_a3)))
        graph_unique = len(set_a4 - set_a2 - set_a3)

        oracle_recalls: Dict[str, float] = {}
        if gold_symbol_ids is not None:
            for arm_name, p_list in pools.items():
                p_sym_ids = {self.symbols[i]["id"] for i in p_list if i < len(self.symbols)}
                oracle_recalls[arm_name] = float(len(p_sym_ids.intersection(gold_symbol_ids)) > 0)

        diag = PoolDiagnostics(
            instance_id=instance_id,
            m_t=m_t,
            a3_fill_count=a3_fill_count,
            a5_fill_count=a5_fill_count,
            jaccard_a4_a2=float(jaccard_a4_a2),
            jaccard_a4_a3=float(jaccard_a4_a3),
            graph_unique_count=graph_unique,
            oracle_recalls=oracle_recalls,
        )

        return pools, diag


def build_downstream_candidates(
    instance_id: str,
    top_candidates_per_arm: Dict[str, List[int]],
    chosen_control: str,  # "A2" or "A3"
    gold_symbol_ids: Set[int],
    symbols: List[Dict[str, Any]],
    seed: int = 20261108,
) -> Dict[str, List[int]]:
    """
    Builds downstream candidate sets D0, D1, D2, D3 (each exactly 5 candidates).
    D0: BM25 top-5
    D1: A4 reranked top-5
    D2: Chosen control (A2 or A3) reranked top-5
    D3: Oracle context (1 gold + 4 non-gold distractors from BM25 top candidates)
    """
    d0 = top_candidates_per_arm.get("B0", [])[:5]
    d1 = top_candidates_per_arm.get("A4", [])[:5]
    d2 = top_candidates_per_arm.get(chosen_control, [])[:5]

    # D3: Oracle context
    rng = np.random.RandomState(seed + (int(hashlib.sha256(instance_id.encode("utf-8")).hexdigest(), 16) % 10000))
    b0_list = top_candidates_per_arm.get("B0", [])

    # Identify gold candidate from universe
    gold_cand_idx = None
    for idx, sym in enumerate(symbols):
        if sym.get("id") in gold_symbol_ids:
            gold_cand_idx = idx
            break

    distractors = [i for i in b0_list if symbols[i].get("id") not in gold_symbol_ids]
    if len(distractors) < 4:
        for idx in range(len(symbols)):
            if symbols[idx].get("id") not in gold_symbol_ids and idx not in distractors:
                distractors.append(idx)
                if len(distractors) >= 4:
                    break
    chosen_distractors = distractors[:4]

    if gold_cand_idx is not None:
        insert_pos = rng.randint(0, 5)
        d3 = list(chosen_distractors)
        d3.insert(insert_pos, gold_cand_idx)
    else:
        d3 = (chosen_distractors + [0])[:5]

    return {
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "D3": d3,
    }
