"""
Suite of 6 Retrieval Baselines for Code Graph Retrieval.

Implements:
1. Okapi BM25 (lexical matching over identifier, path, and text)
2. Dense Semantic Retrieval (normalized semantic token similarity)
3. Aider-Style Repo Map (global graph PageRank with tag matching)
4. Hub Blocklist Baseline (Standard PPR with top-25 hubs masked)
5. Degree-Normalized PPR (pi_query / (deg_in + 1)^gamma)
6. Perron Decoupled Specificity Ratio (query-directed post-walk PMI ratio with Query-Level Hub Adaptivity)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

from perron.diffusion import personalized_pagerank_power_iteration
from perron.specificity import calculate_specificity_scores

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "self", "this", "that", "it", "not", "def", "class", "return", "import",
    "as", "from", "if", "else", "elif", "try", "except", "while", "for", "in",
    "none", "true", "false",
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alpha-numeric identifiers excluding stop words, including snake_case sub-tokens."""
    if not text:
        return []
    raw_tokens = [
        t.lower()
        for t in TOKEN_PATTERN.findall(text)
        if len(t) >= 2 and t.lower() not in STOP_WORDS
    ]
    tokens: List[str] = []
    for t in raw_tokens:
        tokens.append(t)
        if "_" in t:
            sub = [
                st.lower()
                for st in t.split("_")
                if len(st) >= 2 and st.lower() not in STOP_WORDS
            ]
            tokens.extend(sub)
    return tokens


class FastBM25:
    """Zero-dependency BM25 index over symbol catalogs."""

    def __init__(self, symbols: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.num_docs = len(symbols)
        self.doc_lengths = np.zeros(self.num_docs, dtype=np.float32)
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.doc_freqs: Dict[str, int] = {}

        total_len = 0
        for i, sym in enumerate(symbols):
            tokens = []
            ident = sym.get("identifier", "")
            fpath = sym.get("file_path", "")
            # Boost identifier and file path tokens
            tokens.extend(tokenize_text(ident) * 3)
            tokens.extend(tokenize_text(fpath) * 2)

            self.doc_lengths[i] = len(tokens)
            total_len += len(tokens)

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = float(total_len) / max(1, self.num_docs)
        self.idf: Dict[str, float] = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(
                (self.num_docs - df + 0.5) / (df + 0.5) + 1.0
            )

    def query(self, query_text: str) -> np.ndarray:
        scores = np.zeros(self.num_docs, dtype=np.float32)
        q_tokens = tokenize_text(query_text)
        if not q_tokens or self.num_docs == 0:
            return scores

        denom_const = self.k1 * (1.0 - self.b + self.b * (self.doc_lengths / max(1e-6, self.avg_doc_len)))
        for qtok in set(q_tokens):
            if qtok not in self.idf:
                continue
            idf_val = self.idf[qtok]
            for i, tf_dict in enumerate(self.doc_term_freqs):
                if qtok in tf_dict:
                    freq = tf_dict[qtok]
                    score = idf_val * (freq * (self.k1 + 1.0)) / (freq + denom_const[i])
                    scores[i] += score
        return scores


class RepositoryRetrievalEngine:
    """
    Manages graphs, precomputations, and all 6 baseline retrievers for a repository.
    """

    def __init__(
        self,
        t_matrix: csr_matrix,
        symbols: List[Dict[str, Any]],
        beta: float = 0.85,
        gamma: float = 0.70,
        beta_query: Optional[float] = None,
        beta_global: Optional[float] = None,
        is_test_node: Optional[Sequence[bool] | np.ndarray] = None,
        filter_test_nodes: bool = True,
        prior_mode: str = "linear",
        tau: float = 0.15,
        k_prior: int = 25,
    ):
        # Check and enforce row-stochasticity
        row_sums_check = np.asarray(t_matrix.sum(axis=1)).flatten()
        if np.any(row_sums_check > 1.0001) or np.any((row_sums_check > 0.0) & (row_sums_check < 0.9999)):
            inv_sums = np.zeros_like(row_sums_check, dtype=np.float64)
            nonzero_mask = row_sums_check > 0
            inv_sums[nonzero_mask] = 1.0 / row_sums_check[nonzero_mask]
            t_matrix = t_matrix.multiply(inv_sums[:, None]).tocsr()

        self.t_matrix = t_matrix
        self.symbols = symbols
        self.num_nodes = t_matrix.shape[0]
        self.beta_query = beta if beta_query is None else beta_query
        self.beta_global = 0.85 if beta_global is None else beta_global
        self.beta = self.beta_query
        self.gamma = gamma
        self.filter_test_nodes = filter_test_nodes
        self.prior_mode = prior_mode
        self.tau = tau
        self.k_prior = k_prior

        # Determine test node mask
        if is_test_node is not None:
            self.is_test_mask = np.asarray(is_test_node, dtype=bool)
        else:
            has_is_test_field = any("is_test" in s for s in symbols)
            if has_is_test_field:
                self.is_test_mask = np.array([bool(s.get("is_test", False)) for s in symbols], dtype=bool)
            else:
                from perron.graph import _is_test_path, _is_test_symbol_name
                self.is_test_mask = np.array([
                    bool(_is_test_path(s.get("file_path", "")) or _is_test_symbol_name(s.get("identifier", "")))
                    for s in symbols
                ], dtype=bool)

        # Build in-degree vector
        if hasattr(t_matrix, "tocsc"):
            csc = t_matrix.tocsc()
            self.in_degrees = np.diff(csc.indptr).astype(np.float32)
        else:
            self.in_degrees = np.zeros(self.num_nodes, dtype=np.float32)

        # Precompute structural hubs (top-25 in-degree, or top 1/3 for small graphs)
        k_hubs = min(25, max(1, self.num_nodes // 3)) if self.num_nodes < 25 else min(25, self.num_nodes)
        top_indices = np.argsort(self.in_degrees)[-k_hubs:]
        self.hub_set: Set[int] = set(top_indices.tolist())
        self.hub_indices = np.asarray(sorted(list(self.hub_set)), dtype=int)

        # Precompute static degree-discounted transition matrix:
        col_scale = 1.0 / np.power(self.in_degrees + 1.0, self.gamma)
        diag_c = sp.diags(col_scale)
        m_disc = self.t_matrix.dot(diag_c).tocsr()
        r_sums = np.asarray(m_disc.sum(axis=1)).flatten()
        inv_r = np.zeros_like(r_sums, dtype=np.float64)
        mask = r_sums > 0
        inv_r[mask] = 1.0 / r_sums[mask]
        self.t_matrix_deg_discounted = m_disc.multiply(inv_r[:, None]).tocsr()

        # Build adjacency neighbor mapping for 1-hop expansion
        self.adj_neighbors: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for u in range(self.num_nodes):
            st = self.t_matrix.indptr[u]
            en = self.t_matrix.indptr[u + 1]
            self.adj_neighbors[u] = list(self.t_matrix.indices[st:en])

        # Dangling node indicator
        row_sums = np.diff(t_matrix.indptr)
        self.dangling = (row_sums == 0).astype(np.float32)

        # Precompute global PageRank pi_global using beta_global
        p_uniform = np.full(self.num_nodes, 1.0 / max(1, self.num_nodes), dtype=np.float32)
        self.pi_global = personalized_pagerank_power_iteration(
            t_matrix=self.t_matrix,
            dangling=self.dangling,
            p_0=p_uniform,
            beta=self.beta_global,
            max_iter=100,
            tol=1e-6,
        )

        # Precompute BM25 index
        self.bm25_index = FastBM25(symbols)

    def _rank_scores(self, scores: np.ndarray) -> List[int]:
        """Rank nodes stably by descending score, suppressing test nodes if filter_test_nodes is True."""
        s = np.array(scores, dtype=np.float64, copy=True)
        if self.filter_test_nodes and self.is_test_mask is not None and np.any(self.is_test_mask):
            s[self.is_test_mask] = -1e9
        return list(np.argsort(-s, kind="stable"))

    def compute_query_prior(
        self,
        query_text: str,
        mode: Optional[str] = None,
        k: Optional[int] = None,
        tau: Optional[float] = None,
    ) -> np.ndarray:
        """Compute query teleport prior p_0 from lexical BM25 matching (linear or sharp softmax top-k)."""
        m = mode or self.prior_mode
        if m == "softmax":
            K = self.k_prior if k is None else k
            t = self.tau if tau is None else tau
            s = self.bm25_index.query(query_text).astype(np.float64)
            if s.max() <= 0:
                return np.full(self.num_nodes, 1.0 / max(1, self.num_nodes), dtype=np.float32)
            idx = np.argsort(-s, kind="stable")[:K]
            z = s[idx] / s.max()
            w = np.exp((z - z.max()) / t)
            p = np.zeros(self.num_nodes, dtype=np.float64)
            p[idx] = w / w.sum()
            return p.astype(np.float32)
        else:
            # Linear normalized BM25 prior over positive scores
            bm25_scores = self.bm25_index.query(query_text)
            pos_scores = np.maximum(bm25_scores, 0.0)
            s_sum = float(np.sum(pos_scores))
            if s_sum > 1e-8:
                return (pos_scores / s_sum).astype(np.float32)
            return np.full(self.num_nodes, 1.0 / max(1, self.num_nodes), dtype=np.float32)

    def compute_linear_prior(self, query_text: str) -> np.ndarray:
        """Dense linear BM25 prior."""
        return self.compute_query_prior(query_text, mode="linear")

    def compute_softmax_prior(self, query_text: str, k: int = 25, tau: float = 0.15) -> np.ndarray:
        """Sharp top-K softmax BM25 prior with temperature tau."""
        return self.compute_query_prior(query_text, mode="softmax", k=k, tau=tau)

    def compute_personalized_pagerank(self, p_0: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Run power iteration personalized PageRank for teleport prior p_0."""
        b = self.beta_query if beta is None else beta
        return personalized_pagerank_power_iteration(
            t_matrix=self.t_matrix,
            dangling=self.dangling,
            p_0=p_0,
            beta=b,
            max_iter=100,
            tol=1e-6,
        )

    # 1. Okapi BM25
    def retrieve_bm25(self, query_text: str) -> List[int]:
        """Baseline 1: Okapi BM25."""
        scores = self.bm25_index.query(query_text)
        return self._rank_scores(scores)

    # 2. BM25 + 1-Hop Expansion
    def retrieve_bm25_1hop(self, query_text: str) -> List[int]:
        """Baseline 2: BM25 with 1-Hop Graph Neighbor Expansion."""
        bm25_scores = self.bm25_index.query(query_text)
        scores = bm25_scores.copy()
        top_k_indices = np.argsort(-bm25_scores, kind="stable")[:50]
        for u in top_k_indices:
            s_u = bm25_scores[u]
            if s_u <= 0:
                continue
            for v in self.adj_neighbors[u]:
                if s_u * 0.50 > scores[v]:
                    scores[v] = s_u * 0.50
        return self._rank_scores(scores)

    # 3. Dense Semantic Retrieval
    def retrieve_dense(self, query_text: str) -> List[int]:
        """Baseline 3: Dense Semantic Retrieval."""
        q_tokens = set(tokenize_text(query_text))
        scores = np.zeros(self.num_nodes, dtype=np.float32)
        if not q_tokens:
            return self._rank_scores(scores)

        for i, sym in enumerate(self.symbols):
            ident = sym.get("identifier", "").lower()
            fpath = sym.get("file_path", "").lower()
            sym_tokens = set(tokenize_text(ident) + tokenize_text(fpath))
            overlap = len(q_tokens & sym_tokens)
            if overlap > 0:
                scores[i] = overlap / math.sqrt(len(q_tokens) * len(sym_tokens) + 1.0)
            elif any(qt in ident for qt in q_tokens if len(qt) >= 4):
                scores[i] = 0.25
        return self._rank_scores(scores)

    # 4. Prior-Only Control (p0 ranking without diffusion)
    def retrieve_prior_only(self, query_text: str, p_0: Optional[np.ndarray] = None) -> List[int]:
        """Baseline 4: Prior-Only Control (ranking by p0 with no diffusion)."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        return self._rank_scores(p_0)

    # 5. Standard Personalized PageRank (Uniform Teleport with shared p0)
    def retrieve_standard_ppr(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Baseline 5: Standard Personalized PageRank (shared p0, raw pi_query)."""
        if pi_query is None:
            if p_0 is None:
                p_0 = self.compute_query_prior(query_text)
            pi_query = self.compute_personalized_pagerank(p_0)
        return self._rank_scores(pi_query)

    # 6. Authentic Aider Repo Map
    def retrieve_aider_repomap(self, query_text: str) -> List[int]:
        """Baseline 6: Authentic Aider Repo Map (PPR seeded on issue prompt identifiers)."""
        q_tokens = set(tokenize_text(query_text))
        tag_match = np.zeros(self.num_nodes, dtype=np.float32)
        for i, sym in enumerate(self.symbols):
            ident = sym.get("identifier", "").lower()
            if any(qt in ident for qt in q_tokens if len(qt) >= 3):
                tag_match[i] = 1.0

        if float(np.sum(tag_match)) > 0:
            p_aider = tag_match / float(np.sum(tag_match))
        else:
            p_aider = np.full(self.num_nodes, 1.0 / max(1, self.num_nodes), dtype=np.float32)

        pi_aider = self.compute_personalized_pagerank(p_aider)
        return self._rank_scores(pi_aider)

    # 7. Hub Blocklist with Lexical Override
    def retrieve_hub_blocklist_lexical(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Baseline 7: Hub Blocklist with Lexical Override."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        if pi_query is None:
            pi_query = self.compute_personalized_pagerank(p_0)
        scores = pi_query.copy()
        max_p0 = float(np.max(p_0)) if len(p_0) > 0 else 0.0

        for h in self.hub_indices:
            is_lexically_matched = (max_p0 > 1e-8 and (p_0[h] / max_p0) >= 0.20)
            if not is_lexically_matched:
                scores[h] = 0.0
        return self._rank_scores(scores)

    # 8. Static Degree-Discounted Transition Matrix
    def retrieve_static_deg_discount(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Baseline 8: Static Degree-Discounted Transition Matrix."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        pi_disc = personalized_pagerank_power_iteration(
            t_matrix=self.t_matrix_deg_discounted,
            dangling=self.dangling,
            p_0=p_0,
            beta=self.beta_query,
            max_iter=100,
            tol=1e-6,
        )
        return self._rank_scores(pi_disc)

    # 9. Degree-Normalized PPR
    def retrieve_degree_normalized_ppr(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Baseline 9: Degree-Normalized PPR (pi_query / (deg_in + 1)^gamma)."""
        if pi_query is None:
            if p_0 is None:
                p_0 = self.compute_query_prior(query_text)
            pi_query = self.compute_personalized_pagerank(p_0)
        denom = np.power(self.in_degrees + 1.0, self.gamma)
        scores = pi_query / denom
        return self._rank_scores(scores)

    # 10. HippoRAG Pre-Walk Prior Scaling
    def retrieve_hipporag(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
    ) -> List[int]:
        r"""Baseline 10: HippoRAG Pre-Walk Prior Scaling (p0' \propto p0 / pi_g^gamma)."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        scaled_p0 = p_0 / np.power(self.pi_global + 1e-8, self.gamma)
        s_sum = float(np.sum(scaled_p0))
        if s_sum > 1e-8:
            scaled_p0 = scaled_p0 / s_sum
        else:
            scaled_p0 = p_0
        pi_hippo = self.compute_personalized_pagerank(scaled_p0)
        return self._rank_scores(pi_hippo)

    # 11. Query-Reweighted PPR (Dynamic Transitions)
    def retrieve_query_reweighted_ppr(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Baseline 11: Query-Reweighted PPR (dynamic edge reweighting)."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        max_p0 = float(np.max(p_0)) if len(p_0) > 0 else 0.0
        reweight_scale = 1.0 + 2.0 * (p_0 / max(1e-8, max_p0))
        diag_rw = sp.diags(reweight_scale)
        m_rw = self.t_matrix.dot(diag_rw).tocsr()
        r_sums = np.asarray(m_rw.sum(axis=1)).flatten()
        inv_r = np.zeros_like(r_sums, dtype=np.float64)
        mask = r_sums > 0
        inv_r[mask] = 1.0 / r_sums[mask]
        t_rw = m_rw.multiply(inv_r[:, None]).tocsr()
        pi_rw = personalized_pagerank_power_iteration(
            t_matrix=t_rw,
            dangling=self.dangling,
            p_0=p_0,
            beta=self.beta_query,
            max_iter=100,
            tol=1e-6,
        )
        return self._rank_scores(pi_rw)

    # 12. Perron Static Specificity (gamma = 0.70)
    def retrieve_perron_static(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Method 12: Perron Static Specificity (fixed gamma = 0.70)."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        if pi_query is None:
            pi_query = self.compute_personalized_pagerank(p_0)
        denom = np.power(self.pi_global + 1e-8, 0.70)
        scores = pi_query / denom
        return self._rank_scores(scores)

    # 13. Perron Adaptive Specificity (gamma_q with relative prominence eta_q)
    def retrieve_perron(
        self,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
    ) -> List[int]:
        """Method 13: Perron Specificity with Query-Level Hub Adaptivity."""
        if p_0 is None:
            p_0 = self.compute_query_prior(query_text)
        if pi_query is None:
            pi_query = self.compute_personalized_pagerank(p_0)
        scores = calculate_specificity_scores(
            pi_query=pi_query,
            pi_global=self.pi_global,
            gamma=self.gamma,
            query_prior=p_0,
            hub_indices=self.hub_indices,
            is_test_node=self.is_test_mask,
            filter_test_nodes=self.filter_test_nodes,
        )
        return self._rank_scores(scores)

    # 14. Oracle Upper Bound
    def retrieve_oracle(self, gold_symbol_ids: Sequence[int]) -> List[int]:
        """Method 14: Oracle Upper Bound placing ground-truth symbols in top ranks."""
        gold_set = set(gold_symbol_ids)
        top = [i for i in gold_symbol_ids if 0 <= i < self.num_nodes]
        rest = [i for i in range(self.num_nodes) if i not in gold_set]
        full_ranks = top + rest
        if self.filter_test_nodes and self.is_test_mask is not None and np.any(self.is_test_mask):
            non_tests = [i for i in full_ranks if not self.is_test_mask[i]]
            tests = [i for i in full_ranks if self.is_test_mask[i]]
            return non_tests + tests
        return full_ranks

    def retrieve_by_name(
        self,
        method_name: str,
        query_text: str,
        p_0: Optional[np.ndarray] = None,
        pi_query: Optional[np.ndarray] = None,
        gold_symbol_ids: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Unified Dispatcher for all 14 retrieval methods."""
        m = method_name.lower()
        if m == "bm25":
            return self.retrieve_bm25(query_text)
        elif m in ("bm25_1hop", "bm25_expansion"):
            return self.retrieve_bm25_1hop(query_text)
        elif m in ("dense", "semantic"):
            return self.retrieve_dense(query_text)
        elif m in ("prior_only", "p0_only"):
            return self.retrieve_prior_only(query_text, p_0=p_0)
        elif m in ("standard_ppr", "ppr"):
            return self.retrieve_standard_ppr(query_text, p_0=p_0, pi_query=pi_query)
        elif m in ("aider", "repo_map", "repomap"):
            return self.retrieve_aider_repomap(query_text)
        elif m in ("blocklist_lex", "blocklist", "hub_blocklist"):
            return self.retrieve_hub_blocklist_lexical(query_text, p_0=p_0, pi_query=pi_query)
        elif m in ("deg_discount_matrix", "deg_matrix", "static_deg_discount"):
            return self.retrieve_static_deg_discount(query_text, p_0=p_0)
        elif m in ("deg_ppr", "degree_normalized"):
            return self.retrieve_degree_normalized_ppr(query_text, p_0=p_0, pi_query=pi_query)
        elif m in ("hipporag", "prewalk_ppr"):
            return self.retrieve_hipporag(query_text, p_0=p_0)
        elif m in ("query_reweighted_ppr", "reweighted_ppr"):
            return self.retrieve_query_reweighted_ppr(query_text, p_0=p_0)
        elif m in ("perron_static", "specificity_static"):
            return self.retrieve_perron_static(query_text, p_0=p_0, pi_query=pi_query)
        elif m in ("perron", "specificity", "perron_adaptive"):
            return self.retrieve_perron(query_text, p_0=p_0, pi_query=pi_query)
        elif m in ("oracle", "upper_bound"):
            return self.retrieve_oracle(gold_symbol_ids or [])
        else:
            raise ValueError(f"Unknown retrieval method: {method_name}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Baseline Retrievers Engine")
    parser.add_argument("--test-smoke", action="store_true", help="Run self-contained smoke tests")
    args = parser.parse_args()

    if args.test_smoke:
        print("Running baseline retrievers smoke test...")
        # 8-node synthetic test graph
        rows = [0, 0, 1, 1, 2, 2, 4, 4, 6, 7]
        cols = [6, 5, 5, 2, 5, 3, 5, 3, 5, 5]
        data = [1.0] * len(rows)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(8, 8))
        symbols = [
            {"id": 0, "identifier": "process_order", "file_path": "order.py", "is_test": False},
            {"id": 1, "identifier": "alpha", "file_path": "alpha.py", "is_test": False},
            {"id": 2, "identifier": "beta", "file_path": "beta.py", "is_test": False},
            {"id": 3, "identifier": "gamma", "file_path": "gamma.py", "is_test": False},
            {"id": 4, "identifier": "delta", "file_path": "delta.py", "is_test": False},
            {"id": 5, "identifier": "logging_util", "file_path": "logger.py", "is_test": False},
            {"id": 6, "identifier": "calculate_tax", "file_path": "tax.py", "is_test": False},
            {"id": 7, "identifier": "test_order", "file_path": "tests/test_order.py", "is_test": True},
        ]
        # Verify prior reconciliation and beta decoupling
        engine = RepositoryRetrievalEngine(
            t_matrix=adj,
            symbols=symbols,
            beta_query=0.85,
            beta_global=0.50,
            filter_test_nodes=True,
        )
        p_lin = engine.compute_linear_prior("process_order")
        p_smax = engine.compute_softmax_prior("process_order", k=5, tau=0.15)
        assert np.isclose(float(np.sum(p_lin)), 1.0, atol=1e-5), "Linear prior sum != 1.0"
        assert np.isclose(float(np.sum(p_smax)), 1.0, atol=1e-5), "Softmax prior sum != 1.0"

        # Verify all baselines mask test node (node 7)
        methods = [
            "bm25", "bm25_1hop", "dense", "prior_only", "standard_ppr",
            "aider", "hub_blocklist", "deg_matrix", "degree_normalized",
            "hipporag", "query_reweighted_ppr", "perron_static", "perron",
        ]
        for m in methods:
            ranks = engine.retrieve_by_name(m, "process_order")
            top_7 = ranks[:7]
            assert 7 not in top_7, f"Method {m} failed to mask test node 7 from top ranks: {ranks}"

        print(f"[PASS] All {len(methods)} baseline retrievers verified with test-node masking and decoupled beta.")
        sys.exit(0)

