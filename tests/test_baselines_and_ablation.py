"""
Unit Tests for the 6-Baseline Retrieval Suite and Evaluation Invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from benchmarks.baseline_retrievers import FastBM25, RepositoryRetrievalEngine, tokenize_text
from perron.specificity import calculate_specificity_scores
from benchmarks.eval_harness import (
    compute_bootstrap_ci,
    compute_context_purity,
    compute_file_acc_at_k,
    compute_function_acc_at_k,
    compute_hsi,
    files_match,
)


def create_synthetic_toy_engine() -> RepositoryRetrievalEngine:
    """Create a minimal 6-node synthetic graph engine for unit testing."""
    # Adjacency: 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 5, 0 -> 2, 1 -> 2, 3 -> 2, 4 -> 2, 5 -> 2
    # Node 2 has high in-degree (hub)
    rows = [0, 1, 2, 3, 4, 0, 1, 3, 4, 5]
    cols = [1, 2, 3, 4, 5, 2, 2, 2, 2, 2]
    data = [1.0] * len(rows)
    adj = csr_matrix((data, (rows, cols)), shape=(6, 6))

    symbols = [
        {"id": 0, "identifier": "client_session", "file_path": "client.py", "degree": 1},
        {"id": 1, "identifier": "request_handler", "file_path": "handler.py", "degree": 2},
        {"id": 2, "identifier": "BaseManager", "file_path": "core/base.py", "degree": 5},  # Hub
        {"id": 3, "identifier": "parse_payload", "file_path": "parser.py", "degree": 2},
        {"id": 4, "identifier": "validate_schema", "file_path": "schema.py", "degree": 2},
        {"id": 5, "identifier": "format_response", "file_path": "response.py", "degree": 2},
    ]

    return RepositoryRetrievalEngine(t_matrix=adj, symbols=symbols, beta=0.85, gamma=0.70)


class TestFastBM25:
    """Tests for zero-dependency FastBM25 implementation."""

    def test_tokenize_text(self):
        text = "def test_connection(self, host: str, port: int) -> bool:"
        tokens = tokenize_text(text)
        assert "connection" in tokens
        assert "host" in tokens
        assert "port" in tokens
        assert "self" not in tokens  # Stop word
        assert "def" not in tokens  # Stop word

    def test_bm25_scoring(self):
        symbols = [
            {"identifier": "handle_http_request", "file_path": "http.py"},
            {"identifier": "parse_sql_query", "file_path": "db/sql.py"},
            {"identifier": "format_log_output", "file_path": "utils/log.py"},
        ]
        bm25 = FastBM25(symbols)
        scores = bm25.query("sql query syntax")
        assert len(scores) == 3
        # Symbol 1 should have highest score for SQL query
        assert scores[1] > scores[0]
        assert scores[1] > scores[2]


class TestRepositoryRetrievalEngine:
    """Tests for RepositoryRetrievalEngine baseline methods."""

    def test_initialization_and_hub_detection(self):
        engine = create_synthetic_toy_engine()
        assert engine.num_nodes == 6
        assert len(engine.pi_global) == 6
        # Node 2 has the highest in-degree (5 incoming edges)
        assert 2 in engine.hub_set

    def test_all_baselines_return_valid_permutations(self):
        engine = create_synthetic_toy_engine()
        query = "client request handler"
        methods = [
            "bm25",
            "bm25_1hop",
            "dense",
            "prior_only",
            "standard_ppr",
            "aider",
            "blocklist_lex",
            "deg_discount_matrix",
            "deg_ppr",
            "hipporag",
            "query_reweighted_ppr",
            "perron_static",
            "perron",
            "oracle",
        ]

        for m in methods:
            ranking = engine.retrieve_by_name(m, query, gold_symbol_ids=[2, 3])
            assert len(ranking) == 6
            assert set(ranking) == set(range(6))  # Complete permutation of nodes

    def test_hub_blocklist_suppresses_top_hub_when_unmatched(self):
        engine = create_synthetic_toy_engine()
        # When query does NOT match BaseManager, hub 2 is masked to 0.0
        ranking = engine.retrieve_hub_blocklist_lexical("format client payload")
        assert ranking[-1] == 2 or 2 in ranking[-2:]

    def test_perron_hub_adaptivity_preserves_intentional_target(self):
        engine = create_synthetic_toy_engine()
        # When query explicitly targets BaseManager, Perron query-level hub adaptivity preserves it
        ranking_target = engine.retrieve_perron("BaseManager core defect")
        assert ranking_target[0] == 2 or 2 in ranking_target[:3]

    def test_sandwich_inequality_crossover_property(self):
        """
        Verify the analytical crossover condition:
            gamma* = ln(pi_q(h) / pi_q(l)) / ln(pi_g(h) / pi_g(l))
        For gamma > gamma*, hub ranks below leaf (specificity(h) < specificity(l)).
        For gamma < gamma*, hub ranks strictly above leaf (specificity(h) > specificity(l)).
        """
        pi_g = np.array([0.01, 0.50])
        pi_q = np.array([0.10, 0.90])

        gamma_star = float(np.log(pi_q[1] / pi_q[0]) / np.log(pi_g[1] / pi_g[0]))
        assert 0.50 < gamma_star < 0.60

        # Case 1: gamma = 0.70 > gamma* -> Leaf outranks Hub!
        scores_above = calculate_specificity_scores(pi_q, pi_g, gamma=0.70)
        assert scores_above[0] > scores_above[1], "Hub should be suppressed when gamma > gamma*"

        # Case 2: gamma = 0.35 < gamma* -> Hub outranks Leaf!
        scores_below = calculate_specificity_scores(pi_q, pi_g, gamma=0.35)
        assert scores_below[1] > scores_below[0], "Hub should outrank leaf when gamma < gamma*"

        # Case 3: Monotonicity
        gamma_values = np.linspace(0.1, 0.9, 9)
        ratio_h_over_l = []
        for g in gamma_values:
            sc = calculate_specificity_scores(pi_q, pi_g, gamma=float(g))
            ratio_h_over_l.append(sc[1] / sc[0])

        # As gamma increases, ratio of hub / leaf score must strictly decrease!
        for i in range(len(ratio_h_over_l) - 1):
            assert ratio_h_over_l[i] > ratio_h_over_l[i + 1], "Ratio must be strictly monotonically decreasing in gamma"

    def test_dispatcher_invalid_name_raises(self):
        engine = create_synthetic_toy_engine()
        with pytest.raises(ValueError, match="Unknown retrieval method"):
            engine.retrieve_by_name("unknown_algorithm", "query")


class TestEvaluationHarnessInvariants:
    """Tests for metrics and confidence intervals in eval_harness."""

    def test_file_matching_strict(self):
        assert files_match("requests/sessions.py", "sessions.py")
        assert files_match("sessions.py", "requests/sessions.py")
        assert not files_match("a/b/c.py", "x/y/c.py")
        assert not files_match("sessions.py", "models.py")

    def test_file_acc_at_k(self):
        retrieved = ["a.py", "b.py", "c.py", "d.py"]
        gold = ["c.py"]
        acc = compute_file_acc_at_k(retrieved, gold, (1, 3, 5))
        assert acc[1] == 0.0
        assert acc[3] == 1.0
        assert acc[5] == 1.0

    def test_function_acc_at_k(self):
        retrieved = [10, 20, 30, 40, 50]
        gold = [30]
        acc = compute_function_acc_at_k(retrieved, gold, (1, 2, 3, 5))
        assert acc[1] == 0.0
        assert acc[2] == 0.0
        assert acc[3] == 1.0
        assert acc[5] == 1.0

    def test_hsi_calculation(self):
        retrieved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        hubs = {1, 2, 3}  # 3 of top 10 are hubs
        hsi = compute_hsi(retrieved, hubs, top_k=10)
        assert pytest.approx(hsi, 0.01) == 0.70  # 1 - 3/10 = 0.70

    def test_bootstrap_ci_coverage(self):
        vals = [1.0] * 50 + [0.0] * 50
        mean, lower, upper = compute_bootstrap_ci(vals, n_bootstrap=1000, seed=42)
        assert pytest.approx(mean, 0.01) == 0.50
        assert 0.35 <= lower <= 0.45
        assert 0.55 <= upper <= 0.65

    def test_exact_mcnemar_and_holm_correction(self):
        from benchmarks.eval_harness import compute_exact_mcnemar_test, compute_holm_bonferroni_correction
        y_target = [1, 1, 1, 0, 0, 1, 0]
        y_base   = [0, 0, 1, 0, 1, 1, 0]
        res = compute_exact_mcnemar_test(y_target, y_base)
        assert res["n_target_wins"] == 2
        assert res["n_baseline_wins"] == 1
        assert 0.0 <= res["p_value"] <= 1.0

        p_vals = {"m1": 0.01, "m2": 0.04, "m3": 0.05}
        adj = compute_holm_bonferroni_correction(p_vals)
        assert adj["m1"] <= adj["m2"] <= adj["m3"]
