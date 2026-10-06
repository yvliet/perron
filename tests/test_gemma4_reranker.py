"""
Tests for Gemma 4 Candidate Pools, Cache, and Pointwise Reranker.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from perron.reranker.cache import PointwiseScoreCache
from perron.reranker.candidate_pools import (
    CandidatePoolBuilder,
    is_filter_f_passing,
)
from perron.reranker.scorer import (
    PROMPT_TEMPLATES,
    BreadcrumbRenderer,
    PointwiseReranker,
    normalize_and_truncate_issue_text,
)


class DummyEngine:
    def __init__(self, num_nodes: int = 20):
        self.num_nodes = num_nodes
        self.symbols = [
            {
                "id": i,
                "identifier": f"func_{i}",
                "file_path": f"src/mod_{i % 3}.py",
                "symbol_type": "function",
                "line_start": 10 * i + 1,
                "line_end": 10 * i + 8,
            }
            for i in range(num_nodes)
        ]
        # Make one class, one test node, one hub
        self.symbols[5]["symbol_type"] = "class"
        self.is_test_mask = np.zeros(num_nodes, dtype=bool)
        self.is_test_mask[6] = True  # test node
        self.hub_set = {7}           # hub node

        # Ring graph adjacency
        rows = list(range(num_nodes))
        cols = [(i + 1) % num_nodes for i in range(num_nodes)]
        data = [1.0] * num_nodes
        self.t_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)

        class DummyBM25:
            def __init__(self, n: int):
                self.n = n

            def query(self, text: str) -> np.ndarray:
                arr = np.linspace(10.0, 1.0, self.n, dtype=np.float32)
                return arr

        self.bm25_index = DummyBM25(num_nodes)


def test_filter_f():
    sym_fn = {"symbol_type": "function"}
    sym_cls = {"symbol_type": "class"}
    is_test = np.array([False, True, False])
    hub_set = {2}

    assert is_filter_f_passing(sym_fn, 0, is_test, hub_set) is True
    assert is_filter_f_passing(sym_cls, 0, is_test, hub_set) is False
    assert is_filter_f_passing(sym_fn, 1, is_test, hub_set) is False
    assert is_filter_f_passing(sym_fn, 2, is_test, hub_set) is False


def test_candidate_pools_builder():
    eng = DummyEngine(num_nodes=25)
    builder = CandidatePoolBuilder(engine=eng)

    pools, diag = builder.build_all_pools(
        query_text="fix broken function",
        instance_id="test-task-1",
        gold_symbol_ids={1, 2},
    )

    # Check that arms exist
    assert "A1" in pools
    assert "A2" in pools
    assert "A3" in pools
    assert "A4" in pools
    assert "A5" in pools
    assert "A6" in pools
    assert "B0" in pools
    assert "B1" in pools

    # A1 has length 10
    assert len(pools["A1"]) == 10

    # Controls are matched to M_t
    m_t = diag.m_t
    assert len(pools["A2"]) == m_t
    assert len(pools["A3"]) == m_t
    assert len(pools["A4"]) == m_t

    # Filter F enforced on expanded nodes: no class (5), no test (6), no hub (7) in non-seed expansions
    for arm in ("A2", "A3", "A4"):
        expansions = pools[arm][10:]
        assert 5 not in expansions
        assert 6 not in expansions
        assert 7 not in expansions


def test_score_cache():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cache.db"
        cache = PointwiseScoreCache(db_path)

        assert cache.count() == 0
        cache.put("task-1", 10, "P0", s=2.5, format_mass=0.95, n_prompt_tokens=150)
        assert cache.count() == 1

        val = cache.get("task-1", 10, "P0")
        assert val is not None
        assert abs(val["s"] - 2.5) < 1e-5
        assert abs(val["format_mass"] - 0.95) < 1e-5

        batch_hits = cache.get_many("task-1", [10, 20], "P0")
        assert 10 in batch_hits
        assert 20 not in batch_hits


def test_prompt_renderer_and_scorer():
    renderer = BreadcrumbRenderer()
    crumb = renderer.extract_breadcrumb(
        {"identifier": "hello", "file_path": "foo/bar.py", "line_start": 1, "line_end": 5},
        repo_slug="dummy",
        mode="skeleton",
    )
    assert "hello" in crumb

    issue_truncated = normalize_and_truncate_issue_text("word " * 1500, max_tokens=1000)
    assert "[... truncated ...]" in issue_truncated

    reranker = PointwiseReranker(breadcrumb_renderer=renderer)
    task = {"instance_id": "test-1", "problem_statement": "fix syntax error"}
    symbols = [{"identifier": "fix_syntax", "file_path": "a.py"}]
    scores = reranker.score_candidates_for_task(
        task=task,
        candidate_indices=[0],
        symbols=symbols,
        repo_slug="dummy",
        prompt_variant="P0",
    )
    assert 0 in scores
    assert scores[0]["format_mass"] >= 0.90
