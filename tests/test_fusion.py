"""
Unit tests for BM25 and Perron Fusion engine (T2.3).
"""

from __future__ import annotations

import numpy as np
import pytest

from perron.fusion import (
    compute_instance_fusion_rankings,
    min_max_scale,
    reciprocal_rank_fusion,
    select_fusion_weight,
    weighted_score_fusion,
)


def test_reciprocal_rank_fusion_basic() -> None:
    """Verify RRF calculates scores accurately with 1-indexed ranks."""
    # Ranking A: item 0 at rank 1, item 1 at rank 2, item 2 at rank 3
    # Ranking B: item 1 at rank 1, item 0 at rank 2, item 2 at rank 3
    ranking_a = [0, 1, 2]
    ranking_b = [1, 0, 2]
    k = 60

    # Item 0 score: 1/(60+1) + 1/(60+2) = 1/61 + 1/62
    # Item 1 score: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 (identical to item 0)
    # Item 2 score: 1/(60+3) + 1/(60+3) = 2/63 (strictly lower)
    fused = reciprocal_rank_fusion(ranking_a, ranking_b, k=k)
    assert len(fused) == 3
    assert fused[2] == 2
    assert set(fused[:2]) == {0, 1}


def test_reciprocal_rank_fusion_disjoint_tail() -> None:
    """Verify RRF handles non-overlapping rankings and total_nodes."""
    ranking_a = [0, 1]
    ranking_b = [2, 3]
    fused = reciprocal_rank_fusion(ranking_a, ranking_b, k=60, total_nodes=6)
    assert len(fused) == 6
    assert set(fused[:4]) == {0, 1, 2, 3}
    assert set(fused[4:]) == {4, 5}


def test_min_max_scale_normal_and_edge_cases() -> None:
    """Verify min_max_scale normalizes properly and handles uniform vectors."""
    scores = np.array([10.0, 20.0, 30.0, 50.0])
    scaled = min_max_scale(scores)
    assert pytest.approx(scaled[0]) == 0.0
    assert pytest.approx(scaled[-1]) == 1.0
    assert pytest.approx(scaled[1]) == 0.25

    # Uniform vector
    uniform = np.array([5.0, 5.0, 5.0])
    scaled_u = min_max_scale(uniform)
    assert np.all(scaled_u == 0.0)

    # Empty vector
    assert min_max_scale(np.array([])).size == 0


def test_weighted_score_fusion_extremes() -> None:
    """Verify w=0.0 recovers BM25 ranking and w=1.0 recovers Perron ranking."""
    bm25 = np.array([10.0, 5.0, 1.0])  # ranking: [0, 1, 2]
    perron = np.array([1.0, 8.0, 20.0])  # ranking: [2, 1, 0]

    # w = 0.0 -> purely BM25
    rank_0, scores_0 = weighted_score_fusion(bm25, perron, w=0.0)
    assert rank_0 == [0, 1, 2]

    # w = 1.0 -> purely Perron
    rank_1, scores_1 = weighted_score_fusion(bm25, perron, w=1.0)
    assert rank_1 == [2, 1, 0]


def test_compute_instance_fusion_rankings() -> None:
    """Verify compute_instance_fusion_rankings generates all specified variants."""
    bm25 = np.array([10.0, 2.0, 0.0, 5.0])
    perron = np.array([0.1, 0.9, 0.5, 0.2])

    res = compute_instance_fusion_rankings(bm25, perron)
    assert "rrf_60" in res
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        assert f"weighted_w_{w:.1f}" in res
        assert len(res[f"weighted_w_{w:.1f}"]) == 4


def test_rrf_tie_order_is_input_order_invariant():
    a, b = [0, 1, 2], [1, 0, 2]
    assert reciprocal_rank_fusion(a, b, k=60) == reciprocal_rank_fusion(b, a, k=60) == [0, 1, 2]


def test_select_fusion_weight_tie_rule():
    def cell(f10, f5):
        return {"metrics": {"fn_acc_10": {"mean": f10}, "fn_acc_5": {"mean": f5}}}
    summary = {"weighted_w_0.0": cell(0.20, 0.11), "weighted_w_0.1": cell(0.20, 0.12),
               "weighted_w_0.2": cell(0.20, 0.11), "weighted_w_0.3": cell(0.19, 0.13)}
    assert select_fusion_weight(summary, [0.3, 0.2, 0.1, 0.0]) == 0.1

