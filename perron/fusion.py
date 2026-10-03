"""
BM25 and Perron Diffusion Fusion Engine.

Implements:
1. Reciprocal Rank Fusion (RRF) with k=60 and input-order-independent tie-breaking.
2. Weighted Min-Max Score Fusion for w in {0.0, 0.1, ..., 1.0}:
   score(s) = (1 - w) * norm(bm25) + w * norm(perron)
3. Pre-registered selection rule (select_fusion_weight): choose w with the highest
   dev Fn Acc@10; break ties by dev Fn Acc@5, then by the smaller w.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def reciprocal_rank_fusion(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    k: int = 60,
    total_nodes: Optional[int] = None,
) -> List[int]:
    """
    Combine two rankings using standard Reciprocal Rank Fusion (RRF).

    score(item) = 1.0 / (k + rank_a) + 1.0 / (k + rank_b), ranks 1-indexed.
    Equal scores are ordered by ascending item index, so the output does not
    depend on which ranking is passed first.
    """
    scores: Dict[int, float] = {}
    for r_idx, item in enumerate(ranking_a, 1):
        scores[int(item)] = scores.get(int(item), 0.0) + 1.0 / (k + r_idx)
    for r_idx, item in enumerate(ranking_b, 1):
        scores[int(item)] = scores.get(int(item), 0.0) + 1.0 / (k + r_idx)

    all_items = sorted(scores, key=lambda x: (-scores[x], x))

    if total_nodes is not None and len(all_items) < total_nodes:
        seen = set(all_items)
        all_items.extend(i for i in range(total_nodes) if i not in seen)
    return all_items


def min_max_scale(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a score array into [0.0, 1.0]; uniform arrays map to zeros."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    denom = float(np.max(arr)) - min_v
    if denom < eps:
        return np.zeros_like(arr)
    return (arr - min_v) / denom


def weighted_score_fusion(
    scores_bm25: np.ndarray,
    scores_perron: np.ndarray,
    w: float,
    eps: float = 1e-8,
) -> Tuple[List[int], np.ndarray]:
    """
    fused_score = (1.0 - w) * norm(bm25) + w * norm(perron).
    Stable argsort breaks equal fused scores by ascending index.
    """
    w = max(0.0, min(1.0, float(w)))
    fused = (1.0 - w) * min_max_scale(scores_bm25, eps=eps) + w * min_max_scale(scores_perron, eps=eps)
    ranking = [int(i) for i in np.argsort(-fused, kind="stable")]
    return ranking, fused


def select_fusion_weight(summary: Dict[str, Any], weights: Sequence[float]) -> float:
    """
    Pre-registered dev-split selection rule for weighted fusion.

    Maximize Fn Acc@10; break ties by Fn Acc@5, then by the smaller weight
    (closer to pure BM25). The result does not depend on grid order.
    """
    def key(w: float) -> Tuple[float, float, float]:
        m = summary[f"weighted_w_{w:.1f}"]["metrics"]
        return (m["fn_acc_10"]["mean"], m["fn_acc_5"]["mean"], -float(w))

    return float(max(weights, key=key))


def compute_instance_fusion_rankings(
    scores_bm25: np.ndarray,
    scores_perron: np.ndarray,
    weights: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    rrf_k: int = 60,
) -> Dict[str, List[int]]:
    """Compute all fusion rankings for a single instance given raw score arrays."""
    rankings: Dict[str, List[int]] = {}
    rank_bm25 = [int(i) for i in np.argsort(-scores_bm25, kind="stable")]
    rank_perron = [int(i) for i in np.argsort(-scores_perron, kind="stable")]
    rankings["rrf_60"] = reciprocal_rank_fusion(rank_bm25, rank_perron, k=rrf_k, total_nodes=len(scores_bm25))
    for w in weights:
        rk, _ = weighted_score_fusion(scores_bm25, scores_perron, w=w)
        rankings[f"weighted_w_{w:.1f}"] = rk
    return rankings
