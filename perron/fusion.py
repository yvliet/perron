"""
BM25 and Perron Diffusion Fusion Engine (T2.3).

Implements:
1. Reciprocal Rank Fusion (RRF) with k=60.
2. Weighted Min-Max Score Fusion for w in {0.0, 0.1, ..., 1.0}:
   score(s) = (1 - w) * norm(bm25) + w * norm(perron)
3. Pre-registered Selection Rule: Choose w with highest Fn Acc@10 on dev_val;
   ties broken by smaller w.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
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

    score(item) = 1.0 / (k + rank_a) + 1.0 / (k + rank_b)
    where rank is 1-indexed.
    """
    scores: Dict[int, float] = {}

    for r_idx, item in enumerate(ranking_a, 1):
        scores[item] = scores.get(item, 0.0) + (1.0 / (k + r_idx))

    for r_idx, item in enumerate(ranking_b, 1):
        scores[item] = scores.get(item, 0.0) + (1.0 / (k + r_idx))

    # All items ranked by RRF score descending, stable tie-breaking
    all_items = sorted(scores.keys(), key=lambda x: -scores[x])

    if total_nodes is not None and len(all_items) < total_nodes:
        seen = set(all_items)
        unseen = [i for i in range(total_nodes) if i not in seen]
        all_items.extend(unseen)

    return all_items


def min_max_scale(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize score array into [0.0, 1.0] range via min-max scaling.
    Safely handles uniform / degenerate arrays.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    denom = max_v - min_v
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
    Combine normalized BM25 and Perron scores using weight w in [0.0, 1.0]:
    fused_score = (1.0 - w) * norm(bm25) + w * norm(perron)

    Returns:
    (ranking, fused_scores)
    """
    w = max(0.0, min(1.0, float(w)))
    norm_bm25 = min_max_scale(scores_bm25, eps=eps)
    norm_perron = min_max_scale(scores_perron, eps=eps)

    fused = (1.0 - w) * norm_bm25 + w * norm_perron
    ranking = list(np.argsort(-fused, kind="stable"))
    return ranking, fused


def compute_instance_fusion_rankings(
    scores_bm25: np.ndarray,
    scores_perron: np.ndarray,
    weights: Sequence[float] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ),
    rrf_k: int = 60,
) -> Dict[str, List[int]]:
    """
    Compute all fusion rankings for a single instance given raw score arrays.
    """
    rankings: Dict[str, List[int]] = {}

    # 1. RRF ranking
    rank_bm25 = list(np.argsort(-scores_bm25, kind="stable"))
    rank_perron = list(np.argsort(-scores_perron, kind="stable"))
    rankings["rrf_60"] = reciprocal_rank_fusion(
        rank_bm25, rank_perron, k=rrf_k, total_nodes=len(scores_bm25)
    )

    # 2. Weighted min-max score fusion for each w
    for w in weights:
        w_key = f"weighted_w_{w:.1f}"
        rk, _ = weighted_score_fusion(scores_bm25, scores_perron, w=w)
        rankings[w_key] = rk

    return rankings
