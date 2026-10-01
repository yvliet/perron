"""
Perron Specificity Module.

Computes global PageRank stationary vectors and post-walk hub-damping Specificity ratios.
Decouples query-specific diffusion from static zero-copy CSR matrix structures.
"""

from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
from scipy.sparse import csr_matrix

from perron.diffusion import personalized_pagerank_power_iteration


def compute_global_pagerank(
    t_matrix: csr_matrix,
    dangling: np.ndarray,
    beta: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Compute pre-indexed global PageRank vector pi_global using uniform teleport prior.
    Evaluated once per repository during static index generation.
    """
    num_nodes = t_matrix.shape[0]
    if num_nodes == 0:
        return np.empty(0, dtype=np.float32)
    p_uniform = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
    return personalized_pagerank_power_iteration(
        t_matrix=t_matrix,
        dangling=dangling,
        p_0=p_uniform,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
    )


def calculate_specificity_scores(
    pi_query: np.ndarray,
    pi_global: np.ndarray,
    gamma: float = 0.7,
    epsilon: float = 1e-8,
    is_test_node: Optional[Sequence[bool] | np.ndarray] = None,
    filter_test_nodes: bool = True,
    query_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute post-walk Specificity scores:
        Specificity(v) = pi_query(v) / (pi_global(v) + epsilon)^gamma(v)

    Suppress ubiquitous hub nodes (high pi_global) while amplifying query-specific paths.
    If query_prior is supplied, adapts gamma(v) = gamma * (1 - 0.5 * clip(p_0(v) * |V|, 0, 1))
    to avoid penalizing central architectural hubs when the query explicitly targets them.
    Test nodes serve as bipartite routing bridges during diffusion, but are excluded
    from code context packing when filter_test_nodes is True.
    """
    if len(pi_query) != len(pi_global):
        raise ValueError(
            f"Dimension mismatch between pi_query ({len(pi_query)}) and pi_global ({len(pi_global)})."
        )
    if len(pi_query) == 0:
        return np.empty(0, dtype=np.float32)
    if np.isnan(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be strictly positive, got {epsilon} <= 0.")
    if np.isnan(gamma) or gamma < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma} < 0.")

    # Guard against negative floating-point residuals to prevent NaN generation in power operation
    pi_global_safe = np.nan_to_num(np.maximum(np.asarray(pi_global, dtype=np.float64), 0.0), nan=0.0)
    pi_query_safe = np.nan_to_num(np.maximum(np.asarray(pi_query, dtype=np.float64), 0.0), nan=0.0)

    if query_prior is not None and len(query_prior) == len(pi_query):
        p_0_safe = np.nan_to_num(np.maximum(np.asarray(query_prior, dtype=np.float64), 0.0), nan=0.0)
        num_v = max(1, len(pi_query))
        prior_boost = np.clip(p_0_safe * (num_v / 5.0), 0.0, 1.0)
        effective_gamma = gamma * (1.0 - 0.50 * prior_boost)
    else:
        effective_gamma = gamma

    denom = (pi_global_safe + epsilon) ** effective_gamma
    specificity = np.nan_to_num(pi_query_safe / denom, nan=0.0, posinf=0.0, neginf=0.0)

    if filter_test_nodes and is_test_node is not None:
        mask = np.asarray(is_test_node, dtype=bool)
        if len(mask) != len(specificity):
            raise ValueError(
                f"Length of is_test_node ({len(mask)}) does not match specificity ({len(specificity)})."
            )
        specificity[mask] = 0.0

    return specificity.astype(np.float32)
