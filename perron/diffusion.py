"""
Perron Diffusion Module.

Implements shift-invariant softmax teleportation priors and fast sparse Personalized PageRank (PPR)
power iteration over row-stochastic graph transition matrices.
"""

from __future__ import annotations

from typing import Optional, Sequence
import warnings
import numpy as np
from scipy.sparse import csr_matrix


def compute_softmax_teleport_prior(
    similarities: Sequence[float] | np.ndarray,
    node_indices: Sequence[int] | np.ndarray,
    num_nodes: int,
    tau: float = 0.15,
    top_k: Optional[int] = 20,
) -> np.ndarray:
    """
    Compute shift-invariant softmax teleportation prior vector p_0.

    Uses the log-sum-exp trick and float64 accumulator to shield against
    overflow and underflow precision collapse at sharp temperatures (tau=0.05).

    Equation:
        c = max_i (s_i / tau)
        p_0(i) = exp((s_i / tau) - c) / sum_j exp((s_j / tau) - c)
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}.")
    if np.isnan(tau) or tau <= 0.0:
        raise ValueError(f"tau must be strictly positive, got {tau}.")

    sims = np.asarray(similarities, dtype=np.float64)
    nodes = np.asarray(node_indices, dtype=np.int64)

    if len(sims) != len(nodes):
        raise ValueError(
            f"Length mismatch between similarities ({len(sims)}) and node_indices ({len(nodes)})."
        )

    if len(sims) == 0:
        # Uniform fallback if no similarities provided
        p_0 = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)
        return p_0

    # Sanitize NaNs and Infs in similarity scores
    sims = np.nan_to_num(sims, nan=-1e9, posinf=1.0, neginf=-1e9)

    # Check bounds on node_indices
    if (nodes < 0).any() or (nodes >= num_nodes).any():
        raise IndexError(f"node_indices contains values outside [0, {num_nodes - 1}].")

    # Deduplicate node indices, taking the maximum similarity per unique node
    if len(nodes) > len(np.unique(nodes)):
        node_max_sim: dict[int, float] = {}
        for s_val, n_idx in zip(sims, nodes):
            n_int = int(n_idx)
            s_float = float(s_val)
            if n_int not in node_max_sim or s_float > node_max_sim[n_int]:
                node_max_sim[n_int] = s_float
        nodes = np.array(list(node_max_sim.keys()), dtype=np.int64)
        sims = np.array(list(node_max_sim.values()), dtype=np.float64)

    if top_k is not None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive when specified, got {top_k}.")
        if len(sims) > top_k:
            top_idx = np.argsort(sims)[-top_k:]
            sims = sims[top_idx]
            nodes = nodes[top_idx]

    # Shift-invariant log-sum-exp in float64
    scaled_sims = sims / tau
    c = np.max(scaled_sims)
    exp_sims = np.exp(scaled_sims - c)
    denom = np.sum(exp_sims)

    if denom <= 0.0 or np.isnan(denom):
        probs = np.full(len(sims), 1.0 / len(sims), dtype=np.float32)
    else:
        probs = (exp_sims / denom).astype(np.float32)

    p_0 = np.zeros(num_nodes, dtype=np.float32)
    p_0[nodes] = probs

    p_sum = np.sum(p_0)
    if p_sum > 0:
        p_0 /= p_sum
    else:
        p_0 = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float32)

    return p_0


def personalized_pagerank_power_iteration(
    t_matrix: csr_matrix,
    dangling: np.ndarray,
    p_0: Optional[np.ndarray] = None,
    beta: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> np.ndarray:
    """
    Compute stationary distribution pi via sparse power iteration.

    Equation:
        pi^(t+1) = beta * pi^(t) * T + (beta * (pi^(t) . d) + 1 - beta) * p_0^T

    Invariant:
        Sum of elements in pi is strictly conserved at 1.0 at every iteration.
        Perron-Frobenius spectral gap is invariant at (1 - beta) = 0.15.
        Guarantees convergence up to 100 iterations on deep scale-free graphs.
    """
    if max_iterations is not None:
        max_iter = max_iterations
    if tolerance is not None:
        tol = tolerance

    if p_0 is None:
        p_0 = dangling
        dangling = (np.diff(t_matrix.indptr) == 0).astype(np.float64)

    if t_matrix.shape[0] != t_matrix.shape[1]:
        raise ValueError(f"t_matrix must be square, got shape {t_matrix.shape}.")

    num_nodes = t_matrix.shape[0]
    if len(p_0) != num_nodes:
        raise ValueError(
            f"p_0 length ({len(p_0)}) does not match t_matrix dimension ({num_nodes})."
        )
    if len(dangling) != num_nodes:
        raise ValueError(
            f"dangling length ({len(dangling)}) does not match t_matrix dimension ({num_nodes})."
        )
    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must satisfy 0.0 <= beta < 1.0, got {beta}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be strictly positive, got {max_iter}.")
    if np.isnan(tol) or tol <= 0.0:
        raise ValueError(f"tol must be strictly positive, got {tol}.")

    if num_nodes == 0:
        return np.empty(0, dtype=np.float32)

    p_0_vec = np.nan_to_num(np.maximum(p_0.astype(np.float64), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    p_sum = np.sum(p_0_vec)
    if p_sum > 0:
        p_0_vec /= p_sum
    else:
        p_0_vec = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)

    pi = p_0_vec.copy()
    dangling_vec = dangling.astype(np.float64)
    converged = False

    for _ in range(max_iter):
        # Sparse vector-matrix multiplication: pi @ T
        pi_next = beta * (pi @ t_matrix)

        # Teleportation adjustment for regular teleport and dangling node leakage
        dangling_mass = np.dot(pi, dangling_vec)
        teleport_factor = beta * dangling_mass + (1.0 - beta)
        pi_next += teleport_factor * p_0_vec

        # Check L1 convergence criterion
        diff = np.sum(np.abs(pi_next - pi))
        pi = pi_next

        if diff < tol:
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Personalized PageRank power iteration reached max_iter={max_iter} without converging "
            f"(final L1 residual: {diff:.6e}, tol: {tol:.6e}). "
            f"Consider increasing max_iter for deep graphs.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Normalize to prevent any float32 accumulation drift
    total_mass = np.sum(pi)
    if total_mass > 0:
        pi /= total_mass

    return pi.astype(np.float32)
