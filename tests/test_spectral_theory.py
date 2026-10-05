"""
Pytest suite verifying Perron multigraph spectral diffusion theorems:
- Proposition 2 (Barycentric Convex Hull Placement & Resolvent Superposition)
- Proposition 3 (Exact Asymptotic Log-Linear PMI Decomposition)

Author: Sultan Haikal (GitHub: @yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import pytest


def test_proposition_2_convex_hull_and_warping_bound():
    """
    Verify Proposition 2:
    1. pi(p_0) = sum_i w_tilde_i pi(e_s_i) with w_tilde_i >= 0, sum w_tilde_i = 1.0.
    2. Mass bounds: (1 - beta) <= m_i <= 1.0.
    3. Coordinate warping ratio <= 1 / (1 - beta) = 6.6667.
    """
    beta = 0.85
    min_mass_theoretical = 1.0 - beta  # 0.15
    max_mass_theoretical = 1.0
    max_warping_ratio = 1.0 / (1.0 - beta)  # 6.666666666666667

    n = 4
    indptr = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    indices = np.array([2, 3, 1], dtype=np.int32)
    data = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    t_mat = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    dangling = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    eye = sp.eye(n, format="csr", dtype=np.float64)
    resolvent = sp.linalg.inv((eye - beta * t_mat).tocsc())

    u_basis = []
    masses = []
    for i in range(n):
        e_i = np.zeros(n, dtype=np.float64)
        e_i[i] = 1.0
        # Remediation Gate R1: safe 1D array coercion replacing invalid .toarray()
        u_i = (1.0 - beta) * np.asarray(e_i @ resolvent).ravel()
        m_i = float(np.sum(u_i))
        u_basis.append(u_i)
        masses.append(m_i)

        assert min_mass_theoretical - 1e-9 <= m_i <= max_mass_theoretical + 1e-9, (
            f"Mass m_{i} = {m_i} violates theoretical bounds [{min_mass_theoretical}, {max_mass_theoretical}]"
        )

    min_m = min(masses)
    max_m = max(masses)
    warping_ratio = max_m / min_m
    assert warping_ratio <= max_warping_ratio + 1e-9, (
        f"Warping ratio {warping_ratio:.4f} exceeds {max_warping_ratio:.4f}"
    )

    rng = np.random.default_rng(20261005)
    for _ in range(20):
        w = rng.uniform(0.1, 1.0, size=n)
        w = w / np.sum(w)

        # Remediation Gate R1: safe 1D array coercion
        u_true = (1.0 - beta) * np.asarray(w @ resolvent).ravel()
        pi_true = u_true / np.sum(u_true)

        w_m = np.sum(w * masses)
        w_tilde = (w * masses) / w_m

        assert abs(np.sum(w_tilde) - 1.0) < 1e-12
        assert np.all(w_tilde >= 0.0)

        pi_superposed = np.zeros(n, dtype=np.float64)
        for i in range(n):
            pi_i = u_basis[i] / masses[i]
            pi_superposed += w_tilde[i] * pi_i

        l1_diff = float(np.sum(np.abs(pi_true - pi_superposed)))
        assert l1_diff < 1e-12, f"Convex hull superposition failed: L1 error = {l1_diff}"


def test_proposition_3_asymptotic_pmi_identity():
    """
    Verify Proposition 3:
    ln Specificity(v) = (1 - gamma) ln pi_{q,v} + gamma PMI(v; q) - gamma ln(1 + eps / pi_{g,v})
    Machine precision check and refutation of additive fallacy.
    """
    gamma = 0.70
    eps = 1e-8

    pi_q = np.array([0.40, 0.25, 0.05, 0.001], dtype=np.float64)
    pi_g = np.array([0.10, 0.50, 0.01, 0.0001], dtype=np.float64)

    specificity_direct = pi_q / np.power(pi_g + eps, gamma)
    log_spec_direct = np.log(specificity_direct)

    pmi = np.log(pi_q) - np.log(pi_g)
    reg_residual = -gamma * np.log(1.0 + eps / pi_g)

    log_spec_prop3 = (1.0 - gamma) * np.log(pi_q) + gamma * pmi + reg_residual

    residual_error = np.max(np.abs(log_spec_direct - log_spec_prop3))
    assert residual_error < 1e-14, f"Proposition 3 identity violated: max error {residual_error}"

    for v_idx in range(len(pi_q)):
        r_eps = float(np.abs(reg_residual[v_idx]))
        assert r_eps < 1e-3, f"Regularization residual {r_eps} did not vanish on reachable support"

    additive_fallacy = (1.0 - gamma) * np.log(pi_q) + gamma * pmi - gamma * np.log(eps)
    fallacy_discrepancy = np.mean(np.abs(additive_fallacy - log_spec_direct))
    assert fallacy_discrepancy > 10.0, (
        f"Additive fallacy discrepancy should exceed 10.0 log units, got {fallacy_discrepancy}"
    )
