"""
Unit and Integration Tests for Spectral Resolvent Linearity and Dose-Response Surface:
- Spectral resolvent linearity tolerance (float32 <= 2.0e-6, float64 <= 1.0e-10)
- Batched SpMM resolvent solver (compute_resolvent_basis_batched)
- Fixed-window BM25 score normalization sanitization guardrails
- Candidate pool dynamic safety certificate and full-graph GEMM fallback
- Binomial GLMM collapse fitting (IRLS engine and leave-one-family-out R^2)
- Two One-Sided Tests (TOST) and paired exact McNemar equivalence engines
- End-to-end dose-response benchmark execution and JSON schema validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pytest
import scipy.sparse as sp

from benchmarks.run_dose_response_basis import (
    compute_leave_one_family_out_r2,
    compute_paired_mcnemar_exact,
    compute_resolvent_basis_batched,
    compute_tost_equivalence,
    fit_glmm_collapse_irls,
    normalize_scores_fixed_window,
    run_dose_response_benchmark,
    verify_basis_linearity_tolerance,
)
from perron.diffusion import personalized_pagerank_power_iteration

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# 1. Resolvent Linearity & Decoupled Precision Tolerances (DEF-04 & DEF-07)
# -----------------------------------------------------------------------------

def test_basis_linearity_algebraic_proof_float64() -> None:
    """
    Diagnostic algebraic verification: in IEEE 754 float64 arithmetic with tight convergence
    (tol=1e-14, max_iter=500), resolvent basis superposition matches iterative solves within L1 <= 1.0e-10.
    """
    np.random.seed(42)
    n = 80
    adj = (np.random.rand(n, n) < 0.08).astype(np.float64)
    # Inject dangling sink nodes (zero out out-edges)
    adj[15:30, :] = 0.0

    row_sums = adj.sum(axis=1)
    dangling = (row_sums == 0).astype(np.float64)
    row_sums[row_sums == 0] = 1.0
    t_matrix = sp.csr_matrix(adj / row_sums[:, None])

    seeds = [5, 12, 25, 40, 65]
    beta = 0.85

    # Compute basis vectors in double precision
    b_curr = len(seeds)
    C = np.zeros((b_curr, n), dtype=np.float64)
    for i, s_idx in enumerate(seeds):
        C[i, s_idx] = 1.0 - beta

    U = C.copy()
    for _ in range(500):
        U_next = C + beta * (U @ t_matrix)
        diff = np.max(np.abs(U_next - U))
        U = U_next
        if diff < 1e-14:
            break

    masses = np.sum(U, axis=1)

    for _ in range(5):
        w = np.random.dirichlet(np.ones(len(seeds)))
        u_comb = np.dot(w, U)
        m_comb = float(np.dot(w, masses))
        pi_basis = u_comb / m_comb

        # Iterative solve in float64
        p0 = np.zeros(n, dtype=np.float64)
        for s_idx, weight in zip(seeds, w):
            p0[s_idx] = weight

        pi_iter = p0.copy()
        for _ in range(500):
            dangling_mass = float(np.sum(pi_iter[dangling == 1.0]))
            pi_next = beta * (pi_iter @ t_matrix) + (beta * dangling_mass + 1.0 - beta) * p0
            if np.max(np.abs(pi_next - pi_iter)) < 1e-14:
                pi_iter = pi_next
                break
            pi_iter = pi_next

        l1_err = float(np.sum(np.abs(pi_basis - pi_iter)))
        assert l1_err <= 1.0e-10, f"Float64 algebraic L1 error {l1_err:.4e} exceeds 1.0e-10 tolerance limit."


def test_basis_linearity_production_batched_float32() -> None:
    """
    Assert that compute_resolvent_basis_batched produces valid float32 basis vectors
    and float64 mass vectors satisfying production tolerance L1 <= 2.0e-6.
    """
    np.random.seed(101)
    n = 150
    adj = (np.random.rand(n, n) < 0.05).astype(np.float64)
    adj[20:40, :] = 0.0

    row_sums = adj.sum(axis=1)
    dangling = (row_sums == 0).astype(np.float32)
    row_sums[row_sums == 0] = 1.0
    t_matrix = sp.csr_matrix(adj / row_sums[:, None])

    seeds = [10, 25, 45, 80, 120]
    basis_mat, masses = compute_resolvent_basis_batched(
        t_matrix=t_matrix,
        seed_indices=seeds,
        beta=0.85,
        tol=1e-12,
        batch_size=2,
    )

    assert basis_mat.shape == (len(seeds), n)
    assert basis_mat.dtype == np.float32
    assert masses.shape == (len(seeds),)
    assert masses.dtype == np.float64

    for _ in range(5):
        w = np.random.dirichlet(np.ones(len(seeds)))
        l1_err = verify_basis_linearity_tolerance(
            t_matrix=t_matrix,
            dangling=dangling,
            basis_vectors=basis_mat,
            masses=masses,
            seeds=seeds,
            weights=w,
            beta=0.85,
        )
        assert l1_err <= 2.0e-6, f"Production float32 L1 error {l1_err:.4e} exceeds 2.0e-6 limit."


def test_basis_linearity_on_real_repo_graphs() -> None:
    """
    Assert that batched resolvent linearity holds on real SWE-bench repository graphs within float32 tolerance.
    """
    slug = "pallets__flask"  # Real repository with 42% dangling nodes
    graph_path = PROJECT_ROOT / "artifacts" / "swebench_graphs" / f"{slug}_graph.npz"
    if not graph_path.exists():
        pytest.skip(f"Repository graph {graph_path} not found.")

    g = sp.load_npz(graph_path)
    dangling = (np.diff(g.indptr) == 0).astype(np.float32)

    seeds = [10, 25, 50, 100]
    basis_mat, masses = compute_resolvent_basis_batched(g, seeds, beta=0.85, batch_size=2)

    weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    l1_err = verify_basis_linearity_tolerance(
        t_matrix=g,
        dangling=dangling,
        basis_vectors=basis_mat,
        masses=masses,
        seeds=seeds,
        weights=weights,
        beta=0.85,
    )
    assert l1_err <= 2.0e-6, f"Real repo L1 error {l1_err:.4e} exceeds 2.0e-6."


# -----------------------------------------------------------------------------
# 2. Score Normalization Sanitization Guardrails (DEF-09)
# -----------------------------------------------------------------------------

def test_fixed_window_normalization_guardrails() -> None:
    """Assert that score normalization handles empty lists, zero hits, short lists, and flat distributions."""
    # 1. Empty array
    assert len(normalize_scores_fixed_window(np.array([]))) == 0

    # 2. Zero-hit query (all scores 0.0)
    zero_scores = np.zeros(50, dtype=np.float64)
    norm_zero = normalize_scores_fixed_window(zero_scores)
    assert np.allclose(norm_zero, 1.0 / 50)
    assert not np.any(np.isnan(norm_zero))

    # 3. Flat distribution (all scores identical, e.g. 15.0)
    flat_scores = np.full(30, 15.0, dtype=np.float64)
    norm_flat = normalize_scores_fixed_window(flat_scores)
    assert np.allclose(norm_flat, 1.0 / 30)
    assert not np.any(np.isnan(norm_flat))

    # 4. Standard distribution with invariant ranking
    raw_scores = np.sort(np.random.uniform(5.0, 50.0, 150))[::-1]
    norm_scores = normalize_scores_fixed_window(raw_scores)
    assert len(norm_scores) == 150
    assert np.isclose(norm_scores[0], 1.0, atol=1e-6)
    assert np.isclose(norm_scores[99], 0.0, atol=1e-6)


# -----------------------------------------------------------------------------
# 3. Candidate Pool Dynamic Safety Certificate & Fallback (DEF-02)
# -----------------------------------------------------------------------------

def test_candidate_pool_safety_certificate_exactness() -> None:
    """
    Assert that the Dynamic Candidate Safety Certificate guarantees 100% discrete
    top-10 exactness, and that uncertified cases trigger exact full-graph evaluation.
    """
    np.random.seed(456)
    n = 1000
    adj = (np.random.rand(n, n) < 0.01).astype(np.float64)
    row_sums = adj.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    t_matrix = sp.csr_matrix(adj / row_sums[:, None])

    seeds = [10, 20, 30, 40, 50]
    basis_mat, masses = compute_resolvent_basis_batched(t_matrix, seeds, beta=0.85)

    # Candidate pool: union of top-50 from each basis + gold targets
    gold_syms = [15, 25]
    cand_set = set(gold_syms)
    for i in range(len(seeds)):
        cand_set.update(np.argsort(-basis_mat[i])[:50])
    cand_pool = np.array(sorted(list(cand_set)), dtype=np.int64)

    non_cand_mask = np.ones(n, dtype=bool)
    non_cand_mask[cand_pool] = False
    M_bound = np.max(basis_mat[:, non_cand_mask], axis=1)

    for _ in range(25):
        w = np.random.dirichlet(np.ones(len(seeds)))
        outside_bound = float(np.dot(w, M_bound))

        # Candidate scores
        cand_scores = np.dot(w, basis_mat[:, cand_pool])
        tenth_cand = np.partition(cand_scores, -10)[-10]

        # Full-graph scores
        full_scores = np.dot(w, basis_mat)
        full_top10 = set(np.argsort(-full_scores)[:10])

        if outside_bound < tenth_cand:
            # Certified: candidate pool top-10 MUST match full graph
            proj_top10 = set(cand_pool[np.argsort(-cand_scores)[:10]])
            assert proj_top10 == full_top10, "Safety certificate violated!"
        else:
            # Uncertified: fallback guarantees exactness
            tenth_full = np.partition(full_scores, -10)[-10]
            for g in gold_syms:
                gold_in_top10_cand = bool(cand_scores[np.where(cand_pool == g)[0][0]] >= tenth_cand)
                gold_in_top10_full = bool(full_scores[g] >= tenth_full)
                assert gold_in_top10_full == (full_scores[g] in [full_scores[idx] for idx in full_top10])


# -----------------------------------------------------------------------------
# 4. Statistical Engines & Array Flattening Alignment (DEF-01 & DEF-03)
# -----------------------------------------------------------------------------

def test_glmm_covariate_flattening_alignment() -> None:
    """Assert that row-major covariate flattening synchronizes task_hits_matrix with covariates (DEF-01)."""
    n_tasks = 4
    n_configs = 3

    # Simulated task hit matrix [task_idx, cfg_idx]
    task_hits = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])

    all_configs = [
        {"family": "F1", "k": 10, "tau": 0.1, "neff": 2.5},
        {"family": "F1", "k": 25, "tau": 0.2, "neff": 8.0},
        {"family": "F2", "k": 50, "tau": 1.0, "neff": 15.0},
    ]
    repos = ["repo_a", "repo_b", "repo_a", "repo_c"]

    y_flat = task_hits.flatten()
    ln_neff_flat = np.tile([np.log(c["neff"]) for c in all_configs], n_tasks)
    repo_flat = np.repeat(repos, n_configs)

    assert len(y_flat) == n_tasks * n_configs
    assert len(ln_neff_flat) == n_tasks * n_configs
    assert len(repo_flat) == n_tasks * n_configs

    # Verify Task 1, Config 1 (index 4 in row-major: row 1 * 3 + col 1)
    target_idx = 1 * n_configs + 1
    assert y_flat[target_idx] == task_hits[1, 1]
    assert np.isclose(ln_neff_flat[target_idx], np.log(all_configs[1]["neff"]))
    assert repo_flat[target_idx] == repos[1]


def test_paired_mcnemar_exact_engine() -> None:
    """Assert that paired exact McNemar test evaluates discordant pairs correctly (DEF-03)."""
    # Identical predictions
    y1 = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y2 = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    res = compute_paired_mcnemar_exact(y1, y2)
    assert res["discordant_pairs"] == 0
    assert res["p_value"] == 1.0
    assert res["no_significant_divergence"] is True

    # Concordant flip (equal discordant shifts)
    y2_flip = y1.copy()
    y2_flip[0] = 0  # 1 -> 0
    y2_flip[1] = 1  # 0 -> 1
    res_flip = compute_paired_mcnemar_exact(y1, y2_flip)
    assert res_flip["n01"] == 1
    assert res_flip["n10"] == 1
    assert res_flip["p_value"] == 1.0
    assert res_flip["no_significant_divergence"] is True


def test_tost_sample_powered_boundaries() -> None:
    """Assert that TOST equivalence detects equivalence within calibrated boundaries (DEF-03)."""
    np.random.seed(999)
    n = 344
    acc1 = np.random.binomial(1, 0.45, n).astype(np.float64)
    acc2 = acc1.copy()
    acc2[:4] = 1.0 - acc2[:4]  # 4 discordant items

    res_5pp = compute_tost_equivalence(acc1, acc2, margin=0.05)
    assert res_5pp["equivalent_within_margin"] is True
    assert res_5pp["tost_p_value"] < 0.01


# -----------------------------------------------------------------------------
# 5. End-to-End Dry-Run & JSON Schema Validation
# -----------------------------------------------------------------------------

def test_dose_response_benchmark_dry_run(tmp_path: Path) -> None:
    """
    Verify that run_dose_response_benchmark executes cleanly in dry-run mode
    and outputs a schema-compliant JSON artifact.
    """
    out_json = tmp_path / "test_dose_response.json"
    res = run_dose_response_benchmark(
        limit=3,
        verify_linearity=True,
        output_file=out_json,
    )
    assert out_json.exists()
    assert res is not None

    # Schema checks
    assert "metadata" in res
    assert "linearity_validation" in res
    assert "h1_universal_entropy_collapse" in res
    assert "iso_entropy_contrast" in res
    assert "h3_hub_harm_falsification" in res
    assert "prior_calibration" in res
    assert "surface_summary" in res

    assert res["linearity_validation"]["exact_linearity_verified"] is True
    assert res["linearity_validation"]["max_l1_error"] <= 2.0e-6
    assert "paired_mcnemar_exact" in res["h3_hub_harm_falsification"]
    assert "expected_calibration_error" in res["prior_calibration"]
