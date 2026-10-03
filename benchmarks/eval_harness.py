"""
Task-Grounded Benchmark Evaluation Harness.

Implements standardized evaluation metrics aligned with Agentless and LocAgent:
- File Acc@k (k in {1, 3, 5})
- Function Acc@k (k in {1, 5, 10, 25})
- Context Purity@K (fraction of context relevant to gold defect)
- Hub Suppression Index (HSI, top-10 congestion diagnostic)
- 1,000-sample bootstrap confidence intervals
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple
import numpy as np


def files_match(path_a: str, path_b: str) -> bool:
    """Check if two file paths match strictly by exact match or normalized path suffix."""
    if not path_a or not path_b:
        return False
    p_a = path_a.replace("\\", "/").lstrip("/")
    p_b = path_b.replace("\\", "/").lstrip("/")
    return p_a == p_b or p_a.endswith("/" + p_b) or p_b.endswith("/" + p_a)


def compute_file_acc_at_k(
    retrieved_files: Sequence[str],
    gold_files: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> Dict[int, float]:
    """
    Compute whether at least one gold file appears in the top-k retrieved files.
    Returns dictionary mapping k -> 1.0 (hit) or 0.0 (miss).
    """
    res: Dict[int, float] = {}
    if not gold_files:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        top_k = retrieved_files[:k]
        hit = any(
            any(files_match(rf, gf) for gf in gold_files)
            for rf in top_k
        )
        res[k] = 1.0 if hit else 0.0
    return res


def compute_function_acc_at_k(
    retrieved_symbol_ids: Sequence[int],
    gold_symbol_ids: Sequence[int],
    k_values: Sequence[int] = (1, 5, 10, 25),
) -> Dict[int, float]:
    """
    Compute whether at least one gold function symbol ID appears in the top-k retrieved symbols.
    Returns dictionary mapping k -> 1.0 (hit) or 0.0 (miss).
    """
    res: Dict[int, float] = {}
    gold_set: Set[int] = set(gold_symbol_ids)
    if not gold_set:
        return {k: 0.0 for k in k_values}

    for k in k_values:
        top_k_set = set(retrieved_symbol_ids[:k])
        hit = bool(top_k_set & gold_set)
        res[k] = 1.0 if hit else 0.0
    return res


def compute_context_purity(
    retrieved_symbol_ids: Sequence[int],
    retrieved_files: Sequence[str],
    gold_symbol_ids: Sequence[int],
    gold_files: Sequence[str],
) -> float:
    """
    Compute context purity: proportion of retrieved symbols that belong to gold symbols or gold files.
    """
    if not retrieved_symbol_ids and not retrieved_files:
        return 0.0

    gold_sym_set = set(gold_symbol_ids)
    sym_hits = sum(1 for sid in retrieved_symbol_ids if sid in gold_sym_set)

    total_items = max(1, len(retrieved_symbol_ids))
    return float(sym_hits / total_items)


def compute_coverage_at_budget(
    retrieved_symbol_ids: Sequence[int],
    gold_symbol_ids: Sequence[int],
    symbol_line_counts: Dict[int, int],
    token_budget: int = 4096,
    avg_tokens_per_line: float = 8.0,
) -> float:
    """
    Compute Context Coverage@K: fraction of gold symbols retained within token budget K.
    """
    gold_set = set(gold_symbol_ids)
    if not gold_set:
        return 0.0

    accumulated_tokens = 0.0
    covered_gold: Set[int] = set()

    for sid in retrieved_symbol_ids:
        lines = max(5, symbol_line_counts.get(sid, 15))
        tokens = lines * avg_tokens_per_line
        if accumulated_tokens + tokens > token_budget:
            break
        accumulated_tokens += tokens
        if sid in gold_set:
            covered_gold.add(sid)

    return float(len(covered_gold) / len(gold_set))


def compute_hsi(
    retrieved_symbol_ids: Sequence[int],
    hub_ids: Set[int],
    top_k: int = 10,
) -> float:
    """
    Compute Hub Suppression Index (HSI):
        HSI = 1 - (|Top-k retrieved intersect Hubs| / k)
    """
    if not retrieved_symbol_ids or top_k <= 0:
        return 1.0
    top_k_syms = set(retrieved_symbol_ids[:top_k])
    actual_k = min(top_k, len(retrieved_symbol_ids))
    hub_count = len(top_k_syms & hub_ids)
    return float(1.0 - (hub_count / actual_k))


def compute_relative_hsi(
    retrieved_symbol_ids: Sequence[int],
    in_degrees: np.ndarray,
    percentile: float = 99.0,
    top_k: int = 10,
) -> float:
    """
    Compute HSI against relative in-degree percentiles (e.g. top 1.0% or 0.5% hubs).
    """
    if len(in_degrees) == 0 or not retrieved_symbol_ids or top_k <= 0:
        return 1.0
    threshold = float(np.percentile(in_degrees, percentile))
    actual_k = min(top_k, len(retrieved_symbol_ids))
    hub_count = sum(1 for sid in retrieved_symbol_ids[:actual_k] if 0 <= sid < len(in_degrees) and in_degrees[sid] >= threshold)
    return float(1.0 - (hub_count / actual_k))


def compute_bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute sample mean and bootstrap confidence interval.
    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    n = len(arr)
    sample_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(arr[sample_indices], axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))
    return mean_val, lower, upper


def compute_paired_repo_stratified_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    repo_labels: Sequence[str],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute paired repo-stratified bootstrap confidence interval for difference Delta = Score_A - Score_B.
    Guarantees that resamples preserve the exact repository distribution.
    Returns (mean_delta, ci_lower, ci_upper).
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    assert len(arr_a) == len(arr_b) == len(repo_labels)

    delta = arr_a - arr_b
    mean_delta = float(np.mean(delta))

    # Group sample indices by repository
    repo_indices: Dict[str, np.ndarray] = {}
    for idx, r in enumerate(repo_labels):
        repo_indices.setdefault(r, []).append(idx)
    for r in repo_indices:
        repo_indices[r] = np.asarray(repo_indices[r], dtype=int)

    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_bootstrap, dtype=np.float64)
    n_total = len(delta)

    for b in range(n_bootstrap):
        resampled_indices = []
        for r, idxs in repo_indices.items():
            k = len(idxs)
            picked = idxs[rng.integers(0, k, size=k)]
            resampled_indices.extend(picked)
        boot_deltas[b] = np.mean(delta[resampled_indices])

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_deltas, 100.0 * alpha))
    upper = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha)))
    return mean_delta, lower, upper


def compute_exact_mcnemar_test(
    y_target: Sequence[int | float],
    y_baseline: Sequence[int | float],
) -> Dict[str, Any]:
    """
    Compute exact two-sided McNemar paired test on binary classification/ranking outcomes.
    Uses exact binomial distribution on discordant pairs (n01 vs n10).
    """
    from scipy.stats import binomtest

    t_arr = (np.asarray(y_target) > 0).astype(int)
    b_arr = (np.asarray(y_baseline) > 0).astype(int)
    assert len(t_arr) == len(b_arr)

    # Contingency table
    # n11: both 1; n00: both 0
    # n10: target=1, baseline=0 (target win)
    # n01: target=0, baseline=1 (baseline win)
    n10 = int(np.sum((t_arr == 1) & (b_arr == 0)))
    n01 = int(np.sum((t_arr == 0) & (b_arr == 1)))
    n11 = int(np.sum((t_arr == 1) & (b_arr == 1)))
    n00 = int(np.sum((t_arr == 0) & (b_arr == 0)))

    n_discordant = n10 + n01
    if n_discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test under null hypothesis p = 0.5
        k_min = min(n10, n01)
        res = binomtest(k_min, n_discordant, p=0.5, alternative="two-sided")
        p_value = float(res.pvalue)

    return {
        "n_target_wins": n10,
        "n_baseline_wins": n01,
        "n_both_hit": n11,
        "n_both_miss": n00,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def compute_holm_bonferroni_correction(
    p_values: Dict[str, float]
) -> Dict[str, float]:
    """
    Apply step-down Holm-Bonferroni multi-testing correction.
    Returns dictionary mapping baseline name to adjusted p-value.
    """
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adj_p: Dict[str, float] = {}

    running_max = 0.0
    for rank, (name, pval) in enumerate(items, start=1):
        multiplier = m - rank + 1
        adj_val = min(1.0, pval * multiplier)
        running_max = max(running_max, adj_val)
        adj_p[name] = float(min(1.0, running_max))

    return adj_p
