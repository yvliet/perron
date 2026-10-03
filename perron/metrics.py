"""
Authoritative Evaluation Metrics for Code Graph Retrieval and Agent Navigation.

Implements standardized benchmarks metrics:
- Function Acc@K: Hit (1.0) if ANY ground-truth gold function is present in top K retrieved symbols, else (0.0).
- File Acc@K: Hit (1.0) if file path of ANY gold function is present in top K retrieved files, else (0.0).
- Coverage@4k: Fraction of unique gold functions whose context fits in a packed 4,096-token budget.
- HSI@10: Hub Suppression Index at top 10 = 1.0 - (|top10 ∩ H25| / 10.0).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union


def normalize_file_path(path: str) -> str:
    """Normalize file path for cross-platform and relative matching."""
    if not path:
        return ""
    p = path.replace("\\", "/").strip().lstrip("/")
    if p.startswith("./"):
        p = p[2:]
    return p


def files_match(path_a: str, path_b: str) -> bool:
    """
    Check if two file paths refer to the same file.

    Matches if exact match, or if one path is a valid relative suffix of the other.
    """
    p_a = normalize_file_path(path_a)
    p_b = normalize_file_path(path_b)
    if not p_a or not p_b:
        return False
    if p_a == p_b:
        return True
    return p_a.endswith("/" + p_b) or p_b.endswith("/" + p_a)


def function_acc_at_k(
    retrieved: Sequence[Any],
    gold: Sequence[Any],
    k: int = 1,
) -> float:
    """
    Compute Function Acc@K.

    Returns 1.0 if ANY gold function symbol is in the top-K retrieved symbols, else 0.0.
    Handles empty gold, duplicate IDs, K > len(candidates), and missing IDs gracefully.
    """
    if not gold or k <= 0 or not retrieved:
        return 0.0

    gold_set: Set[Any] = set(gold)
    if not gold_set:
        return 0.0

    # Evaluate the top-k candidates (up to available candidate length)
    top_k_candidates = retrieved[:k]
    # Set intersection handles duplicate IDs in retrieved candidates
    hit = any(cand in gold_set for cand in top_k_candidates)
    return 1.0 if hit else 0.0


def file_acc_at_k(
    retrieved_files: Sequence[str],
    gold_files: Sequence[str],
    k: int = 1,
) -> float:
    """
    Compute File Acc@K.

    Returns 1.0 if ANY gold file matches any file in the top-K retrieved files, else 0.0.
    """
    if not gold_files or k <= 0 or not retrieved_files:
        return 0.0

    cleaned_gold = [normalize_file_path(gf) for gf in gold_files if gf]
    if not cleaned_gold:
        return 0.0

    top_k_files = retrieved_files[:k]
    for rf in top_k_files:
        if any(files_match(rf, gf) for gf in cleaned_gold):
            return 1.0
    return 0.0


def coverage_at_4k(
    retrieved_symbols: Sequence[Any],
    gold_symbols: Sequence[Any],
    symbol_token_costs: Optional[Union[Dict[Any, int], Sequence[int]]] = None,
    budget: int = 4096,
    default_cost: int = 120,
) -> float:
    """
    Compute Coverage@4k: fraction of unique gold functions whose breadcrumbs fit in budget.

    Parameters
    ----------
    retrieved_symbols : Sequence[Any]
        Ranked list of retrieved symbol identifiers.
    gold_symbols : Sequence[Any]
        Ground-truth modified symbol identifiers.
    symbol_token_costs : Dict[Any, int] or Sequence[int], optional
        Token costs per symbol. If dictionary, mapped by symbol ID. If sequence, mapped by index.
        Defaults to `default_cost` (120 tokens ~ 15 lines of code) per symbol.
    budget : int, optional
        Context budget in tokens (default 4096).
    default_cost : int, optional
        Default token cost if unmapped (default 120).

    Returns
    -------
    float
        Fraction in [0.0, 1.0] of unique gold functions packed before budget is exhausted.
    """
    if not gold_symbols:
        return 0.0
    if budget <= 0 or not retrieved_symbols:
        return 0.0

    gold_set: Set[Any] = set(gold_symbols)
    if not gold_set:
        return 0.0

    accumulated_tokens = 0
    covered_gold: Set[Any] = set()
    seen_symbols: Set[Any] = set()

    for idx, sid in enumerate(retrieved_symbols):
        if sid in seen_symbols:
            continue
        seen_symbols.add(sid)

        # Determine token cost
        cost = default_cost
        if symbol_token_costs is not None:
            if isinstance(symbol_token_costs, dict):
                cost = symbol_token_costs.get(sid, default_cost)
            elif isinstance(symbol_token_costs, (list, tuple)) and idx < len(symbol_token_costs):
                cost = symbol_token_costs[idx]

        cost = max(1, int(cost))
        if accumulated_tokens + cost > budget:
            break

        accumulated_tokens += cost
        if sid in gold_set:
            covered_gold.add(sid)

    return float(len(covered_gold) / len(gold_set))


def hsi_at_10(
    top_retrieved: Sequence[Any],
    top25_hubs: Union[Sequence[Any], Set[Any]],
) -> float:
    """
    Compute Hub Suppression Index (HSI) at top-10.

    Formula:
        HSI@10 = 1.0 - (|top10 ∩ H25| / 10.0)

    Measures resistance to congestion by ubiquitous utility hub nodes.
    Higher is better (1.0 = zero hubs in top-10; 0.0 = top-10 is saturated with hubs).
    """
    hub_set: Set[Any] = set(top25_hubs)
    top_10 = top_retrieved[:10]
    # Count unique hub occurrences in the top-10 slice
    hub_overlap = sum(1 for item in top_10 if item in hub_set)
    # Scaled by 10.0 per formal specification
    score = 1.0 - (hub_overlap / 10.0)
    return max(0.0, min(1.0, float(score)))
