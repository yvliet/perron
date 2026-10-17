#!/usr/bin/env python3
"""
PPR + LLM Score Fusion Ablation for Code Defect Candidate Reranking.

Evaluates convex combinations, reciprocal rank fusion (RRF), and Bayesian
log-odds prior weighting of Personalized PageRank (PPR) mass with Gemma 4
pointwise log-odds across SWE-bench Verified (N=344) and Stratum C (N=131).

Investigates whether incorporating structural graph mass into downstream
candidate scoring rescues graph targets or amplifies lexical seed bias.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.run_gemma4_protocol import (
    VERIFIED_GOLD_FILE,
    VERIFIED_TASKS_FILE,
    get_repo_slug,
    load_engines,
)
from perron.reranker.candidate_pools import CandidatePoolBuilder
from perron.specificity import calculate_specificity_scores

OUTPUT_JSON = PROJECT_ROOT / "results" / "ppr_score_fusion_results.json"
SCORES_PARQUET = PROJECT_ROOT / "data_release" / "scores.parquet"
STRATA_FILE = PROJECT_ROOT / "data" / "swebench_strata_annotations.jsonl"


def min_max_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalizes array to [0, 1]. Returns zeros if constant."""
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        return (arr - mn) / (mx - mn + 1e-12)
    return np.zeros_like(arr)


def compute_metrics(
    ranked_cands: List[int],
    gold_ids: Set[int],
) -> Tuple[int, int, int, int, float]:
    """Computes Top-1, 3, 5, 10 hit indicators and reciprocal rank."""
    h1 = int(any(c in gold_ids for c in ranked_cands[:1]))
    h3 = int(any(c in gold_ids for c in ranked_cands[:3]))
    h5 = int(any(c in gold_ids for c in ranked_cands[:5]))
    h10 = int(any(c in gold_ids for c in ranked_cands[:10]))
    rr = 0.0
    for idx, c in enumerate(ranked_cands, 1):
        if c in gold_ids:
            rr = 1.0 / idx
            break
    return h1, h3, h5, h10, rr


def exact_mcnemar_p(hits_a: List[int], hits_b: List[int]) -> Tuple[int, int, float]:
    """Computes discordant pairs and exact two-sided McNemar p-value."""
    n10 = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        return 0, 0, 1.0
    k = min(n10, n01)
    p_val = float(2.0 * stats.binom.cdf(k, n_disc, 0.5))
    return n10, n01, min(1.0, max(0.0, p_val))


def main() -> None:
    print("[INFO] Loading scores.parquet and task metadata...")
    df_scores = pd.read_parquet(SCORES_PARQUET)
    score_map: Dict[Tuple[str, int], float] = {
        (row["instance_id"], int(row["candidate_id"])): float(row["s"])
        for _, row in df_scores.iterrows()
    }

    strata_map: Dict[str, str] = {}
    with open(STRATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                strata_map[d["instance_id"]] = d.get("stratum", "Unknown")

    engines, global_pr, layers = load_engines()
    tasks = [json.loads(l) for l in open(VERIFIED_TASKS_FILE, encoding="utf-8") if l.strip()]
    gold_labels = json.load(open(VERIFIED_GOLD_FILE, encoding="utf-8"))

    builders = {
        slug: CandidatePoolBuilder(engine=eng, layers=layers.get(slug))
        for slug, eng in engines.items()
    }

    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    lambdas = [0.1, 0.25, 0.5, 1.0, 2.0]
    rrf_ks = [10, 60]

    # Tracking per condition
    cohorts = ["all", "stratum_a", "stratum_b", "stratum_c"]
    stats_by_cohort: Dict[str, Dict[str, Any]] = {c: defaultdict(dict) for c in cohorts}
    hits_tracker: Dict[str, Dict[str, List[int]]] = {c: defaultdict(list) for c in cohorts}

    processed_tasks = 0
    for task in tasks:
        iid = task["instance_id"]
        slug = get_repo_slug(task.get("repo", ""))
        eng = engines[slug]
        gold_ids = set(gold_labels[iid].get("gold_symbol_ids", []))
        stratum = strata_map.get(iid, "Unknown")
        c_name = f"stratum_{stratum.lower()}"

        q_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")
        p0 = eng.compute_query_prior(q_text, mode="softmax", k=25, tau=0.15)
        pi_q = eng.compute_personalized_pagerank(p0, beta=0.85)
        pi_g = global_pr[slug]
        s_static = calculate_specificity_scores(
            pi_q, pi_g, gamma=0.70, is_test_node=eng.is_test_mask, filter_test_nodes=True
        )

        pools, _ = builders[slug].build_all_pools(
            query_text=q_text,
            instance_id=iid,
            perron_scores=s_static,
            gold_symbol_ids=gold_ids,
        )
        a4_cands = pools["A4"]
        if not a4_cands:
            continue

        processed_tasks += 1

        # Scores
        llm_raw = np.array([score_map.get((iid, c), -10.0) for c in a4_cands])
        ppr_raw = np.array([float(pi_q[c]) for c in a4_cands])
        spec_raw = np.array([float(s_static[c]) for c in a4_cands])

        norm_llm = min_max_normalize(llm_raw)
        norm_ppr = min_max_normalize(ppr_raw)
        norm_spec = min_max_normalize(spec_raw)

        # Baseline BM25 in pool order
        b0_order = pools["B0"][: len(a4_cands)]

        # 1. RRF
        order_llm = np.argsort(-llm_raw)
        ranks_llm = np.empty_like(order_llm)
        ranks_llm[order_llm] = np.arange(1, len(a4_cands) + 1)

        order_ppr = np.argsort(-ppr_raw)
        ranks_ppr = np.empty_like(order_ppr)
        ranks_ppr[order_ppr] = np.arange(1, len(a4_cands) + 1)

        for k_val in rrf_ks:
            rrf_score = (1.0 / (k_val + ranks_llm)) + (1.0 / (k_val + ranks_ppr))
            ranked = [a4_cands[i] for i in np.argsort(-rrf_score)]
            h1, h3, h5, h10, rr = compute_metrics(ranked, gold_ids)
            for c_key in ["all", c_name]:
                cond_key = f"rrf_k{k_val}"
                hits_tracker[c_key][cond_key].append(h5)
                entry = stats_by_cohort[c_key][cond_key]
                entry["acc1"] = entry.get("acc1", 0) + h1
                entry["acc3"] = entry.get("acc3", 0) + h3
                entry["acc5"] = entry.get("acc5", 0) + h5
                entry["acc10"] = entry.get("acc10", 0) + h10
                entry["mrr"] = entry.get("mrr", 0.0) + rr
                entry["n"] = entry.get("n", 0) + 1

        # 2. Alpha Convex Combination (PPR)
        for alpha in alphas:
            fused = alpha * norm_llm + (1.0 - alpha) * norm_ppr
            ranked = [a4_cands[i] for i in np.argsort(-fused)]
            h1, h3, h5, h10, rr = compute_metrics(ranked, gold_ids)
            for c_key in ["all", c_name]:
                cond_key = f"alpha_ppr_{alpha:.1f}"
                hits_tracker[c_key][cond_key].append(h5)
                entry = stats_by_cohort[c_key][cond_key]
                entry["acc1"] = entry.get("acc1", 0) + h1
                entry["acc3"] = entry.get("acc3", 0) + h3
                entry["acc5"] = entry.get("acc5", 0) + h5
                entry["acc10"] = entry.get("acc10", 0) + h10
                entry["mrr"] = entry.get("mrr", 0.0) + rr
                entry["n"] = entry.get("n", 0) + 1

        # 3. Alpha Convex Combination (Specificity)
        for alpha in alphas:
            fused = alpha * norm_llm + (1.0 - alpha) * norm_spec
            ranked = [a4_cands[i] for i in np.argsort(-fused)]
            h1, h3, h5, h10, rr = compute_metrics(ranked, gold_ids)
            for c_key in ["all", c_name]:
                cond_key = f"alpha_spec_{alpha:.1f}"
                hits_tracker[c_key][cond_key].append(h5)
                entry = stats_by_cohort[c_key][cond_key]
                entry["acc1"] = entry.get("acc1", 0) + h1
                entry["acc3"] = entry.get("acc3", 0) + h3
                entry["acc5"] = entry.get("acc5", 0) + h5
                entry["acc10"] = entry.get("acc10", 0) + h10
                entry["mrr"] = entry.get("mrr", 0.0) + rr
                entry["n"] = entry.get("n", 0) + 1

        # 4. Bayesian Prior Log-Odds
        for lam in lambdas:
            fused = llm_raw + lam * np.log(ppr_raw + 1e-12)
            ranked = [a4_cands[i] for i in np.argsort(-fused)]
            h1, h3, h5, h10, rr = compute_metrics(ranked, gold_ids)
            for c_key in ["all", c_name]:
                cond_key = f"bayes_lambda_{lam:.2f}"
                hits_tracker[c_key][cond_key].append(h5)
                entry = stats_by_cohort[c_key][cond_key]
                entry["acc1"] = entry.get("acc1", 0) + h1
                entry["acc3"] = entry.get("acc3", 0) + h3
                entry["acc5"] = entry.get("acc5", 0) + h5
                entry["acc10"] = entry.get("acc10", 0) + h10
                entry["mrr"] = entry.get("mrr", 0.0) + rr
                entry["n"] = entry.get("n", 0) + 1

    # Format summary results
    final_output: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_tasks_processed": processed_tasks,
        "cohorts": {},
    }

    ref_hits_all = hits_tracker["all"]["alpha_ppr_1.0"]

    for c_key, cond_dict in stats_by_cohort.items():
        cohort_summary: Dict[str, Any] = {}
        ref_hits = hits_tracker[c_key]["alpha_ppr_1.0"]

        for cond, metrics in cond_dict.items():
            n = metrics["n"]
            if n == 0:
                continue
            cond_hits = hits_tracker[c_key][cond]
            n10, n01, p_val = exact_mcnemar_p(cond_hits, ref_hits)
            cohort_summary[cond] = {
                "n": n,
                "acc1_pct": round(metrics["acc1"] / n * 100.0, 2),
                "acc3_pct": round(metrics["acc3"] / n * 100.0, 2),
                "acc5_pct": round(metrics["acc5"] / n * 100.0, 2),
                "acc10_pct": round(metrics["acc10"] / n * 100.0, 2),
                "mrr": round(metrics["mrr"] / n, 4),
                "mcnemar_vs_pure_llm": {
                    "n10": n10,
                    "n01": n01,
                    "delta_pp": round((metrics["acc5"] - stats_by_cohort[c_key]["alpha_ppr_1.0"]["acc5"]) / n * 100.0, 2),
                    "p_value": round(p_val, 4),
                },
            }
        final_output["cohorts"][c_key] = cohort_summary

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"[SUCCESS] Wrote PPR + LLM Score Fusion results to {OUTPUT_JSON}")
    print("\n=== SUMMARY (Stratum C, N=131) ===")
    stratum_c = final_output["cohorts"]["stratum_c"]
    print(f"{'Condition':<20} | {'Acc@1':<8} | {'Acc@5':<8} | {'MRR':<8} | {'Delta Acc@5 vs LLM':<18} | {'p-val':<8}")
    print("-" * 80)
    for k in ["alpha_ppr_0.0", "alpha_ppr_0.2", "alpha_ppr_0.5", "alpha_ppr_0.8", "alpha_ppr_1.0", "rrf_k10", "rrf_k60", "bayes_lambda_0.50"]:
        if k in stratum_c:
            row = stratum_c[k]
            delta = row["mcnemar_vs_pure_llm"]["delta_pp"]
            p_v = row["mcnemar_vs_pure_llm"]["p_value"]
            print(f"{k:<20} | {row['acc1_pct']:<7}% | {row['acc5_pct']:<7}% | {row['mrr']:<8.4f} | {delta:+6.2f} pp          | {p_v:<8.4f}")


if __name__ == "__main__":
    main()
