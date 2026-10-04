#!/usr/bin/env python3
"""
Confirmatory Single-Shot Evaluation on SWE-bench Verified Tasks.

Evaluates frozen retrieval methods across verified bug localization tasks from SWE-bench Verified
matching the 12 canonical repositories, completely uncontaminated by exploratory SWE-bench Lite development.

Implements pre-registered statistical battery:
- Primary metric: Function Acc@10 on callable tasks (S_func)
- Secondary metrics: Function Acc@5, File Acc@5, HSI@10
- Stratum-disaggregated metrics (Stratum A, B, C)
- Paired repo-stratified bootstrap confidence intervals (B = 10,000)
- Exact two-sided McNemar paired tests vs. Perron Decoupled Specificity
- Step-down Holm-Bonferroni multiple testing correction
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import scipy.sparse as sp

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.eval_harness import (
    compute_bootstrap_ci,
    compute_exact_mcnemar_test,
    compute_holm_bonferroni_correction,
    compute_paired_repo_stratified_bootstrap_ci,
)
from benchmarks.gold_patch_parser import GoldPatchParser, get_repo_slug
from perron.metrics import file_acc_at_k, function_acc_at_k, hsi_at_10
from perron.results import get_git_sha
from perron.specificity import calculate_specificity_scores, compute_global_pagerank

SEED = 20261003
GRAPHS_DIR = PROJECT_ROOT / "artifacts" / "swebench_graphs"
TASKS_FILE = PROJECT_ROOT / "data" / "swebench_verified_tasks.jsonl"
GOLD_FILE = PROJECT_ROOT / "data" / "swebench_verified_gold_labels.json"
OUT_FILE = PROJECT_ROOT / "results" / "verified_confirmatory_evaluation.json"

FROZEN_BETA_Q = 0.85
FROZEN_BETA_G = 0.85
FROZEN_GAMMA = 0.70
FROZEN_TAU = 0.15
FROZEN_K_PRIOR = 25


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def query_text(task: Dict[str, Any]) -> str:
    return (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")


def determine_stratum(text: str, gold_names: Sequence[str]) -> str:
    """Classify task into Stratum A (named), B (trace), or C (unnamed)."""
    text_lower = text.lower()
    for name in gold_names:
        short = name.split(".")[-1].lower()
        if len(short) >= 3 and short in text_lower:
            return "A"
    if "traceback" in text_lower or "file \"" in text_lower:
        return "B"
    return "C"


def reciprocal_rank_fusion(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    k: int = 60,
    top_n: int = 25,
) -> List[int]:
    """Reciprocal Rank Fusion across two node ranking lists."""
    scores: Dict[int, float] = defaultdict(float)
    for rank, node_id in enumerate(ranking_a):
        scores[node_id] += 1.0 / (k + rank + 1)
    for rank, node_id in enumerate(ranking_b):
        scores[node_id] += 1.0 / (k + rank + 1)
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [item[0] for item in sorted_items[:top_n]]


def load_verified_gold_labels(parser: GoldPatchParser, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Load or extract gold labels for SWE-bench Verified tasks."""
    if GOLD_FILE.exists():
        print(f"Loading cached gold labels from {GOLD_FILE}...")
        return json.loads(GOLD_FILE.read_text(encoding="utf-8"))

    print(f"Extracting gold labels for {len(tasks)} Verified tasks...")
    labels: Dict[str, Dict[str, Any]] = {}
    for idx, task in enumerate(tasks):
        info = parser.extract_gold_for_task(task)
        labels[info["instance_id"]] = info
        if (idx + 1) % 50 == 0 or idx + 1 == len(tasks):
            print(f"  Extracted {idx + 1}/{len(tasks)} labels...")

    GOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    GOLD_FILE.write_text(json.dumps(labels, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(labels)} gold labels to {GOLD_FILE}")
    return labels


def run_verified_evaluation(limit: Optional[int] = None) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"   SWE-bench Verified Confirmatory Evaluation Gate     ")
    print(f"=======================================================\n")
    print(f"Seed: {SEED}, Frozen beta_q={FROZEN_BETA_Q}, beta_g={FROZEN_BETA_G}, gamma={FROZEN_GAMMA}")

    if not TASKS_FILE.exists():
        raise FileNotFoundError(f"Verified tasks file not found at {TASKS_FILE}. Run scratch/fetch_verified.py first.")

    with open(TASKS_FILE, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit:
        tasks = tasks[:limit]

    parser = GoldPatchParser(graphs_dir=GRAPHS_DIR)
    labels = load_verified_gold_labels(parser, tasks)

    # Load graphs
    graphs = {}
    for p in sorted(GRAPHS_DIR.glob("*_graph.npz")):
        slug = p.stem.replace("_graph", "")
        sym_p = GRAPHS_DIR / f"{slug}_symbols.json"
        if sym_p.exists():
            graphs[slug] = (sp.load_npz(p), json.loads(sym_p.read_text(encoding="utf-8")))

    # Precompute global PageRank vectors
    global_pagerank_085: Dict[str, np.ndarray] = {}
    for slug, (g, syms) in graphs.items():
        dangling = np.zeros(g.shape[0], dtype=np.float32)
        global_pagerank_085[slug] = compute_global_pagerank(g, dangling, beta=FROZEN_BETA_G)

    # Instantiate engines with frozen settings
    engines = {
        slug: RepositoryRetrievalEngine(
            g, syms,
            beta=FROZEN_BETA_Q,
            gamma=FROZEN_GAMMA,
            filter_test_nodes=True,
            prior_mode="softmax",
            tau=FROZEN_TAU,
            k_prior=FROZEN_K_PRIOR,
        )
        for slug, (g, syms) in graphs.items()
    }

    # Methods evaluated
    methods = [
        "bm25",
        "standard_ppr",
        "deg_ppr",
        "hipporag",
        "perron_static",
        "perron_adaptive",
        "rrf_fusion",
    ]

    per_task_results: Dict[str, Dict[str, Any]] = {}
    callable_iids: List[str] = []
    repo_by_iid: Dict[str, str] = {}
    stratum_by_iid: Dict[str, str] = {}

    print(f"\nEvaluating {len(tasks)} tasks across {len(methods)} methods...")
    for idx, task in enumerate(tasks):
        iid = task["instance_id"]
        repo = task.get("repo", "")
        slug = get_repo_slug(repo)
        eng = engines.get(slug)
        gold = labels.get(iid)
        if eng is None or not gold:
            continue

        text = query_text(task)
        gold_ids = set(gold.get("gold_symbol_ids", []))
        gold_files = set(gold.get("gold_files", []))
        has_func = len(gold_ids) > 0
        repo_by_iid[iid] = slug
        if has_func:
            callable_iids.append(iid)

        stratum = determine_stratum(text, gold.get("gold_symbol_names", []))
        stratum_by_iid[iid] = stratum

        # Teleport prior (sharp softmax)
        p_0 = eng.compute_query_prior(text, mode="softmax", k=FROZEN_K_PRIOR, tau=FROZEN_TAU)
        pi_q = eng.compute_personalized_pagerank(p_0, beta=FROZEN_BETA_Q)
        pi_g = global_pagerank_085[slug]

        # 1. BM25
        bm25_rank = eng.retrieve_bm25(text)

        # 2. Standard PPR
        std_rank = eng._rank_scores(pi_q)

        # 3. Degree-Normalized PPR
        denom_deg = (eng.in_degrees + 1.0) ** FROZEN_GAMMA
        deg_rank = eng._rank_scores(pi_q / denom_deg)

        # 4. HippoRAG
        denom_hippo = (pi_g + 1e-8) ** FROZEN_GAMMA
        p0_hippo = p_0 / denom_hippo
        if float(np.sum(p0_hippo)) > 0:
            p0_hippo = p0_hippo / np.sum(p0_hippo)
        pi_hippo = eng.compute_personalized_pagerank(p0_hippo, beta=FROZEN_BETA_Q)
        hippo_rank = eng._rank_scores(pi_hippo)

        # 5. Perron Static Specificity
        s_static = calculate_specificity_scores(
            pi_q, pi_g, gamma=FROZEN_GAMMA,
            is_test_node=eng.is_test_mask,
            filter_test_nodes=True,
        )
        perron_rank = eng._rank_scores(s_static)

        # 6. Perron Adaptive Specificity
        s_adapt = calculate_specificity_scores(
            pi_q, pi_g, gamma=FROZEN_GAMMA,
            is_test_node=eng.is_test_mask,
            filter_test_nodes=True,
            query_prior=p_0,
            hub_indices=eng.hub_indices,
        )
        perron_adapt_rank = eng._rank_scores(s_adapt)

        # 7. RRF Fusion (BM25 + Perron)
        rrf_rank = reciprocal_rank_fusion(bm25_rank, perron_rank, k=60, top_n=25)

        rankings = {
            "bm25": bm25_rank,
            "standard_ppr": std_rank,
            "deg_ppr": deg_rank,
            "hipporag": hippo_rank,
            "perron_static": perron_rank,
            "perron_adaptive": perron_adapt_rank,
            "rrf_fusion": rrf_rank,
        }

        task_eval = {}
        for m, rank_list in rankings.items():
            top10 = rank_list[:10]
            top5 = rank_list[:5]
            top10_ids = [eng.symbols[i]["id"] for i in top10]
            top5_ids = [eng.symbols[i]["id"] for i in top5]
            top5_files = [eng.symbols[i].get("file_path", "") for i in top5]

            fn_10 = function_acc_at_k(top10_ids, list(gold_ids), k=10) if has_func else 0.0
            fn_5 = function_acc_at_k(top5_ids, list(gold_ids), k=5) if has_func else 0.0
            file_5 = file_acc_at_k(top5_files, list(gold_files), k=5)
            hsi = hsi_at_10(top10_ids, eng.hub_set)

            task_eval[m] = {
                "fn_acc_10": fn_10,
                "fn_acc_5": fn_5,
                "file_acc_5": file_5,
                "hsi": hsi,
            }
        per_task_results[iid] = task_eval

        if (idx + 1) % 50 == 0 or idx + 1 == len(tasks):
            print(f"  Evaluated {idx + 1}/{len(tasks)} tasks...")

    n_callable = len(callable_iids)
    n_all = len(per_task_results)
    print(f"\nEvaluation complete: {n_all} total tasks, {n_callable} callable tasks (S_func).")

    # Aggregate summaries
    summary_callable = {}
    summary_all = {}

    for m in methods:
        fn10_vals = [per_task_results[i][m]["fn_acc_10"] for i in callable_iids]
        fn5_vals = [per_task_results[i][m]["fn_acc_5"] for i in callable_iids]
        file5_vals = [per_task_results[i][m]["file_acc_5"] for i in per_task_results]
        hsi_vals = [per_task_results[i][m]["hsi"] for i in per_task_results]

        mean_10, lo_10, hi_10 = compute_bootstrap_ci(fn10_vals, n_bootstrap=10000, ci=0.95, seed=SEED)
        mean_5, lo_5, hi_5 = compute_bootstrap_ci(fn5_vals, n_bootstrap=10000, ci=0.95, seed=SEED)
        mean_file, lo_file, hi_file = compute_bootstrap_ci(file5_vals, n_bootstrap=10000, ci=0.95, seed=SEED)
        mean_hsi, lo_hsi, hi_hsi = compute_bootstrap_ci(hsi_vals, n_bootstrap=10000, ci=0.95, seed=SEED)

        summary_callable[m] = {
            "fn_acc_10": {"mean": mean_10, "ci_lower": lo_10, "ci_upper": hi_10, "hits": int(sum(fn10_vals)), "n": n_callable},
            "fn_acc_5": {"mean": mean_5, "ci_lower": lo_5, "ci_upper": hi_5, "hits": int(sum(fn5_vals)), "n": n_callable},
        }
        summary_all[m] = {
            "file_acc_5": {"mean": mean_file, "ci_lower": lo_file, "ci_upper": hi_file, "hits": int(sum(file5_vals)), "n": n_all},
            "hsi": {"mean": mean_hsi, "ci_lower": lo_hsi, "ci_upper": hi_hsi, "n": n_all},
        }

    # Statistical hypothesis tests vs. Perron Decoupled Static
    perron_fn10 = [per_task_results[i]["perron_static"]["fn_acc_10"] for i in callable_iids]
    repos_callable = [repo_by_iid[i] for i in callable_iids]

    pairwise_tests = {}
    p_values_dict = {}

    competing_methods = [m for m in methods if m != "perron_static"]
    for m in competing_methods:
        base_fn10 = [per_task_results[i][m]["fn_acc_10"] for i in callable_iids]
        mcnemar = compute_exact_mcnemar_test(perron_fn10, base_fn10)
        mean_delta, delta_lo, delta_hi = compute_paired_repo_stratified_bootstrap_ci(
            perron_fn10, base_fn10, repos_callable, n_bootstrap=10000, ci=0.95, seed=SEED
        )
        pairwise_tests[m] = {
            "mcnemar": mcnemar,
            "mean_delta": mean_delta,
            "delta_ci_lower": delta_lo,
            "delta_ci_upper": delta_hi,
        }
        p_values_dict[m] = mcnemar["p_value"]

    # Holm-Bonferroni correction
    holm_adjusted = compute_holm_bonferroni_correction(p_values_dict)
    for m in competing_methods:
        pairwise_tests[m]["p_adjusted_holm"] = holm_adjusted[m]

    # Strata breakdown
    strata_counts = defaultdict(int)
    for i in callable_iids:
        strata_counts[stratum_by_iid[i]] += 1

    strata_results = {}
    for st in ("A", "B", "C"):
        st_iids = [i for i in callable_iids if stratum_by_iid[i] == st]
        n_st = len(st_iids)
        st_summary = {"n": n_st}
        if n_st > 0:
            for m in methods:
                hits = sum(per_task_results[i][m]["fn_acc_10"] for i in st_iids)
                st_summary[m] = {"hits": int(hits), "rate": hits / n_st}
        strata_results[st] = st_summary

    oracle_hits = sum(1 for i in callable_iids if per_task_results[i]["bm25"]["fn_acc_10"] or per_task_results[i]["perron_static"]["fn_acc_10"])
    oracle_union = {
        "hits": int(oracle_hits),
        "n": n_callable,
        "rate": oracle_hits / max(1, n_callable),
    }

    out = {
        "suite": "verified_confirmatory_evaluation",
        "benchmark": "SWE-bench_Verified",
        "git_sha": get_git_sha(),
        "utc_timestamp": now_utc(),
        "seed": SEED,
        "frozen_hyperparameters": {
            "beta_q": FROZEN_BETA_Q,
            "beta_g": FROZEN_BETA_G,
            "gamma": FROZEN_GAMMA,
            "tau": FROZEN_TAU,
            "k_prior": FROZEN_K_PRIOR,
        },
        "num_total_instances": n_all,
        "num_callable_instances": n_callable,
        "methods": methods,
        "summary_callable": summary_callable,
        "summary_all": summary_all,
        "pairwise_vs_perron_static": pairwise_tests,
        "strata_counts": dict(strata_counts),
        "strata_results": strata_results,
        "oracle_union": oracle_union,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[CONFIRMATORY RUN COMPLETE] Output saved to {OUT_FILE}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<20} | {'Fn Acc@10 (S_func)':<22} | {'Fn Acc@5':<12} | {'File Acc@5':<12} | {'HSI@10':<8}")
    print("-" * 80)
    for m in methods:
        fn10 = summary_callable[m]["fn_acc_10"]
        fn5 = summary_callable[m]["fn_acc_5"]
        file5 = summary_all[m]["file_acc_5"]
        hsi = summary_all[m]["hsi"]
        print(f"{m:<20} | {fn10['mean']*100:.1f}% ({fn10['hits']}/{fn10['n']}) [{fn10['ci_lower']*100:.1f}, {fn10['ci_upper']*100:.1f}] | {fn5['mean']*100:.1f}%      | {file5['mean']*100:.1f}%      | {hsi['mean']*100:.1f}%")
    print("=" * 80)

    # Print hypothesis verdicts
    print("\nPre-Registered Hypothesis Outcomes:")
    p1_test = pairwise_tests.get("standard_ppr", {})
    p1_pval = p1_test.get("p_adjusted_holm", 1.0)
    p1_delta = p1_test.get("mean_delta", 0.0)
    print(f"  P1 (Perron vs Standard PPR at b=0.85): Delta={p1_delta*100:+.1f} pp, Holm p={p1_pval:.4e} -> {'CONFIRMED' if p1_delta > 0 and p1_pval < 0.05 else 'NOT SIGNIFICANT'}")

    bm25_delta = pairwise_tests.get("bm25", {}).get("mean_delta", 0.0)
    print(f"  P2 (Perron non-inferiority to BM25): Delta={bm25_delta*100:+.1f} pp (margin -2.0 pp) -> {'CONFIRMED' if bm25_delta >= -0.02 else 'REJECTED'}")

    rrf_hits = summary_callable["rrf_fusion"]["fn_acc_10"]["hits"]
    bm25_hits = summary_callable["bm25"]["fn_acc_10"]["hits"]
    perron_hits = summary_callable["perron_static"]["fn_acc_10"]["hits"]
    print(f"  P3 (RRF Fusion Complementarity): RRF={rrf_hits} hits vs BM25={bm25_hits} vs Perron={perron_hits} -> {'CONFIRMED' if rrf_hits >= max(bm25_hits, perron_hits) else 'NO GAIN'}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE-bench Verified Confirmatory Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to evaluate")
    args = parser.parse_args()
    run_verified_evaluation(limit=args.limit)


if __name__ == "__main__":
    main()
