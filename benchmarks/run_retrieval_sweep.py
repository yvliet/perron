"""
Full 300-Instance Retrieval Sweep across 6 Baselines on SWE-bench Lite.

Evaluates:
1. Okapi BM25
2. Dense Semantic Retrieval
3. Aider-Style Repo Map
4. Hub Blocklist Baseline (Top-25 hubs masked)
5. Degree-Normalized PPR (pi_q / (deg_in + 1)^gamma)
6. Perron Specificity Ratio with Query-Level Hub Adaptivity

Computes:
- File Acc@1, Acc@3, Acc@5
- Function Acc@1, Acc@5, Acc@10, Acc@25
- Context Purity@K (top-25 symbols)
- Diagnostic HSI (top-10)
- 1,000-sample bootstrap 95% confidence intervals on Dev and Held-out splits
- Hub-Gold vs Non-Hub-Gold partition analysis

Saves detailed results to data/retrieval_sweep_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.sparse as sp

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.eval_harness import (
    compute_bootstrap_ci,
    compute_context_purity,
    compute_coverage_at_budget,
    compute_exact_mcnemar_test,
    compute_file_acc_at_k,
    compute_function_acc_at_k,
    compute_holm_bonferroni_correction,
    compute_hsi,
    compute_paired_repo_stratified_bootstrap_ci,
    compute_relative_hsi,
)
from benchmarks.gold_patch_parser import get_repo_slug

BASELINES = [
    ("bm25", "Okapi BM25"),
    ("bm25_1hop", "BM25 + 1-Hop Expansion"),
    ("dense", "Dense Semantic"),
    ("prior_only", "Prior-Only Control (p0)"),
    ("standard_ppr", "Standard PPR (Shared p0)"),
    ("aider", "Authentic Aider Repo Map"),
    ("blocklist_lex", "Hub Blocklist + Lexical Override"),
    ("deg_discount_matrix", "Static Deg-Discount Matrix"),
    ("deg_ppr", "Degree-Normalized PPR"),
    ("hipporag", "HippoRAG Prior Scaling"),
    ("query_reweighted_ppr", "Query-Reweighted PPR"),
    ("perron_static", "Perron Static (gamma=0.70)"),
    ("perron", "Perron Adaptive Specificity"),
    ("oracle", "Oracle Upper Bound"),
]


def load_repository_engines(graphs_dir: Path) -> Dict[str, RepositoryRetrievalEngine]:
    """Load precomputed graphs and symbols for all 12 repositories."""
    engines: Dict[str, RepositoryRetrievalEngine] = {}
    for p in sorted(graphs_dir.glob("*_graph.npz")):
        slug = p.stem.replace("_graph", "")
        sym_p = graphs_dir / f"{slug}_symbols.json"
        if sym_p.exists():
            g = sp.load_npz(p)
            with open(sym_p, "r", encoding="utf-8") as f:
                syms = json.load(f)
            engines[slug] = RepositoryRetrievalEngine(g, syms)
    return engines


def run_sweep(
    graphs_dir: Path = Path("artifacts/swebench_graphs"),
    tasks_file: Path = Path("data/swebench_lite_cache.jsonl"),
    labels_file: Path = Path("data/swebench_lite_gold_labels.json"),
    split_file: Path = Path("data/swebench_lite_split.json"),
    output_file: Path = Path("data/retrieval_sweep_results.json"),
) -> Dict[str, Any]:
    """Execute the full 300-instance evaluation sweep across all 14 baselines."""
    t0 = time.time()
    print("Loading repository retrieval engines...")
    engines = load_repository_engines(graphs_dir)
    print(f"Loaded {len(engines)} repository engines in {time.time() - t0:.2f}s.")

    # Load tasks
    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    # Load gold annotations and split
    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels: Dict[str, Dict[str, Any]] = json.load(f)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        pilot_ids: Set[str] = set(split_data.get("dev_pilot_instances", []))
        val_ids: Set[str] = set(split_data.get("dev_val_instances", []))
        dev_ids: Set[str] = set(split_data.get("dev_instances", []))
        heldout_ids: Set[str] = set(split_data.get("heldout_instances", []))

    print(
        f"Loaded {len(tasks)} tasks: Pilot={len(pilot_ids)}, Val={len(val_ids)}, Held-out={len(heldout_ids)}."
    )

    # Storage for per-instance scores
    instance_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    instance_repos: Dict[str, str] = {}

    sweep_t0 = time.time()
    for idx, task in enumerate(tasks, 1):
        instance_id = task.get("instance_id", "")
        repo = task.get("repo", "")
        instance_repos[instance_id] = repo
        slug = get_repo_slug(repo)
        engine = engines.get(slug)
        if engine is None:
            print(f"Warning: Missing engine for {slug} (task {instance_id})")
            continue

        gold_info = gold_labels.get(instance_id)
        if not gold_info:
            print(f"Warning: Missing gold info for {instance_id}")
            continue

        gold_files = gold_info.get("gold_files", [])
        gold_sym_ids = gold_info.get("gold_symbol_ids", [])
        problem_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")

        # Compute query prior and PPR diffusion once per task
        p_0 = engine.compute_query_prior(problem_text)
        pi_query = engine.compute_personalized_pagerank(p_0)

        instance_results[instance_id] = {}
        line_counts = {
            s["id"]: max(5, (s.get("line_end") or 0) - (s.get("line_start") or 0))
            for s in engine.symbols
        }

        for method_key, _ in BASELINES:
            ranking = engine.retrieve_by_name(
                method_name=method_key,
                query_text=problem_text,
                pi_query=pi_query,
                p_0=p_0,
                gold_symbol_ids=gold_sym_ids,
            )

            # Map ranking to symbol IDs
            ret_sym_ids = [engine.symbols[i]["id"] for i in ranking]

            # Map ranking to unique file paths preserving first-rank order
            seen_files: Set[str] = set()
            ret_files: List[str] = []
            for i in ranking:
                fp = engine.symbols[i].get("file_path", "")
                if fp and fp not in seen_files:
                    seen_files.add(fp)
                    ret_files.append(fp)

            # Compute standardized metrics
            f_acc = compute_file_acc_at_k(ret_files, gold_files, k_values=(1, 3, 5))
            fn_acc = compute_function_acc_at_k(ret_sym_ids, gold_sym_ids, k_values=(1, 5, 10, 25))
            purity = compute_context_purity(
                ret_sym_ids[:25], ret_files[:5], gold_sym_ids, gold_files
            )
            hsi = compute_hsi(ret_sym_ids, engine.hub_set, top_k=10)
            rel_hsi = compute_relative_hsi(ret_sym_ids, engine.in_degrees, percentile=99.5, top_k=10)
            coverage = compute_coverage_at_budget(ret_sym_ids, gold_sym_ids, line_counts, token_budget=4096)

            instance_results[instance_id][method_key] = {
                "file_acc_1": f_acc[1],
                "file_acc_3": f_acc[3],
                "file_acc_5": f_acc[5],
                "fn_acc_1": fn_acc[1],
                "fn_acc_5": fn_acc[5],
                "fn_acc_10": fn_acc[10],
                "fn_acc_25": fn_acc[25],
                "context_purity": purity,
                "coverage_4096": coverage,
                "hsi": hsi,
                "relative_hsi": rel_hsi,
            }

        if idx % 50 == 0 or idx == len(tasks):
            print(f"Processed {idx}/{len(tasks)} tasks ({time.time() - sweep_t0:.2f}s)...")

    # Define analytical subsets
    subsets = {
        "full": list(instance_results.keys()),
        "dev_pilot": [iid for iid in instance_results if iid in pilot_ids],
        "dev_val": [iid for iid in instance_results if iid in val_ids],
        "dev_all": [iid for iid in instance_results if iid in dev_ids],
        "heldout": [iid for iid in instance_results if iid in heldout_ids],
        "heldout_s_func": [
            iid for iid in instance_results
            if iid in heldout_ids and gold_labels.get(iid, {}).get("has_function_target", False)
        ],
        "heldout_s_file": [
            iid for iid in instance_results
            if iid in heldout_ids and not gold_labels.get(iid, {}).get("has_function_target", False)
        ],
        "hub_gold": [
            iid for iid in instance_results
            if gold_labels.get(iid, {}).get("is_hub_gold", False)
        ],
        "non_hub_gold": [
            iid for iid in instance_results
            if not gold_labels.get(iid, {}).get("is_hub_gold", False)
        ],
    }

    metric_names = [
        "file_acc_1",
        "file_acc_3",
        "file_acc_5",
        "fn_acc_1",
        "fn_acc_5",
        "fn_acc_10",
        "fn_acc_25",
        "context_purity",
        "coverage_4096",
        "hsi",
        "relative_hsi",
    ]

    aggregated_results: Dict[str, Any] = {
        "metadata": {
            "num_tasks_total": len(tasks),
            "num_dev_pilot": len(pilot_ids),
            "num_dev_val": len(val_ids),
            "num_heldout": len(heldout_ids),
            "num_heldout_s_func": len(subsets["heldout_s_func"]),
            "num_heldout_s_file": len(subsets["heldout_s_file"]),
            "num_hub_gold": len(subsets["hub_gold"]),
            "num_non_hub_gold": len(subsets["non_hub_gold"]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "subsets": {},
        "statistical_tests": {},
    }

    for subset_name, iid_list in subsets.items():
        subset_summary: Dict[str, Dict[str, Any]] = {}
        for method_key, method_label in BASELINES:
            subset_summary[method_key] = {"label": method_label, "metrics": {}}
            for metric in metric_names:
                vals = [
                    instance_results[iid][method_key][metric]
                    for iid in iid_list
                    if iid in instance_results and method_key in instance_results[iid]
                ]
                mean_val, ci_lower, ci_upper = compute_bootstrap_ci(vals, n_bootstrap=10000)
                subset_summary[method_key]["metrics"][metric] = {
                    "mean": round(mean_val, 4),
                    "ci_lower": round(ci_lower, 4),
                    "ci_upper": round(ci_upper, 4),
                    "count": len(vals),
                }
        aggregated_results["subsets"][subset_name] = subset_summary

    # Execute Pre-Registered Primary Statistical Inference on Heldout S_func
    # Primary Endpoint: Function Acc@10 on heldout_s_func (Callable Target Tasks)
    primary_iids = subsets["heldout_s_func"]
    target_key = "perron_static"
    target_scores = [instance_results[iid][target_key]["fn_acc_10"] for iid in primary_iids]
    target_repos = [instance_repos[iid] for iid in primary_iids]

    raw_p_values: Dict[str, float] = {}
    pairwise_stats: Dict[str, Any] = {}

    for method_key, method_label in BASELINES:
        if method_key == target_key:
            continue
        base_scores = [instance_results[iid][method_key]["fn_acc_10"] for iid in primary_iids]

        # 1. Paired repo-stratified bootstrap (10,000 resamples)
        mean_d, ci_l, ci_u = compute_paired_repo_stratified_bootstrap_ci(
            target_scores, base_scores, target_repos, n_bootstrap=10000, ci=0.95
        )

        # 2. Exact McNemar test
        mcnemar_res = compute_exact_mcnemar_test(target_scores, base_scores)
        raw_p_values[method_key] = mcnemar_res["p_value"]

        pairwise_stats[method_key] = {
            "label": method_label,
            "mean_delta_fn_acc_10": round(mean_d, 4),
            "ci_95_lower": round(ci_l, 4),
            "ci_95_upper": round(ci_u, 4),
            "mcnemar": mcnemar_res,
        }

    # 3. Holm-Bonferroni step-down correction
    holm_p = compute_holm_bonferroni_correction(raw_p_values)
    for m_key in pairwise_stats:
        pairwise_stats[m_key]["mcnemar"]["p_adjusted_holm"] = holm_p[m_key]

    aggregated_results["statistical_tests"]["heldout_s_func_primary"] = pairwise_stats

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "aggregated": aggregated_results,
                "instances": instance_results,
            },
            f,
            indent=2,
        )
    print(f"\nComplete evaluation results saved to {output_file}")

    # Print Formatted Comparison Table for Primary Held-out S_func Split
    print("\n" + "=" * 110)
    print(f"PRE-REGISTERED PRIMARY EVALUATION: HELDOUT CALLABLE TASKS (S_func, N={len(primary_iids)})")
    print("=" * 110)
    header = f"{'Method':<26} | {'Fn Acc@5':<12} | {'Fn Acc@10':<12} | {'File Acc@5':<12} | {'Coverage@4k':<12} | {'McNemar p':<12} | {'Holm p-adj':<12}"
    print(header)
    print("-" * 110)

    s_func_res = aggregated_results["subsets"]["heldout_s_func"]
    for method_key, method_label in BASELINES:
        m = s_func_res[method_key]["metrics"]
        fn5_str = f"{m['fn_acc_5']['mean']*100:.1f}%"
        fn10_str = f"{m['fn_acc_10']['mean']*100:.1f}%"
        f5_str = f"{m['file_acc_5']['mean']*100:.1f}%"
        cov_str = f"{m['coverage_4096']['mean']*100:.1f}%"
        if method_key == "perron":
            p_raw = "-"
            p_holm = "-"
        elif method_key in pairwise_stats:
            p_raw = f"{pairwise_stats[method_key]['mcnemar']['p_value']:.4f}"
            p_holm = f"{pairwise_stats[method_key]['mcnemar']['p_adjusted_holm']:.4f}"
        else:
            p_raw = "-"
            p_holm = "-"
        print(f"{method_label:<26} | {fn5_str:<12} | {fn10_str:<12} | {f5_str:<12} | {cov_str:<12} | {p_raw:<12} | {p_holm:<12}")
    print("=" * 110)

    # Print All 150 Heldout Tasks Table
    print("\n" + "=" * 90)
    print(f"ALL HELDOUT TASKS (N=150) - MEAN METRICS")
    print("=" * 90)
    h_header = f"{'Method':<26} | {'File Acc@1':<14} | {'File Acc@5':<14} | {'Fn Acc@10':<14} | {'Rel HSI':<10}"
    print(h_header)
    print("-" * 90)
    heldout_res = aggregated_results["subsets"]["heldout"]
    for method_key, method_label in BASELINES:
        m = heldout_res[method_key]["metrics"]
        f1_s = f"{m['file_acc_1']['mean']*100:.1f}%"
        f5_s = f"{m['file_acc_5']['mean']*100:.1f}%"
        fn10_s = f"{m['fn_acc_10']['mean']*100:.1f}%"
        hsi_s = f"{m['relative_hsi']['mean']*100:.1f}%"
        print(f"{method_label:<26} | {f1_s:<14} | {f5_s:<14} | {fn10_s:<14} | {hsi_s:<10}")
    print("=" * 90 + "\n")

    return aggregated_results


if __name__ == "__main__":
    run_sweep()
