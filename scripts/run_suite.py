"""
Evaluation suite runner for development partitions (dev_val / dev_pilot).

Strictly enforces G3:
- Tuning, debugging, and baseline evaluation on dev_val (N=100) or dev_pilot (N=50).
- Held-out evaluations are prohibited here and must use scripts/run_heldout.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scipy.sparse as sp

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.eval_harness import (
    compute_bootstrap_ci,
    compute_context_purity,
    compute_exact_mcnemar_test,
    compute_holm_bonferroni_correction,
    compute_paired_repo_stratified_bootstrap_ci,
    compute_relative_hsi,
)
from benchmarks.gold_patch_parser import get_repo_slug
from perron.metrics import (
    coverage_at_4k,
    file_acc_at_k,
    function_acc_at_k,
    hsi_at_10,
)
from perron.results import save_results

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
    """Load precomputed graphs and symbols for all repositories."""
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


def run_dev_suite(
    suite: str = "table2",
    split: str = "dev_val",
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    if split == "heldout":
        raise PermissionError(
            "Held-out evaluation is strictly prohibited in run_suite.py per rule G3! "
            "Use scripts/run_heldout.py instead."
        )

    graphs_dir = PROJECT_ROOT / "artifacts" / "swebench_graphs"
    tasks_file = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
    labels_file = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"

    if output_path is None:
        output_path = PROJECT_ROOT / "results" / f"{split}_{suite}_postaudit.json"

    print(f"Executing suite '{suite}' on partition '{split}' (seed={seed})...")
    t0 = time.time()
    engines = load_repository_engines(graphs_dir)
    print(f"Loaded {len(engines)} repository engines in {time.time() - t0:.2f}s.")

    # Load split partition IDs
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split == "dev_val":
        target_ids: Set[str] = set(split_data.get("dev_val_instances", []))
    elif split == "dev_pilot":
        target_ids = set(split_data.get("dev_pilot_instances", []))
    elif split == "dev_all":
        target_ids = set(split_data.get("dev_instances", []))
    else:
        raise ValueError(f"Unknown dev partition: {split}")

    # Load tasks and gold annotations
    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    tasks = [t for t in all_tasks if t.get("instance_id") in target_ids]
    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    print(f"Target tasks: {len(tasks)} instances in {split}.")

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
            continue

        gold_info = gold_labels.get(instance_id)
        if not gold_info:
            continue

        gold_files = gold_info.get("gold_files", [])
        gold_sym_ids = gold_info.get("gold_symbol_ids", [])
        problem_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")

        p_0 = engine.compute_query_prior(problem_text)
        pi_query = engine.compute_personalized_pagerank(p_0)

        instance_results[instance_id] = {}
        line_costs = {
            s["id"]: max(40, int(max(5, (s.get("line_end") or 0) - (s.get("line_start") or 0)) * 8.0))
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

            ret_sym_ids = [engine.symbols[i]["id"] for i in ranking]
            seen_files: Set[str] = set()
            ret_files: List[str] = []
            for i in ranking:
                fp = engine.symbols[i].get("file_path", "")
                if fp and fp not in seen_files:
                    seen_files.add(fp)
                    ret_files.append(fp)

            f_acc_1 = file_acc_at_k(ret_files, gold_files, k=1)
            f_acc_3 = file_acc_at_k(ret_files, gold_files, k=3)
            f_acc_5 = file_acc_at_k(ret_files, gold_files, k=5)

            fn_acc_1 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=1)
            fn_acc_5 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=5)
            fn_acc_10 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=10)
            fn_acc_25 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=25)

            cov = coverage_at_4k(ret_sym_ids, gold_sym_ids, symbol_token_costs=line_costs, budget=4096)
            hsi = hsi_at_10(ret_sym_ids, engine.hub_set)
            rel_hsi = compute_relative_hsi(ret_sym_ids, engine.in_degrees, percentile=99.5, top_k=10)
            purity = compute_context_purity(ret_sym_ids[:25], ret_files[:5], gold_sym_ids, gold_files)

            instance_results[instance_id][method_key] = {
                "file_acc_1": f_acc_1,
                "file_acc_3": f_acc_3,
                "file_acc_5": f_acc_5,
                "fn_acc_1": fn_acc_1,
                "fn_acc_5": fn_acc_5,
                "fn_acc_10": fn_acc_10,
                "fn_acc_25": fn_acc_25,
                "context_purity": purity,
                "coverage_4096": cov,
                "hsi": hsi,
                "relative_hsi": rel_hsi,
            }

        if idx % 25 == 0 or idx == len(tasks):
            print(f"Processed {idx}/{len(tasks)} tasks ({time.time() - sweep_t0:.2f}s)...")

    # Aggregate metrics
    summary: Dict[str, Dict[str, Any]] = {}
    metric_keys = [
        "file_acc_1", "file_acc_3", "file_acc_5",
        "fn_acc_1", "fn_acc_5", "fn_acc_10", "fn_acc_25",
        "context_purity", "coverage_4096", "hsi", "relative_hsi",
    ]

    for method_key, method_label in BASELINES:
        summary[method_key] = {"label": method_label, "metrics": {}}
        for m_key in metric_keys:
            vals = [
                instance_results[iid][method_key][m_key]
                for iid in instance_results
                if method_key in instance_results[iid]
            ]
            mean_v, ci_l, ci_u = compute_bootstrap_ci(vals, n_bootstrap=1000, seed=seed)
            summary[method_key]["metrics"][m_key] = {
                "mean": round(mean_v, 4),
                "ci_lower": round(ci_l, 4),
                "ci_upper": round(ci_u, 4),
                "count": len(vals),
            }

    # Paired McNemar tests vs perron_static
    target_key = "perron_static"
    target_scores = [instance_results[iid][target_key]["fn_acc_10"] for iid in instance_results]
    target_repos = [instance_repos[iid] for iid in instance_results]
    pairwise_stats: Dict[str, Any] = {}
    raw_p_values: Dict[str, float] = {}

    for method_key, method_label in BASELINES:
        if method_key == target_key:
            continue
        base_scores = [instance_results[iid][method_key]["fn_acc_10"] for iid in instance_results]
        mean_d, ci_l, ci_u = compute_paired_repo_stratified_bootstrap_ci(
            target_scores, base_scores, target_repos, n_bootstrap=1000, ci=0.95
        )
        mcnemar = compute_exact_mcnemar_test(target_scores, base_scores)
        raw_p_values[method_key] = mcnemar["p_value"]
        pairwise_stats[method_key] = {
            "label": method_label,
            "mean_delta_fn_acc_10": round(mean_d, 4),
            "ci_95_lower": round(ci_l, 4),
            "ci_95_upper": round(ci_u, 4),
            "mcnemar": mcnemar,
        }

    holm_p = compute_holm_bonferroni_correction(raw_p_values)
    for m_key in pairwise_stats:
        pairwise_stats[m_key]["mcnemar"]["p_adjusted_holm"] = holm_p[m_key]

    results_payload = {
        "suite": suite,
        "split": split,
        "seed": seed,
        "num_instances": len(instance_results),
        "summary": summary,
        "pairwise_vs_perron_static": pairwise_stats,
        "instances": instance_results,
    }

    # Save using standardized G5 results writer
    saved_path = save_results(
        path=output_path,
        results_data=results_payload,
        split_name=split,
        seed=seed,
        suite=suite,
    )
    print(f"\nResults successfully written to: {saved_path}")

    # Print Formatted Table
    print("\n" + "=" * 105)
    print(f"POST-AUDIT EVALUATION: {suite.upper()} ON {split.upper()} (N={len(instance_results)}, Seed={seed})")
    print("=" * 105)
    print(f"{'Method':<28} | {'Fn Acc@5':<11} | {'Fn Acc@10':<11} | {'File Acc@5':<11} | {'Cov@4k':<10} | {'HSI@10':<9} | {'McNemar p':<10}")
    print("-" * 105)
    for method_key, method_label in BASELINES:
        m = summary[method_key]["metrics"]
        fn5 = f"{m['fn_acc_5']['mean']*100:.1f}%"
        fn10 = f"{m['fn_acc_10']['mean']*100:.1f}%"
        f5 = f"{m['file_acc_5']['mean']*100:.1f}%"
        cov = f"{m['coverage_4096']['mean']*100:.1f}%"
        hsi = f"{m['hsi']['mean']*100:.1f}%"
        p_str = f"{pairwise_stats[method_key]['mcnemar']['p_value']:.4f}" if method_key in pairwise_stats else "-"
        print(f"{method_label:<28} | {fn5:<11} | {fn10:<11} | {f5:<11} | {cov:<10} | {hsi:<9} | {p_str:<10}")
    print("=" * 105 + "\n")

    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation suite on dev partitions.")
    parser.add_argument("--suite", type=str, default="table2", help="Evaluation suite name (default: table2).")
    parser.add_argument("--split", type=str, default="dev_val", help="Dataset partition (default: dev_val).")
    parser.add_argument("--seed", type=int, default=20261003, help="Random seed (default: 20261003).")
    parser.add_argument("--output", type=str, default=None, help="Explicit output path.")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    run_dev_suite(suite=args.suite, split=args.split, seed=args.seed, output_path=out_p)


if __name__ == "__main__":
    main()
