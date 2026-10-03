"""
Evaluate BM25 + Perron fusion methods on dev_val and execute pre-registered selection (T2.3).

Selection Rule (fixed in advance):
- Evaluate w in {0.0, 0.1, ..., 1.0} and RRF (k=60).
- Choose the w with highest Fn Acc@10 on dev_val; ties go to smaller w.
- Output: results/dev_val_fusion.json with full G5 metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.eval_harness import (
    compute_bootstrap_ci,
    compute_context_purity,
    compute_exact_mcnemar_test,
    compute_relative_hsi,
)
from benchmarks.gold_patch_parser import get_repo_slug
from perron.fusion import (
    compute_instance_fusion_rankings,
    min_max_scale,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from perron.metrics import (
    coverage_at_4k,
    file_acc_at_k,
    function_acc_at_k,
    hsi_at_10,
)
from perron.results import save_results

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def run_dev_val_fusion_selection(
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Tuple[Path, float]:
    """
    Run fusion sweep across dev_val (N=100), record metrics, and select best w.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "dev_val_fusion.json"

    graphs_dir = PROJECT_ROOT / "artifacts" / "swebench_graphs"
    tasks_file = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
    labels_file = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"

    print("Executing fusion evaluation on dev_val (N=100)...")
    engines = load_repository_engines(graphs_dir)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    target_ids = set(split_data.get("dev_val_instances", []))
    if len(target_ids) != 100:
        raise ValueError(f"Expected 100 dev_val instances, got {len(target_ids)}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = [t for t in all_tasks if t.get("instance_id") in target_ids]

    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    variant_keys = ["rrf_60"] + [f"weighted_w_{w:.1f}" for w in WEIGHTS]
    instance_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    t0 = time.time()
    for idx, task in enumerate(tasks, 1):
        instance_id = task.get("instance_id", "")
        repo = task.get("repo", "")
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

        # Compute raw BM25 scores and raw Perron Static scores
        bm25_scores = engine.bm25_index.query(problem_text)
        p_0 = engine.compute_query_prior(problem_text)
        pi_query = engine.compute_personalized_pagerank(p_0)
        perron_scores = pi_query / np.power(engine.pi_global + 1e-8, 0.70)

        # Generate fusion rankings
        fusion_rankings = compute_instance_fusion_rankings(
            bm25_scores, perron_scores, weights=WEIGHTS, rrf_k=60
        )

        instance_results[instance_id] = {}
        line_costs = {
            s["id"]: max(40, int(max(5, (s.get("line_end") or 0) - (s.get("line_start") or 0)) * 8.0))
            for s in engine.symbols
        }

        for v_key in variant_keys:
            ranking = fusion_rankings[v_key]
            ret_sym_ids = [engine.symbols[i]["id"] for i in ranking]

            seen_files: Set[str] = set()
            ret_files: List[str] = []
            for i in ranking:
                fp = engine.symbols[i].get("file_path", "")
                if fp and fp not in seen_files:
                    seen_files.add(fp)
                    ret_files.append(fp)

            f_acc_1 = file_acc_at_k(ret_files, gold_files, k=1)
            f_acc_5 = file_acc_at_k(ret_files, gold_files, k=5)
            fn_acc_1 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=1)
            fn_acc_5 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=5)
            fn_acc_10 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=10)
            cov = coverage_at_4k(ret_sym_ids, gold_sym_ids, symbol_token_costs=line_costs, budget=4096)
            hsi = hsi_at_10(ret_sym_ids, engine.hub_set)

            instance_results[instance_id][v_key] = {
                "file_acc_1": f_acc_1,
                "file_acc_5": f_acc_5,
                "fn_acc_1": fn_acc_1,
                "fn_acc_5": fn_acc_5,
                "fn_acc_10": fn_acc_10,
                "coverage_4096": cov,
                "hsi": hsi,
            }

        if idx % 25 == 0 or idx == len(tasks):
            print(f"Processed {idx}/{len(tasks)} dev_val tasks ({time.time() - t0:.2f}s)...")

    # Aggregate metrics
    summary: Dict[str, Dict[str, Any]] = {}
    metric_keys = ["file_acc_1", "file_acc_5", "fn_acc_1", "fn_acc_5", "fn_acc_10", "coverage_4096", "hsi"]

    for v_key in variant_keys:
        summary[v_key] = {"metrics": {}}
        for m_key in metric_keys:
            vals = [instance_results[iid][v_key][m_key] for iid in instance_results]
            mean_v, ci_l, ci_u = compute_bootstrap_ci(vals, n_bootstrap=1000, seed=seed)
            summary[v_key]["metrics"][m_key] = {
                "mean": round(mean_v, 4),
                "ci_lower": round(ci_l, 4),
                "ci_upper": round(ci_u, 4),
                "count": len(vals),
            }

    # Pre-registered selection rule:
    # Choose w with highest Fn Acc@10 on dev_val; ties go to smaller w.
    best_w = WEIGHTS[0]
    best_score = -1.0
    for w in WEIGHTS:
        score = summary[f"weighted_w_{w:.1f}"]["metrics"]["fn_acc_10"]["mean"]
        # If strictly better, or equal and smaller w (since WEIGHTS is ascending, strictly better suffices)
        if score > best_score:
            best_score = score
            best_w = w

    rrf_score = summary["rrf_60"]["metrics"]["fn_acc_10"]["mean"]

    print("\n" + "=" * 80)
    print("DEV_VAL FUSION SWEEP SUMMARY (N=100)")
    print("=" * 80)
    print(f"{'Variant':<20} | {'Fn Acc@5':<12} | {'Fn Acc@10':<12} | {'File Acc@5':<12} | {'HSI@10':<10}")
    print("-" * 80)
    for v_key in variant_keys:
        m = summary[v_key]["metrics"]
        print(f"{v_key:<20} | {m['fn_acc_5']['mean']*100:.1f}%        | {m['fn_acc_10']['mean']*100:.1f}%        | {m['file_acc_5']['mean']*100:.1f}%        | {m['hsi']['mean']*100:.1f}%")
    print("=" * 80)
    print(f"Selected Weight: w = {best_w:.1f} (Fn Acc@10 = {best_score*100:.1f}%)")
    print(f"RRF (k=60) Score: Fn Acc@10 = {rrf_score*100:.1f}%")

    results_payload = {
        "suite": "fusion",
        "split": "dev_val",
        "seed": seed,
        "num_instances": len(instance_results),
        "selected_w": best_w,
        "selected_score": best_score,
        "summary": summary,
        "instances": instance_results,
    }

    saved_path = save_results(
        path=output_path,
        results_data=results_payload,
        split_name="dev_val",
        seed=seed,
        suite="fusion",
    )
    print(f"\nSaved dev_val fusion results to: {saved_path}")
    return saved_path, best_w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perron Fusion Dev-Val Evaluation")
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    run_dev_val_fusion_selection(seed=args.seed)
