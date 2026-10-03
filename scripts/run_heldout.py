"""
Authoritative Held-out Evaluation Suite Runner (Rule G3).

Strictly enforces:
- Held-out evaluations execute ONLY via this runner.
- Every execution logs an immutable entry to RUNLOG.md.
- Adheres to G4 (seeds), G5 (metadata serialization), and G8 (contamination honesty).

Supported Suites:
- table2: Official 14-baseline evaluation matrix on heldout (N=150).
- fusion: Official BM25 + Perron fusion evaluation on heldout (N=150) + complementarity analysis.
- strata: Per-stratum accuracy analysis (re-uses cached rankings from table2/fusion, zero recomputation).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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
from perron.analysis.strata import STRATUM_A, STRATUM_B, STRATUM_C
from perron.fusion import (
    compute_instance_fusion_rankings,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from perron.metrics import (
    coverage_at_4k,
    file_acc_at_k,
    function_acc_at_k,
    hsi_at_10,
)
from perron.results import get_git_sha, save_results

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

FUSION_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def append_to_runlog(
    command_str: str,
    split: str,
    output_file: Path,
    note: str = "post-audit heldout evaluation",
) -> None:
    """Append a single verified record to RUNLOG.md per Rule G3."""
    runlog_file = PROJECT_ROOT / "docs" / "RUNLOG.md"
    if not runlog_file.exists() and (PROJECT_ROOT / "RUNLOG.md").exists():
        runlog_file = PROJECT_ROOT / "RUNLOG.md"
    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_sha = get_git_sha()
    rel_path = output_file.relative_to(PROJECT_ROOT) if output_file.is_relative_to(PROJECT_ROOT) else output_file

    line = f"| {now_utc} | {command_str} | {split} | {git_sha} | {rel_path.as_posix()} | {note} |\n"

    # Ensure RUNLOG.md has header if empty
    if not runlog_file.exists() or runlog_file.stat().st_size == 0:
        runlog_content = (
            "# Perron Evaluation & Run Log\n\n"
            "| UTC | command | split | git sha | output file | note |\n"
            "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        )
        runlog_file.write_text(runlog_content, encoding="utf-8")

    with open(runlog_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(f"Logged run to {runlog_file.relative_to(PROJECT_ROOT)}")


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


def run_table2_suite(
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    """Execute official Table 2 evaluation on the frozen heldout partition (N=150)."""
    suite = "table2"
    command_str = f"python scripts/run_heldout.py --suite {suite} --seed {seed}"
    graphs_dir = PROJECT_ROOT / "artifacts" / "swebench_graphs"
    tasks_file = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
    labels_file = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"

    if output_path is None:
        output_path = PROJECT_ROOT / "results" / f"heldout_{suite}_v2_postaudit.json"

    print("=" * 80)
    print(f"OFFICIAL HELDOUT EVALUATION RUN: suite='{suite}', seed={seed}")
    print(f"Destination: {output_path}")
    print("=" * 80)

    t0 = time.time()
    engines = load_repository_engines(graphs_dir)
    print(f"Loaded {len(engines)} repository engines in {time.time() - t0:.2f}s.")

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    heldout_ids: Set[str] = set(split_data.get("heldout_instances", []))
    if len(heldout_ids) != 150:
        raise ValueError(f"Expected exactly 150 heldout instances, found {len(heldout_ids)}!")

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]

    tasks = [t for t in all_tasks if t.get("instance_id") in heldout_ids]
    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    print(f"Matched {len(tasks)} held-out instances from cache.")

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
            print(f"Evaluated {idx}/{len(tasks)} held-out tasks ({time.time() - sweep_t0:.2f}s)...")

    metric_keys = [
        "file_acc_1", "file_acc_3", "file_acc_5",
        "fn_acc_1", "fn_acc_5", "fn_acc_10", "fn_acc_25",
        "context_purity", "coverage_4096", "hsi", "relative_hsi",
    ]

    def aggregate_subset(iids: List[str]) -> Dict[str, Dict[str, Any]]:
        subset_summary: Dict[str, Dict[str, Any]] = {}
        for method_key, method_label in BASELINES:
            subset_summary[method_key] = {"label": method_label, "metrics": {}}
            for m_key in metric_keys:
                vals = [
                    instance_results[iid][method_key][m_key]
                    for iid in iids
                    if iid in instance_results and method_key in instance_results[iid]
                ]
                mean_v, ci_l, ci_u = compute_bootstrap_ci(vals, n_bootstrap=1000, seed=seed)
                subset_summary[method_key]["metrics"][m_key] = {
                    "mean": round(mean_v, 4),
                    "ci_lower": round(ci_l, 4),
                    "ci_upper": round(ci_u, 4),
                    "count": len(vals),
                }
        return subset_summary

    all_iids = list(instance_results.keys())
    summary_all = aggregate_subset(all_iids)

    # Paired statistical inference vs perron_static
    target_key = "perron_static"
    target_scores = [instance_results[iid][target_key]["fn_acc_10"] for iid in all_iids]
    target_repos = [instance_repos[iid] for iid in all_iids]
    pairwise_stats: Dict[str, Any] = {}
    raw_p_values: Dict[str, float] = {}

    for method_key, method_label in BASELINES:
        if method_key == target_key:
            continue
        base_scores = [instance_results[iid][method_key]["fn_acc_10"] for iid in all_iids]
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
        "split": "heldout",
        "seed": seed,
        "num_instances": len(instance_results),
        "summary": summary_all,
        "pairwise_vs_perron_static": pairwise_stats,
        "instances": instance_results,
    }

    saved_path = save_results(
        path=output_path,
        results_data=results_payload,
        split_name="heldout",
        seed=seed,
        suite=suite,
    )
    append_to_runlog(
        command_str=command_str,
        split="heldout",
        output_file=saved_path,
        note="post-audit heldout evaluation (Table 2)",
    )
    return saved_path


def run_fusion_suite(
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    """
    Execute official BM25 + Perron Fusion evaluation on heldout partition (N=150) (T2.3).
    Also generates results/heldout_complementarity.json (T2.4).
    """
    suite = "fusion"
    command_str = f"python scripts/run_heldout.py --suite {suite} --seed {seed}"
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "heldout_fusion.json"

    print("=" * 80)
    print(f"OFFICIAL HELDOUT FUSION EVALUATION: suite='{suite}', seed={seed}")
    print(f"Destination: {output_path}")
    print("=" * 80)

    # Ingest pre-registered winner from dev_val
    dev_val_fusion_file = PROJECT_ROOT / "results" / "dev_val_fusion.json"
    selected_w = 0.0
    if dev_val_fusion_file.exists():
        with open(dev_val_fusion_file, "r", encoding="utf-8") as f:
            dv_data = json.load(f)
        selected_w = float(dv_data.get("selected_w", 0.0))
        print(f"Loaded pre-registered winning dev_val weight: w = {selected_w:.1f}")

    graphs_dir = PROJECT_ROOT / "artifacts" / "swebench_graphs"
    tasks_file = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
    labels_file = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"

    engines = load_repository_engines(graphs_dir)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    heldout_ids = set(split_data.get("heldout_instances", []))

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = [t for t in all_tasks if t.get("instance_id") in heldout_ids]

    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    variant_keys = ["rrf_60"] + [f"weighted_w_{w:.1f}" for w in FUSION_WEIGHTS]
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

        bm25_scores = engine.bm25_index.query(problem_text)
        p_0 = engine.compute_query_prior(problem_text)
        pi_query = engine.compute_personalized_pagerank(p_0)
        perron_scores = pi_query / np.power(engine.pi_global + 1e-8, 0.70)

        fusion_rankings = compute_instance_fusion_rankings(
            bm25_scores, perron_scores, weights=FUSION_WEIGHTS, rrf_k=60
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
            print(f"Evaluated {idx}/{len(tasks)} held-out tasks ({time.time() - t0:.2f}s)...")

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

    results_payload = {
        "suite": "fusion",
        "split": "heldout",
        "seed": seed,
        "num_instances": len(instance_results),
        "selected_w": selected_w,
        "summary": summary,
        "instances": instance_results,
    }

    saved_path = save_results(
        path=output_path,
        results_data=results_payload,
        split_name="heldout",
        seed=seed,
        suite="fusion",
    )
    append_to_runlog(
        command_str=command_str,
        split="heldout",
        output_file=saved_path,
        note="post-audit heldout fusion evaluation (T2.3)",
    )

    # -------------------------------------------------------------
    # T2.4 Complementarity table computation
    # -------------------------------------------------------------
    print("\nComputing T2.4 Complementarity Table...")
    # Load Table 2 postaudit for exact baseline hits
    t2_postaudit_file = PROJECT_ROOT / "results" / "heldout_table2_v2_postaudit.json"
    bm25_hits: Dict[str, int] = {}
    perron_hits: Dict[str, int] = {}

    if t2_postaudit_file.exists():
        with open(t2_postaudit_file, "r", encoding="utf-8") as f:
            t2_data = json.load(f)
        for iid, r in t2_data.get("instances", {}).items():
            bm25_hits[iid] = int(r.get("bm25", {}).get("fn_acc_10", 0.0) > 0.0)
            perron_hits[iid] = int(r.get("perron_static", {}).get("fn_acc_10", 0.0) > 0.0)

    # Selected fusion variant hits
    selected_key = f"weighted_w_{selected_w:.1f}"
    fusion_hits: Dict[str, int] = {
        iid: int(instance_results[iid][selected_key]["fn_acc_10"] > 0.0)
        for iid in instance_results
    }

    # Counts where (BM25 hit, Perron hit) = (1,1), (1,0), (0,1), (0,0)
    c_11 = sum(1 for iid in instance_results if bm25_hits.get(iid, 0) == 1 and perron_hits.get(iid, 0) == 1)
    c_10 = sum(1 for iid in instance_results if bm25_hits.get(iid, 0) == 1 and perron_hits.get(iid, 0) == 0)
    c_01 = sum(1 for iid in instance_results if bm25_hits.get(iid, 0) == 0 and perron_hits.get(iid, 0) == 1)
    c_00 = sum(1 for iid in instance_results if bm25_hits.get(iid, 0) == 0 and perron_hits.get(iid, 0) == 0)

    # Exact McNemar test for fusion vs BM25
    bm25_list = [bm25_hits.get(iid, 0) for iid in instance_results]
    fusion_list = [fusion_hits.get(iid, 0) for iid in instance_results]
    perron_list = [perron_hits.get(iid, 0) for iid in instance_results]

    mcnemar_fusion_vs_bm25 = compute_exact_mcnemar_test(fusion_list, bm25_list)
    mcnemar_perron_vs_bm25 = compute_exact_mcnemar_test(perron_list, bm25_list)

    comp_payload = {
        "suite": "complementarity",
        "split": "heldout",
        "seed": seed,
        "num_instances": len(instance_results),
        "selected_fusion_variant": selected_key,
        "contingency_table_bm25_perron": {
            "both_hit_1_1": c_11,
            "bm25_only_1_0": c_10,
            "perron_only_0_1": c_01,
            "neither_hit_0_0": c_00,
        },
        "mcnemar_fusion_vs_bm25": mcnemar_fusion_vs_bm25,
        "mcnemar_perron_vs_bm25": mcnemar_perron_vs_bm25,
    }

    comp_path = PROJECT_ROOT / "results" / "heldout_complementarity.json"
    save_results(
        path=comp_path,
        results_data=comp_payload,
        split_name="heldout",
        seed=seed,
        suite="complementarity",
    )
    print(f"Complementarity results saved to: {comp_path}")
    print(f"Contingency: (1,1)={c_11}, (1,0)={c_10}, (0,1)={c_01}, (0,0)={c_00}")
    print(f"McNemar fusion vs BM25: p = {mcnemar_fusion_vs_bm25['p_value']:.4f}")

    return saved_path


def run_strata_suite(
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    """
    Execute per-stratum accuracy analysis on heldout (T2.2).
    Re-uses cached rankings from T1.5 (and fusion if available); zero recomputation.
    """
    suite = "strata"
    command_str = f"python scripts/run_heldout.py --suite {suite} --seed {seed}"
    if output_path is None:
        output_path = PROJECT_ROOT / "results" / "heldout_strata.json"

    print("=" * 80)
    print(f"OFFICIAL HELDOUT STRATA ANALYSIS: suite='{suite}', seed={seed}")
    print(f"Re-using cached per-instance rankings (Zero Recomputation).")
    print(f"Destination: {output_path}")
    print("=" * 80)

    # 1. Load cached rankings from Table 2 postaudit
    t2_file = PROJECT_ROOT / "results" / "heldout_table2_v2_postaudit.json"
    if not t2_file.exists():
        raise FileNotFoundError(f"Missing cached Table 2 results at {t2_file}!")

    with open(t2_file, "r", encoding="utf-8") as f:
        t2_data = json.load(f)
    t2_instances: Dict[str, Dict[str, Any]] = t2_data.get("instances", {})

    # 2. Check if fusion results are available
    fusion_file = PROJECT_ROOT / "results" / "heldout_fusion.json"
    fusion_instances: Dict[str, Dict[str, Any]] = {}
    fusion_key = "weighted_w_0.0"
    if fusion_file.exists():
        with open(fusion_file, "r", encoding="utf-8") as f:
            f_data = json.load(f)
        sel_w = f_data.get("selected_w", 0.0)
        fusion_key = f"weighted_w_{sel_w:.1f}"
        fusion_instances = f_data.get("instances", {})

    # 3. Load strata mapping
    strata_file = PROJECT_ROOT / "data" / "strata.json"
    if not strata_file.exists():
        raise FileNotFoundError(f"Missing strata mapping at {strata_file}!")

    with open(strata_file, "r", encoding="utf-8") as f:
        strata_map: Dict[str, str] = json.load(f)

    # Filter to heldout instances
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    heldout_ids = split_data.get("heldout_instances", [])

    methods_to_evaluate = [
        ("bm25", "Okapi BM25"),
        ("dense", "Dense Semantic"),
        ("perron_static", "Perron Static"),
        ("deg_ppr", "Degree-Norm PPR"),
    ]
    if fusion_instances:
        methods_to_evaluate.append(("fusion", f"Perron+BM25 Fusion ({fusion_key})"))

    strata_results: Dict[str, Any] = {}

    for s_name in [STRATUM_A, STRATUM_B, STRATUM_C]:
        s_iids = [iid for iid in heldout_ids if strata_map.get(iid) == s_name]
        n_count = len(s_iids)
        is_descriptive = n_count < 15

        stratum_payload: Dict[str, Any] = {
            "n": n_count,
            "descriptive_only": is_descriptive,
            "methods": {},
        }

        print(f"\n--- Stratum {s_name} (N={n_count}, descriptive_only={is_descriptive}) ---")

        for m_key, m_label in methods_to_evaluate:
            if m_key == "fusion":
                vals = [
                    fusion_instances[iid][fusion_key]["fn_acc_10"]
                    for iid in s_iids
                    if iid in fusion_instances and fusion_key in fusion_instances[iid]
                ]
            else:
                vals = [
                    t2_instances[iid][m_key]["fn_acc_10"]
                    for iid in s_iids
                    if iid in t2_instances and m_key in t2_instances[iid]
                ]

            if not vals:
                mean_v, ci_l, ci_u = 0.0, 0.0, 0.0
            else:
                # B=10,000 bootstrap CI with instance-level resampling
                mean_v, ci_l, ci_u = compute_bootstrap_ci(
                    vals, n_bootstrap=10000, seed=seed, ci=0.95
                )

            stratum_payload["methods"][m_key] = {
                "label": m_label,
                "fn_acc_10": {
                    "mean": round(mean_v, 4),
                    "ci_95_lower": round(ci_l, 4),
                    "ci_95_upper": round(ci_u, 4),
                    "count": len(vals),
                },
            }
            print(f"  {m_label:<28}: {mean_v*100:5.1f}% [{ci_l*100:5.1f}%, {ci_u*100:5.1f}%]")

        strata_results[f"stratum_{s_name}"] = stratum_payload

    final_payload = {
        "suite": "strata",
        "split": "heldout",
        "seed": seed,
        "bootstrap_iterations": 10000,
        "strata": strata_results,
    }

    saved_path = save_results(
        path=output_path,
        results_data=final_payload,
        split_name="heldout",
        seed=seed,
        suite="strata",
    )
    append_to_runlog(
        command_str=command_str,
        split="heldout",
        output_file=saved_path,
        note="post-audit heldout strata evaluation (T2.2)",
    )
    print(f"\nHeld-out strata results saved to: {saved_path}")
    return saved_path


def run_heldout_suite(
    suite: str = "table2",
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    """Dispatcher for held-out evaluation suites."""
    if suite == "table2":
        return run_table2_suite(seed=seed, output_path=output_path)
    elif suite == "fusion":
        return run_fusion_suite(seed=seed, output_path=output_path)
    elif suite == "strata":
        return run_strata_suite(seed=seed, output_path=output_path)
    else:
        raise ValueError(f"Unknown held-out suite: '{suite}'. Must be 'table2', 'fusion', or 'strata'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perron Official Held-out Evaluation Runner")
    parser.add_argument("--suite", type=str, default="table2", choices=["table2", "fusion", "strata"], help="Suite identifier")
    parser.add_argument("--seed", type=int, default=20261003, help="Random seed (default: 20261003)")
    parser.add_argument("--output", type=str, default=None, help="Custom output path")
    args = parser.parse_args()

    out_p = Path(args.output).resolve() if args.output else None
    run_heldout_suite(suite=args.suite, seed=args.seed, output_path=out_p)


if __name__ == "__main__":
    main()
