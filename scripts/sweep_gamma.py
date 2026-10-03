"""
Sweep gamma parameter on dev_val partition (T2.5).

Evaluates Perron Static across gammas in [0.0, 1.0]:
score(v) = pi_query(v) / (pi_global(v) + 1e-8)^gamma

Computes:
- Fn Acc@10 with 95% bootstrap CI
- HSI@10 with 95% bootstrap CI
- File Acc@5, Coverage@4k
Saves results to results/dev_val_gamma_sweep.json with G5 metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set
import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.eval_harness import compute_bootstrap_ci
from benchmarks.gold_patch_parser import get_repo_slug
from perron.metrics import (
    coverage_at_4k,
    file_acc_at_k,
    function_acc_at_k,
    hsi_at_10,
)
from perron.results import save_results


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


def parse_gamma_list(gammas_str: str) -> List[float]:
    """Parse comma-separated gammas, expanding ellipsis if provided."""
    gammas_str = gammas_str.strip()
    if "..." in gammas_str:
        return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return [float(x.strip()) for x in gammas_str.split(",") if x.strip()]


def run_gamma_sweep(
    split: str = "dev_val",
    gammas: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    seed: int = 20261003,
    output_path: Path | None = None,
) -> Path:
    """Execute gamma sweep on the specified split partition."""
    if split == "heldout":
        raise PermissionError(
            "Held-out gamma sweep is prohibited by Rule G3. Use dev_val for tuning."
        )

    if output_path is None:
        output_path = PROJECT_ROOT / "results" / f"{split}_gamma_sweep.json"

    graphs_dir = PROJECT_ROOT / "artifacts" / "swebench_graphs"
    tasks_file = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
    labels_file = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
    split_file = PROJECT_ROOT / "data" / "swebench_lite_split.json"

    print(f"Executing gamma sweep on {split} (seed={seed})...")
    print(f"Gammas: {[round(g, 2) for g in gammas]}")

    engines = load_repository_engines(graphs_dir)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    if split == "dev_val":
        target_ids = set(split_data.get("dev_val_instances", []))
    elif split == "dev_pilot":
        target_ids = set(split_data.get("dev_pilot_instances", []))
    else:
        raise ValueError(f"Unsupported split for tuning: {split}")

    with open(tasks_file, "r", encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = [t for t in all_tasks if t.get("instance_id") in target_ids]

    with open(labels_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    print(f"Loaded {len(tasks)} tasks on {split}.")

    # Precompute personalized PageRank pi_query per instance once
    instance_data: List[Dict[str, Any]] = []
    t0 = time.time()
    for task in tasks:
        iid = task.get("instance_id", "")
        repo = task.get("repo", "")
        slug = get_repo_slug(repo)
        engine = engines.get(slug)
        if engine is None:
            continue
        gold_info = gold_labels.get(iid)
        if not gold_info:
            continue

        problem_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")
        p_0 = engine.compute_query_prior(problem_text)
        pi_query = engine.compute_personalized_pagerank(p_0)

        line_costs = {
            s["id"]: max(40, int(max(5, (s.get("line_end") or 0) - (s.get("line_start") or 0)) * 8.0))
            for s in engine.symbols
        }

        instance_data.append({
            "iid": iid,
            "engine": engine,
            "pi_query": pi_query,
            "gold_files": gold_info.get("gold_files", []),
            "gold_sym_ids": gold_info.get("gold_symbol_ids", []),
            "line_costs": line_costs,
        })

    print(f"Precomputed PPR in {time.time() - t0:.2f}s for {len(instance_data)} instances.")

    # Sweep each gamma
    sweep_results: Dict[str, Any] = {}
    for g_val in gammas:
        g_key = f"gamma_{g_val:.2f}"
        fn_acc_10_vals: List[float] = []
        hsi_10_vals: List[float] = []
        file_acc_5_vals: List[float] = []
        cov_4k_vals: List[float] = []

        for inst in instance_data:
            engine = inst["engine"]
            pi_query = inst["pi_query"]
            gold_files = inst["gold_files"]
            gold_sym_ids = inst["gold_sym_ids"]
            line_costs = inst["line_costs"]

            denom = np.power(engine.pi_global + 1e-8, g_val)
            scores = pi_query / denom
            ranking = list(np.argsort(-scores, kind="stable"))

            ret_sym_ids = [engine.symbols[i]["id"] for i in ranking]
            seen_files: Set[str] = set()
            ret_files: List[str] = []
            for i in ranking:
                fp = engine.symbols[i].get("file_path", "")
                if fp and fp not in seen_files:
                    seen_files.add(fp)
                    ret_files.append(fp)

            f_acc_5 = file_acc_at_k(ret_files, gold_files, k=5)
            fn_acc_10 = function_acc_at_k(ret_sym_ids, gold_sym_ids, k=10)
            cov = coverage_at_4k(ret_sym_ids, gold_sym_ids, symbol_token_costs=line_costs, budget=4096)
            hsi = hsi_at_10(ret_sym_ids, engine.hub_set)

            fn_acc_10_vals.append(fn_acc_10)
            hsi_10_vals.append(hsi)
            file_acc_5_vals.append(f_acc_5)
            cov_4k_vals.append(cov)

        fn_mean, fn_cil, fn_ciu = compute_bootstrap_ci(fn_acc_10_vals, n_bootstrap=2000, seed=seed)
        hsi_mean, hsi_cil, hsi_ciu = compute_bootstrap_ci(hsi_10_vals, n_bootstrap=2000, seed=seed)
        f5_mean, f5_cil, f5_ciu = compute_bootstrap_ci(file_acc_5_vals, n_bootstrap=2000, seed=seed)
        cov_mean, cov_cil, cov_ciu = compute_bootstrap_ci(cov_4k_vals, n_bootstrap=2000, seed=seed)

        sweep_results[g_key] = {
            "gamma": round(g_val, 2),
            "fn_acc_10": {
                "mean": round(fn_mean, 4),
                "ci_lower": round(fn_cil, 4),
                "ci_upper": round(fn_ciu, 4),
            },
            "hsi_10": {
                "mean": round(hsi_mean, 4),
                "ci_lower": round(hsi_cil, 4),
                "ci_upper": round(hsi_ciu, 4),
            },
            "file_acc_5": {
                "mean": round(f5_mean, 4),
                "ci_lower": round(f5_cil, 4),
                "ci_upper": round(f5_ciu, 4),
            },
            "coverage_4096": {
                "mean": round(cov_mean, 4),
                "ci_lower": round(cov_cil, 4),
                "ci_upper": round(cov_ciu, 4),
            },
        }

    print("\n" + "=" * 75)
    print(f"GAMMA SWEEP RESULTS ON {split} (N={len(instance_data)})")
    print("=" * 75)
    print(f"{'Gamma':<8} | {'Fn Acc@10 (95% CI)':<24} | {'HSI@10 (95% CI)':<24} | {'File Acc@5':<10}")
    print("-" * 75)
    for g_val in gammas:
        g_key = f"gamma_{g_val:.2f}"
        res = sweep_results[g_key]
        fn = res["fn_acc_10"]
        hsi = res["hsi_10"]
        f5 = res["file_acc_5"]
        fn_str = f"{fn['mean']*100:4.1f}% [{fn['ci_lower']*100:4.1f}%, {fn['ci_upper']*100:4.1f}%]"
        hsi_str = f"{hsi['mean']*100:4.1f}% [{hsi['ci_lower']*100:4.1f}%, {hsi['ci_upper']*100:4.1f}%]"
        print(f"{g_val:<8.2f} | {fn_str:<24} | {hsi_str:<24} | {f5['mean']*100:4.1f}%")
    print("=" * 75 + "\n")

    payload = {
        "suite": "gamma_sweep",
        "split": split,
        "seed": seed,
        "num_instances": len(instance_data),
        "gammas": list(gammas),
        "sweep": sweep_results,
    }

    saved = save_results(
        path=output_path,
        results_data=payload,
        split_name=split,
        seed=seed,
        suite="gamma_sweep",
    )
    print(f"Saved gamma sweep results to: {saved}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep gamma parameter on dev_val")
    parser.add_argument("--split", type=str, default="dev_val", help="Partition to evaluate (default: dev_val)")
    parser.add_argument("--gammas", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0", help="Comma-separated gammas")
    parser.add_argument("--seed", type=int, default=20261003, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    gammas = parse_gamma_list(args.gammas)
    out_p = Path(args.output).resolve() if args.output else None
    run_gamma_sweep(split=args.split, gammas=gammas, seed=args.seed, output_path=out_p)


if __name__ == "__main__":
    main()
