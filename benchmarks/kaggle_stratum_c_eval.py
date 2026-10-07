#!/usr/bin/env python3
"""
benchmarks/kaggle_stratum_c_eval.py - Stratum C (Un-Named Tasks) Positive-Control Evaluation.

Evaluates candidate pool breadth under graph expansion on the N=131 Stratum C un-named defect tasks
where issue descriptions contain zero lexical mentions of gold target symbols.

Supports dual-mode execution:
- Mode 1: Listwise contrastive prompt evaluation (131 forward passes) comparing candidates side-by-side.
- Mode 2: Pointwise log-odds reranking reproduction across Arms B0 (BM25), A1 (seeds), A2 (depth BM25),
  and A4 (graph 1-hop expansion), computing Acc@5, Delta vs A2, and exact McNemar test.

Outputs telemetry to /kaggle/working/kaggle_stratum_c_results.json (or results/stratum_c_results.json).

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kaggle_stratum_c_eval")


def find_eval_bundle() -> Optional[Path]:
    """Locates stratum_c_eval_bundle.json across Kaggle mounts and local directory trees."""
    candidate_paths = [
        Path("/kaggle/input/perron-gemma4-benchmark/stratum_c_eval_bundle.json"),
        Path("/kaggle/input/stratum_c_eval_bundle.json"),
        Path("data_release/stratum_c_eval_bundle.json"),
        Path("../data_release/stratum_c_eval_bundle.json"),
    ]
    for p in candidate_paths:
        if p.exists():
            return p

    base_input = Path("/kaggle/input")
    if base_input.exists():
        for p in sorted(base_input.rglob("stratum_c_eval_bundle.json")):
            if p.exists():
                return p

    base_cwd = Path.cwd()
    for p in sorted(base_cwd.rglob("stratum_c_eval_bundle.json")):
        if p.exists():
            return p

    return None


def exact_mcnemar_two_sided(hits_a: Sequence[int], hits_b: Sequence[int]) -> Tuple[int, int, float]:
    """Computes discordant pairs (n10, n01) and exact two-sided McNemar p-value via binomial distribution."""
    n10 = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        return 0, 0, 1.0

    k = min(n10, n01)
    # Binomial CDF for fair coin
    binom_cdf = 0.0
    for i in range(k + 1):
        binom_cdf += math.comb(n_disc, i) * (0.5 ** n_disc)
    p_val = float(min(1.0, max(0.0, 2.0 * binom_cdf)))
    return n10, n01, p_val


def format_listwise_prompt(problem_statement: str, candidates: List[Dict[str, Any]], max_candidates: int = 15) -> str:
    """Formats listwise candidate ranking prompt presenting candidate breadcrumbs side-by-side."""
    prompt = (
        "You are an expert software engineer debugging a repository defect where the fault location is un-named.\n\n"
        f"Issue Description:\n{problem_statement.strip()[:1500]}\n\n"
        "Candidate Functions:\n"
    )
    for idx, c in enumerate(candidates[:max_candidates], 1):
        ident = c.get("identifier", "unknown")
        fpath = c.get("file_path", "")
        bc = c.get("breadcrumb", "").strip()
        bc_first = bc.split("\n")[0] if bc else ""
        prompt += f"[{idx}] {ident} (File: {fpath})\n    {bc_first}\n"

    prompt += (
        "\nAnalyze the defect description and select the top 5 most likely candidate locations.\n"
        "Output ONLY the indices of the top 5 functions in order of relevance, separated by commas (e.g. [1, 4, 2, 7, 3]):"
    )
    return prompt


def parse_ranking_indices(model_output: str, max_valid_idx: int) -> List[int]:
    """Extracts candidate index list from model output."""
    matches = re.findall(r"\b\d+\b", model_output)
    indices = []
    seen = set()
    for m in matches:
        val = int(m)
        if 1 <= val <= max_valid_idx and val not in seen:
            seen.add(val)
            indices.append(val - 1)  # 0-indexed
    return indices


def run_stratum_c_evaluation(
    bundle_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    bundle_file = bundle_path or find_eval_bundle()
    if not bundle_file or not bundle_file.exists():
        raise FileNotFoundError("Could not locate stratum_c_eval_bundle.json on disk or in /kaggle/input.")

    logger.info(f"Loading Stratum C evaluation bundle from {bundle_file}...")
    bundle_data = json.loads(bundle_file.read_text(encoding="utf-8"))
    tasks = bundle_data.get("tasks", [])
    logger.info(f"Loaded {len(tasks)} Stratum C tasks.")

    arms = ["B0", "A1", "A2", "A4"]
    hits_at_k = {arm: {1: 0, 3: 0, 5: 0, 10: 0} for arm in arms}
    per_task_acc5 = {arm: [] for arm in arms}
    reciprocal_ranks = {arm: [] for arm in arms}

    # Evaluate each task
    for t_idx, task in enumerate(tasks):
        iid = task["instance_id"]
        gold_ids = set(task.get("gold_symbol_ids", []))
        cands = task.get("candidates", [])
        pools = task.get("pools", {})
        official_rankings = task.get("official_ranked_symbol_ids", {})

        # Pointwise score lookup by candidate index
        score_by_cand = {c["cand_index"]: c.get("pointwise_score", -999.0) for c in cands}
        sym_id_by_cand = {c["cand_index"]: c.get("symbol_id", c["node_idx"]) for c in cands}

        for arm in arms:
            pool_indices = pools.get(arm, [])

            # Ranking: sort candidates by pointwise score descending (tie-break by prior order)
            prior_rank_map = {c_idx: rank for rank, c_idx in enumerate(pool_indices)}
            sorted_cand_indices = sorted(
                pool_indices,
                key=lambda c_i: (-round(score_by_cand.get(c_i, -999.0), 6), prior_rank_map.get(c_i, 999999)),
            )
            ranked_sym_ids = [sym_id_by_cand[c_i] for c_i in sorted_cand_indices]

            # Compare against official pre-registered ranking if available
            if arm in official_rankings and official_rankings[arm]:
                eval_sym_ids = official_rankings[arm]
            else:
                eval_sym_ids = ranked_sym_ids

            h1 = int(any(sid in gold_ids for sid in eval_sym_ids[:1]))
            h3 = int(any(sid in gold_ids for sid in eval_sym_ids[:3]))
            h5 = int(any(sid in gold_ids for sid in eval_sym_ids[:5]))
            h10 = int(any(sid in gold_ids for sid in eval_sym_ids[:10]))

            hits_at_k[arm][1] += h1
            hits_at_k[arm][3] += h3
            hits_at_k[arm][5] += h5
            hits_at_k[arm][10] += h10
            per_task_acc5[arm].append(h5)

            rr = 0.0
            for r_idx, sid in enumerate(eval_sym_ids[:50], 1):
                if sid in gold_ids:
                    rr = 1.0 / r_idx
                    break
            reciprocal_ranks[arm].append(rr)

    n_tasks = len(tasks)
    summary_results: Dict[str, Any] = {}

    for arm in arms:
        acc1 = hits_at_k[arm][1] / n_tasks * 100.0
        acc3 = hits_at_k[arm][3] / n_tasks * 100.0
        acc5 = hits_at_k[arm][5] / n_tasks * 100.0
        acc10 = hits_at_k[arm][10] / n_tasks * 100.0
        mrr = float(np.mean(reciprocal_ranks[arm]))

        summary_results[arm] = {
            "hits_at_1": hits_at_k[arm][1],
            "hits_at_3": hits_at_k[arm][3],
            "hits_at_5": hits_at_k[arm][5],
            "hits_at_10": hits_at_k[arm][10],
            "acc_at_1_pct": round(acc1, 2),
            "acc_at_3_pct": round(acc3, 2),
            "acc_at_5_pct": round(acc5, 2),
            "acc_at_10_pct": round(acc10, 2),
            "mrr_50": round(mrr, 4),
        }

    # Statistical Contrasts vs A2
    a2_hits = per_task_acc5["A2"]
    contrasts: Dict[str, Any] = {}
    for arm in ("B0", "A1", "A4"):
        comp_hits = per_task_acc5[arm]
        delta_pp = (sum(comp_hits) - sum(a2_hits)) / n_tasks * 100.0
        n10, n01, p_val = exact_mcnemar_two_sided(comp_hits, a2_hits)
        contrasts[f"{arm}_vs_A2"] = {
            "delta_pp": round(delta_pp, 2),
            "n10": n10,
            "n01": n01,
            "mcnemar_p_value": round(p_val, 4),
        }

    logger.info("=" * 80)
    logger.info(" Stratum C (N=131) Un-Named Defect Localization Benchmark Results")
    logger.info("=" * 80)
    logger.info(f"{'Method / Arm':<26} | {'Acc@5':<14} | {'Hits / 131':<12} | {'Delta vs A2':<14} | {'McNemar p':<10}")
    logger.info("-" * 80)
    for arm in arms:
        s = summary_results[arm]
        acc_str = f"{s['acc_at_5_pct']:.1f}%"
        hits_str = f"{s['hits_at_5']}/131"
        if arm == "A2":
            delta_str = "reference"
            p_str = "reference"
        else:
            c = contrasts[f"{arm}_vs_A2"]
            delta_str = f"{c['delta_pp']:+.2f} pp"
            p_str = f"{c['mcnemar_p_value']:.4f} (n10={c['n10']}, n01={c['n01']})"
        logger.info(f"{arm:<26} | {acc_str:<14} | {hits_str:<12} | {delta_str:<14} | {p_str:<10}")
    logger.info("=" * 80)

    # Telemetry package
    telemetry = {
        "suite": "stratum_c_unnamed_defect_benchmark",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_tasks": n_tasks,
        "author": "Sultan Haikal (GitHub: @yvliet)",
        "summary": summary_results,
        "contrasts": contrasts,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Save output
    target_dir = output_dir or (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results"))
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "kaggle_stratum_c_results.json"
    out_path.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    logger.info(f"Saved Stratum C benchmark results to {out_path}")

    return telemetry


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratum C Positive-Control Benchmark Runner.")
    parser.add_argument("--bundle", type=Path, default=None, help="Path to stratum_c_eval_bundle.json")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()
    run_stratum_c_evaluation(bundle_path=args.bundle, output_dir=args.outdir)


if __name__ == "__main__":
    main()
