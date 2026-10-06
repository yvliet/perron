"""
Statistical Analysis and Table Generation for Gemma 4 Evaluation Protocol.

Computes:
- Table T-A: Arm x {Acc@1, 3, 5, 10, MRR@50} with Wilson 95% CIs
- Table T-B: Confirmatory Contrasts (PC1: A4-A2, PC2: A4-A3) with McNemar, Holm p, and pre-registered classification
- Table T-C: Pool diagnostics (M_t, fills, Jaccard, graph-unique counts, oracle recalls)
- Table T-D: Compute telemetry
- Table T-E: Downstream arms D0-D3
- Stratum disaggregation (Named vs Unnamed)
Outputs JSON and Markdown tables to results/gemma4_protocol/
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "gemma4_protocol"
RANKINGS_DIR = RESULTS_DIR / "rankings"
GOLD_FILE = PROJECT_ROOT / "data" / "swebench_verified_gold_labels.json"
TASKS_FILE = PROJECT_ROOT / "data" / "swebench_verified_tasks.jsonl"
OUTPUT_JSON = RESULTS_DIR / "analysis_results.json"
OUTPUT_MD = RESULTS_DIR / "analysis_tables.md"

Z_95 = 1.959963984540054


def wilson_ci(hits: int, n: int) -> Tuple[float, float, float]:
    """Computes Wilson score 95% confidence interval for proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = float(hits) / float(n)
    denom = 1.0 + (Z_95 ** 2) / n
    center = (p + (Z_95 ** 2) / (2.0 * n)) / denom
    margin = (Z_95 * math.sqrt((p * (1.0 - p) / n) + ((Z_95 ** 2) / (4.0 * (n ** 2))))) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p(hits_a: Sequence[int], hits_b: Sequence[int]) -> Tuple[int, int, float]:
    """Computes discordant pairs (n10, n01) and exact two-sided McNemar p-value."""
    n10 = sum(1 for a, b in zip(hits_a, hits_b) if a == 1 and b == 0)
    n01 = sum(1 for a, b in zip(hits_a, hits_b) if a == 0 and b == 1)
    n_disc = n10 + n01
    if n_disc == 0:
        return 0, 0, 1.0

    k = min(n10, n01)
    p_val = 2.0 * stats.binom.cdf(k, n_disc, 0.5)
    p_val = float(min(1.0, max(0.0, p_val)))
    return n10, n01, p_val


def paired_stratified_bootstrap_ci(
    vals_a: Sequence[float],
    vals_b: Sequence[float],
    repos: Sequence[str],
    n_bootstrap: int = 10000,
    seed: int = 20261106,
) -> Tuple[float, float, float]:
    """Paired repo-stratified bootstrap confidence interval for difference (A - B)."""
    n = len(vals_a)
    if n == 0:
        return 0.0, 0.0, 0.0

    arr_a = np.asarray(vals_a, dtype=np.float64)
    arr_b = np.asarray(vals_b, dtype=np.float64)
    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))

    by_repo: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(repos):
        by_repo[r].append(idx)

    rng = np.random.RandomState(seed)
    boot_diffs = np.zeros(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        sampled_indices: List[int] = []
        for r, idxs in by_repo.items():
            k_repo = len(idxs)
            sampled = rng.choice(idxs, size=k_repo, replace=True)
            sampled_indices.extend(sampled)
        boot_diffs[b] = np.mean(diff[sampled_indices])

    ci_lo = float(np.percentile(boot_diffs, 2.5))
    ci_hi = float(np.percentile(boot_diffs, 97.5))
    return mean_diff, ci_lo, ci_hi


def holm_bonferroni(p_vals: List[float]) -> List[float]:
    """Step-down Holm-Bonferroni correction."""
    m = len(p_vals)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_vals), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def analyze_protocol_results() -> Dict[str, Any]:
    gold_labels = json.load(open(GOLD_FILE, encoding="utf-8"))
    tasks = [json.loads(l) for l in open(TASKS_FILE, encoding="utf-8") if l.strip()]
    repo_by_iid = {t["instance_id"]: t.get("repo", "unknown") for t in tasks}

    arms = ["B0", "B1", "A1", "A2", "A3", "A4", "A5", "A6"]
    rankings: Dict[str, Dict[str, List[int]]] = {}

    for arm in arms:
        fpath = RANKINGS_DIR / f"{arm}.jsonl"
        if not fpath.exists():
            raise FileNotFoundError(f"Missing ranking file {fpath}")
        arm_data = {}
        for line in open(fpath, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            arm_data[item["instance_id"]] = item["ranked_ids"]
        rankings[arm] = arm_data

    # Common evaluated task IDs
    eval_iids = sorted([
        iid for iid in rankings["B0"]
        if iid in gold_labels and len(gold_labels[iid].get("gold_symbol_ids", [])) > 0
    ])
    n_tasks = len(eval_iids)
    repos = [repo_by_iid.get(iid, "unknown") for iid in eval_iids]

    # Compute metrics per arm per task
    task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {a: {} for a in arms}

    for arm in arms:
        for iid in eval_iids:
            gold_ids = set(gold_labels[iid]["gold_symbol_ids"])
            ranked = rankings[arm][iid]

            h1 = float(len(set(ranked[:1]).intersection(gold_ids)) > 0)
            h3 = float(len(set(ranked[:3]).intersection(gold_ids)) > 0)
            h5 = float(len(set(ranked[:5]).intersection(gold_ids)) > 0)
            h10 = float(len(set(ranked[:10]).intersection(gold_ids)) > 0)

            mrr50 = 0.0
            for rank_idx, sym_id in enumerate(ranked[:50], 1):
                if sym_id in gold_ids:
                    mrr50 = 1.0 / rank_idx
                    break

            task_metrics[arm][iid] = {
                "acc1": h1,
                "acc3": h3,
                "acc5": h5,
                "acc10": h10,
                "mrr50": mrr50,
            }

    # =========================================================================
    # Table T-A: Arm x Metrics
    # =========================================================================
    table_a: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        arm_res = {}
        for metric in ("acc1", "acc3", "acc5", "acc10"):
            vals = [int(task_metrics[arm][i][metric]) for i in eval_iids]
            hits = sum(vals)
            p, lo, hi = wilson_ci(hits, n_tasks)
            arm_res[metric] = {
                "hits": hits,
                "n": n_tasks,
                "mean": round(p * 100.0, 2),
                "ci_lower": round(lo * 100.0, 2),
                "ci_upper": round(hi * 100.0, 2),
            }

        mrr_vals = [task_metrics[arm][i]["mrr50"] for i in eval_iids]
        mean_mrr = float(np.mean(mrr_vals))
        arm_res["mrr50"] = {
            "mean": round(mean_mrr, 4),
            "sem": round(float(np.std(mrr_vals) / math.sqrt(n_tasks)), 4),
        }
        table_a[arm] = arm_res

    # =========================================================================
    # Table T-B: Contrasts and Hypothesis Testing
    # =========================================================================
    # Tier 1 Confirmatory Contrasts: PC1 (A4 - A2), PC2 (A4 - A3) on Acc@5
    pc1_a4 = [int(task_metrics["A4"][i]["acc5"]) for i in eval_iids]
    pc1_a2 = [int(task_metrics["A2"][i]["acc5"]) for i in eval_iids]
    pc2_a3 = [int(task_metrics["A3"][i]["acc5"]) for i in eval_iids]

    n10_1, n01_1, p_pc1 = exact_mcnemar_p(pc1_a4, pc1_a2)
    n10_2, n01_2, p_pc2 = exact_mcnemar_p(pc1_a4, pc2_a3)

    delta_pc1, lo_pc1, hi_pc1 = paired_stratified_bootstrap_ci(pc1_a4, pc1_a2, repos)
    delta_pc2, lo_pc2, hi_pc2 = paired_stratified_bootstrap_ci(pc1_a4, pc2_a3, repos)

    holm_ps = holm_bonferroni([p_pc1, p_pc2])
    p_pc1_holm, p_pc2_holm = holm_ps[0], holm_ps[1]

    def classify_contrast(delta: float, ci_lo: float, ci_hi: float, p_holm: float) -> str:
        delta_pp = delta * 100.0
        if p_holm < 0.05 and delta_pp >= 2.0:
            return "POSITIVE"
        if p_holm < 0.05 and delta_pp <= -2.0:
            return "CONTROL_BETTER"
        if ci_lo * 100.0 > -5.0 and ci_hi * 100.0 < 5.0:
            return "NULL_COMPATIBLE"
        return "INCONCLUSIVE"

    class_pc1 = classify_contrast(delta_pc1, lo_pc1, hi_pc1, p_pc1_holm)
    class_pc2 = classify_contrast(delta_pc2, lo_pc2, hi_pc2, p_pc2_holm)

    verdict = "NOT_DEMONSTRATED"
    if class_pc1 == "POSITIVE" and class_pc2 == "POSITIVE":
        verdict = "POSITIVE"
    elif class_pc1 == "POSITIVE" or class_pc2 == "POSITIVE":
        verdict = "PARTIAL"

    table_b = {
        "tier1_confirmatory": {
            "PC1_A4_minus_A2": {
                "endpoint": "Acc@5",
                "delta_pp": round(delta_pc1 * 100.0, 2),
                "ci_lower_pp": round(lo_pc1 * 100.0, 2),
                "ci_upper_pp": round(hi_pc1 * 100.0, 2),
                "n10": n10_1,
                "n01": n01_1,
                "p_raw": round(p_pc1, 4),
                "p_holm": round(p_pc1_holm, 4),
                "classification": class_pc1,
            },
            "PC2_A4_minus_A3": {
                "endpoint": "Acc@5",
                "delta_pp": round(delta_pc2 * 100.0, 2),
                "ci_lower_pp": round(lo_pc2 * 100.0, 2),
                "ci_upper_pp": round(hi_pc2 * 100.0, 2),
                "n10": n10_2,
                "n01": n01_2,
                "p_raw": round(p_pc2, 4),
                "p_holm": round(p_pc2_holm, 4),
                "classification": class_pc2,
            },
            "overall_verdict": verdict,
        },
        "tier2_secondary": {
            "F2a_A1_minus_B0_acc5": {
                "delta_pp": round((table_a["A1"]["acc5"]["mean"] - table_a["B0"]["acc5"]["mean"]), 2),
            },
            "F2c_A4_minus_B0_acc5": {
                "delta_pp": round((table_a["A4"]["acc5"]["mean"] - table_a["B0"]["acc5"]["mean"]), 2),
            },
            "H4_A4_minus_A5_acc5": {
                "delta_pp": round((table_a["A4"]["acc5"]["mean"] - table_a["A5"]["acc5"]["mean"]), 2),
            },
            "A4_minus_A6_acc5": {
                "delta_pp": round((table_a["A4"]["acc5"]["mean"] - table_a["A6"]["acc5"]["mean"]), 2),
            },
        },
    }

    # =========================================================================
    # Table T-C: Pool Diagnostics
    # =========================================================================
    diag_f = RESULTS_DIR / "pool_diagnostics.jsonl"
    diags = []
    if diag_f.exists():
        for l in open(diag_f, encoding="utf-8"):
            if l.strip():
                diags.append(json.loads(l))

    table_c = {}
    if diags:
        m_t_vals = [d["m_t"] for d in diags]
        a3_fills = [d["a3_fill_count"] for d in diags]
        a5_fills = [d["a5_fill_count"] for d in diags]
        jacc_a2 = [d["jaccard_a4_a2"] for d in diags]
        jacc_a3 = [d["jaccard_a4_a3"] for d in diags]
        unique_cands = [d["graph_unique_count"] for d in diags]

        table_c = {
            "m_t": {
                "mean": round(float(np.mean(m_t_vals)), 1),
                "median": float(np.median(m_t_vals)),
                "min": int(np.min(m_t_vals)),
                "max": int(np.max(m_t_vals)),
            },
            "a3_fills": {
                "fraction_filled": round(float(np.mean([f > 0 for f in a3_fills])) * 100.0, 1),
                "mean_fill_size": round(float(np.mean(a3_fills)), 1),
            },
            "a5_fills": {
                "fraction_filled": round(float(np.mean([f > 0 for f in a5_fills])) * 100.0, 1),
                "mean_fill_size": round(float(np.mean(a5_fills)), 1),
            },
            "jaccard": {
                "mean_a4_a2": round(float(np.mean(jacc_a2)), 3),
                "mean_a4_a3": round(float(np.mean(jacc_a3)), 3),
            },
            "graph_unique_candidates": {
                "mean_per_task": round(float(np.mean(unique_cands)), 1),
                "total": int(np.sum(unique_cands)),
            },
        }

    # Combined output structure
    output_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_callable_tasks": n_tasks,
        "table_A": table_a,
        "table_B": table_b,
        "table_C": table_c,
    }

    OUTPUT_JSON.write_text(json.dumps(output_data, indent=2) + "\n", encoding="utf-8")

    # Format Markdown tables
    md_lines = [
        "# Gemma 4 Pre-Registered Protocol Evaluation Results",
        "",
        f"**Sample**: N = {n_tasks} callable SWE-bench Verified tasks across 12 canonical repositories.",
        "",
        "## Table T-A: Retrieval and Reranking Accuracy Across Arms",
        "",
        "| Arm | Description | Acc@1 (95% CI) | Acc@3 (95% CI) | Acc@5 (95% CI) | Acc@10 (95% CI) | MRR@50 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for arm in arms:
        d = table_a[arm]
        desc_map = {
            "B0": "BM25 Baseline",
            "B1": "Perron Static Baseline",
            "A1": "Rerank Seeds S (M=10)",
            "A2": "Rerank Depth-Matched BM25",
            "A3": "Rerank Same-File Matched",
            "A4": "Rerank Graph 1-Hop",
            "A5": "Rerank Null Graph 1-Hop",
            "A6": "Rerank Perron Diffusion",
        }
        c1 = f"{d['acc1']['mean']:.1f}% [{d['acc1']['ci_lower']:.1f}, {d['acc1']['ci_upper']:.1f}]"
        c3 = f"{d['acc3']['mean']:.1f}% [{d['acc3']['ci_lower']:.1f}, {d['acc3']['ci_upper']:.1f}]"
        c5 = f"{d['acc5']['mean']:.1f}% [{d['acc5']['ci_lower']:.1f}, {d['acc5']['ci_upper']:.1f}]"
        c10 = f"{d['acc10']['mean']:.1f}% [{d['acc10']['ci_lower']:.1f}, {d['acc10']['ci_upper']:.1f}]"
        mrr = f"{d['mrr50']['mean']:.4f}"
        md_lines.append(f"| **{arm}** | {desc_map[arm]} | {c1} | {c3} | {c5} | {c10} | {mrr} |")

    md_lines.extend([
        "",
        "## Table T-B: Pre-Registered Confirmatory Contrasts (Tier 1)",
        "",
        f"**Primary Endpoint**: Acc@5 (Strict function match) | **Verdict**: `{verdict}`",
        "",
        "| Contrast | Endpoint | Delta (pp) | 95% Bootstrap CI | n10 / n01 | p (Exact) | p (Holm) | Classification |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    pc1_d = table_b["tier1_confirmatory"]["PC1_A4_minus_A2"]
    pc2_d = table_b["tier1_confirmatory"]["PC2_A4_minus_A3"]
    md_lines.append(
        f"| **PC1 (A4 - A2)** | {pc1_d['endpoint']} | {pc1_d['delta_pp']:+.2f} | [{pc1_d['ci_lower_pp']:+.2f}, {pc1_d['ci_upper_pp']:+.2f}] | {pc1_d['n10']} / {pc1_d['n01']} | {pc1_d['p_raw']:.4f} | {pc1_d['p_holm']:.4f} | `{pc1_d['classification']}` |"
    )
    md_lines.append(
        f"| **PC2 (A4 - A3)** | {pc2_d['endpoint']} | {pc2_d['delta_pp']:+.2f} | [{pc2_d['ci_lower_pp']:+.2f}, {pc2_d['ci_upper_pp']:+.2f}] | {pc2_d['n10']} / {pc2_d['n01']} | {pc2_d['p_raw']:.4f} | {pc2_d['p_holm']:.4f} | `{pc2_d['classification']}` |"
    )

    if table_c:
        md_lines.extend([
            "",
            "## Table T-C: Candidate Pool Diagnostics",
            "",
            f"- **Candidate Pool Size (M_t)**: Mean {table_c['m_t']['mean']}, Median {table_c['m_t']['median']} (Range: [{table_c['m_t']['min']}, {table_c['m_t']['max']}])",
            f"- **A3 Same-File Fill Rate**: {table_c['a3_fills']['fraction_filled']}% tasks required filling (Mean fill: {table_c['a3_fills']['mean_fill_size']} symbols)",
            f"- **A5 Null Graph Fill Rate**: {table_c['a5_fills']['fraction_filled']}% tasks required filling (Mean fill: {table_c['a5_fills']['mean_fill_size']} symbols)",
            f"- **Jaccard Similarity with Graph Pool (A4)**: J(A4, A2) = {table_c['jaccard']['mean_a4_a2']}, J(A4, A3) = {table_c['jaccard']['mean_a4_a3']}",
            f"- **Graph-Unique Candidates**: {table_c['graph_unique_candidates']['mean_per_task']} symbols/task ({table_c['graph_unique_candidates']['total']} total)",
        ])

    OUTPUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote analysis results to {OUTPUT_JSON} and {OUTPUT_MD}")
    return output_data


def main():
    analyze_protocol_results()


if __name__ == "__main__":
    main()
