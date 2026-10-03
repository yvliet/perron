"""
Perron Benchmark Module.
Executes the Hub-Gold vs Non-Hub-Gold ablation benchmark across splits
and reports the four canonical metrics: Acc@10(H), Acc@10(N), HSI@10, and HEADLINE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


METHOD_ALIASES: Dict[str, str] = {
    "bm25": "bm25",
    "bm25_1hop": "bm25_1hop",
    "dense": "dense",
    "standard_ppr": "standard_ppr",
    "std_ppr": "standard_ppr",
    "deg_ppr": "deg_ppr",
    "degree_norm_ppr": "deg_ppr",
    "aider": "aider",
    "blocklist": "blocklist_lex",
    "blocklist_lex": "blocklist_lex",
    "deg_discount_matrix": "deg_discount_matrix",
    "hipporag": "hipporag",
    "query_reweighted_ppr": "query_reweighted_ppr",
    "perron_static": "perron_static",
    "perron": "perron",
    "oracle": "oracle",
}


def load_gold_data(gold_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load gold records from data_release/gold.jsonl or fallback to data/."""
    if gold_path and gold_path.exists():
        p = gold_path
    elif (REPO_ROOT / "data_release" / "gold.jsonl").exists():
        p = REPO_ROOT / "data_release" / "gold.jsonl"
    else:
        # Fallback build on the fly
        from benchmarks.gold_patch_parser import GoldPatchParser
        parser = GoldPatchParser()
        parser.export_gold_jsonl()
        p = REPO_ROOT / "data_release" / "gold.jsonl"

    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_benchmark(
    method: str,
    split: str = "dev_val",
    results_path: Optional[Path] = None,
    gold_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run benchmark for a given method and split.
    Returns:
        dict with acc_h, acc_n, hsi, headline, count_h, count_n, count_total.
    """
    canonical_method = METHOD_ALIASES.get(method.lower(), method.lower())

    rp = results_path or (REPO_ROOT / "data" / "retrieval_sweep_results.json")
    if not rp.exists():
        raise FileNotFoundError(f"Missing retrieval results at {rp}")

    with open(rp, "r", encoding="utf-8") as f:
        sweep_data = json.load(f)

    instances_data = sweep_data.get("instances", {})
    gold_records = load_gold_data(gold_path)

    # Filter instances by split
    if split in ("dev_val", "heldout", "pilot"):
        split_records = [g for g in gold_records if g.get("split") == split]
    elif split == "dev":
        split_records = [g for g in gold_records if g.get("split") in ("pilot", "dev_val")]
    elif split in ("full", "all"):
        split_records = gold_records
    else:
        raise ValueError(f"Unknown split: {split}. Choose from dev_val, heldout, dev, full.")

    h_instances = {g["instance_id"] for g in split_records if g.get("is_hub_gold")}
    n_instances = {g["instance_id"] for g in split_records if not g.get("is_hub_gold")}
    all_instances = h_instances | n_instances

    h_accs: List[float] = []
    n_accs: List[float] = []
    hsis: List[float] = []

    for iid in all_instances:
        if iid not in instances_data:
            continue
        m_res = instances_data[iid].get(canonical_method)
        if not m_res:
            continue

        acc = float(m_res.get("fn_acc_10", 0.0)) * 100.0
        hsi_val = float(m_res.get("hsi", 0.0)) * 100.0

        if iid in h_instances:
            h_accs.append(acc)
        else:
            n_accs.append(acc)
        hsis.append(hsi_val)

    mean_h = (sum(h_accs) / len(h_accs)) if h_accs else 0.0
    mean_n = (sum(n_accs) / len(n_accs)) if n_accs else 0.0
    mean_hsi = (sum(hsis) / len(hsis)) if hsis else 0.0
    headline = 0.5 * (mean_h + mean_n)

    return {
        "method": canonical_method,
        "split": split,
        "acc_h": round(mean_h, 2),
        "acc_n": round(mean_n, 2),
        "hsi": round(mean_hsi, 2),
        "headline": round(headline, 2),
        "count_h": len(h_accs),
        "count_n": len(n_accs),
        "count_total": len(hsis),
    }


def format_benchmark_report(result: Dict[str, Any]) -> str:
    """Format benchmark result into clean publication text."""
    lines = [
        f"Perron Hub-Gold Benchmark: {result['method'].upper()} (Split: {result['split']})",
        "=" * 65,
        f"Subset H (Hub-Gold, N={result['count_h']}):        Acc@10 = {result['acc_h']:5.1f}%",
        f"Subset N (Non-Hub-Gold, N={result['count_n']}):    Acc@10 = {result['acc_n']:5.1f}%",
        f"Hub Suppression Index (HSI@10):       HSI    = {result['hsi']:5.1f}%",
        "-" * 65,
        f"HEADLINE METRIC (Macro-Average):       Score  = {result['headline']:5.1f}%",
        "=" * 65,
    ]
    return "\n".join(lines)
