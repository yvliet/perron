"""
Hub-Gold vs Non-Hub-Gold Dissection and Ablation Analysis.

Analyzes the failure modes of naive hub blocklisting and degree-normalization
against Perron's Query-Level Hub Adaptivity across the 300 SWE-bench Lite tasks.
Segments tasks into:
- Hub-Gold instances (N = 47, 15.7%): Ground-truth defect resides in top-25 hub.
- Non-Hub-Gold instances (N = 253, 84.3%): Ground-truth defect resides in non-hub symbols.

Generates structured comparison tables for paper/main.tex and paper/paper_writeup.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_hub_gold_analysis(
    results_file: Path = Path("data/retrieval_sweep_results.json"),
    output_file: Path = Path("data/hub_gold_ablation_results.json"),
) -> Dict[str, Any]:
    """Analyze hub-gold trade-offs and output formatted ablation metrics."""
    if not results_file.exists():
        raise FileNotFoundError(f"Sweep results file not found: {results_file}. Run run_retrieval_sweep.py first.")

    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    subsets = data.get("aggregated", {}).get("subsets", {})
    hg = subsets.get("hub_gold", {})
    nhg = subsets.get("non_hub_gold", {})
    full = subsets.get("full", {})
    heldout = subsets.get("heldout", {})

    methods = [
        "bm25",
        "dense",
        "aider",
        "blocklist",
        "deg_ppr",
        "perron",
    ]

    analysis: Dict[str, Any] = {
        "summary": "Hub-Gold vs Non-Hub-Gold Trade-off Analysis",
        "breakdown": {},
    }

    print("\n" + "=" * 100)
    print("HUB-GOLD VS NON-HUB-GOLD TRADE-OFF ABLATION (SWE-bench Lite, N=300)")
    print("=" * 100)
    print(f"{'Method':<18} | {'Hub-Gold (N=47)':^36} | {'Non-Hub-Gold (N=253)':^36}")
    print(f"{'':<18} | {'File@1':<8} {'File@5':<8} {'Fn@10':<8} {'HSI':<8} | {'File@1':<8} {'File@5':<8} {'Fn@10':<8} {'HSI':<8}")
    print("-" * 100)

    for m in methods:
        hg_m = hg.get(m, {}).get("metrics", {})
        nhg_m = nhg.get(m, {}).get("metrics", {})
        label = hg.get(m, {}).get("label", m)

        hg_f1 = hg_m.get("file_acc_1", {}).get("mean", 0.0) * 100
        hg_f5 = hg_m.get("file_acc_5", {}).get("mean", 0.0) * 100
        hg_fn10 = hg_m.get("fn_acc_10", {}).get("mean", 0.0) * 100
        hg_hsi = hg_m.get("hsi", {}).get("mean", 0.0) * 100

        nhg_f1 = nhg_m.get("file_acc_1", {}).get("mean", 0.0) * 100
        nhg_f5 = nhg_m.get("file_acc_5", {}).get("mean", 0.0) * 100
        nhg_fn10 = nhg_m.get("fn_acc_10", {}).get("mean", 0.0) * 100
        nhg_hsi = nhg_m.get("hsi", {}).get("mean", 0.0) * 100

        analysis["breakdown"][m] = {
            "label": label,
            "hub_gold": {
                "file_acc_1": round(hg_f1, 2),
                "file_acc_5": round(hg_f5, 2),
                "fn_acc_10": round(hg_fn10, 2),
                "hsi": round(hg_hsi, 2),
            },
            "non_hub_gold": {
                "file_acc_1": round(nhg_f1, 2),
                "file_acc_5": round(nhg_f5, 2),
                "fn_acc_10": round(nhg_fn10, 2),
                "hsi": round(nhg_hsi, 2),
            },
        }

        print(
            f"{label:<18} | {hg_f1:>6.1f}%  {hg_f5:>6.1f}%  {hg_fn10:>6.1f}%  {hg_hsi:>6.1f}% | "
            f"{nhg_f1:>6.1f}%  {nhg_f5:>6.1f}%  {nhg_fn10:>6.1f}%  {nhg_hsi:>6.1f}%"
        )
    print("=" * 100 + "\n")

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    return analysis


if __name__ == "__main__":
    run_hub_gold_analysis()
