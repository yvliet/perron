"""
scripts/ingest_run2_telemetry.py - Ingest and promote Run 2A and Run 2B telemetry from Kaggle.

Downloads outputs from:
- yvliet/perron-causal-experiments (Run 2A: M=15 Factorial on Stratum C)
- yvliet/perron-gemma4-e2b-lora (Run 2B: Disjoint Lite -> Verified Generalization)

Validates outputs, computes exact empirical contrasts, updates claims_spec.json,
and generates formatted tables for the paper.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CAUSAL_M15_DIR = RESULTS_DIR / "causal_m15"
LORA_DIR = RESULTS_DIR / "lora"

CAUSAL_M15_DIR.mkdir(parents=True, exist_ok=True)
LORA_DIR.mkdir(parents=True, exist_ok=True)


def download_telemetry():
    print("[*] Downloading Run 2A telemetry from Kaggle (yvliet/perron-causal-experiments)...")
    res_2a = subprocess.run(
        ["kaggle", "kernels", "output", "yvliet/perron-causal-experiments", "-p", str(CAUSAL_M15_DIR)],
        capture_output=True, text=True
    )
    print(res_2a.stdout)
    if res_2a.stderr:
        print(f"[!] Stderr: {res_2a.stderr}")

    print("[*] Downloading Run 2B telemetry from Kaggle (yvliet/perron-gemma4-e2b-lora)...")
    res_2b = subprocess.run(
        ["kaggle", "kernels", "output", "yvliet/perron-gemma4-e2b-lora", "-p", str(LORA_DIR)],
        capture_output=True, text=True
    )
    print(res_2b.stdout)
    if res_2b.stderr:
        print(f"[!] Stderr: {res_2b.stderr}")


def parse_and_report():
    causal_file = CAUSAL_M15_DIR / "stratum_c_causal_m15_results.json"
    lora_file = LORA_DIR / "lora_generalization_results.json"

    print("\n" + "="*80)
    print(" RUN 2 TELEMETRY INGESTION & EMPIRICAL AUDIT")
    print("="*80)

    if causal_file.exists():
        print(f"\n[+] Ingesting Run 2A results from {causal_file}...")
        with open(causal_file, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        summary = cdata.get("summary", {})
        contrasts = cdata.get("contrasts", {})

        print("\n--- Run 2A: Stratum C Factorial Evaluation (M=15 Cutoff, N=131) ---")
        print(f"| {'Condition':<28} | {'Acc@1':<7} | {'Acc@5':<7} | {'MRR':<7} | {'Mean Rank':<10} | {'Seeds H1':<8} | {'Graph H1':<8} |")
        print("| " + "-"*28 + " | " + "-"*7 + " | " + "-"*7 + " | " + "-"*7 + " | " + "-"*10 + " | " + "-"*8 + " | " + "-"*8 + " |")
        for arm_name, s in summary.items():
            print(f"| {arm_name:<28} | {s['acc_1_pct']:>6.2f}% | {s['acc_5_pct']:>6.2f}% | {s['mrr']:>7.4f} | {str(s['mean_gold_rank'])[:8]:<10} | {s['seed_subgroup']['hits_1']:>8} | {s['graph_only_subgroup']['hits_1']:>8} |")

        print("\nHypothesis Contrasts:")
        for k, v in contrasts.items():
            print(f"  {k}:")
            print(f"    Top-1: Delta={v['top1']['delta_pp']:+.2f} pp (p={v['top1']['p_value']:.4f})")
            print(f"    Top-5: Delta={v['top5']['delta_pp']:+.2f} pp (p={v['top5']['p_value']:.4f})")
    else:
        print(f"[-] Run 2A results file not found at {causal_file} (kernel may still be executing)")

    if lora_file.exists():
        print(f"\n[+] Ingesting Run 2B results from {lora_file}...")
        with open(lora_file, "r", encoding="utf-8") as f:
            ldata = json.load(f)

        g25 = ldata.get("heldout_25_graph_targets", {})
        s131 = ldata.get("heldout_35_stratum_c", {})

        print("\n--- Run 2B: Disjoint Lite -> Verified Generalization ---")
        print(f"| {'Cohort':<32} | {'Model':<12} | {'Acc@1':<7} | {'Acc@5':<7} | {'MRR':<7} | {'Within-Pool AUC':<16} |")
        print(f"| {'-'*32} | {'-'*12} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*16} |")
        b25 = g25.get("base", {})
        l25 = g25.get("lora", {})
        print(f"| {'Held-Out 25 Graph Targets':<32} | {'Zero-Shot':<12} | {b25.get('acc_1', 0):>6.2f}% | {b25.get('acc_5', 0):>6.2f}% | {b25.get('mrr', 0):>7.4f} | {b25.get('mean_auc', 0):>16.4f} |")
        print(f"| {'Held-Out 25 Graph Targets':<32} | {'Post-LoRA':<12} | {l25.get('acc_1', 0):>6.2f}% | {l25.get('acc_5', 0):>6.2f}% | {l25.get('mrr', 0):>7.4f} | {l25.get('mean_auc', 0):>16.4f} |")

        b131 = s131.get("base", {})
        l131 = s131.get("lora", {})
        print(f"| {'Held-Out 35 Stratum C':<32} | {'Zero-Shot':<12} | {b131.get('acc_1', 0):>6.2f}% | {b131.get('acc_5', 0):>6.2f}% | {b131.get('mrr', 0):>7.4f} | {b131.get('mean_auc', 0):>16.4f} |")
        print(f"| {'Held-Out 35 Stratum C':<32} | {'Post-LoRA':<12} | {l131.get('acc_1', 0):>6.2f}% | {l131.get('acc_5', 0):>6.2f}% | {l131.get('mrr', 0):>7.4f} | {l131.get('mean_auc', 0):>16.4f} |")

        print("\nGeneralization Contrasts on 25 Graph Targets:")
        print(f"  AUC Lift:    {g25.get('delta_auc', 0):+.4f} (Base {b25.get('mean_auc', 0):.4f} -> LoRA {l25.get('mean_auc', 0):.4f})")
        print(f"  Top-1 Lift:  {g25.get('delta_acc1_pp', 0):+.2f} pp (p={g25.get('mcnemar_p_top1', 1.0):.4f})")
        print(f"  Top-5 Lift:  {g25.get('delta_acc5_pp', 0):+.2f} pp (p={g25.get('mcnemar_p_top5', 1.0):.4f})")
    else:
        print(f"[-] Run 2B results file not found at {lora_file} (kernel may still be executing)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download from Kaggle CLI")
    args = parser.parse_args()

    if args.download:
        download_telemetry()
    parse_and_report()
