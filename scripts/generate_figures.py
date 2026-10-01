#!/usr/bin/env python3
"""
scripts/generate_figures.py - Canonical Unified Figure Generator for Perron.

Usage:
    python scripts/generate_figures.py             # Generates all figures (Figures 1-5)
    python scripts/generate_figures.py --fig 1     # Generates Figure 1 (Perron Pipeline)
    python scripts/generate_figures.py --fig 2     # Generates Figure 2 (Comparative Interfaces)
    python scripts/generate_figures.py --fig 3     # Generates Figure 3 (Action Dynamics & Failures)
    python scripts/generate_figures.py --fig 4     # Generates Figure 4 (Pass@k Scaling)
    python scripts/generate_figures.py --fig 5     # Generates Figure 5 (Statistical Evaluation)
    python scripts/generate_figures.py --list      # Lists all figures and output statuses
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FIGURES_REGISTRY = {
    1: {
        "name": "Figure 1: Pipeline Overview Architecture",
        "description": "Agentless-parity macro pipeline diagram with borderless pastel containers",
        "outputs": ["perron_pipeline.pdf", "perron_pipeline.png"],
        "fn": "generate_figure1_pipeline",
    },
    2: {
        "name": "Figure 2: Comparative Code Editing Interfaces",
        "description": "SWE-agent-parity 3-column comparative interface card diagram",
        "outputs": ["comparative_edit_interfaces.pdf", "comparative_edit_interfaces.png"],
        "fn": "generate_figure2_comparative_interfaces",
    },
    3: {
        "name": "Figure 3: Action Dynamics & Benchmark Failure Modes",
        "description": "Stacked execution turn bar chart and failure mode distribution donut chart",
        "outputs": ["action_and_failure_distribution.pdf", "action_and_failure_distribution.png"],
        "fn": "generate_figure3_action_and_failure",
    },
    4: {
        "name": "Figure 4: Pass@k Sample Budget Scaling",
        "description": "SWE-bench Lite resolve rate scaling across sample budgets k in [1..6]",
        "outputs": ["pass_at_k_scaling.pdf", "pass_at_k_scaling.png"],
        "fn": "generate_figure4_pass_at_k",
    },
    5: {
        "name": "Figure 5: Rigorous Statistical Benchmark Evaluation",
        "description": "Two-panel Raincloud plot (HSI) and Dolan-More performance profile (MRR)",
        "outputs": ["benchmark_statistical_evaluation.pdf", "benchmark_statistical_evaluation.png"],
        "fn": "generate_figure5_statistical_evaluation",
    },
}

def list_figures():
    figures_dir = REPO_ROOT / "paper" / "figures"
    print("=" * 75)
    print("PERRON PUBLICATION FIGURES REGISTRY")
    print("=" * 75)
    for fig_id, meta in sorted(FIGURES_REGISTRY.items()):
        print(f"\n[{fig_id}] {meta['name']}")
        print(f"    Description: {meta['description']}")
        print("    Artifacts:")
        for out in meta["outputs"]:
            out_path = figures_dir / out
            if out_path.exists():
                size_kb = out_path.stat().st_size / 1024
                print(f"      - [EXISTS] {out} ({size_kb:.1f} KB)")
            else:
                print(f"      - [MISSING] {out}")
    print("\n" + "=" * 75)

def run_figure_generation(fig_id=None):
    from paper.generate_all_figures import (
        generate_figure1_pipeline,
        generate_figure2_comparative_interfaces,
        generate_figure3_action_and_failure,
        generate_figure4_pass_at_k,
        generate_figure5_statistical_evaluation,
    )
    
    fn_map = {
        1: generate_figure1_pipeline,
        2: generate_figure2_comparative_interfaces,
        3: generate_figure3_action_and_failure,
        4: generate_figure4_pass_at_k,
        5: generate_figure5_statistical_evaluation,
    }
    
    figures_to_run = [fig_id] if fig_id in fn_map else sorted(fn_map.keys())
    
    print("=" * 75)
    if fig_id in fn_map:
        print(f"Generating Figure {fig_id}: {FIGURES_REGISTRY[fig_id]['name']}")
    else:
        print("Generating All 5 Publication-Grade Figures for Perron")
    print("=" * 75)
    
    for fid in figures_to_run:
        meta = FIGURES_REGISTRY[fid]
        print(f"\n--> Running [{fid}/5] {meta['name']}...")
        fn_map[fid]()
        
    print("\n" + "=" * 75)
    print("VERIFYING GENERATED ARTIFACTS IN paper/figures/:")
    figures_dir = REPO_ROOT / "paper" / "figures"
    all_ok = True
    for fid in figures_to_run:
        for out in FIGURES_REGISTRY[fid]["outputs"]:
            out_path = figures_dir / out
            if out_path.exists() and out_path.stat().st_size > 0:
                size_kb = out_path.stat().st_size / 1024
                print(f"  [OK] {out:<40} ({size_kb:>7.1f} KB)")
            else:
                print(f"  [FAIL] {out:<40} (MISSING OR EMPTY)")
                all_ok = False
    print("=" * 75)
    return 0 if all_ok else 1

def main():
    parser = argparse.ArgumentParser(description="Canonical Unified Figure Generator for Perron.")
    parser.add_argument("--fig", "--figure", dest="figure", type=str, default="all",
                        help="Figure number to generate (1, 2, 3, 4, 5, or 'all')")
    parser.add_argument("--list", action="store_true",
                        help="List all registered figures and their current disk status")
    args = parser.parse_args()
    
    if args.list:
        list_figures()
        return 0
        
    if args.figure.lower() == "all":
        return run_figure_generation(None)
    else:
        try:
            fid = int(args.figure)
            if fid not in FIGURES_REGISTRY:
                print(f"[ERROR] Invalid figure ID: {fid}. Must be between 1 and 5.")
                return 1
            return run_figure_generation(fid)
        except ValueError:
            print(f"[ERROR] Invalid argument: {args.figure}. Expected 1-5 or 'all'.")
            return 1

if __name__ == "__main__":
    sys.exit(main())
