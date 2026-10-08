#!/usr/bin/env python3
"""
scripts/generate_figures.py - Canonical Unified Figure Generator for Perron.

Generates the active publication-grade figures for the manuscript:
    - Figure 1: Architecture & Structural Provenance Pipeline (perron_provenance_pipeline.{pdf, png})
    - Figure 2: Candidate Pool-Size Truncation Curves (truncation_curves.{pdf, png})
    - Figure 3: Break-Even Discrimination Bound & Lexical Bottleneck (breakeven_discrimination_bound.{pdf, png})
    - Figure 4: Causal Provenance Forest Plot & Effect Sizes (causal_provenance_forest.{pdf, png})
    - Figure 5: Four-Stage Bug Localization Loss Funnel (loss_funnel.{pdf, png})
    - Figure 6: Continuous Dose-Response Surface & Damping Sensitivity (neff_dose_response.{pdf, png})

Usage:
    python scripts/generate_figures.py             # Generates all active figures (Figures 1-6)
    python scripts/generate_figures.py --fig 1     # Generates Figure 1
    python scripts/generate_figures.py --fig 2     # Generates Figure 2
    python scripts/generate_figures.py --fig 3     # Generates Figure 3
    python scripts/generate_figures.py --fig 4     # Generates Figure 4
    python scripts/generate_figures.py --fig 5     # Generates Figure 5
    python scripts/generate_figures.py --fig 6     # Generates Figure 6
    python scripts/generate_figures.py --list      # Lists all figures and current disk statuses

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import sys
import argparse
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures import (
    FIGURES_REGISTRY,
    FIGURES_DIR,
    generate_all_figures,
)


def list_figures():
    """Print registered figures and their current disk artifact statuses."""
    print("=" * 75)
    print("PERRON ACTIVE PUBLICATION FIGURES REGISTRY")
    print("=" * 75)
    for fig_id, meta in sorted(FIGURES_REGISTRY.items()):
        print(f"\n[{fig_id}] {meta['name']}")
        print(f"    Description: {meta['description']}")
        print("    Artifacts:")
        for out in meta["outputs"]:
            out_path = FIGURES_DIR / out
            if out_path.exists():
                size_kb = out_path.stat().st_size / 1024
                print(f"      - [EXISTS]  {out:<38} ({size_kb:>7.1f} KB)")
            else:
                print(f"      - [MISSING] {out:<38}")
    print("\n" + "=" * 75)


def run_figure_generation(fig_id=None):
    """Execute figure generation for a single figure or all active figures."""
    figures_to_run = [fig_id] if fig_id in FIGURES_REGISTRY else sorted(FIGURES_REGISTRY.keys())

    print("=" * 75)
    if fig_id in FIGURES_REGISTRY:
        print(f"Generating Figure {fig_id}: {FIGURES_REGISTRY[fig_id]['name']}")
    else:
        print("Generating All Active Publication Figures for Perron (Figures 1-6)")
    print("=" * 75)

    for fid in figures_to_run:
        meta = FIGURES_REGISTRY[fid]
        print(f"\n--> Running [{fid}/6] {meta['name']}...")
        meta["generator"](out_dir=FIGURES_DIR)

    print("\n" + "=" * 75)
    print("VERIFYING GENERATED ARTIFACTS IN paper/figures/:")
    all_ok = True
    for fid in figures_to_run:
        for out in FIGURES_REGISTRY[fid]["outputs"]:
            out_path = FIGURES_DIR / out
            if out_path.exists() and out_path.stat().st_size > 0:
                size_kb = out_path.stat().st_size / 1024
                print(f"  [OK]   {out:<38} ({size_kb:>7.1f} KB)")
            else:
                print(f"  [FAIL] {out:<38} (MISSING OR EMPTY)")
                all_ok = False
    print("=" * 75)
    return 0 if all_ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Canonical Unified Figure Generator for Perron Manuscript Figures."
    )
    parser.add_argument(
        "--fig", "--figure", dest="figure", type=str, default="all",
        help="Figure number to generate (1, 2, 3, 4, 5, 6, or 'all')",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered figures and their current disk status",
    )
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
                print(f"[ERROR] Invalid figure ID: {fid}. Must be between 1 and 6.")
                return 1
            return run_figure_generation(fid)
        except ValueError:
            print(f"[ERROR] Invalid argument: '{args.figure}'. Expected 1-6 or 'all'.")
            return 1


if __name__ == "__main__":
    sys.exit(main())
