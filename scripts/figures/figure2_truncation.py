#!/usr/bin/env python3
"""
scripts/figures/figure2_truncation.py - Figure 2: Candidate Pool-Size Truncation Curves.

Generates:
  - paper/figures/truncation_curves.{pdf, png}

Evaluates candidate pool oracle recall and Gemma 4 reranked Acc@5 across pool
cutoffs M in [10, 15, 20, 30, 50] on SWE-bench Verified (N=344).

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.common import (
    apply_serif_style,
    save_figure,
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_SLATE,
    COLOR_MUTED,
    COLOR_BORDER,
    COLOR_BG_CARD,
    COLOR_RED,
    FIGURES_DIR,
)


def generate_figure2(out_dir: Path = FIGURES_DIR):
    """Generate Figure 2: Pointwise Candidate Pool-Size Truncation Curves."""
    apply_serif_style()
    print("Generating Figure 2: Pointwise Candidate Pool-Size Truncation Curves...")

    M_values = np.array([10, 15, 20, 30, 50])

    # Empirical data from SWE-bench Verified (N=344)
    # Arm A4 (Graph 1-Hop)
    a4_recall = np.array([51.16, 54.94, 55.52, 57.56, 58.43])
    a4_acc5 = np.array([46.22, 46.80, 47.09, 46.51, 47.09])

    # Arm A2 (Depth BM25)
    a2_recall = np.array([51.16, 54.07, 56.10, 57.56, 58.43])
    a2_acc5 = np.array([46.22, 45.93, 45.93, 45.06, 45.06])

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)

    # Plot Recall Curves (Top Tier)
    ax.plot(M_values, a4_recall, marker="o", color=COLOR_GREEN, lw=1.6, markersize=5.5,
            label="Arm A4: Graph 1-Hop Pool Recall (Oracle Ceiling)", zorder=4)
    ax.plot(M_values, a2_recall, marker="s", color=COLOR_BLUE, lw=1.4, ls="--", markersize=5.0,
            label="Arm A2: Depth BM25 Pool Recall (Oracle Ceiling)", zorder=3)

    # Plot Acc@5 Curves (Bottom Tier)
    ax.plot(M_values, a4_acc5, marker="^", color=COLOR_GREEN, lw=1.6, markersize=5.5,
            label=r"Arm A4: Graph 1-Hop Acc@5 ($\mathrm{Gemma\ 4}$ Reranked)", zorder=4)
    ax.plot(M_values, a2_acc5, marker="v", color=COLOR_BLUE, lw=1.4, ls="--", markersize=5.0,
            label=r"Arm A2: Depth BM25 Acc@5 ($\mathrm{Gemma\ 4}$ Reranked)", zorder=3)

    # Annotations & Callout Boxes
    # 1. Recall Ceiling Parity at M=50
    ax.scatter([50], [58.43], color=COLOR_SLATE, s=40, zorder=5)
    ax.text(49.5, 59.8, "Recall Ceiling Parity:\nBoth reach 58.4% (201/344 tasks)\n(18 vs 18 discordant pairs)",
            fontsize=6.8, ha="right", va="bottom", color=COLOR_SLATE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=COLOR_BG_CARD, edgecolor=COLOR_BORDER, lw=0.7))

    # 2. A4 Acc@5 Plateau
    ax.text(21.0, 48.0, r"$\mathbf{A4\ Plateau:}\ 47.1\%$" + "\n(+3 tasks over seeds, +0.87 pp, $p=0.549$)",
            fontsize=6.8, ha="left", va="bottom", color=COLOR_GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GREEN, lw=0.7))

    # 3. A2 Acc@5 Drop (Distractor Accumulation)
    ax.text(32.0, 43.6, r"$\mathbf{A2\ Drop:}\ 45.1\%$" + "\n(-4 tasks, -1.16 pp, $p=0.455$)\nDistractor accumulation",
            fontsize=6.8, ha="left", va="top", color=COLOR_BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=COLOR_BG_CARD, edgecolor=COLOR_BLUE, lw=0.7))

    ax.set_xticks(M_values)
    ax.set_xticklabels([str(m) for m in M_values], fontsize=8.0)
    ax.set_xlim(8, 52)
    ax.set_ylim(41.5, 65.5)

    ax.set_xlabel("Candidate Pool Truncation Cutoff $M$", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Accuracy / Recall on $N=344$ Tasks (%)", fontsize=8.5, fontweight="bold")
    ax.set_title(r"$\mathbf{Pointwise\ Candidate\ Pool\text{-}Size\ Truncation\ Curves\ (N=344)}$",
                 fontsize=9.2, pad=12)

    ax.legend(loc="upper left", frameon=True, framealpha=0.92, facecolor="white",
              edgecolor=COLOR_BORDER, fontsize=7.0, ncol=1)
    ax.grid(True, linestyle=":", alpha=0.5, color=COLOR_BORDER)

    plt.tight_layout()
    save_figure(fig, "truncation_curves", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure2()
