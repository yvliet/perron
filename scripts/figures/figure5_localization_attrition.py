#!/usr/bin/env python3
"""
scripts/figures/figure5_localization_attrition.py - Figure 5: Candidate Localization Pipeline Attrition.

Generates:
  - paper/figures/localization_attrition.{pdf, png}

Two-panel scientific chart on SWE-bench Verified (N=344 callable tasks):
  (a) Stepwise Candidate Localization Pipeline Retention across Stages S0 -> S3.
  (b) The 3.68x Attrition Bottleneck Ratio contrasting Upstream Boundary Loss (41.6 pp net)
      against Downstream Compaction Loss (11.3 pp).

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
    COLOR_RED,
    COLOR_INK,
    COLOR_MUTED,
    COLOR_SLATE,
    FIGURES_DIR,
)


def generate_figure5(out_dir: Path = FIGURES_DIR):
    """Generate Figure 5: Candidate Localization Pipeline Attrition (2-Panel Scientific Chart)."""
    apply_serif_style()
    print("Generating Figure 5: Candidate Localization Pipeline Attrition...")

    fig = plt.figure(figsize=(7.4, 3.2), dpi=300)

    # Clean gridspec layout with generous gutters
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2.1, 1.0], wspace=0.32,
        left=0.08, right=0.94, top=0.86, bottom=0.22
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # -------------------------------------------------------------
    # Subplot (a): Stepwise Pipeline Retention
    # -------------------------------------------------------------
    stages = [
        "Stage 0\nBenchmark\n($N=344$)",
        "Stage 1\nSeed Ceiling\n($N=176$)",
        "Stage 2\nGraph Pool\n($N=201$)",
        "Stage 3\nTop-5 Scored\n($N=162$)"
    ]
    x = np.array([0, 1, 2, 3])
    pcts = [100.0, 51.16, 58.43, 47.09]
    counts = [344, 176, 201, 162]

    # Clean publication palette
    bar_colors = ["#E2E8F0", "#BAE6FD", "#BBF7D0", "#93C5FD"]
    edge_colors = ["#475569", "#0284C7", "#16A34A", "#2563EB"]
    w = 0.38

    ax1.bar(x, pcts, width=w, color=bar_colors, edgecolor=edge_colors, linewidth=1.1, zorder=3)

    # Exact milestone retention labels directly above bars
    for i in range(len(x)):
        ax1.text(x[i], pcts[i] + 1.8, f"{pcts[i]:.1f}%\n({counts[i]})",
                 ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=COLOR_INK)

    # Horizontally aligned Transition Badges at y = 94.0 for collision-free headroom
    badge_y = 94.0

    # Delta 1: S0 -> S1 (-48.8 pp, 168 tasks missing from seeds)
    ax1.text(0.50, badge_y, r"$\mathbf{-48.8\ pp}$" + "\n(168 missing)",
             ha="center", va="center", fontsize=6.2, color="#B91C1C",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.7), zorder=4)
    ax1.annotate("", xy=(0.50, 77.0), xytext=(0.50, 85.0),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8), zorder=3)

    # Delta 2: S1 -> S2 (+7.3 pp, +25 tasks recovered via 1-hop AST calls)
    ax1.text(1.50, badge_y, r"$\mathbf{+7.3\ pp}$" + "\n(+25 recovered)",
             ha="center", va="center", fontsize=6.2, color="#15803D",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#F0FDF4", edgecolor="#16A34A", lw=0.7), zorder=4)
    ax1.annotate("", xy=(1.50, 85.0), xytext=(1.50, 77.0),
                 arrowprops=dict(arrowstyle="->", color="#16A34A", lw=0.8), zorder=3)

    # Delta 3: S2 -> S3 (-11.3 pp, 39 tasks displaced from top-5)
    ax1.text(2.50, badge_y, r"$\mathbf{-11.3\ pp}$" + "\n(39 displaced)",
             ha="center", va="center", fontsize=6.2, color="#B91C1C",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#DC2626", lw=0.7), zorder=4)
    ax1.annotate("", xy=(2.50, 77.0), xytext=(2.50, 85.0),
                 arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8), zorder=3)

    ax1.set_title("(a) Stepwise Candidate Localization Retention", fontsize=8.5, fontweight="bold", pad=8)
    ax1.set_ylabel("Defect Retention Recall (%)", fontsize=7.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=6.8)
    ax1.set_ylim(0, 126)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.22, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # -------------------------------------------------------------
    # Subplot (b): Bottleneck Ratio Contrast
    # -------------------------------------------------------------
    cats = [
        "Upstream\nBoundary Loss\n($S_0 \\to S_2$)",
        "Downstream\nCompaction Loss\n($S_2 \\to S_3$)"
    ]
    losses = [41.6, 11.3]
    loss_counts = [143, 39]
    bx = np.arange(len(cats))

    ax2.bar(bx, losses, width=0.44, color=["#FECACA", "#FED7AA"], edgecolor=["#DC2626", "#EA580C"], lw=1.1, zorder=3)

    for i in range(len(bx)):
        ax2.text(bx[i], losses[i] + 1.2, f"{losses[i]:.1f} pp\n({loss_counts[i]} tasks)",
                 ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=COLOR_INK)

    # Ratio bracket positioned safely above the highest label (y=52.5)
    bracket_y = 52.5
    ax2.plot([0, 0, 1, 1], [bracket_y - 2.5, bracket_y, bracket_y, bracket_y - 2.5], color=COLOR_INK, lw=1.0)
    ax2.text(0.5, bracket_y + 1.5, r"$\mathbf{3.68\times\ Bottleneck\ Ratio}$",
             ha="center", va="bottom", fontsize=7.2, fontweight="bold", color=COLOR_INK)

    ax2.set_title("(b) Attrition Bottleneck Ratio", fontsize=8.5, fontweight="bold", pad=8)
    ax2.set_ylabel("Attrition Loss (Percentage Points)", fontsize=7.8)
    ax2.set_xticks(bx)
    ax2.set_xticklabels(cats, fontsize=6.8)
    ax2.set_ylim(0, 64)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.22, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    save_figure(fig, "localization_attrition", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure5()
