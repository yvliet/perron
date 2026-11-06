#!/usr/bin/env python3
"""
scripts/figures/figure5_loss_funnel.py - Figure 5: Four-Stage Bug Localization Loss Funnel.

Generates:
  - paper/figures/loss_funnel.{pdf, png}

Two-panel scientific chart on SWE-bench Verified (N=344 callable tasks):
  (a) Stepwise Defect Localization Pipeline Retention across Stages S0 -> S3.
  (b) The 3.68x Loss Bottleneck Ratio contrasting Upstream Boundary Loss (41.6 pp net)
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
    """Generate Figure 5: Four-Stage Bug Localization Loss Funnel (2-Panel Scientific Chart)."""
    apply_serif_style()
    print("Generating Figure 5: Four-Stage Bug Localization Loss Funnel...")

    fig = plt.figure(figsize=(7.2, 3.2), dpi=300)

    # Clean gridspec layout with generous gutters
    gs = fig.add_gridspec(
        1, 2, width_ratios=[2.2, 1.0], wspace=0.38,
        left=0.08, right=0.92, top=0.88, bottom=0.18
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ----------------------------------------------------
    # Subplot (a): Stepwise Pipeline Retention & Attrition
    # ----------------------------------------------------
    stages = [
        "Stage 0\nBenchmark\n($N=344$)",
        "Stage 1\nSeed Ceiling\n($N=176$)",
        "Stage 2\nGraph Pool\n($N=201$)",
        "Stage 3\nTop-5 Scored\n($N=162$)"
    ]
    x = np.arange(len(stages))
    pcts = [100.0, 51.16, 58.43, 47.09]
    counts = [344, 176, 201, 162]

    # Academic Okabe-Ito inspired palette
    bar_colors = ["#E2E8F0", "#BAE6FD", "#BBF7D0", "#93C5FD"]
    edge_colors = [COLOR_SLATE, COLOR_BLUE, COLOR_GREEN, COLOR_BLUE]

    ax1.bar(x, pcts, width=0.44, color=bar_colors, edgecolor=edge_colors, linewidth=1.1, zorder=3)

    # Stepwise dashed connection lines
    for i in range(len(x) - 1):
        ax1.plot([x[i], x[i+1]], [pcts[i], pcts[i]], color=COLOR_MUTED, linestyle=":", linewidth=0.8, zorder=2)
        ax1.plot([x[i+1], x[i+1]], [pcts[i], pcts[i+1]], color=COLOR_MUTED, linestyle=":", linewidth=0.8, zorder=2)

    # Milestone values above bars
    for i in range(len(x)):
        ax1.text(x[i], pcts[i] + 3.0, f"{pcts[i]:.1f}%\n({counts[i]})",
                 ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COLOR_INK)

    # Attrition / Expansion Delta Annotations (positioned to eliminate collision)
    ax1.annotate(
        r"$\mathbf{-48.8\ pp}$" + "\n(168 missing)",
        xy=(0.58, 75.0), xytext=(0.58, 88.0),
        ha="center", va="center", fontsize=6.5, color=COLOR_RED,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FEF2F2", edgecolor=COLOR_RED, lw=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.7)
    )

    ax1.annotate(
        r"$\mathbf{+7.3\ pp}$" + "\n(+25 recovered)",
        xy=(1.38, 54.8), xytext=(1.32, 78.0),
        ha="center", va="center", fontsize=6.5, color=COLOR_GREEN,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F0FDF4", edgecolor=COLOR_GREEN, lw=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=0.7)
    )

    ax1.annotate(
        r"$\mathbf{-11.3\ pp}$" + "\n(39 displaced)",
        xy=(2.62, 52.8), xytext=(2.62, 26.0),
        ha="center", va="center", fontsize=6.5, color=COLOR_RED,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#FEF2F2", edgecolor=COLOR_RED, lw=0.6, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=COLOR_RED, lw=0.7)
    )

    ax1.set_title("(a) Stepwise Defect Localization Pipeline Retention", fontsize=8.5, fontweight="bold", pad=8)
    ax1.set_ylabel("Defect Retention Recall (%)", fontsize=7.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=7.0)
    ax1.set_ylim(0, 122)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.25, zorder=0)

    # Secondary y-axis for raw task count
    ax1_sec = ax1.twinx()
    ax1_sec.set_ylim(0, 122 * 3.44)
    ax1_sec.set_ylabel("Task Count ($N=344$)", fontsize=7.8, color=COLOR_MUTED)
    ax1_sec.tick_params(axis="y", labelcolor=COLOR_MUTED, labelsize=6.8)
    ax1_sec.spines["top"].set_visible(False)

    # ----------------------------------------------------
    # Subplot (b): The 3.68x Bottleneck Ratio Contrast
    # ----------------------------------------------------
    cats = [
        "Upstream\nBoundary Loss\n($S_0 \\to S_2$)",
        "Downstream\nCompaction Loss\n($S_2 \\to S_3$)"
    ]
    losses = [41.6, 11.3]
    loss_counts = [143, 39]
    bx = np.arange(len(cats))

    ax2.bar(bx, losses, width=0.44, color=["#FCA5A5", "#FECACA"], edgecolor=COLOR_RED, linewidth=1.1, zorder=3)

    for i in range(len(bx)):
        ax2.text(bx[i], losses[i] + 1.2, f"{losses[i]:.1f} pp\n({loss_counts[i]} tasks)",
                 ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=COLOR_RED)

    # Ratio bracket and formal 3.68x annotation
    ax2.plot([0, 0, 1, 1], [46.0, 48.5, 48.5, 16.0], color=COLOR_INK, lw=0.9)
    ax2.text(0.5, 50.2, r"$\mathbf{3.68\times\ Ratio}$", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=COLOR_INK)

    ax2.set_title("(b) Loss Bottleneck Ratio", fontsize=8.5, fontweight="bold", pad=8)
    ax2.set_ylabel("Attrition Loss (Percentage Points)", fontsize=7.8)
    ax2.set_xticks(bx)
    ax2.set_xticklabels(cats, fontsize=6.8)
    ax2.set_ylim(0, 58)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.25, zorder=0)

    save_figure(fig, "loss_funnel", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure5()
