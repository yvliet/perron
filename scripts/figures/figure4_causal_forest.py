#!/usr/bin/env python3
"""
scripts/figures/figure4_causal_forest.py - Figure 4: Causal Provenance Forest Plot & Effect Sizes.

Generates:
  - paper/figures/causal_provenance_forest.{pdf, png}

Panel (a): Forest plot of defect localization accuracy (Acc@1, Acc@5) with exact 95% Clopper-Pearson CIs.
Panel (b): Relative metric lifts and graph-only subgroup conversion on N=131 Stratum C.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.container import ErrorbarContainer

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.figures.common import (
    apply_serif_style,
    save_figure,
    COLOR_BLUE,
    COLOR_ORANGE,
    COLOR_GREEN,
    COLOR_PURPLE,
    COLOR_SLATE,
    COLOR_MUTED,
    COLOR_BORDER,
    COLOR_BG_CARD,
    FIGURES_DIR,
)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson binomial confidence interval in percent."""
    low = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    high = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return low * 100.0, high * 100.0


def generate_figure4(out_dir: Path = FIGURES_DIR):
    """Generate Figure 4: Causal Provenance Forest Plot & Relative Effect Sizes."""
    apply_serif_style()
    print("Generating Figure 4: Causal Provenance Forest Plot & Effect Sizes...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.8), dpi=300,
                                    gridspec_kw={'width_ratios': [1.10, 0.90]})

    # -------------------------------------------------------------
    # Panel A: Forest Plot of Acc@1 and Acc@5 with Clopper-Pearson CIs
    # -------------------------------------------------------------
    conditions = [
        ("C1: Baseline Isolated", 3, 20, 0.0778, 17.47, COLOR_SLATE),
        ("C2: Call-Path Provenance", 5, 21, 0.0899, 17.40, COLOR_ORANGE),
        ("C0: Unguided Thinking", 13, 24, 0.1358, 17.07, COLOR_GREEN),
        ("C3: Provenance + Thinking", 12, 23, 0.1270, 17.18, COLOR_BLUE),
    ]
    n_total = 131
    y_pos = [1.0, 2.0, 3.0, 4.0]

    for idx, (name, k1, k5, mrr, rank, col) in enumerate(conditions):
        y = y_pos[idx]
        p1 = k1 / n_total * 100.0
        p5 = k5 / n_total * 100.0
        ci1_low, ci1_high = clopper_pearson(k1, n_total)
        ci5_low, ci5_high = clopper_pearson(k5, n_total)

        # Plot Acc@1 (Circle) with percentage centered directly above marker
        ax1.errorbar(p1, y - 0.14, xerr=[[p1 - ci1_low], [ci1_high - p1]],
                     fmt='o', color=col, ecolor=col, elinewidth=1.4, capsize=3.0, capthick=1.0,
                     markersize=5.5, label='Acc@1 (Top-1)' if idx == 0 else '_nolegend_', zorder=5)
        ax1.text(p1, y - 0.03, f"{p1:.1f}%", fontsize=6.8, ha='center', va='bottom', color=col, fontweight='bold', zorder=6)

        # Plot Acc@5 (Square) with percentage centered directly above marker
        ax1.errorbar(p5, y + 0.14, xerr=[[p5 - ci5_low], [ci5_high - p5]],
                     fmt='s', color=col, ecolor=col, elinewidth=1.4, capsize=3.0, capthick=1.0,
                     markersize=5.5, label='Acc@5 (Top-5)' if idx == 0 else '_nolegend_', zorder=5)
        ax1.text(p5, y + 0.25, f"{p5:.1f}%", fontsize=6.8, ha='center', va='bottom', color=col, fontweight='bold', zorder=6)

    # Top-1 McNemar Annotation Bracket for Matched Lift (A4 Direct to C0 Thinking)
    dot_clearance = 0.08
    ax1.plot([4.58, 4.58, 9.92, 9.92],
             [1.0 - 0.14 - dot_clearance, 0.35, 0.35, 3.0 - 0.14 - dot_clearance],
             color=COLOR_GREEN, lw=1.1, ls=':', zorder=1)
    ax1.text(6.8, 0.18, r"$\mathbf{2.17\times\ Matched\ Lift}$" + "\n" + r"($4.58\% \to 9.92\%$)" + "\n" + r"($p = 0.0156^*$)",
             fontsize=6.2, color=COLOR_GREEN, fontweight='bold', ha='center', va='top', zorder=4,
             bbox=dict(boxstyle="round,pad=0.22", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GREEN, lw=0.8))

    # Factorial Parity Annotation (C3 vs C0) at symmetric bottom baseline
    ax1.text(23.2, 0.18,
             r"$\mathbf{Factorial\ Contrast:}$" + "\n" +
             r"$C_3\ \text{vs.}\ C_0\ \text{Exact}\ p = 1.0000$" + "\n" +
             r"$\Delta_{\text{interaction}} = -2.29\text{ pp\ (Parity)}$",
             fontsize=6.0, color=COLOR_BLUE, ha='center', va='top', zorder=4,
             bbox=dict(boxstyle="round,pad=0.22", facecolor='#F0F9FF', edgecolor=COLOR_BLUE, lw=0.8))

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([c[0] for c in conditions], fontsize=7.6)
    ax1.set_xlim(-0.5, 32.0)
    ax1.set_ylim(-0.55, 4.60)
    ax1.set_xlabel("Defect Localization Accuracy (%)")
    ax1.set_title(r"$\mathbf{(a)}$ Stratum C Factorial Localization ($N=131$)", loc='center', fontsize=9.0, pad=32)

    leg = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True,
                     framealpha=0.92, facecolor='white', edgecolor=COLOR_BORDER, fontsize=7.2,
                     handlelength=3.4, handletextpad=0.6,
                     handler_map={ErrorbarContainer: HandlerErrorbar(xerr_size=1.2, marker_pad=0.3)})

    # -------------------------------------------------------------
    # Panel B: Relative Metric Lift vs. Matched Graph Baseline (A4)
    # -------------------------------------------------------------
    metrics_data = [
        ("Top-1 Accuracy\n(Acc@1)", 116.7, 100.0, r"+116.7% (p=0.0156*)", r"+100.0% (p=0.0312*)"),
        ("MRR@50\nScore", 45.4, 36.0, r"+45.4% (0.1358)", r"+36.0% (0.1270)"),
        ("Top-5 Accuracy\n(Acc@5)", 9.1, 4.5, r"+9.1% (p=0.688)", r"+4.5% (p=1.000)"),
    ]

    y_bar = np.arange(len(metrics_data))
    bar_height = 0.30

    bars_c0 = ax2.barh(y_bar + 0.16, [m[1] for m in metrics_data], height=bar_height,
                       color=COLOR_GREEN, alpha=0.85, edgecolor=COLOR_BORDER, lw=0.8,
                       label=r'C0: Unguided Thinking')
    bars_c3 = ax2.barh(y_bar - 0.16, [m[2] for m in metrics_data], height=bar_height,
                       color=COLOR_BLUE, alpha=0.85, edgecolor=COLOR_BORDER, lw=0.8,
                       label=r'C3: Prov. + Thinking')

    for idx, (lbl, v0, v3, a0, a3) in enumerate(metrics_data):
        ax2.text(v0 + 4.0, idx + 0.16, a0, fontsize=6.2, color=COLOR_GREEN, va='center', fontweight='bold')
        ax2.text(v3 + 4.0, idx - 0.16, a3, fontsize=6.2, color=COLOR_BLUE, va='center', fontweight='bold')

    ax2.set_yticks(y_bar)
    ax2.set_yticklabels([m[0] for m in metrics_data], fontsize=7.2)
    ax2.set_xlim(0, 160)
    ax2.set_xticks([0, 40, 80, 120, 160])
    ax2.set_ylim(-0.60, 2.85)
    ax2.set_xlabel("Relative Lift vs. Matched Baseline A4 (%)")
    ax2.set_title(r"$\mathbf{(b)}$ Relative Effect Sizes vs. Matched $A_4$", loc='center', fontsize=9.0, pad=32)

    leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True,
                      framealpha=0.92, facecolor='white', edgecolor=COLOR_BORDER, fontsize=6.8)

    # Clean despine on both panels (zero top/right borders, identical height L-frames with gap to legends)
    for ax in (ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(COLOR_SLATE)
        ax.spines['bottom'].set_color(COLOR_SLATE)

    plt.tight_layout(w_pad=2.2)
    save_figure(fig, "causal_provenance_forest", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure4()
