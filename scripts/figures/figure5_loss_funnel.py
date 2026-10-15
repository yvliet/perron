#!/usr/bin/env python3
"""
scripts/figures/figure5_loss_funnel.py - Figure 5: Four-Stage Bug Localization Loss Funnel.

Generates:
  - paper/figures/loss_funnel.{pdf, png}

Four-stage bug localization loss funnel on SWE-bench Verified (N=344 callable tasks).
Decomposes boundary loss before scoring (41.6 pp) and compaction loss from candidate
pool to top-5 (11.3 pp), establishing the 3.68x Bottleneck Ratio.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

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
    COLOR_BORDER,
    COLOR_BG_CARD,
    FIGURES_DIR,
)


def T(ax, x, y, s, size=6.5, color=COLOR_INK, ha="center", va="center", weight="normal", style="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=z, **kw)


def bezier_band(x0, lo0, hi0, x1, lo1, hi1, frac=0.48):
    dx = (x1 - x0) * frac
    verts = [
        (x0, hi0), (x0 + dx, hi0), (x1 - dx, hi1), (x1, hi1),
        (x1, lo1), (x1 - dx, lo1), (x0 + dx, lo0), (x0, lo0),
        (x0, hi0)
    ]
    codes = [
        MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY
    ]
    return MPath(verts, codes)


def generate_figure5(out_dir: Path = FIGURES_DIR):
    """Generate Figure 5: Four-Stage Bug Localization Loss Funnel."""
    apply_serif_style()
    print("Generating Figure 5: Four-Stage Bug Localization Loss Funnel...")

    W, H = 720, 330
    fig = plt.figure(figsize=(7.2, 3.3), facecolor="white", dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    N = 344
    s = 125.0 / N
    b = 110.0
    pw = 14.0
    xs = [75, 255, 435, 615]
    h0, h1, h2, h3 = 344 * s, 176 * s, 201 * s, 162 * s
    lost1, infl, lost2 = 168 * s, 25 * s, 39 * s

    rx = lambda i: xs[i] + pw / 2
    lx = lambda i: xs[i] - pw / 2

    # 1. Main Throughput Ribbons (Pure Vector Pastel Blue with Clean Border)
    p01 = bezier_band(rx(0), b, b + h1, lx(1), b, b + h1)
    ax.add_patch(PathPatch(p01, facecolor="#E0F2FE", edgecolor=COLOR_BLUE, lw=1.0, alpha=0.9, zorder=2))

    p12 = bezier_band(rx(1), b, b + h1, lx(2), b + infl, b + h2)
    ax.add_patch(PathPatch(p12, facecolor="#E0F2FE", edgecolor=COLOR_BLUE, lw=1.0, alpha=0.9, zorder=2))

    p23 = bezier_band(rx(2), b + lost2, b + h2, lx(3), b + lost2, b + h2)
    ax.add_patch(PathPatch(p23, facecolor="#E0F2FE", edgecolor=COLOR_BLUE, lw=1.0, alpha=0.9, zorder=2))

    # 2. Branching Flow Ribbons
    # Lexical Boundary Loss (Stage 0 -> Diverges upward): 168 tasks
    p_lost1 = bezier_band(rx(0), b + h1, b + h0, 315, b + h1 + 16, b + h0 + 16)
    ax.add_patch(PathPatch(p_lost1, facecolor="#FEE2E2", edgecolor=COLOR_RED, lw=1.0, linestyle="--", alpha=0.85, zorder=2))

    # Graph 1-Hop Expansion (Recovers into Stage 2 from below): +25 tasks
    p_gain = bezier_band(280, b - 38, b - 38 + infl, lx(2), b, b + infl)
    ax.add_patch(PathPatch(p_gain, facecolor="#D1FAE5", edgecolor=COLOR_GREEN, lw=1.0, linestyle="-.", alpha=0.9, zorder=2))

    # Compaction Loss (Stage 2 -> Diverges downward): 39 tasks
    p_lost2 = bezier_band(rx(2), b, b + lost2, 545, b - 38, b - 38 + lost2)
    ax.add_patch(PathPatch(p_lost2, facecolor="#FEE2E2", edgecolor=COLOR_RED, lw=1.0, linestyle="--", alpha=0.85, zorder=2))

    # 3. Stage Pillars & Header Labels
    stages = [
        ("STAGE 0", "Benchmark Set", "344 / 344 tasks", "100.0%", COLOR_INK, (b, b + h0)),
        ("STAGE 1", "Seed Pool Ceiling", "176 / 344 tasks", "51.2%", COLOR_BLUE, (b, b + h1)),
        ("STAGE 2", "Expanded Graph Pool", "201 / 344 tasks", "58.4%", COLOR_GREEN, (b, b + h2)),
        ("STAGE 3", "Top-5 Scored Accuracy", "162 / 344 tasks", "47.1%", COLOR_INK, (b + lost2, b + h2)),
    ]

    for x, (tag, title, cnt, pct, pc, (lo, hi)) in zip(xs, stages):
        ax.add_patch(Rectangle((x - pw / 2, lo), pw, hi - lo, facecolor=COLOR_SLATE, edgecolor=COLOR_INK, lw=0.8, zorder=5))
        ax.plot([x, x], [hi + 4, 275], color=COLOR_BORDER, lw=0.8, ls=":", zorder=1)

        T(ax, x, 316, tag, size=5.5, color=COLOR_MUTED, weight="bold")
        T(ax, x, 304, title, size=7.2, weight="bold")
        T(ax, x, 292, cnt, size=6.2, color=COLOR_MUTED)
        T(ax, x, 280, pct, size=9.5, weight="bold", color=pc)

    # 4. Branch Flow Annotations (High-Contrast Callout Cards)
    ax.text(328, 236, "Boundary Loss 1: \u221248.8 pp\n168 tasks missing from seeds\n(Lexical mismatch)",
            fontsize=6.5, color=COLOR_RED, fontweight="bold", ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_RED, lw=0.8, alpha=0.95))

    ax.text(275, 55, "Net Graph Expansion: +7.3 pp\n+25 tasks recovered via 1-hop calls",
            fontsize=6.5, color=COLOR_GREEN, fontweight="bold", ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_GREEN, lw=0.8, alpha=0.95))

    ax.text(545, 62, "Compaction Loss: \u221211.3 pp\n39 pool tasks displaced to rank > 5\nby distractors",
            fontsize=6.5, color=COLOR_RED, fontweight="bold", ha="left", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLOR_RED, lw=0.8, alpha=0.95))

    # 5. Key Finding Banner at Bottom
    ax.plot([50, 670], [36, 36], color=COLOR_BORDER, lw=0.8, zorder=1)
    ax.text(360, 20,
            r"$\mathbf{Key\ Finding:}\ \mathrm{Upstream\ Boundary\ Loss\ (41.6\ pp\ net)\ exceeds\ Downstream\ Compaction\ Loss\ (11.3\ pp)\ by}\ \mathbf{3.68\times}\ (\mathrm{Bottleneck\ Ratio}).$",
            fontsize=7.2, color=COLOR_SLATE, ha="center", va="center", zorder=6)

    save_figure(fig, "loss_funnel", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure5()
