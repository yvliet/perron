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
import matplotlib.colors as mcolors
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
    COLOR_FAINT,
    COLOR_PALE,
    FIGURES_DIR,
)


def T(ax, x, y, s, size=5.5, color=COLOR_INK, ha="center", va="center", weight="normal",
      style="normal", mono=False, z=6, **kw):
    if mono:
        kw["family"] = "monospace"
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=z, **kw)


def line(ax, xs, ys, color=COLOR_FAINT, lw=0.8, ls="-", z=3):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z)


def rich_line(ax, x, y, parts, ha="left", gap=4.5):
    """Draw a single line made of differently-styled text runs."""
    r = ax.figure.canvas.get_renderer()
    xr = ax.get_xlim()
    scale = (xr[1] - xr[0]) / ax.bbox.width
    objs, widths = [], []
    for s, kw in parts:
        kw = dict(kw)
        t = T(ax, 0, y, s, ha="left", va="baseline", **kw)
        widths.append(t.get_window_extent(r).width * scale)
        objs.append(t)
    total = sum(widths) + gap * (len(widths) - 1)
    cur = x if ha == "left" else (x - total / 2 if ha == "center" else x - total)
    for t, w in zip(objs, widths):
        t.set_x(cur)
        cur += w + gap


def band_path(x0, lo0, hi0, x1, lo1, hi1, frac=0.5):
    dx = (x1 - x0) * frac
    verts = [(x0, hi0), (x0 + dx, hi0), (x1 - dx, hi1), (x1, hi1),
             (x1, lo1), (x1 - dx, lo1), (x0 + dx, lo0), (x0, lo0), (x0, hi0)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    return MPath(verts, codes)


def solid_band(ax, path, color, alpha, z=2):
    ax.add_patch(PathPatch(path, fc=color, ec="none", alpha=alpha, lw=0, zorder=z))


def gradient_band(ax, path, color, a0, a1, z=2):
    clip = PathPatch(path, fc="none", ec="none", lw=0, transform=ax.transData)
    ax.add_patch(clip)
    bb = path.get_extents()
    n = 256
    arr = np.zeros((1, n, 4))
    arr[..., :3] = mcolors.to_rgb(color)
    arr[0, :, 3] = np.linspace(a0, a1, n)
    im = ax.imshow(arr, extent=[bb.x0, bb.x1, bb.y0, bb.y1], origin="lower", aspect="auto",
                   zorder=z, interpolation="bilinear")
    im.set_clip_path(clip)


def generate_figure5(out_dir: Path = FIGURES_DIR):
    """Generate Figure 5: Four-Stage Bug Localization Loss Funnel."""
    apply_serif_style()
    print("Generating Figure 5: Four-Stage Bug Localization Loss Funnel...")

    W, H = 720, 320
    fig = plt.figure(figsize=(7.2, 3.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    N = 344
    s = 130.0 / N
    b = 125.0
    pw = 12.0
    xs = [70, 255, 440, 625]
    h0, h1, h2, h3 = 344 * s, 176 * s, 201 * s, 162 * s
    lost1, infl, lost2 = 168 * s, 25 * s, 39 * s
    rx = lambda i: xs[i] + pw / 2
    lx = lambda i: xs[i] - pw / 2

    # Ribbons
    solid_band(ax, band_path(rx(0), b, b + h1, lx(1), b, b + h1), COLOR_BLUE, 0.30)
    solid_band(ax, band_path(rx(1), b, b + h1, lx(2), b + infl, b + h2), COLOR_BLUE, 0.30)
    solid_band(ax, band_path(rx(2), b + lost2, b + h2, lx(3), b + lost2, b + h2), COLOR_BLUE, 0.30)

    # Losses & gains
    gradient_band(ax, band_path(rx(0), b + h1, b + h0, 335, b + h1 + 16, b + h0 + 16), COLOR_RED, 0.80, 0.08)
    gradient_band(ax, band_path(285, b - 45, b - 45 + infl, lx(2), b, b + infl), COLOR_GREEN, 0.06, 0.85)
    gradient_band(ax, band_path(rx(2), b, b + lost2, 560, b - 45, b - 45 + lost2), COLOR_RED, 0.80, 0.08)

    # Pillars + stage headers
    stages = [
        ("STAGE 0", "Benchmark Set", "344 / 344 tasks", "100.0%", COLOR_INK, (b, b + h0)),
        ("STAGE 1", "Seed Pool Ceiling", "176 / 344 tasks", "51.2%", COLOR_BLUE, (b, b + h1)),
        ("STAGE 2", "Expanded Graph Pool", "201 / 344 tasks", "58.4%", COLOR_GREEN, (b, b + h2)),
        ("STAGE 3", "Top-5 Scored Accuracy", "162 / 344 tasks", "47.1%", COLOR_INK, (b + lost2, b + h2)),
    ]
    for x, (tag, title, cnt, pct, pc, (lo, hi)) in zip(xs, stages):
        ax.add_patch(Rectangle((x - pw / 2, lo), pw, hi - lo, fc=COLOR_INK, ec="none", zorder=5))
        line(ax, [x, x], [hi + 3, 273], color=COLOR_FAINT, lw=0.6, ls=(0, (1.4, 2.2)), z=1)
        T(ax, x, 310, tag, size=5.2, color=COLOR_MUTED, weight="bold")
        T(ax, x, 300, title, size=6.9, weight="bold")
        T(ax, x, 290, cnt, size=6.0, color=COLOR_MUTED)
        T(ax, x, 278, pct, size=9.0, weight="bold", color=pc)

    # Flow annotations
    T(ax, 341, 250, "Boundary Loss 1: \u221248.8 pp", size=6.5, weight="bold", color=COLOR_RED, ha="left")
    T(ax, 341, 241, "168 tasks missing from seeds", size=5.8, color=COLOR_MUTED, ha="left")
    T(ax, 341, 233, "(lexical mismatch)", size=5.8, style="italic", color=COLOR_MUTED, ha="left")

    T(ax, 285, 70, "Net Graph Expansion: +7.3 pp", size=6.9, weight="bold", color=COLOR_GREEN, ha="left")
    T(ax, 285, 61, "+25 tasks recovered via 1-hop calls", size=5.8, color=COLOR_MUTED, ha="left")

    T(ax, 568, 97, "Compaction Loss: \u221211.3 pp", size=6.9, weight="bold", color=COLOR_RED, ha="left")
    T(ax, 568, 88, "39 pool tasks displaced to rank > 5", size=5.8, color=COLOR_MUTED, ha="left")
    T(ax, 568, 80, "by distractors", size=5.8, style="italic", color=COLOR_MUTED, ha="left")

    # Key Finding Callout
    line(ax, [70, 650], [47, 47], color=COLOR_PALE, lw=0.6, z=1)
    rich_line(ax, 360, 33, [
        ("Key Finding:", dict(size=7.2, weight="bold")),
        ("Upstream Boundary Loss", dict(size=7.2, weight="bold", color=COLOR_RED)),
        ("(41.6 pp net) exceeds", dict(size=7.0)),
        ("Downstream Compaction Loss", dict(size=7.2, weight="bold", color=COLOR_RED)),
        ("(11.3 pp)", dict(size=7.0))], ha="center", gap=6.0)
    rich_line(ax, 360, 21, [
        ("by", dict(size=7.0)),
        ("3.68\u00d7", dict(size=8.4, weight="bold", color=COLOR_RED)),
        ("(Bottleneck Ratio).", dict(size=7.0)),
        ("Retrieval boundary is the primary bottleneck.", dict(size=7.2, weight="bold"))],
        ha="center", gap=6.0)

    save_figure(fig, "loss_funnel", out_dir=out_dir)


if __name__ == "__main__":
    generate_figure5()
