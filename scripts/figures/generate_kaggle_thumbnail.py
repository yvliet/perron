#!/usr/bin/env python3
"""
scripts/figures/generate_kaggle_thumbnail.py - Publication-Grade 2:1 Kaggle Thumbnail Banner.

Generates:
  - paper/figures/kaggle_thumbnail_2400x1200.png (Master 2400x1200 Ultra-HD 2:1 Banner)
  - paper/figures/kaggle_thumbnail_1120x560.png  (Retina 2x 1120x560 Banner)
  - paper/figures/kaggle_thumbnail_560x280.png   (Standard 1x 560x280 Card Preview)

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
from PIL import Image

os.environ.setdefault("SOURCE_DATE_EPOCH", "1790812800")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.figures.figure1_pipeline as f1


def generate_kaggle_thumbnails(out_dir=None):
    """
    Renders the architecture and structural provenance pipeline in a strictly 2:1 canvas
    with balanced margins and zero clipping across all tiers.
    """
    if out_dir is None:
        target_dir = REPO_ROOT / "paper" / "figures"
    else:
        target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    master_path = target_dir / "kaggle_thumbnail_2400x1200.png"
    retina_path = target_dir / "kaggle_thumbnail_1120x560.png"
    standard_path = target_dir / "kaggle_thumbnail_560x280.png"

    width_px, height_px = 2400, 1200
    dpi = 300
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])

    # 2:1 Aspect Ratio Canvas (880 x 440 coordinate units)
    # Content spans x in [14, 814], y in [20, 372].
    # Margin budget: 40 units horizontal padding, 43-45 units vertical padding.
    ax.set_xlim(-26, 854)
    ax.set_ylim(-25, 415)
    ax.axis("off")

    cap_t1 = lambda x, y, s: f1.T(ax, x, y, s, size=5.8, weight="bold", color=f1.INK)
    cap_t2 = lambda x, y, s: f1.T(ax, x, y, s, size=5.4, weight="bold", color=f1.INK)

    # =========================================================================
    # TIER 1: OFFLINE INDEXING (Top)
    # =========================================================================
    f1.T(ax, 34, 296, "Offline\nIndexing", size=8.8, weight="bold", color=f1.INK, linespacing=1.2)
    f1.draw_curly_brace(ax, x_spine=76, y_bot=225, y_top=368, w=6.5, r=12.0, color=f1.INK, lw=1.2)

    xb1 = 94
    cap_t1(xb1 + 56, 362, "Codebase AST")
    f1.doc_icon(ax, xb1 + 2, 309, 12, 16)
    f1.T(ax, xb1 + 18, 317, "django/db/models/query.py", size=4.4, mono=True, color=f1.MUTED, ha="left")
    code = [("def bulk_create(objs, bs):", f1.INK, "bold"), ("    ...", f1.MUTED, "normal"),
            ("    _batched_insert(objs, bs)", f1.MUTED, "normal"),
            ("def _batched_insert(objs, bs):", f1.INK, "bold"), ("    ...", f1.MUTED, "normal")]
    for i, (s, c, w) in enumerate(code):
        f1.T(ax, xb1 + 2, 298 - 9.5 * i, s, size=4.8, mono=True, color=c, weight=w, ha="left")
    f1.T(ax, xb1 + 56, 245, r"$|\mathcal{V}|\approx 10^{4}-10^{5}$ AST symbols", size=5.6, color=f1.MUTED)

    f1.stage_arrow(ax, (xb1 + 128, 287), (xb1 + 146, 287))

    xb2 = xb1 + 146
    cap_t1(xb2 + 65, 362, "Typed Edge Extraction")
    ax.add_patch(f1.Circle((xb2 + 10, 287), 7, fc=f1.INK, ec="none", zorder=5))
    f1.T(ax, xb2 + 10, 287, "AST", size=4.2, color="white", weight="bold")
    rel = [("Calls", r"$\mathcal{E}_{\mathrm{call}}$, $\omega=0.50$", "-", False),
           ("Inheritance", r"$\mathcal{E}_{\mathrm{inherit}}$, $\omega=0.25$", (0, (3, 1.6)), False),
           ("Imports", r"$\mathcal{E}_{\mathrm{import}}$, $\omega=0.15$", (0, (1, 1.6)), False),
           ("Reverse calls", r"$\mathcal{E}_{\mathrm{caller}}$, $\omega=0.10$", "-", True)]
    for ry, (name, math, ls, rev) in zip([320, 298, 276, 254], rel):
        ax.add_patch(f1.PathPatch(f1.Path([(xb2 + 17, 287), (xb2 + 28, 287), (xb2 + 28, ry), (xb2 + 40, ry)],
                                    [f1.Path.MOVETO, f1.Path.CURVE4, f1.Path.CURVE4, f1.Path.CURVE4]),
                               fc="none", ec=f1.MUTED, lw=0.6, zorder=3))
        a, b = ((xb2 + 54, ry), (xb2 + 42, ry)) if rev else ((xb2 + 42, ry), (xb2 + 54, ry))
        ax.add_patch(f1.FancyArrowPatch(
            a, b, arrowstyle="-|>,head_length=0.45,head_width=0.22",
            mutation_scale=4.2, lw=0.8, ec=f1.INK, fc=f1.INK, ls=ls, zorder=4, capstyle="round"
        ))
        f1.T(ax, xb2 + 59, ry + 4.2, name, size=5.6, weight="bold", ha="left")
        f1.T(ax, xb2 + 59, ry - 4.2, math, size=5.3, color=f1.MUTED, ha="left")
    f1.T(ax, xb2 + 65, 233, r"$W_{uv}=\sum_{r}\omega_{r}A_{uv}^{(r)}$", size=6.4)

    f1.stage_arrow(ax, (xb2 + 128, 287), (xb2 + 146, 287))

    xb3 = xb2 + 146
    cap_t1(xb3 + 70, 362, r"Directed Code Multigraph $\mathcal{G}$")
    f1.blob(ax, xb3 + 2, 228, 142, 118, "#EEF2F7")
    f1.draw_graph(ax, xb3 + 14, 242, 116, 90, "neutral")

    f1.stage_arrow(ax, (xb3 + 146, 287), (xb3 + 164, 287))

    xb4 = xb3 + 164
    cap_t1(xb4 + 78, 362, "Decoupled CSR Matrix Engine")
    mx, my, c = xb4 + 4, 269, 6
    nz = {(0, 1), (0, 2), (1, 4), (1, 5), (2, 3), (2, 5)}
    for i in range(6):
        for j in range(6):
            ax.add_patch(f1.Rectangle((mx + j * c, my + (5 - i) * c), c, c,
                                   fc=f1.INK if (i, j) in nz else "none", ec=f1.FAINT, lw=0.4, zorder=4))
    f1.T(ax, mx + 18, my + 44, r"sparse $W$", size=5.2, color=f1.MUTED)

    f1.stage_arrow(ax, (mx + 42, 287), (mx + 60, 287))

    ax0 = mx + 94
    rows = [(315, "indptr", [0, 2, 4, 6, 6, 6, 6], None),
            (295, "indices", [1, 2, 4, 5, 3, 5], None),
            (275, "data", [".5", ".5", ".5", ".5", ".25", ".5"],
             [f1.PALE, f1.PALE, f1.PALE, f1.PALE, "#F1F5F9", f1.PALE])]
    for y, nm, vals, fills in rows:
        f1.T(ax, ax0 - 4, y, nm, size=4.8, mono=True, color=f1.MUTED, ha="right")
        f1.cells(ax, ax0, y, vals, fills=fills)
    f1.T(ax, ax0 + 45, 329, r"$W$ stored as three flat arrays", size=5.2, color=f1.MUTED)

    shades = ["#F1F5F9", "#F1F5F9", "#E2E8F0", "#E2E8F0", "#94A3B8", "#64748B"]
    for i, fcol in enumerate(shades):
        ax.add_patch(f1.Rectangle((ax0 + 104, 319 - (i + 1) * 10.5), 12, 10.5, fc=fcol, ec=f1.MUTED, lw=0.5, zorder=4))
    f1.T(ax, ax0 + 110, 338, "degree prior", size=4.8, color=f1.MUTED)
    f1.T(ax, ax0 + 110, 329, r"$\pi_g$", size=6.6)
    f1.line(ax, [ax0 + 120, ax0 + 122, ax0 + 122, ax0 + 120], [256.5, 256.5, 277.5, 277.5], color=f1.MUTED, lw=0.5)
    f1.T(ax, ax0 + 125, 267, "hubs:\nhigh $\\pi_g$", size=4.8, color=f1.MUTED, ha="left", linespacing=1.15)

    f1.T(ax, xb4 + 78, 248, r"$\mathbf{p}_{t+1}=\beta\,\bar{T}^{\top}\mathbf{p}_{t}+(1-\beta)\,\mathbf{p}_{0} \quad (\beta=0.85)$", size=5.6)
    f1.T(ax, xb4 + 89, 233, r"$\leq$ 1.6 ms SpMV  |  $\leq$ 3 MB RAM  |  Zero-Copy mmap",
      size=5.6, weight="bold", color=f1.INK)

    # Middle dashed arrow
    ax.add_patch(f1.FancyArrowPatch(
        (496, 222), (496, 172),
        arrowstyle="-|>,head_length=0.50,head_width=0.25",
        mutation_scale=5.2, lw=0.8, ec=f1.MUTED, fc=f1.MUTED, ls=(0, (2.5, 2)), zorder=2, capstyle="round"
    ))
    f1.T(ax, 502, 197, r"re-used: $W$, $\pi_g$ (mmap)", size=5.0, style="italic", color=f1.MUTED, ha="left")

    # =========================================================================
    # TIER 2: ONLINE RETRIEVAL (Bottom)
    # =========================================================================
    f1.T(ax, 34, 103, "Online\nRetrieval", size=8.8, weight="bold", color=f1.INK, linespacing=1.2)
    f1.draw_curly_brace(ax, x_spine=76, y_bot=28, y_top=178, w=6.5, r=12.0, color=f1.INK, lw=1.2)

    xc1 = 94
    cap_t2(xc1 + 38, 166, "Bug Report Query")
    ax.add_patch(f1.Circle((xc1 + 10, 138), 4.6, fc="none", ec=f1.INK, lw=0.8, zorder=4))
    ax.add_patch(f1.Circle((xc1 + 10, 138), 1.4, fc=f1.INK, ec="none", zorder=4))
    f1.T(ax, xc1 + 20, 139, "django-11951", size=5.8, mono=True, weight="bold", ha="left")
    f1.T(ax, xc1 + 20, 129, "Un-Named Defect (Stratum C)", size=5.0, color=f1.MUTED, ha="left")
    f1.T(ax, xc1 + 4, 114, "Symptom (issue text)", size=4.8, style="italic", color=f1.MUTED, ha="left")
    f1.rich_line(ax, xc1 + 4, 102, [
        ("\u201c", dict(size=5.8)),
        ("bulk_create()", dict(size=5.0, mono=True, color=f1.BLUE, weight="bold")),
        (" fails", dict(size=5.8))], gap=2.0)
    f1.T(ax, xc1 + 4, 93, "with batch_size\u201d", size=5.8, ha="left")
    f1.T(ax, xc1 + 4, 78, "Hidden target (not in issue):", size=4.8, style="italic", color=f1.MUTED, ha="left")
    f1.T(ax, xc1 + 4, 69, "_batched_insert", size=5.4, mono=True, weight="bold", color=f1.GREEN, ha="left")
    f1.cross(ax, xc1 + 7, 58, s=2.2, color=f1.RED, lw=1.0)
    f1.T(ax, xc1 + 13, 58, "never mentioned", size=5.0, style="italic", color=f1.RED, ha="left")

    ax.add_patch(f1.FancyArrowPatch(
        (xc1 + 92, 101), (xc1 + 144, 130),
        arrowstyle="-|>,head_length=0.45,head_width=0.22",
        mutation_scale=4.5, lw=0.8, ec=f1.BLUE, fc=f1.BLUE, connectionstyle="arc3,rad=-0.15", zorder=4
    ))

    xc2 = xc1 + 146
    cap_t2(xc2 + 6, 166, "Seed Personalization")
    f1.T(ax, xc2 + 6, 150, r"teleport $\mathbf{p}_0$", size=5.6, color=f1.MUTED)

    for i in range(6):
        cy = 143 - (i + 1) * 10.5
        fc = f1.BLUE if i == 0 else "white"
        ax.add_patch(f1.Rectangle((xc2, cy), 12, 10.5, fc=fc, ec=f1.MUTED, lw=0.5, zorder=4))
        if i == 0:
            f1.T(ax, xc2 + 6, cy + 5.25, "1", size=5.0, mono=True, color="white", weight="bold")
        else:
            f1.T(ax, xc2 + 6, cy + 5.25, "0", size=4.4, mono=True, color=f1.MUTED)

    f1.T(ax, xc2 + 16, 137.75, "1.0", size=5.0, mono=True, color=f1.BLUE, weight="bold", ha="left")
    f1.T(ax, xc2 + 6, 68, r"$\mathbf{p}_0[\mathrm{seed}]=1.0$", size=5.2)
    f1.T(ax, xc2 + 6, 59, "sparse vector", size=4.8, style="italic", color=f1.MUTED)

    f1.stage_arrow(ax, (xc2 + 36, 101), (xc2 + 54, 101))

    xc3 = xc2 + 54
    cap_t2(xc3 + 89, 166, "Specificity-Damped Diffusion")
    f1.blob(ax, xc3 + 2, 44, 174, 112, "#EEF2F7")
    P = f1.draw_graph(ax, xc3 + 12, 62, 148, 78, "active")
    mx_, my_ = (P["bulk_create"][0] + P["_batched_insert"][0]) / 2, (P["bulk_create"][1] + P["_batched_insert"][1]) / 2
    f1.T(ax, mx_ - 6, my_ + 8, "calls", size=5.2, style="italic", color=f1.AMBER_TXT, weight="bold")
    f1.T(ax, xc3 + 89, 34, r"Hub Suppressed ($\gamma=0.70$, HSI@10 = 99.5%)", size=5.5, color=f1.RED, weight="bold")
    f1.T(ax, xc3 + 89, 24, r"specificity ratio  $\pi_q\,/\,(\pi_g+\epsilon)^{\gamma}$", size=5.4, color=f1.MUTED)

    f1.stage_arrow(ax, (xc3 + 182, 101), (xc3 + 200, 101))

    xc4 = xc3 + 200
    cap_t2(xc4 + 56, 166, "Structural Provenance Verbalizer")
    ax.add_patch(f1.Rectangle((xc4 + 4, 118.5), 1.8, 19, fc=f1.AMBER, ec="none", zorder=5))

    f1.T(ax, xc4 + 11, 132, "# Provenance: called_by_seed", size=4.8, mono=True, weight="bold", color=f1.AMBER_TXT, ha="left")
    f1.T(ax, xc4 + 11, 123, "#   (called by bulk_create)", size=4.8, mono=True, color=f1.AMBER_TXT, ha="left")
    f1.T(ax, xc4 + 11, 112, "def _batched_insert(objs,", size=4.8, mono=True, weight="bold", ha="left")
    f1.T(ax, xc4 + 11, 103, "        batch_size): ...", size=4.8, mono=True, weight="bold", ha="left")
    f1.T(ax, xc4 + 4, 85, "without provenance (baseline):", size=4.8, style="italic", color=f1.MUTED, ha="left")
    f1.T(ax, xc4 + 11, 76, "def _batched_insert(...)", size=4.8, mono=True, color="#94A3B8", ha="left")
    f1.T(ax, xc4 + 11, 67, "isolated: no graph context", size=4.8, style="italic", color=f1.RED, ha="left")

    f1.stage_arrow(ax, (xc4 + 128, 127), (xc4 + 146, 127))

    xc5 = xc4 + 146
    cap_t2(xc5 + 68, 166, "Local Gemma 4 Scorer")
    f1.gemma4_logo(ax, xc5 + 16, 127, R=8.5)
    f1.T(ax, xc5 + 16, 108.5, "Gemma 4", size=5.4, weight="normal", color="#3085FE", ha="center")
    f1.T(ax, xc5 + 16, 101.0, "(Local 2B / 4B)", size=4.6, color=f1.MUTED, ha="center")
    ax.add_patch(f1.Circle((xc5 + 32, 119), 1.6, fc="white", ec=f1.MUTED, lw=0.6, zorder=5))
    ax.add_patch(f1.Circle((xc5 + 37.5, 123.5), 2.3, fc="white", ec=f1.MUTED, lw=0.6, zorder=5))
    ax.add_patch(f1.Ellipse((xc5 + 106, 131), 134, 50, fc="white", ec=f1.MUTED, lw=0.7, zorder=4))
    f1.T(ax, xc5 + 106, 142.5, "<thought>", size=4.2, mono=True, color=f1.MUTED)
    f1.T(ax, xc5 + 106, 135.0, "bulk_create delegates batching", size=4.6)
    f1.T(ax, xc5 + 106, 127.0, "to _batched_insert. Bug localized.", size=4.6)
    f1.T(ax, xc5 + 106, 119.5, "</thought>", size=4.2, mono=True, color=f1.MUTED)
    f1.T(ax, xc5 + 4, 91, "Ranked output", size=4.8, style="italic", color=f1.MUTED, ha="left")
    for y, rk, nm, top in [(82, "1", "_batched_insert", True), (73, "2", "save", False),
                           (64, "3", "Model", False)]:
        f1.T(ax, xc5 + 4, y, rk, size=5.2, mono=True, color=f1.GREEN if top else "#94A3B8", ha="left")
        f1.T(ax, xc5 + 14, y, nm, size=5.2, mono=True, color=f1.GREEN if top else f1.MUTED,
          weight="bold" if top else "normal", ha="left")
    ax.add_patch(f1.Circle((xc5 + 88, 82), 3.4, fc=f1.GREEN, ec="none", zorder=6))
    f1.check(ax, xc5 + 88, 82, s=4.2, lw=0.9)
    f1.T(ax, xc5 + 95, 82, "Top-1", size=5.2, weight="bold", color=f1.GREEN, ha="left")

    f1.T(ax, xc5 + 75, 45, r"Acc@1: 4.6% $\rightarrow$ 9.9%  (p=0.0156)", size=5.6, weight="bold", color=f1.INK)
    f1.T(ax, xc5 + 75, 29, "Matched graph pool: +45.4% MRR", size=5.0, color=f1.MUTED)

    fig.savefig(str(master_path), dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"Rendered: {master_path} ({width_px}x{height_px})")

    # Downsample crisp anti-aliased Retina and standard card previews
    master_img = Image.open(master_path)
    retina_img = master_img.resize((1120, 560), Image.Resampling.LANCZOS)
    retina_img.save(str(retina_path), optimize=True)
    print(f"Rendered: {retina_path} (1120x560)")

    standard_img = master_img.resize((560, 280), Image.Resampling.LANCZOS)
    standard_img.save(str(standard_path), optimize=True)
    print(f"Rendered: {standard_path} (560x280)")

    return {
        "master_2400x1200": master_path,
        "retina_1120x560": retina_path,
        "standard_560x280": standard_path,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    generate_kaggle_thumbnails(out)
