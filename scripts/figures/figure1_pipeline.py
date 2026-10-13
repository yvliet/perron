#!/usr/bin/env python3
"""
scripts/figures/figure1_pipeline.py - Figure 1: Architecture & Structural Provenance Pipeline.
Generates:
  - paper/figures/perron_provenance_pipeline.{pdf, png}
  - scratch/perron_pipeline_clean.{pdf, png}

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import os
import sys
import time
import numpy as np
import matplotlib

os.environ.setdefault("SOURCE_DATE_EPOCH", "1790812800")
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,
                                PathPatch, Polygon, Rectangle)
from matplotlib.path import Path

# --------------------------------------------------------------------------- #
# Global Typography & Palette: Clean Modern Sans-Serif
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
    "font.monospace": ["DejaVu Sans Mono", "Courier New"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "axes.unicode_minus": True,
})

INK = "#1E293B"        # dark slate / black
MUTED = "#64748B"      # muted gray
FAINT = "#CBD5E1"      # faint border
PALE = "#E2E8F0"       # very soft fill
BLUE = "#0072B2"       # lexical seeds / query flow
AMBER = "#E69F00"      # active causal walk
AMBER_TXT = "#B45309"  # darker amber for text legibility
GREEN = "#009E73"      # gold defect / success indicator
RED = "#D55E00"        # suppressed hubs / error indicators


def T(ax, x, y, s, size=5.5, color=INK, ha="center", va="center", weight="normal",
      style="normal", mono=False, z=6, **kw):
    if mono:
        kw["family"] = "monospace"
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=z, **kw)


def rich_line(ax, x, y, parts, ha="left", gap=4.5):
    """Draw a single line made of differently-styled text runs (baseline aligned)."""
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
    return total


def line(ax, xs, ys, color=INK, lw=0.7, ls="-", z=5, alpha=1.0):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
            solid_capstyle="round", dash_capstyle="round")


def stage_arrow(ax, p0, p1, color=MUTED, lw=0.9, ms=5.2, z=4):
    """Unified horizontal inter-stage transition arrow with consistent head & length."""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>,head_length=0.50,head_width=0.25",
        mutation_scale=ms, lw=lw, ec=color, fc=color, zorder=z,
        capstyle="round"
    ))


def check(ax, x, y, s=4.0, color="white", lw=1.0, z=9):
    line(ax, [x - 0.5 * s, x - 0.12 * s, x + 0.55 * s],
         [y + 0.02 * s, y - 0.38 * s, y + 0.48 * s], color=color, lw=lw, z=z)


def cross(ax, x, y, s=3.8, color=RED, lw=1.1, z=8):
    """Clean, solid, perfectly symmetrical red X mark."""
    line(ax, [x - s, x + s], [y - s, y + s], color=color, lw=lw, ls="-", z=z)
    line(ax, [x - s, x + s], [y + s, y - s], color=color, lw=lw, ls="-", z=z)


def draw_curly_brace(ax, x_spine, y_bot, y_top, w=6.5, r=12.0, color=INK, lw=1.2, z=5):
    """
    Draw a mathematically sound curly brace { opening to the right.
    Both top and bottom ends curl smoothly outward to the right (x_spine + w).
    The center cusp points smoothly to the left (x_spine - w).
    """
    y_mid = (y_bot + y_top) / 2.0
    c = 0.55228475  # cubic bezier constant for 90-deg circle quadrant
    
    verts = [
        # 1. Start at top tip: horizontal tangent pointing left
        (x_spine + w, y_top),
        (x_spine + w - c * w, y_top),
        (x_spine, y_top - r + c * r),
        (x_spine, y_top - r),
        
        # 2. Straight line down spine to near center cusp
        (x_spine, y_mid + r),
        
        # 3. Cubic curve into center cusp at (x_spine - w, y_mid)
        (x_spine, y_mid + r - c * r),
        (x_spine - w + c * w, y_mid),
        (x_spine - w, y_mid),
        
        # 4. Cubic curve out of center cusp back to vertical spine at (x_spine, y_mid - r)
        (x_spine - w + c * w, y_mid),
        (x_spine, y_mid - r + c * r),
        (x_spine, y_mid - r),
        
        # 5. Straight line down spine to near bottom tip
        (x_spine, y_bot + r),
        
        # 6. Cubic curve to bottom tip at (x_spine + w, y_bot): horizontal tangent pointing right
        (x_spine, y_bot + r - c * r),
        (x_spine + w - c * w, y_bot),
        (x_spine + w, y_bot),
    ]
    
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
    ]
    
    path = Path(verts, codes)
    patch = PathPatch(path, fc="none", ec=color, lw=lw, zorder=z, capstyle="round")
    ax.add_patch(patch)
    return patch


def smooth_closed(pts):
    """Closed Catmull-Rom spline converted to cubic Bezier segments."""
    P = np.asarray(pts, float)
    n = len(P)
    verts, codes = [P[0]], [Path.MOVETO]
    for i in range(n):
        p0, p1, p2, p3 = P[(i - 1) % n], P[i], P[(i + 1) % n], P[(i + 2) % n]
        verts += [p1 + (p2 - p0) / 6.0, p2 - (p3 - p1) / 6.0, p2]
        codes += [Path.CURVE4] * 3
    verts.append(P[0])
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


BLOB = [(0.02, 0.50), (0.07, 0.80), (0.26, 0.96), (0.55, 0.97), (0.86, 0.93),
        (0.99, 0.64), (0.97, 0.32), (0.80, 0.08), (0.50, 0.03), (0.27, 0.09),
        (0.07, 0.22)]


def blob(ax, x, y, w, h, fc, alpha=1.0, z=1):
    pts = [(x + u * w, y + v * h) for u, v in BLOB]
    ax.add_patch(PathPatch(smooth_closed(pts), fc=fc, ec="none", alpha=alpha, zorder=z))


def doc_icon(ax, x, y, w=12, h=16, color=MUTED):
    f = 3.6
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h - f), (x + w - f, y + h),
                          (x, y + h)], closed=True, fc="white", ec=color, lw=0.7, zorder=4))
    line(ax, [x + w - f, x + w - f, x + w], [y + h, y + h - f, y + h - f], color=color, lw=0.6)
    for i in range(3):
        line(ax, [x + 2.2, x + w - 2.2 - (i == 2) * 3], [y + h - 7 - 3 * i] * 2, color=color, lw=0.5)


def gemma4_logo(ax, cx, cy, R=8.5, color="#3085FE"):
    """Publication-grade vector Google Gemma 4 drafting blueprint star emblem."""
    # 1. Outer Construction Circle (delicate drafting hairline)
    ax.add_patch(Circle((cx, cy), R, fill=False, ec=color, lw=0.45, zorder=3))

    # 2. Cardinal Crosshairs extending to 1.42 * R
    ext = 1.42 * R
    line(ax, [cx, cx], [cy - ext, cy + ext], color=color, lw=0.45, z=3)
    line(ax, [cx - ext, cx + ext], [cy, cy], color=color, lw=0.45, z=3)

    # 3. Inscribed Square with Drafting Line Extensions
    d = R / np.sqrt(2)
    line_ext = 0.28 * R
    # Horizontal framing lines
    line(ax, [cx - d - line_ext, cx + d + line_ext], [cy + d, cy + d], color=color, lw=0.45, z=3)
    line(ax, [cx - d - line_ext, cx + d + line_ext], [cy - d, cy - d], color=color, lw=0.45, z=3)
    # Vertical framing lines
    line(ax, [cx - d, cx - d], [cy - d - line_ext, cy + d + line_ext], color=color, lw=0.45, z=3)
    line(ax, [cx + d, cx + d], [cy - d - line_ext, cy + d + line_ext], color=color, lw=0.45, z=3)
    # Diagonal drafting line (top-left to bottom-right)
    line(ax, [cx - d, cx + d], [cy + d, cy - d], color=color, lw=0.45, z=3)

    # 4. Central 4-pointed Star (tips slightly poke through circle at 1.08 * R)
    R_tip = 1.08 * R
    k = 0.64 * R_tip
    verts = [
        (cx, cy + R_tip),  # Top tip
        (cx, cy + R_tip - k), (cx + R_tip - k, cy), (cx + R_tip, cy),  # to Right tip
        (cx + R_tip - k, cy), (cx, cy - R_tip + k), (cx, cy - R_tip),  # to Bottom tip
        (cx, cy - R_tip + k), (cx - R_tip + k, cy), (cx - R_tip, cy),  # to Left tip
        (cx - R_tip + k, cy), (cx, cy + R_tip - k), (cx, cy + R_tip),  # to Top tip
    ]
    codes = [Path.MOVETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CURVE4, Path.CURVE4]
    star_path = Path(verts, codes)
    star_patch = PathPatch(star_path, facecolor="white", edgecolor=color, lw=0.9, zorder=5)
    ax.add_patch(star_patch)


def cells(ax, x, y, vals, cw=13, ch=11, fills=None, fs=4.6):
    for i, v in enumerate(vals):
        fc = fills[i] if fills else "white"
        ax.add_patch(Rectangle((x + i * cw, y - ch / 2), cw, ch, fc=fc, ec=MUTED, lw=0.5, zorder=4))
        T(ax, x + i * cw + cw / 2, y, str(v), size=fs, mono=True)


NODES = {"bulk_create": (0.20, 0.50), "_batched_insert": (0.58, 0.80), "save": (0.40, 0.16),
         "Model": (0.80, 0.20), "logging": (0.92, 0.66), "isinstance": (0.62, 0.44)}
LABEL = {"bulk_create": (-8, 9, "center"), "_batched_insert": (0, 9, "center"),
         "save": (0, -8, "center"), "Model": (0, -8, "center"),
         "logging": (0, 8, "center"), "isinstance": (6, 0, "left")}
EDGES = [("bulk_create", "_batched_insert", "call"), ("bulk_create", "save", "call"),
         ("_batched_insert", "logging", "call"), ("_batched_insert", "isinstance", "call"),
         ("save", "Model", "inherit"), ("save", "isinstance", "call")]
HUBS = ("logging", "isinstance")
CHAR_W = 3.85


def draw_graph(ax, fx, fy, fw, fh, mode):
    P = {k: (fx + u * fw, fy + v * fh) for k, (u, v) in NODES.items()}
    active = mode == "active"
    R = {k: 3.8 for k in NODES}
    sh = lambda k: R[k] * 0.72 + 0.7

    for a, b, kind in EDGES:
        ls = "-" if kind == "call" else (0, (3, 1.6))
        if not active:
            ax.add_patch(FancyArrowPatch(
                P[a], P[b], arrowstyle="-|>,head_length=0.45,head_width=0.22",
                mutation_scale=4.2, lw=0.6, ec=MUTED, fc=MUTED, ls=ls,
                shrinkA=sh(a), shrinkB=sh(b), zorder=3, capstyle="round"
            ))
        elif (a, b) == ("bulk_create", "_batched_insert"):
            ax.add_patch(FancyArrowPatch(
                P[a], P[b], arrowstyle="-|>,head_length=0.45,head_width=0.22",
                mutation_scale=4.2, lw=0.6, ec=AMBER, fc=AMBER, ls=ls,
                shrinkA=sh(a), shrinkB=sh(b), zorder=7, capstyle="round"
            ))
        elif b in HUBS:
            ax.add_patch(FancyArrowPatch(
                P[a], P[b], arrowstyle="-|>,head_length=0.45,head_width=0.22",
                mutation_scale=4.2, lw=0.6, ec="#E8B79E", fc="#E8B79E", ls=(0, (2, 1.6)),
                shrinkA=sh(a), shrinkB=sh(b), zorder=3, capstyle="round"
            ))
        else:
            ax.add_patch(FancyArrowPatch(
                P[a], P[b], arrowstyle="-|>,head_length=0.45,head_width=0.22",
                mutation_scale=4.2, lw=0.6, ec=FAINT, fc=FAINT, ls=ls,
                shrinkA=sh(a), shrinkB=sh(b), zorder=3, capstyle="round"
            ))

    for k, (x, y) in P.items():
        r = R[k]
        if not active:
            ax.add_patch(Circle((x, y), r, fc=INK, ec="none", lw=0, zorder=6))
            col, wt = INK, "normal"
        elif k == "bulk_create":
            ax.add_patch(Circle((x, y), r, fc=BLUE, ec="none", lw=0, zorder=8))
            col, wt = BLUE, "bold"
        elif k == "_batched_insert":
            ax.add_patch(Circle((x, y), r, fc=GREEN, ec="none", lw=0, zorder=8))
            check(ax, x, y, s=3.2, color="white", lw=0.9)
            col, wt = GREEN, "bold"
        elif k in HUBS:
            ax.add_patch(Circle((x, y), r, fc=PALE, ec="none", lw=0, zorder=6))
            cross(ax, x, y, s=3.4, color=RED, lw=1.1)
            col, wt = "#94A3B8", "normal"
        else:
            ax.add_patch(Circle((x, y), r, fc="#94A3B8", ec="none", lw=0, zorder=6))
            col, wt = MUTED, "normal"
        dx, dy, ha = LABEL[k]
        lx = x + dx + (R[k] if ha == "left" else 0)
        T(ax, lx, y + dy, k, size=4.6, color=col, mono=True, ha=ha, weight=wt)
        if active and k in HUBS:
            w = len(k) * CHAR_W
            x0 = lx if ha == "left" else lx - w / 2
            line(ax, [x0 - 1, x0 + w + 1], [y + dy] * 2, color=RED, lw=0.8, ls="-", z=8)
    return P


def figure1(outdir=None):
    """
    Renders publication-grade Figure 1 architecture diagram.
    Saves to outdir (if provided) and paper/figures/perron_provenance_pipeline.{pdf, png}.
    """
    W, H = 825, 390
    fig = plt.figure(figsize=(8.25, 3.9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    cap = lambda x, y, s: T(ax, x, y, s, size=6.5, weight="bold", color=INK)

    # =========================================================================
    # TIER 1: OFFLINE INDEXING (Top)
    # =========================================================================
    T(ax, 34, 296, "Offline\nIndexing", size=8.8, weight="bold", color=INK, linespacing=1.2)
    draw_curly_brace(ax, x_spine=76, y_bot=225, y_top=368, w=6.5, r=12.0, color=INK, lw=1.2)

    xb1 = 94

    # (1) Codebase AST
    cap(xb1 + 56, 362, "Codebase AST")
    doc_icon(ax, xb1 + 2, 309, 12, 16)
    T(ax, xb1 + 18, 317, "django/db/models/query.py", size=4.4, mono=True, color=MUTED, ha="left")
    code = [("def bulk_create(objs, bs):", INK, "bold"), ("    ...", MUTED, "normal"),
            ("    _batched_insert(objs, bs)", MUTED, "normal"),
            ("def _batched_insert(objs, bs):", INK, "bold"), ("    ...", MUTED, "normal")]
    for i, (s, c, w) in enumerate(code):
        T(ax, xb1 + 2, 298 - 9.5 * i, s, size=4.8, mono=True, color=c, weight=w, ha="left")
    T(ax, xb1 + 56, 245, r"$|\mathcal{V}|\approx 10^{4}-10^{5}$ AST symbols", size=5.6, color=MUTED)

    # Inter-Stage Transition Arrow: length 18
    stage_arrow(ax, (xb1 + 128, 287), (xb1 + 146, 287))

    # (2) Typed Edge Extraction
    xb2 = xb1 + 146
    cap(xb2 + 65, 362, "Typed Edge Extraction")
    ax.add_patch(Circle((xb2 + 10, 287), 7, fc=INK, ec="none", zorder=5))
    T(ax, xb2 + 10, 287, "AST", size=4.2, color="white", weight="bold")
    rel = [("Calls", r"$\mathcal{E}_{\mathrm{call}}$, $\omega=0.50$", "-", False),
           ("Inheritance", r"$\mathcal{E}_{\mathrm{inherit}}$, $\omega=0.25$", (0, (3, 1.6)), False),
           ("Imports", r"$\mathcal{E}_{\mathrm{import}}$, $\omega=0.15$", (0, (1, 1.6)), False),
           ("Reverse calls", r"$\mathcal{E}_{\mathrm{caller}}$, $\omega=0.10$", "-", True)]
    for ry, (name, math, ls, rev) in zip([320, 298, 276, 254], rel):
        ax.add_patch(PathPatch(Path([(xb2 + 17, 287), (xb2 + 28, 287), (xb2 + 28, ry), (xb2 + 40, ry)],
                                    [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                               fc="none", ec=MUTED, lw=0.6, zorder=3))
        a, b = ((xb2 + 54, ry), (xb2 + 42, ry)) if rev else ((xb2 + 42, ry), (xb2 + 54, ry))
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>,head_length=0.45,head_width=0.22",
            mutation_scale=4.2, lw=0.8, ec=INK, fc=INK, ls=ls, zorder=4, capstyle="round"
        ))
        T(ax, xb2 + 59, ry + 4.2, name, size=5.6, weight="bold", ha="left")
        T(ax, xb2 + 59, ry - 4.2, math, size=5.3, color=MUTED, ha="left")
    T(ax, xb2 + 65, 233, r"$W_{uv}=\sum_{r}\omega_{r}A_{uv}^{(r)}$", size=6.4)

    # Inter-Stage Transition Arrow: length 18
    stage_arrow(ax, (xb2 + 128, 287), (xb2 + 146, 287))

    # (3) Directed Code Multigraph G
    xb3 = xb2 + 146
    cap(xb3 + 70, 362, r"Directed Code Multigraph $\mathcal{G}$")
    blob(ax, xb3 + 2, 228, 142, 118, "#EEF2F7")
    draw_graph(ax, xb3 + 14, 242, 116, 90, "neutral")

    # Inter-Stage Transition Arrow: length 18
    stage_arrow(ax, (xb3 + 146, 287), (xb3 + 164, 287))

    # (4) Decoupled CSR Matrix Engine
    xb4 = xb3 + 164
    cap(xb4 + 78, 362, "Decoupled CSR Matrix Engine")
    mx, my, c = xb4 + 4, 269, 6
    nz = {(0, 1), (0, 2), (1, 4), (1, 5), (2, 3), (2, 5)}
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((mx + j * c, my + (5 - i) * c), c, c,
                                   fc=INK if (i, j) in nz else "none", ec=FAINT, lw=0.4, zorder=4))
    T(ax, mx + 18, my + 44, r"sparse $W$", size=5.2, color=MUTED)

    # Inter-Stage Transition Arrow inside Stage 4: length 18
    stage_arrow(ax, (mx + 42, 287), (mx + 60, 287))

    ax0 = mx + 94
    rows = [(315, "indptr", [0, 2, 4, 6, 6, 6, 6], None),
            (295, "indices", [1, 2, 4, 5, 3, 5], None),
            (275, "data", [".5", ".5", ".5", ".5", ".25", ".5"],
             [PALE, PALE, PALE, PALE, "#F1F5F9", PALE])]
    for y, nm, vals, fills in rows:
        T(ax, ax0 - 4, y, nm, size=4.8, mono=True, color=MUTED, ha="right")
        cells(ax, ax0, y, vals, fills=fills)
    T(ax, ax0 + 45, 329, r"$W$ stored as three flat arrays", size=5.2, color=MUTED)

    # Stationary degree prior vector pi_g (6 cells: 12x10.5, ec=MUTED, lw=0.5)
    shades = ["#F1F5F9", "#F1F5F9", "#E2E8F0", "#E2E8F0", "#94A3B8", "#64748B"]
    for i, fcol in enumerate(shades):
        ax.add_patch(Rectangle((ax0 + 104, 319 - (i + 1) * 10.5), 12, 10.5, fc=fcol, ec=MUTED, lw=0.5, zorder=4))
    T(ax, ax0 + 110, 338, "degree prior", size=4.8, color=MUTED)
    T(ax, ax0 + 110, 329, r"$\pi_g$", size=6.6)
    line(ax, [ax0 + 120, ax0 + 122, ax0 + 122, ax0 + 120], [256.5, 256.5, 277.5, 277.5], color=MUTED, lw=0.5)
    T(ax, ax0 + 125, 267, "hubs:\nhigh $\\pi_g$", size=4.8, color=MUTED, ha="left", linespacing=1.15)
    
    # Corrected power iteration formula: row-stochastic T and beta = 0.85
    T(ax, xb4 + 78, 248, r"$\mathbf{p}_{t+1}=\beta\,T^{\top}\mathbf{p}_{t}+(1-\beta)\,\mathbf{p}_{0} \quad (\beta=0.85)$", size=5.6)
    T(ax, xb4 + 89, 233, r"$\leq$ 1.6 ms SpMM  |  $\leq$ 3 MB RAM  |  Zero-Copy mmap",
      size=5.6, weight="bold", color=INK)

    # =========================================================================
    # RE-USED MMAP CONNECTION (Vertical Dashed Arrow)
    # =========================================================================
    ax.add_patch(FancyArrowPatch(
        (496, 222), (496, 172),
        arrowstyle="-|>,head_length=0.50,head_width=0.25",
        mutation_scale=5.2, lw=0.8, ec=MUTED, fc=MUTED, ls=(0, (2.5, 2)), zorder=2, capstyle="round"
    ))
    T(ax, 502, 197, r"re-used: $W$, $\pi_g$ (mmap)", size=5.0, style="italic", color=MUTED, ha="left")

    # =========================================================================
    # TIER 2: ONLINE RETRIEVAL (Bottom)
    # =========================================================================
    T(ax, 34, 103, "Online\nRetrieval", size=8.8, weight="bold", color=INK, linespacing=1.2)
    draw_curly_brace(ax, x_spine=76, y_bot=28, y_top=178, w=6.5, r=12.0, color=INK, lw=1.2)

    xc1 = 94

    # (A) Bug Report Query
    cap(xc1 + 38, 166, "Bug Report Query")
    ax.add_patch(Circle((xc1 + 10, 138), 4.6, fc="none", ec=INK, lw=0.8, zorder=4))
    ax.add_patch(Circle((xc1 + 10, 138), 1.4, fc=INK, ec="none", zorder=4))
    T(ax, xc1 + 20, 139, "django-11951", size=5.8, mono=True, weight="bold", ha="left")
    T(ax, xc1 + 20, 129, "Un-Named Defect (Stratum C)", size=5.0, color=MUTED, ha="left")
    T(ax, xc1 + 4, 114, "Symptom (issue text)", size=4.8, style="italic", color=MUTED, ha="left")
    rich_line(ax, xc1 + 4, 102, [
        ("\u201c", dict(size=5.8)),
        ("bulk_create()", dict(size=5.0, mono=True, color=BLUE, weight="bold")),
        (" fails", dict(size=5.8))], gap=2.0)
    T(ax, xc1 + 4, 93, "with batch_size\u201d", size=5.8, ha="left")
    T(ax, xc1 + 4, 78, "Hidden target (not in issue):", size=4.8, style="italic", color=MUTED, ha="left")
    T(ax, xc1 + 4, 69, "_batched_insert", size=5.4, mono=True, weight="bold", color=GREEN, ha="left")
    cross(ax, xc1 + 7, 58, s=2.2, color=RED, lw=1.0)
    T(ax, xc1 + 13, 58, "never mentioned", size=5.0, style="italic", color=RED, ha="left")

    # Smooth query teleport curve to seed vector
    ax.add_patch(FancyArrowPatch(
        (xc1 + 92, 101), (xc1 + 144, 130),
        arrowstyle="-|>,head_length=0.45,head_width=0.22",
        mutation_scale=4.5, lw=0.8, ec=BLUE, fc=BLUE, connectionstyle="arc3,rad=-0.15", zorder=4
    ))

    # (B) Seed Personalization p0 (Vector exactly matching pi_g in Tier 1)
    xc2 = xc1 + 146
    cap(xc2 + 6, 166, "Seed Personalization")
    T(ax, xc2 + 6, 150, r"teleport $\mathbf{p}_0$", size=5.6, color=MUTED)
    
    for i in range(6):
        cy = 143 - (i + 1) * 10.5
        fc = BLUE if i == 0 else "white"
        ax.add_patch(Rectangle((xc2, cy), 12, 10.5, fc=fc, ec=MUTED, lw=0.5, zorder=4))
        if i == 0:
            T(ax, xc2 + 6, cy + 5.25, "1", size=5.0, mono=True, color="white", weight="bold")
        else:
            T(ax, xc2 + 6, cy + 5.25, "0", size=4.4, mono=True, color=MUTED)
            
    T(ax, xc2 + 16, 137.75, "1.0", size=5.0, mono=True, color=BLUE, weight="bold", ha="left")
    T(ax, xc2 + 6, 68, r"$\mathbf{p}_0[\mathrm{seed}]=1.0$", size=5.2)
    T(ax, xc2 + 6, 59, "sparse vector", size=4.8, style="italic", color=MUTED)

    # Inter-Stage Transition Arrow: length 18
    stage_arrow(ax, (xc2 + 36, 101), (xc2 + 54, 101))

    # (C) HERO: Specificity-Damped Diffusion
    xc3 = xc2 + 54
    cap(xc3 + 89, 166, "Specificity-Damped Diffusion")
    blob(ax, xc3 + 2, 44, 174, 112, "#EEF2F7")
    P = draw_graph(ax, xc3 + 12, 62, 148, 78, "active")
    mx_, my_ = (P["bulk_create"][0] + P["_batched_insert"][0]) / 2, (P["bulk_create"][1] + P["_batched_insert"][1]) / 2
    T(ax, mx_ - 6, my_ + 8, "calls", size=5.2, style="italic", color=AMBER_TXT, weight="bold")
    T(ax, xc3 + 89, 34, r"Hub Suppressed ($\gamma=0.70$, HSI@10 = 99.5%)", size=5.5, color=RED, weight="bold")
    T(ax, xc3 + 89, 24, r"specificity ratio  $\pi_q\,/\,(\pi_g+\epsilon)^{\gamma}$", size=5.4, color=MUTED)

    # Inter-Stage Transition Arrow: length 18
    stage_arrow(ax, (xc3 + 182, 101), (xc3 + 200, 101))

    # (D) Structural Provenance Verbalizer
    xc4 = xc3 + 200
    cap(xc4 + 56, 166, "Structural Provenance Verbalizer")
    ax.add_patch(Rectangle((xc4 + 4, 118.5), 1.8, 19, fc=AMBER, ec="none", zorder=5))
    
    # Corrected verbalizer call direction: called_by_seed (called by bulk_create)
    T(ax, xc4 + 11, 132, "# Provenance: called_by_seed", size=4.8, mono=True, weight="bold", color=AMBER_TXT, ha="left")
    T(ax, xc4 + 11, 123, "#   (called by bulk_create)", size=4.8, mono=True, color=AMBER_TXT, ha="left")
    T(ax, xc4 + 11, 112, "def _batched_insert(objs,", size=4.8, mono=True, weight="bold", ha="left")
    T(ax, xc4 + 11, 103, "        batch_size): ...", size=4.8, mono=True, weight="bold", ha="left")
    T(ax, xc4 + 4, 85, "without provenance (baseline):", size=4.8, style="italic", color=MUTED, ha="left")
    T(ax, xc4 + 11, 76, "def _batched_insert(...)", size=4.8, mono=True, color="#94A3B8", ha="left")
    T(ax, xc4 + 11, 67, "isolated: no graph context", size=4.8, style="italic", color=RED, ha="left")

    # Inter-Stage Transition Arrow into Gemma Chip: length 18
    stage_arrow(ax, (xc4 + 128, 127), (xc4 + 146, 127))

    # (E) Local Gemma 4 Scorer
    xc5 = xc4 + 146
    cap(xc5 + 68, 166, "Local Gemma 4 Scorer")
    gemma4_logo(ax, xc5 + 16, 127, R=8.5)
    T(ax, xc5 + 16, 108.5, "Gemma 4", size=5.4, weight="normal", color="#3085FE", ha="center")
    T(ax, xc5 + 16, 101.0, "(Local 2B / 4B)", size=4.6, color=MUTED, ha="center")
    ax.add_patch(Circle((xc5 + 32, 119), 1.6, fc="white", ec=MUTED, lw=0.6, zorder=5))
    ax.add_patch(Circle((xc5 + 37.5, 123.5), 2.3, fc="white", ec=MUTED, lw=0.6, zorder=5))
    ax.add_patch(Ellipse((xc5 + 106, 131), 134, 50, fc="white", ec=MUTED, lw=0.7, zorder=4))
    T(ax, xc5 + 106, 142.5, "<thought>", size=4.2, mono=True, color=MUTED)
    T(ax, xc5 + 106, 135.0, "bulk_create delegates batching", size=4.6)
    T(ax, xc5 + 106, 127.0, "to _batched_insert. Bug localized.", size=4.6)
    T(ax, xc5 + 106, 119.5, "</thought>", size=4.2, mono=True, color=MUTED)
    T(ax, xc5 + 4, 91, "Ranked output", size=4.8, style="italic", color=MUTED, ha="left")
    for y, rk, nm, top in [(82, "1", "_batched_insert", True), (73, "2", "save", False),
                           (64, "3", "Model", False)]:
        T(ax, xc5 + 4, y, rk, size=5.2, mono=True, color=GREEN if top else "#94A3B8", ha="left")
        T(ax, xc5 + 14, y, nm, size=5.2, mono=True, color=GREEN if top else MUTED,
          weight="bold" if top else "normal", ha="left")
    ax.add_patch(Circle((xc5 + 88, 82), 3.4, fc=GREEN, ec="none", zorder=6))
    check(ax, xc5 + 88, 82, s=4.2, lw=0.9)
    T(ax, xc5 + 95, 82, "Top-1", size=5.2, weight="bold", color=GREEN, ha="left")
    
    T(ax, xc5 + 75, 45, r"Acc@1: 4.6% $\rightarrow$ 9.9%  (p=0.039, 2.2$\times$ Lift)", size=5.6, weight="bold", color=INK)
    T(ax, xc5 + 75, 29, "Matched graph pool: +45.4% MRR", size=5.0, color=MUTED)

    # Save to targets
    paper_fig_dir = os.path.join("paper", "figures")
    os.makedirs(paper_fig_dir, exist_ok=True)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        
    for ext in ("pdf", "png"):
        destinations = [os.path.join(paper_fig_dir, f"perron_provenance_pipeline.{ext}")]
        if outdir and os.path.abspath(str(outdir)) != os.path.abspath(str(paper_fig_dir)):
            destinations.append(os.path.join(str(outdir), f"perron_pipeline_clean.{ext}"))
            
        for dest in destinations:
            saved = False
            for attempt in range(3):
                try:
                    fig.savefig(dest, dpi=300, facecolor="white")
                    print("wrote", dest)
                    saved = True
                    break
                except OSError:
                    time.sleep(0.6)
            if not saved:
                print("failed to write", dest)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "scratch"
    figure1(out_dir)


def generate_figure1(out_dir=None):
    return figure1(out_dir)

if __name__ == '__main__':
    generate_figure1()
