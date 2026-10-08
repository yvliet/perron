#!/usr/bin/env python3
"""
scripts/figures/common.py - Shared Palette, Typography, and Persistence Utilities.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

import os
import sys
import time
from pathlib import Path

# Deterministic reproducible PDF builds
os.environ.setdefault("SOURCE_DATE_EPOCH", "1790812800")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Wong (2011) Okabe-Ito Colorblind-Safe Palette
COLOR_BLUE = "#0072B2"
COLOR_VERMILION = "#D55E00"
COLOR_GREEN = "#009E73"
COLOR_ORANGE = "#E69F00"
COLOR_SKY = "#56B4E9"
COLOR_PURPLE = "#CC79A7"
COLOR_YELLOW = "#F0E442"
COLOR_BLACK = "#000000"

# Architectural & Structural Palette
COLOR_SLATE = "#334155"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#CBD5E1"
COLOR_BG_CARD = "#F8FAFC"
COLOR_RED = "#DC2626"
COLOR_AMBER = "#D97706"
COLOR_INK = "#0F172A"
COLOR_PALE = "#E2E8F0"
COLOR_FAINT = "#CBD5E1"


def apply_serif_style():
    """Apply serif typography matching academic LaTeX publication standards."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.edgecolor": COLOR_SLATE,
        "axes.linewidth": 0.8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 9.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_figure(fig, base_name: str, out_dir: Path = FIGURES_DIR, dpi: int = 300):
    """
    Save figure as both vector PDF and high-resolution raster PNG with retry resilience.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{base_name}.pdf"
    png_path = out_dir / f"{base_name}.png"

    for path in (pdf_path, png_path):
        saved = False
        for attempt in range(3):
            try:
                fig.savefig(path, bbox_inches="tight", dpi=dpi, facecolor="white")
                saved = True
                break
            except OSError:
                time.sleep(0.4)
        if saved:
            size_kb = path.stat().st_size / 1024
            print(f"  [SAVED] {path.name:<38} ({size_kb:>7.1f} KB)")
        else:
            print(f"  [ERROR] Failed to save {path}")

    plt.close(fig)
