"""
scripts/figures - Publication-Grade Modular Figure Generation Package for Perron.

Active Manuscript Figures:
  - Figure 1: Architecture & Structural Provenance Pipeline (perron_provenance_pipeline.{pdf, png})
  - Figure 2: Candidate Pool-Size Truncation Curves (truncation_curves.{pdf, png})
  - Figure 3: Break-Even Discrimination Bound & Lexical Bottleneck (breakeven_discrimination_bound.{pdf, png})
  - Figure 4: Causal Provenance Forest Plot & Effect Sizes (causal_provenance_forest.{pdf, png})
  - Figure 5: Four-Stage Bug Localization Loss Funnel (loss_funnel.{pdf, png})
  - Figure 6: Continuous Dose-Response Surface & Damping Sensitivity (neff_dose_response.{pdf, png})

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from pathlib import Path
from scripts.figures.common import FIGURES_DIR
from scripts.figures.figure1_pipeline import generate_figure1
from scripts.figures.figure2_truncation import generate_figure2
from scripts.figures.figure3_breakeven import generate_figure3
from scripts.figures.figure4_causal_forest import generate_figure4
from scripts.figures.figure5_loss_funnel import generate_figure5
from scripts.figures.figure6_dose_response import generate_figure6

FIGURES_REGISTRY = {
    1: {
        "name": "Figure 1: Architecture & Structural Provenance Pipeline",
        "description": "Five-stage end-to-end provenance pipeline and test-time thought generation",
        "outputs": ["perron_provenance_pipeline.pdf", "perron_provenance_pipeline.png"],
        "generator": generate_figure1,
    },
    2: {
        "name": "Figure 2: Candidate Pool-Size Truncation Curves",
        "description": "Candidate pool oracle recall and Gemma 4 Acc@5 scaling across M in [10, 50]",
        "outputs": ["truncation_curves.pdf", "truncation_curves.png"],
        "generator": generate_figure2,
    },
    3: {
        "name": "Figure 3: Break-Even Discrimination Bound & Lexical Bottleneck",
        "description": "Theoretical break-even curves and within-pool discrimination raincloud distributions",
        "outputs": ["breakeven_discrimination_bound.pdf", "breakeven_discrimination_bound.png"],
        "generator": generate_figure3,
    },
    4: {
        "name": "Figure 4: Causal Provenance Forest Plot & Relative Effect Sizes",
        "description": "Stratum C defect localization forest plot and relative effect size lifts",
        "outputs": ["causal_provenance_forest.pdf", "causal_provenance_forest.png"],
        "generator": generate_figure4,
    },
    5: {
        "name": "Figure 5: Four-Stage Bug Localization Loss Funnel",
        "description": "Sankey loss funnel decomposing upstream boundary vs downstream compaction loss",
        "outputs": ["loss_funnel.pdf", "loss_funnel.png"],
        "generator": generate_figure5,
    },
    6: {
        "name": "Figure 6: Continuous Dose-Response Surface & Damping Sensitivity",
        "description": "Two-panel dose-response curve across 186 configurations and beta sensitivity",
        "outputs": ["neff_dose_response.pdf", "neff_dose_response.png"],
        "generator": generate_figure6,
    },
}


def generate_all_figures(out_dir: Path = FIGURES_DIR):
    """Generate all active manuscript figures (Figures 1-6)."""
    for fig_id, meta in sorted(FIGURES_REGISTRY.items()):
        print(f"\n--> Running [{fig_id}/6] {meta['name']}...")
        meta["generator"](out_dir=out_dir)


__all__ = [
    "FIGURES_REGISTRY",
    "FIGURES_DIR",
    "generate_figure1",
    "generate_figure2",
    "generate_figure3",
    "generate_figure4",
    "generate_figure5",
    "generate_figure6",
    "generate_all_figures",
]
