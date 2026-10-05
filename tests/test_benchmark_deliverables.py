"""
Tests for benchmark and dataset export deliverables:
- scripts/export_stratum_dataset.py
- benchmarks/evaluate_gemma4_localization.py (dry-run mode and target extraction)
- benchmarks/run_beta_sweep.py (--trials Optuna mode)
- scripts/generate_recall_curves.py
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from benchmarks.evaluate_gemma4_localization import run_packing_benchmark
from scripts.generate_recall_curves import load_budget_sweep_data, plot_recall_curves

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_evaluate_gemma4_localization_dry_run(tmp_path: Path) -> None:
    """Verify that evaluate_gemma4_localization runs cleanly in deterministic dry-run mode."""
    res = run_packing_benchmark(
        n_instances=2,
        backend_type="deterministic",
        output_file=tmp_path / "test_gemma4_localization.json",
    )
    assert res is not None
    assert "summary" in res
    assert "bm25_pack" in res["summary"]
    assert "perron_pack" in res["summary"]
    assert "slot_fused_pack" in res["summary"]
    for arm in ["bm25_pack", "perron_pack", "slot_fused_pack"]:
        assert "gold_function_inclusion_rate" in res["summary"][arm]
        assert "top1_function_acc" in res["summary"][arm]
        assert "avg_token_occupancy" in res["summary"][arm]


def test_generate_recall_curves_data_and_plot(tmp_path: Path) -> None:
    """Verify that recall curve data loads and plots cleanly to PDF and PNG."""
    data = load_budget_sweep_data()
    assert "budgets" in data and len(data["budgets"]) >= 5
    assert "curves" in data

    for method, curves in data["curves"].items():
        assert "fn_recall" in curves
        assert "file_recall" in curves
        assert len(curves["fn_recall"]) == len(data["budgets"])

    pdf_p, png_p = plot_recall_curves(data, tmp_path)
    assert pdf_p.exists() and pdf_p.stat().st_size > 0
    assert png_p.exists() and png_p.stat().st_size > 0
