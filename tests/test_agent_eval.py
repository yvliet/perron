"""
Tests for scripts/agent_eval.py and empirical memory telemetry.

Verifies:
- Wilson score interval calculation logic
- Schema compliance of results/agent_summary.json
- Schema and provenance compliance of results/memory_measured.json
- Execution boundary invariant adherence (tests_pass is null)
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import pytest

from scripts.agent_eval import wilson_score_interval, run_agent_evaluation


def test_wilson_score_interval_bounds() -> None:
    """Verify Wilson interval returns valid probabilities [0, 1]."""
    low, high = wilson_score_interval(0, 10)
    assert 0.0 <= low <= high <= 1.0
    assert low == 0.0

    low, high = wilson_score_interval(10, 10)
    assert 0.0 <= low <= high <= 1.0
    assert high == 1.0

    low, high = wilson_score_interval(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8


def test_agent_eval_smoke_execution() -> None:
    """Run an isolated smoke test of run_agent_evaluation and verify output artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "test_agent_summary.json"
        memory_path = Path(tmpdir) / "test_memory_measured.json"
        instance_dir = Path(tmpdir) / "agent"

        summary_data, mem_data = run_agent_evaluation(
            arms=["none", "perron"],
            split="dev_val",
            limit=2,
            seed=20261003,
            token_budget=1000,
            instance_output_dir=instance_dir,
            output_summary_path=summary_path,
            output_memory_path=memory_path,
        )

        assert summary_path.is_file()
        assert memory_path.is_file()
        assert instance_dir.is_dir()

        # Check summary content
        assert summary_data["sample_size"] == 2
        assert "none" in summary_data["arms"]
        assert "perron" in summary_data["arms"]
        assert summary_data["arms"]["none"]["tests_pass"] is None
        assert summary_data["arms"]["perron"]["tests_pass"] is None
        assert summary_data["execution_boundary_enforced"] is True

        # Check memory measured content
        assert "engine_label" in mem_data
        assert "process_memory" in mem_data
        assert mem_data["process_memory"]["peak_process_rss_mb"] > 0
        assert mem_data["hardware_host"]["total_system_ram_gb"] > 0

        # Check provenance metadata in saved JSON
        with open(memory_path, "r", encoding="utf-8") as f:
            saved_mem = json.load(f)
        for key in ["git_sha", "python_version", "package_versions", "seed", "split_name", "hardware", "utc_timestamp"]:
            assert key in saved_mem, f"Missing provenance key: {key}"
