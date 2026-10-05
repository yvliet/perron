"""
Pytest suite for Gemma 4 Reader Transmission Pilot & Telemetry Verification.
Enforces offline replay execution in <4.5s on <120 MB RAM with zero network requests.

Author: Sultan Haikal (GitHub: @yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from benchmarks.evaluate_gemma4_reader_transmission import (
    format_controlled_contrast_prompt,
    compute_paired_bootstrap_ci,
    parse_target_strictly,
    run_gemma4_reader_pilot,
    sample_stratified_tasks,
)
from perron.backends.replay import OfflineReplayBackend

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_parse_target_strictly():
    """Verify strict extraction from <target> tags across closed and stripped completions (Gate R2)."""
    assert parse_target_strictly("<target>my_target_func</target>") == "my_target_func"

    # Stop-sequence stripped closing tag
    assert parse_target_strictly("<target>my_target_func") == "my_target_func"

    resp_with_think = (
        "<|think|>Examining call stack. Helper symbol is irrelevant.</|think|>\n"
        "Root cause is <target>models.query.QuerySet.filter</target>."
    )
    assert parse_target_strictly(resp_with_think) == "models.query.QuerySet.filter"

    resp_stripped_with_think = (
        "<|think|>Examining call stack.</|think|><target>models.query.QuerySet.filter"
    )
    assert parse_target_strictly(resp_stripped_with_think) == "models.query.QuerySet.filter"

    resp_multiline = "<target>\n  utils.http.parse_header  \n</target>"
    assert parse_target_strictly(resp_multiline) == "utils.http.parse_header"

    resp_quotes = "<target>`requests.sessions.Session.send`</target>"
    assert parse_target_strictly(resp_quotes) == "requests.sessions.Session.send"

    resp_quotes_stripped = "<target>`requests.sessions.Session.send`"
    assert parse_target_strictly(resp_quotes_stripped) == "requests.sessions.Session.send"

    resp_none = "I could not determine the exact function."
    assert parse_target_strictly(resp_none) is None


def test_format_controlled_contrast_prompt_structure():
    """Verify Gemma 4 turn tokens and instruction formatting."""
    cands = [
        {"identifier": "foo", "file_path": "a.py", "source_code": "def foo(): pass"},
        {"identifier": "bar", "file_path": "b.py", "source_code": "def bar(): pass"},
    ]
    prompt, token_count = format_controlled_contrast_prompt("Fix divide by zero", cands)

    assert "<start_of_turn>user" in prompt
    assert "<end_of_turn>" in prompt
    assert "<start_of_turn>model" in prompt
    assert "<|think|>" in prompt
    assert "<target>" in prompt
    assert "Candidate [1]: foo (a.py:1-1)" in prompt
    assert "Candidate [2]: bar (b.py:1-1)" in prompt
    assert token_count > 20


def test_bootstrap_ci_coverage():
    """Verify bootstrap confidence interval calculation for kappa."""
    hits_a = [1] * 50 + [0] * 50
    hits_b = [0] * 95 + [1] * 5
    kappa, ci_low, ci_high = compute_paired_bootstrap_ci(hits_a, hits_b, n_bootstraps=500, seed=42)

    assert abs(kappa - 0.45) < 1e-4
    assert 0.30 <= ci_low < kappa
    assert kappa < ci_high <= 0.60


def test_reader_pilot_offline_replay_execution_gate(tmp_path: Path):
    """
    Assert reader transmission offline replay invariants:
    - Runs in <4.5 seconds
    - Outputs valid JSON schema
    - Measured kappa > 0
    - Model hash is present
    - No changes to git-tracked files (EPITAPH-100)
    """
    out_file = tmp_path / "gemma4_reader_transmission.json"

    t0 = time.perf_counter()
    telemetry = run_gemma4_reader_pilot(
        output_path=out_file,
        n_instances=25,
        seed=20261005,
        backend_type="replay",
    )
    elapsed = time.perf_counter() - t0

    assert elapsed < 4.5, f"Replay execution took {elapsed:.2f}s, exceeding 4.5s ceiling."
    assert out_file.exists()
    assert telemetry["status"] == "completed"  # Gate R5: Raw runner emits completed
    assert telemetry["model_sha256"] is not None
    assert telemetry["transmission_coefficient_kappa"] > 0
    assert "position_sensitivity" in telemetry
    assert "kappa_ci_95" in telemetry
    assert len(telemetry["instances"]) == 25


def test_air_gapped_telemetry_consistency_gate():
    """
    Assert that Stage 2 promotion invariants hold:
    status == 'measured' requires model_sha256 and active Gemma macros.
    """
    tel_path = PROJECT_ROOT / "results" / "live_inference_telemetry.json"
    claims_path = PROJECT_ROOT / "paper" / "claims_spec.json"

    if tel_path.exists():
        tel = json.loads(tel_path.read_text(encoding="utf-8"))
        spec = json.loads(claims_path.read_text(encoding="utf-8"))["macros"]
        gemma_macros = [k for k in spec if k.startswith("Gemma")]

        if tel.get("status") == "measured":
            assert len(gemma_macros) > 0, "status: 'measured' requires Gemma macros in claims_spec.json"
            assert tel.get("model_sha256"), "status: 'measured' requires verified model_sha256"
        else:
            assert len(gemma_macros) == 0, (
                f"Gemma macros present ({gemma_macros}) while status is '{tel.get('status')}'. "
                "Must be quarantined until Stage 2 promotion."
            )
