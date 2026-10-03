"""
Unit tests for deterministic lexical stratification (T2.1).

Verifies:
- 6 hand-written test examples (2 per stratum: A_named, B_trace, C_unnamed).
- Deterministic behavior on edge cases.
- data/strata.json schema and completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from perron.analysis.strata import (
    STRATUM_A,
    STRATUM_B,
    STRATUM_C,
    classify_stratum,
    generate_strata_file,
    matches_named_function,
    matches_traceback_gold_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -------------------------------------------------------------------------
# 6 Hand-written test examples (2 per stratum)
# -------------------------------------------------------------------------

def test_stratum_a_example_1_exact_function_name() -> None:
    """Stratum A Example 1: Issue explicitly names the target function."""
    issue = "When calling compute_separable_matrix with empty dimensions, it raises ValueError."
    gold_files = ["astropy/modeling/separable.py"]
    gold_symbols = ["compute_separable_matrix"]

    assert matches_named_function(issue, gold_symbols) is True
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_A


def test_stratum_a_example_2_class_method_qualified() -> None:
    """Stratum A Example 2: Issue names Class.method or method within class."""
    issue = "Bug in QuerySet.filter when combining with Q objects having negation: QuerySet.filter returns duplicate rows."
    gold_files = ["django/db/models/query.py"]
    gold_symbols = ["QuerySet.filter"]

    assert matches_named_function(issue, gold_symbols) is True
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_A


def test_stratum_b_example_1_standard_python_traceback() -> None:
    """Stratum B Example 1: No gold function name, but standard Python traceback points to gold file."""
    issue = """
    Encountered unexpected crash while rendering template with nested variables:
    Traceback (most recent call last):
      File "/app/django/template/base.py", line 450, in render
      File "/app/django/template/defaulttags.py", line 82, in render
      File "/app/django/core/serializers/json.py", line 102, in Serializer
    ZeroDivisionError: division by zero
    """
    gold_files = ["django/core/serializers/json.py"]
    # The gold symbol is internal and NOT mentioned anywhere in the issue text
    gold_symbols = ["_internal_serializer_worker"]

    assert matches_named_function(issue, gold_symbols) is False
    assert matches_traceback_gold_file(issue, gold_files) is True
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_B


def test_stratum_b_example_2_pytest_traceback() -> None:
    """Stratum B Example 2: Pytest-style traceback format pointing to gold file."""
    issue = """
    Running the test suite fails on Python 3.11:
    ___________________________ test_matrix_multiplication ___________________________
    sklearn/decomposition/_pca.py:214: in <module>
        components = svd_flip(u, v)
    E   ValueError: operands could not be broadcast together with shapes
    """
    gold_files = ["sklearn/decomposition/_pca.py"]
    gold_symbols = ["_fit_full_decomposition"]

    assert matches_named_function(issue, gold_symbols) is False
    assert matches_traceback_gold_file(issue, gold_files) is True
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_B


def test_stratum_c_example_1_pure_conceptual_issue() -> None:
    """Stratum C Example 1: High-level feature / conceptual bug with no function name and no traceback."""
    issue = """
    Support for datetime with timezone in parquet reader seems inconsistent when daylight savings starts.
    The timestamp columns are shifted by one hour instead of keeping the UTC epoch.
    Expected UTC preservation without timezone skew.
    """
    gold_files = ["pydata/pandas/io/parquet.py"]
    gold_symbols = ["read_parquet_with_metadata"]

    assert matches_named_function(issue, gold_symbols) is False
    assert matches_traceback_gold_file(issue, gold_files) is False
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_C


def test_stratum_c_example_2_traceback_to_unrelated_files_only() -> None:
    """Stratum C Example 2: Traceback present, but only points to external or unrelated libraries."""
    issue = """
    Import error on startup:
    Traceback (most recent call last):
      File "/usr/lib/python3.10/importlib/__init__.py", line 126, in import_module
      File "/site-packages/numpy/core/__init__.py", line 22, in <module>
    ImportError: cannot import name 'c_api'
    """
    gold_files = ["sympy/core/sympify.py"]
    gold_symbols = ["sympify_strict"]

    assert matches_named_function(issue, gold_symbols) is False
    assert matches_traceback_gold_file(issue, gold_files) is False
    assert classify_stratum(issue, gold_files, gold_symbols) == STRATUM_C


# -------------------------------------------------------------------------
# Integration test for data/strata.json
# -------------------------------------------------------------------------

def test_data_strata_json_generated_and_valid() -> None:
    """Verify data/strata.json exists and maps all 300 instances to A, B, or C."""
    strata_file = PROJECT_ROOT / "data" / "strata.json"
    if not strata_file.exists():
        generate_strata_file(output_path=strata_file)

    assert strata_file.is_file(), "data/strata.json was not generated!"

    with open(strata_file, "r", encoding="utf-8") as f:
        strata = json.load(f)

    # 300 instances in SWE-bench Lite
    assert len(strata) == 300, f"Expected 300 instances in strata.json, got {len(strata)}"

    # All values must be 'A', 'B', or 'C'
    valid_strata = {STRATUM_A, STRATUM_B, STRATUM_C}
    for iid, s in strata.items():
        assert s in valid_strata, f"Instance {iid} has invalid stratum: {s}"

    counts = {s: sum(1 for v in strata.values() if v == s) for s in valid_strata}
    assert counts[STRATUM_A] > 0
    assert counts[STRATUM_B] > 0
    assert counts[STRATUM_C] > 0
