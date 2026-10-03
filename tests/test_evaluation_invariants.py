"""
Unit tests for GoldPatchParser and task-grounded evaluation metrics.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from benchmarks.eval_harness import (
    compute_file_acc_at_k,
    compute_function_acc_at_k,
    compute_context_purity,
    compute_hsi,
    compute_bootstrap_ci,
    files_match,
)
from benchmarks.gold_patch_parser import (
    parse_unified_diff,
    GoldPatchParser,
    build_stratified_split,
)


def test_files_match():
    assert files_match("requests/models.py", "models.py")
    assert files_match("models.py", "requests/models.py")
    assert files_match("a/b/c.py", "a/b/c.py")
    assert not files_match("requests/api.py", "requests/models.py")


def test_parse_unified_diff():
    diff_sample = """diff --git a/requests/models.py b/requests/models.py
--- a/requests/models.py
+++ b/requests/models.py
@@ -100,5 +100,6 @@ def prepare(self):
     pass
"""
    files, hunks = parse_unified_diff(diff_sample)
    assert files == ["requests/models.py"]
    assert len(hunks) == 1
    f, start, count = hunks[0]
    assert f == "requests/models.py"
    assert start == 100
    assert count == 5


def test_compute_file_acc_at_k():
    retrieved = ["requests/api.py", "requests/models.py", "requests/sessions.py"]
    gold = ["models.py"]

    acc = compute_file_acc_at_k(retrieved, gold, k_values=[1, 2, 3])
    assert acc[1] == 0.0
    assert acc[2] == 1.0
    assert acc[3] == 1.0


def test_compute_function_acc_at_k():
    retrieved = [10, 25, 42, 99]
    gold = [42, 105]

    acc = compute_function_acc_at_k(retrieved, gold, k_values=[1, 2, 3, 5])
    assert acc[1] == 0.0
    assert acc[2] == 0.0
    assert acc[3] == 1.0
    assert acc[5] == 1.0


def test_compute_context_purity():
    retrieved_symbols = [1, 2, 3, 4]
    gold_symbols = [2, 4]
    purity = compute_context_purity(retrieved_symbols, [], gold_symbols, [])
    assert purity == 0.5


def test_compute_hsi():
    retrieved = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    hubs = {2, 5}
    hsi = compute_hsi(retrieved, hubs, top_k=10)
    assert np.isclose(hsi, 0.8)


def test_compute_bootstrap_ci():
    values = [0.0, 1.0, 1.0, 1.0, 0.0]
    mean, lower, upper = compute_bootstrap_ci(values, n_bootstrap=500, ci=0.95, seed=42)
    assert np.isclose(mean, 0.6)
    assert lower <= mean <= upper


def test_stratified_split_balance():
    split_path = Path("data/swebench_lite_split.json")
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dev_count"] == 150
        assert data["heldout_count"] == 150
        assert len(set(data["dev_instances"]) & set(data["heldout_instances"])) == 0
