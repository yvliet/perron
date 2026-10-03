"""
Unit tests for authoritative evaluation metrics in perron/metrics.py.

Verifies:
- Function Acc@K (hit if ANY gold function is in top K)
- File Acc@K (hit if file of ANY gold function is in top K)
- Coverage@4k (fraction of gold functions fitting in context budget)
- HSI@10 = 1 - |top10 ∩ H25| / 10

Mandatory test cases included:
- empty gold
- duplicate IDs
- K larger than candidate list
- gold not in graph / candidates
"""

from __future__ import annotations

import pytest
from perron.metrics import (
    coverage_at_4k,
    file_acc_at_k,
    function_acc_at_k,
    hsi_at_10,
)


def test_function_acc_at_k_standard() -> None:
    retrieved = [10, 20, 30, 40, 50]
    gold = [30, 99]

    # At k=1, top is [10] -> miss
    assert function_acc_at_k(retrieved, gold, k=1) == 0.0
    # At k=2, top is [10, 20] -> miss
    assert function_acc_at_k(retrieved, gold, k=2) == 0.0
    # At k=3, top is [10, 20, 30] -> hit (30 in gold)
    assert function_acc_at_k(retrieved, gold, k=3) == 1.0
    # At k=5, hit
    assert function_acc_at_k(retrieved, gold, k=5) == 1.0


def test_function_acc_mandatory_edge_cases() -> None:
    # 1. Empty gold
    assert function_acc_at_k(retrieved=[1, 2, 3], gold=[], k=5) == 0.0
    assert function_acc_at_k(retrieved=[], gold=[], k=5) == 0.0

    # 2. Duplicate IDs in retrieved and gold
    retrieved_dups = [10, 10, 20, 20, 30]
    gold_dups = [20, 20]
    # k=1 has [10] -> 0.0
    assert function_acc_at_k(retrieved_dups, gold_dups, k=1) == 0.0
    # k=3 has [10, 10, 20] -> 1.0
    assert function_acc_at_k(retrieved_dups, gold_dups, k=3) == 1.0

    # 3. K larger than candidate list
    retrieved_short = [10, 20]
    assert function_acc_at_k(retrieved_short, [20], k=100) == 1.0
    assert function_acc_at_k(retrieved_short, [30], k=100) == 0.0

    # 4. Gold not in graph (not in candidate list)
    assert function_acc_at_k(retrieved=[1, 2, 3], gold=[9999, 8888], k=3) == 0.0


def test_file_acc_at_k_standard() -> None:
    retrieved_files = ["src/core.py", "src/utils.py", "src/models/user.py"]
    gold_files = ["src/models/user.py"]

    assert file_acc_at_k(retrieved_files, gold_files, k=1) == 0.0
    assert file_acc_at_k(retrieved_files, gold_files, k=2) == 0.0
    assert file_acc_at_k(retrieved_files, gold_files, k=3) == 1.0


def test_file_acc_mandatory_edge_cases() -> None:
    # 1. Empty gold
    assert file_acc_at_k(["a.py", "b.py"], [], k=5) == 0.0

    # 2. Duplicate IDs / paths
    retrieved = ["a.py", "a.py", "b.py"]
    assert file_acc_at_k(retrieved, ["b.py", "b.py"], k=2) == 0.0
    assert file_acc_at_k(retrieved, ["b.py"], k=3) == 1.0

    # 3. K larger than candidate list
    assert file_acc_at_k(["a.py"], ["a.py"], k=50) == 1.0
    assert file_acc_at_k(["a.py"], ["z.py"], k=50) == 0.0

    # 4. Gold not in graph / repo
    assert file_acc_at_k(["a.py", "b.py"], ["nonexistent/missing.py"], k=2) == 0.0

    # Path normalization matching
    assert file_acc_at_k(["foo/bar.py"], ["bar.py"], k=1) == 1.0
    assert file_acc_at_k(["bar.py"], ["foo/bar.py"], k=1) == 1.0
    assert file_acc_at_k(["foo\\bar.py"], ["foo/bar.py"], k=1) == 1.0


def test_coverage_at_4k_standard() -> None:
    retrieved = [1, 2, 3, 4, 5]
    gold = [2, 4, 6]
    # Each symbol costs 1000 tokens. Budget 4096 fits 4 symbols: [1, 2, 3, 4]
    costs = {1: 1000, 2: 1000, 3: 1000, 4: 1000, 5: 1000}
    # Gold [2, 4] are in the top 4 -> 2 out of 3 gold symbols covered
    score = coverage_at_4k(retrieved, gold, symbol_token_costs=costs, budget=4096)
    assert pytest.approx(score, rel=1e-3) == 2.0 / 3.0


def test_coverage_at_4k_mandatory_edge_cases() -> None:
    # 1. Empty gold
    assert coverage_at_4k([1, 2, 3], [], budget=4096) == 0.0

    # 2. Duplicate IDs
    retrieved_dups = [1, 1, 2, 2, 3]
    # Duplicates should not consume duplicate budget or corrupt coverage
    score = coverage_at_4k(retrieved_dups, [1, 2], budget=4096, default_cost=100)
    assert score == 1.0

    # 3. K larger than candidate list
    assert coverage_at_4k([1], [1], budget=4096) == 1.0

    # 4. Gold not in graph
    assert coverage_at_4k([1, 2, 3], [999, 888], budget=4096) == 0.0

    # Zero budget
    assert coverage_at_4k([1, 2], [1, 2], budget=0) == 0.0


def test_hsi_at_10_standard() -> None:
    # Top 25 hubs
    hubs = set(range(100, 125))

    # Case A: Zero hubs in top-10 -> HSI = 1 - 0/10 = 1.0
    retrieved_clean = list(range(10))
    assert hsi_at_10(retrieved_clean, hubs) == 1.0

    # Case B: 3 hubs in top-10 -> HSI = 1 - 3/10 = 0.7
    retrieved_3_hubs = [100, 1, 2, 101, 3, 4, 102, 5, 6, 7]
    assert pytest.approx(hsi_at_10(retrieved_3_hubs, hubs), rel=1e-4) == 0.7

    # Case C: All 10 are hubs -> HSI = 1 - 10/10 = 0.0
    retrieved_all_hubs = list(range(100, 110))
    assert hsi_at_10(retrieved_all_hubs, hubs) == 0.0


def test_hsi_at_10_mandatory_edge_cases() -> None:
    hubs = {10, 20, 30}

    # 1. Empty candidate list
    assert hsi_at_10([], hubs) == 1.0

    # 2. Fewer candidates than 10
    retrieved_short = [10, 1]  # 1 hub in 2 candidates -> 1 - 1/10 = 0.9
    assert pytest.approx(hsi_at_10(retrieved_short, hubs), rel=1e-4) == 0.9

    # 3. Duplicate IDs in retrieved
    retrieved_dups = [10, 10, 1, 2, 3, 4, 5, 6, 7, 8]  # two occurrences of hub 10
    # Overlap in top 10 elements: two items are in hub_set -> 1 - 2/10 = 0.8
    assert pytest.approx(hsi_at_10(retrieved_dups, hubs), rel=1e-4) == 0.8

    # 4. Empty hubs
    assert hsi_at_10([1, 2, 3], set()) == 1.0
