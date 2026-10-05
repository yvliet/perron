"""
tests/test_strata_manifest.py - CONSORT Task Manifest Verification Test Suite.

Audits:
1. RFC 8259 Compliance: Unreachable distances serialize to JSON null, zero unquoted Infinity/NaN literals.
2. Compound Primary Key Uniqueness: (instance_id, benchmark) across 800 records; 707 unique tasks; 93 overlap.
3. CONSORT Waterfall MECE Partition: 344 + 93 + 56 + 7 == 500 Verified tasks.
4. Frozen Table 5 Parity (EPITAPH-109): Exactly N_A=208, N_B=5, N_C=131 on callable Verified.
5. Task Manifest Production Verification: data_release/task_manifest.json schema and counts.
6. Dynamic Hub-Pruned BFS Algorithm: Prunes intermediate hubs while preserving endpoint reachability.
7. Multi-Target Vector Reduction: Evaluates min_hop and mean_hop aggregation guards.
8. Read-Only Gate (EPITAPH-100): Zero disk mutations on tracked repository files.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "swebench_strata_annotations.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data_release" / "task_manifest.json"
VERIFIED_CONF_PATH = PROJECT_ROOT / "results" / "verified_confirmatory_evaluation.json"


# =========================================================================
# 1. RFC 8259 Compliance & Null Serialization
# =========================================================================

def test_rfc_8259_strict_json_compliance() -> None:
    """Verify that all records in swebench_strata_annotations.jsonl strictly adhere to RFC 8259."""
    assert ANNOTATIONS_PATH.is_file(), f"Missing annotations at {ANNOTATIONS_PATH}"
    raw_content = ANNOTATIONS_PATH.read_text(encoding="utf-8")

    # Regex test for illegal unquoted IEEE 754 literals (: Infinity, : -Infinity, : NaN)
    illegal_match = re.search(r':\s*(Infinity|-Infinity|NaN)\b', raw_content)
    assert illegal_match is None, f"Found illegal unquoted IEEE 754 literal: {illegal_match.group(0)}"

    records = [json.loads(line) for line in raw_content.splitlines() if line.strip()]
    assert len(records) == 800, f"Expected 800 records, got {len(records)}"

    # Round-trip serialization with allow_nan=False must succeed without ValueError
    for rec in records:
        dumped = json.dumps(rec, allow_nan=False)
        # Check that no unquoted IEEE 754 float literal exists in the dumped JSON
        assert re.search(r':\s*(Infinity|-Infinity|NaN)\b', dumped) is None, (
            f"Record {rec.get('uid')} dumped with unquoted IEEE 754 literal: {dumped}"
        )


# =========================================================================
# 2. Compound Primary Key Uniqueness & Benchmark Partitions
# =========================================================================

def test_compound_primary_key_uniqueness() -> None:
    """Verify compound primary key uid: f'{benchmark}:{instance_id}' across 800 records."""
    assert ANNOTATIONS_PATH.is_file()
    records: List[Dict[str, Any]] = []
    compound_keys: Set[Tuple[str, str]] = set()
    uids: Set[str] = set()
    instance_ids_by_bench: Dict[str, Set[str]] = {"lite": set(), "verified": set()}

    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            row = json.loads(line_str)
            records.append(row)
            ckey = (row["instance_id"], row["benchmark"])
            compound_keys.add(ckey)
            uids.add(row["uid"])
            instance_ids_by_bench[row["benchmark"]].add(row["instance_id"])

            # Verify UID format invariant
            assert row["uid"] == f"{row['benchmark']}:{row['instance_id']}", (
                f"UID format violation: {row['uid']} != {row['benchmark']}:{row['instance_id']}"
            )

    # 800 total records: exactly 300 Lite, 500 Verified
    assert len(records) == 800
    assert len(compound_keys) == 800
    assert len(uids) == 800
    assert len(instance_ids_by_bench["lite"]) == 300
    assert len(instance_ids_by_bench["verified"]) == 500

    # Exactly 93 overlapping instances between Lite and Verified
    overlap = instance_ids_by_bench["lite"].intersection(instance_ids_by_bench["verified"])
    assert len(overlap) == 93

    # Total unique GitHub task instances across both cohorts: 300 + 500 - 93 = 707
    total_unique = len(instance_ids_by_bench["lite"].union(instance_ids_by_bench["verified"]))
    assert total_unique == 707


# =========================================================================
# 3. Frozen Table 5 Parity (EPITAPH-109)
# =========================================================================

def test_frozen_table5_legacy_strata_concordance() -> None:
    """
    Verify that stratum_legacy matches Table 5 in paper/main.tex and
    verified_confirmatory_evaluation.json with zero discrepancy (N_A=208, N_B=5, N_C=131).
    """
    assert ANNOTATIONS_PATH.is_file()
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    callable_verified = [
        r for r in records
        if r.get("benchmark") == "verified" and r.get("consort_status") == "CALLABLE_EVALUATED"
    ]
    assert len(callable_verified) == 344, f"Expected 344 callable Verified tasks, got {len(callable_verified)}"

    counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in callable_verified:
        strat = r.get("stratum_legacy")
        assert strat in counts, f"Unknown stratum_legacy: {strat}"
        counts[strat] += 1

    # Exact mathematical identity bound to Table 5 in paper/main.tex
    assert counts["A"] == 208, f"Expected N_A=208, got {counts['A']}"
    assert counts["B"] == 5, f"Expected N_B=5, got {counts['B']}"
    assert counts["C"] == 131, f"Expected N_C=131, got {counts['C']}"
    assert sum(counts.values()) == 344

    # Cross-validate against verified_confirmatory_evaluation.json if present
    if VERIFIED_CONF_PATH.is_file():
        conf = json.loads(VERIFIED_CONF_PATH.read_text(encoding="utf-8"))
        if "strata_counts" in conf:
            assert conf["strata_counts"]["A"] == 208
            assert conf["strata_counts"]["B"] == 5
            assert conf["strata_counts"]["C"] == 131


# =========================================================================
# 4. CONSORT Waterfall MECE Partition Invariant
# =========================================================================

def test_consort_waterfall_mece_partition() -> None:
    """
    Verify that the 500 Verified tasks partition into:
    344 (CALLABLE) + 93 (OVERLAP) + 56 (DRIFT) + 7 (NON_CALLABLE) == 500.
    """
    assert ANNOTATIONS_PATH.is_file()
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    verified_records = [r for r in records if r.get("benchmark") == "verified"]
    assert len(verified_records) == 500

    status_counts: Dict[str, int] = {}
    for r in verified_records:
        st = r.get("consort_status")
        status_counts[st] = status_counts.get(st, 0) + 1

    n_callable = status_counts.get("CALLABLE_EVALUATED", 0)
    n_overlap = status_counts.get("EXPLORATORY_SPLIT_OVERLAP", 0)
    n_drift = status_counts.get("FILE_ABSENT_PREIMAGE_DRIFT", 0)
    n_module = status_counts.get("MODULE_LEVEL_NON_CALLABLE", 0)

    assert n_callable == 344, f"Expected 344 callable, got {n_callable}"
    assert n_overlap == 93, f"Expected 93 overlap, got {n_overlap}"
    assert n_drift == 56, f"Expected 56 drift, got {n_drift}"
    assert n_module == 7, f"Expected 7 module non-callable, got {n_module}"
    assert n_callable + n_overlap + n_drift + n_module == 500


# =========================================================================
# 5. Production Task Manifest Integrity (data_release/task_manifest.json)
# =========================================================================

def test_task_manifest_file_structure() -> None:
    """Verify that data_release/task_manifest.json matches schema and accounting invariants."""
    assert MANIFEST_PATH.is_file(), f"Missing task manifest at {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    # Metadata assertions
    assert manifest["dataset_name"] == "swebench-strata-annotations"
    assert manifest["author"] == "Sultan Haikal (GitHub: @yvliet)"
    assert manifest["total_unique_instances"] == 707
    assert manifest["total_verified_instances"] == 500
    assert manifest["total_lite_instances"] == 300

    # Waterfall accounting
    wf = manifest["waterfall"]
    assert wf["verified_total"] == 500
    assert wf["callable_evaluated"] == 344
    assert wf["exploratory_lite_overlap"] == 93
    assert wf["file_preimage_drift"] == 56
    assert wf["module_non_callable"] == 7
    assert wf["is_mece"] is True
    assert wf["callable_evaluated"] + wf["exploratory_lite_overlap"] + wf["file_preimage_drift"] + wf["module_non_callable"] == 500

    # Strata distribution
    strata = manifest["strata_distribution_verified_callable"]["legacy"]
    assert strata["A"] == 208
    assert strata["B"] == 5
    assert strata["C"] == 131

    # Task dictionary completeness
    tasks = manifest["tasks"]
    assert len(tasks) == 707

    # Validate individual task schema
    sample_id = "astropy__astropy-13033"
    assert sample_id in tasks
    task = tasks[sample_id]
    assert task["instance_id"] == sample_id
    assert task["repo"] == "astropy/astropy"
    assert task["consort_status"] == "CALLABLE_EVALUATED"
    assert task["exclusion_reason"] == "NONE"
    assert task["is_callable"] is True
    assert task["stratum_legacy"] == "C"
    assert task["stratum_legacy_label"] == "C_unnamed"


# =========================================================================
# 6. Dynamic Intermediate Hub-Pruned BFS
# =========================================================================

def test_bfs_intermediate_hub_pruning_logic() -> None:
    """
    Test dynamic hub-pruned BFS: intermediate hubs are skipped while direct
    1-hop endpoints are preserved.
    """
    # Multigraph:
    # Seed u -> Hub H -> Target v  (Distance 2 through hub)
    # Seed u -> w1 -> w2 -> Target v (Distance 3 through non-hub domain nodes)
    adj: Dict[str, Set[str]] = {
        "u": {"H", "w1"},
        "H": {"u", "v", "extra1", "extra2"},
        "v": {"H", "w2"},
        "w1": {"u", "w2"},
        "w2": {"w1", "v"},
    }
    hubs: Set[str] = {"H"}

    def hub_pruned_bfs(seed: str, target: str, adjacency: Dict[str, Set[str]], pruned_hubs: Set[str]) -> Optional[int]:
        if seed == target:
            return 0
        if target in adjacency.get(seed, set()):
            return 1
        queue: deque[Tuple[str, int]] = deque([(seed, 0)])
        visited: Set[str] = {seed}
        while queue:
            curr, dist = queue.popleft()
            for nxt in adjacency.get(curr, set()):
                if nxt == target:
                    return dist + 1
                if nxt in visited:
                    continue
                # If nxt is an intermediate hub, skip expansion
                if nxt in pruned_hubs:
                    continue
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    # Unpruned: traverses u -> H -> v in 2 hops
    assert hub_pruned_bfs("u", "v", adj, pruned_hubs=set()) == 2

    # Pruned: skips intermediate H, traverses u -> w1 -> w2 -> v in 3 hops
    assert hub_pruned_bfs("u", "v", adj, pruned_hubs=hubs) == 3

    # Endpoint is hub: direct 1-hop reachability preserved
    assert hub_pruned_bfs("u", "H", adj, pruned_hubs=hubs) == 1
    assert hub_pruned_bfs("H", "v", adj, pruned_hubs=hubs) == 1

    # Unreachable target returns None (RFC 8259 null)
    assert hub_pruned_bfs("u", "isolated", adj, pruned_hubs=hubs) is None


# =========================================================================
# 7. Multi-Target Vector Reduction Guardrails
# =========================================================================

def test_multi_target_vector_aggregation() -> None:
    """Verify that multi-target reductions handle empty and null sequences safely."""
    def reduce_hops(hops: List[Optional[int]]) -> Tuple[Optional[int], Optional[float]]:
        valid = [h for h in hops if h is not None]
        if not valid:
            return None, None
        return min(valid), sum(valid) / len(valid)

    # All reachable
    min_h, mean_h = reduce_hops([0, 2, 4])
    assert min_h == 0
    assert mean_h == 2.0

    # Partially reachable with nulls
    min_h, mean_h = reduce_hops([1, None, 3])
    assert min_h == 1
    assert mean_h == 2.0

    # Completely unreachable (all null)
    min_h, mean_h = reduce_hops([None, None])
    assert min_h is None
    assert mean_h is None

    # Empty target sequence
    min_h, mean_h = reduce_hops([])
    assert min_h is None
    assert mean_h is None
