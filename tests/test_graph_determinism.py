"""
Deterministic Code Graph Build Verification Test.
Builds 5 distinct SWE-bench Lite instances twice into isolated temporary worktrees,
asserting 100% bit-for-bit identical SHA-256 hashes across all CSR arrays,
Parquet tables, and manifest metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.export_swebench_graphs import (
    build_instance_graph,
    compute_file_sha256,
    load_swebench_instances,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_graph_determinism_across_rebuilds(tmp_path: Path):
    """Assert that building instances twice yields strictly identical SHA-256 digests."""
    cache_file = REPO_ROOT / "benchmarks" / "data" / "swebench_lite_cached.json"
    instances = load_swebench_instances(cache_file)
    assert len(instances) >= 5, "Requires at least 5 instances"

    # Select 5 diverse instances from different repositories
    selected = []
    seen_repos = set()
    for inst in instances:
        repo = inst.get("repo", "")
        if repo not in seen_repos and repo:
            seen_repos.add(repo)
            selected.append(inst)
        if len(selected) == 5:
            break

    repo_graphs_dir = REPO_ROOT / "artifacts" / "swebench_graphs"
    assert repo_graphs_dir.exists(), "Precompiled repo graphs required"

    dir_run1 = tmp_path / "run1"
    dir_run2 = tmp_path / "run2"
    dir_run1.mkdir()
    dir_run2.mkdir()

    checked_files = [
        "csr_data.npy",
        "csr_indices.npy",
        "csr_indptr.npy",
        "csr_dangling.npy",
        "nodes.parquet",
    ]

    for inst in selected:
        iid = inst["instance_id"]
        out1 = dir_run1 / iid
        out2 = dir_run2 / iid

        # Build run 1
        build_instance_graph(inst, repo_graphs_dir, out1)

        # Build run 2
        build_instance_graph(inst, repo_graphs_dir, out2)

        # Compare hashes
        for fname in checked_files:
            p1 = out1 / fname
            p2 = out2 / fname
            assert p1.exists(), f"Run 1 missing {fname}"
            assert p2.exists(), f"Run 2 missing {fname}"

            h1 = compute_file_sha256(p1)
            h2 = compute_file_sha256(p2)
            assert h1 == h2, f"Determinism failure for {iid}/{fname}: {h1} != {h2}"

        # Check manifest match
        with open(out1 / "manifest.json", "r", encoding="utf-8") as f:
            m1 = json.load(f)
        with open(out2 / "manifest.json", "r", encoding="utf-8") as f:
            m2 = json.load(f)

        assert m1["node_count"] == m2["node_count"]
        assert m1["edge_count"] == m2["edge_count"]
        assert m1["sha256"] == m2["sha256"]
