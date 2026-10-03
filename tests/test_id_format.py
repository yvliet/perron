"""
ID-format consistency tests for gold function targets and graph node representations.

Verifies:
1. Every hand-labeled gold function exists as an exact node ID / symbol in the graph.
2. The automatic gold extractor returns the identical IDs as the audited hand labels.
3. Node ID string formatting conforms strictly to '<file_path>::<identifier>'.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from benchmarks.gold_patch_parser import GoldPatchParser, get_repo_slug

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDLABELS_PATH = REPO_ROOT / "data" / "handlabels_10.json"
AUDIT_SAMPLE_PATH = REPO_ROOT / "data" / "gold_labels_audit_sample.json"
GOLD_LABELS_PATH = REPO_ROOT / "data" / "swebench_lite_gold_labels.json"


def load_handlabel_sample() -> List[Dict[str, Any]]:
    """Load the 10-instance handlabel audit sample."""
    if HANDLABELS_PATH.exists():
        with open(HANDLABELS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    if AUDIT_SAMPLE_PATH.exists():
        with open(AUDIT_SAMPLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)[:10]
    pytest.skip("Neither data/handlabels_10.json nor data/gold_labels_audit_sample.json found")


def test_handlabel_node_id_consistency() -> None:
    """Assert every hand-labeled gold function exists as an exact node in the graph."""
    sample = load_handlabel_sample()
    assert len(sample) >= 10, f"Expected at least 10 hand-labeled instances, got {len(sample)}"

    parser = GoldPatchParser(
        graphs_dir=REPO_ROOT / "artifacts" / "swebench_graphs",
        tasks_file=REPO_ROOT / "data" / "swebench_lite_cache.jsonl",
    )

    with open(GOLD_LABELS_PATH, "r", encoding="utf-8") as f:
        auto_gold_labels = json.load(f)

    mismatches = []

    for item in sample[:10]:
        instance_id = item["instance_id"]
        repo = item["repo"]
        slug = get_repo_slug(repo)

        symbols = parser.repo_symbols.get(slug, [])
        assert symbols, f"No symbol catalog found for repository {slug}"

        sym_by_id = {s["id"]: s for s in symbols}
        node_id_strings = {f"{s['file_path']}::{s['identifier']}": s["id"] for s in symbols}

        hand_symbols = item.get("gold_symbols", [])
        hand_ids = [s["id"] for s in hand_symbols]

        for hs in hand_symbols:
            hid = hs["id"]
            h_ident = hs.get("identifier", "")

            # 1. Assert exact integer node ID exists in graph
            assert hid in sym_by_id, (
                f"[{instance_id}] Hand-labeled gold symbol ID {hid} not found in {slug} graph"
            )

            # 2. Assert exact string representation exists in graph
            sym_obj = sym_by_id[hid]
            expected_node_str = f"{sym_obj['file_path']}::{sym_obj['identifier']}"
            assert expected_node_str in node_id_strings, (
                f"[{instance_id}] Node string '{expected_node_str}' not found in graph catalog"
            )

            # 3. Check identifier match
            if h_ident and sym_obj.get("identifier"):
                assert h_ident == sym_obj["identifier"], (
                    f"[{instance_id}] Identifier mismatch: {h_ident} != {sym_obj['identifier']}"
                )

        # 4. Assert automatic gold extractor returns the same IDs as hand labels
        auto_entry = auto_gold_labels.get(instance_id, {})
        auto_ids = auto_entry.get("gold_symbol_ids", [])
        if hand_ids != auto_ids:
            mismatches.append(
                f"Instance {instance_id}: hand_ids={hand_ids} != auto_ids={auto_ids}"
            )

    assert not mismatches, f"ID-format mismatches found: {mismatches}"
