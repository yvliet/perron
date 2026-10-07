"""
tests/test_provenance.py - Unit tests for structural call-path provenance resolution and rendering.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from perron.reranker.provenance import (
    CallPathProvenanceRenderer,
    EdgeType,
    ProvenanceResolver,
    SymbolProvenance,
)


def test_edge_type_human_labels():
    assert "seed" in EdgeType.SEED.to_human_label().lower()
    assert "called by" in EdgeType.CALLED_BY_SEED.to_human_label().lower()
    assert "calls" in EdgeType.CALLS_SEED.to_human_label().lower()
    assert "base class" in EdgeType.BASE_CLASS_OF_SEED.to_human_label().lower()
    assert "subclasses" in EdgeType.SUBCLASSES_SEED.to_human_label().lower()


def test_symbol_provenance_formatting():
    # Direct seed
    p_seed = SymbolProvenance(edge_type=EdgeType.SEED, seed_symbol_id=42, hops=0)
    assert "direct lexical seed" in p_seed.format_comment().lower()

    # Called by seed with identifier and file
    p_called = SymbolProvenance(
        edge_type=EdgeType.CALLED_BY_SEED,
        seed_symbol_id=10,
        seed_identifier="bulk_create",
        seed_file_path="django/db/models/query.py",
        hops=1,
    )
    cmt = p_called.format_comment()
    assert "# Provenance: Called by seed function 'bulk_create'" in cmt
    assert "(django/db/models/query.py)" in cmt

    # Co-located
    p_coloc = SymbolProvenance(
        edge_type=EdgeType.CO_LOCATED,
        seed_symbol_id=5,
        seed_file_path="foo/bar.py",
        hops=1,
    )
    assert "co-located in same file as seed" in p_coloc.format_comment().lower()


def test_call_path_renderer_injection():
    renderer = CallPathProvenanceRenderer()
    prov = SymbolProvenance(
        edge_type=EdgeType.CALLS_SEED,
        seed_symbol_id=20,
        seed_identifier="is_valid_ipv6",
        seed_file_path="django/utils/ipv6.py",
    )

    base_bc = (
        "# File: django/core/validators.py (Lines 10-25)\n"
        "def validate_ipv6(value):\n"
        "    pass"
    )

    rendered = renderer.format_breadcrumb_with_provenance(base_bc, prov)
    lines = rendered.splitlines()
    assert lines[0] == "# File: django/core/validators.py (Lines 10-25)"
    assert lines[1].startswith("# Provenance: Calls seed function 'is_valid_ipv6'")
    assert "def validate_ipv6(value):" in lines[2]

    # Test without leading '# File:' line
    raw_bc = "def standalone():\n    pass"
    rendered_raw = renderer.format_breadcrumb_with_provenance(raw_bc, prov)
    raw_lines = rendered_raw.splitlines()
    assert raw_lines[0].startswith("# Provenance: Calls seed function")
    assert raw_lines[1] == "def standalone():"


def test_provenance_resolver_seed_match():
    resolver = ProvenanceResolver(graphs_dir="artifacts/swebench_graphs")
    seeds = {10, 20, 30}
    prov = resolver.resolve(candidate_symbol_id=20, seed_symbol_ids=seeds, repo_slug="non_existent")
    assert prov.edge_type == EdgeType.SEED
    assert prov.seed_symbol_id == 20
    assert prov.hops == 0


def test_provenance_resolver_unknown_repo():
    resolver = ProvenanceResolver(graphs_dir="artifacts/swebench_graphs")
    prov = resolver.resolve(candidate_symbol_id=999, seed_symbol_ids={1, 2}, repo_slug="totally_fake_repo")
    assert prov.edge_type == EdgeType.UNKNOWN


def test_provenance_resolver_real_astropy():
    graphs_dir = Path("artifacts/swebench_graphs")
    if not (graphs_dir / "astropy__astropy_layers.npz").exists():
        pytest.skip("Artifact graph layers not present in test environment")

    resolver = ProvenanceResolver(graphs_dir=graphs_dir)
    # Check astropy-14539: candidate 4229 (TableDataDiff._diff) calls seed 15227 (where_not_allclose)
    prov = resolver.resolve(
        candidate_symbol_id=4229,
        seed_symbol_ids={15227},
        repo_slug="astropy__astropy",
    )
    assert prov.edge_type == EdgeType.CALLS_SEED
    assert prov.seed_symbol_id == 15227
    assert prov.seed_identifier == "where_not_allclose"
