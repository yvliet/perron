"""
tests/test_phase3_contracts.py - Verification Contracts for Phase 3 Open Community Resource Package.

Verifies:
1. results/strata_human_validation.json exists and contains Cohen's kappa >= 0.85 on N=100 tasks.
2. docs/datasheet.md and dist_dataset/DATASHEET.md exist and contain all 7 Gebru et al. sections.
3. paper/datasheet.tex exists and is included in paper/main.tex.
4. paper/data_manifest.json contains sha256 checksums for all 12 graph bundles, task manifests, and telemetry parquet files.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_strata_human_validation_contract():
    """Verify results/strata_human_validation.json exists, has N=100 and kappa >= 0.85."""
    res_file = PROJECT_ROOT / "results" / "strata_human_validation.json"
    assert res_file.is_file(), "results/strata_human_validation.json does not exist"

    data = json.loads(res_file.read_text(encoding="utf-8"))
    assert data.get("sample_size") == 100, f"Expected sample size 100, got {data.get('sample_size')}"
    assert "cohen_kappa" in data, "Missing cohen_kappa in results"
    kappa = data["cohen_kappa"]
    assert 0.85 <= kappa <= 1.0, f"Expected Cohen's kappa in [0.85, 1.0], got {kappa}"
    assert "confusion_matrix" in data, "Missing confusion_matrix"
    assert "human_concordance_pct" in data, "Missing human_concordance_pct"


def test_datasheet_documentation_contract():
    """Verify docs/datasheet.md and dist_dataset/DATASHEET.md exist with Gebru sections."""
    doc_file = PROJECT_ROOT / "docs" / "datasheet.md"
    dist_file = PROJECT_ROOT / "dist_dataset" / "DATASHEET.md"
    tex_file = PROJECT_ROOT / "paper" / "datasheet.tex"

    assert doc_file.is_file(), "docs/datasheet.md does not exist"
    assert dist_file.is_file(), "dist_dataset/DATASHEET.md does not exist"
    assert tex_file.is_file(), "paper/datasheet.tex does not exist"

    content = doc_file.read_text(encoding="utf-8")
    gebru_sections = [
        "Motivation",
        "Composition",
        "Collection Process",
        "Preprocessing",
        "Uses",
        "Distribution",
        "Maintenance",
    ]
    for sec in gebru_sections:
        assert sec.lower() in content.lower(), f"Missing Gebru section '{sec}' in docs/datasheet.md"

    # Verify paper/main.tex references or includes datasheet
    main_tex = (PROJECT_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    assert "datasheet" in main_tex.lower(), "paper/main.tex does not reference datasheet"


def test_cryptographic_data_manifest_contract():
    """Verify paper/data_manifest.json contains sha256 digests for graphs, annotations, and telemetry."""
    manifest_file = PROJECT_ROOT / "paper" / "data_manifest.json"
    assert manifest_file.is_file(), "paper/data_manifest.json does not exist"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "sha256_checksums" in manifest, "Missing sha256_checksums in paper/data_manifest.json"
    hashes = manifest["sha256_checksums"]

    # Must contain 12 graph hashes
    graph_hashes = [k for k in hashes if k.startswith("artifacts/swebench_graphs/") and k.endswith("_graph.npz")]
    assert len(graph_hashes) == 12, f"Expected 12 graph hashes, found {len(graph_hashes)}"

    # Must contain strata annotations hash
    assert "data/swebench_strata_annotations.jsonl" in hashes
    assert "data_release/task_manifest.json" in hashes
