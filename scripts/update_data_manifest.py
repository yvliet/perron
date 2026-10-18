"""
scripts/update_data_manifest.py - Update paper/data_manifest.json with Phase 3 Cryptographic Checksums.

Computes SHA-256 digests for all 12 graph bundles, task manifests, evaluation
telemetry parquet files, and human validation receipts, recording them in
paper/data_manifest.json.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "paper" / "data_manifest.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def update_manifest():
    assert MANIFEST_PATH.is_file(), f"Missing manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files_to_hash = [
        # 12 Repository AST Multigraphs
        *(sorted((REPO_ROOT / "artifacts" / "swebench_graphs").glob("*.*"))),
        # Strata Annotations & Task Manifests
        REPO_ROOT / "data" / "swebench_strata_annotations.jsonl",
        REPO_ROOT / "dist_dataset" / "swebench_strata_annotations.jsonl",
        REPO_ROOT / "data_release" / "task_manifest.json",
        # Evaluation Telemetry
        REPO_ROOT / "data_release" / "scores.parquet",
        REPO_ROOT / "results" / "scores.parquet",
        REPO_ROOT / "results" / "gemma4_e4b_scores.parquet",
        REPO_ROOT / "results" / "strata_human_validation.json",
        # Datasheet Documentation
        REPO_ROOT / "docs" / "datasheet.md",
        REPO_ROOT / "dist_dataset" / "DATASHEET.md",
        REPO_ROOT / "paper" / "datasheet.tex",
    ]

    checksums: Dict[str, str] = {}
    for p in files_to_hash:
        assert p.is_file(), f"File to hash does not exist: {p}"
        rel_posix = p.relative_to(REPO_ROOT).as_posix()
        checksums[rel_posix] = compute_sha256(p)

    manifest["sha256_checksums"] = checksums

    # Add formal community resource package metadata
    human_val_file = REPO_ROOT / "results" / "strata_human_validation.json"
    human_val = json.loads(human_val_file.read_text(encoding="utf-8"))

    manifest["community_resource_package"] = {
        "title": "Perron Open Code Graph & Task Stratification Resource",
        "author": "Sultan Haikal (GitHub: @yvliet)",
        "license": "Apache-2.0",
        "total_unique_tasks": 707,
        "total_strata_records": 800,
        "swebench_verified_total": 500,
        "swebench_verified_callable": 344,
        "swebench_lite_total": 300,
        "total_graph_repositories": 12,
        "total_graph_nodes": 109240,
        "total_graph_edges": 341892,
        "inter_annotator_validation": {
            "sample_size": human_val["sample_size"],
            "human_concordance_pct": human_val["human_concordance_pct"],
            "cohen_kappa": human_val["cohen_kappa"],
            "cohen_kappa_ci_95": human_val["cohen_kappa_ci_95"],
            "chance_agreement_pe": human_val["chance_agreement_pe"],
            "interpretation": human_val["landis_koch_interpretation"]
        }
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Updated {MANIFEST_PATH} with {len(checksums)} SHA-256 checksums and community resource metadata.")


if __name__ == "__main__":
    update_manifest()
