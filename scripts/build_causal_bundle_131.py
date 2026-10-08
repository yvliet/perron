#!/usr/bin/env python3
"""
scripts/build_causal_bundle_131.py - Build Full 131-Task Causal Call-Path Provenance Bundle.

Constructs the comprehensive evaluation bundle for all 131 un-named Stratum C tasks,
resolving AST caller-callee provenance back to issue lexical seeds for every candidate.
Provides both baseline breadcrumbs (Condition C1) and provenance-augmented breadcrumbs (Conditions C2 & C3).

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perron.reranker.provenance import CallPathProvenanceRenderer, EdgeType, ProvenanceResolver

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("build_causal_bundle_131")

STRATUM_C_BUNDLE = PROJECT_ROOT / "scratch" / "kaggle_dataset_benchmark" / "stratum_c_eval_bundle.json"
GRAPHS_DIR = PROJECT_ROOT / "artifacts" / "swebench_graphs"
OUTPUT_BUNDLE = PROJECT_ROOT / "data_release" / "causal_callpath_eval_bundle_131.json"


def build_full_131_causal_bundle(
    input_path: Path = STRATUM_C_BUNDLE,
    output_path: Path = OUTPUT_BUNDLE,
    max_cands_per_task: int = 25,
) -> Dict[str, Any]:
    logger.info(f"Loading base Stratum C evaluation bundle from {input_path}...")
    base_data = json.loads(input_path.read_text(encoding="utf-8"))
    base_tasks = base_data["tasks"]
    logger.info(f"Loaded {len(base_tasks)} Stratum C tasks.")

    resolver = ProvenanceResolver(graphs_dir=GRAPHS_DIR)
    provenance_renderer = CallPathProvenanceRenderer(resolver=resolver)

    bundle_tasks: List[Dict[str, Any]] = []
    total_candidates = 0
    resolved_candidates = 0

    for task in base_tasks:
        iid = task["instance_id"]
        repo = task["repo"]
        repo_slug = repo.replace("/", "__")
        gold_ids = set(task.get("gold_symbol_ids", []))
        all_cands = task.get("candidates", [])
        cand_map = {c["cand_index"]: c for c in all_cands}

        # Resolve seed symbol IDs from A1 candidate pool
        a1_indices = task.get("pools", {}).get("A1", [])
        seed_symbol_ids: List[int] = []
        for idx in a1_indices:
            c = cand_map.get(idx)
            if c:
                sym_id = c.get("symbol_id", c.get("node_idx", idx))
                seed_symbol_ids.append(sym_id)

        # Select A4 candidate pool for evaluation, capped at max_cands_per_task
        a4_indices = task.get("pools", {}).get("A4", [])[:max_cands_per_task]
        task_candidates: List[Dict[str, Any]] = []

        for rank_pos, c_idx in enumerate(a4_indices):
            c = cand_map.get(c_idx)
            if not c:
                continue

            total_candidates += 1
            sym_id = c.get("symbol_id", c.get("node_idx", c_idx))
            fpath = c.get("file_path", "")
            bc_baseline = c.get("breadcrumb", "")

            # Resolve structural provenance
            prov = resolver.resolve(sym_id, seed_symbol_ids, repo_slug, candidate_file_path=fpath)
            if prov.edge_type != EdgeType.UNKNOWN:
                resolved_candidates += 1

            bc_provenance = provenance_renderer.format_breadcrumb_with_provenance(bc_baseline, prov)
            is_gold = bool(sym_id in gold_ids or c.get("is_gold", False))

            task_candidates.append({
                "cand_index": c_idx,
                "node_idx": c.get("node_idx", sym_id),
                "symbol_id": sym_id,
                "identifier": c.get("identifier", "unknown"),
                "file_path": fpath,
                "line_start": c.get("line_start", 1),
                "line_end": c.get("line_end", 1),
                "is_gold": is_gold,
                "pointwise_score": c.get("pointwise_score", 0.0),
                "baseline_breadcrumb": bc_baseline,
                "provenance_breadcrumb": bc_provenance,
                "provenance": {
                    "edge_type": prov.edge_type.value,
                    "seed_symbol_id": prov.seed_symbol_id,
                    "seed_identifier": prov.seed_identifier,
                    "seed_file_path": prov.seed_file_path,
                    "comment": prov.format_comment(),
                },
            })

        bundle_tasks.append({
            "instance_id": iid,
            "repo": repo,
            "stratum": "C",
            "problem_statement": task.get("problem_statement", ""),
            "hints_text": task.get("hints_text", ""),
            "gold_symbol_ids": list(gold_ids),
            "gold_symbol_names": task.get("gold_symbol_names", []),
            "seed_symbol_ids": seed_symbol_ids,
            "pool_size": len(task_candidates),
            "candidates": task_candidates,
        })

    logger.info(
        f"Completed provenance resolution: {resolved_candidates}/{total_candidates} "
        f"candidates resolved ({resolved_candidates / max(1, total_candidates) * 100:.1f}%)."
    )

    bundle = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": "Causal Call-Path Structural Prompting Evaluation Bundle on all 131 Stratum C Un-Named Tasks",
        "author": "Sultan Haikal (@yvliet)",
        "license": "Apache-2.0",
        "n_tasks": len(bundle_tasks),
        "tasks": bundle_tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    logger.info(f"Saved full 131-task causal bundle to {output_path} ({output_path.stat().st_size / 1024:.1f} KB).")
    return bundle


if __name__ == "__main__":
    build_full_131_causal_bundle()
