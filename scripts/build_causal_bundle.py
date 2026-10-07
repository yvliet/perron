#!/usr/bin/env python3
"""
scripts/build_causal_bundle.py - Build Causal Call-Path Evaluation Bundle.

Extracts the 25 graph-only gold target tasks (where 1-hop graph expansion A4
successfully reached a defect absent from lexical seeds A1), resolves their
structural AST/call-graph provenance back to seeds using ProvenanceResolver,
and packages both baseline and provenance-augmented candidate representations.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.gold_patch_parser import get_repo_slug
from benchmarks.run_gemma4_protocol import load_engines
from perron.reranker.candidate_pools import CandidatePoolBuilder, is_filter_f_passing
from perron.reranker.provenance import CallPathProvenanceRenderer, EdgeType, ProvenanceResolver
from perron.reranker.scorer import BreadcrumbRenderer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("build_causal_bundle")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data_release" / "task_manifest.json"
TASKS_PATH = PROJECT_ROOT / "data" / "swebench_verified_tasks.jsonl"
GOLD_PATH = PROJECT_ROOT / "data" / "swebench_verified_gold_labels.json"
SCORES_PARQUET = PROJECT_ROOT / "data_release" / "scores.parquet"
RANKINGS_DIR = PROJECT_ROOT / "results" / "gemma4_protocol" / "rankings"
CLONES_DIR = PROJECT_ROOT / "artifacts" / "clones"
GRAPHS_DIR = PROJECT_ROOT / "artifacts" / "swebench_graphs"
OUTPUT_BUNDLE = PROJECT_ROOT / "data_release" / "causal_callpath_eval_bundle.json"


def build_causal_bundle(output_path: Path = OUTPUT_BUNDLE) -> Dict[str, Any]:
    logger.info("Initializing Causal Call-Path bundle builder...")

    gold_labels = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["tasks"]
    tasks_by_id = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in open(TASKS_PATH, encoding="utf-8")
        if line.strip()
    }

    def load_rankings(arm: str) -> Dict[str, List[int]]:
        f = RANKINGS_DIR / f"{arm}.jsonl"
        return {json.loads(l)["instance_id"]: json.loads(l)["ranked_ids"] for l in open(f, encoding="utf-8")}

    a1_ranks = load_rankings("A1")
    a4_ranks = load_rankings("A4")

    # 1. Identify the 25 graph-only tasks (gold in A4 candidate pool, absent from A1 candidate pool)
    graph_only_task_ids: List[str] = []
    for inst_id, ranked_ids in a4_ranks.items():
        if inst_id not in gold_labels:
            continue
        g_ids = set(gold_labels[inst_id].get("gold_symbol_ids", []))
        if not g_ids:
            continue
        a1_pool = set(a1_ranks.get(inst_id, []))
        a4_pool = set(ranked_ids)
        if bool(g_ids.intersection(a4_pool)) and not bool(g_ids.intersection(a1_pool)):
            graph_only_task_ids.append(inst_id)

    logger.info(f"Identified {len(graph_only_task_ids)} graph-only gold target tasks.")
    assert len(graph_only_task_ids) == 25, f"Expected 25 tasks, found {len(graph_only_task_ids)}"

    # Load pre-computed scores
    scores_df = pd.read_parquet(SCORES_PARQUET)
    p0_scores_df = scores_df[scores_df["prompt_variant"] == "P0"]
    scores_map = p0_scores_df.set_index(["instance_id", "candidate_id"])["s"].to_dict()

    # Load canonical repository engines & provenance resolver
    engines, _, layers = load_engines()
    builders = {slug: CandidatePoolBuilder(engine=eng, layers=layers.get(slug)) for slug, eng in engines.items()}
    renderer = BreadcrumbRenderer(clones_dir=CLONES_DIR)
    resolver = ProvenanceResolver(graphs_dir=GRAPHS_DIR)
    provenance_renderer = CallPathProvenanceRenderer(resolver=resolver)

    bundle_tasks: List[Dict[str, Any]] = []

    for iid in graph_only_task_ids:
        raw_task = tasks_by_id[iid]
        gold_info = gold_labels[iid]
        gold_ids = set(gold_info.get("gold_symbol_ids", []))
        stratum = manifest.get(iid, {}).get("stratum", "Unknown")

        slug = get_repo_slug(raw_task.get("repo", ""))
        eng = engines.get(slug)
        builder = builders.get(slug)
        if not eng or not builder:
            continue

        q_text = (raw_task.get("problem_statement") or "") + " " + (raw_task.get("hints_text") or "")
        seeds, _, p0 = builder.compute_seeds(q_text)
        seed_set = set(seeds)

        e_graph = builder.compute_graph_expansion(seeds, p0)
        m_t = max(10, min(50, 10 + min(len(e_graph), 40)))
        needed_a2 = m_t - 10
        pool_a4 = list(seeds) + [item[0] for item in e_graph[:needed_a2]]

        candidates_data: List[Dict[str, Any]] = []
        for c_idx, node_u in enumerate(pool_a4):
            sym = eng.symbols[node_u]
            sym_id = sym.get("id", node_u)
            ident = sym.get("identifier", "unknown")
            fpath = sym.get("file_path", "")
            l_start = sym.get("line_start", 1)
            l_end = sym.get("line_end", l_start)

            # Baseline breadcrumb
            bc_baseline = renderer.extract_breadcrumb(sym, slug, mode="skeleton")

            # Resolve structural provenance
            prov = resolver.resolve(node_u, seed_set, slug, candidate_file_path=fpath)
            bc_provenance = provenance_renderer.format_breadcrumb_with_provenance(bc_baseline, prov)

            pt_score = float(scores_map.get((iid, node_u), -999.0))
            is_gold = bool(sym_id in gold_ids or node_u in gold_ids)

            candidates_data.append({
                "cand_index": c_idx,
                "node_idx": node_u,
                "symbol_id": sym_id,
                "identifier": ident,
                "file_path": fpath,
                "line_start": l_start,
                "line_end": l_end,
                "is_gold": is_gold,
                "pointwise_score": pt_score,
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
            "repo": raw_task.get("repo", ""),
            "stratum": stratum,
            "problem_statement": raw_task.get("problem_statement", ""),
            "hints_text": raw_task.get("hints_text", ""),
            "gold_symbol_ids": list(gold_ids),
            "gold_symbol_names": gold_info.get("gold_symbol_names", []),
            "seed_node_indices": list(seeds),
            "pool_size": len(pool_a4),
            "official_a4_ranking": a4_ranks.get(iid, []),
            "candidates": candidates_data,
        })

    bundle = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "description": "Causal Call-Path Structural Prompting Evaluation Bundle on 25 Graph-Only Gold Target Tasks",
        "author": "Sultan Haikal (@yvliet)",
        "license": "Apache-2.0",
        "n_tasks": len(bundle_tasks),
        "tasks": bundle_tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    logger.info(f"Successfully saved causal bundle with {len(bundle_tasks)} tasks to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return bundle


if __name__ == "__main__":
    build_causal_bundle()
