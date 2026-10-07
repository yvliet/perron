#!/usr/bin/env python3
"""
scripts/build_stratum_c_bundle.py - Stratum C (Un-Named Tasks) Evaluation Bundle Builder.

Extracts the 131 Stratum C un-named defect tasks from SWE-bench Verified,
renders candidate AST breadcrumbs across comparison arms (A1 seeds, A2 depth BM25,
A4 graph 1-hop expansion, and B0 unscored baseline), and packages them with
ground-truth labels and pre-computed pointwise model scores into a compact,
offline-executable evaluation bundle.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.gold_patch_parser import get_repo_slug
from benchmarks.run_gemma4_protocol import load_engines
from perron.reranker.candidate_pools import CandidatePoolBuilder, is_filter_f_passing
from perron.reranker.scorer import BreadcrumbRenderer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_stratum_c_bundle")

MANIFEST_PATH = PROJECT_ROOT / "data_release" / "task_manifest.json"
TASKS_PATH = PROJECT_ROOT / "data" / "swebench_verified_tasks.jsonl"
GOLD_PATH = PROJECT_ROOT / "data" / "swebench_verified_gold_labels.json"
SCORES_PARQUET = PROJECT_ROOT / "data_release" / "scores.parquet"
RANKINGS_DIR = PROJECT_ROOT / "results" / "gemma4_protocol" / "rankings"
CLONES_DIR = PROJECT_ROOT / "artifacts" / "clones"
OUTPUT_BUNDLE = PROJECT_ROOT / "data_release" / "stratum_c_eval_bundle.json"


def build_stratum_c_bundle(output_path: Path = OUTPUT_BUNDLE) -> Dict[str, Any]:
    t0 = time.time()
    logger.info("Initializing Stratum C bundle construction...")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing task manifest: {MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    c_tasks_ids = [k for k, v in manifest["tasks"].items() if v.get("stratum_legacy") == "C"]
    logger.info(f"Loaded {len(c_tasks_ids)} Stratum C task IDs from manifest.")

    tasks_by_id = {
        json.loads(line)["instance_id"]: json.loads(line)
        for line in open(TASKS_PATH, encoding="utf-8")
        if line.strip()
    }
    gold_labels = json.loads(GOLD_PATH.read_text(encoding="utf-8"))

    # Load official rankings for ground truth verification
    official_rankings: Dict[str, Dict[str, List[int]]] = {}
    for arm in ("B0", "A1", "A2", "A4"):
        arm_f = RANKINGS_DIR / f"{arm}.jsonl"
        if arm_f.exists():
            official_rankings[arm] = {
                json.loads(l)["instance_id"]: json.loads(l)["ranked_ids"]
                for l in open(arm_f, encoding="utf-8")
                if l.strip()
            }
        else:
            logger.warning(f"Ranking file not found: {arm_f}")

    # Load pre-computed scores
    logger.info(f"Loading pre-computed pointwise scores from {SCORES_PARQUET}...")
    scores_df = pd.read_parquet(SCORES_PARQUET)
    p0_scores_df = scores_df[scores_df["prompt_variant"] == "P0"]
    scores_map = p0_scores_df.set_index(["instance_id", "candidate_id"])["s"].to_dict()
    mass_map = p0_scores_df.set_index(["instance_id", "candidate_id"])["format_mass"].to_dict()
    tokens_map = p0_scores_df.set_index(["instance_id", "candidate_id"])["n_prompt_tokens"].to_dict()

    # Load canonical repository engines
    engines, _, layers = load_engines()
    builders = {
        slug: CandidatePoolBuilder(engine=eng, layers=layers.get(slug))
        for slug, eng in engines.items()
    }
    renderer = BreadcrumbRenderer(clones_dir=CLONES_DIR)

    bundle_tasks: List[Dict[str, Any]] = []
    hits_counter = {"B0": 0, "A1": 0, "A2": 0, "A4": 0}

    for idx, iid in enumerate(c_tasks_ids, 1):
        if iid not in tasks_by_id or iid not in gold_labels:
            logger.warning(f"Task {iid} missing from tasks or gold labels; skipping.")
            continue

        raw_task = tasks_by_id[iid]
        gold_info = gold_labels[iid]
        gold_ids = set(gold_info.get("gold_symbol_ids", []))
        gold_names = gold_info.get("gold_symbol_names", [])
        gold_files = gold_info.get("gold_files", [])

        slug = get_repo_slug(raw_task.get("repo", ""))
        eng = engines.get(slug)
        builder = builders.get(slug)
        if not eng or not builder:
            logger.warning(f"Missing engine for repo {slug}; skipping task {iid}.")
            continue

        q_text = (raw_task.get("problem_statement") or "") + " " + (raw_task.get("hints_text") or "")

        # Candidate pools
        seeds, bm25_scores, p0 = builder.compute_seeds(q_text)
        seed_set = set(seeds)

        # 1-hop graph expansion
        e_graph = builder.compute_graph_expansion(seeds, p0)
        m_t = max(10, min(50, 10 + min(len(e_graph), 40)))

        # A1: S only (10)
        pool_a1 = list(seeds)

        # BM25 stream for A2
        bm25_stream = [
            i for i in sorted(
                range(builder.num_nodes),
                key=lambda idx_n: (-bm25_scores[idx_n], builder.symbols[idx_n].get("identifier", "")),
            )
            if i not in seed_set and is_filter_f_passing(builder.symbols[i], i, builder.is_test_mask, builder.hub_set)
        ]
        needed_a2 = m_t - 10
        pool_a2 = list(seeds) + bm25_stream[:needed_a2]

        # A4: Graph 1-hop expansion
        pool_a4 = list(seeds) + [item[0] for item in e_graph[:needed_a2]]

        # B0: Top-50 BM25
        b0_cands = [
            i for i in sorted(
                range(builder.num_nodes),
                key=lambda idx_n: (-bm25_scores[idx_n], builder.symbols[idx_n].get("identifier", "")),
            )
            if is_filter_f_passing(builder.symbols[i], i, builder.is_test_mask, builder.hub_set)
        ][:50]

        # Union of candidate node indices
        cand_union_indices = sorted(list(set(pool_a1 + pool_a2 + pool_a4 + b0_cands)))

        # Build candidate entries with rendered breadcrumbs
        candidates_list: List[Dict[str, Any]] = []
        node_to_cand_idx: Dict[int, int] = {}

        for c_idx, node_u in enumerate(cand_union_indices):
            node_to_cand_idx[node_u] = c_idx
            sym = eng.symbols[node_u]
            sym_id = sym.get("id", node_u)
            ident = sym.get("identifier", "unknown")
            fpath = sym.get("file_path", "")
            l_start = sym.get("line_start", 1)
            l_end = sym.get("line_end", l_start)

            # Pre-render breadcrumb skeleton
            bc = renderer.extract_breadcrumb(sym, slug, mode="skeleton")
            pt_score = float(scores_map.get((iid, node_u), -999.0))
            fmt_mass = float(mass_map.get((iid, node_u), 0.0))
            n_tokens = int(tokens_map.get((iid, node_u), 0))
            is_gold = bool(sym_id in gold_ids or node_u in gold_ids)

            candidates_list.append({
                "cand_index": c_idx,
                "node_idx": node_u,
                "symbol_id": sym_id,
                "identifier": ident,
                "file_path": fpath,
                "line_start": l_start,
                "line_end": l_end,
                "breadcrumb": bc,
                "pointwise_score": pt_score,
                "format_mass": fmt_mass,
                "n_prompt_tokens": n_tokens,
                "is_gold": is_gold,
            })

        # Pre-calculated candidate pools referencing candidate indices
        pools_cand_indices = {
            "A1": [node_to_cand_idx[u] for u in pool_a1],
            "A2": [node_to_cand_idx[u] for u in pool_a2],
            "A4": [node_to_cand_idx[u] for u in pool_a4],
            "B0": [node_to_cand_idx[u] for u in b0_cands],
        }

        # Official ranked symbol IDs
        task_rankings = {
            arm: official_rankings.get(arm, {}).get(iid, [])
            for arm in ("B0", "A1", "A2", "A4")
        }

        # Track top-5 hits
        for arm in ("B0", "A1", "A2", "A4"):
            ranked = task_rankings[arm]
            if any(sid in gold_ids for sid in ranked[:5]):
                hits_counter[arm] += 1

        bundle_tasks.append({
            "instance_id": iid,
            "repo": raw_task.get("repo", ""),
            "base_commit": raw_task.get("base_commit", ""),
            "problem_statement": raw_task.get("problem_statement", ""),
            "hints_text": raw_task.get("hints_text", ""),
            "gold_symbol_ids": sorted(list(gold_ids)),
            "gold_symbol_names": gold_names,
            "gold_files": gold_files,
            "candidates": candidates_list,
            "pools": pools_cand_indices,
            "official_ranked_symbol_ids": task_rankings,
        })

    logger.info(f"Processed {len(bundle_tasks)} Stratum C tasks.")
    logger.info(f"Verification of Top-5 Hits on N=131:")
    logger.info(f"  B0 (BM25 Top-5): {hits_counter['B0']}/131 ({hits_counter['B0']/131*100:.1f}%) [Target: 20/131, 15.3%]")
    logger.info(f"  A1 (Seeds Only): {hits_counter['A1']}/131 ({hits_counter['A1']/131*100:.1f}%) [Target: 18/131, 13.7%]")
    logger.info(f"  A2 (Depth BM25): {hits_counter['A2']}/131 ({hits_counter['A2']/131*100:.1f}%) [Target: 11/131, 8.4%]")
    logger.info(f"  A4 (Graph 1-Hop): {hits_counter['A4']}/131 ({hits_counter['A4']/131*100:.1f}%) [Target: 16/131, 12.2%]")

    assert hits_counter["B0"] == 20, f"B0 hits mismatch: {hits_counter['B0']} != 20"
    assert hits_counter["A1"] == 18, f"A1 hits mismatch: {hits_counter['A1']} != 18"
    assert hits_counter["A2"] == 11, f"A2 hits mismatch: {hits_counter['A2']} != 11"
    assert hits_counter["A4"] == 16, f"A4 hits mismatch: {hits_counter['A4']} != 16"

    bundle = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "metadata": {
            "title": "Stratum C (Un-Named Tasks) Evaluation Bundle",
            "author": "Sultan Haikal (GitHub: @yvliet)",
            "license": "Apache-2.0",
            "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_tasks": len(bundle_tasks),
            "target_model_family": "Google Gemma 4 (E4B / 26B A4B)",
            "description": (
                "Self-contained evaluation bundle containing pre-rendered candidate AST "
                "breadcrumbs, candidate pool partitions, gold labels, and pre-computed pointwise "
                "scores for all 131 Stratum C un-named defect tasks from SWE-bench Verified."
            ),
            "verification_summary": {
                "b0_hits_at_5": hits_counter["B0"],
                "a1_hits_at_5": hits_counter["A1"],
                "a2_hits_at_5": hits_counter["A2"],
                "a4_hits_at_5": hits_counter["A4"],
            },
        },
        "tasks": bundle_tasks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=1)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    logger.info(f"Bundle successfully written to {output_path} ({file_size_mb:.2f} MB) in {elapsed:.2f}s.")

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stratum C Evaluation Bundle.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_BUNDLE,
        help="Target output bundle JSON file path.",
    )
    args = parser.parse_args()
    build_stratum_c_bundle(output_path=args.output)


if __name__ == "__main__":
    main()
