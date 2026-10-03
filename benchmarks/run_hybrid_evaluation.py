#!/usr/bin/env python3
"""Complementarity and budget-matched fusion controls for BM25 and Perron Static.

For each task, records whether a gold callable appears in: BM25 top-10 and top-20,
Perron Static top-10 and top-20, the union of both top-10 lists (an oracle over
up to 20 slots), RRF (k0 = 60) top-10, and weighted fusion at the dev-selected w*.
Tests: union vs BM25@20 (budget-matched), Perron vs BM25@10, and the two
pre-registered fusion variants vs BM25@10 with Holm correction (m = 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from benchmarks.eval_harness import compute_exact_mcnemar_test, compute_holm_bonferroni_correction  # noqa: E402
from benchmarks.run_beta_sweep import build_engines, iter_tasks, load_graphs, load_split, now_utc, query_text  # noqa: E402
from perron.fusion import reciprocal_rank_fusion, weighted_score_fusion  # noqa: E402
from perron.metrics import function_acc_at_k  # noqa: E402
from perron.results import get_git_sha  # noqa: E402

SEED = 20261003
VARIANTS = ("bm25_at10", "bm25_at20", "perron_at10", "perron_at20", "union_at10x2", "rrf60_at10", "weighted_wstar_at10")


def first_gold_rank(ids: List[Any], gold: List[Any]) -> int:
    gs = set(gold)
    for pos, sid in enumerate(ids, 1):
        if sid in gs:
            return pos
    return -1


def run(split: str) -> Path:
    w_star = float(json.loads((PROJECT_ROOT / "results" / "dev_val_fusion.json").read_text(encoding="utf-8"))["selected_w"])
    tasks, labels = load_split(split)
    engines = build_engines(load_graphs(), 0.85)
    inst: Dict[str, Dict[str, Any]] = {}
    for iid, task, gold, engine in iter_tasks(tasks, labels, engines):
        gold_ids = gold.get("gold_symbol_ids", [])
        text = query_text(task)
        bm25_scores = engine.bm25_index.query(text)
        p_0 = engine.compute_query_prior(text)
        pi_query = engine.compute_personalized_pagerank(p_0)
        perron_scores = pi_query / np.power(engine.pi_global + 1e-8, 0.70)
        r_bm25 = [int(i) for i in np.argsort(-bm25_scores, kind="stable")]
        r_perron = [int(i) for i in np.argsort(-perron_scores, kind="stable")]
        r_rrf = reciprocal_rank_fusion(r_bm25, r_perron, k=60, total_nodes=engine.num_nodes)
        r_w, _ = weighted_score_fusion(bm25_scores, perron_scores, w=w_star)
        ids = {name: [engine.symbols[i]["id"] for i in r] for name, r in
               (("bm25", r_bm25), ("perron", r_perron), ("rrf", r_rrf), ("w", r_w))}
        rec = {
            "callable": bool(gold_ids),
            "bm25_at10": function_acc_at_k(ids["bm25"], gold_ids, k=10),
            "bm25_at20": function_acc_at_k(ids["bm25"], gold_ids, k=20),
            "perron_at10": function_acc_at_k(ids["perron"], gold_ids, k=10),
            "perron_at20": function_acc_at_k(ids["perron"], gold_ids, k=20),
            "rrf60_at10": function_acc_at_k(ids["rrf"], gold_ids, k=10),
            "weighted_wstar_at10": function_acc_at_k(ids["w"], gold_ids, k=10),
            "bm25_gold_rank": first_gold_rank(ids["bm25"], gold_ids) if gold_ids else -1,
        }
        rec["union_at10x2"] = float(rec["bm25_at10"] > 0 or rec["perron_at10"] > 0)
        inst[iid] = rec
    iids = sorted(inst)
    s_func = [i for i in iids if inst[i]["callable"]]
    col = lambda v: [inst[i][v] for i in s_func]  # noqa: E731
    counts = {v: {"k": int(sum(1 for i in iids if inst[i][v] > 0)), "n_all": len(iids), "n_callable": len(s_func)}
              for v in VARIANTS}
    b, p = col("bm25_at10"), col("perron_at10")
    contingency = {
        "both": int(sum(1 for x, y in zip(b, p) if x > 0 and y > 0)),
        "bm25_only": int(sum(1 for x, y in zip(b, p) if x > 0 and y == 0)),
        "perron_only": int(sum(1 for x, y in zip(b, p) if x == 0 and y > 0)),
        "neither": int(sum(1 for x, y in zip(b, p) if x == 0 and y == 0)),
    }
    tests = {
        "union_vs_bm25_at20": compute_exact_mcnemar_test(col("union_at10x2"), col("bm25_at20")),
        "perron_vs_bm25_at10": compute_exact_mcnemar_test(p, b),
        "rrf60_vs_bm25_at10": compute_exact_mcnemar_test(col("rrf60_at10"), b),
        "weighted_wstar_vs_bm25_at10": compute_exact_mcnemar_test(col("weighted_wstar_at10"), b),
    }
    holm = compute_holm_bonferroni_correction({k: tests[k]["p_value"] for k in ("rrf60_vs_bm25_at10", "weighted_wstar_vs_bm25_at10")})
    for k, v in holm.items():
        tests[k]["p_adjusted_holm"] = v
    perron_only = {i: inst[i]["bm25_gold_rank"] for i in s_func if inst[i]["perron_at10"] > 0 and inst[i]["bm25_at10"] == 0}
    out = {
        "suite": "hybrid_eval", "split": split, "seed": SEED, "selected_w": w_star, "rrf_k": 60,
        "num_instances": len(iids), "num_callable_instances": len(s_func),
        "counts": counts, "contingency_s_func": contingency, "tests": tests,
        "perron_only_bm25_rank": perron_only,
        "perron_only_bm25_ranks_sorted": sorted(perron_only.values()),
        "perron_only_beyond_bm25_top20": int(sum(1 for r in perron_only.values() if r > 20 or r == -1)),
        "note": "union_at10x2 uses up to 20 slots; compare it only with bm25_at20.",
        "instances": inst, "git_sha": get_git_sha(), "utc_timestamp": now_utc(),
    }
    path = PROJECT_ROOT / "results" / f"{split}_hybrid_eval.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT).as_posix()}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Perron complementarity and budget-matched fusion controls")
    parser.add_argument("--split", choices=["dev_val", "heldout"], required=True)
    run(parser.parse_args().split)


if __name__ == "__main__":
    main()
