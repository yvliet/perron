#!/usr/bin/env python3
"""In-degree of top-10 candidates vs gold targets on held-out callable tasks (beta = 0.85).

In-degree is the binary column count of the row-normalized transition matrix
(RepositoryRetrievalEngine.in_degrees). Writes results/heldout_top10_indegree.json.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from benchmarks.run_beta_sweep import build_engines, iter_tasks, load_graphs, load_split, now_utc, query_text  # noqa: E402
from perron.metrics import function_acc_at_k  # noqa: E402
from perron.results import get_git_sha  # noqa: E402

COLLAPSED = ("standard_ppr", "aider", "blocklist_lex", "deg_discount_matrix", "hipporag", "query_reweighted_ppr")
REFERENCE = ("perron_static", "bm25")
OUT = PROJECT_ROOT / "results" / "heldout_top10_indegree.json"


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "max": float(a.max())}


def main() -> None:
    tasks, labels = load_split("heldout")
    engines = build_engines(load_graphs(), 0.85)
    gold_deg = []
    top_deg = {m: [] for m in COLLAPSED + REFERENCE}
    hits = {m: 0 for m in COLLAPSED + REFERENCE}
    names = {m: Counter() for m in COLLAPSED + REFERENCE}
    name_deg = {}
    n_tasks = 0
    for iid, task, gold, engine in iter_tasks(tasks, labels, engines):
        gold_ids = gold.get("gold_symbol_ids", [])
        if not gold_ids:
            continue
        n_tasks += 1
        id_to_row = {s["id"]: i for i, s in enumerate(engine.symbols)}
        gold_deg += [float(engine.in_degrees[id_to_row[g]]) for g in gold_ids if g in id_to_row]
        text = query_text(task)
        p_0 = engine.compute_query_prior(text)
        pi_query = engine.compute_personalized_pagerank(p_0)
        for m in COLLAPSED + REFERENCE:
            ranking = engine.retrieve_by_name(method_name=m, query_text=text, p_0=p_0, pi_query=pi_query,
                                              gold_symbol_ids=gold_ids)
            ids = [engine.symbols[i]["id"] for i in ranking[:10]]
            hits[m] += int(function_acc_at_k(ids, gold_ids, k=10) > 0)
            for i in ranking[:10]:
                deg = float(engine.in_degrees[i])
                top_deg[m].append(deg)
                ident = engine.symbols[i].get("identifier", "")
                names[m][ident] += 1
                name_deg[ident] = max(name_deg.get(ident, 0.0), deg)
    methods = {}
    for m in COLLAPSED + REFERENCE:
        res = mannwhitneyu(top_deg[m], gold_deg, alternative="two-sided", method="asymptotic")
        rb = 2.0 * float(res.statistic) / (len(top_deg[m]) * len(gold_deg)) - 1.0
        methods[m] = {**stats(top_deg[m]), "fn_acc_10_hits": hits[m], "mannwhitney_u": float(res.statistic),
                      "mannwhitney_p": float(res.pvalue), "rank_biserial": rb,
                      "top_identifiers": [{"identifier": k, "count": c, "in_degree": name_deg[k]}
                                          for k, c in names[m].most_common(10)]}
    out = {"suite": "top10_indegree", "split": "heldout", "beta": 0.85, "num_tasks": n_tasks,
           "in_degree_definition": "binary column count of the row-normalized transition matrix",
           "gold": stats(gold_deg), "methods": methods, "git_sha": get_git_sha(), "utc_timestamp": now_utc()}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
