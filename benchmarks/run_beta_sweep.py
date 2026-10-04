#!/usr/bin/env python3
"""Dev-split damping sweep and single-shot held-out evaluation at the selected beta.

--split dev_val : evaluates every graph method for each beta in BETAS on dev_val
                  (N=100), selects beta* per method (max dev Fn Acc@10, then max
                  dev Fn Acc@5, then larger beta), checks the beta = 0 identity,
                  and writes results/dev_val_beta_sweep.json.
--split heldout : reads beta* from results/dev_val_beta_sweep.json, evaluates each
                  method once on the frozen held-out split (N=150), and writes
                  results/heldout_tuned_baselines.json.
A fresh engine is built per beta because pi_global depends on beta.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine  # noqa: E402
from benchmarks.eval_harness import (  # noqa: E402
    compute_bootstrap_ci,
    compute_exact_mcnemar_test,
    compute_holm_bonferroni_correction,
)
from benchmarks.gold_patch_parser import get_repo_slug  # noqa: E402
from perron.metrics import file_acc_at_k, function_acc_at_k, hsi_at_10  # noqa: E402
from perron.results import get_git_sha  # noqa: E402
from perron.specificity import calculate_specificity_scores, compute_global_pagerank  # noqa: E402

SEED = 20261003
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85)
BETAS_FACTORIAL = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)
GAMMAS_FACTORIAL = (0.3, 0.5, 0.7, 0.85, 1.0)
SWEEP_METHODS = ("standard_ppr", "aider", "blocklist_lex", "deg_discount_matrix",
                 "deg_ppr", "hipporag", "query_reweighted_ppr", "perron_static")
FIXED_METHODS = ("bm25", "bm25_1hop", "dense", "prior_only", "oracle")
IDENTITY_METHODS = ("standard_ppr", "deg_discount_matrix", "query_reweighted_ppr", "hipporag", "perron_static")
HOLM_FAMILY = ("bm25", "bm25_1hop", "dense", "standard_ppr", "aider", "blocklist_lex",
               "deg_discount_matrix", "deg_ppr", "hipporag", "query_reweighted_ppr")
GRAPHS_DIR = PROJECT_ROOT / "artifacts" / "swebench_graphs"
SPLIT_KEYS = {"dev_val": ("dev_val_instances", 100), "heldout": ("heldout_instances", 150)}
DEV_OUT = PROJECT_ROOT / "results" / "dev_val_beta_sweep.json"
HELDOUT_OUT = PROJECT_ROOT / "results" / "heldout_tuned_baselines.json"
FACTORIAL_OUT = PROJECT_ROOT / "results" / "exploratory_factorial_sweep.json"


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bkey(b: float) -> str:
    return f"beta_{b:.2f}"


def load_split(split: str):
    key, expected = SPLIT_KEYS[split]
    split_data = json.loads((PROJECT_ROOT / "data" / "swebench_lite_split.json").read_text(encoding="utf-8"))
    ids = set(split_data[key])
    if len(ids) != expected:
        raise SystemExit(f"{split}: expected {expected} instances, found {len(ids)}")
    with open(PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    tasks = sorted((t for t in tasks if t.get("instance_id") in ids), key=lambda t: t["instance_id"])
    labels = json.loads((PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json").read_text(encoding="utf-8"))
    return tasks, labels


def load_graphs() -> Dict[str, Any]:
    graphs = {}
    for p in sorted(GRAPHS_DIR.glob("*_graph.npz")):
        slug = p.stem.replace("_graph", "")
        sym_p = GRAPHS_DIR / f"{slug}_symbols.json"
        if sym_p.exists():
            graphs[slug] = (sp.load_npz(p), json.loads(sym_p.read_text(encoding="utf-8")))
    if not graphs:
        raise SystemExit("no graphs under artifacts/swebench_graphs")
    return graphs


def build_engines(graphs: Dict[str, Any], beta: float) -> Dict[str, RepositoryRetrievalEngine]:
    return {slug: RepositoryRetrievalEngine(g, syms, beta=beta, gamma=0.70) for slug, (g, syms) in graphs.items()}


def query_text(task: Dict[str, Any]) -> str:
    return (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")


def evaluate_methods(engine, task, gold, methods: Sequence[str], keep_top10: bool = False) -> Dict[str, Dict[str, Any]]:
    text = query_text(task)
    gold_ids = gold.get("gold_symbol_ids", [])
    gold_files = gold.get("gold_files", [])
    p_0 = engine.compute_query_prior(text)
    pi_query = engine.compute_personalized_pagerank(p_0)
    out: Dict[str, Dict[str, Any]] = {}
    for m in methods:
        ranking = engine.retrieve_by_name(method_name=m, query_text=text, p_0=p_0,
                                          pi_query=pi_query, gold_symbol_ids=gold_ids)
        ids = [engine.symbols[i]["id"] for i in ranking]
        files: List[str] = []
        seen = set()
        for i in ranking:
            fp = engine.symbols[i].get("file_path", "")
            if fp and fp not in seen:
                seen.add(fp)
                files.append(fp)
            if len(files) >= 5:
                break
        rec = {
            "fn_acc_5": function_acc_at_k(ids, gold_ids, k=5),
            "fn_acc_10": function_acc_at_k(ids, gold_ids, k=10),
            "file_acc_5": file_acc_at_k(files, gold_files, k=5),
            "hsi": hsi_at_10(ids, engine.hub_set),
        }
        if keep_top10:
            rec["top10"] = [int(i) for i in ranking[:10]]
        out[m] = rec
    return out


def iter_tasks(tasks, labels, engines):
    for task in tasks:
        iid = task["instance_id"]
        engine = engines.get(get_repo_slug(task.get("repo", "")))
        gold = labels.get(iid)
        if engine is None or not gold:
            continue
        yield iid, task, gold, engine


def count(values: Sequence[float]) -> Dict[str, Any]:
    k = int(sum(1 for v in values if v > 0))
    n = len(values)
    return {"k": k, "n": n, "mean": k / n if n else 0.0}


def run_dev() -> None:
    tasks, labels = load_split("dev_val")
    graphs = load_graphs()
    sweep: Dict[str, Dict[str, Any]] = defaultdict(dict)
    identity: Dict[str, Dict[str, Any]] = {}
    reference: Dict[str, Any] = {}
    n_tasks = None
    for b in BETAS:
        engines = build_engines(graphs, b)
        methods = SWEEP_METHODS + (("prior_only",) if b == 0.0 else ())
        per_task = {}
        for iid, task, gold, engine in iter_tasks(tasks, labels, engines):
            per_task[iid] = evaluate_methods(engine, task, gold, methods, keep_top10=(b == 0.0))
        n_tasks = len(per_task)
        for m in SWEEP_METHODS:
            sweep[m][bkey(b)] = {
                "fn_acc_10": count([r[m]["fn_acc_10"] for r in per_task.values()]),
                "fn_acc_5": count([r[m]["fn_acc_5"] for r in per_task.values()]),
            }
        if b == 0.0:
            reference["prior_only"] = {"fn_acc_10": count([r["prior_only"]["fn_acc_10"] for r in per_task.values()])}
            for m in IDENTITY_METHODS:
                set_eq = sum(1 for r in per_task.values() if set(r[m]["top10"]) == set(r["prior_only"]["top10"]))
                hit_eq = sum(1 for r in per_task.values() if r[m]["fn_acc_10"] == r["prior_only"]["fn_acc_10"])
                identity[m] = {"top10_set_equal_tasks": set_eq, "fn_acc_10_equal_tasks": hit_eq}
        del engines
    beta_star = {
        m: max(BETAS, key=lambda b: (sweep[m][bkey(b)]["fn_acc_10"]["k"], sweep[m][bkey(b)]["fn_acc_5"]["k"], b))
        for m in SWEEP_METHODS
    }
    out = {
        "suite": "beta_sweep", "split": "dev_val", "num_instances": n_tasks, "seed": SEED,
        "betas": list(BETAS), "selection_rule": "max(dev fn_acc_10 hits, dev fn_acc_5 hits, beta)",
        "sweep": sweep, "beta_star": beta_star, "identity_methods": list(IDENTITY_METHODS),
        "identity_check": identity, "reference": reference,
        "git_sha": get_git_sha(), "utc_timestamp": now_utc(),
    }
    DEV_OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    bad = [m for m, c in identity.items() if c["top10_set_equal_tasks"] != n_tasks or c["fn_acc_10_equal_tasks"] != n_tasks]
    if n_tasks != 100 or bad:
        raise SystemExit(f"dev gate failed: n_tasks={n_tasks}, identity mismatches={bad}")
    print(f"beta*: {beta_star}")


def summarize(per_task: Dict[str, Dict[str, Any]], m: str, iids: Sequence[str]) -> Dict[str, Any]:
    res = {}
    for metric in ("fn_acc_10", "file_acc_5", "hsi"):
        vals = [float(per_task[i][m][metric]) for i in iids]
        mean, lo, hi = compute_bootstrap_ci(vals, n_bootstrap=10000, ci=0.95, seed=SEED)
        entry = {"n": len(vals), "mean": mean, "ci_lower": lo, "ci_upper": hi}
        if metric == "fn_acc_10":
            entry["k"] = int(sum(1 for v in vals if v > 0))
        res[metric] = entry
    return res


def run_heldout() -> None:
    dev_bytes = DEV_OUT.read_bytes()
    dev = json.loads(dev_bytes)
    beta_star = {m: float(b) for m, b in dev["beta_star"].items()}
    tasks, labels = load_split("heldout")
    graphs = load_graphs()
    groups: Dict[float, List[str]] = defaultdict(list)
    for m in SWEEP_METHODS:
        groups[beta_star[m]].append(m)
    groups[0.85].extend(FIXED_METHODS)
    per_task: Dict[str, Dict[str, Any]] = defaultdict(dict)
    callable_ids = set()
    for b, methods in sorted(groups.items()):
        engines = build_engines(graphs, b)
        for iid, task, gold, engine in iter_tasks(tasks, labels, engines):
            per_task[iid].update(evaluate_methods(engine, task, gold, methods))
            if gold.get("gold_symbol_ids"):
                callable_ids.add(iid)
        del engines
    all_ids = sorted(per_task)
    s_func = sorted(callable_ids)
    methods = list(SWEEP_METHODS) + list(FIXED_METHODS)
    summary = {m: summarize(per_task, m, all_ids) for m in methods}
    summary_s_func = {m: summarize(per_task, m, s_func) for m in methods}
    for m in methods:
        summary_s_func[m]["beta"] = beta_star.get(m)
        summary[m]["beta"] = beta_star.get(m)
    target = [per_task[i]["perron_static"]["fn_acc_10"] for i in s_func]
    pairwise = {m: {"mcnemar": compute_exact_mcnemar_test(target, [per_task[i][m]["fn_acc_10"] for i in s_func])}
                for m in HOLM_FAMILY}
    holm = compute_holm_bonferroni_correction({m: pairwise[m]["mcnemar"]["p_value"] for m in HOLM_FAMILY})
    for m in HOLM_FAMILY:
        pairwise[m]["p_adjusted_holm"] = holm[m]
    out = {
        "suite": "tuned_baselines", "split": "heldout", "seed": SEED,
        "num_instances": len(all_ids), "num_callable_instances": len(s_func),
        "dev_source": "results/dev_val_beta_sweep.json",
        "dev_source_sha256": hashlib.sha256(dev_bytes).hexdigest(),
        "dev_source_utc_timestamp": dev["utc_timestamp"],
        "beta_star": beta_star, "holm_family": list(HOLM_FAMILY), "holm_family_size": len(HOLM_FAMILY),
        "summary": summary, "summary_s_func": summary_s_func,
        "pairwise_vs_perron_static": pairwise,
        "instances": {i: {m: {k: per_task[i][m][k] for k in ("fn_acc_10", "file_acc_5", "hsi")} for m in methods}
                      for i in all_ids},
        "git_sha": get_git_sha(), "utc_timestamp": now_utc(),
    }
    HELDOUT_OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {HELDOUT_OUT.relative_to(PROJECT_ROOT).as_posix()}")


def run_factorial(
    output_path: Path = FACTORIAL_OUT,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Gate G1 Factorial Sweep:
    Evaluates 2x2 factorial combinations:
      Prior Dispersion: Dense Linear Prior vs. Sharp Top-25 Softmax Prior (tau=0.15)
      Graph Topology: Raw AST Graph (test nodes unmasked) vs. Sanitized Graph (test nodes masked)
    across:
      betas_q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85]
      gammas in [0.3, 0.5, 0.7, 0.85, 1.0]
      betas_g in [beta_q, 0.85]
    over all 300 exploratory SWE-bench Lite instances.
    """
    print(f"[FACTORIAL SWEEP] Loading graphs and tasks (limit={limit})...")
    with open(PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit:
        tasks = tasks[:limit]

    labels = json.loads((PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json").read_text(encoding="utf-8"))
    graphs = load_graphs()

    # Precompute global PageRank vectors per repo slug
    global_pagerank_085: Dict[str, np.ndarray] = {}
    global_pagerank_by_beta: Dict[str, Dict[float, np.ndarray]] = defaultdict(dict)
    for slug, (g, syms) in graphs.items():
        dangling = np.zeros(g.shape[0], dtype=np.float32)
        global_pagerank_085[slug] = compute_global_pagerank(g, dangling, beta=0.85)
        for b in BETAS_FACTORIAL:
            global_pagerank_by_beta[slug][b] = compute_global_pagerank(g, dangling, beta=b)

    # Initialize engines once across all repositories
    engines = {
        slug: RepositoryRetrievalEngine(
            g, syms,
            beta=0.85, gamma=0.70,
            tau=0.15,
            k_prior=25,
        )
        for slug, (g, syms) in graphs.items()
    }

    # Precompute per-task data and query priors (linear and softmax)
    task_data = []
    n_callable = 0
    for task in tasks:
        iid = task["instance_id"]
        slug = get_repo_slug(task.get("repo", ""))
        eng = engines.get(slug)
        gold = labels.get(iid)
        if eng is None or not gold:
            continue
        gold_ids = set(gold.get("gold_symbol_ids", []))
        gold_files = set(gold.get("gold_files", []))
        text = query_text(task)
        has_func = len(gold_ids) > 0
        if has_func:
            n_callable += 1
        p_0_linear = eng.compute_query_prior(text, mode="linear")
        p_0_softmax = eng.compute_query_prior(text, mode="softmax", k=25, tau=0.15)
        task_data.append((iid, slug, eng, text, gold_ids, gold_files, has_func, p_0_linear, p_0_softmax))

    conditions = [
        ("linear__raw", "linear", False),
        ("linear__sanitized", "linear", True),
        ("softmax__raw", "softmax", False),
        ("softmax__sanitized", "softmax", True),
    ]

    factorial_results: Dict[str, Any] = {}
    n_total = len(task_data)

    for cond_name, pmode, filter_tests in conditions:
        print(f"\n[FACTORIAL CONDITION: {cond_name}] (prior={pmode}, filter_test_nodes={filter_tests})", flush=True)
        # Configure test filtering on all engines
        for eng in engines.values():
            eng.filter_test_nodes = filter_tests

        cond_data: Dict[str, Any] = {
            "bm25": {"fn_acc_10": 0, "fn_acc_5": 0, "file_acc_5": 0, "test_pct_10": 0.0},
            "by_beta": {},
        }

        # Evaluate BM25
        bm25_hits_10 = 0
        bm25_hits_5 = 0
        bm25_file_hits = 0
        bm25_test_nodes = 0
        for iid, slug, eng, text, gold_ids, gold_files, has_func, p_lin, p_soft in task_data:
            ranking = eng.retrieve_bm25(text)
            top10 = ranking[:10]
            top5 = ranking[:5]
            if has_func:
                if any(eng.symbols[i]["id"] in gold_ids for i in top10):
                    bm25_hits_10 += 1
                if any(eng.symbols[i]["id"] in gold_ids for i in top5):
                    bm25_hits_5 += 1
            top5_files = {eng.symbols[i].get("file_path", "") for i in top5}
            if top5_files & gold_files:
                bm25_file_hits += 1
            if eng.is_test_mask is not None:
                bm25_test_nodes += sum(1 for i in top10 if eng.is_test_mask[i])

        cond_data["bm25"] = {
            "fn_acc_10": bm25_hits_10 / max(1, n_callable),
            "fn_acc_5": bm25_hits_5 / max(1, n_callable),
            "file_acc_5": bm25_file_hits / max(1, n_total),
            "test_pct_10": (bm25_test_nodes / (n_total * 10)) * 100.0,
            "hits_10": bm25_hits_10,
            "total_callable": n_callable,
        }

        # Grid evaluation across betas
        for b_q in BETAS_FACTORIAL:
            b_key = f"beta_{b_q:.2f}"
            beta_record: Dict[str, Any] = {
                "standard_ppr": {},
                "deg_ppr": {},
                "perron_decoupled": {},
                "perron_coupled": {},
                "perron_adaptive": {},
                "hipporag": {},
            }

            std_hits_10 = 0
            std_hits_5 = 0
            std_file_hits = 0
            std_test_nodes = 0

            deg_hits = {g: 0 for g in GAMMAS_FACTORIAL}
            perron_dec_hits = {g: 0 for g in GAMMAS_FACTORIAL}
            perron_coup_hits = {g: 0 for g in GAMMAS_FACTORIAL}
            perron_adapt_hits = {g: 0 for g in GAMMAS_FACTORIAL}
            hippo_hits = {g: 0 for g in (0.5, 0.7)}

            for iid, slug, eng, text, gold_ids, gold_files, has_func, p_lin, p_soft in task_data:
                p_0 = p_soft if pmode == "softmax" else p_lin
                pi_q = eng.compute_personalized_pagerank(p_0, beta=b_q)
                pi_g_085 = global_pagerank_085[slug]
                pi_g_coup = global_pagerank_by_beta[slug][b_q]

                # Standard PPR
                std_rank = eng._rank_scores(pi_q)
                std_top10 = std_rank[:10]
                std_top5 = std_rank[:5]
                if has_func:
                    if any(eng.symbols[i]["id"] in gold_ids for i in std_top10):
                        std_hits_10 += 1
                    if any(eng.symbols[i]["id"] in gold_ids for i in std_top5):
                        std_hits_5 += 1
                top5_files = {eng.symbols[i].get("file_path", "") for i in std_top5}
                if top5_files & gold_files:
                    std_file_hits += 1
                if eng.is_test_mask is not None:
                    std_test_nodes += sum(1 for i in std_top10 if eng.is_test_mask[i])

                # Gammas
                for gamma in GAMMAS_FACTORIAL:
                    # Degree normalized
                    denom_deg = (eng.in_degrees + 1.0) ** gamma
                    deg_rank = eng._rank_scores(pi_q / denom_deg)
                    if has_func and any(eng.symbols[i]["id"] in gold_ids for i in deg_rank[:10]):
                        deg_hits[gamma] += 1

                    # Perron Decoupled (beta_g = 0.85)
                    s_dec = calculate_specificity_scores(
                        pi_q, pi_g_085, gamma=gamma,
                        is_test_node=eng.is_test_mask,
                        filter_test_nodes=filter_tests,
                    )
                    s_dec_rank = eng._rank_scores(s_dec)
                    if has_func and any(eng.symbols[i]["id"] in gold_ids for i in s_dec_rank[:10]):
                        perron_dec_hits[gamma] += 1

                    # Perron Coupled (beta_g = beta_q)
                    s_coup = calculate_specificity_scores(
                        pi_q, pi_g_coup, gamma=gamma,
                        is_test_node=eng.is_test_mask,
                        filter_test_nodes=filter_tests,
                    )
                    s_coup_rank = eng._rank_scores(s_coup)
                    if has_func and any(eng.symbols[i]["id"] in gold_ids for i in s_coup_rank[:10]):
                        perron_coup_hits[gamma] += 1

                    # Perron Adaptive (beta_g = 0.85)
                    s_adapt = calculate_specificity_scores(
                        pi_q, pi_g_085, gamma=gamma,
                        is_test_node=eng.is_test_mask,
                        filter_test_nodes=filter_tests,
                        query_prior=p_0,
                        hub_indices=eng.hub_indices,
                    )
                    s_adapt_rank = eng._rank_scores(s_adapt)
                    if has_func and any(eng.symbols[i]["id"] in gold_ids for i in s_adapt_rank[:10]):
                        perron_adapt_hits[gamma] += 1

                    # HippoRAG (canonical gamma=0.70)
                    if gamma in (0.5, 0.7):
                        denom_h = (pi_g_085 + 1e-8) ** gamma
                        p0_h = p_0 / denom_h
                        s_h = float(np.sum(p0_h))
                        if s_h > 0:
                            p0_h = p0_h / s_h
                        pi_h = eng.compute_personalized_pagerank(p0_h, beta=b_q)
                        h_rank = eng._rank_scores(pi_h)
                        if has_func and any(eng.symbols[i]["id"] in gold_ids for i in h_rank[:10]):
                            hippo_hits[gamma] += 1

            beta_record["standard_ppr"] = {
                "fn_acc_10": std_hits_10 / max(1, n_callable),
                "fn_acc_5": std_hits_5 / max(1, n_callable),
                "file_acc_5": std_file_hits / max(1, n_total),
                "test_pct_10": (std_test_nodes / (n_total * 10)) * 100.0,
                "hits_10": std_hits_10,
            }
            beta_record["deg_ppr"] = {f"gamma_{g:.2f}": deg_hits[g] / max(1, n_callable) for g in GAMMAS_FACTORIAL}
            beta_record["perron_decoupled"] = {f"gamma_{g:.2f}": perron_dec_hits[g] / max(1, n_callable) for g in GAMMAS_FACTORIAL}
            beta_record["perron_coupled"] = {f"gamma_{g:.2f}": perron_coup_hits[g] / max(1, n_callable) for g in GAMMAS_FACTORIAL}
            beta_record["perron_adaptive"] = {f"gamma_{g:.2f}": perron_adapt_hits[g] / max(1, n_callable) for g in GAMMAS_FACTORIAL}
            beta_record["hipporag"] = {f"gamma_{g:.2f}": hippo_hits[g] / max(1, n_callable) for g in (0.5, 0.7)}

            cond_data["by_beta"][b_key] = beta_record
            print(f"  beta={b_q:.2f} -> Standard PPR: {std_hits_10}/{n_callable} ({std_hits_10/max(1, n_callable)*100:.1f}%), Perron Decoupled (g=0.70): {perron_dec_hits[0.70]}/{n_callable} ({perron_dec_hits[0.70]/max(1, n_callable)*100:.1f}%), Test nodes: {beta_record['standard_ppr']['test_pct_10']:.1f}%")

        factorial_results[cond_name] = cond_data

    out = {
        "suite": "exploratory_factorial_sweep",
        "seed": SEED,
        "num_instances": len(tasks),
        "num_callable_instances": n_callable,
        "betas": list(BETAS_FACTORIAL),
        "gammas": list(GAMMAS_FACTORIAL),
        "conditions": [c[0] for c in conditions],
        "results": factorial_results,
        "git_sha": get_git_sha(),
        "utc_timestamp": now_utc(),
    }
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\n[FACTORIAL SWEEP COMPLETE] Telemetry saved to {out_p}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Perron dev damping sweep, held-out evaluation, and factorial sweep")
    parser.add_argument("--split", choices=["dev_val", "heldout", "factorial"], default=None)
    parser.add_argument("--factorial", action="store_true", help="Run 2x2 factorial exploratory sweep")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks for fast runs")
    args = parser.parse_args()

    if args.factorial or args.split == "factorial":
        run_factorial(limit=args.limit)
    elif args.split == "dev_val":
        run_dev()
    elif args.split == "heldout":
        run_heldout()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

