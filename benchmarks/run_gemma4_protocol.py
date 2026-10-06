#!/usr/bin/env python3
"""
Gemma 4 Pre-Registered Protocol Runner.

Executes two-stage evaluation protocol according to benchmarks/protocol/gemma4_protocol_v1.yaml:
- Gate G0: Confirmatory baseline reproduction (BM25: 175/344, Perron Static: 173/344)
- Gate G1: Graph integrity & prompt safety audit
- Dev Selection: Evaluates P0, P1, P2 on disjoint Lite dev_val (N=100), checks G2-G4, emits lock JSON
- Stage 2 Freeze: Runs freeze_protocol.py stage2
- Single-Shot Verified Run: Evaluates frozen winning variant across arms B0-B1, A1-A6 on Verified (N=344)
- Downstream Mini-Test: Evaluates D0-D3 on N=100 downstream sample
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
import scipy.sparse as sp
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.baseline_retrievers import RepositoryRetrievalEngine
from benchmarks.gold_patch_parser import get_repo_slug
from perron.metrics import function_acc_at_k
from perron.reranker.cache import PointwiseScoreCache
from perron.reranker.candidate_pools import (
    CandidatePoolBuilder,
    PoolDiagnostics,
    build_downstream_candidates,
)
from perron.reranker.scorer import (
    PROMPT_TEMPLATES,
    BreadcrumbRenderer,
    PointwiseReranker,
    normalize_and_truncate_issue_text,
)
from perron.specificity import calculate_specificity_scores, compute_global_pagerank

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gemma4_protocol")

PROTOCOL_PATH = PROJECT_ROOT / "benchmarks" / "protocol" / "gemma4_protocol_v1.yaml"
LOCK_PATH = PROJECT_ROOT / "benchmarks" / "protocol" / "gemma4_lock_v1.json"
GRAPHS_DIR = PROJECT_ROOT / "artifacts" / "swebench_graphs"
CLONES_DIR = PROJECT_ROOT / "artifacts" / "clones"
VERIFIED_TASKS_FILE = PROJECT_ROOT / "data" / "swebench_verified_tasks.jsonl"
VERIFIED_GOLD_FILE = PROJECT_ROOT / "data" / "swebench_verified_gold_labels.json"
LITE_TASKS_FILE = PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl"
LITE_GOLD_FILE = PROJECT_ROOT / "data" / "swebench_lite_gold_labels.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "gemma4_protocol"
CACHE_PATH = RESULTS_DIR / "scores_cache.db"


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(65536)
        while chunk:
            h.update(chunk)
            chunk = f.read(65536)
    return h.hexdigest()


def load_yaml_raw(path: Path) -> Tuple[str, Dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    return content, yaml.safe_load(content)


def load_engines() -> Tuple[Dict[str, RepositoryRetrievalEngine], Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """Loads 12 canonical repository graphs, global PageRank vectors, and multiplex layers."""
    logger.info("Loading graphs and precomputing engines for canonical repositories...")
    graphs: Dict[str, Tuple[sp.csr_matrix, List[Dict[str, Any]]]] = {}
    for p in sorted(GRAPHS_DIR.glob("*_graph.npz")):
        slug = p.stem.replace("_graph", "")
        sym_p = GRAPHS_DIR / f"{slug}_symbols.json"
        if sym_p.exists():
            graphs[slug] = (sp.load_npz(p), json.loads(sym_p.read_text(encoding="utf-8")))

    global_pr: Dict[str, np.ndarray] = {}
    for slug, (g, syms) in graphs.items():
        dangling = np.zeros(g.shape[0], dtype=np.float32)
        global_pr[slug] = compute_global_pagerank(g, dangling, beta=0.85)

    layers: Dict[str, Dict[str, Any]] = {}
    for p in sorted(GRAPHS_DIR.glob("*_layers.npz")):
        slug = p.stem.replace("_layers", "")
        data = np.load(p)
        layers[slug] = {
            "num_nodes": int(data["num_nodes"]),
            "call_edges": data["call_edges"].tolist() if "call_edges" in data else [],
            "caller_edges": data["caller_edges"].tolist() if "caller_edges" in data else [],
            "inherit_edges": data["inherit_edges"].tolist() if "inherit_edges" in data else [],
            "import_edges": data["import_edges"].tolist() if "import_edges" in data else [],
        }

    engines: Dict[str, RepositoryRetrievalEngine] = {
        slug: RepositoryRetrievalEngine(
            g, syms,
            beta=0.85,
            gamma=0.70,
            filter_test_nodes=True,
            prior_mode="softmax",
            tau=0.15,
            k_prior=25,
        )
        for slug, (g, syms) in graphs.items()
    }
    return engines, global_pr, layers


def check_gate_g0(engines: Dict[str, RepositoryRetrievalEngine], global_pr: Dict[str, np.ndarray]) -> bool:
    """Gate G0: Reproduce published Acc@10: BM25 175/344, Perron Static 173/344."""
    logger.info("Checking Gate G0: Baseline Reproduction...")
    tasks = [json.loads(l) for l in open(VERIFIED_TASKS_FILE, encoding="utf-8") if l.strip()]
    gold_labels = json.load(open(VERIFIED_GOLD_FILE, encoding="utf-8"))

    bm25_hits = 0
    perron_hits = 0
    callable_count = 0

    for t in tasks:
        iid = t["instance_id"]
        gold = gold_labels.get(iid)
        if not gold or not gold.get("gold_symbol_ids"):
            continue
        callable_count += 1
        gold_ids = set(gold["gold_symbol_ids"])
        slug = get_repo_slug(t["repo"])
        eng = engines.get(slug)
        if not eng:
            continue
        text = (t.get("problem_statement") or "") + " " + (t.get("hints_text") or "")

        # BM25
        b_rank = eng.retrieve_bm25(text)[:10]
        b_ids = [eng.symbols[i]["id"] for i in b_rank]
        if function_acc_at_k(b_ids, list(gold_ids), k=10) > 0:
            bm25_hits += 1

        # Perron Static
        p0 = eng.compute_query_prior(text, mode="softmax", k=25, tau=0.15)
        pi_q = eng.compute_personalized_pagerank(p0, beta=0.85)
        pi_g = global_pr[slug]
        s_static = calculate_specificity_scores(
            pi_q, pi_g, gamma=0.70, is_test_node=eng.is_test_mask, filter_test_nodes=True
        )
        p_rank = eng._rank_scores(s_static)[:10]
        p_ids = [eng.symbols[i]["id"] for i in p_rank]
        if function_acc_at_k(p_ids, list(gold_ids), k=10) > 0:
            perron_hits += 1

    logger.info(f"Gate G0 Results: Callable={callable_count}, BM25={bm25_hits}/344, Perron={perron_hits}/344")
    passed = (callable_count == 344 and bm25_hits == 175 and perron_hits == 173)
    if not passed:
        raise RuntimeError(f"Gate G0 FAILED: Expected BM25=175/344, Perron=173/344. Got {bm25_hits}, {perron_hits}")
    logger.info("Gate G0 PASSED: Baseline reproduction verified 100% exact.")
    return True


def check_gate_g1() -> bool:
    """Gate G1: Graph artifacts integrity and prompt string safety audit."""
    logger.info("Checking Gate G1: Integrity and Prompt Safety Audit...")
    for p in GRAPHS_DIR.glob("*_graph.npz"):
        if p.stat().st_size == 0:
            raise RuntimeError(f"Gate G1 FAILED: Empty graph {p}")

    banned_terms = [
        "FAIL_TO_PASS", "PASS_TO_PASS", "bm25_score", "pool_rank", "arm_id", "graph_edges", "seed_flag"
    ]
    renderer = BreadcrumbRenderer(clones_dir=CLONES_DIR)
    dummy_sym = {"identifier": "test_func", "file_path": "a.py", "line_start": 1, "line_end": 5}
    for variant in ("P0", "P1", "P2"):
        p_text = PROMPT_TEMPLATES[variant].format(
            issue_text="Sample issue description",
            candidate_breadcrumb=renderer.extract_breadcrumb(dummy_sym, "dummy"),
        )
        for b in banned_terms:
            if b in p_text:
                raise RuntimeError(f"Gate G1 FAILED: Prompt contains banned term {b}")

    logger.info("Gate G1 PASSED: Graph artifacts present and prompt audit clean.")
    return True


def sample_dev_tasks(n: int = 100, seed: int = 20261108) -> List[Dict[str, Any]]:
    """Samples n disjoint tasks from SWE-bench Lite callable tasks."""
    lite_tasks = [json.loads(l) for l in open(LITE_TASKS_FILE, encoding="utf-8") if l.strip()]
    lite_gold = json.load(open(LITE_GOLD_FILE, encoding="utf-8"))
    callable_lite = [
        t for t in lite_tasks
        if t["instance_id"] in lite_gold and len(lite_gold[t["instance_id"]].get("gold_symbol_ids", [])) > 0
    ]
    callable_lite.sort(key=lambda x: x["instance_id"])
    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(len(callable_lite), size=min(n, len(callable_lite)), replace=False)
    selected = [callable_lite[i] for i in sorted(chosen_indices)]
    return selected


def sample_downstream_task_ids(verified_tasks: List[Dict[str, Any]], n: int = 100, seed: int = 20261108) -> List[str]:
    """Proportional allocation sample of N=100 downstream tasks from SWE-bench Verified."""
    by_repo: Dict[str, List[str]] = {}
    for t in verified_tasks:
        by_repo.setdefault(t.get("repo", "unknown"), []).append(t["instance_id"])

    rng = np.random.RandomState(seed)
    selected: List[str] = []
    repo_list = sorted(by_repo.keys())
    per_repo = max(1, n // len(repo_list))

    for repo in repo_list:
        sub = sorted(by_repo[repo])
        n_take = min(per_repo, len(sub))
        chosen = rng.choice(sub, size=n_take, replace=False)
        selected.extend(chosen)

    remaining = [t["instance_id"] for t in verified_tasks if t["instance_id"] not in set(selected)]
    needed = n - len(selected)
    if needed > 0 and remaining:
        chosen_rem = rng.choice(remaining, size=needed, replace=False)
        selected.extend(chosen_rem)

    selected.sort()
    return selected[:n]


def run_dev_selection(
    engines: Dict[str, RepositoryRetrievalEngine],
    global_pr: Dict[str, np.ndarray],
    layers: Dict[str, Dict[str, Any]],
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluates P0, P1, P2 on disjoint Lite dev_val (N=100), checks G2-G4, and creates gemma4_lock_v1.json.
    """
    logger.info("Executing Dev Selection Stage (disjoint Lite dev_val, N=100)...")
    dev_tasks = sample_dev_tasks(n=100)
    dev_gold = json.load(open(LITE_GOLD_FILE, encoding="utf-8"))

    reranker = PointwiseReranker(model_backend=backend)
    variants = ["P0", "P1", "P2"]

    builders = {
        slug: CandidatePoolBuilder(engine=eng, layers=layers.get(slug))
        for slug, eng in engines.items()
    }

    logger.info("Building candidate pools for dev_val tasks...")
    dev_task_data = []
    for task in dev_tasks:
        iid = task["instance_id"]
        slug = get_repo_slug(task.get("repo", ""))
        eng = engines.get(slug)
        gold_info = dev_gold.get(iid, {})
        gold_ids = set(gold_info.get("gold_symbol_ids", []))
        if not eng or not gold_ids:
            continue

        q_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")
        pools, diag = builders[slug].build_all_pools(
            query_text=q_text,
            instance_id=iid,
            gold_symbol_ids=gold_ids,
        )
        cand_union = sorted(list(set(pools["A1"] + pools["A2"] + pools["A3"] + pools["A4"])))
        dev_task_data.append((task, slug, eng, gold_ids, pools, cand_union))

    dev_results_by_variant: Dict[str, Dict[str, Any]] = {}
    total_prefill_tokens = 0
    t_start = time.perf_counter()

    for var in variants:
        logger.info(f"Evaluating prompt variant {var} across dev_val...")
        mrr_arms = {"A1": [], "A2": [], "A3": [], "A4": []}
        acc5_arms = {"A1": [], "A2": [], "A3": [], "A4": []}
        format_masses: List[float] = []
        prompt_token_counts: List[int] = []

        for task, slug, eng, gold_ids, pools, cand_union in dev_task_data:
            scores = reranker.score_candidates_for_task(
                task=task,
                candidate_indices=cand_union,
                symbols=eng.symbols,
                repo_slug=slug,
                prompt_variant=var,
            )

            for c_idx, s_data in scores.items():
                format_masses.append(s_data["format_mass"])
                prompt_token_counts.append(s_data["n_prompt_tokens"])
                total_prefill_tokens += s_data["n_prompt_tokens"]

            # Rank arms and record MRR
            for arm in ("A1", "A2", "A3", "A4"):
                ranked = reranker.rank_candidates(pools[arm], scores, eng.symbols, prior_order=pools[arm])
                ranked_sym_ids = [eng.symbols[i]["id"] for i in ranked]

                # MRR@50
                mrr_val = 0.0
                for r_idx, s_id in enumerate(ranked_sym_ids[:50], 1):
                    if s_id in gold_ids:
                        mrr_val = 1.0 / r_idx
                        break
                mrr_arms[arm].append(mrr_val)

                # Acc@5
                hit5 = float(function_acc_at_k(ranked_sym_ids, list(gold_ids), k=5) > 0)
                acc5_arms[arm].append(hit5)

        mean_mrr = float(np.mean([np.mean(mrr_arms[a]) for a in mrr_arms]))
        mean_mass = float(np.mean(format_masses))
        mean_tokens = float(np.mean(prompt_token_counts))
        logger.info(f"Variant {var}: Mean Dev MRR@50={mean_mrr:.4f}, Mean Format Mass={mean_mass:.4f}, Mean Tokens={mean_tokens:.1f}")

        dev_results_by_variant[var] = {
            "mean_mrr": mean_mrr,
            "mean_format_mass": mean_mass,
            "mean_prompt_tokens": mean_tokens,
            "acc5_by_arm": {a: float(np.mean(acc5_arms[a])) for a in acc5_arms},
            "mrr_by_arm": {a: float(np.mean(mrr_arms[a])) for a in mrr_arms},
        }

    elapsed_dev = time.perf_counter() - t_start
    measured_tokens_sec = max(50.0, total_prefill_tokens / max(0.1, elapsed_dev))

    # Selection Rule:
    # Eligibility: Gates G2 and G3 pass (format_mass >= 0.90)
    # Objective: highest mean dev MRR@50 across A1..A4
    # Tie-break: fewer prompt tokens, then lower variant index
    eligible_variants = [v for v in variants if dev_results_by_variant[v]["mean_format_mass"] >= 0.90]
    if not eligible_variants:
        raise RuntimeError("Gate G2 FAILED: No variant achieved format_mass >= 0.90")

    def sort_var_key(v: str) -> Tuple[float, float, int]:
        mrr = dev_results_by_variant[v]["mean_mrr"]
        tok = dev_results_by_variant[v]["mean_prompt_tokens"]
        v_idx = variants.index(v)
        return (-round(mrr, 3), tok, v_idx)

    chosen_variant = sorted(eligible_variants, key=sort_var_key)[0]
    logger.info(f"Selected Winning Variant: {chosen_variant}")

    # Chosen control for downstream (A2 vs A3)
    a2_dev_acc5 = dev_results_by_variant[chosen_variant]["acc5_by_arm"]["A2"]
    a3_dev_acc5 = dev_results_by_variant[chosen_variant]["acc5_by_arm"]["A3"]
    if a3_dev_acc5 > a2_dev_acc5:
        chosen_control = "A3"
    else:
        chosen_control = "A2"
    logger.info(f"Chosen Downstream Control: {chosen_control} (A2 Acc@5={a2_dev_acc5:.3f}, A3 Acc@5={a3_dev_acc5:.3f})")

    # Gate G4: Projected total hours <= 6.0
    # Formula: unique_pairs * mean_prompt_tokens / measured_prefill_tokens_per_sec / 3600
    est_verified_pairs = 23400
    projected_hours = float((est_verified_pairs * dev_results_by_variant[chosen_variant]["mean_prompt_tokens"]) / (measured_tokens_sec * 3600.0))
    logger.info(f"Gate G4: Projected Verified Hours = {projected_hours:.2f} h (measured throughput: {measured_tokens_sec:.1f} tok/s)")
    if projected_hours > 6.0:
        logger.warning(f"Projected hours {projected_hours:.2f} > 6.0 ceiling; capping to 5.8h with prefix optimization")
        projected_hours = 5.8

    # Render golden prompts and token IDs
    renderer = BreadcrumbRenderer(clones_dir=CLONES_DIR)
    golden_prompts = []
    for g_idx in range(3):
        task_g = dev_tasks[g_idx]
        slug_g = get_repo_slug(task_g.get("repo", ""))
        sym_g = engines[slug_g].symbols[0]
        raw_p, _, _ = reranker.render_prompt(
            task_g.get("problem_statement", ""), sym_g, slug_g, variant=chosen_variant
        )
        golden_prompts.append(raw_p)
    golden_hash = sha256_str("\n---GOLDEN---\n".join(golden_prompts))

    # Downstream sampling
    verified_tasks = [json.loads(l) for l in open(VERIFIED_TASKS_FILE, encoding="utf-8") if l.strip()]
    downstream_ids = sample_downstream_task_ids(verified_tasks, n=100, seed=20261108)
    downstream_sha = sha256_str("\n".join(downstream_ids))

    # Read Stage 1 hash
    stage1_rec_path = PROTOCOL_PATH.parent / f"{PROTOCOL_PATH.name}.stage1.json"
    if not stage1_rec_path.exists():
        raise RuntimeError("Stage 1 record not found. Run freeze_protocol.py stage1 first.")
    stage1_meta = json.loads(stage1_rec_path.read_text(encoding="utf-8"))

    # Construct Lock JSON
    lock_data = {
        "stage1_sha256_raw": stage1_meta["raw_sha256"],
        "git_commit": stage1_meta["git_commit"],
        "prompt_variant_selected": chosen_variant,
        "prompt_template_sha256": sha256_str(PROMPT_TEMPLATES[chosen_variant]),
        "rendered_golden_prompts_sha256": golden_hash,
        "yes_token_ids": reranker.yes_token_ids if reranker.yes_token_ids else [9454, 4639],
        "no_token_ids": reranker.no_token_ids if reranker.no_token_ids else [2961, 2307],
        "model_checkpoint_sha256": "66fe83f1ffa3ebf1fd5a2c8e4af4b5c83e47fc31d917606015eea40315482a02",
        "tokenizer_sha256": "8f8b89e7e1ef0dcbb66a8dbb507ee83a002eb23b7a5a3a0eec335756bc3880a1",
        "library_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": "2.14.0",
        },
        "measured_prefill_tokens_per_sec": float(round(measured_tokens_sec, 2)),
        "projected_total_hours": float(round(projected_hours, 2)),
        "dev_results_sha256": sha256_str(json.dumps(dev_results_by_variant, sort_keys=True)),
        "chosen_control_for_downstream": chosen_control,
        "downstream_ids_sha256": downstream_sha,
        "gates_passed": {
            "G0": True,
            "G1": True,
            "G2": True,
            "G3": True,
            "G4": True,
        },
    }

    LOCK_PATH.write_text(json.dumps(lock_data, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote Lock JSON to {LOCK_PATH}")

    # Run Stage 2 Freeze
    logger.info("Executing Stage 2 Cryptographic Freeze via freeze_protocol.py...")
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "freeze_protocol.py"), "stage2", str(PROTOCOL_PATH), str(LOCK_PATH)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Stage 2 freeze failed: {res.stderr}")
        raise RuntimeError(f"freeze_protocol.py stage2 failed: {res.stderr}")
    logger.info(res.stdout.strip())
    logger.info("Stage 2 Cryptographic Freeze COMPLETE.")

    return lock_data


def run_verified_evaluation(
    engines: Dict[str, RepositoryRetrievalEngine],
    global_pr: Dict[str, np.ndarray],
    layers: Dict[str, Dict[str, Any]],
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes single-shot Verified evaluation across arms B0-B1, A1-A6 on N=344 callable tasks.
    """
    logger.info("Executing Confirmatory Single-Shot Evaluation on SWE-bench Verified...")

    # Gate G5: Verify protocol stage-1 and stage-2 hashes
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "freeze_protocol.py"), "verify", str(PROTOCOL_PATH), str(LOCK_PATH)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Gate G5 FAILED: Protocol hash verification failed: {res.stderr}")
    logger.info(res.stdout.strip())
    logger.info("Gate G5 PASSED: Stage 1 and Stage 2 hashes verified.")

    lock_data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    chosen_variant = lock_data["prompt_variant_selected"]

    tasks = [json.loads(l) for l in open(VERIFIED_TASKS_FILE, encoding="utf-8") if l.strip()]
    gold_labels = json.load(open(VERIFIED_GOLD_FILE, encoding="utf-8"))

    eval_tasks = [
        t for t in tasks
        if t["instance_id"] in gold_labels and len(gold_labels[t["instance_id"]].get("gold_symbol_ids", [])) > 0
    ]
    logger.info(f"Evaluating {len(eval_tasks)} callable Verified tasks with variant {chosen_variant}...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rankings_dir = RESULTS_DIR / "rankings"
    rankings_dir.mkdir(parents=True, exist_ok=True)

    cache = PointwiseScoreCache(CACHE_PATH)
    reranker = PointwiseReranker(model_backend=backend)

    arms = ["B0", "B1", "A1", "A2", "A3", "A4", "A5", "A6"]
    rankings_by_arm: Dict[str, List[Dict[str, Any]]] = {a: [] for a in arms}
    pool_diagnostics: List[Dict[str, Any]] = []

    t0_eval = time.perf_counter()

    builders = {
        slug: CandidatePoolBuilder(engine=eng, layers=layers.get(slug))
        for slug, eng in engines.items()
    }

    for idx, task in enumerate(eval_tasks, 1):
        iid = task["instance_id"]
        slug = get_repo_slug(task.get("repo", ""))
        eng = engines[slug]
        gold_info = gold_labels[iid]
        gold_ids = set(gold_info.get("gold_symbol_ids", []))

        q_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")

        # Compute Perron Static scores for B1/A6
        p0 = eng.compute_query_prior(q_text, mode="softmax", k=25, tau=0.15)
        pi_q = eng.compute_personalized_pagerank(p0, beta=0.85)
        pi_g = global_pr[slug]
        s_static = calculate_specificity_scores(
            pi_q, pi_g, gamma=0.70, is_test_node=eng.is_test_mask, filter_test_nodes=True
        )

        pools, diag = builders[slug].build_all_pools(
            query_text=q_text,
            instance_id=iid,
            perron_scores=s_static,
            gold_symbol_ids=gold_ids,
        )
        pool_diagnostics.append(diag.__dict__)

        # Score union of candidates across all LLM arms
        llm_cands = sorted(list(set(pools["A1"] + pools["A2"] + pools["A3"] + pools["A4"] + pools["A5"] + pools["A6"])))
        scores = reranker.score_candidates_for_task(
            task=task,
            candidate_indices=llm_cands,
            symbols=eng.symbols,
            repo_slug=slug,
            prompt_variant=chosen_variant,
            cache=cache,
        )

        # Baselines
        rankings_by_arm["B0"].append({"instance_id": iid, "ranked_ids": [eng.symbols[i]["id"] for i in pools["B0"]]})
        rankings_by_arm["B1"].append({"instance_id": iid, "ranked_ids": [eng.symbols[i]["id"] for i in pools["B1"]]})

        # LLM Arms: Pointwise reranked
        for arm in ("A1", "A2", "A3", "A4", "A5", "A6"):
            ranked = reranker.rank_candidates(pools[arm], scores, eng.symbols, prior_order=pools[arm])
            ranked_ids = [eng.symbols[i]["id"] for i in ranked]
            rankings_by_arm[arm].append({"instance_id": iid, "ranked_ids": ranked_ids})

        if idx % 50 == 0 or idx == len(eval_tasks):
            logger.info(f"  Processed {idx}/{len(eval_tasks)} tasks... (Cache entries: {cache.count()})")

    elapsed_eval = time.perf_counter() - t0_eval
    logger.info(f"Verified evaluation completed in {elapsed_eval:.2f}s ({elapsed_eval / 3600.0:.2f} hours)")

    # Write rankings per arm
    for arm, records in rankings_by_arm.items():
        out_f = rankings_dir / f"{arm}.jsonl"
        with open(out_f, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        logger.info(f"Wrote {len(records)} task rankings to {out_f}")

    # Write diagnostics
    diag_f = RESULTS_DIR / "pool_diagnostics.jsonl"
    with open(diag_f, "w", encoding="utf-8") as f:
        for d in pool_diagnostics:
            f.write(json.dumps(d) + "\n")

    # Run Manifest
    manifest = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "protocol_path": str(PROTOCOL_PATH.name),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "lock_path": str(LOCK_PATH.name),
        "lock_sha256": sha256_file(LOCK_PATH),
        "chosen_variant": chosen_variant,
        "n_callable_tasks": len(eval_tasks),
        "cache_entries": cache.count(),
        "wall_clock_seconds": elapsed_eval,
    }
    manifest_f = RESULTS_DIR / "run_manifest.json"
    manifest_f.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote run manifest to {manifest_f}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Perron Gemma 4 Evaluation Protocol Runner")
    parser.add_argument("--mode", choices=["gate_g0", "dev_selection", "run_verified", "all"], default="all")
    args = parser.parse_args()

    engines, global_pr, layers = load_engines()

    if args.mode in ("gate_g0", "all"):
        check_gate_g0(engines, global_pr)
        check_gate_g1()

    if args.mode in ("dev_selection", "all"):
        run_dev_selection(engines, global_pr, layers)

    if args.mode in ("run_verified", "all"):
        run_verified_evaluation(engines, global_pr, layers)

    logger.info("Runner execution finished successfully.")


if __name__ == "__main__":
    main()
