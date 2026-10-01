#!/usr/bin/env python3
"""
Automated Multi-Objective Bayesian Hyperparameter Optimization for Perron.

Optimizes specificity diffusion hyperparameters:
  - beta: PageRank random walk damping factor [0.50, 0.95]
  - gamma: Specificity hub-penalty damping exponent [0.10, 1.50]
  - tau: Shift-invariant softmax teleport temperature [0.01, 0.20]
  - k: Candidate seed count [5, 50]
  - mu: Context frontier packing connectivity bonus [0.0, 1.0]
  - lambda_caller: Caller (reverse) edge transition weight [0.05, 0.50]
  - token_budget: Context packing token ceiling [1024, 4096]

Objectives:
  1. Maximize Function-Level Recall@Budget (R_func)
  2. Maximize Hub Suppression Index (HSI)
  3. Minimize Diffusion Latency (ms on CPU)

Emits:
  - benchmarks/optuna_study.db (SQLite database)
  - benchmarks/optuna_results.json (Pareto optimal trials and statistics)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perron.matrix import build_static_transition_matrix
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import ASTContextSymbol, pack_context_subgraphs
from benchmarks.diagnostic import generate_synthetic_repo_graph


def evaluate_trial_configuration(
    beta: float,
    gamma: float,
    tau: float,
    top_k: int,
    mu: float,
    lambda_caller: float,
    token_budget: int,
    num_instances: int = 25,
    graph_size: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Evaluates a single hyperparameter configuration across simulated instances.
    Returns: (mean_func_recall, mean_hsi, mean_latency_ms)
    """
    rng = np.random.default_rng(seed)
    call_edges, caller_edges, symbols, hubs = generate_synthetic_repo_graph(
        num_nodes=graph_size, num_hubs=20, seed=seed
    )
    hubs_set = set(hubs)

    t_matrix, dangling = build_static_transition_matrix(
        num_nodes=graph_size,
        call_edges=call_edges,
        caller_edges=caller_edges,
        lambda_call=1.0,
        lambda_caller=lambda_caller,
    )
    pi_global = compute_global_pagerank(t_matrix, dangling, max_iter=100)

    # Build non-hub local adjacency
    adj: Dict[int, List[int]] = {i: [] for i in range(graph_size)}
    for u, v in call_edges:
        adj[u].append(v)
    for u, v in caller_edges:
        adj[u].append(v)

    func_recalls: List[float] = []
    hsi_scores: List[float] = []
    latencies_ms: List[float] = []

    for _ in range(num_instances):
        target = int(rng.integers(len(hubs) + 10, graph_size - 10))

        # 1-2 hop symptom selection
        neighbors_1 = [v for v in adj.get(target, []) if v not in hubs_set and v != target]
        if neighbors_1:
            v1 = int(rng.choice(neighbors_1))
            neighbors_2 = [v for v in adj.get(v1, []) if v not in hubs_set and v != target and v != v1]
            symptom = int(rng.choice(neighbors_2)) if neighbors_2 else v1
        else:
            symptom = max(0, min(graph_size - 1, target - 1))

        sims = np.clip(rng.normal(0.10, 0.03, size=graph_size), 0.0, 1.0)
        sims[symptom] = float(rng.uniform(0.80, 0.95))
        for n in adj.get(symptom, []):
            if n not in hubs_set and n != target:
                sims[n] = max(sims[n], float(rng.uniform(0.40, 0.60)))
        for h in rng.choice(hubs, size=min(4, len(hubs)), replace=False):
            sims[h] = float(rng.uniform(0.40, 0.65))

        t0 = time.perf_counter()
        p_0 = compute_softmax_teleport_prior(
            similarities=sims,
            node_indices=np.arange(graph_size),
            num_nodes=graph_size,
            tau=tau,
            top_k=top_k,
        )

        pi_q = personalized_pagerank_power_iteration(
            t_matrix=t_matrix,
            dangling=dangling,
            p_0=p_0,
            beta=beta,
            max_iter=100,
        )

        scores = calculate_specificity_scores(
            pi_query=pi_q,
            pi_global=pi_global,
            gamma=gamma,
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

        # HSI (Hub Suppression Index in top 10)
        ranked = np.argsort(scores)[::-1]
        top10_hubs = sum(1 for n in ranked[:10] if n in hubs_set)
        hsi = 1.0 - (top10_hubs / 10.0)
        hsi_scores.append(hsi * 100.0)

        # Context packing Function Recall
        packed, _ = pack_context_subgraphs(
            symbols=symbols,
            specificity_scores=scores,
            adjacency_matrix=t_matrix,
            token_budget=token_budget,
            mu=mu,
        )
        packed_ids = {s.node_id for s in packed}
        func_recalls.append(1.0 if target in packed_ids else 0.0)

    return float(np.mean(func_recalls) * 100.0), float(np.mean(hsi_scores)), float(np.mean(latencies_ms))


def run_optuna_study(
    n_trials: int = 50,
    seed: int = 42,
    db_path: Path = REPO_ROOT / "benchmarks" / "optuna_study.db",
) -> Dict[str, Any]:
    """
    Executes multi-objective Optuna Bayesian optimization study.
    """
    print(f"Initializing Optuna Multi-Objective Study (trials={n_trials}, seed={seed})...")

    # Remove stale SQLite DB if exists to guarantee clean run
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    storage_url = f"sqlite:///{db_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=seed)

    study = optuna.create_study(
        study_name="perron_specificity_diffusion_hpo",
        storage=storage_url,
        directions=["maximize", "maximize", "minimize"],
        sampler=sampler,
        load_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> Tuple[float, float, float]:
        beta = trial.suggest_float("beta", 0.60, 0.95, step=0.05)
        gamma = trial.suggest_float("gamma", 0.20, 1.20, step=0.05)
        tau = trial.suggest_float("tau", 0.02, 0.15, step=0.01)
        top_k = trial.suggest_int("top_k", 10, 40, step=5)
        mu = trial.suggest_float("mu", 0.10, 0.90, step=0.10)
        lambda_caller = trial.suggest_float("lambda_caller", 0.10, 0.40, step=0.05)
        token_budget = trial.suggest_categorical("token_budget", [1024, 2048, 3072, 3480, 4096])

        func_rec, hsi, latency_ms = evaluate_trial_configuration(
            beta=beta,
            gamma=gamma,
            tau=tau,
            top_k=top_k,
            mu=mu,
            lambda_caller=lambda_caller,
            token_budget=token_budget,
            num_instances=20,
            graph_size=1000,
            seed=seed + trial.number,
        )
        return func_rec, hsi, latency_ms

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    pareto_trials = study.best_trials
    print(f"\nOptimization Complete! Total trials: {len(study.trials)}, Pareto optimal trials: {len(pareto_trials)}")

    results = {
        "n_trials": n_trials,
        "n_pareto": len(pareto_trials),
        "pareto_front": [
            {
                "trial_number": t.number,
                "values": {
                    "func_recall_pct": t.values[0],
                    "hsi_pct": t.values[1],
                    "latency_ms": t.values[2],
                },
                "params": t.params,
            }
            for t in pareto_trials
        ],
        "default_recommended_params": {
            "beta": 0.85,
            "gamma": 0.70,
            "tau": 0.05,
            "top_k": 25,
            "mu": 0.50,
            "lambda_caller": 0.20,
            "token_budget": 3480,
        },
    }

    out_file = REPO_ROOT / "benchmarks" / "optuna_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved Pareto results to: {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Perron AutoML Optuna HPO Suite")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_optuna_study(n_trials=args.n_trials, seed=args.seed)


if __name__ == "__main__":
    main()
