#!/usr/bin/env python3
"""
Automated Kaggle GPU Batch Runner for Gemma 4 SWE-bench Lite Evaluation.
Executes live inference on Kaggle dual-T4 GPUs across 50 SWE-bench Lite tasks
in an A/B benchmark (Condition A: Naive Sliding Window vs Condition B: Perron Packing).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perron.agent import PerronAgent, TrajectoryResult
from perron.backends.transformers_backend import TransformersBackend
from perron.tester import EphemeralWorktree

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("kaggle_runner")


def resolve_model_path(base_model_name: str) -> str:
    """Resolve model path from Kaggle mounts or local cache."""
    candidates = [
        Path(f"/kaggle/input/{base_model_name}"),
        Path(f"/kaggle/input/{base_model_name.replace('/', '-')}-it"),
        Path(f"/kaggle/input/gemma-4-e4b-it"),
        Path(f"/kaggle/input/gemma-4-e4b"),
        Path(base_model_name),
    ]
    for c in candidates:
        if c.exists():
            logger.info(f"Found pre-mounted model at: {c}")
            return str(c)
    return base_model_name


def run_benchmark(
    tasks_file: Path,
    output_file: Path,
    base_model: str,
    limit: int = 50,
    condition: str = "perron",
) -> None:
    logger.info(f"Starting Kaggle SWE-bench Lite Evaluation (Condition: {condition})...")

    if tasks_file.suffix == ".jsonl":
        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = [json.loads(line) for line in f if line.strip()][:limit]
    else:
        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)[:limit]

    resolved_path = resolve_model_path(base_model)
    logger.info(f"Instantiating TransformersBackend for {resolved_path} in 4-bit...")

    backend = TransformersBackend(
        model_name_or_path=resolved_path,
        load_in_4bit=True,
    )

    results: List[Dict] = []
    success_count = 0
    syntax_error_count = 0
    total_tokens_spent = 0

    for idx, task in enumerate(tasks, 1):
        instance_id = task.get("instance_id", f"task_{idx}")
        problem_statement = task.get("problem_statement", "")
        repo = task.get("repo", "psf/requests")
        logger.info(f"[{idx}/{len(tasks)}] Evaluating {instance_id} ({repo})...")

        repo_dir = REPO_ROOT / "artifacts" / "clones" / repo.replace("/", "__")
        if not repo_dir.exists():
            repo_dir = REPO_ROOT

        with EphemeralWorktree(repo_dir) as worktree:
            target_worktree = getattr(worktree, "worktree_dir", worktree)
            start_t = time.time()
            if condition == "perron":
                agent = PerronAgent(
                    repo_dir=target_worktree,
                    backend=backend,
                    token_budget=3480,
                    gamma=0.70,
                    beta=0.85,
                    max_turns=2,
                )
            else:
                agent = PerronAgent(
                    repo_dir=target_worktree,
                    backend=backend,
                    token_budget=4096,
                    max_turns=2,
                )

            traj: TrajectoryResult = agent.solve_issue(issue_text=problem_statement, instance_id=instance_id)
            latency = time.time() - start_t

            if traj.resolved:
                success_count += 1
            is_syntax_error = (not traj.patch_applied and traj.failure_category == "syntax_error")
            if is_syntax_error:
                syntax_error_count += 1

            task_tokens = sum(t.token_cost for t in traj.turns)
            total_tokens_spent += task_tokens

            results.append({
                "instance_id": instance_id,
                "repo": repo,
                "resolved": traj.resolved,
                "patch_applied": traj.patch_applied,
                "syntax_error": is_syntax_error,
                "turns": len(traj.turns),
                "total_tokens": task_tokens,
                "latency_sec": latency,
            })
            logger.info(f"Task {instance_id} -> Resolved={traj.resolved}, Time={latency:.1f}s")

    summary = {
        "condition": condition,
        "total_tasks": len(tasks),
        "resolved_count": success_count,
        "pass_at_1": (success_count / len(tasks)) * 100.0,
        "syntax_error_rate": (syntax_error_count / len(tasks)) * 100.0,
        "avg_tokens_per_task": total_tokens_spent / max(1, len(tasks)),
        "tasks": results,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n=======================================================")
    logger.info(f"EVALUATION COMPLETE ({condition.upper()}):")
    logger.info(f"Pass@1: {summary['pass_at_1']:.1f}% ({success_count}/{len(tasks)})")
    logger.info(f"Syntax Error Rate: {summary['syntax_error_rate']:.1f}%")
    logger.info(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Gemma 4 Live Evaluator.")
    default_tasks = (
        REPO_ROOT / "benchmarks" / "data" / "swebench_lite_cached.json"
        if (REPO_ROOT / "benchmarks" / "data" / "swebench_lite_cached.json").exists()
        else REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    )
    parser.add_argument("--tasks", type=Path, default=default_tasks)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "telemetry" / "kaggle_live_gemma_results.json")
    parser.add_argument("--model", type=str, default="google/gemma-4-e4b-it")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--condition", choices=["perron", "naive"], default="perron")
    args = parser.parse_args()

    run_benchmark(args.tasks, args.output, args.model, limit=args.limit, condition=args.condition)
