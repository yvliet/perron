"""
Real benchmark evaluation runner for Perron on SWE-bench and real repository issue instances.
Supports running against local Gemma 4 backends (TransformersBackend, VllmBackend) and OfflineReplayBackend.
Provides ephemeral git worktree isolation to guarantee clean repository state across benchmark instances.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perron.agent import PerronAgent, TrajectoryResult
from perron.backends.base import ModelBackend
from perron.backends.replay import OfflineReplayBackend
from perron.backends.transformers_backend import TransformersBackend
from perron.backends.vllm_backend import VllmBackend
from perron.patch import compute_git_patch
from perron.tester import EphemeralWorktree


def _execute_instance(
    instance_id: str,
    issue_text: str,
    test_file: Optional[str],
    test_filter: Optional[str],
    repo_dir: Path,
    backend: ModelBackend,
    token_budget: int = 3480,
    max_turns: int = 5,
    test_timeout: float = 45.0,
    rollback_on_test_failure: bool = False,
    assume_resolved_without_test: bool = False,
) -> Dict[str, Any]:
    """
    Executes a single issue repair attempt within a prepared target directory.
    """
    agent = PerronAgent(
        repo_dir=repo_dir,
        backend=backend,
        token_budget=token_budget,
        max_turns=max_turns,
        test_timeout=test_timeout,
        rollback_on_test_failure=rollback_on_test_failure,
        assume_resolved_without_test=assume_resolved_without_test,
    )

    t0 = time.perf_counter()
    with agent:
        agent.initialize_graph()
        result: TrajectoryResult = agent.solve_issue(
            issue_text=issue_text,
            test_file=test_file,
            test_filter=test_filter,
            instance_id=instance_id,
        )

    duration = time.perf_counter() - t0
    patch = compute_git_patch(repo_dir, initial_snapshots=getattr(agent, "all_initial_snapshots", None))

    return {
        "instance_id": instance_id,
        "resolved": result.resolved,
        "total_turns": result.total_turns,
        "duration_seconds": duration,
        "patch_applied": result.patch_applied,
        "test_passed": result.test_passed,
        "failure_category": result.failure_category,
        "applied_edits_count": len(result.applied_edits),
        "patch": patch,
    }


def run_instance(
    instance: Dict[str, Any],
    backend: ModelBackend,
    base_repo_dir: Path,
    token_budget: int = 3480,
    max_turns: int = 5,
    test_timeout: float = 45.0,
    rollback_on_test_failure: bool = False,
    use_worktree: bool = True,
    assume_resolved_without_test: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a single issue instance within an isolated repository environment.
    """
    instance_id = instance.get("instance_id") or instance.get("id", "custom_task")
    issue_text = instance.get("problem_statement") or instance.get("issue_text") or instance.get("issue", "")
    test_file = instance.get("test_file")
    test_filter = instance.get("test_filter")
    base_commit = instance.get("base_commit", "HEAD")

    # Handle instance-specific canned responses for OfflineReplayBackend
    inst_backend = backend
    if isinstance(backend, OfflineReplayBackend) and "model_responses" in instance:
        inst_backend = OfflineReplayBackend(canned_responses=instance["model_responses"])

    # If instance defines self-contained source and test files, set up in dedicated workspace
    if "file" in instance and "code" in instance:
        with tempfile.TemporaryDirectory(prefix="perron_inst_") as tmpdir:
            inst_repo = Path(tmpdir)
            src_file = inst_repo / instance["file"]
            src_file.parent.mkdir(parents=True, exist_ok=True)
            src_file.write_text(instance["code"], encoding="utf-8")

            if "test_file" in instance and "test_code" in instance and instance["test_code"]:
                t_file = inst_repo / instance["test_file"]
                t_file.parent.mkdir(parents=True, exist_ok=True)
                t_file.write_text(instance["test_code"], encoding="utf-8")

            # Initialize git so diff calculation and worktrees function
            subprocess.run(["git", "init"], cwd=str(inst_repo), capture_output=True, check=False)
            subprocess.run(["git", "config", "user.name", "Sultan Haikal"], cwd=str(inst_repo), capture_output=True, check=False)
            subprocess.run(["git", "config", "user.email", "yvliet@users.noreply.github.com"], cwd=str(inst_repo), capture_output=True, check=False)
            subprocess.run(["git", "add", "."], cwd=str(inst_repo), capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(inst_repo), capture_output=True, check=False)

            if use_worktree:
                with EphemeralWorktree(inst_repo) as target_repo:
                    return _execute_instance(
                        instance_id=instance_id,
                        issue_text=issue_text,
                        test_file=test_file,
                        test_filter=test_filter,
                        repo_dir=target_repo,
                        backend=inst_backend,
                        token_budget=token_budget,
                        max_turns=max_turns,
                        test_timeout=test_timeout,
                        rollback_on_test_failure=rollback_on_test_failure,
                        assume_resolved_without_test=assume_resolved_without_test,
                    )
            else:
                return _execute_instance(
                    instance_id=instance_id,
                    issue_text=issue_text,
                    test_file=test_file,
                    test_filter=test_filter,
                    repo_dir=inst_repo,
                    backend=inst_backend,
                    token_budget=token_budget,
                    max_turns=max_turns,
                    test_timeout=test_timeout,
                    rollback_on_test_failure=rollback_on_test_failure,
                    assume_resolved_without_test=assume_resolved_without_test,
                )

    if use_worktree:
        with EphemeralWorktree(base_repo_dir, commit_or_branch=base_commit) as target_repo:
            return _execute_instance(
                instance_id=instance_id,
                issue_text=issue_text,
                test_file=test_file,
                test_filter=test_filter,
                repo_dir=target_repo,
                backend=inst_backend,
                token_budget=token_budget,
                max_turns=max_turns,
                test_timeout=test_timeout,
                rollback_on_test_failure=rollback_on_test_failure,
                assume_resolved_without_test=assume_resolved_without_test,
            )
    else:
        return _execute_instance(
            instance_id=instance_id,
            issue_text=issue_text,
            test_file=test_file,
            test_filter=test_filter,
            repo_dir=base_repo_dir,
            backend=inst_backend,
            token_budget=token_budget,
            max_turns=max_turns,
            test_timeout=test_timeout,
            rollback_on_test_failure=rollback_on_test_failure,
            assume_resolved_without_test=assume_resolved_without_test,
        )


def run_evaluation_suite(
    instances: List[Dict[str, Any]],
    backend: ModelBackend,
    base_repo_dir: Path,
    output_path: Optional[Path] = None,
    use_worktree: bool = True,
    assume_resolved_without_test: bool = False,
) -> Dict[str, Any]:
    """
    Runs a batch of evaluation instances and records aggregate metrics and patches.
    """
    results: List[Dict[str, Any]] = []
    resolved_count = 0

    print(f"Starting Perron Evaluation Suite across {len(instances)} instance(s)...")
    for idx, inst in enumerate(instances):
        iid = inst.get("instance_id") or inst.get("id", f"inst_{idx}")
        print(f"[{idx+1}/{len(instances)}] Evaluating {iid}...")
        res = run_instance(
            instance=inst,
            backend=backend,
            base_repo_dir=base_repo_dir,
            use_worktree=use_worktree,
            assume_resolved_without_test=assume_resolved_without_test,
        )
        results.append(res)
        if res["resolved"]:
            resolved_count += 1
            print(f"  -> RESOLVED in {res['duration_seconds']:.2f}s ({res['total_turns']} turns)")
        else:
            print(f"  -> UNRESOLVED ({res.get('failure_category', 'Unknown')})")

    summary = {
        "total_instances": len(instances),
        "resolved_count": resolved_count,
        "resolve_rate": resolved_count / max(1, len(instances)),
        "results": results,
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved evaluation results to {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perron Real Evaluation Runner")
    parser.add_argument("--repo-dir", type=str, default=".", help="Target repository directory")
    parser.add_argument("--backend", "--mode", dest="backend", type=str, default="replay", choices=["replay", "transformers", "vllm"])
    parser.add_argument("--model-name", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output", type=str, default="benchmarks/real_eval_results.json")
    parser.add_argument("--manifest", type=str, default=None, help="Task manifest path")
    parser.add_argument("--limit", type=int, default=None, help="Maximum instances to evaluate")
    parser.add_argument("--no-worktree", action="store_true", help="Disable ephemeral git worktree isolation")
    parser.add_argument("--assume-resolved-without-test", action="store_true", help="Permit unverified resolution when test_file is None")
    args = parser.parse_args()

    target_repo = Path(args.repo_dir).resolve()
    if args.backend == "transformers":
        model_backend = TransformersBackend(model_name_or_path=args.model_name)
    elif args.backend == "vllm":
        model_backend = VllmBackend(base_url=args.vllm_url, model_name=args.model_name)
    else:
        model_backend = OfflineReplayBackend()

    instances = []
    if args.manifest and Path(args.manifest).exists():
        with open(args.manifest, "r", encoding="utf-8") as f:
            instances = json.load(f)
    else:
        from benchmarks.repair_eval import BENCHMARK_CASES
        instances = list(BENCHMARK_CASES)

    if args.limit is not None and args.limit > 0:
        instances = instances[:args.limit]

    out_file = REPO_ROOT / args.output
    summary_res = run_evaluation_suite(
        instances=instances,
        backend=model_backend,
        base_repo_dir=target_repo,
        output_path=out_file,
        use_worktree=not args.no_worktree,
        assume_resolved_without_test=args.assume_resolved_without_test,
    )
    print(f"Evaluation complete: {summary_res['resolved_count']}/{summary_res['total_instances']} resolved.")
