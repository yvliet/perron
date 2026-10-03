"""
benchmarks/run_swebench_lite.py - Automated SWE-bench Lite batch evaluation runner.

Executes offline diagnostic evaluation across SWE-bench Lite task instances or representative
archetypes, leveraging Perron's targeted test isolation, AST patching, and deterministic timeout guards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure perron package is discoverable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.repair_eval import BENCHMARK_CASES
from benchmarks.run_real_evaluation import run_evaluation_suite
from perron.backends.replay import OfflineReplayBackend
from perron.backends.transformers_backend import TransformersBackend
from perron.backends.vllm_backend import VllmBackend


def load_benchmark_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Loads SWE-bench Lite evaluation instances or defaults to verified archetype instances."""
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    cache_file = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    if cache_file.exists():
        from benchmarks.swebench_loader import load_cached_swebench_lite
        return load_cached_swebench_lite(cache_file)
    print(f"[*] Task manifest not found at {manifest_path}; defaulting to {len(BENCHMARK_CASES)} representative defect archetypes.")
    return list(BENCHMARK_CASES)


def evaluate_batch(
    instances: List[Dict[str, Any]],
    output_path: Path,
    mode: str = "replay",
    model_name: str = "google/gemma-4-31b-it",
    vllm_url: str = "http://localhost:8000/v1",
    base_repo_dir: Path = REPO_ROOT,
    limit: int = 10,
    use_worktree: bool = True,
) -> Dict[str, Any]:
    """
    Evaluates an automated slice of SWE-bench Lite task instances via run_evaluation_suite.
    """
    if mode == "transformers":
        backend = TransformersBackend(model_name_or_path=model_name)
    elif mode == "vllm":
        backend = VllmBackend(base_url=vllm_url, model_name=model_name)
    else:
        backend = OfflineReplayBackend()

    eval_instances = instances[:limit] if limit else instances

    summary = run_evaluation_suite(
        instances=eval_instances,
        backend=backend,
        base_repo_dir=base_repo_dir,
        output_path=output_path,
        use_worktree=use_worktree,
        assume_resolved_without_test=False,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Perron Automated SWE-bench Lite Batch Evaluator")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/swebench_lite_cache.jsonl", help="Task manifest path")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "telemetry/swebench_batch_results.json", help="Output telemetry path")
    parser.add_argument("--mode", "--backend", dest="mode", type=str, default="replay", choices=["replay", "transformers", "vllm"], help="Execution mode / backend")
    parser.add_argument("--limit", type=int, default=10, help="Max instances to evaluate")
    parser.add_argument("--repo-dir", type=Path, default=REPO_ROOT, help="Base repository directory")
    args = parser.parse_args()

    instances = load_benchmark_manifest(args.manifest)
    evaluate_batch(
        instances=instances,
        output_path=args.output,
        mode=args.mode,
        base_repo_dir=args.repo_dir,
        limit=args.limit,
        use_worktree=True,
    )


if __name__ == "__main__":
    main()
