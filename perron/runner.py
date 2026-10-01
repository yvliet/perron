"""
Perron Headless Agent Runner & CLI Entrypoint.

Invoked by evaluation harnesses, automated batch runners, or offline container environments.
Parses agent.yaml configuration, initializes PerronAgent with specified model/diffusion settings,
executes autonomous problem repair, and outputs the unified git diff patch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perron.agent import PerronAgent, TrajectoryResult
from perron.backends.base import ModelBackend
from perron.backends.replay import OfflineReplayBackend
from perron.backends.transformers_backend import TransformersBackend
from perron.backends.vllm_backend import VllmBackend
from perron.patch import compute_git_patch
from perron.tester import EphemeralWorktree


def load_agent_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Parses agent.yaml configuration file using PyYAML.
    """
    if not config_path.is_file():
        return {}

    content = config_path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(content) or {}
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for agent.yaml parsing. Please install pyyaml."
        ) from e


from perron.backends.gguf_backend import GgufBackend


def create_backend_from_config(
    model_cfg: Dict[str, Any],
    runtime_cfg: Optional[Dict[str, Any]] = None,
) -> ModelBackend:
    """
    Instantiates the model backend specified in agent.yaml.
    Supports environment variable overrides for offline container execution:
    GEMMA_CHECKPOINT_DIR, MODEL_PATH, or local cache paths.
    """
    backend_type = model_cfg.get("backend", "transformers").lower()
    checkpoint_candidates = [
        os.environ.get("GEMMA_CHECKPOINT_DIR"),
        os.environ.get("MODEL_PATH"),
        "/kaggle/input/gemma-4/transformers",
        "/kaggle/input/gemma-4",
        "./models/gemma-4",
    ]
    resolved_path = None
    for cand in checkpoint_candidates:
        if cand and Path(cand).exists():
            resolved_path = str(Path(cand).resolve())
            break
    base_model = resolved_path or model_cfg.get("base", "google/gemma-4/transformers/gemma-4-e2b-it-qat-mobile-transformers")
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path and not Path(adapter_path).exists():
        adapter_path = None

    quantization = model_cfg.get("quantization", "4bit")
    offline_mode = bool(
        model_cfg.get("offline_mode", False)
        or (runtime_cfg.get("offline_mode", False) if runtime_cfg else False)
    )

    if backend_type == "transformers":
        load_in_4bit = quantization in ("4bit", "q4_k_m")
        load_in_8bit = quantization == "8bit"
        return TransformersBackend(
            model_name_or_path=base_model,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            local_files_only=offline_mode,
        )
    elif backend_type == "gguf":
        model_path = os.environ.get("GGUF_MODEL_PATH") or model_cfg.get("model_path")
        if not model_path or not Path(model_path).exists():
            print(f"[RUNNER WARNING] GGUF model path not found on disk; falling back to TransformersBackend({base_model})")
            return TransformersBackend(
                model_name_or_path=base_model,
                adapter_path=adapter_path,
                local_files_only=offline_mode,
            )
        return GgufBackend(model_path=model_path, model_name=base_model)
    elif backend_type == "vllm":
        vllm_url = model_cfg.get("base_url", "http://localhost:8000/v1")
        return VllmBackend(base_url=vllm_url, model_name=base_model)
    elif backend_type == "replay":
        return OfflineReplayBackend()
    else:
        raise ValueError(f"Unknown model backend type: {backend_type}")


def run_runner(
    repo_dir: Path,
    issue_text: str,
    config_path: Optional[Path] = None,
    output_patch_path: Optional[Path] = None,
    test_file: Optional[str] = None,
    test_filter: Optional[str] = None,
    instance_id: str = "task_instance",
    max_turns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Core execution logic for headless agent runner.
    """
    cfg_file = config_path or (REPO_ROOT / "agent.yaml")
    config = load_agent_yaml(cfg_file)

    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})
    runtime_cfg = config.get("runtime", {})

    token_budget = model_cfg.get("max_context_tokens", 3480)
    effective_max_turns = max_turns if max_turns is not None else model_cfg.get("max_turns", 5)
    test_timeout = 45.0
    for tool in config.get("tools", []):
        if isinstance(tool, dict) and tool.get("name") == "isolated_test_runner":
            test_timeout = float(tool.get("parameters", {}).get("timeout_seconds", 45.0))
            break

    beta = diffusion_cfg.get("beta", 0.85)
    gamma = diffusion_cfg.get("gamma", 0.70)
    tau = diffusion_cfg.get("tau", 0.15)

    backend = create_backend_from_config(model_cfg, runtime_cfg)

    with EphemeralWorktree(repo_dir) as worktree_repo:
        agent = PerronAgent(
            repo_dir=worktree_repo,
            backend=backend,
            token_budget=token_budget,
            max_turns=effective_max_turns,
            test_timeout=test_timeout,
            assume_resolved_without_test=(test_file is None),
            beta=beta,
            gamma=gamma,
            tau=tau,
        )

        with agent:
            agent.initialize_graph()
            result: TrajectoryResult = agent.solve_issue(
                issue_text=issue_text,
                test_file=test_file,
                test_filter=test_filter,
                instance_id=instance_id,
            )

        patch_content = compute_git_patch(worktree_repo, initial_snapshots=getattr(agent, "all_initial_snapshots", None))

    if output_patch_path:
        output_patch_path.parent.mkdir(parents=True, exist_ok=True)
        output_patch_path.write_text(patch_content, encoding="utf-8")
        print(f"[RUNNER] Patch written to {output_patch_path} ({len(patch_content)} bytes)")

    return {
        "instance_id": instance_id,
        "resolved": result.resolved,
        "patch_applied": result.patch_applied,
        "total_turns": result.total_turns,
        "duration_seconds": result.total_duration_seconds,
        "patch_size_bytes": len(patch_content),
        "patch": patch_content,
    }


def main():
    parser = argparse.ArgumentParser(description="Perron Autonomous Developer Agent Headless Runner")
    parser.add_argument("--repo-dir", type=str, default=".", help="Target repository directory")
    parser.add_argument("--issue", type=str, default="", help="Issue description text")
    parser.add_argument("--issue-file", type=str, default=None, help="Path to file containing issue description")
    parser.add_argument("--output-patch", type=str, default=None, help="Path to save generated git patch")
    parser.add_argument("--config", type=str, default="agent.yaml", help="Path to agent.yaml")
    parser.add_argument("--test-file", type=str, default=None, help="Optional test file to verify")
    parser.add_argument("--test-filter", type=str, default=None, help="Optional test filter expression")
    parser.add_argument("--instance-id", type=str, default="task_0", help="Unique task identifier")
    parser.add_argument("--max-turns", type=int, default=None, help="Maximum agent turns")

    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    config_file = Path(args.config).resolve()

    if args.issue_file and Path(args.issue_file).is_file():
        issue_text = Path(args.issue_file).read_text(encoding="utf-8")
    else:
        issue_text = args.issue or "Fix issue in repository"

    out_patch = Path(args.output_patch).resolve() if args.output_patch else None

    result = run_runner(
        repo_dir=repo_dir,
        issue_text=issue_text,
        config_path=config_file,
        output_patch_path=out_patch,
        test_file=args.test_file,
        test_filter=args.test_filter,
        instance_id=args.instance_id,
        max_turns=args.max_turns,
    )

    if not out_patch and result["patch"]:
        print(result["patch"])


if __name__ == "__main__":
    main()
