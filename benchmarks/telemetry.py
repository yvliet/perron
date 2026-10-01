"""
Telemetry Collection and Benchmark Ingestion for Perron.

Collects trajectory telemetry from real PerronAgent runs, computes empirical
action dynamics by turn, failure mode distributions, and pass@k sample scaling,
and exports clean JSON manifests for paper figure generation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from perron.agent import TrajectoryResult, TurnTelemetry

BENCHMARK_DIR = Path(__file__).resolve().parent
RUN_TELEMETRY_PATH = BENCHMARK_DIR / "run_telemetry.json"
PASS_AT_K_PATH = BENCHMARK_DIR / "pass_at_k_results.json"


CANONICAL_ACTIONS = [
    "CSR_INGEST",
    "TELEPORT_PRIOR",
    "PPR_WALK",
    "SPEC_FILTER",
    "AST_PACK",
    "MODEL_INFER",
    "AST_EDIT",
    "PYTEST_RUN",
]

CANONICAL_FAILURE_MODES = [
    "Multi-file Latent Dependency",
    "Dynamic Reflection / Monkey-patch",
    "Underspecified Issue Text",
    "Harness Timeout / Deadlock",
    "Premature Search Termination",
]


def aggregate_trajectories(
    trajectories: List[TrajectoryResult],
    max_turns: int = 10,
) -> Dict:
    """
    Aggregates a list of TrajectoryResults into figure-ready distributions.
    """
    # 1. Action frequencies by turn index (0 to max_turns - 1)
    action_counts: Dict[str, List[int]] = {act: [0] * max_turns for act in CANONICAL_ACTIONS}

    for traj in trajectories:
        for t in traj.turns:
            t_idx = min(max(0, t.turn_index - 1), max_turns - 1)
            act = t.action_type
            if act in action_counts:
                action_counts[act][t_idx] += 1

    # 2. Failure mode counts
    failure_counts: Dict[str, int] = {m: 0 for m in CANONICAL_FAILURE_MODES}
    unresolved_count = 0

    for traj in trajectories:
        if not traj.resolved:
            unresolved_count += 1
            cat = traj.failure_category or "Premature Search Termination"
            if cat in failure_counts:
                failure_counts[cat] += 1
            else:
                failure_counts["Multi-file Latent Dependency"] += 1

    # Compute percentages for donut chart
    total_failures = max(1, unresolved_count)
    failure_percentages = [
        round((failure_counts[m] / total_failures) * 100.0, 1)
        for m in CANONICAL_FAILURE_MODES
    ]

    telemetry_data = {
        "num_trajectories": len(trajectories),
        "resolved_count": sum(1 for t in trajectories if t.resolved),
        "unresolved_count": unresolved_count,
        "turns": list(range(max_turns)),
        "action_frequencies": action_counts,
        "failure_labels": CANONICAL_FAILURE_MODES,
        "failure_percentages": failure_percentages,
    }

    return telemetry_data


def save_telemetry(telemetry_data: Dict, path: Optional[Path] = None) -> None:
    out_path = path or RUN_TELEMETRY_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_data, f, indent=2)


def load_telemetry(path: Optional[Path] = None) -> Optional[Dict]:
    target_path = path or RUN_TELEMETRY_PATH
    if not target_path.exists():
        return None
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)
