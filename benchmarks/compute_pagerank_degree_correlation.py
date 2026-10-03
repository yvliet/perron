#!/usr/bin/env python3
"""Spearman correlation between global PageRank and in-degree per repository graph.

Writes results/pagerank_degree_spearman.json. Engines use the harness defaults
(beta = 0.85), matching the pi_global used by every retrieval method.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr  # noqa: E402

from perron.results import get_git_sha  # noqa: E402
from scripts.run_heldout import load_repository_engines  # noqa: E402


def main() -> None:
    engines = load_repository_engines(PROJECT_ROOT / "artifacts" / "swebench_graphs")
    per_repo = {
        slug: float(spearmanr(engine.pi_global, engine.in_degrees).statistic)
        for slug, engine in sorted(engines.items())
    }
    values = list(per_repo.values())
    out = {
        "suite": "pagerank_degree_spearman",
        "num_repositories": len(per_repo),
        "per_repo": per_repo,
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "min_repo": min(per_repo, key=per_repo.get),
        "max_repo": max(per_repo, key=per_repo.get),
        "git_sha": get_git_sha(),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path = PROJECT_ROOT / "results" / "pagerank_degree_spearman.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(PROJECT_ROOT)}: min={out['min']:.3f} max={out['max']:.3f} mean={out['mean']:.3f}")


if __name__ == "__main__":
    main()
