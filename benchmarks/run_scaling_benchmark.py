#!/usr/bin/env python3
"""
benchmarks/run_scaling_benchmark.py - Master runner for cross-architecture scaling benchmark.

Orchestrates sequential execution of 5 models across isolated OS subprocesses to ensure
clean memory reclamation and generate verifiable empirical receipts.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("perron.scaling_master")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = PROJECT_ROOT / "data_release" / "stratum_c_eval_bundle.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "scaling_benchmark_results.json"
TMP_DIR = PROJECT_ROOT / "scratch" / "scaling_worker_tmp"

MODELS_CONFIG = [
    {
        "id": "google/gemma-4-e2b-it",
        "name": "Gemma 4 E2B",
        "family": "Google Gemma 4",
        "params_b": 2.0,
        "active_params_b": 2.0,
        "archetype": "PLE Dense Edge",
        "tier": "pointwise",
    },
    {
        "id": "google/gemma-4-e4b-it",
        "name": "Gemma 4 E4B",
        "family": "Google Gemma 4",
        "params_b": 4.0,
        "active_params_b": 4.0,
        "archetype": "PLE Dense Edge",
        "tier": "hybrid_anchor",
    },
    {
        "id": "microsoft/Phi-4-mini-instruct",
        "name": "Phi-4-mini",
        "family": "Microsoft Phi",
        "params_b": 3.8,
        "active_params_b": 3.8,
        "archetype": "High-Density Synthetic Reasoning",
        "tier": "pointwise",
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "name": "DeepSeek R1 Distill 7B",
        "family": "DeepSeek AI",
        "params_b": 7.0,
        "active_params_b": 7.0,
        "archetype": "RL Reasoning Distill",
        "tier": "pointwise",
    },
    {
        "id": "google/gemma-4-26b-a4b-it",
        "name": "Gemma 4 26B-A4B",
        "family": "Google Gemma 4",
        "params_b": 26.0,
        "active_params_b": 3.8,
        "archetype": "Dynamic Sparse MoE",
        "tier": "hybrid_anchor",
    },
]


def check_gpu_telemetry() -> Dict[str, Any]:
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        count = torch.cuda.device_count() if cuda_ok else 0
        names = [torch.cuda.get_device_name(i) for i in range(count)] if cuda_ok else []
        return {"cuda_available": cuda_ok, "gpu_count": count, "gpu_names": names}
    except Exception as e:
        return {"cuda_available": False, "gpu_count": 0, "error": str(e)}


def run_benchmark(bundle_path: Path, output_path: Path, models: List[Dict[str, Any]]):
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    telemetry = check_gpu_telemetry()
    logger.info(f"Master Runner starting | Telemetry: {telemetry}")

    model_results = []
    overall_start_time = time.time()

    for m in models:
        m_id = m["id"]
        safe_name = m_id.replace("/", "__")
        worker_out = TMP_DIR / f"worker_{safe_name}.json"

        logger.info(f"Launching isolated worker for {m_id} (tier={m['tier']})...")
        cmd = [
            sys.executable,
            "-m",
            "perron.reranker.scaling_worker",
            "--model_id",
            m_id,
            "--tier",
            m["tier"],
            "--bundle",
            str(bundle_path),
            "--output",
            str(worker_out),
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        t1 = time.time()

        if proc.returncode != 0:
            logger.error(f"Worker for {m_id} failed with exit code {proc.returncode}:\n{proc.stderr}")
            # If in dry-run environment where transformers / weights are not present, log error
            res = {
                "model_id": m_id,
                "name": m["name"],
                "family": m["family"],
                "params_b": m["params_b"],
                "active_params_b": m["active_params_b"],
                "archetype": m["archetype"],
                "tier": m["tier"],
                "status": "failed",
                "error": proc.stderr[:500],
                "elapsed_seconds": t1 - t0,
            }
        else:
            with open(worker_out, "r", encoding="utf-8") as f:
                worker_data = json.load(f)
            res = {**m, **worker_data, "status": "success"}
            logger.info(
                f"Worker for {m_id} succeeded in {t1 - t0:.1f}s | "
                f"Seed AUC={res.get('mean_seed_auc', 0):.4f} | Graph AUC={res.get('mean_graph_auc', 0):.4f}"
            )

        model_results.append(res)

    total_elapsed = time.time() - overall_start_time

    # Compute correlation across successfully evaluated models
    valid_models = [m for m in model_results if m.get("status") == "success"]
    if len(valid_models) >= 3:
        capacities = [m["params_b"] for m in valid_models]
        graph_aucs = [m["mean_graph_auc"] for m in valid_models]
        tau_val, tau_pval = stats.kendalltau(capacities, graph_aucs)
    else:
        tau_val, tau_pval = None, None

    final_payload = {
        "metadata": {
            "protocol_id": "PREREG-SCALING-v1",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "telemetry": telemetry,
            "total_elapsed_seconds": total_elapsed,
            "n_models_evaluated": len(model_results),
            "n_models_succeeded": len(valid_models),
        },
        "hypothesis_tests": {
            "H1_monotonic_scaling": {
                "kendall_tau": float(tau_val) if tau_val is not None else None,
                "p_value": float(tau_pval) if tau_pval is not None else None,
                "supported": bool(tau_val is not None and tau_val >= 0.70 and tau_pval < 0.05),
            },
        },
        "models": model_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    logger.info(f"Master benchmark execution complete. Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Perron Scaling Benchmark Master Runner")
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE), help="Path to evaluation bundle")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output JSON")
    args = parser.parse_args()

    run_benchmark(Path(args.bundle), Path(args.output), MODELS_CONFIG)


if __name__ == "__main__":
    main()
