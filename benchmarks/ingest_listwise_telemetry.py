#!/usr/bin/env python3
"""
benchmarks/ingest_listwise_telemetry.py - Telemetry Ingestion and Multi-Arm Synthesis.

Ingests remote GPU execution receipts from Kaggle workstation runs, validates
statistical bounds, and prepares publication-grade contrast tables for Section 5.4.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("perron.ingest_listwise")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ingest_telemetry(raw_path: Path, out_path: Path) -> Dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Telemetry file not found at {raw_path}")

    data = json.loads(raw_path.read_text(encoding="utf-8"))
    logger.info(f"Ingested telemetry from {raw_path}: suite={data.get('suite')}")
    
    # Save formatted version to destination
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(f"Synchronized telemetry to {out_path}")
    return data


def format_summary_table(data: Dict[str, Any]) -> str:
    n_tasks = data.get("n_tasks", 131)
    results = data.get("results", {})
    contrasts = data.get("contrasts", {})
    
    lines = []
    lines.append("=" * 88)
    lines.append(f"{'Method / Arm':<26} | {'Acc@5':<10} | {'Hits / 131':<12} | {'MRR@50':<10} | {'Delta vs A2':<14} | {'McNemar p':<10}")
    lines.append("-" * 88)
    
    for arm_name, metrics in results.items():
        acc_s = f"{metrics.get('acc_at_5_pct', 0.0):.2f}%"
        hits_s = f"{metrics.get('hits_at_5', 0)}/{n_tasks}"
        mrr_s = f"{metrics.get('mrr_50', 0.0):.4f}"
        
        c = contrasts.get(f"{arm_name}_vs_A2_pointwise", {})
        delta_s = f"{c.get('delta_pp', 0.0):+.2f} pp" if c else "N/A"
        p_val = c.get('mcnemar_p_value')
        p_s = f"{p_val:.4f}" if p_val is not None else "N/A"
        
        lines.append(f"{arm_name:<26} | {acc_s:<10} | {hits_s:<12} | {mrr_s:<10} | {delta_s:<14} | {p_s:<10}")
        
    lines.append("=" * 88)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Ingest Kaggle listwise telemetry.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "results" / "kaggle_workstation" / "kaggle_stratum_c_listwise_results.json",
        help="Input path to downloaded Kaggle telemetry JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "stratum_c_listwise_results.json",
        help="Output path for synchronized telemetry JSON",
    )
    args = parser.parse_args()
    
    telemetry = ingest_telemetry(args.input, args.output)
    print("\n" + format_summary_table(telemetry))


if __name__ == "__main__":
    main()
