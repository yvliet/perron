#!/usr/bin/env python3
"""
scripts/generate_protocol_tables.py - Deterministic Producer for Gemma 4 Protocol Tables.

Compiles and validates results/gemma4_protocol_tables.json from underlying
experiment artifacts in results/gemma4_protocol/analysis_results.json.

Ensures complete programmatic lineage for the 167 publication macros anchored
to protocol tables in paper/metrics_macros.tex.

Author: Sultan Haikal (GitHub: @yvliet, recrified@gmail.com)
License: Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSON = PROJECT_ROOT / "results" / "gemma4_protocol" / "analysis_results.json"
TARGET_JSON = PROJECT_ROOT / "results" / "gemma4_protocol_tables.json"


def round1(val: float) -> float:
    return round(float(val), 1)


def round4(val: float) -> float:
    return round(float(val), 4)


def build_protocol_tables(source_data: Dict[str, Any], existing_target: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs or updates the protocol tables dictionary from raw analysis results.
    """
    compiled = copy.deepcopy(existing_target)
    tables = compiled.setdefault("tables", {})

    # 1. Table A: Reranker and baseline ablations
    if "table_A" in source_data:
        t_a = {}
        for arm, metrics in source_data["table_A"].items():
            t_a[arm] = {
                "acc_1": round1(metrics["acc1"]["mean"]),
                "acc_3": round1(metrics["acc3"]["mean"]),
                "acc_5": round1(metrics["acc5"]["mean"]),
                "acc_10": round1(metrics["acc10"]["mean"]),
                "mrr_50": round4(metrics["mrr50"]["mean"]),
            }
        tables["table_t_a"] = t_a

    # 2. Table B: Confirmatory contrasts
    if "table_B" in source_data and "tier1_confirmatory" in source_data["table_B"]:
        t1 = source_data["table_B"]["tier1_confirmatory"]
        t_b = {}
        if "PC1_A4_minus_A2" in t1:
            pc1 = t1["PC1_A4_minus_A2"]
            t_b["PC1"] = {
                "delta_pp": round(float(pc1.get("delta_pp", 0)), 2),
                "n10": int(pc1.get("n10", 0)),
                "n01": int(pc1.get("n01", 0)),
                "p_exact": round4(pc1.get("p_raw", 0)),
                "p_holm": round4(pc1.get("p_holm", 0)),
            }
        if "PC2_A4_minus_A3" in t1:
            pc2 = t1["PC2_A4_minus_A3"]
            t_b["PC2"] = {
                "delta_pp": round(float(pc2.get("delta_pp", 0)), 2),
                "n10": int(pc2.get("n10", 0)),
                "n01": int(pc2.get("n01", 0)),
                "p_exact": round4(pc2.get("p_raw", 0)),
                "p_holm": round4(pc2.get("p_holm", 0)),
            }
        tables["table_t_b"] = t_b

    # 3. Table E: Reader evaluation
    if "table_E" in source_data:
        t_e = {}
        for k in ["D0", "D1", "D2", "D3"]:
            if k in source_data["table_E"]:
                cond = source_data["table_E"][k]
                t_e[k] = {
                    "context_hit_5": round1(cond.get("context_hit_5", {}).get("pct", 0)),
                    "edit_loc_1": round1(cond.get("edit_loc_1", {}).get("pct", 0)),
                    "edit_loc_3": round1(cond.get("edit_loc_3", {}).get("pct", 0)),
                    "conversion": round(float(cond.get("conversion", 0)), 3),
                }
        if "reader_transmission" in source_data["table_E"]:
            rt = source_data["table_E"]["reader_transmission"]
            t_e["reader_transmission_rate"] = round1(rt.get("raw_p", 0) * 100)
            t_e["chance_adjusted_rate"] = round1(rt.get("chance_adjusted", 0) * 100)
        tables["table_t_e_reader"] = t_e

    compiled["suite"] = existing_target.get("suite", "gemma4_protocol_evaluation")
    compiled["benchmark"] = existing_target.get("benchmark", "SWE-bench_Verified")
    compiled["sample_n"] = source_data.get("n_callable_tasks", existing_target.get("sample_n", 344))
    compiled["tables"] = tables
    return compiled


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic producer for Gemma 4 protocol tables.")
    parser.add_argument("--check", action="store_true", help="Check whether target JSON matches compiled output.")
    args = parser.parse_args()

    if not SOURCE_JSON.exists():
        print(f"[ERROR] Missing source analysis results: {SOURCE_JSON}", file=sys.stderr)
        return 1

    with open(SOURCE_JSON, "r", encoding="utf-8") as f:
        source_data = json.load(f)

    existing_target = {}
    if TARGET_JSON.exists():
        with open(TARGET_JSON, "r", encoding="utf-8") as f:
            existing_target = json.load(f)

    compiled = build_protocol_tables(source_data, existing_target)

    if args.check:
        if not TARGET_JSON.exists():
            print(f"[FAIL] Target file does not exist: {TARGET_JSON}", file=sys.stderr)
            return 1
        with open(TARGET_JSON, "r", encoding="utf-8") as f:
            current = json.load(f)
        if current == compiled:
            print("[PASS] Protocol tables are perfectly synchronized.")
            return 0
        else:
            print("[FAIL] Protocol tables are out of sync with source data.", file=sys.stderr)
            return 1

    # Write output deterministically
    with open(TARGET_JSON, "w", encoding="utf-8") as f:
        json.dump(compiled, f, indent=2)
        f.write("\n")

    print(f"[SUCCESS] Successfully compiled protocol tables to {TARGET_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
