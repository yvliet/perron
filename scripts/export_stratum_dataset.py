#!/usr/bin/env python3
"""
scripts/export_stratum_dataset.py - Lexical Accessibility Strata Dataset Exporter.

Classifies SWE-bench Lite (N=300) and SWE-bench Verified (N=500) instances into
three canonical lexical accessibility strata:
- Stratum A (A_named): Gold symbol/method identifier is explicitly named in the prompt.
- Stratum B (B_module / B_trace): Enclosing file or module is named (e.g. in tracebacks), but gold symbol is omitted.
- Stratum C (C_latent / C_unnamed): Neither gold symbol nor enclosing file appears in the prompt.

Exports the annotated benchmark dataset to:
data/swebench_strata_annotations.jsonl

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

from perron.analysis.strata import (
    STRATA_LABELS,
    STRATUM_A,
    STRATUM_B,
    STRATUM_C,
    classify_stratum,
)


def load_lite_tasks(root: Path) -> List[Dict[str, Any]]:
    lite_path = root / "data" / "swebench_lite_cache.jsonl"
    if not lite_path.exists():
        raise FileNotFoundError(f"Missing Lite cache at {lite_path}")
    with open(lite_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold_labels(root: Path) -> Dict[str, Dict[str, Any]]:
    lite_labels_path = root / "data" / "swebench_lite_gold_labels.json"
    verif_labels_path = root / "data" / "swebench_verified_gold_labels.json"

    labels: Dict[str, Dict[str, Any]] = {}
    if lite_labels_path.exists():
        with open(lite_labels_path, "r", encoding="utf-8") as f:
            labels.update(json.load(f))
    if verif_labels_path.exists():
        with open(verif_labels_path, "r", encoding="utf-8") as f:
            labels.update(json.load(f))
    return labels


def load_verified_tasks(root: Path) -> List[Dict[str, Any]]:
    parquet_path = root / "scratch" / "swebench_verified.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            tasks: List[Dict[str, Any]] = []
            for _, row in df.iterrows():
                tasks.append({
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "problem_statement": row["problem_statement"],
                    "hints_text": row.get("hints_text", "") or "",
                    "base_commit": row.get("base_commit", ""),
                })
            return tasks
        except ImportError:
            pass

    # Fallback to pre-cached tasks file + Lite overlap
    tasks_file = root / "data" / "swebench_verified_tasks.jsonl"
    tasks: List[Dict[str, Any]] = []
    seen = set()
    if tasks_file.exists():
        with open(tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    tasks.append(item)
                    seen.add(item["instance_id"])

    # Supplement overlapping Lite tasks that belong to SWE-bench Verified
    lite_tasks = load_lite_tasks(root)
    for lt in lite_tasks:
        if lt["instance_id"] not in seen:
            tasks.append(lt)
            seen.add(lt["instance_id"])

    return tasks


def export_strata_dataset(
    output_path: Optional[Path] = None,
    include_lite: bool = True,
    include_verified: bool = True,
) -> Path:
    if output_path is None:
        output_path = REPO_ROOT / "data" / "swebench_strata_annotations.jsonl"

    labels = load_gold_labels(REPO_ROOT)
    records: List[Dict[str, Any]] = []

    counts_lite = {STRATUM_A: 0, STRATUM_B: 0, STRATUM_C: 0}
    counts_verif = {STRATUM_A: 0, STRATUM_B: 0, STRATUM_C: 0}

    # 1. Process SWE-bench Lite
    if include_lite:
        lite_tasks = load_lite_tasks(REPO_ROOT)
        for task in lite_tasks:
            iid = task["instance_id"]
            issue = task.get("problem_statement", "")
            gold_info = labels.get(iid, {})
            gold_files = gold_info.get("gold_files", [])
            gold_symbols = gold_info.get("gold_symbol_names", [])

            stratum = classify_stratum(issue, gold_files, gold_symbols)
            counts_lite[stratum] += 1

            records.append({
                "instance_id": iid,
                "benchmark": "lite",
                "repo": task.get("repo", ""),
                "stratum": stratum,
                "stratum_label": STRATA_LABELS.get(stratum, stratum),
                "gold_targets": gold_symbols,
                "gold_files": gold_files,
                "problem_statement_snippet": issue[:200].strip(),
            })

    # 2. Process SWE-bench Verified
    if include_verified:
        verif_tasks = load_verified_tasks(REPO_ROOT)
        for task in verif_tasks:
            iid = task["instance_id"]
            issue = task.get("problem_statement", "")
            gold_info = labels.get(iid, {})
            gold_files = gold_info.get("gold_files", [])
            gold_symbols = gold_info.get("gold_symbol_names", [])

            stratum = classify_stratum(issue, gold_files, gold_symbols)
            counts_verif[stratum] += 1

            records.append({
                "instance_id": iid,
                "benchmark": "verified",
                "repo": task.get("repo", ""),
                "stratum": stratum,
                "stratum_label": STRATA_LABELS.get(stratum, stratum),
                "gold_targets": gold_symbols,
                "gold_files": gold_files,
                "problem_statement_snippet": issue[:200].strip(),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Exported {len(records)} total records to {output_path}")
    if include_lite:
        n_l = sum(counts_lite.values())
        print(f"  SWE-bench Lite (N={n_l}):")
        print(f"    Stratum A (Named):   {counts_lite[STRATUM_A]:>3} ({counts_lite[STRATUM_A]/n_l*100:5.1f}%)")
        print(f"    Stratum B (Module):  {counts_lite[STRATUM_B]:>3} ({counts_lite[STRATUM_B]/n_l*100:5.1f}%)")
        print(f"    Stratum C (Latent):  {counts_lite[STRATUM_C]:>3} ({counts_lite[STRATUM_C]/n_l*100:5.1f}%)")

    if include_verified:
        n_v = sum(counts_verif.values())
        print(f"  SWE-bench Verified (N={n_v}):")
        print(f"    Stratum A (Named):   {counts_verif[STRATUM_A]:>3} ({counts_verif[STRATUM_A]/n_v*100:5.1f}%)")
        print(f"    Stratum B (Module):  {counts_verif[STRATUM_B]:>3} ({counts_verif[STRATUM_B]/n_v*100:5.1f}%)")
        print(f"    Stratum C (Latent):  {counts_verif[STRATUM_C]:>3} ({counts_verif[STRATUM_C]/n_v*100:5.1f}%)")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SWE-bench Lexical Accessibility Strata Dataset")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output JSONL file path")
    parser.add_argument("--lite-only", action="store_true", help="Export only SWE-bench Lite")
    parser.add_argument("--verified-only", action="store_true", help="Export only SWE-bench Verified")
    args = parser.parse_args()

    include_lite = not args.verified_only
    include_verified = not args.lite_only

    export_strata_dataset(
        output_path=args.output,
        include_lite=include_lite,
        include_verified=include_verified,
    )


if __name__ == "__main__":
    main()
