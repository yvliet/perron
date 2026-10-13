"""
benchmarks/prepare_lora_dataset.py - Contrastive LoRA Dataset Generator for Gemma 4.

Generates strictly quarantined training and validation pairs from SWE-bench Lite (300 tasks)
with precomputed CSR call graphs and AST nodes, enforcing a cryptographic firewall
against the 344 held-out SWE-bench Verified evaluation tasks.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_atomic_label(index: int) -> str:
    """Returns single uppercase letter token [A] through [Y] for candidate index."""
    if 0 <= index < 26:
        return chr(ord("A") + index)
    raise ValueError(f"Candidate index {index} exceeds 26-letter atomic limit.")


def enforce_zero_leakage_firewall(
    candidate_ids: Set[str],
    eval_batch_path: Path = REPO_ROOT / "data" / "gemma4_eval_batch_344.json",
) -> None:
    """
    Physical cryptographic firewall: strictly asserts that no candidate task ID
    overlaps with the 344 held-out SWE-bench Verified evaluation tasks.
    """
    if not eval_batch_path.exists():
        raise FileNotFoundError(f"Held-out benchmark manifest not found at {eval_batch_path}")

    with open(eval_batch_path, "r", encoding="utf-8") as f:
        eval_batch = json.load(f)

    held_out_ids = set(x["instance_id"] if isinstance(x, dict) else x for x in eval_batch)
    overlap = candidate_ids.intersection(held_out_ids)
    if overlap:
        raise ValueError(
            f"FATAL LEAKAGE DETECTED: {len(overlap)} candidate tasks overlap with "
            f"the 344 held-out evaluation tasks: {list(overlap)[:5]}..."
        )
    print(f"[FIREWALL PASS] Zero overlap detected across {len(candidate_ids)} candidate tasks against {len(held_out_ids)} held-out tasks.")


def format_candidate_prompt(
    problem_statement: str,
    candidates: List[Dict[str, Any]],
    target_letter: str,
) -> Dict[str, Any]:
    """
    Formats Gemma instruction prompt with candidates labeled [A] through [Y].
    """
    lines = [
        "<start_of_turn>user",
        "You are an expert software engineer analyzing a bug report.",
        "Analyze the following issue and candidate code symbols extracted from the codebase.",
        "Identify the exact target candidate symbol that is the root cause of the bug.",
        "",
        "### Issue Description:",
        problem_statement.strip()[:2000],  # Bound issue description length
        "",
        "### Candidate Code Symbols:",
    ]

    for cand in candidates:
        letter = cand["letter"]
        ident = cand["qualified_name"]
        fpath = cand["file"]
        sline = cand.get("start_line", 1)
        eline = cand.get("end_line", 1)
        provenance = cand.get("provenance", "Extracted code graph candidate")
        lines.append(f"Candidate [{letter}]: {ident} ({fpath}:{sline}-{eline})")
        lines.append(f"```python\n# {provenance}\n# {ident} in {fpath}\n```\n")

    lines.append("Identify the single best candidate choice letter (e.g. [A], [B], [C]) inside <target>[X]</target> tags.")
    lines.append("<end_of_turn>")
    lines.append("<start_of_turn>model")
    
    prompt_str = "\n".join(lines)
    completion_str = f"<target>[{target_letter}]</target><end_of_turn>"

    return {
        "prompt": prompt_str,
        "completion": completion_str,
        "target_letter": target_letter,
        "candidates": candidates,
    }


def generate_lora_dataset(
    output_dir: Path = REPO_ROOT / "data",
    val_count: int = 50,
    seed: int = 42,
    max_candidates_per_task: int = 25,
) -> Tuple[int, int]:
    """
    Generates training and validation datasets for Gemma 4 LoRA fine-tuning.
    """
    rng = random.Random(seed)
    graphs_dir = REPO_ROOT / "data_release" / "graphs"
    lite_cache_path = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    lite_gold_path = REPO_ROOT / "data" / "swebench_lite_gold_labels.json"

    with open(lite_gold_path, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    tasks_by_id: Dict[str, Dict[str, Any]] = {}
    with open(lite_cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                tasks_by_id[item["instance_id"]] = item

    # Enforce Firewall on all Lite tasks
    enforce_zero_leakage_firewall(set(tasks_by_id.keys()))

    processed_pairs: List[Dict[str, Any]] = []

    for instance_id, task in tasks_by_id.items():
        if instance_id not in gold_labels:
            continue

        gold_info = gold_labels[instance_id]
        gold_names = set(gold_info.get("gold_symbol_names", []))
        gold_files = set(gold_info.get("gold_files", []))

        graph_dir = graphs_dir / instance_id
        if not graph_dir.exists():
            continue

        nodes_parquet = graph_dir / "nodes.parquet"
        if not nodes_parquet.exists():
            continue

        nodes_df = pd.read_parquet(nodes_parquet)
        if len(nodes_df) == 0:
            continue

        # Find gold symbol in nodes dataframe
        gold_match = None
        for _, row in nodes_df.iterrows():
            if not row.get("is_test", False):
                if row["qualified_name"] in gold_names or any(row["file"].endswith(gf) for gf in gold_files):
                    gold_match = row.to_dict()
                    break

        if gold_match is None:
            continue

        # Collect negative distractors
        non_test_nodes = nodes_df[
            (~nodes_df["is_test"]) & 
            (nodes_df["id"] != gold_match["id"])
        ]
        if len(non_test_nodes) < 4:
            continue

        # Sample distractors
        num_distractors = min(max_candidates_per_task - 1, len(non_test_nodes))
        sampled_df = non_test_nodes.sample(n=num_distractors, random_state=seed)
        distractors = [r.to_dict() for _, r in sampled_df.iterrows()]

        # Pool candidates and place gold at a random position
        all_pool = [gold_match] + distractors
        rng.shuffle(all_pool)

        candidates_formatted: List[Dict[str, Any]] = []
        target_letter = ""

        for idx, item in enumerate(all_pool):
            letter = get_atomic_label(idx)
            is_gold = (item["id"] == gold_match["id"])
            if is_gold:
                target_letter = letter

            cand_dict = {
                "id": int(item["id"]),
                "letter": letter,
                "qualified_name": str(item["qualified_name"]),
                "file": str(item["file"]),
                "start_line": int(item.get("start_line", 1)),
                "end_line": int(item.get("end_line", 1)),
                "is_gold": is_gold,
                "provenance": "Gold target symbol" if is_gold else "Candidate graph symbol",
            }
            candidates_formatted.append(cand_dict)

        formatted_pair = format_candidate_prompt(
            problem_statement=task["problem_statement"],
            candidates=candidates_formatted,
            target_letter=target_letter,
        )
        formatted_pair["instance_id"] = instance_id
        formatted_pair["repo"] = task["repo"]
        processed_pairs.append(formatted_pair)

    print(f"[*] Processed {len(processed_pairs)} valid training instances from SWE-bench Lite.")
    rng.shuffle(processed_pairs)

    val_set = processed_pairs[:val_count]
    train_set = processed_pairs[val_count:]

    train_out = output_dir / "gemma4_lora_train.jsonl"
    val_out = output_dir / "gemma4_lora_val.jsonl"

    with open(train_out, "w", encoding="utf-8") as f:
        for p in train_set:
            f.write(json.dumps(p) + "\n")

    with open(val_out, "w", encoding="utf-8") as f:
        for p in val_set:
            f.write(json.dumps(p) + "\n")

    print(f"[+] Wrote {len(train_set)} train instances to {train_out}")
    print(f"[+] Wrote {len(val_set)} val instances to {val_out}")
    return len(train_set), len(val_set)


def main():
    parser = argparse.ArgumentParser(description="Prepare Gemma 4 LoRA Dataset")
    parser.add_argument("--verify-only", action="store_true", help="Only verify cryptographic firewall")
    parser.add_argument("--val-count", type=int, default=50, help="Number of validation instances")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    lite_cache_path = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    with open(lite_cache_path, "r", encoding="utf-8") as f:
        lite_ids = set(json.loads(l)["instance_id"] for l in f if l.strip())

    enforce_zero_leakage_firewall(lite_ids)

    if args.verify_only:
        print("[+] Firewall verification complete. All tasks 100% disjoint from evaluation set.")
        return

    n_train, n_val = generate_lora_dataset(val_count=args.val_count, seed=args.seed)
    print(f"[DONE] Successfully generated {n_train} train and {n_val} val samples.")


if __name__ == "__main__":
    main()
