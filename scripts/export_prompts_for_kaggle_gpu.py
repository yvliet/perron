"""
Export 19,298 pre-rendered candidate prompts for live Gemma 4 E2B GPU inference.

Matches the exact row set from results/scores.parquet (344 SWE-bench Verified tasks).
Outputs data/gemma4_e2b_prompts_19298.parquet.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import pandas as pd

from perron.reranker.scorer import PointwiseReranker
from benchmarks.gold_patch_parser import get_repo_slug

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_prompts")


def export_prompts(
    scores_parquet_path: Path = Path("results/scores.parquet"),
    tasks_file: Path = Path("data/swebench_verified_tasks.jsonl"),
    gold_file: Path = Path("data/swebench_verified_gold_labels.json"),
    graphs_dir: Path = Path("artifacts/swebench_graphs"),
    out_parquet: Path = Path("data/gemma4_e2b_prompts_19298.parquet"),
) -> Path:
    logger.info("Loading reference candidate pairs from %s...", scores_parquet_path)
    df_ref = pd.read_parquet(scores_parquet_path)
    total_rows = len(df_ref)
    logger.info("Total rows to render: %d across %d tasks", total_rows, df_ref["instance_id"].nunique())

    logger.info("Loading tasks and gold labels...")
    tasks_by_id = {}
    with open(tasks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                tasks_by_id[d["instance_id"]] = d

    with open(gold_file, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    # Cache repo symbols to prevent redundant file reads
    symbols_cache = {}

    def get_symbol_dict(slug: str, candidate_id: int):
        if slug not in symbols_cache:
            sym_path = graphs_dir / f"{slug}_symbols.json"
            if not sym_path.exists():
                raise FileNotFoundError(f"Symbols file not found: {sym_path}")
            symbols_cache[slug] = json.loads(sym_path.read_text(encoding="utf-8"))
        sym_list = symbols_cache[slug]
        if candidate_id >= len(sym_list):
            raise IndexError(f"Candidate id {candidate_id} exceeds {len(sym_list)} symbols for {slug}")
        return sym_list[candidate_id]

    reranker = PointwiseReranker()
    rendered_rows = []
    t_start = time.perf_counter()

    for idx, row in df_ref.iterrows():
        iid = str(row["instance_id"])
        cid = int(row["candidate_id"])
        variant = str(row.get("prompt_variant", "P0"))

        task = tasks_by_id.get(iid)
        if not task:
            raise KeyError(f"Task {iid} not found in {tasks_file}")

        slug = get_repo_slug(task.get("repo", ""))
        sym = get_symbol_dict(slug, cid)

        issue_text = (task.get("problem_statement") or "") + " " + (task.get("hints_text") or "")
        raw_prompt, _, _ = reranker.render_prompt(issue_text, sym, slug, variant=variant)

        gold_ids = set(gold_labels.get(iid, {}).get("gold_symbol_ids", []))
        is_gold = 1 if cid in gold_ids else 0

        rendered_rows.append({
            "instance_id": iid,
            "candidate_id": cid,
            "prompt_variant": variant,
            "symbol_name": sym.get("identifier", ""),
            "file_path": sym.get("file_path", ""),
            "raw_prompt": raw_prompt,
            "is_gold": is_gold,
        })

        if (idx + 1) % 2500 == 0 or (idx + 1) == total_rows:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / max(elapsed, 1e-5)
            logger.info("Rendered %d / %d prompts (%.1f prompts/s, %.1fs elapsed)", idx + 1, total_rows, rate, elapsed)

    df_out = pd.DataFrame(rendered_rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_parquet, index=False, compression="snappy")
    file_size_mb = out_parquet.stat().st_size / (1024 * 1024)
    logger.info("Successfully exported %d prompts to %s (%.2f MB)", len(df_out), out_parquet, file_size_mb)
    return out_parquet


if __name__ == "__main__":
    export_prompts()
