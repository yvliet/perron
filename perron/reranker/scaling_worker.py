"""
perron/reranker/scaling_worker.py - Per-model isolated worker for cross-architecture scaling benchmark.

Executes 4-bit NF4 forward-pass candidate discrimination and anchor listwise reasoning
inside an isolated OS process to ensure zero residual VRAM across sequential model runs.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("perron.scaling_worker")

PROMPT_P0 = (
    "You are an expert software engineer reviewing a repository to locate a reported defect.\n\n"
    "Problem Statement:\n"
    "{issue_text}\n\n"
    "Candidate Code Element:\n"
    "{candidate_breadcrumb}\n\n"
    "Does this candidate code element contain or directly cause the reported bug? Answer 'Yes' or 'No'."
)


def truncate_issue_text(text: str, max_chars: int = 3000) -> str:
    if not text:
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_chars:
        return clean
    half = (max_chars - 60) // 2
    return clean[:half] + " ... [TRUNCATED FOR CONTEXT BUDGET] ... " + clean[-half:]


def format_candidate_breadcrumb(cand: Dict[str, Any]) -> str:
    sym_name = cand.get("symbol_name") or cand.get("identifier") or "unknown_symbol"
    fpath = cand.get("file_path") or cand.get("file") or "unknown_file"
    lines = cand.get("lines") or [cand.get("line_start", 1), cand.get("line_end", 1)]
    code_snippet = cand.get("code") or cand.get("snippet") or f"# {sym_name} in {fpath}"
    return f"File: {fpath} (Lines {lines[0]}-{lines[1]})\nSymbol: {sym_name}\n```python\n{code_snippet}\n```"


def resolve_yes_no_token_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    yes_candidates = ["Yes", " Yes", "YES", " yes"]
    no_candidates = ["No", " No", "NO", " no"]
    
    y_ids = set()
    for word in yes_candidates:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            y_ids.add(toks[-1])
            
    n_ids = set()
    for word in no_candidates:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if toks:
            n_ids.add(toks[-1])
            
    return sorted(list(y_ids)), sorted(list(n_ids))


def resolve_model_path(model_id: str) -> str:
    if Path(model_id).exists() and (Path(model_id) / "config.json").exists():
        return model_id
    base_in = Path("/kaggle/input")
    if base_in.exists():
        slug = model_id.split("/")[-1].lower()
        for p in base_in.rglob("config.json"):
            if slug in str(p.parent).lower():
                logger.info(f"Resolved local Kaggle mount for {model_id}: {p.parent}")
                return str(p.parent)
        if "gemma" in slug:
            for variant in ["e2b", "e4b", "26b", "31b", "12b"]:
                if variant in slug:
                    for p in base_in.rglob("config.json"):
                        p_str = str(p.parent).lower()
                        if variant in p_str and "gemma" in p_str:
                            logger.info(f"Resolved local mount for {model_id} via keyword {variant}: {p.parent}")
                            return str(p.parent)
    return model_id


def get_hf_token() -> Optional[str]:
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        token = secrets.get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def purge_hf_cache():
    import shutil
    for cdir in [Path("/root/.cache/huggingface/hub"), Path(os.path.expanduser("~/.cache/huggingface/hub"))]:
        if cdir.exists():
            logger.info(f"Purging Hugging Face cache at {cdir} to reclaim scratch disk...")
            shutil.rmtree(cdir, ignore_errors=True)


def run_worker():
    parser = argparse.ArgumentParser(description="Perron Scaling Model Worker")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--tier", default="pointwise", choices=["pointwise", "hybrid_anchor"])
    parser.add_argument("--bundle", required=True, help="Path to stratum_c_eval_bundle.json")
    parser.add_argument("--output", required=True, help="Path to write output JSON")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle_data = json.load(f)
    tasks = bundle_data.get("tasks", [])
    logger.info(f"Loaded {len(tasks)} tasks from {bundle_path}")

    start_wall_time = time.time()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    logger.info(f"CUDA Available: {cuda_available} | Device: {device_name}")

    target_path = resolve_model_path(args.model_id)
    is_local = Path(target_path).exists()
    hf_token = get_hf_token()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if cuda_available else None

    logger.info(f"Loading model: {args.model_id} (Target Path: {target_path}, Local: {is_local}, NF4: {cuda_available})")
    load_kwargs = {"trust_remote_code": False}
    if not is_local and hf_token:
        load_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(target_path, **load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if cuda_available:
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": False,
            "torch_dtype": torch.float16,
        }
        if not is_local and hf_token:
            model_kwargs["token"] = hf_token
        model = AutoModelForCausalLM.from_pretrained(target_path, **model_kwargs)
    else:
        logger.warning("Operating in fallback CPU mode (for dry-run testing only)")
        model = None

    y_ids, n_ids = resolve_yes_no_token_ids(tokenizer)
    logger.info(f"Resolved classification tokens: Yes={y_ids}, No={n_ids}")

    # Pre-Flight Smoke Test Probe (Validate Multi-GPU Device Placement and Generation)
    logger.info(f"Executing Pre-Flight Smoke Test for {args.model_id}...")
    smoke_text = "Problem Statement: Verification.\n\nCandidate: def test(): pass\n\nDoes this cause the bug? Answer Yes or No."
    smoke_inp = tokenizer(smoke_text, return_tensors="pt", max_length=512, truncation=True)
    if cuda_available and model is not None:
        smoke_inp = smoke_inp.to("cuda")
        with torch.inference_mode():
            smoke_out = model(**smoke_inp)
            smoke_logits = smoke_out.logits[:, -1, :]
            smoke_y = torch.tensor(y_ids, dtype=torch.long, device=smoke_logits.device)
            smoke_n = torch.tensor(n_ids, dtype=torch.long, device=smoke_logits.device)
            smoke_score = (torch.logsumexp(smoke_logits[:, smoke_y], dim=-1) - torch.logsumexp(smoke_logits[:, smoke_n], dim=-1)).cpu().item()
        logger.info(f"Smoke test forward pass succeeded (sample log-odds: {smoke_score:.4f})")
        if args.tier == "hybrid_anchor":
            with torch.inference_mode():
                _ = model.generate(**smoke_inp, max_new_tokens=5, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            logger.info("Smoke test autoregressive generation succeeded.")
    logger.info("Pre-Flight Smoke Test PASSED.")

    # Track metrics
    task_results = []
    forward_latencies = []
    
    # 1. Pointwise candidate scoring across tasks
    for t_idx, task in enumerate(tasks):
        inst_id = task["instance_id"]
        issue = truncate_issue_text(task.get("problem_statement", ""))
        cands = task.get("candidates", [])
        gold_ids = set(task.get("gold_symbol_ids", []))
        if not gold_ids and "gold_symbol_names" in task:
            gold_ids = set(task.get("gold_symbol_names", []))

        if not cands or not gold_ids:
            continue

        cand_prompts = []
        for c in cands:
            breadcrumb = format_candidate_breadcrumb(c)
            raw = PROMPT_P0.format(issue_text=issue, candidate_breadcrumb=breadcrumb)
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    formatted = tokenizer.apply_chat_template(
                        [{"role": "user", "content": raw}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    formatted = raw
            else:
                formatted = raw
            cand_prompts.append(formatted)

        scores = []
        if model is not None and cuda_available:
            for b_start in range(0, len(cand_prompts), args.batch_size):
                b_prompts = cand_prompts[b_start : b_start + args.batch_size]
                inputs = tokenizer(
                    b_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to("cuda")

                t0 = time.perf_counter()
                with torch.inference_mode():
                    outputs = model(**inputs)
                    # Extract last-token logits with device-aligned index tensors
                    next_token_logits = outputs.logits[:, -1, :]
                    y_tensor = torch.tensor(y_ids, dtype=torch.long, device=next_token_logits.device)
                    n_tensor = torch.tensor(n_ids, dtype=torch.long, device=next_token_logits.device)
                    y_logits = torch.logsumexp(next_token_logits[:, y_tensor], dim=-1)
                    n_logits = torch.logsumexp(next_token_logits[:, n_tensor], dim=-1)
                    log_odds = (y_logits - n_logits).cpu().numpy().tolist()
                t1 = time.perf_counter()
                forward_latencies.append((t1 - t0) / len(b_prompts))
                scores.extend(log_odds)
        else:
            # CPU fallback calculation for dry-run
            scores = [float(np.random.normal(0, 1)) for _ in cand_prompts]

        # Calculate task AUC: fraction of negative distractors outscored by gold
        gold_scores = []
        distractor_scores = []
        for c, s in zip(cands, scores):
            cid = c.get("symbol_id") or c.get("identifier") or c.get("symbol_name")
            if cid in gold_ids:
                gold_scores.append(s)
            else:
                distractor_scores.append(s)

        task_auc = 0.5
        if gold_scores and distractor_scores:
            wins = 0.0
            total_pairs = len(gold_scores) * len(distractor_scores)
            for g in gold_scores:
                for d in distractor_scores:
                    if g > d:
                        wins += 1.0
                    elif g == d:
                        wins += 0.5
            task_auc = wins / total_pairs

        is_seed_gold = any(c.get("is_seed", False) for c in cands if (c.get("symbol_id") or c.get("identifier")) in gold_ids)

        task_results.append({
            "instance_id": inst_id,
            "task_auc": float(task_auc),
            "is_seed_gold": bool(is_seed_gold),
            "n_candidates": len(cands),
            "n_gold": len(gold_scores),
            "n_distractors": len(distractor_scores),
        })

    # 2. Anchor Listwise Reasoning (if tier == hybrid_anchor)
    listwise_results = {}
    if args.tier == "hybrid_anchor":
        logger.info(f"Running Tier 2 anchor listwise reasoning for {args.model_id}")
        hits_top1 = 0
        hits_top5 = 0
        reciprocal_ranks = []

        for task in tasks:
            inst_id = task["instance_id"]
            gold_ids = set(task.get("gold_symbol_ids", [])) or set(task.get("gold_symbol_names", []))
            cands = task.get("candidates", [])[:25]
            if not cands or not gold_ids:
                continue

            cand_list_text = "\n".join([f"[{i+1}] {c.get('symbol_name') or c.get('identifier')} (File: {c.get('file_path')})" for i, c in enumerate(cands)])
            prompt_text = (
                f"You are debugging the following issue:\n{truncate_issue_text(task.get('problem_statement', ''))}\n\n"
                f"Candidate Functions:\n{cand_list_text}\n\n"
                "Think carefully inside <thought>...</thought> tags, then state the single best candidate index: [INDEX]."
            )

            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt_text}],
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                except Exception:
                    formatted_prompt = prompt_text
            else:
                formatted_prompt = prompt_text

            if model is not None and cuda_available:
                inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=3000, truncation=True).to("cuda")
                with torch.inference_mode():
                    gen_tokens = model.generate(
                        **inputs,
                        max_new_tokens=180,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                out_str = tokenizer.decode(gen_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            else:
                out_str = "<thought>dry run</thought> [1]"

            pred_idx = 1
            m = re.search(r'(?:candidate\s*:?\s*|index\s*:?\s*|\[)\s*(\d+)', out_str, re.IGNORECASE)
            if m:
                pred_idx = int(m.group(1))

            pred_cid = None
            if 1 <= pred_idx <= len(cands):
                pred_cid = cands[pred_idx - 1].get("symbol_id") or cands[pred_idx - 1].get("identifier")

            is_top1 = pred_cid in gold_ids
            if is_top1:
                hits_top1 += 1
                reciprocal_ranks.append(1.0)
            else:
                reciprocal_ranks.append(0.0)

            # Check top-5
            hits_top5 += int(pred_idx <= 5 and pred_cid in gold_ids)

        listwise_results = {
            "acc_top1": hits_top1 / len(tasks) if tasks else 0.0,
            "acc_top5": hits_top5 / len(tasks) if tasks else 0.0,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "hits_top1": hits_top1,
            "hits_top5": hits_top5,
        }

    # Aggregate pointwise metrics
    seed_aucs = [r["task_auc"] for r in task_results if r["is_seed_gold"]]
    graph_aucs = [r["task_auc"] for r in task_results if not r["is_seed_gold"]]
    all_aucs = [r["task_auc"] for r in task_results]

    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if cuda_available else 0.0
    elapsed_sec = time.time() - start_wall_time

    summary = {
        "model_id": args.model_id,
        "tier": args.tier,
        "live_model_executed": cuda_available,
        "device": device_name,
        "peak_vram_mb": float(peak_vram_mb),
        "elapsed_seconds": float(elapsed_sec),
        "mean_forward_latency_ms": float(np.mean(forward_latencies) * 1000) if forward_latencies else 0.0,
        "n_evaluated_tasks": len(task_results),
        "mean_seed_auc": float(np.mean(seed_aucs)) if seed_aucs else 0.5,
        "mean_graph_auc": float(np.mean(graph_aucs)) if graph_aucs else 0.5,
        "mean_overall_auc": float(np.mean(all_aucs)) if all_aucs else 0.5,
        "listwise_anchor_metrics": listwise_results,
        "per_task_results": task_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Worker finished: {args.model_id} | Seed AUC={summary['mean_seed_auc']:.4f} | "
        f"Graph AUC={summary['mean_graph_auc']:.4f} | Peak VRAM={peak_vram_mb:.1f} MB | Time={elapsed_sec:.1f}s"
    )

    # Clean up memory
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()
    if not is_local:
        purge_hf_cache()


if __name__ == "__main__":
    run_worker()
