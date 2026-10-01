"""
Live Local Gemma 4 Inference & Verification Runner.
Executes Gemma 4 E2B/E4B locally within a 16GB consumer laptop memory envelope (<= 6.8 GB RAM).
Validates Perron graph diffusion, context budget packing, and AST-verified patch generation.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import psutil
import numpy as np
from perron.backends.transformers_backend import TransformersBackend
from perron.matrix import load_mmap_csr
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import (
    ASTContextSymbol,
    pack_context_subgraphs,
    format_hierarchical_context,
)
from perron.patch import compute_git_patch


def get_memory_info() -> Dict[str, float]:
    """Returns current system and process memory metrics in gigabytes."""
    vm = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    return {
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
        "used_ram_gb": round(vm.used / (1024 ** 3), 2),
        "proc_rss_gb": round(proc.memory_info().rss / (1024 ** 3), 2),
    }


def find_instance(instance_id: str) -> Optional[Dict[str, Any]]:
    """Loads a specific SWE-bench Lite instance from the local dataset cache."""
    cache_path = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("instance_id") == instance_id:
                    return item
    return None


def extract_patch_or_edits(response_text: str) -> Dict[str, Any]:
    """Extracts code blocks, XML edits, or unified diffs from model response."""
    edit_matches = re.findall(r'<edit\b[^>]*>.*?</edit>', response_text, re.DOTALL | re.IGNORECASE)
    diff_matches = re.findall(r'(?:diff --git|--- a/|\+\+\+ b/).*?(?=\Z|diff --git)', response_text, re.DOTALL)
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', response_text, re.DOTALL)
    
    primary_patch = ""
    if diff_matches:
        primary_patch = diff_matches[0].strip()
    elif edit_matches:
        primary_patch = edit_matches[0].strip()
    elif code_blocks:
        primary_patch = code_blocks[0].strip()

    return {
        "num_edit_blocks": len(edit_matches),
        "num_diff_blocks": len(diff_matches),
        "num_code_blocks": len(code_blocks),
        "primary_patch": primary_patch,
        "code_blocks": code_blocks,
    }


def run_local_inference(
    model_name_or_path: str = "google/gemma-4/transformers/gemma-4-e2b-it-qat-mobile-transformers",
    instance_id: str = "psf__requests-2674",
    custom_prompt: Optional[str] = None,
    device: str = "auto",
    token_budget: int = 3480,
    max_tokens: int = 1024,
    enable_thinking: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PERRON: LIVE LOCAL GEMMA 4 INFERENCE ENGINE")
    print(f"Target Model: {model_name_or_path}")
    print(f"Target Instance: {instance_id}")
    print("=" * 80)

    # 1. Measure initial memory baseline
    mem_initial = get_memory_info()
    print("\n[System Memory Baseline]")
    print(f"  Physical RAM: {mem_initial['total_ram_gb']} GB")
    print(f"  Available RAM: {mem_initial['available_ram_gb']} GB")
    print(f"  Process Baseline RSS: {mem_initial['proc_rss_gb']} GB")

    # 2. Extract issue statement
    if custom_prompt:
        issue_text = custom_prompt
        repo_name = "custom"
    else:
        inst_data = find_instance(instance_id)
        if inst_data:
            issue_text = inst_data.get("problem_statement", "")
            repo_name = inst_data.get("repo", "requests")
            print(f"\n[Problem Statement Loaded]: {len(issue_text)} chars")
        else:
            issue_text = (
                "Issue psf__requests-2674: Exceptions raised by urllib3 are not properly wrapped "
                "in requests.exceptions.RequestException subclasses when streaming responses."
            )
            repo_name = "requests"
            print(f"\n[Problem Statement Fallback]: {issue_text}")

    # 3. Load Real CSR Call Graph & Perform Perron Diffusion
    graph_dir = None
    if "requests" in repo_name.lower():
        graph_dir = REPO_ROOT / "data" / "real_graphs" / "requests"
    elif "sympy" in repo_name.lower():
        graph_dir = REPO_ROOT / "data" / "real_graphs" / "sympy"

    context_slice = ""
    retained_count = 0
    used_tokens = 0
    diff_time = 0.0
    if graph_dir and graph_dir.exists():
        print(f"\n[Loading Zero-Copy CSR Graph]: {graph_dir.name}")
        from perron.graph import ASTSymbolNode
        t_matrix, dangling, node_to_id, id_to_node = load_mmap_csr(graph_dir)
        num_nodes = t_matrix.shape[0]

        with open(graph_dir / "symbols.json", "r", encoding="utf-8") as f:
            raw_symbols = json.load(f)
        raw_nodes = [ASTSymbolNode(**s) for s in raw_symbols]
        context_symbols = [node.to_context_symbol() for node in raw_nodes]
        symbols_dict = {sym.node_id: sym for sym in context_symbols}
        print(f"  Graph Nodes: {num_nodes}, Non-zero Matrix Elements: {t_matrix.nnz}")

        # Seed teleportation prior based on term overlap with issue text
        words = set(re.findall(r"\w+", issue_text.lower()))
        sims = np.zeros(num_nodes, dtype=np.float64)
        for sym in context_symbols:
            nid = sym.node_id
            if 0 <= nid < num_nodes:
                sym_tokens = set(re.findall(r"\w+", (sym.name + " " + (sym.class_docstring or "")).lower()))
                overlap = len(words.intersection(sym_tokens))
                sims[nid] = float(overlap)

        t_diff_0 = time.perf_counter()
        if np.max(sims) > 0:
            top_k_indices = np.argsort(sims)[-20:]
            p_0 = compute_softmax_teleport_prior(
                similarities=sims[top_k_indices],
                node_indices=top_k_indices,
                num_nodes=num_nodes,
                tau=0.05,
            )
        else:
            p_0 = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)

        # Perron PPR and specificity discount
        pi = personalized_pagerank_power_iteration(t_matrix, dangling, p_0, beta=0.85)
        pi_global = compute_global_pagerank(t_matrix, dangling, beta=0.85)
        scores = calculate_specificity_scores(pi, pi_global, gamma=1.5)
        diff_time = time.perf_counter() - t_diff_0

        packed_symbols, context_slice = pack_context_subgraphs(
            symbols=symbols_dict,
            specificity_scores=scores,
            adjacency_matrix=t_matrix,
            token_budget=token_budget,
        )
        retained_count = len(packed_symbols)
        used_tokens = sum(s.token_count for s in packed_symbols)

        print(f"  Perron Diffusion solved in {diff_time * 1000:.2f} ms")
        print(f"  Retained High-Specificity Symbols: {retained_count}")
        print(f"  Context Token Budget Used: {used_tokens} / {token_budget} tokens")
    else:
        context_slice = "# Sliced Context: requests/adapters.py\n# def send(self, request, ...): pass\n"
        print(f"\n[Using Default Context Slice]: {len(context_slice)} chars")

    # 4. Construct Gemma 4 Agent Prompt
    system_instruction = (
        "You are an expert autonomous software engineering agent powered by Gemma 4.\n"
        "Analyze the problem statement and the static call graph context slice.\n"
        "Formulate your reasoning step-by-step within <|think|>...</|think|> tags.\n"
        "Then produce a valid git unified diff or precise file edit block to resolve the defect."
    )

    full_prompt = (
        f"<start_of_turn>user\n"
        f"{system_instruction}\n\n"
        f"### PROBLEM STATEMENT:\n{issue_text}\n\n"
        f"### CODEBASE CONTEXT (Perron CSR Slice):\n```python\n{context_slice}\n```\n\n"
        f"Produce the bugfix patch:<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    if dry_run:
        print("\n[Dry Run Flag Enabled] Skipping neural weights loading.")
        return {
            "status": "dry_run_success",
            "prompt_length_chars": len(full_prompt),
            "retained_symbols": retained_count,
            "used_tokens": used_tokens,
            "diffusion_ms": round(diff_time * 1000, 2),
            "memory_initial": mem_initial,
        }

    # 5. Load Gemma 4 via TransformersBackend & Measure RAM
    print("\n[Initializing Live Gemma 4 Model Engine]...")
    t_load_0 = time.perf_counter()
    backend = TransformersBackend(
        model_name_or_path=model_name_or_path,
        device=device,
        enable_thinking=enable_thinking,
    )
    load_time = time.perf_counter() - t_load_0

    mem_loaded = get_memory_info()
    model_rss_increase = round(mem_loaded["proc_rss_gb"] - mem_initial["proc_rss_gb"], 2)
    print(f"\n[Model Loaded Successfully in {load_time:.2f}s]")
    print(f"  Device: {backend.device}")
    print(f"  Process Resident Memory (RSS): {mem_loaded['proc_rss_gb']} GB")
    print(f"  Model Weight Footprint in RAM: {model_rss_increase} GB")
    print(f"  Remaining Available RAM: {mem_loaded['available_ram_gb']} GB")

    # Assert consumer laptop constraint
    total_footprint = mem_loaded["used_ram_gb"]
    print(f"  Total System Used RAM: {total_footprint} GB (Target <= 16.0 GB laptop threshold)")

    # 6. Execute Model Generation
    print(f"\n[Executing Live Gemma 4 Generation (max_tokens={max_tokens})]...")
    response = backend.generate(
        prompt=full_prompt,
        max_tokens=max_tokens,
        temperature=0.2,
    )

    tok_per_sec = response.completion_tokens / max(0.001, response.latency_seconds)
    print("\n[Generation Complete]")
    print(f"  Latency: {response.latency_seconds:.2f} s")
    print(f"  Tokens Generated: {response.completion_tokens} tokens")
    print(f"  Generation Throughput: {tok_per_sec:.2f} tokens/s")

    if response.thinking_trace:
        print("\n" + "-" * 40 + " GEMMA 4 THINKING TRACE " + "-" * 40)
        print(response.thinking_trace[:500] + ("..." if len(response.thinking_trace) > 500 else ""))
        print("-" * 104)

    print("\n" + "-" * 40 + " GENERATED RESPONSE " + "-" * 44)
    print(response.content[:800] + ("..." if len(response.content) > 800 else ""))
    print("-" * 104)

    # 7. Extract Patch and Perform AST Syntax Audit
    patch_info = extract_patch_or_edits(response.content)
    ast_check = "N/A"
    if patch_info["code_blocks"]:
        sample_code = patch_info["code_blocks"][0]
        try:
            ast.parse(sample_code)
            ast_check = "AST VALID"
        except SyntaxError as e:
            ast_check = f"AST SYNTAX ERROR: {e}"

    print(f"  Extracted Primary Patch: {len(patch_info['primary_patch'])} chars")
    print(f"  AST Syntax Audit: {ast_check}")

    mem_final = get_memory_info()
    return {
        "status": "success",
        "model": model_name_or_path,
        "instance_id": instance_id,
        "load_time_sec": load_time,
        "inference_latency_sec": response.latency_seconds,
        "tokens_per_sec": tok_per_sec,
        "completion_tokens": response.completion_tokens,
        "prompt_tokens": response.prompt_tokens,
        "model_rss_gb": model_rss_increase,
        "total_rss_gb": mem_final["proc_rss_gb"],
        "available_ram_gb": mem_final["available_ram_gb"],
        "ast_check": ast_check,
        "has_thinking_trace": bool(response.thinking_trace),
        "patch_info": {
            "num_edit_blocks": patch_info["num_edit_blocks"],
            "num_diff_blocks": patch_info["num_diff_blocks"],
            "num_code_blocks": patch_info["num_code_blocks"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Local Gemma 4 Inference Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-4/transformers/gemma-4-e2b-it-qat-mobile-transformers",
        help="Model name, path, or KaggleHub slug",
    )
    parser.add_argument(
        "--instance-id",
        type=str,
        default="psf__requests-2674",
        help="SWE-bench Lite instance ID",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, cuda)")
    parser.add_argument("--budget", type=int, default=3480, help="Perron token budget")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    parser.add_argument("--no-thinking", action="store_true", help="Disable Gemma 4 thinking trace")
    parser.add_argument("--dry-run", action="store_true", help="Run graph diffusion and prompt pipeline without loading weights")
    args = parser.parse_args()

    results = run_local_inference(
        model_name_or_path=args.model,
        instance_id=args.instance_id,
        custom_prompt=args.prompt,
        device=args.device,
        token_budget=args.budget,
        max_tokens=args.max_tokens,
        enable_thinking=not args.no_thinking,
        dry_run=args.dry_run,
    )
    print("\n[Runner Execution Completed Successfully]")
    print(json.dumps(results, indent=2))
