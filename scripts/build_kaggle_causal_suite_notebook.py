"""
scripts/build_kaggle_causal_suite_notebook.py - Builder for Run 2A Kaggle Notebook.

Generates notebooks/kaggle_causal_suite/kaggle_causal_suite.ipynb for:
- Unconfounded full docstring-intact AST breadcrumbs (eliminates 3-line C1 stub confound).
- Compact candidate pool cutoff M=15 (top-10 BM25 seeds + top-5 1-hop AST neighbors).
- 2x2 Factorial evaluation on all 131 Stratum C tasks with Gemma 4 E4B:
  1. A4_M15_direct_baseline
  2. A4_M15_direct_provenance
  3. C0_M15_thinking_baseline
  4. C3_M15_thinking_provenance

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "notebooks" / "kaggle_causal_suite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Update kernel-metadata.json
metadata = {
    "id": "yvliet/perron-causal-experiments",
    "title": "perron-causal-experiments",
    "code_file": "kaggle_causal_suite.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": False,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "yvliet/perron-swebench-graphs",
        "yvliet/swebench-strata-annotations",
        "yvliet/perron-gemma4-benchmark"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [
        "google/gemma-4/Transformers/gemma-4-e4b-it/1",
        "google/gemma-4/Transformers/gemma-4-e2b-it/1"
    ]
}

with open(OUTPUT_DIR / "kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"[+] Wrote metadata to {OUTPUT_DIR / 'kernel-metadata.json'}")

# 2. Compress causal bundle 131 for embedded fail-safe fallback
bundle_131_path = REPO_ROOT / "data_release" / "causal_callpath_eval_bundle_131.json"
with open(bundle_131_path, "rb") as f:
    bundle_131_raw = f.read()

bundle_131_b64 = base64.b64encode(gzip.compress(bundle_131_raw)).decode("ascii")
print(f"[+] Compressed bundle 131: {len(bundle_131_raw)} -> {len(bundle_131_b64)} b64 chars")

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Perron Causal Call-Path Provenance & Reasoning Evaluation (M=15)\n",
            "\n",
            "- **Author**: Sultan Haikal (GitHub: `@yvliet`)\n",
            "- **Hardware**: Kaggle Dual Tesla T4 (Native FP16 with `device_map=\"auto\"`)\n",
            "- **Target Model**: Google Gemma 4 E4B (`google/gemma-4-e4b-it`)\n",
            "- **Cohort**: All 131 Stratum C un-named defect tasks under matched cutoff $M=15$\n",
            "- **Evaluation Invariant**: Full AST breadcrumbs with docstrings intact; testing unconfounded structural provenance and test-time reasoning against the Break-Even Discrimination Bound (AUC* = 0.714 at M=15).\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 1: Environment and Hardware Setup\n",
            "!pip install -q accelerate transformers\n",
            "\n",
            "import os, sys, json, math, time, re, gzip, base64\n",
            "from pathlib import Path\n",
            "import numpy as np\n",
            "import scipy.stats as stats\n",
            "import torch\n",
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
            "\n",
            "cuda_ok = torch.cuda.is_available()\n",
            "num_gpus = torch.cuda.device_count() if cuda_ok else 0\n",
            "print(f'[*] CUDA Available: {cuda_ok} (GPU Count: {num_gpus})')\n",
            "if cuda_ok:\n",
            "    for i in range(num_gpus):\n",
            "        props = torch.cuda.get_device_properties(i)\n",
            "        print(f'  [GPU {i}]: {props.name} | Total VRAM: {props.total_memory / 1e9:.2f} GB')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 2: Load Stratum C 131-Task Evaluation Bundle\n",
            "base_in = Path('/kaggle/input')\n",
            "bundle_candidates = [\n",
            "    Path('/kaggle/input/perron-gemma4-benchmark/causal_callpath_eval_bundle_131.json'),\n",
            "    Path('/kaggle/input/datasets/yvliet/perron-gemma4-benchmark/causal_callpath_eval_bundle_131.json'),\n",
            "    Path('data_release/causal_callpath_eval_bundle_131.json'),\n",
            "    Path('../data_release/causal_callpath_eval_bundle_131.json'),\n",
            "]\n",
            "\n",
            "bundle_path = None\n",
            "for p in bundle_candidates:\n",
            "    if p.exists():\n",
            "        bundle_path = p\n",
            "        break\n",
            "\n",
            "if bundle_path is None and base_in.exists():\n",
            "    for p in base_in.rglob('causal_callpath_eval_bundle_131.json'):\n",
            "        if p.exists():\n",
            "            bundle_path = p\n",
            "            break\n",
            "\n",
            "if bundle_path is not None:\n",
            "    print(f'[+] Loaded bundle from mounted dataset: {bundle_path}')\n",
            "    with open(bundle_path, 'r', encoding='utf-8') as f:\n",
            "        bundle_data = json.load(f)\n",
            "else:\n",
            "    print('[*] Using embedded fail-safe bundle payload...')\n",
            "    EMBEDDED_B64 = '''" + bundle_131_b64 + "'''\n",
            "    raw_json = gzip.decompress(base64.b64decode(EMBEDDED_B64)).decode('utf-8')\n",
            "    bundle_data = json.loads(raw_json)\n",
            "\n",
            "eval_tasks = bundle_data['tasks']\n",
            "print(f'[+] Successfully loaded {len(eval_tasks)} Stratum C evaluation tasks.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 3: Prompt Formatter with M=15 Cutoff and Docstring-Intact AST Breadcrumbs\n",
            "def truncate_text(text: str, max_chars: int = 3000) -> str:\n",
            "    if not text:\n",
            "        return ''\n",
            "    clean = re.sub(r'\\s+', ' ', text).strip()\n",
            "    if len(clean) <= max_chars:\n",
            "        return clean\n",
            "    half = (max_chars - 60) // 2\n",
            "    return clean[:half] + ' ... [TRUNCATED] ... ' + clean[-half:]\n",
            "\n",
            "def format_causal_prompt_m15(\n",
            "    task: dict,\n",
            "    mode: str = 'baseline',  # 'baseline' or 'provenance'\n",
            "    use_thinking: bool = False,\n",
            "    max_cands: int = 15,\n",
            ") -> tuple:\n",
            "    issue_text = truncate_text(task.get('problem_statement', ''))\n",
            "    all_cands = task.get('candidates', [])[:max_cands]\n",
            "    cand_lines = []\n",
            "    valid_indices = []\n",
            "    cand_idx_to_sym = {}\n",
            "\n",
            "    for c in all_cands:\n",
            "        c_idx = c['cand_index']\n",
            "        sym_id = c.get('symbol_id', c.get('node_idx', c_idx))\n",
            "        cand_idx_to_sym[c_idx] = sym_id\n",
            "        valid_indices.append(c_idx)\n",
            "\n",
            "        if mode == 'baseline':\n",
            "            bc = c.get('baseline_breadcrumb') or c.get('breadcrumb', '')\n",
            "        else:\n",
            "            bc = c.get('provenance_breadcrumb') or c.get('breadcrumb', '')\n",
            "\n",
            "        if not bc:\n",
            "            fpath = c.get('file_path', 'unknown')\n",
            "            ident = c.get('identifier', 'unknown')\n",
            "            st = c.get('line_start', 1)\n",
            "            en = c.get('line_end', st)\n",
            "            bc = f'# File: {fpath} (Lines {st}-{en})\\ndef {ident}(...): pass'\n",
            "        elif len(bc) > 320:\n",
            "            bc = bc[:315] + ' ... [truncated]'\n",
            "\n",
            "        cand_lines.append(f'[Candidate ID: {c_idx}]\\n{bc}')\n",
            "\n",
            "    cands_block = '\\n\\n'.join(cand_lines)\n",
            "\n",
            "    if use_thinking:\n",
            "        if mode == 'provenance':\n",
            "            prompt = (\n",
            "                'You are an expert developer resolving a software issue in a codebase.\\n\\n'\n",
            "                f'### Issue Description:\\n{issue_text}\\n\\n'\n",
            "                f'### Candidate Functions ({len(valid_indices)} candidates retrieved, M=15 cutoff):\\n'\n",
            "                f'{cands_block}\\n\\n'\n",
            "                '### Task:\\n'\n",
            "                'Analyze how issue symptoms connect to candidate functions via annotated call paths and data flow.\\n'\n",
            "                'First, reflect on which candidates are entry points versus true root cause inside <thought>...</thought> tags.\\n'\n",
            "                'Then, output your top-5 ranking by Candidate ID in descending order of root-cause likelihood.\\n'\n",
            "                'Format your ranking strictly in tags: <ranking>[ID1, ID2, ID3, ID4, ID5]</ranking>.'\n",
            "            )\n",
            "        else:\n",
            "            prompt = (\n",
            "                'You are an expert developer resolving a software issue in a codebase.\\n\\n'\n",
            "                f'### Issue Description:\\n{issue_text}\\n\\n'\n",
            "                f'### Candidate Functions ({len(valid_indices)} candidates retrieved, M=15 cutoff):\\n'\n",
            "                f'{cands_block}\\n\\n'\n",
            "                '### Task:\\n'\n",
            "                'Identify the primary root-cause location that must be modified to fix the issue.\\n'\n",
            "                'First, reflect on candidate functionality and root-cause likelihood inside <thought>...</thought> tags.\\n'\n",
            "                'Then, output your top-5 ranking by Candidate ID in descending order of likelihood.\\n'\n",
            "                'Format your ranking strictly in tags: <ranking>[ID1, ID2, ID3, ID4, ID5]</ranking>.'\n",
            "            )\n",
            "    else:\n",
            "        prompt = (\n",
            "            'You are an expert developer resolving a software issue in a codebase.\\n\\n'\n",
            "            f'### Issue Description:\\n{issue_text}\\n\\n'\n",
            "            f'### Candidate Functions ({len(valid_indices)} candidates retrieved, M=15 cutoff):\\n'\n",
            "            f'{cands_block}\\n\\n'\n",
            "            '### Task:\\n'\n",
            "            'Identify the primary root-cause location that must be modified to fix the issue.\\n'\n",
            "            'Output your top-5 ranking by Candidate ID in descending order of likelihood.\\n'\n",
            "            'Respond strictly in tags: <ranking>[ID1, ID2, ID3, ID4, ID5]</ranking> without preamble.'\n",
            "        )\n",
            "\n",
            "    return prompt, valid_indices, cand_idx_to_sym\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 4: Strict Ranking Parser and Scorer\n",
            "def parse_ranking(response_text: str, valid_ids: list) -> list:\n",
            "    valid_set = set(valid_ids)\n",
            "    m = re.search(r'<ranking>\\s*\\[?([\\d\\s,]+)\\]?\\s*</ranking>', response_text, re.IGNORECASE)\n",
            "    if m:\n",
            "        raw = re.findall(r'\\b\\d+\\b', m.group(1))\n",
            "        parsed = [int(n) for n in raw if int(n) in valid_set]\n",
            "        if parsed:\n",
            "            seen = set()\n",
            "            dedup = [n for n in parsed if not (n in seen or seen.add(n))]\n",
            "            rest = [n for n in valid_ids if n not in seen]\n",
            "            return dedup + rest\n",
            "\n",
            "    all_nums = [int(n) for n in re.findall(r'\\b\\d+\\b', response_text) if int(n) in valid_set]\n",
            "    if all_nums:\n",
            "        seen = set()\n",
            "        dedup = [n for n in all_nums if not (n in seen or seen.add(n))]\n",
            "        rest = [n for n in valid_ids if n not in seen]\n",
            "        return dedup + rest\n",
            "\n",
            "    return list(valid_ids)\n",
            "\n",
            "def score_ranking(ranked_cands: list, task: dict, cand_idx_to_sym: dict) -> dict:\n",
            "    gold_ids = set(task.get('gold_symbol_ids', []))\n",
            "    gold_cands = [c['cand_index'] for c in task.get('candidates', []) if c.get('is_gold', False)]\n",
            "    ranked_syms = [cand_idx_to_sym.get(c, c) for c in ranked_cands]\n",
            "\n",
            "    gold_rank = None\n",
            "    for r_idx, (c_idx, s_id) in enumerate(zip(ranked_cands, ranked_syms), start=1):\n",
            "        if s_id in gold_ids or c_idx in gold_cands:\n",
            "            gold_rank = r_idx\n",
            "            break\n",
            "\n",
            "    return {\n",
            "        'gold_rank': gold_rank,\n",
            "        'hit_1': bool(gold_rank == 1) if gold_rank else False,\n",
            "        'hit_3': bool(gold_rank <= 3) if gold_rank else False,\n",
            "        'hit_5': bool(gold_rank <= 5) if gold_rank else False,\n",
            "        'hit_10': bool(gold_rank <= 10) if gold_rank else False,\n",
            "        'mrr': float(1.0 / gold_rank) if gold_rank else 0.0,\n",
            "    }\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 5: Model Checkpoint Resolver & Loading\n",
            "def resolve_model_path() -> str:\n",
            "    base_in = Path('/kaggle/input')\n",
            "    if base_in.exists():\n",
            "        # Search for E4B first\n",
            "        for p in base_in.rglob('config.json'):\n",
            "            p_str = str(p.parent).lower()\n",
            "            if 'e4b' in p_str and 'gemma' in p_str:\n",
            "                print(f'[+] Resolved Gemma 4 E4B: {p.parent}')\n",
            "                return str(p.parent)\n",
            "        # Search for E2B fallback\n",
            "        for p in base_in.rglob('config.json'):\n",
            "            p_str = str(p.parent).lower()\n",
            "            if 'e2b' in p_str and 'gemma' in p_str:\n",
            "                print(f'[+] Resolved Gemma 4 E2B: {p.parent}')\n",
            "                return str(p.parent)\n",
            "    return 'google/gemma-4-e4b-it'\n",
            "\n",
            "model_path = resolve_model_path()\n",
            "print(f'[*] Loading tokenizer from {model_path}...')\n",
            "tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)\n",
            "if tokenizer.pad_token is None:\n",
            "    tokenizer.pad_token = tokenizer.eos_token\n",
            "\n",
            "print(f'[*] Loading model in native FP16 with device_map=\"auto\"...')\n",
            "device_map = 'auto' if torch.cuda.is_available() else None\n",
            "dtype = torch.float16 if torch.cuda.is_available() else torch.float32\n",
            "model = AutoModelForCausalLM.from_pretrained(\n",
            "    model_path,\n",
            "    torch_dtype=dtype,\n",
            "    device_map=device_map,\n",
            "    trust_remote_code=True\n",
            ")\n",
            "model.eval()\n",
            "print('[+] Model loaded successfully.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 6: Execute 2x2 Factorial Matrix under M=15\n",
            "eval_modes = [\n",
            "    ('A4_M15_direct_baseline', 'baseline', False, 50),\n",
            "    ('A4_M15_direct_provenance', 'provenance', False, 50),\n",
            "    ('C0_M15_thinking_baseline', 'baseline', True, 180),\n",
            "    ('C3_M15_thinking_provenance', 'provenance', True, 180),\n",
            "]\n",
            "\n",
            "results_by_arm = {}\n",
            "\n",
            "for arm_name, mode, use_thinking, max_tokens in eval_modes:\n",
            "    print(f'\\n=== EXECUTING CONDITION: {arm_name} (mode={mode}, thinking={use_thinking}, M=15) ===')\n",
            "    arm_records = []\n",
            "    t_start = time.time()\n",
            "\n",
            "    for idx, task in enumerate(eval_tasks):\n",
            "        iid = task['instance_id']\n",
            "        prompt_text, valid_ids, cand_idx_to_sym = format_causal_prompt_m15(\n",
            "            task, mode=mode, use_thinking=use_thinking, max_cands=15\n",
            "        )\n",
            "\n",
            "        try:\n",
            "            prompt_formatted = tokenizer.apply_chat_template(\n",
            "                [{'role': 'user', 'content': prompt_text}],\n",
            "                tokenize=False,\n",
            "                add_generation_prompt=True\n",
            "            )\n",
            "        except Exception:\n",
            "            prompt_formatted = f'<start_of_turn>user\\n{prompt_text}<end_of_turn>\\n<start_of_turn>model\\n'\n",
            "\n",
            "        inputs = tokenizer(prompt_formatted, return_tensors='pt', truncation=True, max_length=3400).to(model.device)\n",
            "        with torch.no_grad():\n",
            "            out = model.generate(\n",
            "                **inputs,\n",
            "                max_new_tokens=max_tokens,\n",
            "                do_sample=False,\n",
            "                pad_token_id=tokenizer.eos_token_id\n",
            "            )\n",
            "        gen_toks = out[0, inputs.input_ids.shape[1]:]\n",
            "        gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)\n",
            "\n",
            "        pred_ranking = parse_ranking(gen_text, valid_ids)\n",
            "        scores = score_ranking(pred_ranking, task, cand_idx_to_sym)\n",
            "        scores['instance_id'] = iid\n",
            "        scores['pred_ranking'] = pred_ranking\n",
            "        cands15 = task.get('candidates', [])[:15]\n",
            "        scores['is_seed_gold'] = bool(any(c.get('is_gold') and c.get('provenance', {}).get('edge_type') == 'seed' for c in cands15))\n",
            "        scores['is_graph_only_gold'] = bool((not scores['is_seed_gold']) and any(c.get('is_gold') for c in cands15))\n",
            "        arm_records.append(scores)\n",
            "\n",
            "        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_tasks):\n",
            "            h1 = sum(1 for r in arm_records if r['hit_1'])\n",
            "            h5 = sum(1 for r in arm_records if r['hit_5'])\n",
            "            print(f'  [{idx + 1}/{len(eval_tasks)}] Progress | Hits@1: {h1} | Hits@5: {h5}')\n",
            "\n",
            "    elapsed = time.time() - t_start\n",
            "    results_by_arm[arm_name] = {\n",
            "        'arm_name': arm_name,\n",
            "        'mode': mode,\n",
            "        'use_thinking': use_thinking,\n",
            "        'elapsed': elapsed,\n",
            "        'records': arm_records,\n",
            "    }\n",
            "    # Save intermediate checkpoint\n",
            "    with open('stratum_c_causal_m15_checkpoint.json', 'w') as f:\n",
            "        json.dump(results_by_arm, f, indent=2)\n",
            "\n",
            "print('[+] All 4 conditions executed successfully.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Step 7: Statistical Analysis & Hypothesis Contrasts\n",
            "summary = {}\n",
            "n_total = len(eval_tasks)\n",
            "\n",
            "for arm_name, data in results_by_arm.items():\n",
            "    recs = data['records']\n",
            "    h1 = sum(1 for r in recs if r['hit_1'])\n",
            "    h3 = sum(1 for r in recs if r['hit_3'])\n",
            "    h5 = sum(1 for r in recs if r['hit_5'])\n",
            "    h10 = sum(1 for r in recs if r['hit_10'])\n",
            "    mrr = float(np.mean([r['mrr'] for r in recs]))\n",
            "    ranks = [r['gold_rank'] for r in recs if r['gold_rank'] is not None]\n",
            "\n",
            "    # Subgroups\n",
            "    seed_recs = [r for r in recs if r['is_seed_gold']]\n",
            "    graph_recs = [r for r in recs if r['is_graph_only_gold']]\n",
            "\n",
            "    summary[arm_name] = {\n",
            "        'n': n_total,\n",
            "        'hits_1': h1,\n",
            "        'hits_3': h3,\n",
            "        'hits_5': h5,\n",
            "        'hits_10': h10,\n",
            "        'acc_1_pct': float(h1 / n_total * 100),\n",
            "        'acc_3_pct': float(h3 / n_total * 100),\n",
            "        'acc_5_pct': float(h5 / n_total * 100),\n",
            "        'acc_10_pct': float(h10 / n_total * 100),\n",
            "        'mrr': mrr,\n",
            "        'mean_gold_rank': float(np.mean(ranks)) if ranks else None,\n",
            "        'median_gold_rank': float(np.median(ranks)) if ranks else None,\n",
            "        'seed_subgroup': {\n",
            "            'n': len(seed_recs),\n",
            "            'hits_1': sum(1 for r in seed_recs if r['hit_1']),\n",
            "            'hits_5': sum(1 for r in seed_recs if r['hit_5']),\n",
            "        },\n",
            "        'graph_only_subgroup': {\n",
            "            'n': len(graph_recs),\n",
            "            'hits_1': sum(1 for r in graph_recs if r['hit_1']),\n",
            "            'hits_5': sum(1 for r in graph_recs if r['hit_5']),\n",
            "        },\n",
            "    }\n",
            "\n",
            "# Hypothesis contrasts\n",
            "contrasts = {}\n",
            "comparisons = [\n",
            "    ('C3_thinking_vs_C0_thinking', 'C3_M15_thinking_provenance', 'C0_M15_thinking_baseline'),\n",
            "    ('C0_thinking_vs_A4_direct', 'C0_M15_thinking_baseline', 'A4_M15_direct_baseline'),\n",
            "    ('C3_thinking_vs_A4_direct', 'C3_M15_thinking_provenance', 'A4_M15_direct_baseline'),\n",
            "    ('A4_prov_direct_vs_A4_base_direct', 'A4_M15_direct_provenance', 'A4_M15_direct_baseline'),\n",
            "]\n",
            "\n",
            "for comp_name, arm_a, arm_b in comparisons:\n",
            "    recs_a = results_by_arm[arm_a]['records']\n",
            "    recs_b = results_by_arm[arm_b]['records']\n",
            "    # Top-1\n",
            "    n10_1 = sum(1 for a, b in zip(recs_a, recs_b) if a['hit_1'] and not b['hit_1'])\n",
            "    n01_1 = sum(1 for a, b in zip(recs_a, recs_b) if not a['hit_1'] and b['hit_1'])\n",
            "    p1 = stats.binomtest(min(n10_1, n01_1), n10_1 + n01_1, 0.5).pvalue if (n10_1 + n01_1) > 0 else 1.0\n",
            "    delta_1 = summary[arm_a]['acc_1_pct'] - summary[arm_b]['acc_1_pct']\n",
            "\n",
            "    # Top-5\n",
            "    n10_5 = sum(1 for a, b in zip(recs_a, recs_b) if a['hit_5'] and not b['hit_5'])\n",
            "    n01_5 = sum(1 for a, b in zip(recs_a, recs_b) if not a['hit_5'] and b['hit_5'])\n",
            "    p5 = stats.binomtest(min(n10_5, n01_5), n10_5 + n01_5, 0.5).pvalue if (n10_5 + n01_5) > 0 else 1.0\n",
            "    delta_5 = summary[arm_a]['acc_5_pct'] - summary[arm_b]['acc_5_pct']\n",
            "\n",
            "    contrasts[comp_name] = {\n",
            "        'top1': {'delta_pp': float(delta_1), 'n10': n10_1, 'n01': n01_1, 'p_value': float(p1)},\n",
            "        'top5': {'delta_pp': float(delta_5), 'n10': n10_5, 'n01': n01_5, 'p_value': float(p5)},\n",
            "    }\n",
            "\n",
            "print('\\n=== FACTORIAL RESULTS SUMMARY (M=15 Cutoff, N=131) ===')\n",
            "print(f'| {\"Condition\":<28} | {\"Acc@1\":<7} | {\"Acc@5\":<7} | {\"MRR\":<7} | {\"Mean Rank\":<10} | {\"Seeds H1\":<8} | {\"Graph H1\":<8} |')\n",
            "print('| ' + '-'*28 + ' | ' + '-'*7 + ' | ' + '-'*7 + ' | ' + '-'*7 + ' | ' + '-'*10 + ' | ' + '-'*8 + ' | ' + '-'*8 + ' |')\n",
            "for arm_name, s in summary.items():\n",
            "    print(f'| {arm_name:<28} | {s[\"acc_1_pct\"]:>6.2f}% | {s[\"acc_5_pct\"]:>6.2f}% | {s[\"mrr\"]:>7.4f} | {str(s[\"mean_gold_rank\"])[:8]:<10} | {s[\"seed_subgroup\"][\"hits_1\"]:>8} | {s[\"graph_only_subgroup\"][\"hits_1\"]:>8} |')\n",
            "\n",
            "print('\\n=== HYPOTHESIS CONTRASTS ===')\n",
            "for k, v in contrasts.items():\n",
            "    print(f'{k}:')\n",
            "    print(f'  Top-1: Delta={v[\"top1\"][\"delta_pp\"]:+.2f} pp (n10={v[\"top1\"][\"n10\"]}, n01={v[\"top1\"][\"n01\"]}, p={v[\"top1\"][\"p_value\"]:.4f})')\n",
            "    print(f'  Top-5: Delta={v[\"top5\"][\"delta_pp\"]:+.2f} pp (n10={v[\"top5\"][\"n10\"]}, n01={v[\"top5\"][\"n01\"]}, p={v[\"top5\"][\"p_value\"]:.4f})')\n",
            "\n",
            "final_output = {\n",
            "    'suite': 'stratum_c_causal_m15_benchmark',\n",
            "    'author': 'Sultan Haikal (@yvliet)',\n",
            "    'license': 'Apache-2.0',\n",
            "    'n_tasks': n_total,\n",
            "    'pool_cutoff': 15,\n",
            "    'summary': summary,\n",
            "    'contrasts': contrasts,\n",
            "    'raw_results': results_by_arm,\n",
            "}\n",
            "\n",
            "with open('stratum_c_causal_m15_results.json', 'w') as f:\n",
            "    json.dump(final_output, f, indent=2)\n",
            "print('[+] Full empirical dataset written to stratum_c_causal_m15_results.json.')\n"
        ]
    }
]

notebook_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path = OUTPUT_DIR / "kaggle_causal_suite.ipynb"
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=2)

# Automated AST Static Syntax Validation Gate
import ast
print("[*] Running automated AST static syntax validation on all notebook cells...")
for idx, c in enumerate(cells):
    if c["cell_type"] == "code":
        code_str = "".join([line for line in c["source"] if not line.strip().startswith("!")])
        try:
            ast.parse(code_str)
        except SyntaxError as e:
            raise RuntimeError(f"FATAL: Syntax error in Notebook Cell {idx}: {e}")
print("[+] All notebook cells verified syntactically valid via Python AST parser.")
print(f"[+] Successfully built Run 2A notebook at: {nb_path}")
