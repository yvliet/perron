# SWE-bench Lexical Accessibility Strata Annotations & Evaluation Parquet

## Overview
This dataset provides per-instance lexical accessibility stratification across 801 real-world software engineering benchmarks (SWE-bench Lite and SWE-bench Verified) alongside 19,298 evaluation passes from the Perron study.
Authored by Sultan Haikal (@yvliet) as part of the Perron spectral diffusion framework for Google DeepMind's Gemma 4 Developer Agent challenge.

## Motivation & Problem Statement
Code retrieval evaluations in software engineering agents frequently conflate two distinct retrieval regimes:
1. **Lexically Direct Queries (Stratum A)**: Problem statements that explicitly mention the target function, class, or method name. In this regime, standard inverted-index retrievers (BM25) and dense embeddings excel.
2. **Structural & Behavioral Queries (Stratum C)**: Problem statements describing symptoms, test failures, or behavioral defects without naming the underlying code entity. In this regime, lexical retrieval funnels away from targets, demanding topological graph diffusion.

## Stratum Taxonomy
Each task is annotated into one of three mutually exclusive strata:
- **Stratum A (Named Targets)**: Gold modified symbol names appear verbatim in the problem description.
- **Stratum B (Trace Targets)**: File paths, module names, or runtime error traces appear in the problem description without explicit function names.
- **Stratum C (Un-named Targets)**: Neither function nor class names appear in the prompt; resolving the defect requires structural traversal from symptom entrypoints.

## Schema: `swebench_strata_annotations.jsonl`
- `instance_id` (str): SWE-bench task identifier (e.g. `astropy__astropy-12907`, `django__django-11019`).
- `benchmark` (str): Benchmark split origin (`lite` or `verified`).
- `repo` (str): Source repository slug (`astropy/astropy`, `django/django`, `sympy/sympy`, etc.).
- `stratum` (str): Stratum classification (`A`, `B`, or `C`).
- `stratum_label` (str): Full stratum descriptor (`A_named`, `B_trace`, `C_unnamed`).
- `gold_targets` (List[str]): Ground-truth modified symbol names extracted from gold resolution patch.
- `gold_files` (List[str]): Ground-truth modified file paths.
- `problem_statement_snippet` (str): First 200 characters of the raw issue description.

## Schema: `scores.parquet` (Evaluation Manifest)
`scores.parquet` packages all 19,298 evaluation passes across benchmark strata:
- `instance_id` (large_string): SWE-bench task identifier (e.g. `astropy__astropy-12907`, `django__django-11019`).
- `candidate_id` (int64): 0-indexed candidate position within candidate ranking pool.
- `prompt_variant` (large_string): Model and retrieval condition (e.g. `gemma-4-e2b-it:A4_graph_1hop`, `A2_depth_bm25`, `C0_direct`).
- `s` (double): Pointwise scoring logit / score assigned to candidate symbol.
- `format_mass` (double): Normalized prompt format probability mass.
- `n_prompt_tokens` (int64): Candidate context prompt token length.

### Parquet Verification & Compatibility
`scores.parquet` is 100% compliant with Apache Parquet format (PyArrow 14+, Fastparquet, Polars, DuckDB):
```python
import polars as pl

df = pl.read_parquet("dist_dataset/scores.parquet")
print(f"Loaded {len(df):,} evaluation records across {df['instance_id'].n_unique()} benchmark tasks.")
```
Programmatic verification is enforced via Gate 20 of `scripts/verify_paper.py`, confirming schema types and exact 19,298 row integrity.

## Dataset Statistics
- **Total Annotated Instances**: 801
- **SWE-bench Lite Coverage**: 300 instances
- **SWE-bench Verified Coverage**: 500 instances
- **Total Empirical Evaluation Records**: 19,298 rows (`scores.parquet`)
- **Stratum Distribution**:
  - Stratum A: ~38%
  - Stratum B: ~18%
  - Stratum C: ~44%

## Usage
```python
import json

with open("swebench_strata_annotations.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"Loaded {len(records)} stratified instances.")
```

## Citation & Attribution
- Author: Sultan Haikal (GitHub: `@yvliet`)
- Repository: https://github.com/yvliet/perron
- License: Creative Commons Zero v1.0 Universal (CC0-1.0)
