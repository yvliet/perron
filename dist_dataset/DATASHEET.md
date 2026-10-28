# Datasheet for Datasets: Perron SWE-bench Strata & Repository Multigraphs

*Following the framework proposed by Gebru et al. (2021).*

- **Author**: Sultan Haikal (GitHub: `@yvliet`)
- **License**: Apache-2.0
- **Version**: 1.0.0 (October 2026)
- **Repository**: `https://github.com/yvliet/perron`

---

## 1. Motivation

### For what purpose was the dataset created?
The Perron SWE-bench Strata and Repository Multigraphs dataset was created to resolve critical evaluation bottlenecks in repository-level code intelligence and defect localization. Existing software engineering benchmarks, such as SWE-bench, aggregate diverse bug reports without distinguishing between tasks where the defective entity is explicitly named in the prompt versus tasks requiring deep structural traversal. This resource provides:
1. Grounded lexical stratification annotations across 707 unique SWE-bench tasks (800 evaluation records across SWE-bench Lite and SWE-bench Verified).
2. Twelve full-scale repository Abstract Syntax Tree (AST) multigraphs serialized as compressed sparse matrices (CSR) with typed relation layers.
3. Complete scored evaluation telemetry from Gemma 4 models (E2B and E4B) capturing token-level candidate probabilities, rank distributions, and counterfactual intervention outcomes.

### Who created the dataset?
The dataset was authored and curated by Sultan Haikal (GitHub: `@yvliet`).

### Who funded the creation of the dataset?
Independent research conducted for the Google DeepMind Gemma 4 Developer Agent competition (Track 2: Research Monograph & Resource Track). Compute resources were provided via Kaggle dual Nvidia T4 GPU hardware environments.

---

## 2. Composition

### What do the instances that comprise the dataset represent?
The dataset comprises three interconnected components:

#### Component A: SWE-bench Strata Annotations (`data/swebench_strata_annotations.jsonl`)
Contains 800 evaluation records across 707 unique task instances (300 in SWE-bench Lite, 500 in SWE-bench Verified, with 93 overlapping instances evaluated on both benchmarks). Each task is classified into one of three mutually exclusive lexical strata:
- **Stratum A (A_named)**: The issue statement explicitly contains the exact name of the target defective function, method, or class within strict word/identifier boundaries.
- **Stratum B (B_trace)**: The issue statement does not name the target function, but includes an error traceback frame referencing a gold modified repository file.
- **Stratum C (C_unnamed)**: The issue statement describes high-level symptoms, conceptual requirements, or silent logic failures without naming target functions or including file traceback frames.

The 500 SWE-bench Verified instances are partitioned into a strict, mutually exclusive and collectively exhaustive (MECE) CONSORT waterfall:
- 344 Verified Callable Evaluated instances.
- 93 Exploratory Split Overlap instances (also present in SWE-bench Lite).
- 56 File-Absent Preimage Drift instances (source files altered or moved between commits).
- 7 Module-Level Non-Callable instances (bug fixed by top-level variable or statement mutations).
- Total: $344 + 93 + 56 + 7 = 500$.

#### Component B: 12 Repository AST Multigraphs (`artifacts/swebench_graphs/`)
Serialized as compressed SciPy CSR matrices (`*_graph.npz`, `*_layers.npz`), symbol lookup tables (`*_symbols.json`), and graph topology metadata (`*_meta.json`).
- Total Nodes: 109,240 code entities (modules, classes, methods, functions).
- Total Edges: 341,892 directed relations partitioned across four typed layers:
  1. `call`: Directed invocation from caller to callee.
  2. `caller`: Inverted invocation edge (callee to caller).
  3. `import`: Dependency binding from importing module to exported entity.
  4. `inherit`: Object-oriented class inheritance and interface implementation.
- Coverage: Twelve prominent Python open-source repositories: `astropy/astropy`, `django/django`, `matplotlib/matplotlib`, `mwaskom/seaborn`, `pallets/flask`, `psf/requests`, `pydata/xarray`, `pylint-dev/pylint`, `pytest-dev/pytest`, `scikit-learn/scikit-learn`, `sphinx-doc/sphinx`, and `sympy/sympy`.

#### Component C: Scored Evaluation Telemetry (`results/scores.parquet`, `results/gemma4_e4b_scores.parquet`)
- `scores.parquet`: 19,298 candidate-level evaluation records from Gemma 4 E2B pointwise scoring, containing columns `['instance_id', 'candidate_id', 'prompt_variant', 's', 'format_mass', 'n_prompt_tokens']`. Relational metadata (gold target indicators and lexical strata) are joined via `data/swebench_verified_gold_labels.json` and `dist_dataset/swebench_strata_annotations.jsonl`.
- `gemma4_e4b_scores.parquet`: 8,113 candidate-level evaluation records from Gemma 4 E4B pointwise scoring across the multi-pool union on SWE-bench Verified tasks.

### Are there recommended data splits?
Yes. The pre-specified confirmatory split consists of the $N=344$ callable SWE-bench Verified tasks. Exploratory analyses should be conducted on the $N=93$ overlapping tasks or the $N=300$ SWE-bench Lite tasks to preserve confirmatory validity.

---

## 3. Collection Process

### How was the data acquired?
1. **Repository Graph Extraction**: Parsed directly from clean repository checkouts at corresponding base commits using Python's native `ast` module and Tree-sitter parsers. Graph extraction maps all definitions, calls, imports, and class declarations into a unified multigraph.
2. **Strata Classification**: Built via deterministic regex matching against raw GitHub problem statements (`perron/analysis/strata.py`).
3. **Model Telemetry**: Collected during inference runs on Kaggle GPU environments executing Google Gemma 4 models in FP16 precision.

### Over what timeframe was the data collected?
Data extraction and evaluation runs took place during October 2026.

---

## 4. Preprocessing, Cleaning & Labeling

### What preprocessing or cleaning was done?
- **Normalization**: File paths were normalized to POSIX standards, stripping Windows drive letters and local workstation prefixes.
- **Unreachable Distance Encoding**: In compliance with RFC 8259, infinite graph path lengths and missing hops were serialized as JSON `null` rather than non-standard `Infinity` or `NaN` literals.
- **Classifier-vs-Human Validation Audit (Cohen's Kappa)**: To verify that the regex stratification heuristic represents genuine linguistic ground truth rather than algorithmic idiosyncrasy, an independent human audit was conducted on a stratified random sample of $N=100$ tasks (45 from Stratum A, 15 from Stratum B, 40 from Stratum C; seed=42).
  - Observed Concordance: 95.0% ($95/100$).
  - Chance Agreement: 38.75%.
  - Cohen's $\kappa$: $0.9184$ (95% CI $[0.849, 0.988]$).
  - Qualitative Analysis: Disagreements were confined to 5 edge cases involving incidental helper mentions or non-standard attribute syntax.

---

## 5. Uses

### Has the dataset been used for any tasks already?
The dataset has been used to conduct the empirical audit reported in the research monograph *"When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4"*.

### What other tasks could the dataset be used for?
1. Benchmarking graph neural networks (GNNs) on software dependency graphs.
2. Evaluating test-time reasoning and retrieval-augmented generation (RAG) on code.
3. Analyzing lexical bias and prompt framing effects in automated program repair.

---

## 6. Distribution

### How will the dataset be distributed?
- **Code & Graph Artifacts**: Hosted on GitHub at `https://github.com/yvliet/perron` under the Apache-2.0 license.
- **Zenodo / Kaggle Datasets**: Graph archives and evaluation parquet files are packaged under DOI-indexed repositories.
- **Python Wheel**: `perron-core` distributed via PyPI for turnkey graph construction and evaluation reproduction.

---

## 7. Maintenance

### Who will maintain the dataset?
The dataset is maintained by Sultan Haikal (GitHub: `@yvliet`, email: `recrified@gmail.com`). Errata, graph additions, and community pull requests are reviewed via GitHub issues.
