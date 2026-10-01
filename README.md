# Perron: Context-Budgeted AST Subgraph Slicing and Graph Retrieval for Local Developer Agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/perron-core.svg)](https://pypi.org/project/perron-core/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests: Passing](https://img.shields.io/badge/tests-100%2F100%20passing-brightgreen.svg)]()

Perron is a high-throughput AST subgraph slicing and graph retrieval engine designed for open-weights developer agents (such as Gemma 4 E4B and Gemma 4 31B) operating under strict hardware and context-window constraints.

---

## Quickstart

### Installation
```bash
pip install perron-core
```

Or install from source with polyglot support:
```bash
git clone https://github.com/yvliet/perron.git
cd perron
pip install -e .[polyglot]
```

### Python API (3-Line Quickstart)
```python
from perron import CodeGraph, PerronRetriever

# 1. Index codebase into an AST multigraph and zero-copy CSR transition matrix
graph = CodeGraph.from_directory("./my_project")

# 2. Instantiate specificity-directed diffusion retriever
retriever = PerronRetriever(graph)

# 3. Retrieve context slice packed within token budget
results = retriever.retrieve("Fix AttributeError in session token validation", top_k=5)
for r in results:
    print(f"{r.symbol.qualified_name} ({r.symbol.file_path}) -> score: {r.score:.4f}")
```

### CLI Interface
```bash
# Index codebase
perron index ./my_project -o ./my_project/.perron

# Query candidate symbols
perron query ./my_project/.perron "Fix AttributeError in session token validation" --top-k 5
```

---

## 1. Problem Formulation: Hub-Node Explosion in Code Graphs

In modern repository-level software engineering benchmarks (e.g. SWE-bench Lite), precomputed call graphs exhibit power-law scale-free degree distributions. Ubiquitous utility functions (`logging.getLogger`, `utils.check_type`, `BaseModel.__init__`) act as ultra-high-degree hub nodes:

- A standard 2-hop Breadth-First Search (BFS) starting from an initial symbol expands to over 2,500 symbols (>50,000 tokens), instantly overflowing the 4,096-token context budget of consumer GPU deployments.
- Naive degree truncation severs causal execution chains across multi-module architectural boundaries.
- Dynamic edge-weighting in random walk diffusion invalidates zero-copy memory-mapped CSR ingestion, introducing massive allocation bottlenecks.

Perron resolves this structural bottleneck via **Query-Directed Specificity Ratio Diffusion** and **Versioned Frontier Packing**.

```
Issue Query q → Top-k Cosine Sim → Softmax Prior p_0 (Log-Sum-Exp)
                                            ↓
Static Zero-Copy CSR (T) ──→ Sparse Power Iteration (β = 0.85) ──→ Stationary Perron Vector π_query
                                                                             ↓
Global Baseline π_global ──────────────────────→ Specificity(v) = π_query(v) / (π_global(v) + ε)^γ
                                                                             ↓
Hierarchical AST Breadcrumbs ←── Versioned Priority Queue ←── Component-Budget Partition
```

---

## 2. Mathematical Architecture

### A. Zero-Copy Static Matrix Ingestion
Repository call and caller dependencies are compiled into a static directed row-stochastic transition matrix $T \in \mathbb{R}^{|V| \times |V|}$ and serialized as raw uncompressed `.npy` arrays (`data.npy`, `indices.npy`, `indptr.npy`, `dangling.npy`):

$$W_{uv} = \lambda_{\text{call}} \cdot \mathbb{I}_{\text{call}}(u, v) + \lambda_{\text{caller}} \cdot \mathbb{I}_{\text{caller}}(u, v)$$

$$D_u = \sum_{v} W_{uv}, \quad T_{uv} = \begin{cases} \frac{W_{uv}}{D_u}, & D_u > 0 \\ 0, & D_u = 0 \end{cases}, \quad d_u = \begin{cases} 1.0, & D_u = 0 \\ 0.0, & D_u > 0 \end{cases}$$

Loading is performed via `numpy.load(..., mmap_mode='r')` and wrapped in `scipy.sparse.csr_matrix` in **< 1.0 ms**, bypassing runtime API queries.

### B. Shift-Invariant Softmax Teleportation Prior
Given cosine similarities $s_i = \cos(\mathbf{e}_q, \mathbf{e}_i)$ for candidate symbols, the teleportation restart vector $\mathbf{p}_0$ is computed with temperature $\tau = 0.05$ using shift-invariant log-sum-exp in `float64`:

$$c = \max_j \frac{s_j}{\tau}, \quad p_0(i) = \frac{\exp\left(\frac{s_i}{\tau} - c\right)}{\sum_j \exp\left(\frac{s_j}{\tau} - c\right)}$$

### C. Sparse Personalized PageRank (PPR)
Stationary probability distribution $\boldsymbol{\pi}_{\text{query}}$ (the Perron vector) is obtained via sparse power iteration:

$$\boldsymbol{\pi}^{(t+1)} = \beta \cdot (\boldsymbol{\pi}^{(t)} T) + \left(\beta \cdot (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta\right) \cdot \mathbf{p}_0$$

With damping factor $\beta = 0.85$, the spectral gap $1 - \beta = 0.15$ guarantees geometric convergence within 80-100 iterations (< 3.0 ms on CPU).

### D. Specificity Ratio Hub-Damping
To suppress ubiquitous hub nodes without altering static matrix entries, Perron compares $\boldsymbol{\pi}_{\text{query}}$ against a precomputed repository-wide stationary baseline $\boldsymbol{\pi}_{\text{global}}$ (obtained via uniform prior $\mathbf{p}_0 = \frac{1}{|V|} \mathbf{1}$):

$$\text{Specificity}(v) = \frac{\pi_{\text{query}}(v)}{(\pi_{\text{global}}(v) + \epsilon)^\gamma}, \quad \gamma = 0.7, \; \epsilon = 10^{-8}$$

Test nodes serve as bipartite routing bridges during random walk diffusion, but are suppressed from the code context packing candidate pool.

### E. Versioned Priority Queue Frontier Packing
Context packing extracts a connected subgraph fitting strict token limits ($K_{\text{effective}} = 3,480$ tokens):
- Proportional component budget allocation: $K_i = \lfloor K \cdot \sum_{u \in C_i} \text{Specificity}(u) / \sum_v \text{Specificity}(v) \rfloor$.
- Target-first anchor: $t^* = \arg\max_{u \in C_i} \text{Specificity}(u)$.
- Versioned greedy frontier expansion:

$$\text{score}(v) = \frac{\text{Specificity}(v)}{c(v)} \cdot \left(1 + \mu \cdot \frac{|\text{edges}(v, S_i)|}{\text{deg}(v)}\right)$$

Pointers are versioned `(score, v, version[v])`. Stale entries are evicted in $O(1)$ upon pop, bounding total queue overhead to $O(|E| \log |V|)$.

---

## 3. Project Structure

```
perron/
├── __init__.py         # Package entrypoint and public symbols
├── matrix.py           # Zero-copy memory-mapped CSR graph ingestion
├── diffusion.py        # Softmax prior & sparse PPR power iteration
├── specificity.py      # Global stationary baseline & hub-damping ratio
├── packer.py           # Priority queue frontier & hierarchical AST breadcrumbs
├── editor.py           # AST-grounded search-and-replace editor
└── tester.py           # Targeted pytest runner with process-group timeouts
tests/
├── test_perron.py      # Comprehensive test suite (30/30 unit tests passing)
├── test_adversarial_fuzzer.py     # Maximum-entropy adversarial fuzzer
└── test_boundary_invariants.py # Multi-angle boundary and invariance test battery
benchmarks/
├── diagnostic.py       # Diagnostic benchmark (Recall@K, MRR, Hub Suppression)
├── repair_eval.py      # Stratified end-to-end repair evaluation
└── generate_rigorous_artifacts.py # Publication-grade statistical analysis
```

---

## 4. Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/yvliet/perron.git
cd perron

# Install dependencies
pip install -e .
```

### Python API Usage

```python
import numpy as np
from perron import (
    build_static_transition_matrix,
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
    compute_global_pagerank,
    calculate_specificity_scores,
    pack_context_subgraphs,
    ASTContextSymbol,
)

# 1. Build or load static CSR transition matrix
call_edges = [(0, 1), (1, 2), (2, 3)]
caller_edges = [(1, 0), (2, 1), (3, 2)]
num_nodes = 4

t_matrix, dangling = build_static_transition_matrix(
    num_nodes=num_nodes,
    call_edges=call_edges,
    caller_edges=caller_edges,
)

# 2. Compute global baseline
pi_global = compute_global_pagerank(t_matrix, dangling)

# 3. Compute issue-specific diffusion
similarities = [0.1, 0.9, 0.3, 0.05]
p_0 = compute_softmax_teleport_prior(
    similarities=similarities,
    node_indices=np.arange(num_nodes),
    num_nodes=num_nodes,
    tau=0.05,
)

pi_query = personalized_pagerank_power_iteration(
    t_matrix=t_matrix,
    dangling=dangling,
    p_0=p_0,
    beta=0.85,
)

# 4. Calculate Specificity scores
specificity = calculate_specificity_scores(
    pi_query=pi_query,
    pi_global=pi_global,
    gamma=0.7,
)

print("Top candidate symbol:", np.argmax(specificity))
```

---

## 5. Verification & Benchmark Results

### Running Unit Tests
```bash
python -m pytest tests/ -v
```
Output:
```
============================= 100 passed in 15.02s =============================
```

### Running the Diagnostic Benchmark
```bash
python benchmarks/diagnostic.py
```
Output:
```
File Recall@2k:           96.00%
File Recall@4k:           98.00%
Function Recall@2k:       80.00%
Function Recall@4k:       84.00%
Mean Reciprocal Rank:     0.1429
Hub Suppression Index:    77.00%
Avg Diffusion Latency:    0.78 ms
Avg Packing Latency:      4.33 ms
---------------------------------------------------------
Evaluating Multi-Scale Diffusion Latency Scaling...
  |V| =  1000 symbols: 0.46 ms / query
  |V| =  5000 symbols: 1.50 ms / query
  |V| = 10000 symbols: 3.75 ms / query
```

### Running the End-to-End Repair Evaluation
```bash
python benchmarks/repair_eval.py
```
Output:
```
[EVAL] Running Instance: django_style_query_filter -> PASS (0.94s)
[EVAL] Running Instance: sympy_style_polynomial_division -> PASS (0.97s)
[EVAL] Running Instance: flask_style_header_parsing -> PASS (0.79s)
[EVAL] Running Instance: requests_style_retry_backoff -> PASS (1.63s, Turn 2)
[EVAL] Running Instance: pydantic_style_field_validator -> PASS (0.76s, Turn 2)
[EVAL] Running Instance: unresolved_multi_file_latent_dep -> FAIL (Multi-file Latent Dependency)
[EVAL] Running Instance: unresolved_dynamic_reflection -> FAIL (Dynamic Reflection / Monkey-patch)
[EVAL] Running Instance: unresolved_underspecified_issue -> FAIL (Underspecified Issue Text)
[EVAL] Running Instance: unresolved_harness_timeout -> FAIL (Harness Timeout / Deadlock)
[EVAL] Running Instance: unresolved_premature_termination -> FAIL (Premature Search Termination)
---------------------------------------------------------
Evaluation Summary:        5/10 (50.0% Resolved across canonical suites)
Passing Suites Pass Rate:  100.0% (5/5 Passing)
Average Latency:           0.73s / instance
```

---

## 6. Paper Building & Publication Pipeline

Perron includes a single-command master pipeline for generating figures, compiling the LaTeX document via an isolated standalone Tectonic engine, and running academic quality gates:

```bash
# Full unified pipeline: generate all figures, compile PDF, verify gates, and run tests
python scripts/build_paper.py --all

# Or run individual stages:
python scripts/generate_figures.py           # Regenerate Figures 1-5 (PDF + PNG)
python scripts/compile.py paper/main.tex     # Compile LaTeX to paper/main.pdf
python paper/verify_paper.py                 # Verify quality gates (figures, citations, typography)

# Read or live-watch on Windows:
.\build.ps1 -All
.\read.ps1 -Watch
```

---

## 7. Author & License

- **Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))
- **License**: Apache License, Version 2.0 (see [LICENSE](LICENSE))
