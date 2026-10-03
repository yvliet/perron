# Dataset Card: `swebench-lite-codegraphs`

> **Author**: Sultan Haikal (GitHub: `@yvliet`)  
> **License**: Apache 2.0 (Graph Topology & Metadata)  
> **Release Target**: Open-Source Research Benchmark for Local Software Engineering Agents  
> **Compatible Formats**: Compressed Sparse Row (CSR) `mmap` arrays, JSON multigraphs, and NetworkX / SciPy matrices.

---

## 1. Dataset Summary

`swebench-lite-codegraphs` provides the first standardized, task-grounded multigraph benchmark for software engineering retrieval across all 300 instances of the **SWE-bench Lite** benchmark.

Traditional SWE benchmarks distribute issues and test harnesses but omit the structural topological call graphs of the target software. Consequently, prior graph retrieval methods relied on unstandardized, ad-hoc AST extractors that suffered from cross-directory naming collisions and heuristic failure modes.

`swebench-lite-codegraphs` solves this by delivering:
1. **Verifiable AST Multigraphs**: Deterministically extracted AST symbols and 4-relation directed multiplex edges (`calls`, `callers`, `inherits`, `imports`) for base-commit syntax trees.
2. **Stationary Spectral Priors**: Precomputed stationary Global PageRank distributions ($\pi_{\text{global}}$) under Brauer spectral teleportation ($\beta = 0.85$), bounding spectral radius $|\lambda_2| \le 0.85$.
3. **Task-Grounded Target Labels**: Exact unified diff pre-image line range interval mappings ($I_{\text{hunk}} = [\text{start}, \text{start} + \text{count} - 1]$) identifying ground-truth function and container targets across all 300 instances.
4. **License-Decoupled Distribution**: Mathematical graph structures and metadata are distributed under Apache 2.0, with a reproducible hydration engine (`scripts/reconstruct_dataset.py`) to hydrate raw source code text directly from git checkouts, respecting upstream licenses.

---

## 2. Upstream Repository Licenses & Legal Compliance

The 300 instances of SWE-bench Lite originate from 12 prominent open-source Python software repositories with varied licensing terms:

| Repository Slug | Task Count | Upstream License | Distribution Strategy |
| :--- | :---: | :--- | :--- |
| `django/django` | 114 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `sympy/sympy` | 44 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `pytest-dev/pytest` | 26 | MIT | Topology distributed; text hydrated via git |
| `matplotlib/matplotlib` | 24 | PSF / Permissive | Topology distributed; text hydrated via git |
| `sphinx-doc/sphinx` | 18 | BSD-2-Clause | Topology distributed; text hydrated via git |
| `scikit-learn/scikit-learn` | 16 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `pylint-dev/pylint` | 15 | GPL-2.0-or-later | Topology distributed; text hydrated via git |
| `astropy/astropy` | 13 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `pallets/flask` | 10 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `mwaskom/seaborn` | 9 | BSD-3-Clause | Topology distributed; text hydrated via git |
| `psf/requests` | 6 | Apache-2.0 | Topology distributed; text hydrated via git |
| `pydata/xarray` | 5 | Apache-2.0 | Topology distributed; text hydrated via git |

### License Compliance Architecture
To strictly adhere to open-source governance and avoid redistribution complications with copyleft licenses (e.g. `pylint` under GPL-2.0):
- **Graph Metadata**: Symbol identifiers, relative file paths, line numbers, edge lists, in/out degrees, $\pi_{\text{global}}$ scores, and content SHA-256 digests are distributed under **Apache 2.0**.
- **Source Code Text**: Verbatim copyrightable code text is not bundled in the graph topology release. Users execute `scripts/reconstruct_dataset.py` to hydrate text locally from upstream git repositories at the exact base commit.

---

## 3. Dataset Schema & Structure

### Node Specification
Each AST symbol record contains:
```json
{
  "node_id": 1420,
  "file_path": "requests/sessions.py",
  "identifier": "Session.request",
  "symbol_type": "method",
  "start_line": 460,
  "end_line": 520,
  "parent_class": "Session",
  "docstring": "Constructs a :class:`Request <Request>`, prepares it and sends it.",
  "content_sha256": "8a3f82b7...",
  "in_degree": 42,
  "out_degree": 12,
  "pi_global": 0.0031842
}
```

### Multiplex Directed Edge Specification
Edges are encoded as a row-stochastic transition matrix in Compressed Sparse Row (CSR) format:
- `calls` (weight: 0.50): Direct caller-to-callee function/method invocations.
- `callers` (weight: 0.25): Bidirectional callee-to-caller reachability.
- `inherits` (weight: 0.15): Class inheritance and base-class hierarchies.
- `imports` (weight: 0.10): Module-level symbol import bindings.

---

## 4. Benchmark Splitting & Quarantine Protocol

To guarantee zero data contamination and support blinded evaluations, instances are partitioned via `data/swebench_lite_split.json`:
- **`dev_pilot` ($N = 50$)**: Historical exploratory subset (6 Requests + 44 SymPy). Quarantined strictly to algorithm discovery.
- **`dev_val` ($N = 100$)**: Uncontaminated tuning split used for baseline hyperparameter calibration.
- **`heldout` ($N = 150$)**: Strictly blinded evaluation split evaluated once with frozen parameters.

---

## 5. Usage & Hydration

### Step 1: Install Perron
```bash
pip install perron-core
```

### Step 2: Hydrate Source Text from Git
```bash
python scripts/reconstruct_dataset.py \
    --graph-meta data/swebench_lite_codegraphs_meta.json \
    --repo-dir ./repos/requests \
    --output ./hydrated/requests_graph.json
```

### Step 3: Run Model Context Protocol (MCP) Server
```bash
perron-mcp --repo ./repos/requests
```
