# Perron SWE-bench Lite Code Graph Resource

An open-source dataset of zero-copy Compressed Sparse Row (CSR) AST multigraphs and ground-truth localization targets derived from 300 task instances in SWE-bench Lite.

---

## 1. Resource Overview

Modern developer agents frequently saturate their context windows when navigating large codebases due to scale-free call graph hub explosion. The Perron Code Graph dataset provides canonical, pre-indexed topological representations of repositories, allowing researchers and agents to execute sub-millisecond spectral diffusion, personalized PageRank (PPR), and structural localization without runtime AST parsing overhead.

- **Instances**: 300 SWE-bench Lite tasks across 12 canonical Python repositories.
- **Topological Representation**: Directed multigraphs capturing `call`, `caller`, `inheritance`, and `import` relations.
- **Data Format**: Zero-copy memory-mapped CSR transition matrices (`.npy`), symbol tables (`nodes.parquet`), and manifest metadata (`manifest.json`).
- **Ground Truth**: Ground-truth patch-to-symbol mappings (`gold.jsonl`) with exact in-degree percentiles and hub annotations.
- **License**: Apache-2.0 for all derived multigraph structures and metadata.

---

## 2. Directory Layout

```
data_release/
├── graphs/
│   ├── <instance_id>/
│   │   ├── csr_data.npy         # Float32 row-stochastic non-zero edge weights
│   │   ├── csr_indices.npy      # Int32 target column indices
│   │   ├── csr_indptr.npy       # Int32 row pointers (length N + 1)
│   │   ├── csr_dangling.npy     # Float32 indicator vector for zero-outdegree sink nodes
│   │   ├── nodes.parquet        # Structured AST symbol table
│   │   └── manifest.json        # Instance metadata and SHA256 integrity hashes
│   └── ...
├── gold.jsonl                   # Ground truth localization targets (300 tasks)
├── croissant.json               # MLCommons Croissant 1.0 metadata specification
├── DATASHEET.md                 # Gebru et al. Dataset Datasheet
├── VALIDATION.md                # Topological and integrity validation report
└── README.md                    # Resource documentation (this file)
```

---

## 3. Gold Target Extraction Methodology

Ground-truth targets in `gold.jsonl` are extracted through deterministic AST pre-image line interval intersections:
1. **Patch Slicing**: Human unified diffs (`git diff`) are parsed into modified file paths and line ranges `[start, start + count)`.
2. **AST Pre-Image Intersection**: For each modified range, the extractor maps line numbers to the enclosing AST symbol definitions (`function`, `method`, `class`) in the base repository commit.
3. **Innermost Scope Resolution**: If a hunk intersects nested scopes (e.g., a function inside a class), the innermost callable scope is designated as the primary target.
4. **Hub Quantification**: In-degree percentiles are computed across all symbols in the repository. Tasks whose target symbols fall in the top-1% in-degree (or top-25 hubs) are flagged with `is_hub_gold: true`.

### Schema: `gold.jsonl`

| Field | Type | Description |
| :--- | :--- | :--- |
| `instance_id` | `str` | Unique SWE-bench Lite task identifier (e.g. `django__django-11099`). |
| `repo` | `str` | Upstream GitHub repository identifier. |
| `gold_files` | `List[str]` | List of relative file paths modified by the gold solution patch. |
| `gold_functions` | `List[str]` | Qualified names of functions/methods modified by the gold patch. |
| `gold_in_degree_percentile` | `float` | In-degree percentile ($[0, 100]$) of the gold symbol in the repository. |
| `is_hub_gold` | `bool` | Flag indicating whether the target function is a high-degree hub ($\ge 99.0\%$). |
| `stratum` | `str` | Lexical target stratum (`A`: unique filename, `B`: stem matches, `C`: lexical sink). |
| `split` | `str` | Partition designation (`pilot`, `dev_val`, or `heldout`). |

---

## 4. Legal Clean-Room Architecture

SWE-bench Lite incorporates repositories with varying licenses (BSD-3-Clause, Apache-2.0, MIT, and GPL-2.0).
To eliminate copyleft license aggregation risks:
1. **Mathematical Structure Only**: The release bundle contains strictly mathematical multigraph topology (matrices, symbol identifiers, degree distributions, line offsets).
2. **Zero Verbatim Source Code**: Raw source code bodies are omitted from the distributed arrays.
3. **Hydration Pipeline**: On-demand hydration of full source text from official upstream Git repositories at immutable commit SHAs is supported via `scripts/reconstruct_dataset.py`.

---

## 5. Quickstart Ingestion

```python
from perron.loaders import load_instance_graph

# Load zero-copy CSR multigraph
graph = load_instance_graph("data_release/graphs/django__django-11099")

print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
print(f"Top-5 hubs: {graph.top_hubs(k=5)}")
```
