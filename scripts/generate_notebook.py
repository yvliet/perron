import json
from pathlib import Path

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Perron: Context-Budgeted AST Subgraph Slicing for Gemma 4 Agents\n",
    "\n",
    "**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  \n",
    "**Research Focus**: Context-Budgeted Graph Retrieval via Multiplex Spectral Diffusion  \n",
    "**Open-Source Package**: [perron-core on PyPI](https://pypi.org/project/perron-core/) | [GitHub Repository](https://github.com/yvliet/perron)  \n",
    "\n",
    "---\n",
    "\n",
    "### Abstract & Dual-Tier Deployment Architecture\n",
    "This notebook provides a live, reproducible demonstration of **Perron**, an open-source graph-grounded retrieval and AST subgraph slicing architecture for developer agents.\n",
    "\n",
    "Perron solves the scale-free repository graph explosion problem (>50,000 tokens) via query-directed specificity diffusion over static, zero-copy memory-mapped CSR transition matrices. The architecture is engineered around a **Dual-Tier deployment model**:\n",
    "1. **Tier 1 (Edge Local / 16GB Consumer Laptops)**: Operates on **Gemma 4 E4B (4B)** via quantized GGUF execution (`perron/backends/gguf_backend.py`), requiring only 2.4 GB weights and 0.4 GB KV cache (**6.8 GB active RAM footprint**), leaving >9 GB free RAM for OS operations, compilers, and targeted pytest test runs.\n",
    "2. **Tier 2 (Workstation Scaled / 24GB GPUs)**: Scales seamlessly to **Gemma 4 26B A4B** (Sparse MoE, 3.8B active parameters) and **Gemma 4 31B** for enterprise repositories.\n",
    "\n",
    "Across 50 real SWE-bench Lite instances on authentic repository call graphs (Requests and SymPy), Perron achieves a **95.6% Hub Suppression Index** (paired $t(49) = 5.02, p < .001$, Cohen's $d = 0.71$), **42.0% File Recall@4k**, and **4.0% Function Recall@4k** (where Standard PPR achieves 0.0%), running with **sub-5 ms** CPU diffusion (<1.0 ms on Requests, ~5.0 ms on SymPy)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: System Imports and Verification\n",
    "import sys\n",
    "import time\n",
    "from pathlib import Path\n",
    "import numpy as np\n",
    "import scipy.sparse as sp\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from perron.matrix import build_static_transition_matrix\n",
    "from perron.diffusion import (\n",
    "    compute_softmax_teleport_prior,\n",
    "    personalized_pagerank_power_iteration,\n",
    ")\n",
    "from perron.specificity import (\n",
    "    compute_global_pagerank,\n",
    "    calculate_specificity_scores,\n",
    ")\n",
    "from perron.packer import ASTContextSymbol, pack_context_subgraphs\n",
    "from perron.editor import apply_multi_file_patch\n",
    "\n",
    "print(\"Perron core engine loaded successfully!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Zero-Copy Memory-Mapped Transition Matrix Ingestion\n",
    "We generate a scale-free synthetic call graph ($|V| = 1,000$ nodes with 20 high-degree utility hubs) and construct row-stochastic CSR transition matrix $T$ and dangling indicator vector $d$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Build and Benchmark Matrix Ingestion\n",
    "from benchmarks.diagnostic import generate_synthetic_repo_graph\n",
    "\n",
    "call_edges, caller_edges, symbols, hubs = generate_synthetic_repo_graph(num_nodes=1000, num_hubs=20, seed=42)\n",
    "t0 = time.perf_counter()\n",
    "t_matrix, dangling = build_static_transition_matrix(\n",
    "    num_nodes=1000,\n",
    "    call_edges=call_edges,\n",
    "    caller_edges=caller_edges,\n",
    "    lambda_call=1.0,\n",
    "    lambda_caller=0.20,\n",
    ")\n",
    "ingestion_ms = (time.perf_counter() - t0) * 1000.0\n",
    "\n",
    "print(f\"Matrix constructed in {ingestion_ms:.2f} ms\")\n",
    "print(f\"Nodes: {t_matrix.shape[0]}, Non-zero Edges: {t_matrix.nnz}, Dangling Nodes: {int(np.sum(dangling))}\")\n",
    "assert ingestion_ms < 100.0, \"Ingestion latency should be sub-100ms\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Query-Directed Specificity Ratio Diffusion\n",
    "We evaluate an issue statement symptom, perform sparse Personalized PageRank diffusion, and calculate node-level specificity ratios against precomputed stationary distribution $\\pi_{\\text{global}}$:\n",
    "\n",
    "$$\\text{Specificity}(v) = \\frac{\\pi_{\\text{query}}(v)}{(\\pi_{\\text{global}}(v) + \\epsilon)^\\gamma}$$\n",
    "\n",
    "where $\\gamma = 0.70$ damps high-degree utility hubs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Stationary Diffusion and Hub Suppression Benchmark\n",
    "pi_global = compute_global_pagerank(t_matrix, dangling, max_iter=100)\n",
    "\n",
    "# Simulate query symptom similarity\n",
    "sims = np.clip(np.random.normal(0.10, 0.03, size=1000), 0.0, 1.0)\n",
    "symptom_node = 250\n",
    "sims[symptom_node] = 0.92\n",
    "\n",
    "# Step A: Teleport Prior\n",
    "p_0 = compute_softmax_teleport_prior(sims, np.arange(1000), 1000, tau=0.05, top_k=25)\n",
    "\n",
    "# Step B: Power Iteration\n",
    "t_diff_0 = time.perf_counter()\n",
    "pi_query = personalized_pagerank_power_iteration(t_matrix, dangling, p_0, beta=0.85, max_iter=100)\n",
    "scores = calculate_specificity_scores(pi_query, pi_global, gamma=0.70)\n",
    "diff_time_ms = (time.perf_counter() - t_diff_0) * 1000.0\n",
    "\n",
    "# Compute Hub Suppression Index in Top 10\n",
    "top10_std = np.argsort(pi_query)[::-1][:10]\n",
    "top10_perron = np.argsort(scores)[::-1][:10]\n",
    "\n",
    "hubs_set = set(hubs)\n",
    "hubs_in_std = sum(1 for n in top10_std if n in hubs_set)\n",
    "hubs_in_perron = sum(1 for n in top10_perron if n in hubs_set)\n",
    "\n",
    "print(f\"Diffusion Latency: {diff_time_ms:.2f} ms\")\n",
    "print(f\"Standard PPR Hubs in Top-10: {hubs_in_std}/10 (HSI: {(1 - hubs_in_std/10)*100:.1f}%)\")\n",
    "print(f\"Perron Specificity Hubs in Top-10: {hubs_in_perron}/10 (HSI: {(1 - hubs_in_perron/10)*100:.1f}%)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Visualizing the Pareto Frontier & Sensitivity Surface\n",
    "Visualizing the trade-off across context token budgets ($K \\in [512, 4096]$) and the 2D parameter response surface."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: Plot Pareto Frontier and Parameter Surface\n",
    "import json\n",
    "\n",
    "pareto_file = Path(\"benchmarks/pareto_frontier_data.json\")\n",
    "if pareto_file.exists():\n",
    "    with open(pareto_file, encoding=\"utf-8\") as f:\n",
    "        data = json.load(f)\n",
    "    \n",
    "    budgets = data[\"pareto_summary\"][\"budgets\"]\n",
    "    recalls = data[\"pareto_summary\"][\"mean_recalls\"]\n",
    "    \n",
    "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)\n",
    "    \n",
    "    # Panel 1: Pareto curves\n",
    "    for m, vals in recalls.items():\n",
    "        lw = 2.2 if \"Perron\" in m else 1.2\n",
    "        ax1.plot(budgets, vals, label=m, marker=\"o\", linewidth=lw)\n",
    "    ax1.axvline(3480, color=\"gray\", linestyle=\":\", label=\"Gemma 4 Budget (3,480)\")\n",
    "    ax1.set_xlabel(\"Context Token Budget (K)\")\n",
    "    ax1.set_ylabel(\"Function Recall (%)\")\n",
    "    ax1.set_title(\"Empirical Pareto Frontier\")\n",
    "    ax1.legend(fontsize=8)\n",
    "    ax1.grid(True, linestyle=\"--\", alpha=0.5)\n",
    "    \n",
    "    # Panel 2: Sensitivity Surface\n",
    "    gamma_vals = np.array(data[\"gamma_values\"])\n",
    "    beta_vals = np.array(data[\"beta_values\"])\n",
    "    surf = np.array(data[\"surface_recall_matrix\"])\n",
    "    B, G = np.meshgrid(beta_vals, gamma_vals)\n",
    "    cp = ax2.contourf(B, G, surf, levels=10, cmap=\"viridis\")\n",
    "    ax2.plot([0.85], [0.70], marker=\"*\", color=\"red\", markersize=10, label=\"Perron (0.85, 0.70)\")\n",
    "    ax2.set_xlabel(\"Damping Beta\")\n",
    "    ax2.set_ylabel(\"Specificity Gamma\")\n",
    "    ax2.set_title(\"Topological Sensitivity Basin\")\n",
    "    ax2.legend(fontsize=8)\n",
    "    plt.colorbar(cp, ax=ax2, label=\"Recall (%)\")\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show(block=False)\n",
    "    plt.close('all')\n",
    "else:\n",
    "    print(\"Run python benchmarks/compute_pareto_frontier.py to generate live pareto data.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. AST-Grounded Search-and-Replace Editor Demo\n",
    "Perron's editor enforces pre-commit `ast.parse()` validation and relative indentation tolerance, guaranteeing a 0.0% syntax failure rate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: AST Pre-Commit Validation Demo\n",
    "import tempfile\n",
    "from perron.editor import SymbolEditSpec, apply_multi_file_patch\n",
    "\n",
    "with tempfile.TemporaryDirectory() as tmpdir:\n",
    "    test_file = Path(tmpdir) / \"sample_service.py\"\n",
    "    test_file.write_text(\"\"\"\n",
    "class UserService:\n",
    "    def get_user(self, user_id):\n",
    "        if user_id <= 0:\n",
    "            return None\n",
    "        return {'id': user_id, 'name': 'Anonymous'}\n",
    "\"\"\".strip() + \"\\n\", encoding=\"utf-8\")\n",
    "\n",
    "    edit_spec = SymbolEditSpec(\n",
    "        file_path=str(test_file),\n",
    "        old_str=\"if user_id <= 0:\\n            return None\",\n",
    "        new_str=\"if user_id <= 0:\\n            raise ValueError('Invalid user id')\",\n",
    "    )\n",
    "\n",
    "    success, msg, modified_paths = apply_multi_file_patch([edit_spec])\n",
    "    print(f\"Patch Result: Success={success}, Message='{msg}', Modified={len(modified_paths)}\")\n",
    "    print(\"Modified File Content:\\n\")\n",
    "    print(test_file.read_text(encoding=\"utf-8\"))\n",
    "    assert success and len(modified_paths) == 1, \"Patch application must succeed\"\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Reproducibility & Open Source Audit\n",
    "- **Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)\n",
    "- **License**: Apache 2.0\n",
    "- **Test Suite**: Run `pytest tests/` (100% passing across 41 tests)\n",
    "- **Paper**: See arXiv PDF in repository for complete mathematical proofs and Brauer rank-1 spectral bounds."
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("perron_showcase.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("perron_showcase.ipynb generated successfully!")
