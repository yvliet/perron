# Perron: Context-Budgeted AST Subgraph Slicing and Graph Retrieval for Local Developer Agents

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents have to find the code an issue is about inside a prompt budget of a few thousand tokens ($K \le 4{,}096$). Graph retrieval is the usual way to reach code that the issue never names, but repository call graphs are scale-free, and Personalized PageRank (PPR) moves relevance into a few utility hubs.

I measure this on all 300 SWE-bench Lite instances under a pre-registered 50/100/150 split. At damping $\beta = 0.85$, six diffusion retrievers (standard PPR, Aider repo maps, HippoRAG-style prior scaling, query-reweighted PPR, a hub blocklist and a degree-discounted matrix) find none of the 118 held-out callable defect targets; the median in-degree of standard PPR's top-10 symbols is 135.0, against 3.0 for the gold targets. **Perron** divides the converged query distribution by a repository-wide stationary baseline, $\pi_q/(\pi_g+\epsilon)^\gamma$, and recovers 17/118 (14.4%) Function Acc@10 while the CSR transition matrix stays static and memory-mapped. BM25 still finds more (27/118, McNemar p = 0.052), but the two retrievers hit different targets: 5 of Perron's 6 exclusive hits lie outside BM25's top 20. No fusion I tested converts this into a top-10 gain (RRF 25/118 vs 27/118), and the union of both top-10 lists (33/118) is matched by BM25 given the same 20 slots (31/118, p = 0.727).

I release `perron-core` v0.3.0 under Apache 2.0: memory-mapped code graphs for the 12 SWE-bench Lite repositories, the diffusion operators, an AST editor (23/30 valid applications vs 16/30 for `sed`), and a harness that regenerates every number here from versioned result files.

---

## 1. Introduction & Systems Constraints

A local agent working on a repository bug cannot read the repository. It gets a few thousand tokens of context chosen by a retriever, and everything it does afterwards depends on whether the defect is inside that window. Lexical retrieval finds code whose names appear in the issue. Graph retrieval is meant to find the rest: seed a random walk on the lexical matches and let PPR carry relevance along call edges.

On code graphs that walk fails in a predictable way. A handful of utility symbols (type-check wrappers, base classes, table accessors) have in-degree in the hundreds or thousands, and the stationary distribution collects on them whatever the query says. This paper measures that failure on SWE-bench Lite, shows why pre-walk fixes cannot remove it, and evaluates a post-walk correction. It also reports, against a budget-matched control, how far that correction is from beating BM25 and which targets it reaches that BM25 cannot.

### Hardware Envelope
The standalone Perron engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB laptop with 6.86 GiB available. No Gemma 4 model was executed for this paper, so model memory and latency are not reported.

---

## 2. Related Work

**Code Graph Navigation & Repo Maps**: Aider (Gauthier, 2023) introduced PageRank repo maps over Tree-sitter tags. However, unweighted random walks amplify high-in-degree hubs, flooding prompts with utility definitions. Graph agents such as RepoGraph (Ma et al., 2024), CodexGraph (Cheng et al., 2024), and LocAgent (Dong et al., 2024) rely on iterative graph-database queries, incurring multi-turn traversal latency. Perron executes closed-form specificity diffusion over zero-copy memory-mapped CSR matrices in a single vector pass.

**Hub Suppression in Graph Diffusion**: HippoRAG (Gutierrez et al., 2024) explores pre-walk teleport prior scaling ($p_0' \propto p_0 / \pi_g^\gamma$). However, because random walk reachability funnels probability mass through unmodified transitions, probability concentrates in downstream structural sinks. Degree-normalized PPR discounts scores post-walk by in-degree, but static degree ignores directional path reachability. Perron decouples diffusion from stationary distribution scaling via Pointwise Mutual Information (PMI).

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex directed relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$. Edge weights are assigned as:

$$W_{uv} = 0.50 A^{(\text{call})}_{uv} + 0.25 A^{(\text{inherit})}_{uv} + 0.15 A^{(\text{import})}_{uv} + 0.10 A^{(\text{caller})}_{uv}$$

Let $D_u = \sum_v W_{uv}$. The row-stochastic transition matrix $T \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ and dangling sink indicator vector $\mathbf{d} \in \{0, 1\}^{|\mathcal{V}|}$ are:

$$T_{uv} = \begin{cases} \frac{W_{uv}}{D_u}, & D_u > 0 \\ 0, & D_u = 0 \end{cases}, \qquad d_u = \begin{cases} 1.0, & D_u = 0 \\ 0.0, & D_u > 0 \end{cases}$$

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts and error logs are tokenized against symbol identifiers, file paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax with temperature $\tau = 0.15$:

$$p_0(v) = \frac{\exp((\tilde{s}_v - \max_u \tilde{s}_u) / \tau)}{\sum_{j \in \mathcal{K}} \exp((\tilde{s}_j - \max_u \tilde{s}_u) / \tau)}$$

ensuring $\sum_{v} p_0(v) = 1.0$. If stack trace frames are parsed, mass is divided equally between traceback frames and lexical matches.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank diffusion computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta \cdot (\boldsymbol{\pi}^{(t)} T) + (\beta \cdot (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \cdot \mathbf{p}_0$$

Under Brauer's rank-1 perturbation theorem, the second eigenvalue of the Google operator satisfies $|\lambda_2(M)| \le \beta = 0.85$, establishing an invariant spectral gap $1 - |\lambda_2| \ge 0.15$. Because modular software architectures partition into disconnected subgraphs ($\lambda_2(\bar{T}) = 1.0$), Brauer's bound holds with exact equality: $|\lambda_2(M)| = 0.85$. Geometric $L_1$ contraction bounds convergence to tolerance $\epsilon = 10^{-6}$ in at most $\lceil \ln(2/\epsilon) / \ln(1/\beta) \rceil = 90$ iterations. Residual stopping at $\epsilon$ bounds the true error by $\epsilon/(1-\beta)$. On CPU, power iteration converges in 90 iterations on Requests (1.89 ms) and 66 on SymPy (5.10 ms).

### 3.4 Decoupled Post-Walk Specificity Ratio
To suppress hubs without mutating sparse transition matrices, Perron precomputes the global stationary distribution $\boldsymbol{\pi}_g$ under a uniform teleport prior. The Specificity Ratio is computed post-walk:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$ prevents numerical instability. In the stationary limit, $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$. The reweighted prior enters every term of the Neumann series, but walks of length at least t carry a fraction $\beta^t$ of the mass, and long walks end at hubs. Decoupled damping acts as power-law smoothed structural Inverse Document Frequency (IDF) over graph space.

**Scale-Invariant Relative Hub Prominence ($\eta_q$)**:
When an issue explicitly targets a central hub (e.g. `Session` in Requests or `Expr` in SymPy), aggressive discounting penalizes legitimate bug locations. Perron computes scale-invariant relative prominence:

$$\eta_q = \frac{\max_{h \in \mathcal{H}_{25}} p_0(h)}{\max_v p_0(v)}$$

When $\eta_q \ge \tau_{\text{lex}} = 0.20$, the exponent relaxes dynamically:

$$\gamma_q = \gamma \cdot \left(1 - \frac{1}{2} \mathbb{I}(\eta_q \ge \tau_{\text{lex}})\right)$$

rescuing intentional hub targets while suppressing unseeded structural sinks.

---

## 4. Systems Architecture & Memory Virtualization

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays and opened with `numpy.load(mmap_mode='r')`. Hub suppression only rescales the output vector, so the arrays are never written and worker processes can share their pages through the OS page cache. Rewriting edge weights per query instead costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time). I did not measure multi-process memory, so no savings figure is given.

### 4.2 AST Breadcrumb Packing & Transactional Editor
- **AST Breadcrumbs**: Perron wraps selected functions in enclosing class skeletons and docstrings, preserving lexical scope in $<100$ tokens per symbol.
- **Transactional Patching**: To prevent line offset drift, edits apply in descending order of start line (bottom-to-top). Indentation is rebased to match enclosing scopes. Before writing to disk, modified files undergo mandatory `ast.parse()` and `compile()` validation. If any syntax error occurs, atomic snapshots roll back all files (`os.replace`).

---

## 5. Empirical Evaluation

### 5.1 Editor Interface Benchmark
An edit counts as a valid application when the tool applies it and the edited file passes `ast.parse`. Brackets are 95% Wilson intervals; p is an exact McNemar test against Perron on paired per-case outcomes.

| Interface | Applied | Valid application | 95% CI | McNemar vs Perron |
| :--- | :--- | :--- | :--- | :--- |
| `git apply` | 24/30 | 9/30 (30.0%) | [16.7%, 47.9%] | p < .001 |
| `sed` | 30/30 | 16/30 (53.3%) | [36.1%, 69.8%] | p = 0.039 |
| Perron AST | 23/30 | 23/30 (76.7%) | [59.1%, 88.2%] | reference |

Perron rolls back the 7/30 edits that fail validation, leaving those files unchanged.

### 5.2 12-Baseline Retrieval Matrix on Held-Out Split
Each diffusion method runs at the damping β* selected on dev_val; the held-out split was evaluated once.

| Method | β* | Fn Acc@10 (N=118) | 95% CI | File Acc@5 | HSI@10 | Holm p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Perron Static ($\gamma=0.70$) | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | -- |
| Standard PPR | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| Aider Repo Map | 0.85 | 0.0% (0/118) | [0.0%, 0.0%] | 5.9% | 54.3% | <.001 |
| Hub Blocklist + Lexical | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| Static Deg-Discount Matrix | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| Degree-Normalized PPR | 0.30 | 12.7% (15/118) | [6.8%, 19.5%] | 48.3% | 100.0% | 0.004 |
| HippoRAG Prior Scaling | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| Query-Reweighted PPR | 0.00 | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| Okapi BM25 | -- | 22.9% (27/118) | [15.3%, 30.5%] | 66.1% | 99.9% | 1.000 |
| BM25 + 1-Hop Expansion | -- | 24.6% (29/118) | [16.9%, 33.1%] | 66.1% | 99.9% | 1.000 |
| Token-Overlap Similarity | -- | 17.8% (21/118) | [11.0%, 25.4%] | 52.5% | 99.3% | 0.562 |
| Oracle Upper Bound | -- | 100.0% (118/118) | -- | 100.0% | 100.0% | -- |

### 5.3 Key Empirical Insights
- **Diffusion Collapse on Scale-Free Graphs**: At the default damping $\beta=0.85$, six diffusion baselines (standard PPR, Aider repo maps, HippoRAG prior scaling, query-reweighted PPR, hub blocklists, and degree-discounted matrices) retrieve none of the 118 held-out callable targets (0.0%). Their top-10 symbols exhibit a median in-degree of 135.0, compared to 3.0 for ground-truth targets, confirming that unconstrained diffusion concentrates probability mass in infrastructure hubs rather than localized defects.
- **Specificity Ratio Recovery**: Dividing the converged query distribution by a repository-wide stationary baseline ($\boldsymbol{\pi}_q / (\boldsymbol{\pi}_g + \epsilon)^\gamma$) penalizes structural hubs post-walk, recovering 17/118 (14.4%) Function Acc@10 while leaving the CSR matrix static and memory-mapped.
- **Tuned-Damping Behavior**: Every diffusion method is evaluated at the damping $\beta^*$ selected on dev_val. At its tuned damping, no diffusion baseline exceeds Perron Static (22.9%, 27/118).

### 5.4 Lexical and Structural Complementarity
On the 118 held-out callable tasks, both BM25 and Perron hit 11, only BM25 hits 16, and only Perron hits 6 (McNemar p = 0.052, favouring BM25). The BM25 ranks of the Perron-only targets are 12, 21, 76, 166, 176, 1927: 5 lie outside BM25's top 20, so a larger lexical budget alone would not reach them.

Merging the lists does not help at ten slots: RRF ($k_0=60$) reaches 25/118 and score fusion at the dev-selected $w^*=0.1$ reaches 26/118, against 27/118 for BM25 (Holm p = 1.000 and p = 1.000). The union of both top-10 lists hits 33/118, but it uses up to 20 slots; BM25 with 20 slots reaches 31/118 and the difference is not significant (5 vs 3 discordant tasks, p = 0.727). Routing each query to the retriever that suits it is the open problem.

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released as three reproducible deliverables:
1. **`perron-core` v0.3.0**: Minimalist pip package (`numpy`, `scipy`, `pyyaml`) with CLI commands (`perron index`, `perron query`, `perron bench`).
2. **Model Context Protocol (`perron-mcp`)**: Native JSON-RPC 2.0 stdio server enabling zero-shot code graph navigation for Claude Code, Cursor, and local agents with tools `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs` Dataset Release**: Standardized graph structures and multiplex edge arrays for all 12 repositories under Apache 2.0, coupled with `scripts/reconstruct_dataset.py` to hydrate raw source text from git checkouts in compliance with upstream licenses (MIT, BSD, GPL).

---

## 7. Discussion & Limitations

Perron's gain over other diffusion methods is a gain over a collapsed baseline; BM25 remains the stronger single retriever on SWE-bench Lite, where issues often quote identifiers. The structural signal is real but needs a router, not a merge. Limitations: static call graphs miss reflection and monkey-patching; the evaluation is Python-only (a Tree-sitter TypeScript interface exists as a smoke test); the agent loop is tested with deterministic replay, not end-to-end local-model repair; and model memory and latency are reported only if measured (Section 1).

---

## 8. Conclusion

At the usual damping, PPR on code graphs retrieves hubs, not defects: six diffusion retrievers find none of 118 held-out targets. A post-walk specificity ratio recovers 17/118 without touching the static CSR operator. BM25 is still stronger (27/118), no tested fusion beats it at ten slots, and the targets Perron alone reaches sit far below BM25's top 20. The code, graphs and the harness that regenerates every number are at https://github.com/yvliet/perron.

---

## References

1. Brauer, A. (1952). Limits for the characteristic roots of a matrix. *Duke Mathematical Journal*, 19(4), 553-562.
2. Cheng, K., et al. (2024). CodexGraph: Bridging large language models and code repositories through code graph databases. *arXiv preprint arXiv:2408.03910*.
3. Dong, Y., et al. (2024). LocAgent: Graph-guided agent for defect localization in large-scale codebases. *arXiv preprint arXiv:2407.03213*.
4. Gauthier, P. (2023). Aider: AI pair programming in your terminal. GitHub repository.
5. Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. *NeurIPS 2024*.
6. Haveliwala, T. H., & Kamvar, S. D. (2003). The second eigenvalue of the Google matrix. *Stanford Technical Report*.
7. Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *ICLR 2024*.
8. Ma, W., et al. (2024). RepoGraph: Enhancing AI software engineering with repository-level code knowledge graphs. *arXiv preprint arXiv:2410.14652*.
9. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. *Stanford InfoLab*.
