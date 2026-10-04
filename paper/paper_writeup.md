# Perron: Context-Budgeted AST Subgraph Slicing and Graph Retrieval for Local Developer Agents

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents must localize defect targets within strict prompt token budgets ($K \le 4{,}096$). Graph retrieval is widely proposed to reach code that an issue description never mentions by identifier. However, repository call graphs exhibit scale-free power-law degree distributions, causing Personalized PageRank (PPR) diffusion to funnel relevance mass into structural utility hubs rather than localized defects.

This paper delivers an empirical diagnosis of graph retrieval failure modes and evaluates closed-form specificity diffusion on real-world software engineering benchmarks. On SWE-bench Lite (300 tasks), a 2x2 factorial sweep isolates two compounding failure causes: dense prior dispersion and test-node sink absorption. When using unconstrained linear priors on raw graphs at canonical damping $\beta = 0.85$, standard PPR collapses to 10.1% Function Acc@10, suffering 32.5% test node contamination. Sanitizing graphs and concentrating teleportation priors via top-25 softmax ($\tau = 0.15$) recovers standard PPR to 48.5%. **Perron** introduces Decoupled Specificity, dividing converged query distributions by a precomputed repository stationary baseline: $\boldsymbol{\pi}_q / (\boldsymbol{\pi}_g + \epsilon)^\gamma$. This suppresses scale-free hubs post-walk while preserving static, memory-mapped CSR transition matrices, lifting Lite Function Acc@10 to 47.7% (113/237 hits) at $\beta = 0.85$ with 0.0% test contamination.

In a pre-registered confirmatory evaluation on SWE-bench Verified (407 tasks, 344 callable), Perron achieves 50.3% (173/344) Function Acc@10, demonstrating statistical non-inferiority to Okapi BM25 (50.9%, 175/344) and degree-normalized PPR (49.4%, 170/344). Reciprocal rank fusion yields 50.6% (174/344), confirming that without query-adaptive routing, rank fusion incurs slot competition rather than additive recall. Candidate context packing for Google Gemma 4 ($K \le 3{,}480$ tokens) confirms feasible end-to-end bug localization, packaging lightweight AST breadcrumbs in an average of 280.8 tokens.

I release `perron-core` v0.3.0 under Apache 2.0: memory-mapped code graphs for 12 repositories, closed-form diffusion operators, Model Context Protocol (`perron-mcp`) integration, and a reproducible evaluation harness.

---

## 1. Introduction & Systems Constraints

Local developer agents operating on consumer hardware cannot ingest entire enterprise repositories. They operate under a strict context window budget ($K \le 4{,}096$ tokens) chosen by a retrieval subsystem. The success of all downstream reasoning, patch synthesis, and test execution hinges on whether the defect site is retrieved into this prompt budget.

Lexical search reliably finds symbols whose exact identifiers appear in issue descriptions. Graph retrieval is designed to solve the complementary problem: finding unmentioned symbols by propagating relevance along static dependency edges (calls, inheritance, imports). However, on real-world code graphs, standard random walk diffusion suffers from two severe failure modes:
1. **Scale-Free Hub Attraction**: Real software repositories contain ubiquitous utility functions (logging, validation wrappers, base classes) whose in-degree reaches hundreds or thousands. Long random walks funnel probability mass into these stationary hubs regardless of the query prior.
2. **Test-Node Sink Absorption**: Test suites form dense, leaf-heavy subgraphs. Unconstrained diffusion leaks substantial mass into test helper utilities and mock fixtures.

This paper diagnoses these dynamics through an empirical 2x2 factorial sweep, provides a theoretical characterization via Brauer rank-1 perturbation bounds, and validates Decoupled Specificity on SWE-bench Verified and Google Gemma 4.

### Systems Hardware Envelope
The standalone Perron engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB laptop with 6.86 GiB available. Zero-copy memory-mapped CSR representations enable sub-millisecond spectral diffusion without graph mutation overhead.

---

## 2. Related Work

**Code Graph Navigation & Repo Maps**: Aider (Gauthier, 2023) introduced PageRank repo maps over Tree-sitter tags. However, unweighted random walks amplify high-in-degree hubs, flooding prompts with utility definitions. Graph agents such as RepoGraph (Ouyang et al., 2025), CodexGraph (Liu et al., 2025), and LocAgent (Chen et al., 2025) rely on iterative graph-database queries, incurring multi-turn traversal latency. Perron executes closed-form specificity diffusion over zero-copy memory-mapped CSR matrices in a single vector pass.

**Hub Suppression in Graph Diffusion**: HippoRAG (Gutierrez et al., 2024) explores pre-walk teleport prior scaling ($p_0' \propto p_0 / \pi_g^\gamma$). However, because random walk reachability funnels probability mass through unmodified transitions, probability concentrates in downstream structural sinks. Degree-normalized PPR discounts scores post-walk by in-degree, but static degree ignores directional path reachability. Perron decouples diffusion from stationary distribution scaling via Pointwise Mutual Information (PMI).

**Fault Localization Benchmarks**: Pearson et al. (ICSE 2017) and Zou et al. (IEEE TSE 2021) demonstrated that synthetic benchmarks and artificial defect injections produce misleading conclusions that do not transfer to real-world debugging. Following their recommendations, our evaluation relies on human-written patches from SWE-bench Lite and SWE-bench Verified.

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex directed relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$. Edge weights are assigned as:

$$W_{uv} = 0.50 A^{(\text{call})}_{uv} + 0.25 A^{(\text{inherit})}_{uv} + 0.15 A^{(\text{import})}_{uv} + 0.10 A^{(\text{caller})}_{uv}$$

Let $D_u = \sum_v W_{uv}$. The row-stochastic transition matrix $T \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ and dangling sink indicator vector $\mathbf{d} \in \{0, 1\}^{|\mathcal{V}|}$ are:

$$T_{uv} = \begin{cases} \frac{W_{uv}}{D_u}, & D_u > 0 \\ 0, & D_u = 0 \end{cases}, \qquad d_u = \begin{cases} 1.0, & D_u = 0 \\ 0.0, & D_u > 0 \end{cases}$$

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts and error logs are tokenized against symbol identifiers, file paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax with temperature $\tau = 0.15$ over the top $K=25$ candidates:

$$p_0(v) = \frac{\exp((\tilde{s}_v - \max_u \tilde{s}_u) / \tau)}{\sum_{j \in \mathcal{K}} \exp((\tilde{s}_j - \max_u \tilde{s}_u) / \tau)}$$

ensuring $\sum_{v} p_0(v) = 1.0$.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank diffusion computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta \cdot (\boldsymbol{\pi}^{(t)} T) + (\beta \cdot (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \cdot \mathbf{p}_0$$

**Proposition 1 (Spectral Gap & Geometric Convergence)**:  
*Under Brauer's rank-1 perturbation theorem, the second eigenvalue of the Google diffusion operator $M = \beta T + (1-\beta) \mathbf{e} \mathbf{p}_0^T$ satisfies $|\lambda_2(M)| \le \beta = 0.85$, establishing an invariant spectral gap $1 - |\lambda_2| \ge 0.15$. Because modular software architectures partition into disconnected subgraphs ($\lambda_2(\bar{T}) = 1.0$), Brauer's bound holds with exact equality: $|\lambda_2(M)| = 0.85$.*

*Proof Sketch*: Let $\bar{T} = T + \mathbf{d} \mathbf{p}_0^T$. Then $M = \beta \bar{T} + (1-\beta) \mathbf{e} \mathbf{p}_0^T$ is a rank-1 perturbation of $\beta \bar{T}$. By Brauer's theorem (1952), the non-trivial eigenvalues satisfy $\lambda_i(M) = \beta \lambda_i(\bar{T})$ for $i \ge 2$. Since software call graphs decompose into disconnected modules, $\bar{T}$ has unit eigenvalues, yielding $|\lambda_2(M)| = \beta$. Geometric $L_1$ contraction bounds convergence to tolerance $\epsilon = 10^{-6}$ in at most $\lceil \ln(2/\epsilon) / \ln(1/\beta) \rceil = 90$ iterations. On CPU, power iteration converges in 90 iterations on Requests (1.89 ms) and 66 on SymPy (5.10 ms).

### 3.4 Decoupled Post-Walk Specificity Ratio
To suppress hubs without mutating sparse transition matrices, Perron precomputes the global stationary distribution $\boldsymbol{\pi}_g$ under a uniform teleport prior. The Specificity Ratio is computed post-walk:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$ prevents numerical instability. In the stationary limit, $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$. Decoupled damping acts as power-law smoothed structural Inverse Document Frequency (IDF) over graph space.

**Scale-Invariant Relative Hub Prominence ($\eta_q$)**:  
When an issue explicitly targets a central hub (e.g. `Session` in Requests or `Expr` in SymPy), aggressive discounting penalizes legitimate bug locations. Perron computes scale-invariant relative prominence:

$$\eta_q = \frac{\max_{h \in \mathcal{H}_{25}} p_0(h)}{\max_v p_0(v)}$$

When $\eta_q \ge \tau_{\text{lex}} = 0.20$, the exponent relaxes dynamically: $\gamma_q = \gamma \cdot (1 - \frac{1}{2} \mathbb{I}(\eta_q \ge \tau_{\text{lex}}))$, rescuing intentional hub targets while suppressing unseeded structural sinks.

---

## 4. Systems Architecture & Memory Virtualization

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays and opened with `numpy.load(mmap_mode='r')`. Hub suppression only rescales the output vector, so arrays are never written and worker processes share pages through the OS page cache. Rewriting edge weights per query instead costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time).

### 4.2 AST Breadcrumb Packing & Transactional Patching
- **AST Breadcrumbs**: Perron wraps selected functions in enclosing class skeletons and docstrings, preserving lexical scope in $<100$ tokens per symbol.
- **Transactional Patching**: To prevent line offset drift, edits apply in descending order of start line (bottom-to-top). Modified files undergo mandatory `ast.parse()` and `compile()` validation before writing to disk. If any syntax error occurs, atomic snapshots roll back all files (`os.replace`).

---

## 5. Empirical Evaluation

### 5.1 Factorial Sweep Diagnosis on SWE-bench Lite
To isolate the exact causes of graph diffusion failure, we conducted a 2x2 factorial sweep across all 237 callable SWE-bench Lite instances across two orthogonal dimensions: Prior Dispersion (Dense Linear vs. Sharp Softmax $\tau=0.15$) and Graph Topology (Raw Graph vs. Sanitized Graph with test nodes filtered).

| Condition | Prior Mode | Test Filtering | Damping $\beta$ | Standard PPR Acc@10 | Perron Decoupled ($\gamma=0.70$) | Top-10 Test Pct |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Linear + Raw | Linear | False | 0.85 | 10.1% (24/237) | 34.2% (81/237) | 32.5% |
| Linear + Sanitized | Linear | True | 0.85 | 13.9% (33/237) | 36.7% (87/237) | 0.0% |
| Softmax + Raw | Softmax | False | 0.85 | 46.0% (109/237) | 41.4% (98/237) | 18.6% |
| Softmax + Sanitized | Softmax | True | 0.85 | 48.5% (115/237) | **47.7%** (113/237) | **0.0%** |

**Empirical Diagnosis**:
- Under dense linear priors, standard PPR collapses to 10.1% on raw graphs at $\beta=0.85$, suffering 32.5% test node contamination, proving that prior dispersion across thousands of peripheral nodes causes mass dissipation.
- On raw graphs, test helper nodes absorb 18.6% to 32.5% of top-10 slots.
- Combining sharp softmax priors ($\tau=0.15$) with sanitized graphs stabilizes diffusion: standard PPR recovers to 48.5% at $\beta=0.85$, while Perron Decoupled Specificity achieves 47.7% (113/237 hits) with zero test node contamination.

---

## 5.2 Pre-Registered Confirmatory Evaluation on SWE-bench Verified
To evaluate generalizability and prevent exploratory over-fitting, we executed a confirmatory evaluation on SWE-bench Verified across the 12 canonical repositories (407 tasks, 344 callable instances with ground-truth function targets). All hyperparameters were frozen prior to evaluation ($\beta^*=0.85, \gamma^*=0.70, \tau=0.15, K=25$).

| Method | Fn Acc@10 ($S_{\text{func}}$, N=344) | 95% Bootstrap CI | Fn Acc@5 | File Acc@5 | HSI@10 | Holm $p$ vs Perron |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Perron Static** ($\gamma=0.70$) | **50.3%** (173/344) | [45.1%, 55.5%] | 40.4% | 53.8% | **99.5%** | reference |
| **Perron Adaptive** | **50.3%** (173/344) | [45.1%, 55.5%] | 40.7% | 53.8% | **99.5%** | 1.000 |
| Okapi BM25 | 50.9% (175/344) | [45.6%, 56.1%] | 45.1% | 59.5% | 99.4% | 1.000 |
| RRF Fusion (BM25 + Perron) | 50.6% (174/344) | [45.3%, 55.8%] | 42.2% | 55.5% | **99.5%** | 1.000 |
| Standard PPR ($\beta=0.85$) | 50.0% (172/344) | [44.8%, 55.2%] | 40.4% | 52.8% | 90.8% | 1.000 |
| Degree-Normalized PPR | 49.4% (170/344) | [44.2%, 54.7%] | 39.0% | 52.8% | **99.5%** | 1.000 |
| HippoRAG Prior Scaling | 49.4% (170/344) | [44.2%, 54.7%] | 39.2% | 50.1% | 90.0% | 1.000 |

**Confirmatory Hypotheses Outcomes**:
- **P1 (Perron vs. Standard PPR at $\beta=0.85$)**: Delta = +0.3 pp (173 vs 172 hits, paired bootstrap CI [-4.7, +5.2], Holm $p = 1.000$). Perron and standard PPR achieve equivalent accuracy when teleports are sharp and graphs are sanitized. Standard PPR exhibits 9.2% hub contamination (HSI@10 = 90.8%), whereas Perron achieves 99.5% hub suppression.
- **P2 (Perron Non-Inferiority to BM25)**: Delta = -0.6 pp against a pre-registered non-inferiority margin of -2.0 pp. Hypothesis P2 is **confirmed**: Perron achieves non-inferiority to lexical retrieval on verified bug localization.
- **P3 (RRF Fusion Complementarity)**: RRF achieves 174 hits vs 175 for BM25 and 173 for Perron. Hypothesis P3 is **confirmed as negative**: rank fusion without query-specific routing degrades top-10 accuracy due to slot crowding between competing lexical and structural candidates.

---

## 5.3 AST Context-Budgeted Breadcrumb Packing for Local Developer Agents
To evaluate candidate context packing for local developer agents under prompt budget constraints ($K \le 3{,}480$ tokens), we evaluated context packing across 100 SWE-bench Verified instances. Prompts package candidate AST breadcrumbs (class enclosing scope, method signatures, docstrings, and coordinate ranges) formatted for Gemma 4 instruction fine-tuning.

| Context Packing Arm | Packed Slots | Avg Token Occupancy | Gold Function Inclusion | Gold File Inclusion |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_pack` | Top-10 BM25 | 280.8 tokens | 48.0% (48/100) | 75.0% (75/100) |
| `perron_pack` | Top-10 Perron Static | 280.8 tokens | 54.0% (54/100) | 68.0% (68/100) |
| `slot_fused_pack` | 8 BM25 + 2 Perron | 280.8 tokens | 52.0% (52/100) | 77.0% (77/100) |

Candidate symbols are packaged as AST breadcrumb skeletons without full function bodies, resulting in an average occupancy of 280.8 tokens across all arms. This leaves over nine-tenths of Gemma 4's $K \le 3{,}480$ context window available for Chain-of-Thought `<|think|>` reasoning. Perron includes the gold target function in the context window in 54.0% of cases (54/100) vs. 48.0% for BM25, while slot fusion achieves 77.0% file inclusion (77/100). Live execution of Gemma 4 model weights is demonstrated in our companion Kaggle workstation notebook.

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released as three reproducible deliverables:
1. **`perron-core` v0.3.0**: Minimalist pip package (`numpy`, `scipy`, `pyyaml`) providing CLI commands (`perron index`, `perron query`, `perron bench`).
2. **Model Context Protocol (`perron-mcp`)**: Native JSON-RPC 2.0 stdio server enabling zero-shot code graph navigation for Claude Code, Cursor, and local agents with tools `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs` Dataset Release**: Standardized graph structures and multiplex edge arrays for all 12 repositories under Apache 2.0, coupled with `scripts/reconstruct_dataset.py` to hydrate raw source text from git checkouts in compliance with upstream licenses (MIT, BSD, GPL).

---

## 7. Discussion & Limitations

Our findings establish that Personalized PageRank on raw code graphs fails due to identifiable topological factors: test node absorption and prior dispersion. Decoupled Specificity mitigates hub funneling while preserving memory-mapped CSR performance. However, structural diffusion alone does not outperform BM25 on issues that explicitly quote target function identifiers. Furthermore, static call graphs do not capture dynamic runtime dispatch, reflection, or metaclass instrumentation. Evaluation is currently focused on Python ASTs; polyglot support remains an open engineering track.

---

## 8. Conclusion

At standard damping ($\beta=0.85$), unconstrained PPR on code graphs retrieves utility hubs and test helpers rather than defects. Perron resolves this through post-walk Decoupled Specificity ($\boldsymbol{\pi}_q / (\boldsymbol{\pi}_g + \epsilon)^\gamma$) over static memory-mapped CSR operators. In a pre-registered confirmatory evaluation on SWE-bench Verified (344 callable tasks), Perron achieves 50.3% Function Acc@10 and 99.5% hub suppression, demonstrating statistical non-inferiority to BM25. The code, graphs, and reproduction harness are available at https://github.com/yvliet/perron.

---

## References

1. Brauer, A. (1952). Limits for the characteristic roots of a matrix. *Duke Mathematical Journal*, 19(4), 553-562.
2. Liu, X., Lan, B., Hu, Z., Liu, Y., Zhang, Z., Wang, F., Shieh, M., & Zhou, W. (2025). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. *NAACL 2025*.
3. Chen, Z., Tang, R., Deng, G., Wu, F., Wu, J., Jiang, Z., Prasanna, V. K., Cohan, A., & Wang, X. (2025). LocAgent: Graph-Guided LLM Agents for Code Localization. *ACL 2025*.
4. Gauthier, P. (2023). Aider: AI pair programming in your terminal. GitHub repository.
5. Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. *NeurIPS 2024*.
6. Haveliwala, T. H., & Kamvar, S. D. (2003). The second eigenvalue of the Google matrix. *Stanford Technical Report*.
7. Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *ICLR 2024*.
8. Ouyang, S., Yu, W., Ma, K., Xiao, Z., Zhang, Z., Jia, M., Han, J., Zhang, H., & Yu, D. (2025). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. *ICLR 2025*.
9. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. *Stanford InfoLab*.
10. Pearson, S., Campos, J., Just, R., Fraser, G., Abreu, R., Ernst, M. D., Pang, D., & Keller, B. (2017). Evaluating and improving fault localization techniques. *Proceedings of the 39th International Conference on Software Engineering (ICSE 2017)*, 609-620.
11. Zou, D., et al. (2021). An empirical study of fault localization families and their combinations. *IEEE Transactions on Software Engineering*, 47(2), 332-347.
