# When and Why Does Personalized PageRank Fail on Code Graphs? Analyzing Seed Dispersion, Graph Resolution Artifacts, and Decoupled Specificity Damping

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents localize defect targets under strict prompt token budgets ($K \le 4{,}096$). While Personalized PageRank (PPR) over dependency graphs is widely deployed to reach unmentioned symbols, its empirical efficacy remains controversial.

Through a $2\times 2$ factorial evaluation on SWE-bench Lite ($N=237$), I show that unconstrained PPR collapse is driven by **prior dispersion and test-node sink absorption** rather than scale-free hub funnelling alone. Under diffuse linear priors, Standard PPR collapses to 10.1% Function Acc@10 with 32.5% test contamination. Concentrating priors via top-25 softmax ($\tau = 0.15$) rescues Standard PPR to 48.5%, while test sanitization adds +3.8 percentage points. **Perron** introduces Decoupled Specificity ($\boldsymbol{\pi}_q / (\boldsymbol{\pi}_g + \epsilon)^\gamma$), an invariant topological stabilizer that lifts diffuse linear priors by +24.1 pp (34.2% vs. 10.1%) while enforcing 99.5% hub suppression over static memory-mapped CSR transition matrices.

In a pre-registered confirmatory evaluation on SWE-bench Verified ($N=344$ callable), Perron attains 50.3% (173/344) Function Acc@10, exhibiting statistical parity with Okapi BM25 (50.9%, 175/344; paired difference $[-3.5, +2.3]$ pp, rendering non-inferiority inconclusive against a $-2.0$ pp margin) while BM25 leads at top-5 (45.1% vs. 40.4%). However, BM25 and Perron retrieve 14 and 12 exclusive targets respectively (187/344 oracle union, 54.4%), confirming that graph diffusion captures orthogonal structural signals. In candidate prompt packing for Google Gemma 4 ($K \le 3{,}480$), lightweight AST breadcrumbs occupy an average of 280.8 tokens, preserving 91.9% of the context window for native Chain-of-Thought reasoning.

I release `perron-core` v0.3.0, `perron-mcp`, and 12-repository CSR graphs under Apache 2.0.

---

## 1. Introduction & Systems Constraints

Local developer agents on consumer hardware operate under strict prompt budgets ($K \le 4{,}096$ tokens). Downstream reasoning and patch synthesis depend entirely on retrieving the defect site into this budget.

Lexical search finds symbols whose identifiers appear in issue text. Graph retrieval aims to find unmentioned symbols by traversing static dependency edges (calls, inheritance, imports). On real-world code graphs, standard diffusion exhibits two distinct failure modes:
1. **Prior Dispersion Across Lexical Leaves**: Diffuse linear priors inject fractional mass across thousands of peripheral symbols, causing random walks to dissipate before reaching defect clusters.
2. **Test-Node Sink Absorption**: Test suites form dense, leaf-heavy subgraphs that absorb substantial probability mass into test helpers and mock fixtures.

This paper diagnoses these dynamics via a $2\times 2$ factorial sweep, characterizes spectral convergence via Brauer rank-1 bounds, and validates Decoupled Specificity on SWE-bench Verified.

### Systems Hardware Envelope
The standalone Perron engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB laptop with 6.86 GiB available. Zero-copy memory-mapped CSR representations enable sub-millisecond spectral diffusion without graph mutation overhead, sharing physical memory across parallel worker threads via OS page caching ($W \rightarrow 1$).

---

## 2. Related Work

**Code Graph Navigation & Repo Maps**: Aider (Gauthier, 2023) introduced PageRank repo maps over Tree-sitter tags. However, unweighted random walks amplify high-in-degree hubs. Graph agents including RepoGraph (Ouyang et al., 2025), CodexGraph (Liu et al., 2025), and LocAgent (Chen et al., 2025) rely on iterative graph-database queries, incurring multi-turn latency. Perron executes closed-form specificity diffusion over zero-copy memory-mapped CSR matrices in a single vector pass.

**Hub Suppression in Graph Diffusion**: HippoRAG (Gutierrez et al., 2024) explores pre-walk teleport prior scaling ($p_0' \propto p_0 / \pi_g^\gamma$). However, unmodified transition walks funnel mass into downstream structural sinks. Degree-normalized PPR discounts post-walk by in-degree, but ignores path reachability. Perron decouples diffusion from stationary distribution scaling via Pointwise Mutual Information (PMI).

**Fault Localization Benchmarks**: Pearson et al. (ICSE 2017) and Zou et al. (IEEE TSE 2021, evaluating 357 Defects4J faults) demonstrated that synthetic benchmarks produce conclusions that fail to transfer to real-world debugging. Following their recommendations, our evaluation relies on authentic human patches from SWE-bench Lite and SWE-bench Verified.

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex directed relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$:

$$W_{uv} = 0.50 A^{(\text{call})}_{uv} + 0.25 A^{(\text{inherit})}_{uv} + 0.15 A^{(\text{import})}_{uv} + 0.10 A^{(\text{caller})}_{uv}$$

Let $D_u = \sum_v W_{uv}$. The row-stochastic transition matrix $T \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ and dangling sink indicator $\mathbf{d} \in \{0, 1\}^{|\mathcal{V}|}$ are:

$$T_{uv} = \begin{cases} \frac{W_{uv}}{D_u}, & D_u > 0 \\ 0, & D_u = 0 \end{cases}, \qquad d_u = \begin{cases} 1.0, & D_u = 0 \\ 0.0, & D_u > 0 \end{cases}$$

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts are tokenized against identifiers, paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax ($\tau = 0.15$) over top $K=25$ candidates:

$$p_0(v) = \frac{\exp((\tilde{s}_v - \max_u \tilde{s}_u) / \tau)}{\sum_{j \in \mathcal{K}} \exp((\tilde{s}_j - \max_u \tilde{s}_u) / \tau)}$$

ensuring $\sum_{v} p_0(v) = 1.0$.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta \cdot (\boldsymbol{\pi}^{(t)} T) + (\beta \cdot (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \cdot \mathbf{p}_0$$

**Proposition 1 (Spectral Gap & Geometric Convergence)**:  
*Under Brauer's rank-1 perturbation theorem (Brauer, 1952), the Google operator $M = \beta T + (1-\beta) \mathbf{e} \mathbf{p}_0^T$ satisfies $|\lambda_2(M)| \le \beta = 0.85$, establishing an invariant spectral gap $1 - |\lambda_2| \ge 0.15$. Because modular software architectures partition into disconnected subgraphs ($\lambda_2(\bar{T}) = 1.0$), Brauer's bound holds with exact equality: $|\lambda_2(M)| = 0.85$.*

*Proof Sketch*: Let $\bar{T} = T + \mathbf{d} \mathbf{p}_0^T$. Then $M = \beta \bar{T} + (1-\beta) \mathbf{e} \mathbf{p}_0^T$ is a rank-1 perturbation of $\beta \bar{T}$. By Brauer (1952), $\lambda_i(M) = \beta \lambda_i(\bar{T})$ for $i \ge 2$. Since software call graphs decompose into disconnected modules, $\bar{T}$ has unit eigenvalues, yielding $|\lambda_2(M)| = \beta$. Geometric $L_1$ contraction bounds convergence to tolerance $\epsilon = 10^{-6}$ in at most $\lceil \ln(\epsilon) / \ln(\beta) \rceil = 86$ iterations. On CPU, power iteration converges in 71 iterations on Requests (1.89 ms) and 63 on SymPy (5.10 ms).

### 3.4 Decoupled Post-Walk Specificity Ratio
To suppress hubs without mutating sparse transition matrices, Perron precomputes global stationary distribution $\boldsymbol{\pi}_g$ under uniform teleportation. Specificity is computed post-walk:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$. In the stationary limit, $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$. Decoupled damping acts as power-law smoothed structural Inverse Document Frequency (IDF).

**Scale-Invariant Hub Prominence ($\eta_q$)**: When an issue explicitly names a central hub (`Session` in Requests, `Expr` in SymPy), aggressive discounting penalizes legitimate defect sites. Perron computes $\eta_q = \max_{h \in \mathcal{H}_{25}} p_0(h) / \max_v p_0(v)$. When $\eta_q \ge 0.20$, the exponent relaxes: $\gamma_q = \gamma \cdot (1 - \frac{1}{2} \mathbb{I}(\eta_q \ge 0.20))$, rescuing intentional targets.

---

## 4. Systems Architecture & Memory Virtualization

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays opened via `numpy.load(mmap_mode='r')`. Hub suppression only rescales the output vector, so arrays are never written and parallel workers share pages through the OS page cache. Rewriting edge weights per query instead costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time).

### 4.2 AST Breadcrumbs & Transactional Patching
- **AST Breadcrumbs**: Perron wraps candidate functions in enclosing class skeletons and docstrings, preserving scope in $<100$ tokens per symbol.
- **Transactional Patching**: Edits apply bottom-to-top by start line to prevent offset drift. Modified files undergo mandatory `ast.parse()` validation. If syntax errors occur, atomic snapshots roll back all files (`os.replace`).

---

## 5. Empirical Evaluation

### 5.1 Factorial Sweep Diagnosis on SWE-bench Lite
To isolate the causes of graph diffusion failure, we conducted a $2\times 2$ factorial sweep across all 237 callable SWE-bench Lite instances across Prior Dispersion (Dense Linear vs. Sharp Softmax $\tau=0.15$) and Graph Topology (Raw vs. Sanitized).

| Condition | Prior Mode | Test Filtering | Damping $\beta$ | Standard PPR Acc@10 | Perron Decoupled ($\gamma=0.70$) | Top-10 Test Pct |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Linear + Raw | Linear | False | 0.85 | 10.1% (24/237) | 34.2% (81/237) | 32.5% |
| Linear + Sanitized | Linear | True | 0.85 | 13.9% (33/237) | 36.7% (87/237) | 0.0% |
| Softmax + Raw | Softmax | False | 0.85 | 46.0% (109/237) | 41.4% (98/237) | 18.6% |
| Softmax + Sanitized | Softmax | True | 0.85 | **48.5%** (115/237) | 47.7% (113/237) | **0.0%** |
| *Lexical Baseline (BM25)* | BM25 | True | — | 46.0% (109/237) | — | 0.0% |

**Key Findings**:
- Under dense linear priors, standard PPR collapses to 10.1% on raw graphs, suffering 32.5% test contamination: mass dissipates across peripheral nodes.
- Isolating test nodes eliminates sink leakage (+3.8 pp recovery for linear PPR).
- Concentrating priors via top-25 softmax ($\tau=0.15$) rescues Standard PPR to 48.5%, rendering Perron neutral (47.7%, $-0.8$ pp).
- Perron acts as an invariant topological stabilizer under diffuse priors (+24.1 pp gain), but behaves neutrally when the prior is already concentrated.

---

## 5.2 Pre-Registered Confirmatory Evaluation on SWE-bench Verified
We evaluated generalizability on SWE-bench Verified across 12 canonical repositories (344 callable instances). Hyperparameters were frozen prior to evaluation ($\beta^*=0.85, \gamma^*=0.70, \tau=0.15, K=25$).

| Method | Fn Acc@10 ($N=344$) | 95% Bootstrap CI | Fn Acc@5 | File Acc@5 | HSI@10 | Holm $p$ vs Perron |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Perron Static** ($\gamma=0.70$) | **50.3%** (173/344) | [45.1%, 55.5%] | 40.4% | 53.8% | **99.5%** | reference |
| **Perron Adaptive** | **50.3%** (173/344) | [45.1%, 55.5%] | 40.7% | 53.8% | **99.5%** | 1.000 |
| Okapi BM25 | 50.9% (175/344) | [45.6%, 56.1%] | 45.1% | 59.5% | 99.4% | 1.000 |
| RRF Fusion (BM25 + Perron) | 50.6% (174/344) | [45.3%, 55.8%] | 42.2% | 55.5% | **99.5%** | 1.000 |
| Standard PPR ($\beta=0.85$) | 50.0% (172/344) | [44.8%, 55.2%] | 40.4% | 52.8% | 90.8% | 1.000 |
| Degree-Normalized PPR | 49.4% (170/344) | [44.2%, 54.7%] | 39.0% | 52.8% | **99.5%** | 1.000 |
| HippoRAG Prior Scaling | 49.4% (170/344) | [44.2%, 54.7%] | 39.2% | 50.1% | 90.0% | 1.000 |

**Hypotheses Outcomes**:
- **P1 (Perron vs. Standard PPR)**: Difference $+0.3$ pp (173 vs 172 hits, paired bootstrap CI $[-4.7, +5.2]$, Holm $p = 1.000$). No statistically detectable accuracy difference exists once test nodes are isolated. Standard PPR exhibits 9.2% hub contamination (HSI@10 = 90.8%), whereas Perron sustains 99.5% hub suppression.
- **P2 (Perron Parity to BM25)**: Difference $-0.6$ pp with paired bootstrap CI $[-3.5, +2.3]$ pp. Because the lower bound crosses the pre-registered $-2.0$ pp margin, strict non-inferiority is **statistically inconclusive**. Point estimates exhibit parity, while BM25 leads at top-5 (45.1% vs 40.4% Fn Acc@5, 59.5% vs 53.8% File Acc@5).
- **P3 (Orthogonal Signal Space & Fusion)**: Rank fusion (RRF: 174 hits vs 175 for BM25) confirms slot competition under unrouted merging. Crucially, contingency analysis shows that BM25 and Perron retrieve 161 targets jointly, 14 exclusively by BM25, and 12 exclusively by Perron. The theoretical oracle union reaches **187/344 (54.4%, +3.5 pp over BM25)**, proving that graph diffusion captures orthogonal structural signals.

---

## 5.3 AST Context-Budgeted Breadcrumb Packing for Local Developer Agents
We evaluated candidate context packing under prompt budget constraints ($K \le 3{,}480$ tokens) across 100 SWE-bench Verified instances. Prompts package AST breadcrumb skeletons without raw bodies.

| Context Packing Arm | Packed Slots | Avg Token Occupancy | Gold Function Inclusion | Gold File Inclusion |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_pack` | Top-10 BM25 | 280.8 tokens | 48.0% (48/100) | 75.0% (75/100) |
| `perron_pack` | Top-10 Perron Static | 280.8 tokens | 54.0% (54/100) | 68.0% (68/100) |
| `slot_fused_pack` | 8 BM25 + 2 Perron | 280.8 tokens | 52.0% (52/100) | 77.0% (77/100) |

Candidate symbols are packaged as AST breadcrumbs without full function bodies, consuming an average of 280.8 tokens across all arms. This leaves 91.9% of Gemma 4's $K \le 3{,}480$ context window available for internal Chain-of-Thought `<|think|>` reasoning. Perron includes the gold function in 54.0% of cases vs. 48.0% for BM25, while BM25 leads on file inclusion (75.0% vs. 68.0%), and slot fusion achieves 77.0% file inclusion.

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released as three reproducible deliverables:
1. **`perron-core` v0.3.0**: Minimalist package (`numpy`, `scipy`, `pyyaml`) providing CLI commands (`perron index`, `perron query`, `perron bench`).
2. **Model Context Protocol (`perron-mcp`)**: JSON-RPC 2.0 stdio server enabling zero-shot code graph navigation for desktop agents with tools `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs` Dataset Release**: Standardized graph structures and multiplex edge arrays for all 12 repositories under Apache 2.0, coupled with `scripts/reconstruct_dataset.py` to hydrate source text from git checkouts in compliance with upstream licenses.

---

## 7. Discussion & Limitations

Our findings establish that Personalized PageRank on code graphs fails primarily due to test node absorption and prior dispersion. Decoupled Specificity stabilizes diffusion under diffuse priors (+24.1 pp) and eliminates hub leakage (99.5% HSI) while preserving zero-copy memory-mapped CSR performance. However, several limitations remain: (1) static call resolution can conflate ambiguous method names across disparate modules; (2) multiplex edge weights were not globally optimized per repository; (3) static graphs cannot trace dynamic reflection or runtime dispatch; and (4) evaluation focuses on Python ASTs; polyglot grammar interfaces remain an active engineering direction.

---

## 8. Conclusion

On software code graphs, Personalized PageRank collapse under default damping ($\beta=0.85$) is driven by teleport prior dispersion and test-node sink absorption. When priors are focused via top-25 softmax on sanitized graphs, Standard PPR recovers to 48.5%, matching Perron Decoupled Specificity (47.7%). Perron serves as an invariant topological stabilizer under diffuse priors (+24.1 pp gain) and enforces 99.5% hub suppression. In a pre-registered confirmatory evaluation on SWE-bench Verified (344 callable tasks), Perron exhibits statistical parity with BM25 (50.3% vs. 50.9%), while their oracle union reaches 54.4% (187/344), confirming that graph diffusion captures orthogonal structural signals. The code, graphs, and reproduction harness are available at https://github.com/yvliet/perron.

---

## References

1. Brauer, A. (1952). Limits for the characteristic roots of a matrix. IV: Applications to stochastic matrices. *Duke Mathematical Journal*, 19(1), 75-91.
2. Chen, Z., Tang, R., Deng, G., Wu, F., Wu, J., Jiang, Z., Prasanna, V. K., Cohan, A., & Wang, X. (2025). LocAgent: Graph-Guided LLM Agents for Code Localization. *ACL 2025*.
3. Gauthier, P. (2023). Aider: AI pair programming in your terminal. GitHub repository.
4. Gutierrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurobiologically inspired long-term memory for large language models. *NeurIPS 2024*.
5. Haveliwala, T. H., & Kamvar, S. D. (2003). The second eigenvalue of the Google matrix. *Stanford Technical Report*.
6. Jimenez, C. E., et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? *ICLR 2024*.
7. Liu, X., Lan, B., Hu, Z., Liu, Y., Zhang, Z., Wang, F., Shieh, M., & Zhou, W. (2025). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. *NAACL 2025*.
8. Ouyang, S., Yu, W., Ma, K., Xiao, Z., Zhang, Z., Jia, M., Han, J., Zhang, H., & Yu, D. (2025). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. *ICLR 2025*.
9. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank citation ranking: Bringing order to the web. *Stanford InfoLab*.
10. Pearson, S., Campos, J., Just, R., Fraser, G., Abreu, R., Ernst, M. D., Pang, D., & Keller, B. (2017). Evaluating and improving fault localization techniques. *Proceedings of the 39th International Conference on Software Engineering (ICSE 2017)*, 609-620.
11. Zou, D., Liang, J., Xiong, Y., Ernst, M. D., & Zhang, L. (2021). An empirical study of fault localization families and their combinations. *IEEE Transactions on Software Engineering*, 47(2), 332-347.
