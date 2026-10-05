# When and Why Does Personalized PageRank Fail on Code Graphs? Analyzing Seed Dispersion, Prior Calibration, and Decoupled Specificity Damping

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents localize defect targets under strict prompt token budgets ($K \le 4{,}096$). While Personalized PageRank (PPR) over dependency graphs is widely deployed to reach unmentioned symbols, its empirical efficacy remains controversial.

Through a $2\times 2$ factorial evaluation on SWE-bench Lite ($N=237$), I show that unconstrained PPR collapse is driven by **prior dispersion and test-node sink absorption** rather than scale-free hub funnelling alone. Under diffuse linear priors, Standard PPR collapses to 10.1% Function Acc@10 with 32.5% test contamination. Concentrating priors via top-25 softmax ($\tau = 0.15$) rescues Standard PPR to 48.5%, while test sanitization adds +3.8 percentage points. **Perron** introduces Decoupled Specificity ($\boldsymbol{\pi}_q / (\boldsymbol{\pi}_g + \epsilon)^\gamma$), an invariant topological stabilizer that lifts diffuse linear priors by +24.1 pp (34.2% vs. 10.1%) while restoring hub suppression to 99.5% (matching BM25's 99.4% baseline) over static memory-mapped CSR transition matrices.

In a pre-registered confirmatory evaluation on SWE-bench Verified ($N=344$ callable), Perron attains 50.3% (173/344) Function Acc@10, showing no statistically detectable difference at top-10 (non-inferiority inconclusive against the $-2.0$ pp margin), with disjoint hits reflecting top-10 boundary rank permutations rather than an orthogonal structural discovery advantage. BM25 leads at top-5 (45.1% vs. 40.4%). In candidate prompt packing for Google Gemma 4, lightweight AST breadcrumbs occupy an average of 283.8 whitespace words, preserving 91.8% of the 3,480-token prompt budget for native `<|think|>` Chain-of-Thought reasoning.

I release `perron-core` v0.3.0, `perron-mcp`, and 12-repository CSR graphs under Apache 2.0.

---

## 1. Introduction & Systems Constraints

Local developer agents on consumer hardware operate under strict prompt budgets ($K \le 4{,}096$ tokens), where downstream reasoning and patch synthesis depend on retrieving the defect site within budget.

While lexical search finds symbols explicitly named in issue text, graph retrieval aims to reach unmentioned symbols by traversing static dependency edges. On real-world code graphs, standard diffusion fails via:
1. **Prior Dispersion Across Lexical Leaves**: Diffuse linear priors inject fractional mass across thousands of peripheral symbols, dissipating random walks before reaching defect clusters.
2. **Test-Node Sink Absorption**: Dense test-suite subgraphs siphon probability mass into test helpers and mock fixtures.

This paper diagnoses these dynamics via a $2\times 2$ factorial sweep, characterizes spectral convergence via Brauer rank-1 bounds, and validates Decoupled Specificity on SWE-bench Verified.

### Systems Hardware Envelope
The engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB host with 6.86 GiB available. Zero-copy memory-mapped CSR storage enables spectral diffusion in 1.89 ms on Requests and 5.10 ms on SymPy, sharing physical memory across worker threads via OS page caching ($W \rightarrow 1$).

---

## 2. Related Work

**Code Graph Navigation & Repo Maps**: Aider (Gauthier, 2023) introduced PageRank repo maps over Tree-sitter tags; however, unweighted walks amplify central hubs. Multi-turn graph agents like RepoGraph (Ouyang et al., 2025), CodexGraph (Liu et al., 2025), and LocAgent (Chen et al., 2025) rely on iterative graph queries, incurring substantial latency. Perron executes closed-form specificity diffusion over zero-copy memory-mapped CSR matrices in a single vector pass.

**Hub Suppression in Graph Diffusion**: HippoRAG (Gutierrez et al., 2024) scales pre-walk teleport priors via Inverse Passage Frequency ($s_i = |P_i|^{-1}$), but walks still funnel into structural sinks. Degree-normalized PPR discounts post-walk by in-degree, ignoring reachability. Perron decouples diffusion from stationary scaling via Pointwise Mutual Information (PMI).

**Fault Localization Benchmarks & Scalable Diffusion**: Scalable diffusion builds on push-based approximate PPR (Andersen et al., 2006), APPNP (Klicpera et al., 2019), PPRGo (Bojchevski et al., 2020), and stationary-normalized rooted PageRank (Liben-Nowell & Kleinberg, 2007). In fault localization, Pearson et al. (ICSE 2017) and Zou et al. (IEEE TSE 2021) showed that synthetic benchmarks fail to transfer to real-world debugging; my evaluation relies on authentic human patches from SWE-bench Lite and Verified.

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex directed relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$:

$$W_{uv} = 0.50 A^{(\text{call})}_{uv} + 0.25 A^{(\text{inherit})}_{uv} + 0.15 A^{(\text{import})}_{uv} + 0.10 A^{(\text{caller})}_{uv}$$

Let $D_u = \sum_v W_{uv}$. The row-stochastic transition matrix $T \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$ and dangling sink indicator $\mathbf{d} \in \{0, 1\}^{|\mathcal{V}|}$ are:

$$T_{uv} = \begin{cases} \frac{W_{uv}}{D_u}, & D_u > 0 \\ 0, & D_u = 0 \end{cases}, \qquad d_u = \begin{cases} 1.0, & D_u = 0 \\ 0.0, & D_u > 0 \end{cases}$$

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts are tokenized against identifiers, paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax ($\tau = 0.15$) over top $K=25$ candidates:

$$p_0(v) = \begin{cases} \frac{\exp((\tilde{s}_v - \max_u \tilde{s}_u) / \tau)}{\sum_{j \in \mathcal{K}} \exp((\tilde{s}_j - \max_u \tilde{s}_u) / \tau)}, & v \in \mathcal{K} \\ 0.0, & v \notin \mathcal{K} \end{cases}$$

where $\tilde{s}_v$ denotes min-max normalized BM25 relevance scores over the top $K=25$ candidates, ensuring $\sum_{v} p_0(v) = 1.0$.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta \cdot (\boldsymbol{\pi}^{(t)} T) + (\beta \cdot (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \cdot \mathbf{p}_0$$

**Proposition 1 (Spectral Gap & Geometric Convergence)**:  
*Under Brauer's rank-1 perturbation theorem (Brauer, 1952), the Google operator $M = \beta (T + \mathbf{d}\mathbf{p}_0^T) + (1-\beta) \mathbf{1} \mathbf{p}_0^T$ satisfies $|\lambda_2(M)| \le \beta = 0.85$, establishing an invariant spectral gap $1 - |\lambda_2| \ge 0.15$. Because modular software architectures partition into disconnected subgraphs ($\lambda_2(\bar{T}) = 1.0$), Brauer's bound holds with exact equality: $|\lambda_2(M)| = 0.85$.*

*Proof Sketch*: Let $\bar{T} = T + \mathbf{d} \mathbf{p}_0^T$. Then $M = \beta \bar{T} + (1-\beta) \mathbf{1} \mathbf{p}_0^T$ is a rank-1 perturbation of $\beta \bar{T}$. By Brauer (1952), $\lambda_i(M) = \beta \lambda_i(\bar{T})$ for $i \ge 2$. Because the maximum initial $L_1$ error between two probability vectors on the simplex satisfies $\|\mathbf{e}^{(0)}\|_1 \le 2.0$, contraction bounds convergence to tolerance $\epsilon = 10^{-6}$ in at most $\lceil \ln(2/\epsilon) / \ln(1/\beta) \rceil = 90$ iterations. On CPU, power iteration converges in 90 iterations on Requests (1.89 ms) and 66 on SymPy (5.10 ms), matching the worst-case bound on modular graphs.

### 3.4 Decoupled Post-Walk Specificity Ratio
To suppress hubs without mutating sparse transition matrices, Perron precomputes global stationary distribution $\boldsymbol{\pi}_g$ under uniform teleportation. Specificity is computed post-walk:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$. In the stationary limit, $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$. Decoupled damping acts as power-law smoothed structural Inverse Document Frequency (IDF), analogous to popularity discounting in bipartite networks. Let $\mathcal{H}_{25}$ denote the top-25 in-degree hub symbols. Hub suppression is measured via the Hub Suppression Index: $\text{HSI}(q) = 1 - (|R_{10}(q) \cap \mathcal{H}_{25}| / 10)$.

**Scale-Invariant Hub Prominence ($\eta_q$)**: When an issue explicitly names a central hub (`Session` in Requests, `Expr` in SymPy), aggressive discounting penalizes legitimate defect sites. Perron computes $\eta_q = \max_{h \in \mathcal{H}_{25}} p_0(h) / \max_v p_0(v)$. When $\eta_q \ge 0.20$, the exponent relaxes: $\gamma_q = \gamma \cdot (1 - \frac{1}{2} \mathbb{I}(\eta_q \ge 0.20))$, rescuing intentional targets. Figure 2 visualizes this Pareto trade-off on the pre-walk development validation split ($N=100$).

![Figure 2: Specificity Exponent Sweep on dev_val (N=100)](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/specificity_gamma_sweep.png)

---

## 4. Systems Architecture & Memory Virtualization

![Figure 1: Perron Pipeline Architecture](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/perron_pipeline.png)

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays opened via `numpy.load(mmap_mode='r')`. Hub suppression only rescales the output vector, allowing parallel workers to share physical memory via the OS page cache ($W \rightarrow 1$). In contrast, rewriting edge weights per query costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time).

### 4.2 AST Breadcrumbs & Transactional Patching
- **AST Breadcrumbs**: Perron wraps candidate functions in enclosing class skeletons and docstrings ($<100$ tokens per symbol).
- **Transactional Patching**: Edits apply bottom-to-top by start line with mandatory `ast.parse()` validation and atomic rollback on syntax failures (`os.replace`).

![Figure 3: Comparative Edit Interfaces](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/comparative_edit_interfaces.png)

---

## 5. Empirical Evaluation

### 5.1 Factorial Sweep Diagnosis on SWE-bench Lite
To isolate the causes of graph diffusion failure, I conducted a $2\times 2$ factorial sweep across all 237 callable SWE-bench Lite instances across Prior Dispersion (Dense Linear vs. Sharp Softmax $\tau=0.15$) and Graph Topology (Raw vs. Sanitized). Tasks filter to callable function edits.

| Condition | Prior Mode | Test Filtering | Damping $\beta$ | Standard PPR Acc@10 | Perron Decoupled ($\gamma=0.70$) | Top-10 Test Pct |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Linear + Raw | Linear | False | 0.85 | 10.1% (24/237) | 34.2% (81/237) | 32.5% |
| Linear + Sanitized | Linear | True | 0.85 | 13.9% (33/237) | 36.7% (87/237) | 0.0% |
| Softmax + Raw | Softmax | False | 0.85 | 46.0% (109/237) | 41.4% (98/237) | 18.6% |
| Softmax + Sanitized | Softmax | True | 0.85 | **48.5%** (115/237) | 47.7% (113/237) | **0.0%** |
| *Lexical Baseline (BM25)* | BM25 | True | - | 46.0% (109/237) | - | 0.0% |

**Key Findings**:
- Under dense linear priors, standard PPR collapses to 10.1% on raw graphs from 32.5% test contamination.
- Isolating test nodes eliminates sink leakage (+3.8 pp recovery).
- Concentrating priors via top-25 softmax ($\tau=0.15$) rescues Standard PPR to 48.5%, rendering Perron neutral (47.7%, $-0.8$ pp).
- Perron acts as an invariant topological stabilizer under diffuse priors (+24.1 pp gain), remaining neutral under concentrated priors.

---

### 5.2 Damping Sensitivity and Topological Regularization
On SWE-bench Lite, varying damping $\beta \in [0.50, 0.85]$ demonstrates that lower damping restricts random walks tightly around issue seeds, whereas unconstrained diffusion at $\beta=0.85$ requires calibrated softmax priors to prevent mass dispersion.

### 5.3 Pre-Registered Confirmatory Evaluation on SWE-bench Verified
I evaluated generalizability on SWE-bench Verified across 12 canonical repositories (344 callable instances, filtering out non-callable file additions). Hyperparameters were frozen prior to evaluation ($\beta=0.85$, $\gamma^*=0.70$, $\tau=0.15$, $K=25$).

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
- **P1 (Perron vs. Standard PPR)**: Difference $+0.3$ pp (173 vs 172 hits, paired bootstrap CI $[-2.9, +3.5]$, Holm $p = 1.000$). Isolating test nodes leaves no detectable accuracy difference. Standard PPR exhibits 9.2% hub contamination (HSI@10 = 90.8%), whereas Perron sustains 99.5% hub suppression.
- **P2 (Perron Parity to BM25)**: Difference $-0.6$ pp (paired bootstrap CI $[-3.5, +2.3]$ pp). Because the lower bound crosses the pre-registered $-2.0$ pp margin, strict non-inferiority is **statistically inconclusive**. Point estimates show parity, while BM25 leads at top-5 (45.1% vs 40.4% Fn Acc@5, 59.5% vs 53.8% File Acc@5).
- **P3 (Boundary Jitter & Fusion)**: RRF fusion (174 hits vs 175 for BM25) confirms slot competition under unrouted merging. BM25 and Perron retrieve 161 targets jointly, 157 concordant misses, 14 exclusively by BM25, and 12 exclusively by Perron ($p = 0.845$, two-sided McNemar). In Stratum C (unnamed targets, $N=131$), BM25 hits 28 to Perron's 27 ($p = 1.000$). The theoretical oracle union reaches 187/344 (54.4%), but unrouted rank fusion fails to surpass standalone BM25: disjoint hits reflect top-10 boundary rank permutations rather than an orthogonal structural discovery mechanism.

---

### 5.4 AST Context-Budgeted Breadcrumb Packing for Local Developer Agents
I evaluated candidate context packing under prompt budget constraints ($K \le 3{,}480$ tokens) across all 344 SWE-bench Verified callable instances, packaging AST breadcrumb skeletons without raw function bodies.

| Context Packing Arm | Packed Slots | Avg Token Occupancy | Gold Function Inclusion | Gold File Inclusion |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_pack` | Top-10 BM25 | 283.8 tokens | 51.5% (177/344) | 72.4% (249/344) |
| `perron_pack` | Top-10 Perron | 283.8 tokens | 51.2% (176/344) | 67.2% (231/344) |
| `slot_fused_pack` | 8 BM25 + 2 Perron | 283.8 tokens | **52.3%** (180/344) | **72.7%** (250/344) |

Across arms, AST breadcrumbs average 283.8 whitespace words, preserving 91.8% of prompt capacity for Gemma 4 `<|think|>` reasoning. Table 3 inclusion credits enclosing class scopes (yielding 177 and 176 hits vs. 175 and 173 in Table 2). Standalone Perron matches BM25 in function inclusion (51.2% vs. 51.5%), while BM25 leads in file inclusion (72.4% vs. 67.2%). Slot fusion balances coverage (52.3% function, 72.7% file) at 513.96 ms joint pipeline latency (1.89 ms CSR diffusion vs. AST parsing and IO overhead).

### 5.5 Ablation Study
Removing Decoupled Specificity ($\gamma=0$) drops HSI from 99.5% to 90.8% and collapses diffuse linear priors by -24.1 pp. Test-suite sanitization prevents an additional 3.8 pp sink leakage under unconstrained random walks.

![Figure 4: Diagnostic Retrieval and Hub Suppression Profile](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/benchmark_statistical_evaluation.png)

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released as five reproducible deliverables under Apache 2.0 / CC0-1.0:
1. **`perron-core` v0.3.0**: Python library on [PyPI](https://pypi.org/project/perron-core/0.3.0/) with zero-copy CSR diffusion and CLI (`perron index`, `perron query`).
2. **`perron-mcp`**: Stdio MCP server exposing `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs`**: Precomputed CSR graph topologies for all 12 repositories, reconstructed via `scripts/reconstruct_dataset.py`.
4. **`swebench-strata-annotations`**: 801 per-instance lexical accessibility annotations on [Kaggle Datasets](https://www.kaggle.com/datasets/yvliet/swebench-strata-annotations).
5. **Gemma 4 Workstation Kernel**: Interactive GPU notebook on [Kaggle](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation).

---

## 7. Discussion & Limitations

Personalized PageRank fails on code graphs primarily from test node absorption and prior dispersion. Decoupled Specificity stabilizes diffuse priors (+24.1 pp) and suppresses hub accumulation (99.5% vs. 90.8% HSI) with zero-copy CSR efficiency. Four limitations remain: (1) static call resolution can conflate ambiguous method names; (2) multiplex edge weights [0.50, 0.25, 0.15, 0.10] are fixed across repositories rather than tuned per-project; (3) static graphs miss dynamic reflection; and (4) evaluation focuses on Python ASTs; polyglot grammar interfaces remain an active direction.

---

## 8. Conclusion

On software code graphs, Personalized PageRank collapse under default damping ($\beta=0.85$) is driven by teleport prior dispersion and test-node sink absorption. When priors are focused via top-25 softmax on sanitized graphs, Standard PPR recovers to 48.5%, matching Perron Decoupled Specificity (47.7%). Perron serves as an invariant topological stabilizer under diffuse priors (+24.1 pp gain) and restores hub suppression to 99.5%. In a pre-registered confirmatory evaluation on SWE-bench Verified (344 callable tasks), Perron shows no statistically detectable difference at top-10 vs BM25 (non-inferiority inconclusive against $-2.0$ pp). While an unconstrained oracle union reaches 54.4% (187/344), unrouted rank fusion fails to exceed standalone BM25, confirming that disjoint hits reflect top-10 boundary rank permutations rather than an orthogonal structural advantage. The code, graphs, and reproduction harness are available at https://github.com/yvliet/perron.

---

## References

1. Brauer, A. *Duke Math. J.*, 1952.
2. Chen, Z., et al. LocAgent. *ACL*, 2025.
3. Gauthier, P. Aider, 2023.
4. Gutierrez, B. J., et al. HippoRAG. *NeurIPS*, 2024.
5. Haveliwala, T. H. & Kamvar, S. D. *Stanford Tech. Rep.*, 2003.
6. Jimenez, C. E., et al. SWE-bench. *ICLR*, 2024.
7. Liu, X., et al. CodexGraph. *NAACL*, 2025.
8. Ouyang, S., et al. RepoGraph. *ICLR*, 2025.
9. Page, L., et al. PageRank. *Stanford InfoLab*, 1999.
10. Pearson, S., et al. Fault Localization. *ICSE*, 2017.
11. Zou, D., et al. Fault Localization Families. *IEEE TSE*, 2021.
