# Does Graph Structure Help a Small LLM Localize Bugs? A Diagnostic Study of Personalized PageRank, Topological Regularization, and Gemma 4 Reranking on Code Graphs

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents localize defect targets within constrained prompt budgets ($K \le 4{,}096$). On SWE-bench Verified ($N=344$), under a pre-specified, hash-committed protocol, candidate pools scored pointwise by frozen `gemma-4-e2b-it` achieve 47.1% Acc@5 under graph 1-hop expansion ($A_4$) vs. 45.1% for depth BM25 ($A_2$, $\Delta = +2.03$ pp) and 46.2% for seed reranking ($A_1$, $\Delta = +0.87$ pp, net 3 tasks, $p=0.5488$), classified as null-compatible ($p_{\text{holm}} = 0.287, 0.302$). Un-reranked BM25 preserves superior rank order (MRR 0.344 vs 0.341). Downstream testing ($N=100$) reveals that candidate recall (35%--38%), rather than reader discrimination (74.0% oracle top-1, 67.5% chance-adjusted), forms the decisive bottleneck. Across a 63,984-configuration SpMM sweep, random-walk retrieval collapses onto a dose-response curve governed by prior effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($p < .001$, GLMM). Softmax concentration ($\tau = 0.15$) recovers PPR from 10.1% to 48.5%, while TOST confirms hubs do not degrade localization (equivalence within $\pm 9.3$ pp, $p < .001$). **Perron** stabilizes diffuse priors (+24.1 pp) and enforces 99.5% hub suppression over zero-copy CSR matrices. I release `scores.parquet`, `perron-core`, and benchmarks under Apache 2.0.

---

## 1. Introduction & Systems Constraints

Local developer agents operate under prompt budgets ($K \le 4{,}096$ tokens) where defect localization is token-constrained. Lexical search locates symbols named in issue text, while graph retrieval aims to reach unmentioned symbols along static dependency edges. On code graphs, unconstrained diffusion collapses via:
1. **Prior Dispersion Across Lexical Leaves**: Diffuse linear priors inject fractional mass across thousands of peripheral symbols, dissipating random walks.
2. **Test-Node Sink Absorption**: Dense test subgraphs siphon probability mass into test helpers and mock fixtures.

I diagnose these dynamics via a $2\times 2$ factorial sweep, establish the $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ dose-response law, evaluate Decoupled Specificity on SWE-bench Verified, and assess local LLM reranking under a two-stage pre-registered Gemma 4 protocol.

### Systems Hardware Envelope
The engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB host, completing sparse CSR diffusion in 1.89 ms on Requests and 5.10 ms on SymPy via memory-mapped page caching. Reader ingestion was evaluated on Kaggle GPU across $N=100$ SWE-bench Verified tasks using Gemma 4 E2B in FP16 with PLE on a Tesla T4 (2.37 GiB peak VRAM, 3505.0 ms mean TTFT, 11.7 tokens/sec decode). In downstream reader testing, Gemma 4 achieves 74.0% top-1 localization on oracle contexts (67.5% chance-adjusted, Section 5.4), whereas exploratory pilot generation under an unconstrained 256-token budget exhausted tokens in internal reasoning before emitting closing tags, showing the necessity for decoupled reasoning and extraction budgets.

![Figure 1](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/perron_pipeline.png)

---

## 2. Related Work

**Code Graph Navigation**: Aider (Gauthier, 2023) uses PageRank repo maps over Tree-sitter tags; however, unweighted walks amplify hubs. Multi-turn agents like RepoGraph (Ouyang et al., 2025) and LocAgent (Chen et al., 2025) rely on iterative LLM reasoning. Perron executes closed-form specificity diffusion over zero-copy CSR matrices in a single vector pass.

**Hub Suppression**: HippoRAG (Gutierrez et al., 2024) scales pre-walk priors by Inverse Passage Frequency, but walks still funnel into sinks. Degree-normalized PPR discounts by in-degree, ignoring reachability. Perron decouples diffusion from stationary scaling via Pointwise Mutual Information.

**Fault Localization**: Scalable diffusion builds on push-based approximate PPR (Andersen et al., 2006), APPNP (Klicpera et al., 2019), and rooted PageRank (Liben-Nowell & Kleinberg, 2007). In fault localization, Pearson et al. (2017) and Zou et al. (2021) showed synthetic benchmarks fail to transfer; my evaluation uses authentic human patches from SWE-bench Lite and Verified.

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$:

$$W_{u,v} = 0.50 A^{(\text{call})}_{u,v} + 0.25 A^{(\text{inherit})}_{u,v} + 0.15 A^{(\text{import})}_{u,v} + 0.10 A^{(\text{caller})}_{u,v}$$

Matrix $T$ normalizes out-degree $D_u = \sum_v W_{u,v}$ ($T_{u,v} = W_{u,v}/D_u$); dangling indicator $d_u = 1.0$ when $D_u = 0$ else $0.0$.

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts are tokenized against identifiers, paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax ($\tau = 0.15$) over top $K=25$ candidates: for $v \in \mathcal{K}$, $p_0(v) \propto \exp((\tilde{s}_v - \max_u \tilde{s}_u)/\tau)$, and $0$ otherwise.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta (\boldsymbol{\pi}^{(t)} T) + (\beta (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \mathbf{p}_0$$

**Proposition 1 (Spectral Gap & Convergence)**:  
*By Brauer's theorem (Brauer, 1952), Google operator $M$ satisfies |\lambda_2(M)| \le \beta = 0.85, establishing spectral gap $1 - |\lambda_2| \ge 0.15$.* Geometric contraction bounds $\epsilon = 10^{-6}$ convergence in $\lceil \ln(2/\epsilon) / \ln(1/\beta) \rceil = 90$ iterations (Requests: 90 iterations, 1.89 ms; SymPy: 66 iterations, 5.10 ms).

### 3.4 Decoupled Post-Walk Specificity Ratio
Perron precomputes global stationary distribution $\boldsymbol{\pi}_g$ under uniform teleportation, adapting stationary-normalized PageRank (Liben-Nowell & Kleinberg, 2007) with exponent $\gamma_q$:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$. With uniform prior, $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$, yielding interpolation $(1-\gamma)\ln \pi_q(v) + \gamma \text{PMI}(v; q)$. Hub suppression is measured by $\text{HSI}(q) = 1 - (|R_{10}(q) \cap \mathcal{H}_{25}| / 10)$.

**Scale-Invariant Hub Prominence ($\eta_q$)**: When an issue names a central hub, prominence $\eta_q = \max_{h \in \mathcal{H}_{25}} p_0(h) / \max_v p_0(v)$ relaxes the exponent: $\gamma_q = \gamma \cdot (1 - \frac{1}{2} \mathbb{I}(\eta_q \ge 0.20))$, rescuing intentional targets (Figure 2, dev_val $N=100$).

![Figure 2](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/specificity_gamma_sweep.png)

---

## 4. Systems Architecture & Memory Virtualization

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays opened via `numpy.load(mmap_mode='r')`. Specificity output rescaling allows parallel workers to share physical memory via OS page caching. Dynamically rewriting edge weights per query costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time).

### 4.2 AST Breadcrumbs & Transactional Patching
- **AST Breadcrumbs**: Perron wraps candidate functions in enclosing class skeletons and docstrings ($<100$ tokens per symbol).
- **Transactional Patching**: Edits apply bottom-to-top by start line with mandatory `ast.parse()` validation and atomic rollback on syntax failures (`os.replace`).

![Figure 3](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/comparative_edit_interfaces.png)

---

## 5. Empirical Evaluation

### 5.1 Continuous Dose-Response Surface and Resolvent Linearity (H1, H2 & H3)
Via resolvent linearity ($\mathbf{u}(p_0) = \sum_{i=1}^K w_i \mathbf{u}(e_{s_i})$), I evaluated 186 configurations across four prior families on SWE-bench Verified ($N=344$, 63,984 SpMM evaluations in 26 min).

![Figure 4](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/neff_dose_response.png)

Retrieval accuracy collapses onto an empirical dose-response curve governed by prior effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$. A binomial GLMM confirms vanishing cutoff effects:

$$\text{logit}(P(Y_{ij}=1)) = \beta_0 + \beta_1 \ln n_{\text{eff}} + \beta_2 (\ln n_{\text{eff}})^2 + a \ln k + b \ln \tau + \gamma_j$$

with $\beta_1 = 0.325, \beta_2 = -0.075, a = -0.008, b = -0.036$. The GLMM vertex sits at $n_{\text{eff}}^* \approx 8.81$ (50.9% hit rate, matching BM25 top-10 exactly due to seed retention). Calibrated default ($k=25, \tau=0.15 \implies n_{\text{eff}} \approx 6.035$, 49.4% hit rate) sits within the $[4, 8]$ plateau (pseudo-$R^2 = 0.197$, confirming **H1**).

**Hypothesis H2 (Iso-Entropy Contrast)**: Power-law and softmax priors matched on $n_{\text{eff}} \approx 6.0$ ($k=10$ vs $k=50$) exhibit no significant difference ($p = 0.535$, McNemar).

**Hypothesis H3 (Hub Impact and Equivalence)**: On tasks where hubs enter top-10 slots under Standard PPR ($N=109$), paired McNemar ($p = 0.683$) and TOST show equivalence within $\pm 9.3$ pp ($p < .001$); across all tasks, equivalence holds within $\pm 5.0$ pp ($p < .001$). Graph diffusion produces zero positive ranking lift over lexical seeds (50.9% vs 50.3%). Decoupled Specificity functions as an adaptive prior regularizer (+24.1 pp under diffuse linear) and aesthetic hub filter (99.5% HSI) rather than an accuracy remedy.

---

## 5.2 Topological Reachability and Relation-Stratified Null Models
Boolean reachability from lexical top-10 seeds ($N=344$) covers 50.9% of gold targets, expanding to 57.8% at 1 hop (median $|N_1|=27$) and 66.9% at 2 hops (median $|N_2|=324$). Hub pruning drops 2-hop reachability to 62.2% (214/344).

Degree-preserving Maslov-Sneppen null models (100 replicates) yield 63.8% 2-hop reachability ($z_E = +6.46, p < .001$). Same-file containment captures 66.9%, matching 2-hop reachability via symmetric discordance: 90.1% concordance (310/344), where 17 cross-file graph hits balance 17 intra-file targets lacking static call edges.

---

## 5.3 Pre-Registered Confirmatory Evaluation on SWE-bench Verified
I evaluated generalizability on SWE-bench Verified across 12 canonical repositories (344 callable instances). Protocols and hyperparameters were frozen under pre-registration Git commit `6908e78` (protocol SHA-256: `89416e40...`, split SHA-256: `95c7ef3c...`) prior to evaluation ($\beta=0.85$, $\gamma^*=0.70$, $\tau=0.15$, $K=25$).

| Method | Fn Acc@10 | 95% CI | Fn Acc@5 | HSI@10 | Holm $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Perron Static** ($\gamma=0.70$) | **50.3%** | [45.1%, 55.5%] | 40.4% | **99.5%** | ref |
| Okapi BM25 | 50.9% | [45.6%, 56.1%] | 45.1% | 99.4% | 1.000 |
| RRF Fusion (BM25 + Perron) | 50.6% | [45.3%, 55.8%] | 42.2% | **99.5%** | 1.000 |
| Standard PPR ($\beta=0.85$) | 50.0% | [44.8%, 55.2%] | 40.4% | 90.8% | 1.000 |
| Degree-Normalized PPR | 49.4% | [44.2%, 54.7%] | 39.0% | **99.5%** | 1.000 |
| HippoRAG Prior Scaling | 49.4% | [44.2%, 54.7%] | 39.2% | 90.0% | 1.000 |

**Hypotheses Outcomes**:
- **P1 (Perron vs. Standard PPR)**: Difference $+0.3$ pp (173 vs 172 hits, 95% CI $[-2.9, +3.5]$, Holm $p = 1.000$). Standard PPR suffers 9.2% hub contamination, whereas Perron achieves 99.5% HSI.
- **P2 (Parity to BM25)**: Difference $-0.6$ pp (95% CI $[-3.5, +2.3]$ pp); non-inferiority against $-2.0$ pp is inconclusive.
- **P3 (Fusion)**: RRF fusion (174 hits vs 175 BM25, $p=0.845$) confirms disjoint hits reflect boundary rank permutations rather than orthogonal discovery.

---

## 5.4 Pre-Specified Evaluation of Graph Proposals and Pointwise Reranking on SWE-bench Verified

To evaluate whether local models exploit graph proposals under strict context budgets, I executed a pre-specified, hash-committed protocol on SWE-bench Verified ($N=344$ callable tasks). Candidate pools $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean $24.5$) were scored pointwise via frozen `gemma-4-e2b-it` single-forward log-odds ($s(t, v) = \text{logsumexp}(z[Y]) - \text{logsumexp}(z[N])$) under prompt P0 with thinking disabled.

| Arm | Description | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| B0 | BM25 Baseline | 25.3% | 38.7% | 44.8% | 51.2% | **0.344** |
| B1 | Perron Baseline | 22.7% | 35.2% | 40.4% | 50.3% | 0.316 |
| A1 | Rerank Seeds $S$ ($M=10$) | 24.7% | 39.5% | 46.2% | 51.2% | 0.335 |
| A2 | Rerank Depth BM25 | 23.8% | 38.1% | 45.1% | 51.2% | 0.330 |
| A3 | Rerank Same-File | 24.1% | 39.2% | 45.6% | 52.0% | 0.338 |
| **A4** | **Rerank Graph 1-Hop** | **24.7%** | **39.2%** | **47.1%** | **52.3%** | 0.341 |
| A5 | Rerank Null Graph | 24.1% | 39.0% | 45.9% | 51.2% | 0.332 |
| A6 | Rerank Perron Diffusion | 23.6% | 38.7% | 45.6% | 52.6% | 0.330 |

- **Confirmatory Contrasts & Realized Power**: Graph reranking (A4) achieves 47.1% Acc@5 vs. 45.1% for depth BM25 (A2, $\Delta = +2.03$ pp, 95% CI $[-0.29, +4.36]$, $p_{\text{holm}} = 0.287$) and 45.6% for same-file (A3, $\Delta = +1.45$ pp, 95% CI $[-0.58, +3.78]$, $p_{\text{holm}} = 0.302$), classified as `NULL_COMPATIBLE` within $[-5.0, +5.0]$ pp bounds (verdict: `NOT_DEMONSTRATED`). Under observed discordance ($n_d = 15\text{--}17$ pairs, 4.4%--4.9%), realized MDD is $3.4\text{--}3.7$ pp; while the CI upper bound is $+4.36$ pp, observed gains fall below detection resolution.
- **Seed Dominance & Rank Order**: Comparing graph 1-hop (A4) to seed reranking (A1, 46.2% Acc@5) yields net 3 tasks ($\Delta = +0.87$ pp, $p=0.549$), costing 4,988 extra forward passes. Un-reranked BM25 (B0) retains higher MRR (0.344 vs 0.341) and top-1 accuracy (25.3% vs 24.7%) than A4. Controls (A2, A3) dilute pools with distractors, lowering Acc@5 below A1; A4 appears superior not from topological discovery, but from less distractor dilution.

| Arm ($N=100$) | Candidate Context (Top-5) | Hit@5 | Loc@1 | Loc@3 | Conversion |
| :--- | :--- | :---: | :---: | :---: | :---: |
| D0 | BM25 Top-5 | 35.0% | 33.0% | 35.0% | 0.943 |
| D1 | A4 Graph Top-5 | 37.0% | 31.0% | 37.0% | 0.838 |
| D2 | A3 Control Top-5 | 38.0% | 30.0% | 37.0% | 0.789 |
| D3 | Oracle Context (Transmission) | 100.0% | **74.0%** | **85.0%** | 0.740 |

- **Reader Transmission & Distractor Interference**: On BM25 contexts (D0), Hit@5 is 35.0% and Loc@1 is 33.0% (conversion 0.943), matching a baseline selecting candidate 1 (33.0%). On Graph contexts (D1), Hit@5 rises to 37.0%, but Loc@1 drops to 31.0% (conversion 0.838), showing structural neighbors confuse the 2B reader. On oracle context (D3), Gemma 4 achieves 74.0% top-1 accuracy (67.5% chance-adjusted, 95% Wilson CI [64.6, 81.6]%), confirming that retrieval candidate recall (35%--38%), rather than reader discrimination, forms the primary bottleneck.

### 5.5 Ablation Study
Removing Decoupled Specificity ($\gamma=0$) drops HSI from 99.5% to 90.8% and collapses diffuse linear priors by -24.1 pp. Test-suite sanitization prevents an additional 3.8 pp sink leakage under unconstrained random walks.

![Figure 5](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/benchmark_statistical_evaluation.png)

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released across six deliverables under Apache 2.0 / CC0-1.0:
1. **`perron-core` v0.3.0**: Python library on PyPI with zero-copy CSR diffusion and CLI (`perron index`, `perron query`).
2. **`perron-mcp`**: Stdio MCP server with `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs`**: Precomputed CSR graph topologies for 12 repositories via `scripts/reconstruct_dataset.py`.
4. **`swebench-strata-annotations`**: 800 per-instance annotations across SWE-bench Lite and Verified on Kaggle.
5. **`gemma4-verified-reranking-scores`**: 19,298 scored candidate tuples (`scores.parquet`) and per-arm rankings for all 344 Verified tasks under Apache 2.0.
6. **Gemma 4 Workstation Kernel**: Interactive GPU notebook on Kaggle.

---

## 7. Discussion & Limitations

PPR fails on code graphs from test node absorption and prior dispersion. Decoupled Specificity stabilizes diffuse priors (+24.1 pp) and suppresses hubs (99.5% vs. 90.8% HSI). Four limitations remain: (1) static call resolution conflates ambiguous names; (2) edge weights are fixed; (3) static graphs miss runtime reflection; and (4) while reader transmission demonstrates 74.0% top-1 conversion on oracle contexts ($N=100$), full generative patch synthesis was not evaluated, as small-model reflection loops require decoupled extraction budgets.

---

## 8. Conclusion

On software code graphs, Personalized PageRank collapse under default damping ($\beta=0.85$) is driven by teleport prior dispersion and test-node sink absorption. Softmax prior concentration ($\tau=0.15$) recovers PPR to 48.5%, while Perron's Decoupled Specificity provides an adaptive regularizer (+24.1 pp under diffuse linear) and enforces 99.5% hub suppression. On SWE-bench Verified ($N=344$), standalone Perron matches BM25 at top-10 (50.3% vs 50.9%, non-inferiority inconclusive against $-2.0$ pp). Pointwise reranking under Gemma 4 demonstrates that graph 1-hop expansion does not outperform lexical seed reranking (+0.87 pp, $p=0.5488$), with un-reranked BM25 preserving superior rank order (MRR 0.344 vs 0.341), proving that retrieval candidate recall, rather than local reader discrimination, forms the decisive agentic bottleneck. Open code, graphs, and reproduction scripts are released at <https://github.com/yvliet/perron> and <https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation>.

---

## References

1. Brauer. *Duke Math. J.*, 1952.
2. Chen et al. *ACL*, 2025.
3. Gauthier. Aider, 2023.
4. Gutierrez et al. *NeurIPS*, 2024.
5. Haveliwala & Kamvar. *Stanford*, 2003.
6. Jimenez et al. *ICLR*, 2024.
7. Liu et al. *NAACL*, 2025.
8. Ouyang et al. *ICLR*, 2025.
9. Page et al. *Stanford*, 1999.
10. Pearson et al. *ICSE*, 2017.
11. Zou et al. *IEEE TSE*, 2021.
12. Liben-Nowell & Kleinberg. *JASIST*, 2007.
13. Andersen et al. *FOCS*, 2006.
14. Klicpera et al. *ICLR*, 2019.
15. Bojchevski et al. *KDD*, 2020.
