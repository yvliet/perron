# When and Why Does Personalized PageRank Fail on Code Graphs? Analyzing Seed Dispersion, Prior Calibration, and Decoupled Specificity Damping

**Author**: Sultan Haikal (GitHub: [@yvliet](https://github.com/yvliet))  
**Repository**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron)  
**License**: Apache 2.0  

---

## Abstract

Local developer agents localize defect targets under strict token budgets ($K \le 4{,}096$). Across a $2\times 2$ factorial evaluation on SWE-bench Lite ($N=237$) and an SpMM sweep of 63,984 configurations on SWE-bench Verified ($N=344$), I show that random-walk retrieval collapses onto a continuous dose-response curve governed by prior effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($p < .001$, GLMM). Diffuse linear priors collapse Personalized PageRank (PPR) to 10.1% Function Acc@10 with 32.5% test contamination. Softmax prior concentration ($\tau = 0.15$) recovers PPR to 48.5%, and test sanitization adds +3.8 pp. Intermediate structural hubs alter candidate order without degrading defect localization (equivalence within $\pm 9.3$ pp on hub tasks, $p < .001$). **Perron** decouples stationary specificity over static CSR matrices, stabilizing diffuse priors (+24.1 pp) and restoring hub suppression to 99.5% (matching BM25's 99.4%). On SWE-bench Verified, Perron attains 50.3% (173/344) Function Acc@10 vs. BM25's 50.9% (non-inferiority inconclusive against a $-2.0$ pp margin), where disjoint hits reflect top-10 boundary permutations rather than orthogonal discovery. In candidate prompt packing for Google Gemma 4, AST breadcrumbs occupy $370\text{--}425$ SentencePiece BPE tokens (283.8 whitespace words), reserving 88.7% of reasoning capacity (worst-case $\ge 87.8\%$). I release `perron-core` v0.3.0, MCP tooling, and 12-repository CSR graphs under Apache 2.0.

---

## 1. Introduction & Systems Constraints

Local developer agents operate under prompt budgets ($K \le 4{,}096$ tokens), where defect localization is token-constrained.

Lexical search locates symbols named in issue text, while graph retrieval aims to reach unmentioned symbols along static dependency edges. On real code graphs, diffusion fails via:
1. **Prior Dispersion Across Lexical Leaves**: Diffuse linear priors inject fractional mass across thousands of peripheral symbols, dissipating random walks.
2. **Test-Node Sink Absorption**: Dense test subgraphs siphon probability mass into test helpers and mock fixtures.

I diagnose these failure modes via a $2\times 2$ factorial sweep, characterize spectral convergence via Brauer bounds, establish the $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ dose-response law across continuous prior surfaces, and evaluate Decoupled Specificity on SWE-bench Verified.

### Systems Hardware Envelope
The standalone engine peaks at 144.33 MB resident memory (graph load 91.4 MB) on a 15.68 GiB host, completing sparse CSR diffusion in 1.89 ms on Requests and 5.10 ms on SymPy via OS page cache memory-mapping. Reader ingestion was evaluated under a controlled contrast protocol on Kaggle GPU across $N=100$ SWE-bench Verified tasks using Gemma 4 E2B in unquantized FP16 with native PLE on a Tesla T4 (2.37 GiB peak VRAM, 3548.1 ms mean TTFT, 11.6 tokens/sec decode throughput). Under a 256-token budget, `<|think|>` reflection exhausted generation before emitting closing `<target>` tags across all conditions (transmission kappa 0.0), showing reader transmission requires dedicated generation allocation.

---

## 2. Related Work

**Code Graph Navigation**: Aider (Gauthier, 2023) uses PageRank repo maps over Tree-sitter tags; however, unweighted walks amplify hubs. Multi-turn graph agents like RepoGraph (Ouyang et al., 2025) and LocAgent (Chen et al., 2025) rely on iterative LLM reasoning. Perron executes closed-form specificity diffusion over zero-copy CSR matrices in a single vector pass.

**Hub Suppression**: HippoRAG (Gutierrez et al., 2024) scales pre-walk priors by Inverse Passage Frequency ($s_i = |P_i|^{-1}$), but walks still funnel into sinks. Degree-normalized PPR discounts by in-degree, ignoring reachability. Perron decouples diffusion from stationary scaling via Pointwise Mutual Information.

**Fault Localization**: Scalable diffusion builds on push-based approximate PPR (Andersen et al., 2006), APPNP (Klicpera et al., 2019), PPRGo (Bojchevski et al., 2020), and stationary-normalized rooted PageRank (Liben-Nowell & Kleinberg, 2007). In fault localization, Pearson et al. (2017) and Zou et al. (2021) showed synthetic benchmarks fail to transfer; my evaluation uses authentic human patches from SWE-bench Lite and Verified.

---

## 3. Mathematical Methodology

### 3.1 Multiplex Transition Operator
A codebase is modeled as a directed multigraph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with AST symbol nodes and multiplex relations $r \in \{\text{call}, \text{inherit}, \text{import}, \text{caller}\}$:

$$W_{uv} = 0.50 A^{(\text{call})}_{uv} + 0.25 A^{(\text{inherit})}_{uv} + 0.15 A^{(\text{import})}_{uv} + 0.10 A^{(\text{caller})}_{uv}$$

Matrix $T$ normalizes out-degree $D_u = \sum_v W_{uv}$ ($T_{uv} = W_{uv}/D_u$); dangling indicator $d_u = 1.0$ when $D_u = 0$ else $0.0$.

### 3.2 Teleportation Prior Vector ($p_0$)
Issue prompts are tokenized against identifiers, paths, and docstrings using Okapi BM25 ($k_1 = 1.5, b = 0.75$). Candidate scores undergo shift-invariant softmax ($\tau = 0.15$) over top $K=25$ candidates: for $v \in \mathcal{K}$, $p_0(v) \propto \exp((\tilde{s}_v - \max_u \tilde{s}_u)/\tau)$, and $0$ otherwise.

### 3.3 Spectral Convergence and Brauer Rank-1 Bounds
Personalized PageRank computes stationary vector $\boldsymbol{\pi}_q$:

$$\boldsymbol{\pi}^{(t+1)} = \beta (\boldsymbol{\pi}^{(t)} T) + (\beta (\boldsymbol{\pi}^{(t)} \mathbf{d}) + 1 - \beta) \mathbf{p}_0$$

**Proposition 1 (Spectral Gap & Convergence)**:  
*Under Brauer's theorem (Brauer, 1952), Google operator $M$ satisfies |\lambda_2(M)| \le \beta = 0.85, establishing spectral gap $1 - |\lambda_2| \ge 0.15$ (with equality for disconnected subgraphs).* Geometric contraction bounds $\epsilon = 10^{-6}$ convergence in $\lceil \ln(2/\epsilon) / \ln(1/\beta) \rceil = 90$ iterations (Requests: 90 iterations, 1.89 ms; SymPy: 66 iterations, 5.10 ms).

### 3.4 Decoupled Post-Walk Specificity Ratio
To suppress hubs without mutating sparse transition matrices, Perron precomputes global stationary distribution $\boldsymbol{\pi}_g$ under uniform teleportation, adapting stationary-normalized rooted PageRank (Liben-Nowell & Kleinberg, 2007) with an exponent $\gamma_q$:

$$\text{Specificity}(v) = \frac{\pi_q(v)}{(\pi_g(v) + \epsilon)^{\gamma_q}}$$

where $\epsilon = 10^{-8}$. With uniform query prior $P(Q=q) = 1/|\mathcal{V}|$, stationary $\ln(\pi_q(v) / \pi_g(v)) = \text{PMI}(v; q)$, yielding interpolation $(1-\gamma)\ln \pi_q(v) + \gamma \text{PMI}(v; q)$. Hub suppression is measured by $\text{HSI}(q) = 1 - (|R_{10}(q) \cap \mathcal{H}_{25}| / 10)$.

**Scale-Invariant Hub Prominence ($\eta_q$)**: When an issue explicitly names a central hub (`Session` in Requests, `Expr` in SymPy), prominence $\eta_q = \max_{h \in \mathcal{H}_{25}} p_0(h) / \max_v p_0(v)$ relaxes the exponent: $\gamma_q = \gamma \cdot (1 - \frac{1}{2} \mathbb{I}(\eta_q \ge 0.20))$, rescuing intentional targets (Figure 2, dev_val $N=100$).

![Figure 2](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/specificity_gamma_sweep.png)

---

## 4. Systems Architecture & Memory Virtualization

![Figure 1](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/perron_pipeline.png)

### 4.1 Shared Read-Only Graph Storage
The transition matrix is stored as raw CSR arrays opened via `numpy.load(mmap_mode='r')`. Specificity output rescaling allows parallel workers to share physical memory via OS page caching. Dynamically rewriting edge weights per query costs 319.0 ms on a 1M-node synthetic graph (17.5% of diffusion time).

### 4.2 AST Breadcrumbs & Transactional Patching
- **AST Breadcrumbs**: Perron wraps candidate functions in enclosing class skeletons and docstrings ($<100$ tokens per symbol).
- **Transactional Patching**: Edits apply bottom-to-top by start line with mandatory `ast.parse()` validation and atomic rollback on syntax failures (`os.replace`).

![Figure 3](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/comparative_edit_interfaces.png)

---

## 5. Empirical Evaluation

### 5.1 Continuous Dose-Response Surface and Resolvent Linearity (H1, H2 & H3)
Exploiting resolvent linearity ($\mathbf{u}(p_0) = \sum_{i=1}^K w_i \mathbf{u}(e_{s_i})$), I precomputed basis vectors for top-100 seeds on SWE-bench Verified ($N=344$ callable), evaluating 186 continuous configurations across four prior families (softmax, power, linear, rank-zeta, 63,984 evaluations in 26 min via batched SpMM).

![Figure 5](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/neff_dose_response.png)

As shown in Figure 5, retrieval accuracy collapses onto an invariant dose-response curve governed by prior effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ across $n_{\text{eff}} \in [4, 8]$. A binomial GLMM confirms vanishing cutoff effects and modest temperature sensitivity:

$$\text{logit}(P(Y_{ij}=1)) = \beta_0 + \beta_1 \ln n_{\text{eff}} + \beta_2 (\ln n_{\text{eff}})^2 + a \ln k + b \ln \tau + \gamma_j$$

with $\beta_1 = 0.325$, $\beta_2 = -0.075$, $a = -0.008$, and $b = -0.036$. The analytical GLMM vertex sits at $\ln(n_{\text{eff}}^*) = -\beta_1 / (2\beta_2) \approx 2.176 \implies n_{\text{eff}}^* \approx 8.81$ ($50.9\%$ hit rate). The calibrated library default ($k=25, \tau=0.15 \implies n_{\text{eff}} \approx 6.035$, $49.4\%$ hit rate) sits on the conservative side of the $[4, 8]$ plateau to minimize sink leakage. Task-level cross-validation yields pseudo-$R^2 = 0.197$, confirming **Hypothesis H1 (Support Law)**. 

**Hypothesis H2 (Iso-Entropy Contrast)**: Comparing power-law and softmax priors matched on $n_{\text{eff}} \approx 6.0$ ($k=10$ vs. $k=50$) shows no significant difference (p = 0.535, McNemar, 31 discordant tasks).

**Hypothesis H3 (Hub Impact and Equivalence)**: On tasks where hubs ($\mathcal{H}_{25}$) enter top-10 slots under Standard PPR ($N=109$), paired McNemar (p = 0.683) and Two One-Sided Tests demonstrate equivalence within a $\pm 9.3$ pp margin (p < .001); across all tasks, equivalence holds within $\pm 5.0$ pp (p < .001). Graph diffusion achieves zero positive ranking lift over lexical seeds (seeds hit 50.9\%, walks reach 50.3\%). Hubs alter candidate ranking without displacing gold targets. Decoupled Specificity acts as an adaptive prior regularizer (+24.1 pp under diffuse linear priors) and an aesthetic hub filter (99.5% HSI), rather than an accuracy fix for structural hub bias.

---

## 5.2 Topological Reachability and Relation-Stratified Null Models
I analyzed Boolean reachability from lexical seeds on SWE-bench Verified ($N=344$ callable tasks). Lexical top-10 seeds hit 50.9% of gold targets. Graph expansion covers 57.8% at 1 hop (median $|N_1|=27$) and 66.9% at 2 hops (median $|N_2|=324$), adding at most 7.0 to 16.0 pp while expanding candidate sets to 27 to 324 nodes. Pruning hubs drops 2-hop reachability to 62.2% (214 of 344); lexical expansion reaches 54.1% (186 of 344).

Degree-preserving Maslov-Sneppen null models (100 replicates, $E_{\text{caller}} = E_{\text{call}}^T$) yield 63.8% 2-hop reachability ($z_E = +6.46$, p < .001, paired difference 95% bootstrap CI $[-0.42, +6.41]$ pp).

Same-file containment captures 66.9%, matching 2-hop graph reachability via symmetric discordance: methods attain 90.1 percent concordance (310 concordant tasks of 344), where 17 cross-file graph hits outside seed files balance 17 intra-file targets lacking static AST call edges.

---

## 5.3 Pre-Registered Confirmatory Evaluation on SWE-bench Verified
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
- **P1 (Perron vs. Standard PPR)**: Difference $+0.3$ pp (173 vs 172 hits, paired bootstrap CI $[-2.9, +3.5]$, Holm $p = 1.000$). Standard PPR suffers 9.2% hub contamination (HSI@10 = 90.8%), whereas Perron sustains 99.5% hub suppression.
- **P2 (Parity to BM25)**: Difference $-0.6$ pp (paired bootstrap CI $[-3.5, +2.3]$ pp). Crossing $-2.0$ pp leaves non-inferiority **inconclusive**. Point estimates show parity; BM25 leads at top-5 (45.1% vs 40.4% Fn Acc@5, 59.5% vs 53.8% File Acc@5).
- **P3 (Boundary Jitter & Fusion)**: RRF fusion (174 hits vs 175 for BM25) confirms slot competition: 161 joint hits, 157 misses, 14 exclusively by BM25, and 12 exclusively by Perron (p = 0.845, McNemar). In Stratum C ($N=131$), BM25 hits 28 to Perron's 27 (p = 1.000). While oracle union reaches 187/344 (54.4%), unrouted fusion fails to beat BM25, confirming disjoint hits reflect boundary rank permutations.

---

## 5.4 AST Context-Budgeted Breadcrumb Packing for Local Developer Agents
I evaluated candidate context packing under prompt budget constraints ($K \le 3{,}480$ tokens) across all 344 SWE-bench Verified callable instances, packaging AST breadcrumb skeletons without raw function bodies.

| Context Packing Arm | Packed Slots | Candidate Occupancy | Gold Function Inclusion | Gold File Inclusion |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_pack` | Top-10 BM25 | 370–425 BPE (283.8 w) | 51.5% (177/344) | 72.4% (249/344) |
| `perron_pack` | Top-10 Perron | 370–425 BPE (283.8 w) | 51.2% (176/344) | 67.2% (231/344) |
| `slot_fused_pack` | 8 BM25 + 2 Perron | 370–425 BPE (283.8 w) | **52.3%** (180/344) | **72.7%** (250/344) |

Across arms, prompts contain an identical mean of 283.8 whitespace words formatting 10 candidate slots under the same template ($370\text{--}425$ Gemma SentencePiece BPE tokens, mean 394 tokens), consuming 11.3 percent of prompt budget and leaving 88.7% open for reasoning (worst-case $\ge 87.8\%$). Inclusion credits enclosing class scopes (yielding 177 and 176 hits vs. 175 and 173 in Table 2). Perron matches BM25 in function inclusion (51.2% vs. 51.5%), while BM25 preserves higher file grounding (72.4% vs. 67.2%). Slot fusion balances coverage (52.3% function, 72.7% file) at 513.96 ms pipeline latency (diffusion takes 1.89 ms on Requests and 5.10 ms on SymPy; AST parsing accounts for the remainder).

### 5.5 Ablation Study
Removing Decoupled Specificity ($\gamma=0$) drops HSI from 99.5% to 90.8% and collapses diffuse linear priors by -24.1 pp. Test-suite sanitization prevents an additional 3.8 pp sink leakage under unconstrained random walks.

![Figure 4](https://raw.githubusercontent.com/yvliet/perron/main/paper/figures/benchmark_statistical_evaluation.png)

---

## 6. Open-Source Ecosystem & Systems Deliverables

Perron is released across five reproducible deliverables under Apache 2.0 / CC0-1.0:
1. **`perron-core` v0.3.0**: Python library on [PyPI](https://pypi.org/project/perron-core/0.3.0/) with zero-copy CSR diffusion and CLI (`perron index`, `perron query`).
2. **`perron-mcp`**: Stdio MCP server exposing `retrieve_context`, `inspect_symbol_breadcrumbs`, and `build_code_graph`.
3. **`swebench-lite-codegraphs`**: Precomputed CSR graph topologies for 12 repositories via `scripts/reconstruct_dataset.py`.
4. **`swebench-strata-annotations`**: 800 per-instance annotations across SWE-bench Lite and Verified (707 unique instances across 93 overlap) on [Kaggle](https://www.kaggle.com/datasets/yvliet/swebench-strata-annotations).
5. **Gemma 4 Workstation Kernel**: Interactive GPU notebook on [Kaggle](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation).

---

## 7. Discussion & Limitations

PPR fails on code graphs from test node absorption and prior dispersion. Decoupled Specificity stabilizes diffuse priors (+24.1 pp) and suppresses hubs (99.5% vs. 90.8% HSI). Four limitations remain: (1) static call resolution conflates ambiguous names; (2) edge weights are fixed; (3) static graphs miss runtime reflection; and (4) while reader transmission was benchmarked on Gemma 4 E2B ($N=100$), generative patch synthesis was not evaluated, as small-model reflection loops require higher generation budgets than the 256-token transmission envelope.

---

## 8. Conclusion

On software code graphs, Personalized PageRank collapse under default damping ($\beta=0.85$) is driven by teleport prior dispersion and test-node sink absorption. Concentrating priors via top-25 softmax recovers Standard PPR to 48.5%, matching Perron (47.7% on Lite). Perron acts as a topological regularizer under diffuse priors (+24.1 pp) and restores hub suppression to 99.5%. On SWE-bench Verified (344 callable tasks), Perron shows no statistically detectable difference vs. BM25 (non-inferiority inconclusive against $-2.0$ pp). While oracle union reaches 54.4% (187/344), unrouted rank fusion fails to exceed BM25, confirming disjoint hits reflect boundary rank permutations. Open code, graphs, and reproduction scripts are released at <https://github.com/yvliet/perron> and <https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation>.

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
12. Liben-Nowell, D. & Kleinberg, J. The Link-Prediction Problem. *JASIST*, 2007.
