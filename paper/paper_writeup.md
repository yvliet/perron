# Unlocking Code Graphs for Small Language Models: Provenance-Guided Reasoning and Spectral Diffusion with Gemma 4
**Diagnostic study, theoretical bounds, and turnkey CSR diffusion on SWE-bench Verified**

**Author**: Sultan Haikal (recrified@gmail.com · [@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

*Official Writeup (≤ 3,000 words). Target Tracks: Overall Best Paper & Best New Resource. Repo: https://github.com/yvliet/perron.*

---

## Abstract

Deploying structural code graphs in local developer agents requires balancing topological expansion against language model discrimination. Across 344 held-out SWE-bench Verified tasks, naive Personalized PageRank yields indistinguishable localization from lexical retrieval (47.1% vs 45.1%, exact binomial McNemar $p=0.287$; excluding true advantages $\ge +4.4$ pp, $p < 0.05$). This limitation is characterized by an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$), where Gemma 4 E2B pairwise discrimination degrades from $\text{AUC}=0.7966$ on seeds to $\text{AUC}=0.6596$ on neighbors, converting expanded pools into distractors. Consistent with relational reasoning limits in transformers (Sanford et al., 2024), an exploratory $2 \times 2$ factorial evaluation on all 131 un-named Stratum C defects reveals test-time reasoning traces (`<thought>`) quadruple Top-1 accuracy from 2.29% to 9.92% (exact binomial $p = 0.0063^*$, discordant pairs $n_{10}=11, n_{01}=1$) and increase MRR@50 by +74.6% (0.0778 to 0.1358). Factorial comparison against combined provenance prompting (9.16%) confirms empirical parity ($p = 1.0000$, $\Delta_{\text{interaction}} = -2.29$ pp), demonstrating that intermediate reasoning tokens allow model representations to trace relational dependencies directly. **Perron** provides an open-source CSR diffusion engine executing vectorized spectral convergence in 1.89--5.10 ms within $\le 3$ MB zero-copy RAM ($144.33$ MB peak process RSS), preserving 91.9% headroom. Downstream evaluation demonstrates that AST surgical replacement achieves 0.0% syntax errors ($p < .001$ vs unified diffs), resolving patch generation fragility independent of retrieval depth.

---

## 1. Introduction

Local developer agents must localize defects inside tight token budgets ($K \le 4,096$) without cloud APIs. Across 344 held-out SWE-bench Verified tasks [10], naive graph expansion yields statistically indistinguishable top-5 accuracy from depth-matched BM25 (47.1% vs 45.1%, Holm $p=0.287$). This failure reflects an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$), where models suffer a **Lexical Abstraction Bottleneck** (discrimination drops from 0.7966 on seeds to 0.6596 on neighbors). I resolve this via three contributions: (1) **factorial test-time reasoning** showing `<thought>` traces quadruple Top-1 accuracy on un-named defects ($2.29\% \to 9.92\%$, exact $p = 0.0063^*$, discordant $n_{10}=11, n_{01}=1$, MRR $+74.6\%$), with parity under combined provenance ($9.16\%$, $p = 1.0000$, $\Delta_{\text{interaction}} = -2.29$ pp); (2) **Perron**, a CSR engine executing vectorized spectral convergence in 1.89--5.10 ms within $\le 3$ MB zero-copy RAM ($144.33$ MB process RSS); and (3) a **benchmark toolkit** releasing 19,298 evaluations (`scores.parquet`) and 800 strata annotations. **Scope**: Focuses on retrieval and context packing ($K \le 4,096$); localization bounds downstream patch resolution [5]. PC1/PC2 ($N=344$) are confirmatory; Stratum C factorial evaluation ($N=131$) is exploratory; AST editing ($N=30$) is a syntax safety utility (0.0% syntax errors vs 50.0% for unified diffs, $p < .001$).

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo [13] and APPNP [7] decouple PageRank from neural prediction; HippoRAG [8] applies PPR to text knowledge graphs. Perron extends this decoupled lineage to static code multigraphs in zero-copy CSR.
- **Graph Complexity**: Fatemi et al. [11] showed verbalization shifts LLM reasoning by 61.8%; Sanford et al. [12] proved connectivity requires $\mathsf{LogDepthPause}$ intermediate tokens. RepoGraph [2] noted 2-hop flattening degrades resolution.
- **Agent Pipelines**: SWE-agent [14] and Agentless [5] show hierarchical localization matches complex agent loops; LocAgent [4] fine-tunes graph-traversal agents. Perron evaluates zero-shot graph expansion under compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [10] across 12 repos: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), Stratum C (Un-Named, $N=131$, 38.1%). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes call, inherit, import, caller edges in zero-copy CSR ($<1.2$ ms attach, $1.89\text{--}5.10$ ms solve, $<3$ MB delta, $144.3$ MB process RSS). Breadcrumbs average 283.8 tokens (median 264), preserving 91.9% headroom. Specificity damping enforces $\text{HSI@10} = 99.5\%$.
- **Pools & Arms**: Seeds $S$ are top-10 BM25 symbols; filter $F$ prunes test nodes and top-25 hubs. Cap $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5). Arms: **B0** (unscored BM25), **A1** (seeds $M=10$), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion). Pruning yields $+2$ tasks over BFS ($201$ vs $199$).
- **Scoring & Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds: $s(t, v) = \operatorname{logsumexp}(z[Y]) - \operatorname{logsumexp}(z[N])$ (19,298 passes). Dual T4s validate listwise `gemma-4-e4b-it`. Confirmatory family: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).
- **Reader & Editor**: $N=100$ tasks compare BM25 (D0), graph (D1), same-file (D2), and oracle (D3). $N=30$ tasks benchmark AST surgical editing against unified diffs.

---

## 4. Empirical Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5; Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, $\Delta = -0.6$ pp, Holm $p=1.000$) and 40.4% Acc@5; standard PPR hits 50.0% (172/344). Tag maps (Aider) reach 40.0% File Acc@5 (0.0% Fn Acc@10); dense bi-encoders reach 38.4% Fn Acc@10. On Stratum C (131 tasks), BM25 hits 28 and Perron 27.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. 1-hop covers 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment (66.9%). 2-hop reachability exceeds null graphs (63.8%) by +3.02 pp (CI [−0.42, +6.41]).

### 4.3 Pointwise Candidate Reranking: Confirmatory Matched-Control Family (N=344)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 51.2% | 0.3315 |
| **A6** | Perron diffusion pool | 23.6% | 45.6% | 52.6% | 0.3301 |

- **Confirmatory (Acc@5) & Power Bounds**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). With 17 discordant tasks ($p_{\text{disc}} = 4.94\%$), MDE at 80% power is 3.27 pp ($N = 942$ required for $+2.03$ pp). Under TOST, true population advantages $\ge +4.4$ pp are formally excluded ($p < 0.05$).
- **Exploratory**: $A_4 - A_1$: $+0.87$ pp (CI [−1.92, +3.66], discordant 7/4, $p=0.549$). Null control $A_4 - A_5$: $+1.16$ pp ($p=0.344$). Diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). BM25 (**B0**) maintains higher MRR@50 (0.344 vs 0.341).
- **Model Sensitivity & Lexical Bottleneck**: BM25 pool ROC-AUC is 0.8245. Gemma 4 achieves 0.7796 ROC-AUC, dropping from 0.7966 on seeds to 0.6596 on graph targets (0.137 gap). RRF ($k_{\text{rrf}}=60$) reaches 46.51% (160/344).

### 4.4 Diagnostic Investigations & Exploratory Provenance Reasoning (N=131)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Break-Even Discrimination Bound**: In two-stage retrieval, expected outranking distractors is $\mathbb{E}[X \mid s^*] = (M - 1)(1 - \text{AUC})$. Landing in top-$k$ requires $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $\text{AUC} \ge 0.830$. Because Gemma 4 achieves only $\text{AUC} = 0.6596$ on graph targets, expected distractors is $8.00$, burying targets at rank $9.00$ ($17.68$ at $M=50$).
- **Stratum Asymmetry (25 Graph-Only Targets)**: In Stratum A, 11 reached targets yield 6 top-5 conversions (54.5%, net $+5$ tasks, discordant 7/2, $p=0.1797$). In Stratum C, 14 reached targets yield 0 conversions (0.0%, net $-2$ tasks, discordant 0/2, $p=0.5000$), cancelling to $+3$ tasks aggregate (+0.87 pp, $p=0.549$).

| Factorial Condition (N=131) | Acc@1 | Acc@5 | Hits | MRR | McNemar p (Acc@1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| C1: Baseline Isolated (No Prov, No Think) | 2.29% | 15.27% | 20 | 0.0778 | reference |
| C2: Call-Path Provenance (Prov, No Think) | 3.82% | 16.03% | 21 | 0.0899 | p = 0.5000 |
| C0: Unguided Thinking (No Prov, Think) | **9.92%** | **18.32%** | **24** | **0.1358** | **p = 0.0063\*** |
| C3: Provenance + Thinking (Prov, Think) | 9.16% | 17.56% | 23 | 0.1270 | **p = 0.0117\*** |
| *Factorial Contrast: C3 vs C0* | *-0.76 pp* | *-0.76 pp* | *-1* | *-0.0088* | *p = 1.0000 (Parity)* |
| *Subgroup: Seed-Connected (N=28)* | *C1: 10.7%* | *C0: 42.9%* | *+9 hits* | *--* | *Local disambiguation* |
| *Subgroup: Graph-Only (N=7)* | *C1: 0.0%* | *C0: 14.3%* | *+1 hit* | *--* | *28.6% Acc@5 (Exploratory)* |

- **Factorial Reasoning & Candidate Grounding**: Isolated snippets (C1) yield 2.29% Top-1; static edge annotations alone (C2, 3.82%, $p=0.5000$) yield negligible change. Enabling `<thought>` traces (C0) quadruples Top-1 to **9.92% (13/131 hits, $+7.63$ pp, exact $p = 0.0063^*$, discordant $n_{10}=11, n_{01}=1$)** and increases MRR@50 by +74.6% to **0.1358** (~120 tokens, 10-15s on T4, <3s on desktop). Provenance plus thinking (C3, 9.16%, $p = 0.0117^*$) matches unguided thinking ($\Delta = -0.76$ pp, $p = 1.0000$, $\Delta_{\text{interaction}} = -2.29$ pp; 11 identical hits), showing thinking tokens allow representations to trace relational paths directly. Partitioning the 35 in-pool defects by seed proximity reveals where reflection takes effect: (1) seed-adjacent targets ($N=28$) improve from $10.7\% \to 42.9\%$ Top-1 (+32.2 pp) by resolving ambiguity among local callers; (2) distant graph targets ($N=7$) achieve exploratory recovery of 2/7 targets into Top-5 (28.6%, 1 Top-1 vs 0% in C1). Confirming distant target separation requires $N \ge 942$ tasks.
- **Selective Gating & Listwise Scaling**: Symbol density gating achieves 47.38% Acc@5 (163/344), while an oracle stratum router achieves 47.67% (+2.90 pp over BM25), matching the empirical two-arm oracle ceiling (48.55%). On Stratum C ($N=131$), listwise execution ($M \le 25$, E4B) raises depth BM25 to 16.79% (+8.40 pp, $p_{\text{holm}}=0.0102$ vs pointwise A2 8.40%); under identical listwise execution, graph expansion ties depth BM25 at 16.79% (22/131 hits, $p=1.000$), preserving the null result.
- **Four-Stage Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed ceiling: 51.2% (176) $\to$ (3) Graph ceiling: 58.4% (201, $+7.3$ pp) $\to$ (4) Top-5: 47.1% (162). Boundary loss (41.6 pp) exceeds compaction loss (11.3 pp) by $3.68\times$ at $k=5$.

### 4.5 Downstream Reader & Syntactic AST Patch Tooling (N=100 & N=30)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, predicting candidate 1 yields 33.0% EditLoc@1 from positional bias. In D3, gold is identified in 74.0% of tasks (CI [64.6, 81.6]%). Across 30 defect repairs, unified diff git apply suffered 50.0% syntax errors and 20.0% hunk rejections (30.0% valid). Perron's AST editor attained 0.0% syntax errors and 76.7% valid application (23.3% safe rollback, McNemar $p < .001$ vs git apply), verifying AST surgical replacement provides syntax safety against indentation errors and hunk drift.

### 4.6 Diffusion Diagnostics (Post-Hoc)
On Lite ($N=237$), PPR drops to 10.1% Acc@10 under diffuse priors; top-25 softmax prior ($\tau=0.15$) restores 48.5% (BM25: 46.0%). Specificity damping raises HSI from 90.8% to 99.5%, preventing hub absorption.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but differences are null-compatible with gains $> +4.4$ pp excluded (PC1).
- **Distractor Degradation vs Lexical Bottleneck**: Listwise prompting enables comparative ranking (+8.40 pp on Stratum C, $p=0.0034$), but listwise graph and depth expansion tie at 16.79% ($\Delta = 0.00$ pp), showing compact models evaluate relevance primarily via surface tokens.
- **Operating Regimes for Local Developer Agents**: The $3.68\times$ bottleneck ratio defines operational guidelines: (1) **First-Pass Static Filtering**: Zero-copy CSR diffusion grounds candidate symbols in $<5$ ms within $\le 3$ MB of RAM, capturing the 58.4% static reachability ceiling while preserving 91.9% context headroom; (2) **Test-Time Reflection**: Allocate $\sim 120$ `<thought>` tokens to discriminate local call-trees without prompt edge pollution; (3) **Dynamic Execution Fallback**: For the 41.6 pp boundary loss where defects involve dynamic dispatch, reflection, or implicit framework configuration, static graphs fail by construction. Agents must switch to runtime execution traces and test assertions rather than expanding static graph radius.
- **Post-Training & Feature Noise**: Post-training compact models on call-path provenance can eliminate test-time token overhead. Auxiliary heuristics exacerbate distractors: churn correlates weakly with defects ($\text{AUC} < 0.60$), while complexity tracks file size. Sparse AST topology with provenance avoids heuristic inflation.
- **Limitations**: (1) Python ASTs; (2) compact models (E2B/E4B); (3) 1-hop pools ($M \le 50$); (4) reader/editor sample size ($N=100$ and $N=30$); (5) frozen zero-shot scorers.

---

## 6. Community Resource Toolkit & Specification

Perron is released as an open-source resource ecosystem under Apache 2.0:
- **Binary CSR Multigraphs**: Precomputed multigraphs for 12 repos in zero-copy CSR ($<1.2$ ms attach, $1.89\text{--}5.10$ ms solve, $<3$ MB delta).
- **Evaluation Archive (`scores.parquet`)**: 19,298 scored evaluations across 344 tasks with log-odds under Apache 2.0.
- **Ground Truth & Strata**: Function targets, hub flags, and Strata A/B/C annotations across 800 tasks (300 Lite + 500 Verified, with 344 callable Verified tasks forming the primary benchmark; CC0-1.0).
- **Engine Performance Benchmark**:
| Engine | Warm Attach | PPR Solve | RSS (Graph / Total) | Throughput | HSI@10 | Integration |
| :--- | :---: | :---: | :---: | :---: | :---: | : --- |
| **NetworkX** | $1{,}420$ ms | $320.0$ ms | $132.5$ / $185.4$ MB | 3.1 qps | 90.8% | Graph object |
| **SciPy Sparse** | $85$ ms | $28.4$ ms | $91.4$ / $144.2$ MB | 35.2 qps | 90.8% | Sparse matrix |
| **Perron (Ours)** | **$<1.2$ ms** | **$1.89\text{--}5.10$ ms** | **$<3$ / $144.33$ MB** | **$196\text{--}529$ qps** | **$99.5\%$** | **3-Line / MCP** |

- **Turnkey API & MCP Server**: Drop-in `perron.diffuse_with_provenance()` Python interface and stdio JSON-RPC server (`perron-mcp`) for IDEs and agents.
- **Kaggle Workstation**: Reproduction notebook on Dual Tesla T4 GPUs (`yvliet/perron-gemma-4-workstation`), validating context packing, scoring, and causal evaluation.

---

## 7. Conclusion

On un-named Stratum C defects, test-time reasoning tokens quadruple Top-1 localization from 2.29% to 9.92% (exact binomial $p = 0.0063^*$, discordant pairs $n_{10}=11, n_{01}=1$, MRR $+74.6\%$), with empirical parity under topological prompt annotations ($p = 1.0000$). Paired with Perron's high-throughput CSR diffusion engine, these results establish that compact open models can navigate non-lexical code graphs when provided an allocated inference-time reasoning budget.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] H. Chen et al. *LocAgent.* arXiv:2503.09080, 2025.
- [5] C. S. Xia et al. *Agentless.* arXiv:2407.01489, 2024.
- [6] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [7] J. Klicpera et al. *Predict then Propagate.* ICLR 2019.
- [8] B. Gutierrez et al. *HippoRAG.* NeurIPS 2024.
- [9] S. Robertson et al. *Okapi at TREC-3.* 1994.
- [10] C. E. Jimenez et al. *SWE-bench Verified.* 2024.
- [11] B. Fatemi et al. *Talk like a Graph.* ICLR 2024.
- [12] C. Sanford et al. *Understanding Transformers via Graph Connectivity.* COLM 2024.
- [13] A. Bojchevski et al. *Scaling Graph Neural Networks with Approximate PageRank.* KDD 2020.
- [14] J. Yang et al. *SWE-agent.* arXiv:2405.15793, 2024.
- [15] R. Rafailov et al. *Direct Preference Optimization.* NeurIPS 2023.
- [16] E. Brochu et al. *A Tutorial on Bayesian Optimization.* arXiv:1012.2599, 2010.
- [17] T. Akiba et al. *Optuna.* KDD 2019.
