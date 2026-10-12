# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified
**Pre-registered evaluation, candidate pool controls, and turnkey CSR diffusion on SWE-bench Verified**

**Author**: Sultan Haikal (recrified@gmail.com · [@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

---

## Abstract

When does candidate expansion pay off for a small reranker, and can you predict it before running the experiment? Deploying structural code graphs in local developer agents requires balancing topological recall against language model discrimination. Across 344 held-out SWE-bench Verified tasks, naive Personalized PageRank yields indistinguishable localization from lexical retrieval alone (47.1% vs 45.1%, exact binomial McNemar $p=0.287$; excluding true advantages $\ge +4.4$ pp, $p < 0.05$). This limitation reflects an expected-rank **Break-Even Discrimination Bound** ($	ext{AUC}^* \ge 1 - rac{k-1}{M-1}$), where Gemma 4 E2B pairwise discrimination degrades from $	ext{AUC}=0.7966$ on lexical seeds to $	ext{AUC}=0.6596$ on structural neighbors, converting expanded pools into distractors. Pre-registered cross-architecture scaling (`PREREG-SCALING-v1`) on Kaggle Dual Tesla T4 GPUs confirms monotonic intra-family scaling in Gemma 4 ($0.8544 	o 0.9274$) and shows Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek-R1-Distill-7B (7.0B, 0.6489), while isolating a consumer hardware boundary on Gemma 4 26B-A4B ($44.0$ GB unquantized weights physically exceeding $29.12$ GB Dual T4 VRAM). Consistent with relational reasoning limits in transformers (Sanford et al., 2024), an exploratory $2 	imes 2$ factorial evaluation on all 131 un-named Stratum C defects reveals test-time reasoning traces (`<thought>`, $C_0$) raise Top-1 accuracy from 2.29% to 9.92% (13 hits; exact binomial $p = 0.0063^*$, discordant $n_{10}=11, n_{01}=1$, MRR $+74.6\%$), matching explicit provenance prompting ($C_3$, 9.16%, $p = 1.0000$). Pre-registered causal interventions (`PREREG-CAUSAL-v1`, $N=58$) show that stripping user-defined identifiers via AST $lpha$-renaming incurs only a $-1.73$ pp Top-1 drop ($12.07\% 	o 10.34\%$, exact $p=1.0000$), confirming that control-flow syntax provides a resilient structural baseline without surface names, while inline comment provenance doubles Top-1 accuracy to 25.86% (+13.79 pp lift, exact binomial McNemar $p = 0.0386^*$, Wilcoxon $p = 0.0103^*$), significantly outperforming disconnected global edge lists (15.52%) and adjacency maps (15.52%). **Perron** provides an open-source CSR diffusion engine converging in 1.89--5.10 ms within $\le 3$ MB zero-copy RAM ($144.33$ MB process RSS), preserving 91.9% context headroom.

---

## 1. Introduction

Local developer agents designed for offline execution must localize software defects inside a working context of a few thousand tokens ($K \le 4,096$), eliminating dependence on costly cloud APIs and preventing private source code leakage. Standard lexical search (e.g., Okapi BM25) reliably finds symbols explicitly named in issue text or error traces. To locate fault sites that the issue never mentions, existing agent architectures (including Aider repository maps, RepoGraph, and LocAgent) deploy random walks or graph traversal tools over AST dependency networks.

The working hypothesis underlying graph-augmented localization rests on two joint conditions: (1) **Recall Expansion**: Traversal along structural call edges must place gold defect targets into the candidate pool that lexical search misses; and (2) **Rank Discrimination**: A compact language model must rank those newly introduced targets above the distractors that graph expansion inevitably introduces.

Evaluating both assumptions on 344 held-out SWE-bench Verified tasks [10] with frozen `gemma-4-e2b-it` exposes a fundamental disconnect: while 1-hop graph expansion adds +7.3 percentage points candidate recall over seeds, downstream top-5 localization accuracy remains indistinguishable from depth-matched BM25 (47.1% vs 45.1%, Holm $p=0.287$).

This paper investigates this disconnect through five interconnected contributions:
1. **The Empirical Paradox**: Across 19,298 scored evaluations, naive Personalized PageRank expansion achieves zero statistical advantage over lexical retrieval under frozen small scorers, confirming candidate pool parity (RQ1, RQ2).
2. **The Predictive Account**: I formalize the expected-rank **Break-Even Discrimination Bound** ($	ext{AUC}^* \ge 1 - rac{k-1}{M-1}$) and identify the **Lexical Abstraction Bottleneck**: Gemma 4 discrimination collapses from 0.7966 ROC-AUC on seeds to 0.6596 on graph neighbors, falling below the expected-rank threshold required to benefit from pool expansion (RQ3).
3. **Cross-Architecture Scaling Surface**: Pre-registered protocol `PREREG-SCALING-v1` on Dual Tesla T4 GPUs demonstrates monotonic intra-family scaling within Gemma 4 ($0.8544 	o 0.9274$), proves Gemma 4 E4B achieves representational efficiency over larger controls (Phi-4-mini and DeepSeek-7B), and isolates the physical quantization boundary for sparse MoE architectures on consumer GPUs.
4. **Factorial Relational Reasoning**: On 131 un-named Stratum C defects, test-time reasoning traces (`<thought>`, $C_0$) raise Top-1 localization accuracy from 2.29% to 9.92% ($+7.63$ pp, exact binomial McNemar $p = 0.0063^*$, discordant pairs $n_{10}=11, n_{01}=1$; MRR $+74.6\%$). Contrast against combined provenance prompting ($C_3$, 9.16%) reveals statistical parity ($p = 1.0000$, $\Delta_{	ext{interaction}} = -2.29$ pp), showing test-time reflection allows model representations to trace relational paths directly.
5. **Causal Lexical Dissection & Graph Serialization**: Protocol `PREREG-CAUSAL-v1` ($N=58$) reveals that syntax-preserving AST $lpha$-renaming reduces Top-1 localization by only $-1.73$ pp ($p=1.0000$), while inline comment provenance achieves 25.86% Acc@1 (+13.79 pp lift, $p = 0.0386^*$, Wilcoxon $p = 0.0103^*$), more than doubling baseline localization and outperforming disconnected edge lists (15.52%).

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo [13] and APPNP [7] decouple PageRank from neural prediction; HippoRAG [8] applies PPR to text knowledge graphs. Perron extends this decoupled lineage to static code multigraphs in zero-copy CSR.
- **Graph Complexity & Reasoning**: Fatemi et al. [11] showed verbalization shifts LLM reasoning by 61.8%; Sanford et al. [12] (NeurIPS 2024) proved connectivity requires $\mathsf{LogDepthPause}$ intermediate tokens. RepoGraph [2] noted 2-hop flattening degrades resolution.
- **Agent Pipelines**: SWE-agent [14] and Agentless [5] show hierarchical localization matches complex agent loops; LocAgent [4] fine-tunes graph-traversal agents. Perron evaluates zero-shot graph expansion under compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [10] across 12 repos: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), Stratum C (Un-Named, $N=131$, 38.1%). Parameters ($eta=0.85, \gamma=0.70, 	au=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes call, inherit, import, caller edges in zero-copy CSR ($<1.2$ ms attach, $1.89	ext{--}5.10$ ms solve, $<3$ MB delta, $144.3$ MB process RSS). Breadcrumbs average 283.8 tokens, preserving 91.9% headroom. Damping enforces $	ext{HSI@10} = 99.5\%$.
- **Pools & Arms**: Seeds $S$ are top-10 BM25 symbols; filter $F$ prunes test nodes and top-25 hubs. Cap $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5). Arms: **B0** (unscored BM25), **A1** (seeds $M=10$), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion).
- **Scoring & Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds: $s(t, v) = \operatorname{logsumexp}(z[Y]) - \operatorname{logsumexp}(z[N])$ (19,298 passes). Dual T4s validate listwise `gemma-4-e4b-it`. Confirmatory family: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5; Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, $\Delta = -0.6$ pp, Holm $p=1.000$) and 40.4% Acc@5; standard PPR hits 50.0% (172/344). Tag maps (Aider) reach 40.0% File Acc@5 (0.0% Fn Acc@10); dense bi-encoders reach 38.4% Fn Acc@10. On Stratum C (131 tasks), BM25 hits 28 and Perron 27. Top-10 seeds cover 50.9% of gold targets. 1-hop covers 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment (66.9%). 2-hop reachability exceeds null graphs (63.8%) by +3.02 pp (CI [−0.42, +6.41]).

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Family (N=344)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 51.2% | 0.3315 |
| **A6** | Perron diffusion pool | 23.6% | 45.6% | 52.6% | 0.3301 |

- **Confirmatory (Acc@5) & Power Bounds**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). With 17 discordant tasks ($p_{	ext{disc}} = 4.94\%$), MDE at 80% power is 3.27 pp ($N = 942$ required for $+2.03$ pp). Under TOST, true population advantages $\ge +4.4$ pp are formally excluded ($p < 0.05$).
- **Exploratory Contrasts**: $A_4 - A_1$: $+0.87$ pp ($p=0.549$); null control $A_4 - A_5$: $+1.16$ pp ($p=0.344$); diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). BM25 (**B0**) maintains higher MRR@50 (0.344 vs 0.341). Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph targets (0.137 gap). RRF ($k_{	ext{rrf}}=60$) reaches 46.51% (160/344).

### 4.3 The Break-Even Discrimination Bound & Stratum Asymmetry

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Break-Even Discrimination Bound**: Expected outranking distractors is $\mathbb{E}[X \mid s^*] = (M - 1)(1 - 	ext{AUC})$. Landing in top-$k$ requires $	ext{AUC}^*_{	ext{breakeven}} \ge 1 - rac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $	ext{AUC} \ge 0.830$. Because Gemma 4 reaches only $	ext{AUC} = 0.6596$ on graph targets, expected distractors is $8.00$, placing targets at rank $9.00$ ($17.68$ at $M=50$).
- **Stratum Asymmetry (25 Graph-Only Targets)**: In Stratum A, 11 reached targets yield 6 top-5 conversions (54.5%, net $+5$ tasks, discordant 7/2, $p=0.1797$). In Stratum C, 14 reached targets yield 0 conversions (0.0%, net $-2$ tasks, discordant 0/2, $p=0.5000$), yielding $+3$ tasks aggregate (+0.87 pp, $p=0.549$).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

To evaluate whether the Lexical Abstraction Bottleneck reflects the compact scale of 2B models, protocol `PREREG-SCALING-v1` measured pointwise discrimination on held-out Stratum C tasks ($N=58$) across four architectures executed on Kaggle Dual Tesla T4 GPUs:

| Model ID | Architecture Archetype | Active / Total Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `google/gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `google/gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `microsoft/Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,230s |
| `google/gemma-4-26b-a4b-it` | Dynamic MoE (128 experts) | 3.8B / 26.0B | *Hardware Bound* | *44.0 GB req.* | *Exceeds T4* |

1. **Intra-Family Monotonic Scaling**: Discrimination scales monotonically within Gemma 4 ($0.8544 	o 0.9274$), confirming pre-registered Hypothesis $H_1$.
2. **Representational Efficiency**: Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (0.7836) and DeepSeek R1 Distill 7B (0.6489). DeepSeek-7B falls squarely below the Lexical Abstraction ceiling ($0.6489 \le 0.6596$), proving general reasoning models cannot bypass topological code graph structure.
3. **MoE Hardware Boundary**: In `transformers`, Gemma 4 26B-A4B stores its 128 experts as fused 3D `nn.Parameter` tensors, bypassing `bitsandbytes` 4-bit `nn.Linear` quantization loops. The unquantized 16-bit weights require $44.0$ GB VRAM, physically exceeding dual Tesla T4 capacity ($29.12$ GB). While dense edge models ($\le 4$B) operate cleanly within consumer budgets ($\le 9.6$ GB), offline sparse MoE agents require native pre-quantized expert formats.

### 4.5 Factorial Provenance Reasoning & Candidate Controls (N=131)

| Factorial Condition (N=131) | Acc@1 | Acc@5 | Hits | MRR | McNemar p (Acc@1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| C1: Baseline Isolated (No Prov, No Think) | 2.29% | 15.27% | 20 | 0.0778 | reference |
| C2: Call-Path Provenance (Prov, No Think) | 3.82% | 16.03% | 21 | 0.0899 | p = 0.5000 |
| C0: Unguided Thinking (No Prov, Think) | **9.92%** | **18.32%** | **24** | **0.1358** | **p = 0.0063\*** |
| C3: Provenance + Thinking (Prov, Think) | 9.16% | 17.56% | 23 | 0.1270 | **p = 0.0117\*** |
| *Factorial Contrast: C3 vs C0* | *-0.76 pp* | *-0.76 pp* | *-1* | *-0.0088* | *p = 1.0000 (Parity)* |
| *Subgroup: Seed-Connected (N=28)* | *C1: 10.7%* | *C0: 42.9%* | *+9 hits* | *--* | *Local disambiguation* |
| *Subgroup: Graph-Only (N=7)* | *C1: 0.0%* | *C0: 14.3%* | *+1 hit* | *--* | *28.6% Acc@5 (Exploratory)* |

- **Factorial Reasoning & Candidate Grounding**: All four factorial conditions evaluate candidate pool $A_4[:25]$ (Condition C1's 20 hits matching $A_1$ listwise reflects zero conversion on distant targets rather than a seeds-only pool). Isolated snippets (C1) yield 2.29% Top-1; static edge annotations alone (C2, 3.82%, $p=0.5000$) yield negligible change. Enabling `<thought>` traces (C0) raises Top-1 accuracy from 2.29% (3 hits) to **9.92% (13/131 hits, $+7.63$ pp, exact $p = 0.0063^*$, discordant $n_{10}=11, n_{01}=1$)** and increases MRR@50 by +74.6% to **0.1358** (~120 tokens, 10-15s on T4, <3s on desktop). Provenance plus thinking (C3, 9.16%, $p = 0.0117^*$) matches unguided thinking ($\Delta = -0.76$ pp, $p = 1.0000$, $\Delta_{	ext{interaction}} = -2.29$ pp; 11 identical hits), showing thinking tokens allow representations to trace relational paths directly. Partitioning the 35 in-pool defects shows seed-adjacent targets ($N=28$) improve from $10.7\% 	o 42.9\%$ Top-1 (+32.2 pp) by resolving ambiguity among local callers; distant graph targets ($N=7$) achieve exploratory recovery of 2/7 targets into Top-5 (28.6%, 1 Top-1 vs 0.0% in C1).
- **Selective Gating & Candidate Controls**: Gating by symbol density attains 47.38% Acc@5 (163/344); an oracle stratum router reaches 47.67% (+2.90 pp over BM25). On Stratum C ($N=131$), non-reasoning listwise ranking ($M \le 25$) ties depth BM25 at 16.79% (22/131 hits, $p=1.000$). Under test-time reasoning (Control 1: `<thought>` $\le 180$ tokens), graph expansion ($A_4$) breaks this parity, reaching 18.32% Acc@5 (24/131, 9.92% Top-1, MRR 0.1358) and outperforming depth BM25 (12.98%, 17/131) by +5.34 pp (+7 tasks) and seeds (10.69%) by +7.63 pp (+10 tasks). Control 2 confirms the 0.137 seed-neighbor AUC gap reflects candidate selection conditioning (counterfactual BM25 drops by 0.183 on the same pairs). Boundary loss (41.6 pp) exceeds compaction loss (11.3 pp) by $3.68	imes$ at $k=5$.

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

To isolate the causal mechanism of the Lexical Abstraction Bottleneck, protocol `PREREG-CAUSAL-v1` executed two controlled interventions across 58 held-out Stratum C tasks on Kaggle Dual Tesla T4 GPUs with `google/gemma-4-e2b-it`:

| Condition | Acc@1 | Acc@5 | MRR | Contrasts vs. Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **Experiment A: Causal Lexical Intervention (AST $lpha$-Renaming, N=58)** | | | | |
| $O_0$: Natural Code (Identifiers Intact) | 12.07% | 27.59% | 0.1804 | reference (7/58 hits) |
| $O_1$: Obfuscated Code ($lpha$-Renamed) | 10.34% | 24.14% | 0.1686 | -1.73 pp (6/58, exact $p = 1.0000$) |
| **Experiment B: Multi-Encoding Graph Representations (N=58)** | | | | |
| $E_0$: Isolated Code (Baseline Snippet) | 12.07% | 27.59% | 0.1804 | reference (7/58 hits) |
| $E_1$: Inline Comment Provenance (`# Provenance`) | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** (15/58, exact **$p = 0.0386^*$**) |
| $E_2$: Linearized Traversal Trace (`[Call Trace]`) | 17.24% | 32.76% | 0.2440 | +5.17 pp (10/58, $p = 0.4531$) |
| $E_3$: Subgraph Edge List (`### Edges: (u, rel, v)`) | 15.52% | 32.76% | 0.2419 | +3.45 pp (9/58, $p = 0.6875$) |
| $E_4$: Hierarchical Adjacency Map (`u -> [v, w]`) | 15.52% | 34.48% | 0.2362 | +3.45 pp (9/58, $p = 0.6875$) |

- **Lexical Independence**: Stripping user-defined function, parameter, and variable names via an AST `NodeTransformer` ($O_1$) lowers Top-1 localization by only 1.73 pp ($12.07\% 	o 10.34\%$, discordant $n_{10}=4, n_{01}=3$, $p=1.0000$) and Top-5 accuracy by $-3.45$ pp ($27.59\% 	o 24.14\%$). While surface lexical tokens provide positive discrimination signals, the model retains 10.34% Top-1 localization without lexical identifiers, indicating that control-flow syntax, operator structure, and statement geometry establish a resilient structural baseline.
- **Graph Serialization Hierarchy**: Testing the graph serialization taxonomy of Fatemi et al. [11] reveals that graph prompt format strongly governs code localization. Co-locating structural relations directly above symbol signatures ($E_1$: Inline Comment Provenance) achieves 25.86% Acc@1, 39.66% Acc@5, and an MRR of 0.3097, more than doubling Top-1 localization over isolated baseline snippets (+13.79 pp lift, exact binomial McNemar $p = 0.0386^*$, Wilcoxon signed-rank $W = 31.5, p = 0.0103^*$). Disconnected global preambles ($E_3$ Subgraph Edge Lists at 15.52% and $E_4$ Adjacency Maps at 15.52%) underperform inline comments: in multi-line code repositories, separating the graph edge list from code definitions forces long-range self-attention cross-referencing across thousands of tokens, whereas inline provenance co-locates relational context directly at the candidate's syntactic entry point.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but differences are null-compatible with gains $> +4.4$ pp excluded (PC1).
- **Distractor Degradation vs Lexical Bottleneck**: Non-reasoning listwise prompting enables comparative ranking (+8.40 pp on Stratum C, $p=0.0034$) where graph and depth BM25 tie at 16.79%. Test-time reasoning resolves this tie (+5.34 pp, 24 vs 17 hits) by allowing the model to trace non-lexical call paths.
- **Operating Regimes for Local Developer Agents**: The $3.68	imes$ bottleneck ratio defines operational guidelines: (1) **First-Pass Static Filtering**: Zero-copy CSR diffusion grounds candidate symbols in $<5$ ms within $\le 3$ MB RAM, capturing the 58.4% static reachability ceiling while preserving 91.9% context headroom; (2) **Test-Time Reflection**: Allocate $\sim 120$ `<thought>` tokens to discriminate local call-trees without prompt edge pollution; (3) **Dynamic Execution Fallback**: For the 41.6 pp boundary loss involving dynamic dispatch or framework configuration, agents must switch to test execution traces rather than expanding static graph radius.
- **Post-Training & Feature Noise**: Auxiliary heuristics exacerbate distractors: churn correlates weakly with defects ($	ext{AUC} < 0.60$), while complexity tracks file size. Sparse AST topology with provenance avoids heuristic inflation.
- **Limitations**: (1) Python ASTs; (2) compact models (E2B/E4B); (3) 1-hop pools ($M \le 50$); (4) reader/editor sample size ($N=100$ and $N=30$); (5) frozen zero-shot scorers.

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **Binary CSR Multigraphs**: Precomputed multigraphs for 12 repos in zero-copy CSR ($<1.2$ ms attach, $1.89	ext{--}5.10$ ms solve, $<3$ MB delta).
- **Evaluation Archive (`scores.parquet`)**: 19,298 evaluations across 344 tasks with log-odds.
- **Ground Truth & Strata**: Function targets and Strata A/B/C annotations across 800 tasks (300 Lite + 500 Verified; 344 callable Verified primary benchmark; CC0-1.0).
- **Engine Performance & Downstream Tooling**:
  - CSR diffusion solves in $1.89	ext{--}5.10$ ms ($196	ext{--}529$ qps) with $<3$ MB delta and $99.5\%$ HSI@10, outperforming NetworkX ($320$ ms) and SciPy ($28.4$ ms).
  - Downstream edit reader ($N=100$) reaches 74.0% EditLoc@1 under oracle context.
  - AST surgical patch editor ($N=30$) eliminates indentation errors, achieving 0.0% syntax errors vs 50.0% for unified diffs ($p < .001$).
- **Turnkey API & MCP Server**: Drop-in `perron.diffuse_with_provenance()` Python API and stdio JSON-RPC server (`perron-mcp`).
- **Kaggle Workstation & Scaling Benchmark**: Dual Tesla T4 reproduction notebooks validating context packing, causal controls (`yvliet/perron-gemma-4-workstation`), and cross-architecture scaling (`yvliet/perron-scaling-benchmark`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion pays off only when paired with test-time reasoning compute or localized inline comment provenance. Reasoning traces raise Top-1 localization from 2.29% to 9.92% (exact binomial $p = 0.0063^*$, discordant pairs $n_{10}=11, n_{01}=1$, MRR $+74.6\%$), while inline comment provenance doubles Top-1 accuracy to 25.86% ($+13.79$ pp lift, $p=0.0386^*$, Wilcoxon $p=0.0103^*$). Cross-architecture scaling confirms monotonic discrimination gains in Gemma 4 ($0.8544 	o 0.9274$) while quantifying the memory boundary for sparse MoE models on consumer GPUs. Paired with Perron's high-throughput CSR diffusion engine, these results establish that compact open models can navigate non-lexical code graphs when provided an allocated inference-time reasoning budget or localized structural provenance.

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
- [12] C. Sanford et al. *Understanding Transformer Reasoning Capabilities via Graph Algorithms.* NeurIPS 2024.
- [13] A. Bojchevski et al. *Scaling Graph Neural Networks with Approximate PageRank.* KDD 2020.
- [14] J. Yang et al. *SWE-agent.* arXiv:2405.15793, 2024.
- [15] R. Rafailov et al. *Direct Preference Optimization.* NeurIPS 2023.
- [16] E. Brochu et al. *A Tutorial on Bayesian Optimization.* arXiv:1012.2599, 2010.
- [17] T. Akiba et al. *Optuna.* KDD 2019.
