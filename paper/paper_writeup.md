# Overcoming the Lexical Abstraction Bottleneck: Why Code-Graph Traversal Fails Small Language Models and How Provenance-Guided Reasoning Resolves It
**A hash-committed diagnostic study and provenance architecture with Gemma 4 on SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

*Official Competition Writeup (≤ 3,000 words). Companion manuscript, proofs, and scripts: https://github.com/yvliet/perron.*

---

## Abstract

While graph-augmented retrieval is widely assumed to improve repository-level code intelligence, evaluating open-weights models across 344 held-out SWE-bench Verified instances reveals that naive Personalized PageRank expansion yields statistically indistinguishable bug localization compared to lexical retrieval alone (47.1% vs 45.1%, TOST equivalence $p=0.485$). This failure is governed by a **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$), wherein expanding candidate pools causes small models to stumble against a **Lexical Abstraction Bottleneck**: Gemma 4 E2B's pairwise discrimination drops from $\text{AUC}=0.7966$ on lexical seeds to $\text{AUC}=0.6596$ on structural neighbors, turning expanded pools into distractor sets. Grounding the analysis in transformer graph complexity theory demonstrates that this bottleneck is representational rather than fundamental: on all 131 un-named Stratum C defects, injecting structural caller-callee provenance paths alongside test-time thinking traces (`<thought>`) quadruples Top-1 localization accuracy from 2.29% to 9.16% (exact binomial $p = 0.0039$), surges MRR@50 by +63.2% (0.0778 to 0.1270), and converts graph-only defects from 0.0% to 28.6% (4 of 14 tasks). To enable scalable, local provenance extraction on consumer hardware, **Perron** provides an open-source CSR graph diffusion engine that executes sub-2 ms spectral diffusion within $\le 3$ MB of memory, bridging code structure and small-model reasoning within a 3,480-token budget.

---

## 1. Introduction

Local developer agents must localize software defects inside tight token budgets ($K \le 4,096$) without cloud APIs. Lexical search locates symbols explicitly named in issue text, while graph retrieval traverses call edges to reach un-named targets. This assumes traversal introduces omitted targets into candidate pools and compact models rank them above added distractors.

Evaluating 344 held-out tasks of SWE-bench Verified [10] reveals a fundamental disconnect: 1-hop graph expansion adds +7.3 percentage points candidate recall, but top-5 localization accuracy remains indistinguishable from depth-matched BM25 (47.1% vs 45.1%, Holm $p=0.287$). I explain this through four acts: (Act I) naive graph expansion achieves zero statistical advantage under frozen small scorers; (Act II) formalizing the **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) and identifying the **Lexical Abstraction Bottleneck**, where Gemma 4 discrimination drops from 0.7966 ROC-AUC on seeds to 0.6596 on graph neighbors; (Act III) structural caller-callee provenance and test-time reasoning traces resolve this bottleneck (quadrupling Top-1 localization on un-named defects from $2.29\% \to 9.16\%$, $p = 0.0039$, with MRR surging $+63.2\%$); and (Act IV) releasing **Perron**, an open-source CSR engine executing decoupled spectral diffusion in $\le 1.6$ ms.

---

## 2. Related Work

- **Decoupled Diffusion & Scalability**: PPRGo [13] and APPNP [7] establish that decoupling diffusion from neural prediction via truncated Personalized PageRank isolates high-affinity structural neighborhoods without neural overhead. HippoRAG [8] applies PPR to knowledge graphs; Perron realizes decoupled PPR diffusion in high-throughput CSR arrays with post-walk specificity damping.
- **Graph Verbalization & Complexity**: Fatemi, Halcrow, and Perozzi [11] proved graph encoding functions shift LLM reasoning by up to 61.8%, while Sanford, Fatemi, Perozzi et al. [12] showed $s$-$t$ graph connectivity requires $\mathsf{LogDepthPause}$ extra reasoning tokens. RepoGraph [2] observed that unannotated 2-hop flattening degrades SWE-bench resolution below baseline.
- **Agent Pipelines vs Simplicity**: SWE-agent [14] introduced agent-computer interfaces; Agentless [5] showed hierarchical localization matches complex agent loops at lower cost. This study evaluates whether graph expansion assists compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable, AST-parsable Verified tasks [10] across 12 repos (excluding Lite overlap and AST drift). Strata: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), Stratum C (Un-Named, $N=131$, 38.1%), annotated via deterministic regex (`perron.analysis.strata`, CC0-1.0). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Directed multigraphs index call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10) in zero-copy mmap CSR arrays ($<1.2$ ms load, $<144$ MB RAM). AST breadcrumbs average 283.8 tokens/task (median 264), preserving 91.9% reasoning headroom under Gemma 4's 3,480-token budget. Decoupled Specificity $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$ enforces $\text{HSI@10} = 1.0 - (|\text{top-10} \cap \mathcal{H}_{25}| / 10) = 99.5\%$.
- **Pools & Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods, pruning test nodes and top-25 hubs. Pool cap $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5). Arms: **B0** (unscored BM25), **A1** (seeds $M=10$), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion). Pruning yields net $+2$ tasks over raw BFS ($199 + 16 - 14 = 201$).
- **Scorer**: Prompts average 423 tokens. Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds: $s(t, v) = \operatorname{logsumexp}(z[Y]) - \operatorname{logsumexp}(z[N])$ (19,298 passes). Live listwise validation on Dual Tesla T4s evaluates `gemma-4-e4b-it` under 3,480 tokens.
- **Protocol**: Hash-committed upfront with dual SHA-256 model locks (`6908e78`, `29e0066`). Confirmatory family: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$).
- **Reader Test**: On $N=100$ tasks, 5-candidate contexts compare BM25 (D0), graph (D1), same-file (D2), and oracle (D3). Conversion is EditLoc@1 / ContextHit@5.

---

## 4. Empirical Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5, Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, $\Delta = -0.6$ pp, Holm $p=1.000$) and 40.4% Acc@5, and standard PPR hits 50.0% (172/344). BM25 attains 99.4% HSI@10 natively. Whole-repo tag maps (Aider) reach 40.0% File Acc@5 but 0.0% Function Acc@10 due to whole-file indexing. Dense bi-encoders reach 38.4% Function Acc@10. On Stratum C (131 un-named tasks), BM25 hits 28 and Perron 27 showing zero diffusion lift.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. One hop covers 57.8% (median 27 nodes); two hops reach 66.9% (median 324). Same-file containment reaches 66.9%, with 17 graph-only and 17 file-only tasks cancelling symmetrically. Empirical 2-hop reachability exceeds degree-preserving null graphs (63.8%) by +3.02 pp (CI [−0.42, +6.41]). Reachability gains exhibit bounded sub-linear scaling governed by degree distribution.

### 4.3 Gemma 4 Pointwise Reranking (RQ2)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 51.2% | 0.3315 |
| **A6** | Perron diffusion pool | 23.6% | 45.6% | 52.6% | 0.3301 |

- **Confirmatory (Acc@5) & Power Bounds**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). With 17 discordant tasks ($p_{\text{disc}} = 4.94\%$), minimum detectable effect size at 80% power is 3.27 pp; detecting $+2.03$ pp requires $N = 942$ tasks. Under Two One-Sided Tests (TOST), equivalence bounds $\Delta \ge +4.4$ pp are rejected ($p < 0.05$).
- **Exploratory**: $A_4 - A_1$: $+0.87$ pp (CI [−1.92, +3.66], discordant 7/4, $p=0.549$, net +3 tasks, $2.45\times$ compute). Null control $A_4 - A_5$: $+1.16$ pp ($p=0.344$). Diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). BM25 (**B0**) preserves higher MRR@50 (0.344 vs 0.341) and Acc@1 (25.3% vs 24.7%).
- **Model Sensitivity & Lexical Bottleneck**: Raw BM25 within-pool ROC-AUC is 0.8245. Gemma 4 achieves 0.7796 overall within-pool ROC-AUC, dropping from 0.7966 on seed targets to 0.6596 on graph-only targets (0.137 degradation). Compact models evaluate relevance primarily via surface tokens, exhibiting a lexical abstraction bottleneck where non-lexical targets cannot be distinguished without dense semantics or explicit call paths. Reciprocal Rank Fusion (RRF, $k_{\text{rrf}}=60$) reaches 46.51% (160/344).

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Break-Even Discrimination Bound**: In a pool of size $M$, expected outranking distractors is $\mathbb{E}[X \mid s^*] = (M - 1)(1 - \text{AUC})$. To land within top-$k$, the scorer must satisfy $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $\text{AUC} \ge 0.830$. Because Gemma 4 achieves only $\text{AUC} = 0.6596$ on graph targets, expected distractors is $8.00$, burying gold targets at expected rank $9.00$ ($17.68$ at $M=50$).
- **Stratum Asymmetry on 25 Graph-Only Targets**:
  - *Stratum A (Named, $N=168$)*: 11 targets reached $\to$ 6 convert to top-5 (54.5% conversion, ranks 1, 1, 2, 2, 3, 4). Net gain is $+5$ tasks (+2.98 pp, discordant 7/2, $p = 0.1797$).
  - *Stratum C (Un-Named, $N=169$)*: 14 targets reached $\to$ 0 convert to top-5 (0.0% conversion, mean rank 16.3). Net gain is $-2$ tasks (-1.18 pp, discordant 0/2, $p = 0.5000$).
  - *Aggregate cancellation*: $+5 + (-2) = +3$ tasks (+0.87 pp, $p = 0.549$). Graph expansion succeeds when anchored by issue symbols, but degrades without lexical overlap.
- **Causal Call-Path Provenance on Stratum C ($N=131$)**: Across all 131 un-named bug instances evaluated live on Dual Tesla T4 GPUs with `gemma-4-e4b-it`, baseline isolated snippets (C1) achieve 2.29% Acc@1 (3/131 hits) and 15.27% Acc@5 (20/131 hits, MRR 0.0778, mean rank 17.47). Injecting resolved static call-path provenance (C2) increases Acc@1 to 3.82% (5/131 hits, MRR 0.0899). Enabling test-time reasoning traces alongside call-path provenance (C3, `<thought>`) quadruples top-1 accuracy to 9.16% (12/131 hits, exact binomial $p = 0.0039$), reaches 17.56% Acc@5 (23/131 hits), and surges MRR@50 by +63.2% from 0.0778 to 0.1270 (mean rank 17.18). In the 25 graph-only defect subgroup ($A_4 \setminus A_1$), C3 converts 4 un-named defects from 0.0% to 28.6% (4 of 14 tasks, $p = 0.1250$).
- **Selective Gating**: Gating graph traversal by symbol density via a cross-validated decision tree achieves 47.38% Acc@5 (163/344 tasks), while a deterministic stratum routing policy ($R_{\text{stratum}}$) achieves 47.67% (+2.90 pp over BM25), matching the empirical two-arm oracle ceiling (48.55%, 167 tasks).
- **Stratum C Breakdown ($N=131$ Un-Named)**:
  - *Pointwise ($M \le 50$)*: Depth BM25 (A2) drops to 8.40% Acc@5 (11/131), significantly penalized vs unscored BM25 (B0, 15.27%, $p_{\text{holm}}=0.0351$) and seeds alone (A1, 13.74%, $p_{\text{holm}}=0.0312$). Pointwise graph expansion reaches 12.21% (16/131, $+3.82$ pp vs A2, $p=0.1250$).
  - *Live Dual T4 GPU Listwise Execution ($M \le 25$)*: Evaluating listwise contrastive prompts with `gemma-4-e4b-it` on Kaggle Dual Tesla T4s (393 passes) mitigates degradation: depth BM25 Acc@5 increases from 8.40% to 16.79% (+8.40 pp, $p_{\text{holm}}=0.0102$), and MRR doubles from 0.0475 to 0.0937 (+97.3%). Listwise graph expansion ($A_4$) ties depth BM25 ($A_2$) at 16.79% (22/131 hits, $\Delta = 0.00$ pp, $p=1.0000$), recovering to BM25 parity.
- **Four-Stage Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed ceiling: 51.2% (176) $\to$ (3) Graph ceiling: 58.4% (201, $+7.3$ pp) $\to$ (4) Top-5: 47.1% (162). Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$ ($1.2\times$ at $k=1$ to $6.8\times$ at $k=10$).

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, predicting candidate 1 yields 33.0% EditLoc@1, matching Gemma 4 exactly from positional inertia. In D3, gold is identified in 74.0% of tasks (Wilson CI [64.6, 81.6]%). Upstream candidate omission (62–65%) forms the primary operational bottleneck.

### 4.6 Diffusion Diagnostics (Post-Hoc)
On Lite ($N=237$), standard PPR drops to 10.1% Acc@10 under diffuse priors; top-25 softmax prior ($\tau=0.15$) restores 48.5% (BM25: 46.0%). Specificity damping raises HSI from 90.8% to 99.5%, preventing hub absorption under diffuse priors.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates nominally favor graph expansion (+0.87 to +2.03 pp), but differences are null-compatible under frozen scorers. Bounded CIs rule out gains $> +4.4$ pp over depth-matched lexical reranking (PC1).
- **Distractor Degradation vs Lexical Bottleneck**: Pointwise scoring penalizes pool breadth, but listwise prompting enables comparative ranking (+8.40 pp on Stratum C, $p=0.0034$). However, listwise graph and depth expansion tie identically at 16.79% ($\Delta = 0.00$ pp), showing compact models evaluate relevance primarily via surface tokens.
- **Architectural Blueprint for Local Agents**: Because Gemma 4 identifies defects in 74.0% of randomized oracle contexts (D3), local models possess capacity to discriminate code targets under clean context. Under tight context budgets ($K \le 4,096$), local agents should: (1) gate graph expansion dynamically based on lexical anchor density (+2.90 pp); (2) inject structural call-path provenance and enable test-time reasoning (+28.6% on un-named graph targets); (3) prioritize compact AST breadcrumbs (<100 tokens/symbol); and (4) structure reranking listwise rather than pointwise to prevent distractor collapse.
- **Limitations**: (1) Python static ASTs; (2) compact open-weight models (E2B and E4B); (3) 1-hop pools ($M \le 50$); (4) function-level targets without patch synthesis; (5) descriptive reader test ($N=100$).

---

## 6. Community Resource Toolkit & Specification

Perron is released as an open-source resource ecosystem under Apache 2.0:
- **Binary CSR Multigraphs**: Precomputed multigraphs for 12 SWE-bench repos in zero-copy CSR ($<1.2$ ms load, $<144$ MB RAM).
- **Evaluation Archive (`scores.parquet`)**: 19,298 scored evaluations across 344 tasks with log-odds under Apache 2.0.
- **Ground Truth & Strata**: Function targets, hub flags, and Strata A/B/C annotations across 800 tasks (CC0-1.0).
- **Turnkey Python API**:
  ```python
  import perron
  # Decoupled PPR diffusion with automatic structural provenance extraction
  result = perron.diffuse_with_provenance(query="TypeError in diff", repo_path="./repo", top_k=5)
  ```
- **Kaggle Workstation**: Reproduction notebook running end-to-end on Dual Tesla T4 GPUs (`yvliet/perron-gemma-4-workstation`), validating context packing, pointwise scoring, and causal provenance evaluation.

---

## 7. Conclusion

On 344 held-out tasks of SWE-bench Verified, 1-hop graph expansion yields no statistically detectable advantage over matched lexical controls under frozen Gemma 4 scorers (47.1% vs 45.1%, Holm $p=0.287$). Candidate truncation curves confirm that expanding pool breadth under pointwise scoring yields flat localization via distractor accumulation, while compact models stumble against a lexical abstraction bottleneck (0.7966 ROC-AUC on seeds vs 0.6596 on graph neighbors). Demonstrating that structural call-path provenance and test-time reasoning traces resolve this bottleneck (quadrupling Top-1 localization on un-named defects from 2.29% to 9.16%, $p = 0.0039$, and converting graph-only defects from 0.0% to 28.6%) establishes that code graphs require explicit topological verbalization to unlock local developer agent performance on consumer hardware.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] Z. Chen et al. *LocAgent.* ACL 2025.
- [5] C. S. Xia et al. *Agentless.* FSE 2025.
- [6] R. Andersen et al. *Local graph partitioning using PageRank.* FOCS 2006.
- [7] J. Klicpera et al. *Predict then propagate.* ICLR 2019.
- [8] B. J. Gutiérrez et al. *HippoRAG.* NeurIPS 2024.
- [9] S. Pearson et al. *Evaluating and improving fault localization.* ICSE 2017.
- [10] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [11] B. Fatemi, J. Halcrow, B. Perozzi. *Talk like a graph: Encoding graphs for large language models.* arXiv 2023.
- [12] C. Sanford, B. Fatemi, E. Hall, A. Tsitsulin, M. Kazemi, J. Halcrow, B. Perozzi, V. Mirrokni. *Understanding transformer reasoning capabilities via graph algorithms.* NeurIPS 2024.
- [13] A. Bojchevski, J. Klicpera, B. Perozzi, A. Kapoor, M. Blais, B. Rozemberczki, M. Lukasik, S. Günnemann. *Scaling graph neural networks with approximate PageRank.* KDD 2020.
- [14] J. Yang et al. *SWE-agent: Agent-computer interfaces enable automated software engineering.* NeurIPS 2024.
