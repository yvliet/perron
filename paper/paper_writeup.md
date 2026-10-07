# Graph Expansion Adds Little to Lexical Retrieval for Small-LLM Bug Localization
**A hash-committed diagnostic study with Gemma 4 on a 344-task held-out subset of SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

*Official Competition Writeup (≤ 3,000 words). Unabridged companion research manuscript, proofs, and scripts available at https://github.com/yvliet/perron.*

---

## Abstract

Does call-graph expansion improve bug localization over lexical retrieval for small local models? Evaluating 344 held-out tasks of SWE-bench Verified with Perron and frozen Gemma 4 scorers reveals that while 1-hop expansion adds 7.3 pp pool recall over seeds (51.2% to 58.4%), Gemma 4 converts only 0.87 pp into top-5 accuracy (+3 tasks, $p=0.549$). Versus depth-matched BM25, graph expansion yields 47.1% vs 45.1% Acc@5 (+2.03 pp, 95% CI [−0.29, +4.36], Holm $p=0.287$). On Stratum C ($N=131$ un-named tasks), live GPU listwise ranking mitigates pool degradation (+8.40 pp, $p=0.0034$), though graph and depth BM25 tie at 16.79% due to a lexical abstraction bottleneck (ROC-AUC drops from 0.7966 on seeds to 0.6596 on graph-only targets). Reader evaluation shows local reasoning is intact under oracle context (74.0% accuracy), but upstream boundary loss (41.6 pp) exceeds compaction loss (11.3 pp) by $3.68\times$ at $k=5$. All 19,298 evaluations, CSR multigraphs, and code are released under Apache 2.0.

---

## 1. Introduction

Local developer agents localize code defects within tight token budgets without cloud APIs. Lexical search locates symbols explicitly named in issue text; graph retrieval traverses call edges to reach un-named targets. This assumes traversal introduces omitted targets into candidate pools, and compact models rank them above added distractors.

I evaluate both assumptions on 344 held-out tasks of SWE-bench Verified using Perron and frozen Gemma 4 scorers across three questions: (RQ1) Does PPR diffusion beat BM25? (RQ2) Does graph 1-hop expansion outperform depth-matched and same-file controls? (RQ3) Where is accuracy lost across the pipeline? Pre-registered confirmatory contrasts evaluate PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$). Contributions include matched controls, causal call-path interventions, selective routing, and 19,298 scored evaluations.

---

## 2. Related Work

- **Graphs & Agents**: Prior systems (Aider [1], RepoGraph [2], CodexGraph [3], Agentless [6]) evaluate frontier cloud models. I evaluate whether graph expansion assists compact 2B/4B open models.
- **Random-Walk Retrieval**: Personalized PageRank [7, 8, 9] and HippoRAG [10] motivate graph diffusion; Perron decouples walks from stationary baselines.
- **Reranking**: Pointwise scorers [13] accumulate distractors as pools expand; listwise rerankers [14, 15] enable comparative cross-candidate attention.
- **Empirical Rigor**: Pearson et al. [11] showed synthetic fault results fail to transfer to real code, motivating authentic SWE-bench evaluation [12].

---

## 3. Study Design

- **Benchmark & Strata**: 344 callable, AST-parsable Verified tasks [12] across 12 repos (excluding Lite overlaps, AST drift, non-callable). Strata: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), Stratum C (Un-Named, $N=131$, 38.1%), annotated via deterministic regex (`perron.analysis.strata`, CC0-1.0). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Directed multigraphs index call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10) in zero-copy mmap CSR arrays ($<5$ ms load, 144 MB RAM). AST breadcrumbs average 283.8 tokens/task (median 264), preserving 91.9% reasoning headroom under Gemma 4's 3,480-token budget. Decoupled Specificity $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$ enforces $\text{HSI@10} = 1.0 - (|\text{top-10} \cap \mathcal{H}_{25}| / 10)$.
- **Pools & Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods, pruning test nodes and top-25 hubs. Pool cap $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5). Arms: **B0** (unscored BM25), **A1** (seeds $M=10$), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion). Pruning yields net $+2$ tasks over raw BFS ($199 + 16 - 14 = 201$).
- **Scorer**: Prompts average 423 tokens. Frozen `gemma-4-e2b-it` (FP16, thinking disabled) scores candidates via log-odds: $s(t, v) = \operatorname{logsumexp}(z[Y]) - \operatorname{logsumexp}(z[N])$ (19,298 passes, `logits_to_keep=1`). Live listwise validation on Dual Tesla T4s evaluates `gemma-4-e4b-it` under 3,480 tokens.
- **Protocol**: Hash-committed upfront with dual SHA-256 model locks. Pre-specified confirmatory family: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$).
- **Reader Test**: On $N=100$ tasks, 5-candidate contexts compare BM25 (D0), graph (D1), same-file (D2), and oracle (D3). Conversion is EditLoc@1 / ContextHit@5.

---

## 4. Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5, Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, $\Delta = -0.6$ pp, Holm $p=1.000$) and 40.4% Acc@5, and standard PPR hits 50.0% (172/344). BM25 attains 99.4% HSI@10 natively. Whole-repo tag maps (Aider) reach 40.0% File Acc@5 but 0.0% Function Acc@10 due to whole-file indexing. Dense bi-encoders reach 38.4% Function Acc@10. On Stratum C (131 un-named tasks), BM25 hits 28 and Perron 27 showing zero diffusion lift.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. One hop covers 57.8% (median 27 nodes); two hops reach 66.9% (median 324). Same-file containment reaches 66.9%, with 17 graph-only and 17 file-only tasks cancelling symmetrically. Empirical 2-hop reachability exceeds degree-preserving null graphs (63.8%) by +3.02 pp (CI [−0.42, +6.41]). Reachability gains are modest and degree-driven.

### 4.3 Gemma 4 Pointwise Reranking (RQ2)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |

- **Confirmatory (Acc@5) & Power Bounds**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). With 17 discordant tasks ($p_{\text{disc}} = 4.94\%$), minimum detectable effect size at 80% power is 3.27 pp (11.3 tasks); detecting $+2.03$ pp requires $N = 942$ tasks. Under Two One-Sided Tests (TOST), we reject equivalence bounds $\Delta \ge +4.4$ pp ($p < 0.05$), ruling out large localization gains from unguided 1-hop expansion.
- **Exploratory**: $A_4 - A_1$: $+0.87$ pp (CI [−1.92, +3.66], discordant 7/4, $p=0.549$, net +3 tasks, $2.45\times$ compute). Null control $A_4 - A_5$: $+1.16$ pp (CI [−0.58, +2.91], $p=0.344$). Diffusion $A_4 - A_6$: $+1.45$ pp (CI [+0.00, +3.20], $p=0.180$). BM25 (**B0**) preserves higher MRR@50 (0.344 vs 0.341) and Acc@1 (25.3% vs 24.7%). Non-Django ($N=230$) $A_4 - A_2$ is $+2.17$ pp ($p=0.227$).
- **Model Sensitivity & Lexical Abstraction Bottleneck**: Raw BM25 within-pool ROC-AUC is 0.8245, explaining why unscored BM25 preserves higher top-1 accuracy (25.3%) and MRR@50 (0.3439). Gemma 4 achieves 0.7796 overall within-pool ROC-AUC, dropping from 0.7966 on seed targets to 0.6596 on graph-only targets (0.137 degradation). Compact models evaluate relevance primarily via surface tokens, exhibiting a lexical abstraction bottleneck where non-lexical targets cannot be distinguished without dense semantics or explicit call paths. Reciprocal Rank Fusion (RRF, $k_{\text{rrf}}=60$) combining BM25 rank with Gemma score reaches 46.51% (160/344).

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Break-Even Discrimination Bound**: In a pool of size $M$, expected outranking distractors is $\mathbb{E}[X \mid s^*] = (M - 1)(1 - \text{AUC})$. To land within top-$k$, the scorer must satisfy $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $\text{AUC} \ge 0.830$. Because Gemma 4 achieves only $\text{AUC} = 0.6596$ on graph targets, expected distractors is $8.00$, burying gold targets at expected rank $9.00$ ($17.68$ at $M=50$).
- **Stratum Asymmetry on 25 Graph-Only Targets**: Of 25 targets reachable only through 1-hop expansion, conversion is polarized by issue text:
  - *Stratum A (Named, $N=168$)*: 11 targets reached $\to$ 6 convert to top-5 (54.5% conversion, ranks 1, 1, 2, 2, 3, 4). With 1 seed promotion and 2 displacements, net gain is $+5$ tasks (+2.98 pp, discordant 7/2, $p = 0.1797$).
  - *Stratum C (Un-Named, $N=169$)*: 14 targets reached $\to$ 0 convert to top-5 (0.0% conversion, mean rank 16.3). With 2 displaced seed hits, net gain is $-2$ tasks (-1.18 pp, discordant 0/2, $p = 0.5000$).
  - *Aggregate cancellation*: $+5 + (-2) = +3$ tasks (+0.87 pp, $p = 0.549$). Graph expansion succeeds when anchored by issue symbols, but degrades into pure noise without lexical overlap.
- **Causal Call-Path Provenance ($N=25$)**: To test the lexical bottleneck causally, I audited the 25 graph-only tasks ($A_4 \setminus A_1$). 24 of 25 (96.0%) share direct static edges with seeds (10 callees, 14 callers). Injecting call-path provenance (`# Provenance: Called by seed...`) elevates Acc@5 from 12.0% to 40.0% (10/25, +28.0 pp, $p=0.0156$), converting Stratum C from 0.0% to 35.7% (5/14). Test-time reasoning (`<thought>`) reaches 56.0% (14/25, +44.0 pp, $p=0.0010$, mean rank 4.05).
- **Stratum-Aware Selective Routing ($R_{\text{stratum}}$)**: Routing Stratum A/B to Graph ($A_4$) and Stratum C to Seeds ($A_1$) achieves 164/344 hits (47.67%), beating BM25 (+2.90 pp, +10 tasks), seeds (+1.45 pp), and static graph (+0.58 pp, 0 losses). Provenance routing on Stratum C reaches 169/344 (49.13%, +4.36 pp).
- **Stratum C Breakdown ($N=131$ Un-Named)**:
  - *Pointwise ($M \le 50$)*: Depth BM25 (A2) drops to 8.40% Acc@5 (11/131, MRR 0.0475), significantly penalized vs unscored BM25 (B0, 15.27%, 20/131, $p=0.0117 \to p_{\text{holm}}=0.0351$) and seeds alone (A1, 13.74%, 18/131, $p=0.0156 \to p_{\text{holm}}=0.0312$). Pointwise graph expansion reaches 12.21% (16/131, $+3.82$ pp vs A2, $p=0.1250, p_{\text{holm}}=0.1250$).
  - *Live Dual T4 GPU Listwise Execution ($M \le 25$)*: Evaluating listwise contrastive prompts with `gemma-4-e4b-it` on Kaggle Dual Tesla T4s (393 passes) mitigates candidate pool degradation: depth BM25 Acc@5 increases from 8.40% to 16.79% (+8.40 pp, $p=0.0034 \to p_{\text{holm}}=0.0102$), and MRR doubles from 0.0475 to 0.0937 (+97.3%). Notably, listwise graph expansion ($A_4$) ties depth BM25 ($A_2$) at 16.79% (22/131 hits, $\Delta = 0.00$ pp, $n_{10}=0, n_{01}=0, p=1.0000$). Listwise prompting enables comparative cross-candidate ranking, recovering performance back to BM25 parity, but performance remains constrained by surface lexical alignment.
- **Four-Stage Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed ceiling: 51.2% (176) $\to$ (3) Graph ceiling: 58.4% (201, $+7.3$ pp) $\to$ (4) Top-5: 47.1% (162). Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$ (spanning $1.2\times$ at $k=1$ to $6.8\times$ at $k=10$): expansion adds 7.3 pp recall, but Gemma 4 converts only $\sim 1$ pp.

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, predicting candidate 1 yields 33.0% EditLoc@1, matching Gemma 4 exactly (33.0%) from positional inertia. In D3, gold is identified in 74.0% of tasks (Wilson CI [64.6, 81.6]%). While the reader misses 26% of oracle cases, retrieval supplies gold in only 35–38%, forming the main bottleneck.

### 4.6 Diffusion Diagnostics & Continuous Dose-Response (Post-Hoc)
On Lite ($N=237$), standard PPR drops to 10.1% Acc@10 under diffuse priors; top-25 softmax prior ($\tau=0.15$) restores 48.5% (BM25: 46.0%). Specificity damping raises HSI from 90.8% to 99.5%, preventing hub absorption under diffuse priors.

---

## 5. Discussion

- **Empirical verdict**: Point estimates nominally favor graph expansion (+0.87 to +2.03 pp), but differences are statistically null-compatible under frozen small scorers. Bounded CIs rule out large gains ($> +4.4$ pp, $15$ tasks) over depth-matched lexical reranking (PC1).
- **Distractor degradation vs Lexical Bottleneck**: Pointwise scoring penalizes pool breadth, but listwise prompting enables comparative ranking (+8.40 pp on Stratum C, $p=0.0034$). However, listwise graph and depth expansion tie identically at 16.79% ($\Delta = 0.00$ pp). Compact models like Gemma 4 evaluate relevance predominantly via surface lexical tokens rather than structural execution paths.
- **Architectural blueprint for local agents**: Because Gemma 4 identifies defects in 74.0% of oracle contexts (D3), local models possess effective code discrimination once retrieved. Under tight context budgets ($K \le 4,096$), local agents should: (1) route graph hops selectively ($\\pi^*$, +2.90 pp); (2) inject structural call-path provenance (+28.0 pp on graph targets); (3) prioritize compact AST breadcrumbs (<100 tokens/symbol); and (4) structure reranking listwise rather than pointwise (+8.40 pp, $p=0.0034$).
- **Reusable audit framework**: Any retrieval stage feeding a local model can be audited similarly: report pool recall, evaluate matched lexical controls, and decompose upstream boundary loss from ranking compaction.
- **Limitations**: (1) Python static ASTs; (2) compact open-weight models (E2B and E4B); (3) 1-hop pools ($M \le 50$); (4) function-level targets without patch synthesis; (5) descriptive reader test ($N=100$).

---

## 6. Community Resource Toolkit & Specification

To establish an open benchmark and acceleration suite for local developer agent research, Perron is released as a production-grade resource ecosystem under Apache 2.0:

- **Binary CSR Multigraphs**: Precomputed multigraphs for 12 canonical SWE-bench repos in zero-copy CSR ($<2$ ms load, $<145$ MB RAM).
- **Evaluation Archive (`scores.parquet`)**: 19,298 scored candidate evaluations across 344 tasks with log-odds and format mass.
- **Ground Truth & Strata**: Function targets, hub flags, and Strata A/B/C annotations across 800 tasks (CC0-1.0).
- **Turnkey Python API**:
  ```python
  from perron import CodeGraph
  graph = CodeGraph.load("graphs/astropy.csr")
  top_candidates = graph.diffuse(issue_text, gamma=0.70, top_k=25)
  ```
- **Kaggle Workstation**: Public reproduction notebook running end-to-end on free Dual Tesla T4 GPUs (`yvliet/perron-gemma-4-workstation`), validating context packing, pointwise scoring, and live listwise contrastive reranking.

---

## 7. Conclusion

On 344 held-out tasks of SWE-bench Verified, 1-hop graph expansion yields no statistically detectable advantage over matched lexical controls under frozen Gemma 4 scorers, nor does it expand reachability beyond lexical depth. While adding 7.3 pp pool recall over seeds, the model converts only 0.87 pp into top-5 accuracy (+3 tasks, $p=0.549$). Candidate truncation curves confirm that expanding pool breadth under pointwise scoring yields flat localization via distractor accumulation; on un-named tasks, pointwise pool expansion reduces accuracy relative to unscored BM25 ($p=0.0117$). While listwise contrastive prompting mitigates pool degradation (+8.40 pp, $p=0.0034$), graph expansion ties lexical depth at 16.79% ($\Delta = 0.00$ pp, $p=1.0000$), reflecting a lexical abstraction bottleneck where compact models struggle on targets lacking surface token overlap. Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$ ($1.2\times$ to $6.8\times$). Null-compatible estimates (+0.87 to +2.03 pp) present an unfavourable trade-off against a $2.45\times$ compute increase under pointwise scoring. Local developer agents are more reliably guided toward high-precision lexical retrieval, compact AST breadcrumbs, and listwise ranking.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] Z. Chen et al. *LocAgent.* ACL 2025.
- [5] Y. Zhang et al. *AutoCodeRover.* ISSTA 2024.
- [6] C. S. Xia et al. *Agentless.* FSE 2025.
- [7] R. Andersen et al. *Local graph partitioning using PageRank.* FOCS 2006.
- [8] J. Klicpera et al. *Predict then propagate.* ICLR 2019.
- [9] D. Liben-Nowell, J. Kleinberg. *Link prediction in social networks.* JASIST 2007.
- [10] B. J. Gutiérrez et al. *HippoRAG.* NeurIPS 2024.
- [11] S. Pearson et al. *Evaluating and improving fault localization.* ICSE 2017.
- [12] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [13] R. Nogueira et al. *Document ranking with sequence-to-sequence.* Findings of EMNLP 2020.
- [14] W. Sun et al. *Is ChatGPT good at search?* EMNLP 2023.
- [15] R. Pradeep et al. *RankZephyr.* arXiv 2023.
- [16] T. H. Haveliwala, S. D. Kamvar. *Second eigenvalue of the Google matrix.* Stanford TR, 2003.
