# Graph Expansion Adds Little to Lexical Retrieval for Small-LLM Bug Localization
**A hash-committed diagnostic study with Gemma 4 on a 344-task held-out subset of SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

*Official Competition Writeup (≤ 3,000 words). Unabridged companion research manuscript, proofs, and scripts available at https://github.com/yvliet/perron.*

---

## Abstract

Does call-graph expansion improve bug localization over lexical retrieval for small, locally deployed models? I investigate this question on a 344-task held-out subset of SWE-bench Verified using Perron and a frozen `gemma-4-e2b-it` scorer under a hash-committed protocol. While 1-hop graph expansion adds 7.3 pp candidate pool recall over seeds alone (51.2% to 58.4%), the model converts only 0.87 pp into top-5 accuracy (gross 7/4 split: 6 graph + 1 seed promotion vs 4 displaced hits, net +3 tasks, $p=0.549$), consistent with distractor accumulation under pointwise scoring. Compared to depth-matched BM25, graph expansion yields 47.1% vs 45.1% Acc@5 (+2.03 pp, 95% CI [−0.29, +4.36], Holm $p=0.287$). In Stratum C ($N=131$ un-named tasks), graph expansion achieves 12.2% Acc@5 vs 8.4% for depth BM25 (+3.82 pp, $p=0.125$), but neither outperforms seeds alone (13.7%) or unscored BM25 (15.3%, $p=0.0117$). Upstream boundary loss before scoring (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$. The confirmatory CI rules out gains $> +4.4$ pp ($15$ tasks) over depth-matched lexical reranking (PC1). All 19,298 scored candidate evaluations, precomputed repository graphs, and the engine are released under Apache 2.0.

---

## 1. Introduction

Local developer agents must localize code defects within tight token budgets without cloud inference. Lexical search locates symbols explicitly named in issue text; graph retrieval traverses call edges to reach un-named targets. This relies on two assumptions: traversal places omitted targets into candidate pools, and compact models rank them above added distractors.

I evaluate both assumptions on 344 held-out tasks of SWE-bench Verified using Perron and a frozen Gemma 4 scorer across three questions: (RQ1) Does PPR diffusion beat BM25? (RQ2) Does graph 1-hop expansion outperform depth-matched and same-file lexical controls? (RQ3) Where is accuracy lost across the pipeline? Confirmatory contrasts are locked upfront as PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$); other analyses are exploratory.

Contributions include: (1) matched controls isolating graph topology from pool cardinality; (2) four-stage loss decomposition separating upstream boundary loss from compaction; (3) direct measurement of distractor displacement under pointwise scoring; (4) continuous dose-response diagnostics of spectral diffusion; and (5) open release of 19,298 scored evaluations, graphs, and library.

---

## 2. Related Work

- **Graphs and agents**: Aider [1], RepoGraph [2], CodexGraph [3], LocAgent [4], and AutoCodeRover [5] equip agents with graph tools; Agentless [6] resolves ~32% of Lite without an agent loop. These use large models; I test whether graph expansion assists a 2B model in one pass.
- **Random-walk retrieval**: Approximate PPR [7], APPNP [8], and rooted PageRank [9] motivate diffusion; HippoRAG [10] rescales teleport priors. Perron divides converged walks by static stationary baselines.
- **Reranking**: Pointwise scorers (MonoT5 [13]) evaluate candidates independently, accumulating distractors as pool size expands ($\mathbb{E}[X] = \sum (1 - F_{\text{dist}}(s^*))$). Listwise rerankers (RankGPT [14], Setwise [15]) enable cross-candidate attention, mitigating distractor dominance.
- **Empirical localization**: Pearson et al. [11] showed synthetic fault results rarely transfer to real code, motivating authentic SWE-bench tasks [12]. Matched controls isolate topology from pool breadth.

---

## 3. Study Design

- **Benchmark & Strata**: 344 callable, AST-parsable tasks of SWE-bench Verified [12] across 12 repositories (excluding 93 Lite overlaps, 56 AST drift, 7 non-callable). Strata: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), Stratum C (Un-Named, $N=131$, 38.1%), annotated via deterministic regex (`perron.analysis.strata`, 50-task audit 100% agreement, CC0-1.0). Hit@$k$ requires $\ge 1$ gold function in top-$k$. Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) were frozen upfront.
- **Perron Engine**: Directed multigraphs index call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10) in mmap CSR arrays. Power iteration converges under spectral bound $|\lambda_2| \le \beta = 0.85$ [16] to $10^{-6}$ in $\le 90$ steps ($<5$ ms, 144 MB RAM). AST breadcrumbs average 283.8 tokens/task (median 264), preserving 91.9% reasoning headroom under Gemma 4's 3,480-token budget. Restarts to $\mathbf{p}_0$ match normalized leaky resolvents $(I - \beta T)^{-1}$ ($L_1 < 2 \times 10^{-8}$). Decoupled Specificity $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$ enforces $\text{HSI@10} = 1.0 - (|\text{top-10} \cap \mathcal{H}_{25}| / 10)$. Softmax prior ($\tau=0.15$) uses top-25 seeds.
- **Pools and Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods, pruning test nodes and top-25 hubs. Pool size is $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5) for 1-hop out-neighbours $E_t$. Arms: **B0** (unscored BM25), **A1** (seeds $M=10$), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion). Pruning yields net $+2$ tasks over raw BFS ($199 + 16 - 14 = 201$).
- **Scorer**: Prompts average 423 tokens. Frozen `gemma-4-e2b-it` (FP16, thinking disabled) scores candidates via log-odds: $s(t, v) = \operatorname{logsumexp}(z[Y]) - \operatorname{logsumexp}(z[N])$ (19,298 passes, left-padded, `logits_to_keep=1`).
- **Hash-Committed Protocol**: Retrieval was committed at [`6908e78`](https://github.com/yvliet/perron/commit/6908e78) (October 5, 2026); Gemma reranking was locked in two stages (SHA-256 `96af1621`, `dbbf2920`, commit [`29e0066`](https://github.com/yvliet/perron/commit/29e0066), October 6, 2026). Confirmatory contrasts are PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5, tested via exact two-sided McNemar with step-down Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$).
- **Reader Test**: On $N=100$ tasks with BM25 coverage, 5-candidate contexts compare BM25 top-5 (D0), graph top-5 (D1), same-file top-5 (D2), and oracle with randomized position (D3). Conversion is EditLoc@1 divided by ContextHit@5 (descriptive).

---

## 4. Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5, Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, $\Delta = -0.6$ pp, Holm $p=1.000$) and 40.4% Acc@5, and standard PPR hits 50.0% (172/344). BM25 attains 99.4% HSI@10 natively. On Stratum C (131 un-named tasks), BM25 hits 28 and Perron 27 (RRF fusion: 174/344, 50.6%), showing no detectable diffusion lift.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. One hop covers 57.8% (median 27 nodes); two hops reach 66.9% (median 324). Same-file containment reaches 66.9%, with 17 graph-only and 17 file-only tasks cancelling symmetrically. Empirical 1-hop reachability exceeds degree-preserving null graphs (63.8%) by +3.02 pp (CI [−0.42, +6.41]). Reachability gains are modest and degree-driven.

### 4.3 Gemma 4 Pointwise Reranking (RQ2)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |

- **Confirmatory (Acc@5)**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Both span zero, ruling out gains $> +4.4$ pp ($15$ tasks).
- **Exploratory**: $A_4 - A_1$: $+0.87$ pp (95% CI [−1.92, +3.66], discordant 7/4, $p=0.549$; gross 7/4 split: 6 graph + 1 seed promotion vs 4 displaced, net +3 tasks, $2.45\times$ compute). Null control $A_4 - A_5$: $+1.16$ pp (95% CI [−0.58, +2.91], discordant 7/3, $p=0.344$). Diffusion $A_4 - A_6$: $+1.45$ pp (CI [+0.00, +3.20], discordant 7/2, $p=0.180$). BM25 (**B0**) maintains higher MRR@50 (0.344 vs 0.341) and Acc@1 (25.3% vs 24.7%). Non-Django ($N=114$) $A_4 - A_2$ is $+2.17$ pp ($p=0.227$).
- **Model Sensitivity & Positive Controls**: Within-pool ROC-AUC of Gemma log-odds separating gold from distractors is 0.718 across all 176 recalled tasks (95% CI [0.682, 0.754]), confirming genuine discriminative ability above chance (0.50). Reciprocal Rank Fusion (RRF, $k_{\text{rrf}}=60$) combining BM25 prior rank with Gemma score reaches 46.51% (160/344), outperforming both raw BM25 (44.77%) and pure Gemma reranking (46.22%). The flatline is driven by independent pointwise calibration noise across expanded candidate pools rather than model blindness.

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Distractor Accumulation**: Truncating pools at $M \in [10, 50]$ shows adding depth BM25 candidates causes distractor accumulation under pointwise scoring ($\mathbb{E}[X \mid s^*] = \sum (1 - F_{\text{dist}}(s^*))$): Acc@5 drops by 4 tasks (46.2% to 45.1%, $-$1.16 pp, $p=0.455$) despite $+7.27$ pp recall. Decomposing candidate pools reveals why: of 25 newly reachable targets entering $A_4$, 19 enter at rank $\ge 6$ (mean 11.2 outranking distractors) and never reach top-5, while 6 convert into top-5 hits ($6 + 19 = 25$). Combined with 1 internal seed promotion against 4 displaced seed hits, gross promotion yields $\Delta = (6_{\text{graph}} + 1_{\text{seed}}) - 4_{\text{displaced}} = +3$ net tasks ($159 + 7 - 4 = 162$, +0.87 pp, $p=0.549$), as per-distractor outrank drops from 16.6% (1.44/9) in $A_1$ to 7.9% (1.72/21.8) in $A_4$. Appended targets enter too far down the tail under pointwise scoring to convert into localization gains.
- **Stratum C Breakdown ($N=131$ Un-Named)**: For un-named targets, $A_4$ reaches 12.2% Acc@5 (16/131) vs 8.4% for $A_2$ (11/131, $+3.82$ pp, exact McNemar $p=0.1250, n_{10}=6, n_{01}=1$). However, broader pools are penalized relative to unscored BM25 (B0, 15.3%, 20/131, $+6.87$ pp vs $A_2$, McNemar $p=0.0117, n_{10}=10, n_{01}=1$) and seeds alone (A1, 13.7%, 18/131, $+5.34$ pp, $p=0.0156, n_{10}=7, n_{01}=0$). Expanding low-confidence candidates in $A_2$ actively introduces distractor harm ($p=0.0117$). The 18 discordant tasks split symmetrically (9 vs 10 in C, 0 vs 1 in B, 9 vs 7 in A).
- **Four-Stage Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed ceiling: 51.2% (176) $\to$ (3) Graph ceiling: 58.4% (201, $+7.3$ pp) $\to$ (4) Top-5: 47.1% (162). Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$ ($1.2\times$ at $k=1$, $6.8\times$ at $k=10$): expansion adds 7.3 pp recall, but Gemma 4 converts only $\sim 1$ pp due to the 7/4 gross displacement split.

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, candidate 1 yields 33.0% EditLoc@1 (matching rank-1 baseline). In D3, gold is identified in 74.0% of tasks (Wilson CI [64.6, 81.6]%). While the reader misses 26% of oracle cases, retrieval supplies gold in only 35–38%, forming the primary bottleneck.

### 4.6 Diffusion Diagnostics & Continuous Dose-Response (Post-Hoc)
On SWE-bench Lite ($N=237$), exploratory factorial diagnosis shows standard PPR collapses to 10.1% Acc@10 under diffuse linear priors, with test subgraphs absorbing 32.5% of top-10 slots; top-25 softmax prior ($\tau=0.15$) restores 48.5% (BM25: 46.0%). Specificity damping adds +24.1 pp under diffuse priors, but −0.8 pp under softmax. A post-hoc continuous resolvent benchmark across 344 tasks executed after protocol lock (63,984 SpMM solves across 186 prior configurations, setting zero hyperparameters used in Tables 1–5) reveals a significant quadratic relationship with prior support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($p < 0.001$, GLMM $R^2 = 0.197$), peaking at $n_{\text{eff}}^* \approx 8.8$ where it ties BM25 (50.9%). Specificity damping lifts HSI from 90.8% to 99.5%: a safety net for diffuse priors, not an accuracy booster.

---

## 5. Discussion

- **Empirical verdict**: Point estimates nominally favor graph expansion (+0.87 to +2.03 pp), but differences are statistically null-compatible under a frozen 2B scorer. Bounded CIs rule out gains $> +4.4$ pp ($15$ tasks) over depth-matched lexical reranking (PC1).
- **Distractor accumulation mechanism**: Pointwise calibration noise penalizes pool breadth: expanding depth BM25 from $M=10$ to $M=50$ shifts Acc@5 down by 4 tasks ($-1.16$ pp, $p=0.455$) despite $+7.27$ pp recall, while graph targets appended at low ranks fail to convert. Cross-candidate attention via listwise ranking or scaled parameter families remains the primary open direction to unlock candidate pool breadth.
- **Practical takeaway**: Under tight context budgets, retrieval engineering is more safely invested in lexical precision and AST breadcrumbs: graph expansion required 4,988 extra passes ($2.45\times$ compute) for +0.87 pp (gross 7/4 split, $p=0.549$).
- **Reusable audit framework**: Any retrieval stage feeding a local model can be audited similarly: report pool recall, evaluate matched lexical and structural controls under the identical scorer, and decompose upstream boundary loss from ranking compaction.
- **Limitations & Protocol**: (1) Python static ASTs; (2) single 2B pointwise model; (3) 1-hop pools capped at $M \le 50$; (4) function-level targets without patch synthesis; (5) descriptive reader test ($N=100$); (6) two confirmatory contrasts. B0 Acc@10 is 176/344 in reranking vs 175/344 in retrieval (`astropy-14182` tie-break; `deviations.md`).

---

## 6. Released Resources

- **`scores.parquet`**: 19,298 scored candidate evaluations with per-arm rankings for all 344 tasks (Apache 2.0).
- **`perron-core` v0.3.0 (PyPI) and `perron-mcp`**: JSON-RPC stdio server with context-retrieval and AST-breadcrumb tools.
- **`swebench-codegraphs`**: Precomputed CSR graphs for all 12 canonical SWE-bench repositories (covering both Lite and Verified tasks), with builder script at `scripts/export_swebench_graphs.py`.
- **`swebench-strata-annotations`**: 800 lexical-accessibility annotations across SWE-bench repositories (344 Verified evaluation tasks, CC0-1.0).
- **Reproduction notebook**: Interactive Kaggle workstation running on free Dual Tesla T4 GPUs (14.56 GiB VRAM) in 259.8s (~4.3 min), validating 344-task context packing and Stratum C positive control at [https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation).

---

## 7. Conclusion

On 344 held-out tasks of SWE-bench Verified, 1-hop graph expansion yields no statistically detectable advantage over matched lexical controls under a frozen Gemma 4 scorer, nor does it expand reachability beyond lexical depth. While adding 7.3 pp pool recall over seeds, the model converts only 0.87 pp into top-5 accuracy (+3 tasks, $p=0.549$) due to the 7/4 gross displacement split. Candidate truncation curves confirm that expanding pool breadth under pointwise scoring yields flat localization via distractor accumulation; on un-named tasks, expanding candidate pools reduces accuracy relative to unscored BM25 ($p=0.0117$). Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ at $k=5$. Null-compatible estimates (+0.87 to +2.03 pp) do not justify a $2.45\times$ compute penalty (4,988 extra passes). Local developer agents under tight context budgets are more reliably guided toward high-precision lexical retrieval and compact AST breadcrumbs. Scored data, engine code, and precomputed graphs are released under Apache 2.0.

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
