# Graph Expansion Adds Little to Lexical Retrieval for Small-LLM Bug Localization
**A hash-committed diagnostic study with Gemma 4 on SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

---

## Abstract

Local developer agents must localize software defects within a working context of a few thousand tokens. Graph walks over call graphs are widely used to reach symbols the issue never names. I test whether they help a small local model. On SWE-bench Verified ($N=344$ callable tasks), under a hash-committed protocol, a frozen `gemma-4-e2b-it` scores candidate pools pointwise. Graph 1-hop expansion reaches 47.1% Acc@5, against 45.1% for depth-matched BM25 (+2.03 pp, 95% CI [−0.29, +4.36], Holm $p=0.287$) and 45.6% for same-file expansion (+1.45 pp, Holm $p=0.302$). Both pools recall gold in 201/344 tasks (58.4%), with 18 discordant tasks each way. Compared to reranking seeds alone (46.2%), graph expansion adds only 3 tasks (+0.87 pp, $p=0.549$). In Stratum C ($N=131$ un-named tasks), graph expansion achieves 12.2% Acc@5 vs 8.4% for depth BM25 (+3.82 pp, $p=0.125$), but neither beats seeds alone (13.7%) or unscored BM25 (15.3%). Most loss occurs before scoring: 41.6 pp between task set and pool, 11.3 pp between pool and top-5. In an $N=100$ reader test, the model finds gold in 74% of oracle contexts, while retrieval supplies it in only 35–38% of tasks. The paired 95% bootstrap CI [−0.29, +4.36] pp spans zero, ruling out large gains (> 4.4 pp) while remaining consistent with small or null differences. I release 19,298 scored candidate evaluations, the library, and precomputed graphs under Apache 2.0.

---

## 1. Introduction

Local agents must localize bugs within a few thousand tokens, without cloud inference. Lexical search finds symbols named in issue text. Graph retrieval aims to reach un-named symbols via call edges. The working assumption requires two conditions: traversal must place missing gold targets into candidate pools, and a small model must rank them above added distractors.

I test both conditions on SWE-bench Verified with Perron, a memory-mapped CSR graph engine, and a frozen Gemma 4 scorer. Three questions guide the study:

- **RQ1**: Does Personalized PageRank (PPR) diffusion, with or without hub damping, beat BM25 at function-level retrieval?
- **RQ2**: Under one frozen scorer, does graph 1-hop expansion beat lexical-depth and same-file pools of equal size?
- **RQ3**: At which pipeline stage is localization accuracy lost?

### Contributions
- **A matched-control design**: depth-matched BM25, same-file, and degree-preserving null-graph pools scored by the same model, with two confirmatory contrasts fixed upfront.
- **A loss decomposition**: separating pool ceiling, rank compaction, and reader error; 7.3 pp of added oracle recall converts to $\sim 1$ pp of top-5 accuracy.
- **Diffusion diagnostics**: analyzing why diffusion collapses under diffuse priors on code graphs, and what specificity damping repairs.
- **Released resources**: open-source scored candidate evaluations, engine library, and precomputed graphs.

Only two contrasts (PC1, PC2) are confirmatory. Every other analysis is labelled exploratory.

---

## 2. Related Work

- **Graphs and agents for localization**: Aider [1] ranks tags with PageRank. RepoGraph [2], CodexGraph [3], and LocAgent [4] give agents graph tools; AutoCodeRover [5] navigates ASTs iteratively. Agentless [6] resolves 27.3% of SWE-bench Lite with hierarchical localization and no agent loop, showing simple pipelines are strong baselines. These systems use large hosted models; I evaluate whether graph expansion assists a 2B model in one scoring pass.
- **Random-walk retrieval**: Approximate PPR [7], APPNP [8], and rooted PageRank [9] motivate diffusion. HippoRAG [10] rescales teleport priors by inverse passage frequency. Perron divides converged walks by static stationary vectors, leaving matrices unmutated.
- **Pointwise versus listwise reranking**: Pointwise scorers (MonoT5 [13]) evaluate candidates independently, accumulating distractors as pool size grows ($\mathbb{E}[X] = \sum (1 - F_{\text{dist}}(s^*))$). Listwise models (RankT5 [14]) permit cross-candidate competition, mitigating distractor dominance.
- **Evaluation**: Pearson et al. [11] showed that findings on artificial faults rarely transfer to real ones, motivating authentic SWE-bench tasks [12]. Matched-pool controls isolate what graph structure contributes beyond pool size.

---

## 3. Study Design

- **Benchmark & Strata**: SWE-bench Verified [12] contains 500 tasks across 12 repositories. Excluding 93 Lite overlap tasks leaves 407 held-out tasks; removing 56 AST drift exclusions (legacy syntax incompatibilities in base commits parsed identically across arms) and 7 non-callable patches leaves 344 callable tasks (all Python). Tasks are stratified by lexical visibility: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), and Stratum C (Un-Named, $N=131$, 38.1%), annotated via deterministic regex (`perron.analysis.strata`, CC0-1.0). Hyperparameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) were frozen prior to evaluation.
- **Perron Engine**: Directed multigraphs store call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10) in mmap CSR arrays. Power iteration converges at rate $\beta$ in $\le 90$ steps to $10^{-6}$ (1.89 ms Requests, 5.10 ms SymPy, 144 MB RAM). Decoupled Specificity scores $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$, damping against stationary vector $\pi_g$ without matrix mutation. The library prior uses softmax ($\tau=0.15$) over top-25 BM25 candidates; §4.1 uses a BM25-proportional prior.
- **Pools and Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods and removes test nodes and the 25 highest in-degree hubs. $E_t$ denotes 1-hop out-neighbours of $S$ passing $F$. Pool size is $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5).
  - **B0**: BM25 order, unscored
  - **A1**: Seeds only ($M=10$)
  - **A2**: Seeds plus next BM25 candidates ($M_t$)
  - **A3**: Seeds plus same-file symbols ($M_t$)
  - **A4**: Seeds plus $E_t$ ordered by 1-step walk adjacency ($M_t$)
  - **A5**: Degree-preserving null graph
  - **A6**: Perron diffusion pool
  - *Pool Recall Discrepancy (201 vs 199 tasks)*: Masking test seeds rescues 16 source tasks, while Filter $F$ hub pruning drops 14 tasks, yielding net $+2$ over 1-hop BFS ($199 + 16 - 14 = 201$).
- **Scorer**: Pointwise prompt contexts average 423 tokens (issue text, instructions, and $<100$-token AST breadcrumbs). A frozen `gemma-4-e2b-it` (FP16, thinking disabled) scores candidates via affirmative minus negative token logsumexp logits. Prompt was chosen on disjoint development tasks ($N=100$). Scoring required 19,298 forward passes (2.37 GiB peak VRAM, 3.5 s TTFT on Tesla T4).
- **Hash-Committed Protocol & Statistics**: Retrieval was committed at `6908e78`; Gemma reranking was locked in two hashed stages (SHA-256 `96af1621`, `dbbf2920`, commit `29e0066`). Confirmatory contrasts are PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5, tested via exact two-sided McNemar with step-down Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$). Everything else is exploratory.
- **Reader Test**: On $N=100$ tasks stratified by repository size with BM25 coverage, a 5-candidate context is built from BM25 top-5 (D0), graph top-5 (D1), same-file top-5 (D2), or an oracle with the gold target at a random position (D3). Conversion is EditLoc@1 divided by ContextHit@5 (descriptive).

---

## 4. Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
At Function Acc@10, BM25 hits 175/344 (50.9%), RRF fusion 174, Perron Static 173 (50.3%), and standard PPR 172. Perron minus BM25 is −0.6 pp (95% CI [−3.5, +2.3]). BM25 leads at Acc@5 (45.1% vs 40.4%) and sustains 99.4% Hub Suppression Index (HSI@10) without graph machinery. On Stratum C (131 un-named tasks), BM25 hits 28 and Perron 27; fusion yields 174/344 (50.6%). Diffusion provides no detectable lexical lift over BM25.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. One hop covers 57.8% (median 27 nodes); two hops reach 66.9% (median 324). Same-file containment also reaches 66.9%, with 17 graph-only and 17 file-only tasks cancelling symmetrically. Null graphs reach 63.8% (+3.02 pp, CI [−0.42, +6.41]). Reachability gains are modest, degree-driven, and introduce dozens of distractors.

### 4.3 Gemma 4 Pointwise Reranking (RQ2)

| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |

- **Confirmatory Contrasts on Acc@5**:
  - **PC1 ($A_4 - A_2$)**: $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$).
  - **PC2 ($A_4 - A_3$)**: $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$).
  - Both contrasts are null-compatible (CIs span zero, Holm $p > 0.05$), excluding large effects (> 4.4 pp).
- **Exploratory Contrasts**:
  - **$A_4 - A_1$**: $+0.87$ pp (+3 tasks, discordant 7/4, $p=0.549$), requiring 4,988 extra forward passes.
  - **$A_4 - A_5$ (Null Graph Control)**: $+1.16$ pp (95% CI [−0.58, +2.91], discordant 7/3, $p=0.344$), indicating empirical graph topology adds no detectable gain over degree-matched random graphs.
  - **$A_4 - A_6$ (Diffusion Pool)**: $+1.45$ pp (95% CI [+0.00, +3.20], discordant 7/2, $p=0.180$).
  - Unscored BM25 (**B0**) maintains nominally higher MRR@50 (0.344 vs 0.341, Wilcoxon $p=0.72$) and top-1 accuracy (25.3% vs 24.7%) than $A_4$.
  - Excluding Django ($N=230$), $A_4 - A_2$ is $+2.17$ pp ($p=0.227$).

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Distractor Accumulation**: Truncating matched pools at $M \in [10, 50]$ (mean 24.5 candidates) shows that adding depth BM25 candidates is consistent with distractor accumulation under pointwise scoring ($\mathbb{E}[X \mid s^*] = \sum (1 - F_{\text{dist}}(s^*))$): Acc@5 drops by 4 tasks (46.2% to 45.1%, $-$1.16 pp, exact McNemar $p=0.455$) despite adding +7.27 pp of recall. In $A_4$, Acc@5 plateaus (+3 tasks over seeds, +0.87 pp, $p=0.549$).
- **Stratum C Breakdown ($N=131$ Un-Named Tasks)**: For un-named targets, $A_4$ reaches 12.2% Acc@5 (16/131) vs 8.4% for $A_2$ (11/131, $+3.82$ pp, $p=0.125$), but neither beats seeds alone (13.7%, 18/131) or unscored BM25 (15.3%, 20/131). The 18 graph-only and 18 lexical-only discordant recall tasks split symmetrically across strata (9 vs 10 in C, 0 vs 1 in B, 9 vs 7 in A), indicating graph expansion does not selectively target un-named bugs over lexical search.
- **Four-Stage Loss Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed pool ceiling: 51.2% (176, loss 48.8 pp) $\to$ (3) Graph pool ceiling: 58.4% (201, $+7.3$ pp, loss 41.6 pp) $\to$ (4) Scored top-5: 47.1% (162, loss 11.3 pp). Upstream boundary loss exceeds compaction loss by $3.68\times$ ($41.6 / 11.3$): expansion adds 7.3 pp oracle recall, but Gemma 4 converts only $\sim 1$ pp into ranked accuracy.

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, choosing candidate 1 yields 33.0% EditLoc@1 (rank-1 prior). In D3, gold is found in 74.0% of tasks (Wilson CI [64.6, 81.6]%; 67.5% chance-adjusted). While the reader misses a quarter of oracle cases, retrieval supplies gold in only 35–38% of tasks, limiting end-to-end performance.

### 4.6 Diffusion Diagnostics on SWE-bench Lite (Exploratory, N=237)
Under a diffuse linear prior, standard PPR collapses to 10.1% Acc@10; test subgraphs absorb 32.5% of top-10 slots. Top-25 softmax prior ($\tau=0.15$) on sanitized graphs restores 48.5%, versus 46.0% for BM25. Specificity damping adds +24.1 pp under diffuse priors, but −0.8 pp under softmax. Across 186 configurations, accuracy tracks effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ (peaking near 8.8). Damping lifts HSI from 90.8% to 99.5%: a safeguard for diffuse priors, not an accuracy gain.

---

## 5. Discussion

- **What the evidence supports**: Point estimates nominally favor graph expansion (+0.87 to +2.03 pp), but differences are statistically null-compatible under a frozen 2B scorer. Bounded CIs rule out large gains (> 4.4 pp) while remaining consistent with small or null effects.
- **Distractor mechanism**: Truncation curves are consistent with distractor accumulation under pointwise scoring: expanding depth BM25 from $M=10$ to $M=50$ adds $+7.27$ pp recall but drops Acc@5 by 4 tasks ($-1.16$ pp, $p=0.455$). Because expected distractors scale as $\sum (1 - F_{\text{dist}}(s^*))$, pointwise scoring penalizes pool breadth; listwise ranking remains a promising hypothesis.
- **Practical implication**: At a 4,096-token budget with a 2B scorer, first-stage lexical precision and AST breadcrumbs are the safer investment: graph expansion required 4,988 extra passes for +0.87 pp (+3 tasks, $p=0.549$).
- **A reusable audit**: Any retrieval stage feeding a small model can be audited using the same framework: report pool oracle recall, add depth-matched and same-file controls under the same scorer, and decompose loss before and after ranking.
- **Limitations**: (1) Python-only ASTs missing dynamic dispatch. (2) Single 2B model, prompt, and pointwise scoring. (3) 1-hop pools capped at $M \le 50$. (4) Function-level targets on 344 tasks without patch synthesis. (5) Reader test is descriptive ($N=100$). (6) Only PC1/PC2 are confirmatory. (7) Edge weights and $\gamma$ were fixed globally.
- **Protocol disclosure**: B0 Acc@10 is 176/344 in reranking but 175/344 in retrieval: on `astropy__astropy-14182`, raw token ranking kept gold at rank 10 while AST deduplication moved it to 11. PC1/PC2 are unaffected; raw counts are recorded in `deviations.md`.

---

## 6. Released Resources

- **`scores.parquet`**: 19,298 scored candidate evaluations with per-arm rankings for all 344 tasks (Apache 2.0).
- **`perron-core` v0.3.0 (PyPI) and `perron-mcp`**: JSON-RPC stdio server with context-retrieval and AST-breadcrumb tools.
- **`swebench-lite-codegraphs`**: Precomputed CSR graphs for the 12 SWE-bench Lite repositories.
- **`swebench-strata-annotations`**: 800 lexical-accessibility annotations (CC0-1.0).
- **Reproduction notebook**: [https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation)

---

## 7. Conclusion

On SWE-bench Verified, graph 1-hop expansion does not demonstrate a statistically detectable advantage over matched lexical pools under a frozen Gemma 4 scorer, nor does it widen candidate reachability beyond lexical depth. While graph expansion adds 7.3 pp of pool recall over seeds alone, the small model converts only $\sim 1$ pp into top-5 accuracy. Candidate pool truncation curves show that expanding pool breadth under pointwise scoring yields flat or declining localization, consistent with distractor accumulation; on un-named tasks, neither expansion beats seeds alone or unscored BM25. Upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$. While point estimates nominally favor graph expansion (+0.87 to +2.03 pp), gains are statistically undetectable in 344 tasks and do not justify a 5x compute explosion. Local agent pipelines under tight compute budgets are more safely guided toward high-precision lexical retrieval and compact AST breadcrumbs, leaving listwise reranking as an open question for future study. Scores, code, and precomputed graphs are released under Apache 2.0.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph: Enhancing AI software engineering with repository-level code graph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph: Bridging LLMs and code repositories via code graph databases.* NAACL 2025.
- [4] Z. Chen et al. *LocAgent: Graph-guided LLM agents for code localization.* ACL 2025.
- [5] Y. Zhang et al. *AutoCodeRover: Autonomous program improvement.* ISSTA 2024.
- [6] C. S. Xia et al. *Agentless: Demystifying LLM-based software engineering agents.* FSE 2025.
- [7] R. Andersen, F. Chung, K. Lang. *Local graph partitioning using PageRank vectors.* FOCS 2006.
- [8] J. Klicpera et al. *Predict then propagate: Combining neural networks with personalized PageRank.* ICLR 2019.
- [9] D. Liben-Nowell, J. Kleinberg. *The link-prediction problem for social networks.* JASIST 2007.
- [10] B. J. Gutiérrez et al. *HippoRAG: Neurobiologically inspired long-term memory for LLMs.* NeurIPS 2024.
- [11] S. Pearson et al. *Evaluating and improving fault localization.* ICSE 2017.
- [12] C. E. Jimenez et al. *SWE-bench: Can language models resolve real-world GitHub issues?* ICLR 2024.
- [13] R. Nogueira et al. *Document ranking with a pretrained sequence-to-sequence model.* Findings of EMNLP 2020.
- [14] H. Zhuang et al. *RankT5: Fine-tuning T5 for text ranking with ranking losses.* SIGIR 2023.
