# Graph Expansion Adds Little to Lexical Retrieval for Small-LLM Bug Localization
**A hash-committed diagnostic study with Gemma 4 on SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

---

## Abstract

Local developer agents must localize software defects within tight token budgets without cloud inference. Graph walks over call graphs aim to reach symbols issue descriptions omit. I test whether 1-hop graph expansion assists a compact local model (`gemma-4-e2b-it`) on SWE-bench Verified ($N=344$ callable tasks). Under a hash-committed protocol, graph 1-hop expansion reaches 47.1% Acc@5 vs 45.1% for depth-matched BM25 (+2.03 pp, 95% CI [−0.29, +4.36], Holm $p=0.287$) and 45.6% for same-file expansion (+1.45 pp, Holm $p=0.302$). Both pools recall gold in 201/344 tasks (58.4%), with 18 discordant tasks each way. Compared to reranking seeds alone (46.2%), graph expansion rescues 7 newly reachable targets into top-5 while distractors displace 4 seed hits, producing a gross 7/4 split and net gain of only 3 tasks (+0.87 pp, $p=0.549$). In Stratum C ($N=131$ un-named tasks), graph expansion achieves 12.2% Acc@5 vs 8.4% for depth BM25 (+3.82 pp, $p=0.125$), but neither beats seeds alone (13.7%) or unscored BM25 (15.3%). At $k=5$, retrieval boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ (spanning $1.2\times$ at $k=1$ to $6.8\times$ at $k=10$). In an $N=100$ reader test, the model identifies gold in 74% of oracle contexts, while retrieval supplies it in only 35–38% of tasks. The paired 95% bootstrap CI [−0.29, +4.36] pp spans zero, ruling out gains larger than $+4.4$ pp ($15$ tasks) over depth-matched lexical reranking (PC1). I release 19,298 scored candidate evaluations, the library, and precomputed graphs under Apache 2.0.

---

## 1. Introduction

Local agents must localize software defects within tight token budgets without cloud inference. Lexical search locates symbols named in issue descriptions. Graph retrieval traverses call edges to reach un-named symbols. This approach relies on two conditions: graph traversal must place omitted targets into candidate pools, and a small model must rank them above added distractors.

I evaluate both assumptions on SWE-bench Verified using Perron, a memory-mapped CSR graph engine, and a frozen Gemma 4 scorer, addressing three research questions:
- **RQ1**: Does Personalized PageRank (PPR) diffusion, with or without hub damping, outperform BM25 at function-level retrieval?
- **RQ2**: Under a frozen small scorer, does graph 1-hop expansion outperform lexical-depth and same-file pools of equal size?
- **RQ3**: At which pipeline stage is localization accuracy lost?

### Contributions
- **Matched-control design**: depth-matched BM25, same-file, and degree-preserving null-graph pools evaluated under identical scoring budgets, with two confirmatory contrasts locked upfront.
- **Four-stage loss decomposition**: separating boundary ceiling, pool reachability, rank compaction, and reader comprehension.
- **Diffusion diagnostics**: identifying why diffusion degrades under diffuse priors on code graphs, and demonstrating specificity damping mechanics.
- **Released open resources**: 19,298 scored candidate evaluations, high-throughput engine library, and precomputed repository graphs.

Only contrasts PC1 and PC2 are confirmatory; all other analyses are exploratory.

---

## 2. Related Work

- **Graphs and agents for localization**: Aider [1] ranks tags with PageRank. RepoGraph [2], CodexGraph [3], and LocAgent [4] give agents graph tools; AutoCodeRover [5] navigates ASTs iteratively. Agentless [6] resolves 27.3% of SWE-bench Lite with hierarchical localization and no agent loop, showing simple pipelines are strong baselines. These systems use large hosted models; I evaluate whether graph expansion assists a 2B model in one scoring pass.
- **Random-walk retrieval**: Approximate PPR [7], APPNP [8], and rooted PageRank [9] motivate diffusion. HippoRAG [10] rescales teleport priors by inverse passage frequency. Perron divides converged walks by static stationary vectors, leaving matrices unmutated.
- **Pointwise versus listwise reranking**: Pointwise scorers (MonoT5 [13]) evaluate candidates independently, accumulating distractors as pool size grows ($\mathbb{E}[X] = \sum (1 - F_{\text{dist}}(s^*))$). Listwise and setwise LLM rerankers (RankGPT [14], Setwise [15]) permit joint cross-candidate attention, mitigating the distractor dominance inherent to independent pointwise scoring.
- **Empirical evaluation**: Pearson et al. [11] showed artificial fault findings rarely transfer to real code, motivating authentic SWE-bench tasks [12]. Matched controls isolate graph topology from pool breadth.

---

## 3. Study Design

- **Benchmark & Strata**: SWE-bench Verified [12] contains 500 tasks across 12 repositories. Excluding 93 Lite overlaps, 56 base-commit AST drift exclusions, and 7 non-callable patches yields 344 callable Python tasks. Tasks are stratified by lexical visibility: Stratum A (Named, $N=208$, 60.5%), Stratum B (Traceback, $N=5$, 1.5%), and Stratum C (Un-Named, $N=131$, 38.1%), annotated via deterministic regex (`perron.analysis.strata`, verified by manual spot-check audit across 50 randomly sampled tasks with 100% agreement, CC0-1.0). Hyperparameters ($eta=0.85, \gamma=0.70, 	au=0.15, K=25$) were frozen prior to evaluation.
- **Perron Engine**: Directed multigraphs index call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10) in mmap CSR arrays. Power iteration converges at rate $eta$ in $\le 90$ steps to $10^{-6}$ ($<5$ ms, 144 MB RAM). Decoupled Specificity scores $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$, damping against stationary vector $\pi_g$ The library prior uses softmax ($	au=0.15$) over top-25 BM25 seeds.
- **Pools and Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods and removes test nodes and the 25 highest in-degree hubs. $E_t$ denotes 1-hop out-neighbours of $S$ passing $F$. Pool size is $M_t = 10 + \min(|E_t|, 40) \in [10, 50]$ (mean 24.5).
  - **B0**: BM25 order, unscored
  - **A1**: Seeds only ($M=10$)
  - **A2**: Seeds plus next BM25 candidates ($M_t$)
  - **A3**: Seeds plus same-file symbols ($M_t$)
  - **A4**: Seeds plus $E_t$ ordered by 1-step walk adjacency ($M_t$)
  - **A5**: Degree-preserving null graph
  - **A6**: Perron diffusion pool
  - *Pool Discrepancy*: Masking test seeds rescues 16 source tasks while Filter $F$ hub pruning drops 14, yielding net $+2$ over 1-hop BFS ($199 + 16 - 14 = 201$).
- **Scorer**: Prompts average 423 tokens (issue text, instructions, $<100$-token AST breadcrumbs). A frozen `gemma-4-e2b-it` (FP16, thinking disabled) scores candidates via affirmative minus negative logsumexp logits (19,298 forward passes, 2.37 GiB peak VRAM on Tesla T4).
- **Hash-Committed Protocol**: Retrieval was committed and pushed publicly at `6908e78` (October 5, 2026); Gemma reranking was locked in two hashed stages (SHA-256 `96af1621`, `dbbf2920`, commit `29e0066`, October 6, 2026). Confirmatory contrasts are PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5, tested via exact two-sided McNemar with step-down Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$).
- **Reader Test**: On $N=100$ tasks stratified by repository size with BM25 coverage, a 5-candidate context is built from BM25 top-5 (D0), graph top-5 (D1), same-file top-5 (D2), or an oracle with the gold target at a random position (D3). Conversion is EditLoc@1 divided by ContextHit@5 (descriptive).

---

## 4. Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
On Stratum C (131 un-named tasks), BM25 hits 28 and Perron 27 (fusion: 174/344, 50.6%). Diffusion provides no detectable lexical lift.

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

- **Confirmatory Contrasts on Acc@5**:
  - **PC1 ($A_4 - A_2$)**: $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$).
  - **PC2 ($A_4 - A_3$)**: $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$).
  - Both contrasts are null-compatible (CIs span zero, Holm $p > 0.05$), excluding large gains (> 4.4 pp over depth-matched lexical reranking).
- **Exploratory Contrasts**:
  - **$A_4 - A_1$**: $+0.87$ pp (95% CI [−1.92, +3.66], discordant 7/4, $p=0.549$). Gemma 4 promotes 7 newly reachable targets into top-5 while distractors displace 4 seed hits (gross 7/4 split, net +3 tasks: $159+7-4=162$), requiring 4,988 extra forward passes (a $2.45\times$ compute increase over seeds).
  - **$A_4 - A_5$ (Null Graph Control)**: $+1.16$ pp (95% CI [−0.58, +2.91], discordant 7/3, $p=0.344$), indicating empirical graph topology adds no detectable gain over degree-matched random graphs.
  - **$A_4 - A_6$ (Diffusion Pool)**: $+1.45$ pp (95% CI [+0.00, +3.20], discordant 7/2, $p=0.180$).
  - Unscored BM25 (**B0**) maintains nominally higher MRR@50 (0.344 vs 0.341) and top-1 accuracy (25.3% vs 24.7%) than $A_4$.
  - Excluding Django ($N=230$), $A_4 - A_2$ is $+2.17$ pp ($p=0.227$).

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 54.1% | 56.1% | 57.6% | 58.4% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.9% | 45.1% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.9% | 55.5% | 57.6% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.8% | **47.1%** | 46.5% | **47.1%** |

- **Distractor Accumulation**: Truncating matched pools at $M \in [10, 50]$ (mean 24.5 candidates) shows that adding depth BM25 candidates is consistent with distractor accumulation under pointwise scoring ($\mathbb{E}[X \mid s^*] = \sum (1 - F_{\text{dist}}(s^*))$): Acc@5 drops by 4 tasks (46.2% to 45.1%, $-$1.16 pp, exact McNemar $p=0.455$) despite adding +7.27 pp of recall. In $A_4$, Acc@5 plateaus (+3 tasks over seeds, +0.87 pp, $p=0.549$).
- **Stratum C Breakdown ($N=131$ Un-Named Tasks)**: For un-named targets, $A_4$ reaches 12.2% Acc@5 (16/131) vs 8.4% for $A_2$ (11/131, $+3.82$ pp, $p=0.125$), but neither beats seeds alone (13.7%, 18/131) or unscored BM25 (15.3%, 20/131). The 18 discordant recall tasks split symmetrically across strata (9 vs 10 in C, 0 vs 1 in B, 9 vs 7 in A), confirming graph expansion does not selectively target un-named bugs.
- **Four-Stage Loss Funnel**: (1) Benchmark: 100% (344) $\to$ (2) Seed pool ceiling: 51.2% (176, loss 48.8 pp) $\to$ (3) Graph pool ceiling: 58.4% (201, $+7.3$ pp, loss 41.6 pp) $\to$ (4) Scored top-5: 47.1% (162, loss 11.3 pp). At $k=5$, upstream boundary loss exceeds compaction loss by $3.68\times$ ($41.6 / 11.3$, spanning $1.2\times$ at $k=1$ to $6.8\times$ at $k=10$): expansion adds 7.3 pp oracle recall, but Gemma 4 converts only $\sim 1$ pp into ranked accuracy due to the 7/4 gross displacement split.

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, choosing candidate 1 yields 33.0% EditLoc@1 (rank-1 prior). In D3, gold is found in 74.0% of tasks (Wilson CI [64.6, 81.6]%). While the reader misses a quarter of oracle cases, retrieval supplies gold in only 35–38% of tasks, limiting end-to-end performance.

### 4.6 Diffusion Diagnostics on SWE-bench Lite (Exploratory, N=237)
Under diffuse linear priors, standard PPR collapses to 10.1% Acc@10; test subgraphs absorb 32.5% of top-10 slots. Top-25 softmax prior ($\tau=0.15$) restores 48.5% (BM25: 46.0%). Specificity damping adds +24.1 pp under diffuse priors, but −0.8 pp under softmax. Across 186 configurations, accuracy tracks effective support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ (peaking near 8.8). Damping lifts HSI from 90.8% to 99.5%: a safeguard for diffuse priors, not an accuracy gain.

---

## 5. Discussion

- **What the evidence supports**: Point estimates nominally favor graph expansion (+0.87 to +2.03 pp), but differences are statistically null-compatible under a frozen 2B scorer. Bounded CIs rule out gains larger than $+4.4$ pp ($15$ tasks) over depth-matched lexical reranking while remaining consistent with small or null effects.
- **Distractor mechanism**: Truncation curves are consistent with distractor accumulation under pointwise scoring: expanding depth BM25 from $M=10$ to $M=50$ adds $+7.27$ pp recall but drops Acc@5 by 4 tasks ($-1.16$ pp, $p=0.455$). Because expected distractors scale as $\sum (1 - F_{\text{dist}}(s^*))$, pointwise scoring penalizes pool breadth; listwise ranking remains a promising hypothesis.
- **Practical implication**: Under a 4,096-token budget with a 2B scorer, lexical precision and AST breadcrumbs are the safer investment: graph expansion required 4,988 extra passes ($2.45\times$ compute) for +0.87 pp (+3 tasks net, gross 7/4 split, $p=0.549$).
- **A reusable audit**: Any retrieval stage feeding a small model can be audited using the same framework: report pool oracle recall, add depth-matched and same-file controls under the same scorer, and decompose loss before and after ranking.
- **Limitations**: (1) Python-only static ASTs without dynamic dispatch. (2) Single 2B model under pointwise scoring. (3) 1-hop pools capped at $M \le 50$. (4) Function-level targets on 344 tasks without patch synthesis. (5) Descriptive reader test ($N=100$). (6) Two confirmatory contrasts (PC1/PC2).
- **Protocol disclosure**: B0 Acc@10 is 176/344 in reranking vs 175/344 in retrieval (`astropy__astropy-14182` tie-break; see `deviations.md`). Confirmatory PC1/PC2 are unaffected.

---

## 6. Released Resources

- **`scores.parquet`**: 19,298 scored candidate evaluations with per-arm rankings for all 344 tasks (Apache 2.0).
- **`perron-core` v0.3.0 (PyPI) and `perron-mcp`**: JSON-RPC stdio server with context-retrieval and AST-breadcrumb tools.
- **`swebench-lite-codegraphs`**: Precomputed CSR graphs for the 12 SWE-bench Lite repositories.
- **`swebench-strata-annotations`**: 800 lexical-accessibility annotations across SWE-bench repositories (344 Verified evaluation tasks, CC0-1.0).
- **Reproduction notebook**: [https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation)

---

## 7. Conclusion

On SWE-bench Verified, graph 1-hop expansion provides no statistically detectable benefit over matched lexical pools under a frozen Gemma 4 scorer, nor does it widen reachability beyond lexical depth. While graph expansion adds 7.3 pp of pool recall over seeds alone, Gemma 4 converts only $\sim 1$ pp into top-5 accuracy due to the 7/4 gross displacement split. Candidate pool truncation curves show that expanding pool breadth under pointwise scoring yields flat or declining localization, consistent with distractor accumulation; on un-named tasks, neither expansion outperforms seeds alone or unscored BM25. At $k=5$, upstream boundary loss (41.6 pp) exceeds downstream compaction loss (11.3 pp) by $3.68\times$ (spanning $1.2\times$ at $k=1$ to $6.8\times$ at $k=10$). The small gains (+0.87 to +2.03 pp) are statistically null-compatible and do not justify a $2.45\times$ compute penalty (4,988 extra forward passes). Local agent pipelines under tight inference budgets are more effectively guided toward high-precision lexical retrieval and compact AST breadcrumbs. Scored data, engine code, and precomputed graphs are released under Apache 2.0.

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
- [14] W. Sun et al. *Is ChatGPT good at search? Investigating large language models as re-ranking agents.* EMNLP 2023.
- [15] R. Pradeep et al. *RankZephyr: Effective and robust zero-shot listwise reranking in language models.* arXiv 2023.
