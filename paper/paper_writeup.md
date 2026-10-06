# Graph Expansion Adds Little to Lexical Retrieval for Small-LLM Bug Localization
**A pre-registered diagnostic study with Gemma 4 on SWE-bench Verified**

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet)) · **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) · **License**: Apache 2.0

---

## Abstract

Local developer agents must find the code an issue concerns within a prompt of a few thousand tokens. Graph walks over call graphs are widely used to reach symbols the issue never names. I test whether they help a small local model. On SWE-bench Verified ($N=344$ callable tasks), under a hash-committed protocol, a frozen `gemma-4-e2b-it` scores candidate pools pointwise. Graph 1-hop expansion reaches 47.1% Acc@5, against 45.1% for depth-matched BM25 (+2.03 pp, 95% CI [−0.29, +4.36], Holm $p=0.287$) and 45.6% for same-file expansion (+1.45 pp, Holm $p=0.302$). Both pools contain the gold function in 201/344 tasks (58.4%), with 18 discordant tasks each way. Expansion adds 7.3 pp of oracle recall over the seed pool, yet the scorer converts about 1 pp of it. Most loss occurs before scoring: 41.6 pp between task set and pool, 11.3 pp between pool and top-5. In a descriptive $N=100$ reader test, the model finds the gold target in 74% of oracle contexts, while retrieval supplies it in only 35–38% of tasks. The study is Python-only, uses one small model, and cannot exclude gains of about 4 pp. I release 19,298 scored candidate evaluations, the library, and precomputed graphs under Apache 2.0.

---

## 1. Introduction

Local agents must localize bugs inside a few thousand tokens, without cloud inference. Lexical search finds symbols the issue names. Graph retrieval, such as Aider's PageRank repository maps, RepoGraph, and LocAgent, aims to reach symbols it does not name by following call edges. The working assumption has two conditions: expansion must put gold targets into the pool that lexical search misses, and a small model must then rank them above the distractors the expansion also brings.

I test both conditions on SWE-bench Verified with Perron, a memory-mapped CSR graph engine, and a frozen Gemma 4 scorer. Three questions guide the study:

- **RQ1**: Does Personalized PageRank (PPR) diffusion, with or without hub damping, beat BM25 at function-level retrieval?
- **RQ2**: Under one frozen scorer, does graph 1-hop expansion beat lexical-depth and same-file pools of equal size?
- **RQ3**: At which pipeline stage is localization accuracy lost?

### Contributions
- **A matched-control design**: depth-matched BM25, same-file, and degree-preserving null-graph pools are scored by the same model, with two confirmatory contrasts fixed before the run.
- **A loss decomposition**: separating pool ceiling, rank compaction, and reader error, highlighting that 7.3 pp of added oracle recall converts to about 1 pp of ranked accuracy.
- **Diffusion diagnostics**: analyzing why diffusion fails under diffuse priors on code graphs, and what specificity damping repairs.
- **Released resources**: open-source scored candidate evaluations, engine code, and precomputed repository graphs.

Only two contrasts (PC1, PC2) are confirmatory. Every other analysis is labelled exploratory.

---

## 2. Related Work

- **Graphs and agents for localization**: Aider [1] ranks Tree-sitter tags with PageRank. RepoGraph [2], CodexGraph [3], and LocAgent [4] give multi-turn agents graph tools, and AutoCodeRover [5] navigates ASTs through iterative search. Agentless [6] resolves 27.3% of SWE-bench Lite with hierarchical localization and no agent loop, demonstrating that simple pipelines are strong baselines. These systems use large hosted models. I evaluate the question for a 2B local model with one scoring pass per candidate.
- **Random-walk retrieval**: Push-based approximate PPR [7], APPNP [8], and rooted PageRank for link prediction [9] motivate diffusion. HippoRAG [10] rescales the teleport prior by inverse passage frequency. Perron instead divides the converged walk by a repository-wide stationary vector, leaving the matrix static.
- **Evaluation**: Pearson et al. [11] showed that fault-localization results on artificial faults need not transfer to real ones, motivating authentic issues from SWE-bench [12]. I add matched-pool controls that isolate what the graph contributes beyond pool size.

---

## 3. Study Design

- **Benchmark**: SWE-bench Verified [12] contains 500 tasks across 12 repositories. Excluding 93 Lite overlap tasks leaves 407 held-out tasks. Excluding 56 commit AST file drifts and 7 module-level non-callable patches leaves 344 callable tasks with an identifiable function gold target (all Python). Hyperparameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) were fixed on development data and frozen before evaluation.
- **Perron Engine**: A directed multigraph carries call, inherit, import, and caller edges (weights 0.50, 0.25, 0.15, 0.10). CSR arrays are opened read-only via numpy memory-mapping, sharing pages without heap copies. Power iteration converges at rate $\beta$ in $\le 90$ iterations to $10^{-6}$ (1.89 ms on Requests, 5.10 ms on SymPy, 144 MB peak resident memory). Decoupled Specificity ranks nodes as $\pi_q(v) / (\pi_g(v)+\epsilon)^\gamma$, damping post-walk against stationary vector $\pi_g$ without matrix mutation. The library prior uses softmax ($\tau=0.15$) over top-25 BM25 candidates; §4.1 uses a BM25-proportional prior.
- **Pools and Arms**: Seeds $S$ are top-10 BM25 symbols. Filter $F$ keeps functions/methods and removes test nodes and the 25 highest in-degree hubs. $E_t$ denotes 1-hop out-neighbours of $S$ passing $F$. Pool size is $M_t = 10 + \min(|E_t|, 40)$, ranging 10–50 (mean 24.5).
  - **B0**: BM25 order, unscored
  - **A1**: Seeds only ($M=10$)
  - **A2**: Seeds plus next BM25 candidates ($M_t$)
  - **A3**: Seeds plus same-file symbols ($M_t$)
  - **A4**: Seeds plus $E_t$ ordered by 1-step walk adjacency ($M_t$)
  - **A5**: Degree-preserving null graph
  - **A6**: Perron diffusion pool
  - *Pool Recall Discrepancy (201 vs 199 tasks)*: A4 pool extraction reaches 201/344 recall (58.4%), exceeding raw 1-hop BFS reachability (199/344, 57.8%). Masking test-file seeds rescues 16 source tasks (e.g. `sympy-13372`), while Filter $F$ hub pruning drops 14 tasks (e.g. `sphinx-7590`), yielding net $+2$ ($199 + 16 - 14 = 201$).
- **Scorer**: Each candidate is scored pointwise in a 423-token context (issue text, instructions, and $<100$-token AST breadcrumbs). A frozen `gemma-4-e2b-it` (FP16, thinking disabled) scores candidates via logsumexp difference between affirmative and negative token logits. The prompt was chosen on disjoint development tasks ($N=100$). Scoring required 19,298 forward passes (2.37 GiB peak VRAM, 3.5 s time-to-first-token on Tesla T4).
- **Hash-Committed Protocol & Statistics**: Retrieval was committed at `6908e78`; Gemma reranking was locked in two hashed stages (SHA-256 `96af1621`, `dbbf2920`, commit `29e0066`). Confirmatory contrasts are PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5, tested via exact two-sided McNemar with step-down Holm correction. CIs use repository-stratified bootstrap ($B=10{,}000$). Empirical minimum detectable difference is $3.4\text{--}3.7$ pp at 80% power. Everything else is exploratory.
- **Reader Test**: On $N=100$ tasks stratified by repository size with BM25 coverage, a 5-candidate context is built from BM25 top-5 (D0), graph top-5 (D1), same-file top-5 (D2), or an oracle with the gold target at a random position (D3). Conversion is EditLoc@1 divided by ContextHit@5. This test is descriptive.

---

## 4. Results

### 4.1 Standalone Retrieval (RQ1, Hash-Committed)
At Function Acc@10, BM25 hits 175/344 (50.9%), RRF fusion 174, Perron Static 173 (50.3%), and standard PPR 172. Perron minus BM25 is −0.6 pp (95% CI [−3.5, +2.3]); non-inferiority against a −2.0 pp margin is inconclusive. BM25 leads at Acc@5 (45.1% vs 40.4%) and naturally sustains 99.4% Hub Suppression Index (HSI@10) without graph machinery. On Stratum C (131 tasks whose gold symbol the issue never mentions), BM25 hits 28 and Perron 27. Fusion does not beat BM25, and an oracle union reaches only 187/344 (54.4%). Diffusion adds no detectable lexical lift over BM25.

### 4.2 Reachability Versus Ranking (Exploratory)
Top-10 seeds cover 50.9% of gold targets. One hop covers 57.8% (median 27 candidates) and two hops 66.9% (median 324). Same-file containment also reaches 66.9%, with 17 graph-only and 17 file-only tasks cancelling in an exact symmetric discordance. Degree-preserving null graphs reach 63.8%; the +3.02 pp difference has CI [−0.42, +6.41]. Reachability gains are modest, largely explained by degree structure, and must be ranked against 27 to 324 candidates.

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
  - Both contrasts are null-compatible within the pre-registered $\pm 5.0$ pp equivalence bounds. Realized minimum detectable difference (MDD) at 80% power is $3.4\text{--}3.7$ pp, so gains below that threshold were not detectable.
- **Exploratory Contrasts**:
  - **$A_4 - A_1$**: $+0.87$ pp (discordant 7/4, $p=0.549$), requiring 4,988 extra forward passes.
  - The null-graph control (**A5**) scores 45.9% and the diffusion pool (**A6**) scores 45.6%.
  - Un-reranked BM25 (**B0**) keeps nominally higher MRR@50 and top-1 accuracy than $A_4$, but differences are within noise (paired Wilcoxon $p=0.72$).
  - Excluding Django ($N=230$), $A_4 - A_2$ is $+2.17$ pp ($p=0.227$).

### 4.4 Candidate Pool Truncation Curves & Loss Decomposition (RQ3)

| Truncation Cutoff | M=10 | M=15 | M=20 | M=30 | M=50 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A2 (Depth BM25) Recall** | 51.2% | 55.2% | 58.7% | 62.8% | 67.2% |
| **A2 (Depth BM25) Acc@5** | **46.2%** | 45.9% | 45.6% | 45.4% | 45.1% |
| **A4 (Graph 1-Hop) Recall** | 51.2% | 54.1% | 55.5% | 56.7% | 58.4% |
| **A4 (Graph 1-Hop) Acc@5** | 46.2% | 46.5% | **47.1%** | **47.1%** | **47.1%** |

- **Distractor Accumulation**: Truncating pools at $M \in [10, 50]$ confirms the distractor accumulation mechanism. In $A_2$, expanding $M$ from 10 to 50 raises pool recall by $+15.99$ pp (51.2% to 67.2%), yet Acc@5 drops monotonically from 46.2% to 45.1% ($-$1.16 pp) as added distractors outscore gold targets ($\mathbb{E}[X \mid s^*] = \sum (1 - F_{\text{dist}}(s^*))$). In $A_4$, Acc@5 rises to 47.1% at $M=20$ and plateaus through $M=50$.
- **Stratum C Breakdown ($N=131$ Un-Named Tasks)**: Where issues omit gold symbols, $A_4$ achieves 12.2% Acc@5 (16/131) vs 8.4% for $A_2$ (11/131, $+3.82$ pp, McNemar $p=0.125$), while unscored BM25 achieves 15.3% (B0, 20/131). Of 25 extra tasks recalled by graph expansion over seeds, 15 (60.0%) fall in Stratum A (already named), 1 in Stratum B, and 9 in Stratum C (36.0%), showing that graph expansion primarily duplicates lexical search.
- **Four-Stage Loss Funnel**: (1) Benchmark tasks: 100% (344) $\to$ (2) Seed pool ceiling: 51.2% (176, boundary loss 48.8 pp) $\to$ (3) Graph pool ceiling: 58.4% (201, $+7.3$ pp gain, boundary loss 41.6 pp) $\to$ (4) Scored top-5: 47.1% (162, compaction loss 11.3 pp). Upstream boundary loss exceeds downstream compaction loss by $3.68\times$ ($41.6 / 11.3$). Expansion adds 7.3 pp of oracle recall, yet Gemma 4 converts only $\sim 1$ pp into ranked accuracy.

### 4.5 Downstream Reader (Descriptive, N=100)

| Arm | Context Source | ContextHit@5 | EditLoc@1 | Conversion |
| :--- | :--- | :---: | :---: | :---: |
| **D0** | BM25 top-5 | 35.0% | 33.0% | 0.943 |
| **D1** | Graph top-5 | 37.0% | 31.0% | 0.838 |
| **D2** | Same-file top-5 | 38.0% | 30.0% | 0.789 |
| **D3** | Oracle (randomized position) | 100.0% | 74.0% | 0.740 |

In D0, always choosing candidate 1 also yields 33.0% EditLoc@1, indicating that Gemma 4 relies on a default rank-1 positional prior when context lacks clear discriminative features. In D3, the model finds gold 74.0% of the time (Wilson CI [64.6, 81.6]%; 67.5% chance-adjusted). While the reader misses about a quarter of oracle cases, retrieval supplies gold in only 35–38% of tasks and remains the decisive bottleneck.

### 4.6 Diffusion Diagnostics on SWE-bench Lite (Exploratory, N=237)
Under a diffuse linear prior, standard PPR collapses to 10.1% Acc@10; test subgraphs absorb 32.5% of top-10 slots. A top-25 softmax prior ($\tau=0.15$) on test-sanitized graphs restores 48.5%, versus 46.0% for BM25. Specificity damping gains +24.1 pp under the diffuse prior, but −0.8 pp (sanitized) and −4.6 pp (raw graph) under the softmax prior.

Across 186 prior configurations (63,984 evaluations on Verified), accuracy tracks effective prior support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$, with a fitted mixed-model vertex near $n_{\text{eff}}=8.8$. On the 109 tasks where hubs enter standard PPR's top-10, paired McNemar finds no detectable hub penalty ($p=0.688, n_{01}=4, n_{10}=2$). Damping lifts the Hub Suppression Index from 90.8% to 99.5% without changing accuracy. It is a safeguard for badly specified priors and hub crowding, not an accuracy gain.

---

## 5. Discussion

- **What the evidence supports**: On Verified Python tasks, graph 1-hop pools were not distinguishable from lexical-depth or same-file pools under a fixed small scorer. The intervals still admit gains near 4 pp: the result is "not demonstrated, and large effects excluded", not "no effect".
- **Distractor mechanism**: Truncation curves confirm that pool expansion degrades pointwise scoring: expanding depth BM25 from $M=10$ to $M=50$ adds $+15.99$ pp recall but reduces Acc@5 by $-1.16$ pp. Because expected distractors outscoring gold scale as $\sum (1 - F_{\text{dist}}(s^*))$, pointwise scoring penalizes pool breadth. Listwise ranking is the natural remedy.
- **Practical implication**: At a 4,096-token budget with a 2B scorer, first-stage lexical recall and AST-breadcrumb packing are the safer investment. Graph expansion cost 4,988 extra passes for +0.87 pp.
- **A reusable audit**: Any retrieval stage feeding a small model can be audited using the same framework: report pool oracle recall, add depth-matched and same-file controls under the same scorer, and decompose loss before and after ranking.
- **Limitations**: (1) Python-only ASTs missing dynamic dispatch. (2) Single 2B model, prompt, and pointwise scoring. (3) 1-hop pools capped at $M \le 50$. (4) Function-level targets on 344 tasks without patch synthesis. (5) Reader test is descriptive ($N=100$). (6) Only PC1/PC2 are confirmatory. (7) Edge weights and $\gamma$ were fixed globally.
- **Protocol disclosure**: B0 Acc@10 is 176/344 in the reranking protocol but 175/344 in the retrieval comparison. On `astropy__astropy-14182`, raw-token ranking kept the gold target at rank 10 while AST-key deduplication moved it to rank 11. PC1 and PC2 do not involve B0 and are unaffected; raw counts are recorded in `deviations.md`.

---

## 6. Released Resources

- **`scores.parquet`**: 19,298 scored candidate evaluations with per-arm rankings for all 344 tasks (Apache 2.0).
- **`perron-core` v0.3.0 (PyPI) and `perron-mcp`**: JSON-RPC stdio server with context-retrieval and AST-breadcrumb tools.
- **`swebench-lite-codegraphs`**: Precomputed CSR graphs for the 12 SWE-bench Lite repositories.
- **`swebench-strata-annotations`**: 800 lexical-accessibility annotations (CC0-1.0).
- **Reproduction notebook**: [https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation)

---

## 7. Conclusion

On SWE-bench Verified, graph 1-hop expansion neither beat matched lexical pools under a frozen Gemma 4 scorer nor widened the candidate ceiling beyond lexical depth. Most accuracy was lost before scoring, and extra recall converted poorly. Meaningful gains likely require higher first-stage recall or listwise ranking rather than adding local graph neighbours. The scores, code, and precomputed graphs are released to facilitate further experimentation.

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
