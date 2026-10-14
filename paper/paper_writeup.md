# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, naive Personalized PageRank expansion yields bug localization statistically indistinguishable from lexical retrieval (Acc@5 47.1% vs 45.1%, exact binomial McNemar $p = 0.1435$, Holm $p = 0.287$; 95% CI excludes advantages $\ge +4.4$ pp). We characterize this limitation via an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$), observing that Gemma 4 pairwise discrimination falls from $\text{AUC} = 0.7966$ on seeds to $0.6596$ on graph neighbors, rendering expanded pools distractor sets under compact scorers. In an exploratory follow-up across 131 un-named defects, allocating short test-time reasoning compute ($\le 180$ tokens) raises Top-1 accuracy across candidate pools ($6 \to 13$ hits on the matched graph pool, Holm $p = 0.0469^*$), but leaves Top-5 accuracy null-compatible ($p = 0.6875$) as reasoning traces consume token budget ahead of trailing ranks. We release depth-matched control pools, 19,298 scored candidate evaluations, and the Perron Compressed Sparse Row diffusion engine.

---

## 1. Introduction

Local developer agents must localize defects within tight context budgets ($K \le 4,096$) without cloud APIs. Graph-augmented localization assumes: (1) **Recall Expansion** introduces targets missed by lexical search; and (2) **Rank Discrimination** ranks those targets above distractors. On 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy is indistinguishable from depth-matched BM25 (47.1% vs 45.1%, Holm $p=0.287$).

We investigate this disconnect through five core contributions:
1. **Matched-Control Parity**: PPR expansion yields no statistical advantage over lexical retrieval under frozen scorers across 19,298 evaluations (RQ1, RQ2).
2. **Predictive Bound**: The Break-Even Bound ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) shows that Gemma 4's discrimination drop to 0.6596 on graph targets turns expanded pools into distractors, driven by candidate selection conditioning (RQ3).
3. **Scaling Surface**: Protocol `PREREG-SCALING-v1` confirms monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) on reachable targets and identifies the sparse MoE 52.0 GB runtime VRAM hardware boundary.
4. **Reasoning Lift**: Reasoning traces elevate Top-1 across candidate pools (+15 net tasks, $16 \to 31$) and double matched graph Top-1 ($4.58\% \to 9.92\%$, $+5.34$ pp, exact $p = 0.0156^*$, Holm $p = 0.0469^*$). Acc@5 shifts remain null-compatible ($+1.53$ pp, $p=0.6875$), with lexical drops reflecting decoding truncation.
5. **Open Resource**: Perron provides a 0.76--1.89 ms CSR diffusion engine and AST editor.

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo and APPNP decouple PageRank propagation from neural prediction; HippoRAG applies PPR to text knowledge graphs, and Dong et al. (2024) show graph reranking connects isolated chunks in RAG. Perron extends this decoupled lineage to static code multigraphs in zero-copy CSR.
- **Graph Complexity & Reasoning**: Fatemi et al. (2024) showed verbalization shifts LLM reasoning by 61.8%; Sanford et al. (NeurIPS 2024) proved connectivity requires logarithmic depth or intermediate reasoning tokens. RepoGraph noted 2-hop flattening degrades resolution from 29.67% to 26.00%. Our audit shows 96.0% of graph targets are direct 1-hop callers or callees.
- **Agent Pipelines**: SWE-agent and Agentless show hierarchical localization matches complex agent loops; LocAgent fine-tunes graph-traversal agents. Perron evaluates zero-shot graph expansion under compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). 52 of 56 AST exclusions belong to Stratum C (39 in SymPy) from Python 2 syntax parsing failures on patches. Hyperparameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes call, inherit, import, caller edges in CSR ($<1.2$ ms attach, $0.76\text{--}1.89$ ms solve, $<3$ MB delta, $\text{HSI@10} = 99.5\%$, 283.8 tokens/symbol, 91.9% headroom).
- **Pools & Arms**: Seeds are top-10 BM25 symbols ($M_t \in [10, 50]$, mean 24.5): **B0** (unscored), **A1** (seeds), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion).
- **Scoring & Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds (19,298 passes). Confirmatory: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability
 
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); standard PPR reaches 50.0% (172/344). Tag maps reach 40.0% File Acc@5; bi-encoders reach 38.4%. Top-10 seeds cover 50.9% of targets. 1-hop reaches 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment (66.9%) and exceeding null graphs (63.8%, +3.02 pp).
 
### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation
 
| Arm | Candidate Composition | Acc@1 | Acc@5 | Acc@10 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 51.2% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 51.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 51.2% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 52.0% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 52.3% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 51.2% | 0.3315 |
| **A6** | Perron diffusion pool | 23.6% | 45.6% | 52.6% | 0.3301 |
 
- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Gains $\ge +4.4$ pp are excluded ($p < 0.05$, TOST). Exploratory: $A_4 - A_1$: $+0.87$ pp ($p=0.549$); null $A_4 - A_5$: $+1.16$ pp ($p=0.344$); diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph targets.
 
### 4.3 The Break-Even Discrimination Bound & Stratum Asymmetry
 
Expanding candidate pools from $M=10$ to $M=50$ increases recall from 51.2% to 58.4%, but Acc@5 plateaus (46.2% to 47.1% for A4; 46.2% to 45.1% for depth BM25). Landing in top-$k$ requires $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $\text{AUC} \ge 0.830$. Because Gemma 4 reaches only $\text{AUC} = 0.6596$ on graph targets competing against seed distractors, targets fall to expected rank $9.00$ ($17.68$ at $M=50$).
Stratum analysis exposes polarization: on Stratum A, 11 reached targets yield 6 top-5 conversions (54.5%, net $+5$ tasks, $p=0.1797$); on Stratum C, 14 reached targets yield 0 conversions (net $-2$ tasks, $p=0.5000$). The aggregate gain of $+3$ tasks (+0.87 pp, $p=0.549$) sums $+5$ named wins and $-2$ un-named losses.

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Kaggle Dual Tesla T4 GPUs on reachable Stratum C tasks ($N=58$). $\text{AUC}_{\text{tail}}$ evaluates discrimination within reachable graph candidates on Stratum C ($N=58$), distinct from benchmark discrimination ($N=344, \text{AUC}=0.6596$) where graph neighbors compete against top-10 seed distractors:

| Model ID | Architecture Archetype | Active / Total Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `google/gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `google/gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `microsoft/Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,230s |
| `google/gemma-4-26b-a4b-it` | Dynamic MoE (128 experts) | 3.8B / 26.0B | *Hardware Bound* | *52.0 GB req.* | *Exceeds T4* |

1. **Monotonic Scaling**: Tail discrimination on reachable candidates ($\text{AUC}_{\text{tail}}$) scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), confirming Hypothesis $H_1$.
2. **Representational Efficiency**: Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek R1 Distill 7B (7.0B, 0.6489; paired bootstrap 95% CI $[+0.21, +0.34]$).
3. **MoE Hardware Boundary**: Gemma 4 26B-A4B stores 128 experts as fused 3D tensors, bypassing 4-bit quantization. Weights require 52.0 GB runtime VRAM (44.0 GB disk), exceeding dual T4 capacity (29.12 GB). While dense edge models ($\le 4$B) fit consumer budgets ($\le 9.6$ GB), sparse MoE agents require pre-quantized formats.

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix (N=131)

| Candidate Pool Architecture | Readout Mode | Acc@1 | Acc@5 | Hits / 131 | MRR@50 | $\Delta_{\text{reasoning}}$ Acc@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Unscored BM25 ($B_0$, $M=25$) | Plain Order | 2.29% | 15.27% | 20 | 0.0862 | baseline |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Direct Readout | 3.05% | 15.27% | 20 | 0.0778 | reference |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Thinking Traces | 6.87% | 10.69% | 14 | 0.0865 | $-4.58$ pp ($-6$ tasks) |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% | 22 | 0.0937 | reference |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Thinking Traces | 6.87% | 12.98% | 17 | 0.0967 | $-3.82$ pp ($-5$ tasks) |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% | 22 | 0.0934 | reference |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Thinking Traces ($C_0$) | **9.92%** | **18.32%** | **24** | **0.1358** | **+1.53 pp (+2 tasks)** |

- **Baseline Parity & Position Bias**: Under direct listwise readout, graph expansion ($A_4^{(25)}$) and depth BM25 ($A_2^{(25)}$) tie at 16.79% Acc@5 (22/131 hits) and 4.58% Top-1 (6/131), matching unscored BM25 ($B_0$, 15.27% Acc@5, exact $p=0.75$, MRR 0.0862). While direct readout exhibits seed position bias, reasoning selects position 1 in only 6.87% of tasks (9/131), matching sha256 order permutation and confirming active candidate discrimination.
- **Cluster Dependence & Pool Holm Tests**: Evaluating pools independently under step-down Holm correction isolates where reasoning acts across non-independent presentations ($A_1 \subset A_2 \approx A_4$): on matched graph pool $A_4^{(25)}$, reasoning more than doubles Top-1 from 4.58% (6 hits) to **9.92% (13 hits; $+5.34$ pp, exact binomial McNemar $p=0.0156^*$, Holm $p=0.0469^*$)**, with Acc@5 at 18.32% (24 hits, $+1.53$ pp, discordant $n_{10}=4, n_{01}=2$, exact $p=0.6875$) and MRR rising $+45.4\%$ ($0.0934 \to 0.1358$). Seeds ($A_1^{(10)}$) shift $4 \to 9$ hits ($p_{\text{holm}}=0.1250$); depth BM25 ($A_2^{(25)}$) shifts $6 \to 9$ hits ($p_{\text{holm}}=0.3750$). Across all 393 presentations, reasoning yields $+15$ net top-1 hits ($16 \to 31$).
- **Decoding Truncation Mechanics**: Acc@5 drops on lexical pools ($-4.58$ pp seeds, $-3.82$ pp depth BM25) while gaining $+1.53$ pp (+2 tasks, $p=0.6875$) on graphs. Under $\le 180$ tokens, reflection traces consume generation bandwidth, truncating trailing ranks 2--5. The contrast ($\Delta A_4 - \Delta A_2 = +5.34$ pp, 24 vs 17 hits) reflects pool composition under token limits.
- **Topological Provenance ($C_1\text{--}C_3$)**: Un-thought provenance ($C_2$, 3.82% Top-1, $p=0.5000$) gives modest lift over isolated breadcrumbs ($C_1$, 2.29%). Adding provenance to thinking ($C_3$, 9.16%, exact $p=0.0312^*$ vs $A_4$ Direct) matches unguided thinking ($C_0$, 9.92%; $\Delta = -0.76$ pp, exact McNemar $p=1.0000$, TOST $p=0.1745$).

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated 58 reachable held-out Stratum C defects on Kaggle Dual Tesla T4 GPUs ($N=58$):

| Condition | Acc@1 | Acc@5 | MRR | Contrasts vs. Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0$: Natural Code (Identifiers Intact)** | 12.07% | 27.59% | 0.1654 | reference (7/58 hits) |
| **$O_1$: Obfuscated Code ($\alpha$-Renamed)** | 10.34% | 24.14% | 0.1448 | $-1.73$ pp ($6/58$, exact $p = 1.0000$) |
| **$E_0$: Isolated Code (Baseline Snippet)** | 12.07% | 27.59% | 0.1654 | reference (7/58 hits) |
| **$E_1$: Inline Comment Provenance** | **25.86%** | **39.66%** | **0.2982** | **+13.79 pp** ($15/58$, $p=0.0386^*$, $p_{\text{holm}}=0.1543$) |
| **$E_2$: Linearized Traversal Trace** | 17.24% | 32.76% | 0.2227 | $+5.17$ pp ($10/58$, $p = 0.4531$) |
| **$E_3$: Subgraph Edge List** | 15.52% | 32.76% | 0.2084 | $+3.45$ pp ($9/58$, $p = 0.6875$) |
| **$E_4$: Hierarchical Adjacency Map** | 15.52% | 34.48% | 0.2185 | $+3.45$ pp ($9/58$, $p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) lowers Top-1 by $-1.73$ pp (6 vs 7 hits, exact $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), confirming control-flow syntax sustains discrimination without identifiers, though power on $N=58$ cannot rule out moderate effects.
2. **Topological Audit & Prompt Serialization**: 24 of 25 graph-only targets (96.0%) are direct 1-hop callers or callees of seeds (10 callees, 14 callers, 1 co-located class). Inline comment provenance ($E_1$) reaches 25.86% Top-1 ($+13.79$ pp over $E_0$, unadjusted $p=0.0386^*$, step-down $p_{\text{holm}}=0.1543$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$). In $E_1$ ($N=58$, Gemma 4 E2B), reachable targets benefit from seed identifiers bridging candidates to issue tokens, contrasting with $C_2$ ($N=131$, Gemma 4 E4B, $+1.53$ pp) where 96 targets are omitted.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but differences remain null-compatible with gains $> +4.4$ pp excluded (PC1).
- **Selection Conditioning Counterfactual**: Gemma 4's discrimination drop on graph targets (0.7966 on seeds vs 0.6596 on graph neighbors) is driven by candidate selection conditioning: depth BM25 non-graph targets ($A_2 \setminus A_1$, 18 tasks) exhibit identical attenuation (mean 0.6380 vs graph 0.6253, equivalence $p=0.72$). BM25 sub-chance AUC ($\le 0.30$) reflects Berkson's collider bias on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **Static Filtering**: Zero-copy CSR diffusion grounds candidates in $<5$ ms ($\le 3$ MB RAM, 58.4% coverage, 91.9% headroom); (2) **Test-Time Reflection**: Allocate $\sim 120$ `<thought>` tokens to trace local call paths; (3) **Dynamic Fallback**: For dynamic dispatch, switch to execution traces; (4) **Post-Training**: Gemma 4 E2B LoRA tuning on ranking cross-entropy achieves 100.0% Top-5 discrimination ($N=50$, loss $0.639 \to 0.0002$).
- **Threats & Tooling**: (1) Python ASTs; (2) Compact models (E2B/E4B); (3) Pre-training contamination risk on open repos (Django, SymPy); $\alpha$-renaming confirms syntax fallback ($10.34\%$ vs $12.07\%$, $p=1.0000$, wide 95% CI [$-$11.2, +7.7] pp); (4) AST editor achieves 0.0% syntax errors via atomic rollbacks (76.7% valid apply, 23.3% safe rollback vs 50.0% syntax errors under `git apply`; Aider 0.0% reflects retrieval recall under truncation, not editing).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **Binary CSR Multigraphs & Archive**: Precomputed multigraphs for 12 repos in zero-copy CSR format, `scores.parquet` (19,298 evaluations), and 800 strata annotations (CC0-1.0).
- **Engine Performance & Tooling**: CSR diffusion converges in $0.76\text{--}1.89$ ms ($<3$ MB RAM, $0.021\text{--}0.077$ ms SpMV latency), outperforming NetworkX ($320$ ms). Downstream AST editing achieves 0.0% syntax errors via atomic rollback vs 50.0% for unified diffs ($p < .001$).
- **API & Workstations**: High-level `perron.diffuse_with_provenance()` API, JSON-RPC MCP server (`perron-mcp`), Dual Tesla T4 notebooks (`yvliet/perron-gemma-4-workstation`), and fine-tuned Gemma 4 E2B LoRA weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion succeeds only with test-time reasoning compute: allocating reasoning tokens produces $+15$ net top-1 hits across all 393 candidate presentations ($16 \to 31$). On the matched graph pool, reasoning traces double Top-1 localization from 4.58% to 9.92% (exact binomial McNemar $p = 0.0156^*$, step-down Holm $p = 0.0469^*$; MRR $+45.4\%$), achieving parity with explicit edge annotations ($p = 1.0000$). While Acc@5 exhibits pool-dependent behavior (+2 tasks on graphs vs $-5$ on depth BM25 from decoding truncation), cross-architecture scaling confirms monotonic discrimination gains in Gemma 4 ($0.8544 \to 0.9274$) while quantifying memory boundaries for sparse MoE models on consumer GPUs (52.0 GB runtime VRAM). Paired with Perron's high-throughput CSR diffusion engine, these results establish that compact open models navigate non-lexical code graphs when provided an allocated inference-time reasoning budget.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] H. Chen et al. *LocAgent.* arXiv:2503.09080, 2025.
- [5] C. S. Xia et al. *Agentless.* arXiv:2407.01489, 2024.
- [6] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [7] J. Klicpera et al. *Predict then Propagate.* ICLR 2019.
- [8] A. Fatemi et al. *Talk like a graph.* NeurIPS 2024.
- [9] C. Sanford et al. *Understanding transformers on random graphs.* NeurIPS 2024.
- [10] J. Dong, B. Fatemi, B. Perozzi, L. F. Yang, and A. Tsitsulin. *Don't Forget to Connect! Improving RAG with Graph-based Reranking.* arXiv:2405.18414, 2024.
