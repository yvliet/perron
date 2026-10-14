# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, naive Personalized PageRank expansion yields bug localization statistically indistinguishable from lexical retrieval (Acc@5 47.1% vs 45.1%, exact binomial McNemar $p = 0.1435$, Holm $p = 0.287$; 95% CI excludes advantages $\ge +4.4$ pp). I characterize this limitation via an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$), observing that Gemma 4 pairwise discrimination falls from $\text{AUC} = 0.7966$ on seeds to $0.6596$ on graph neighbors, turning expanded pools into distractor sets under compact scorers. Across 131 un-named defects, test-time reasoning ($\le 180$ tokens) doubles Top-1 on matched graphs ($6 \to 13$ hits, Holm $p = 0.0469^*$) via distractor immunity on lexical seeds, while Top-5 remains null-compatible ($p = 0.6875$) under backfilling. I release depth-matched control pools, 19,298 scored candidate evaluations, and the Perron Compressed Sparse Row diffusion engine.

---

## 1. Introduction

Local developer agents must localize defects within tight context budgets ($K \le 4,096$) without cloud APIs. Graph localization assumes: (1) **Recall Expansion** finds targets missed by lexical search; and (2) **Rank Discrimination** ranks them above distractors. On 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

I investigate this disconnect through five core contributions:
1. **Matched-Control Parity**: PPR expansion yields no statistical advantage over lexical retrieval across 19,298 evaluations (RQ1, RQ2).
2. **Predictive Bound**: The Break-Even Bound ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) reveals that discrimination drops to 0.6596 on graph targets via selection conditioning, creating distractors (RQ3).
3. **Scaling Surface**: Monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) on reachable targets and the sparse MoE 52.0 GB runtime VRAM boundary.
4. **Reasoning Lift & Distractor Immunity**: On matched graphs ($A_4$), reasoning traces double Top-1 from 4.58% to 9.92% ($+5.34$ pp, Holm $p=0.0469^*$; MRR $+45.4\%$). Subgroup conversion reveals gains concentrate in lexical seeds (converting 6 of 28 seeds [21.4%] to 12 of 28 [42.9%], $p=0.0312^*$) rather than pure graph targets (0 of 7 to 1 of 7, 14.3%, $p=1.0000$), demonstrating distractor immunity. Backfilling restores lexical recall parity (+1.53 pp contrast, $p=0.6875$).
5. **Open Resource**: Perron provides a 0.76-1.89 ms CSR diffusion engine and AST editor.

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo and APPNP [7] decouple PageRank propagation from neural prediction; HippoRAG applies PPR to knowledge graphs, and Dong et al. [10] show graph reranking connects chunks in RAG. Perron extends this lineage to code multigraphs in zero-copy CSR.
- **Graph Complexity & Reasoning**: Fatemi et al. [8] showed verbalization shifts LLM reasoning by 61.8%; Sanford et al. [9] proved connectivity requires logarithmic depth or intermediate reasoning tokens. RepoGraph [2] noted 2-hop flattening degrades resolution from 29.67% to 26.00%. My audit shows 96.0% of graph targets are direct 1-hop callers or callees.
- **Agent Pipelines**: Aider [1], CodexGraph [3], and Agentless [5] show structured localization matches complex agent loops; LocAgent [4] fine-tunes graph agents on SWE-bench [6]. Perron evaluates zero-shot graph expansion under compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). 52 of 56 AST exclusions belong to Stratum C (39 in SymPy, Python 2 syntax). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes call, inherit, import, caller edges in CSR ($<1.2$ ms attach, $0.76\text{--}1.89$ ms solve, $<3$ MB delta, $\text{HSI@10} = 99.5\%$, 283.8 tokens/task, $<100$/symbol, 91.9% headroom).
- **Pools & Arms**: Seeds are top-10 BM25 symbols ($M_t \in [10, 50]$, mean 24.5): **B0** (unscored), **A1** (seeds), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion).
- **Scoring & Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds (19,298 passes). Confirmatory: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability
 
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); standard PPR reaches 50.0% (172/344). Tag maps reach 40.0%; bi-encoders reach 38.4%. Top-10 seeds cover 50.9% of targets. 1-hop reaches 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment (66.9%).
 
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
 
- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Gains $\ge +4.4$ pp are excluded ($p < 0.05$, TOST). Exploratory: $A_4 - A_1$: $+0.87$ pp ($p=0.549$); null $A_4 - A_5$: $+1.16$ pp ($p=0.344$); diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). Because 96.0% of reachable targets lie at 1 hop, multi-step diffusion dilutes precision into hubs ($A_6$ 45.6% vs $A_4$ 47.1% Acc@5), making 1-hop adjacency ($A_4$) the primary graph pool for downstream factorial tests. Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph targets.
 
#### 4.3 The Break-Even Discrimination Bound & Stratum Asymmetry
 
Expanding candidate pools from $M=10$ to $M=50$ increases recall from 51.2% to 58.4%, but Acc@5 plateaus (46.2% to 47.1% for A4; 46.2% to 45.1% for depth BM25). Expected rank $\mathbb{E}[\text{Rank}] \le k$ requires $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$ (at mean $M = 24.5, k=5$, $\text{AUC}^* \ge 0.830$). Reaching only $\text{AUC} = 0.6596$ on graph targets against seed distractors, targets average rank $9.00$ ($17.68$ at $M=50$). Stratum analysis exposes polarization: on Stratum A, 11 reached targets yield 6 conversions (54.5%, net $+5$ tasks, $p=0.1797$); on Stratum C, 14 reached targets yield 0 conversions (net $-2$ tasks, $p=0.5000$), totaling $+3$ tasks (+0.87 pp, $p=0.549$).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Dual Tesla T4 GPUs across the 58 Stratum C tasks reachable in the candidate pool union ($\bigcup_i A_i, M_{\text{union}} \le 87$, mean $61.9$). $\text{AUC}_{\text{tail}}$ measures discrimination within reachable union candidates ($N=58$), distinct from full-benchmark discrimination ($N=344, \text{AUC}=0.6596$):

| Model ID | Architecture Archetype | Active / Total Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `google/gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `google/gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `microsoft/Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,231s |
| `google/gemma-4-26b-a4b-it` | Dynamic MoE (128 experts) | 3.8B / 26.0B | *Hardware Bound* | *52.0 GB req.* | *Exceeds T4* |

1. **Monotonic Scaling**: Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), confirming Hypothesis $H_1$.
2. **Representational Efficiency**: Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek R1 Distill 7B (7.0B, 0.6489; 95% CI $[+0.21, +0.34]$).
3. **MoE Hardware Boundary**: Gemma 4 26B-A4B experts bypass 4-bit quantization, requiring 52.0 GB runtime VRAM (44.0 GB disk), exceeding dual T4 capacity (29.12 GB). Dense edge models ($\le 4$B) fit consumer budgets ($\le 9.6$ GB); sparse MoE agents require pre-quantized formats.

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix (N=131)

| Candidate Pool Architecture | Readout Mode | Acc@1 | Acc@5 (Raw / Backfill) | Hits / 131 | MRR@50 | $\Delta_{\text{reasoning}}$ Acc@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Unscored BM25 ($B_0$, $M=25$) | Plain Order | 2.29% | 15.27% / 15.27% | 20 | 0.0862 | baseline |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Direct Readout | 3.05% | 15.27% / 15.27% | 20 | 0.0778 | reference |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Thinking Traces | 6.87% | 10.69% / 15.27% | 14 / 20 | 0.0865 | $-4.58$ pp raw ($\pm 0.00$ pp backfill) |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% / 16.79% | 22 | 0.0937 | reference |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Thinking Traces | 6.87% | 12.98% / 16.79% | 17 / 22 | 0.0967 | $-3.82$ pp raw ($\pm 0.00$ pp backfill) |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% / 16.79% | 22 | 0.0934 | reference |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Thinking Traces ($C_0$) | **9.92%** | **18.32% / 18.32%** | **24** | **0.1358** | **+1.53 pp (+2 tasks)** |

- **Baseline Parity & Position Invariance**: Under direct listwise readout ($M \le 25$), graph expansion ($A_4^{(25)}$) and depth BM25 ($A_2^{(25)}$) tie at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131), matching unscored BM25 ($B_0$, 15.27%, exact $p=0.75$). Reasoning traces select position 1 in only 6.87% of tasks (9/131, matching sha256 permutations), confirming active discrimination.
- **Cluster Dependence & Subgroup Conversion**: Under step-down Holm correction across presentations ($A_1 \subset A_2 \approx A_4$): on matched graph pool $A_4^{(25)}$, reasoning more than doubles Top-1 from 4.58% (6 hits) to **9.92% (13 hits; $+5.34$ pp, exact $p=0.0156^*$, Holm $p=0.0469^*$)**, with Acc@5 at 18.32% (24 hits, $+1.53$ pp, $p=0.6875$) and MRR $+45.4\%$ ($0.0934 \to 0.1358$). Seeds ($A_1^{(10)}$) shift $4 \to 9$ hits ($p_{\text{holm}}=0.1250$); depth BM25 ($A_2^{(25)}$) shifts $6 \to 9$ hits ($p_{\text{holm}}=0.3750$). Subgroup conversion confirms 12 of 13 hits are top-10 lexical seeds (converting 6 of 28 seeds [21.4%] to 12 of 28 [42.9%], $p=0.0312^*$) while pure graph additions convert 0 of 7 to 1 of 7 targets (14.3%, $p=1.0000$), demonstrating reasoning acts as a distractor filter protecting lexical seeds rather than navigating deep graphs.
- **Decoding Truncation & Backfilling**: Without backfilling, reflection traces truncate output lists before rank 5 ($-4.58$ pp raw on seeds, $-3.82$ pp on depth BM25). Deterministic backfilling restores baseline parity (15.27% on $A_1$, 16.79% on $A_2$), shrinking the apparent differential contrast from an artifactual $+5.34$ pp (24 vs 17 hits) to an honest $+1.53$ pp (24 vs 22 hits, $p=0.6875$).
- **Topological Provenance ($C_1\text{--}C_3$)**: Stripping docstrings to 3-line stubs ($C_1$) lowers Top-1 to 2.29% (3 hits); static provenance ($C_2$, 3.82%, 5 hits) is null-compatible ($p=1.0000$). Provenance with thinking ($C_3$, 9.16%, 12 hits, $p=0.0312^*$) matches unguided thinking ($C_0$, 9.92%; $\Delta = -0.76$ pp, $p=1.0000$, TOST $p=0.1745$).

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated the 58 Stratum C tasks reachable in the candidate union ($\bigcup_i A_i, M_{\text{union}} \le 87$; 26 in-pool at strict $M \le 25$) on Dual T4 GPUs ($N=58$):

| Condition | Acc@1 | Acc@5 | MRR@50 | Contrasts vs. Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0$: Natural Code (Identifiers Intact)** | 12.07% | 27.59% | 0.1804 | reference (7/58 hits) |
| **$O_1$: Obfuscated Code ($\alpha$-Renamed)** | 10.34% | 24.14% | 0.1686 | $-1.73$ pp ($6/58$, exact $p = 1.0000$) |
| **$E_0$: Isolated Code (Baseline Snippet)** | 12.07% | 27.59% | 0.1804 | reference (7/58 hits) |
| **$E_1$: Inline Comment Provenance** | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** ($15/58$, $p=0.0386^*$, $p_{\text{holm}}=0.1543$) |
| **$E_2$: Linearized Traversal Trace** | 17.24% | 32.76% | 0.2440 | $+5.17$ pp ($10/58$, $p = 0.4531$) |
| **$E_3$: Subgraph Edge List** | 15.52% | 32.76% | 0.2419 | $+3.45$ pp ($9/58$, $p = 0.6875$) |
| **$E_4$: Hierarchical Adjacency Map** | 15.52% | 34.48% | 0.2362 | $+3.45$ pp ($9/58$, $p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) lowers Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), showing syntax sustains discrimination without identifiers.
2. **Topological Audit & Prompt Serialization**: 24 of 25 graph-only targets (96.0%) in $A_4 \setminus A_1$ are direct 1-hop callers or callees of seeds (10 callees, 14 callers). Inline comment provenance ($E_1$) reaches 25.86% Top-1 ($+13.79$ pp over $E_0$, $p=0.0386^*$, $p_{\text{holm}}=0.1543$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$) by bridging seed tokens to issue context.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but gains $\ge +4.4$ pp are excluded (PC1).
- **Selection Conditioning Counterfactual**: Gemma 4's discrimination drop on graph targets (0.7966 on seeds vs 0.6596 on graph neighbors) stems from candidate selection conditioning: depth BM25 targets ($A_2 \setminus A_1$, 18 tasks) show identical attenuation (mean 0.6380 vs graph 0.6253, equivalence $p=0.72$). BM25 sub-chance AUC ($\le 0.30$) reflects Berkson's collider bias on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **Static Filtering**: Zero-copy CSR diffusion grounds candidates in $<5$ ms ($\le 3$ MB RAM, 58.4% coverage, 91.9% headroom); (2) **Test-Time Reflection**: Allocate $\sim 120$ `<thought>` tokens to trace local call paths; (3) **Dynamic Fallback**: For dynamic dispatch, switch to execution traces; (4) **Post-Training Optimization Smoke Test**: As a parameter-efficient smoke test on Dual T4 VRAM, LoRA tuning (`gemma-4-e2b-it`, $r=16$) on provenance ranking cross-entropy ($N=50$ pairs) confirms gradient stability without numerical divergence (loss $0.6391 \to 0.0002$).
- **Threats & Tooling**: (1) Python ASTs; (2) Compact models (E2B/E4B); (3) Pre-training contamination risk; $\alpha$-renaming confirms syntax fallback ($10.34\%$ vs $12.07\%$, $p=1.0000$); (4) AST editor achieves 0.0% syntax errors via atomic rollbacks (76.7% valid apply, 23.3% safe rollback vs 50.0% syntax errors under `git apply`; Aider 0.0% reflects retrieval recall under truncation).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **Binary CSR Multigraphs & Archive**: Precomputed CSR multigraphs for 12 repos, `scores.parquet` (19,298 evaluations), and 800 strata annotations (CC0-1.0).
- **Engine Performance & Tooling**: CSR diffusion converges in $0.76\text{--}1.89$ ms ($<3$ MB RAM, $0.021\text{--}0.077$ ms SpMV latency), outperforming NetworkX ($320$ ms). Downstream AST editing achieves 0.0% syntax errors via atomic rollback vs 50.0% for unified diffs ($p < .001$).
- **API & Workstations**: `perron.diffuse_with_provenance()` API, JSON-RPC MCP server (`perron-mcp`), Dual Tesla T4 notebooks (`yvliet/perron-gemma-4-workstation`), and fine-tuned Gemma 4 E2B LoRA weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion succeeds only under test-time reasoning: holding the pool fixed at graph 1-hop ($A_4$), reasoning traces double Top-1 localization from 4.58% (6 hits) to 9.92% (13 hits; $+5.34$ pp, exact $p = 0.0156^*$, Holm $p = 0.0469^*$; MRR $+45.4\%$). Subgroup decomposition confirms 12 of 13 hits reside in top-10 lexical seeds (converting 6 of 28 seeds [21.4%] to 12 of 28 [42.9%], $p=0.0312^*$) while pure 1-hop graph additions convert 0 of 7 to 1 of 7 targets (14.3%, $p=1.0000$), demonstrating distractor immunity protecting lexical anchors rather than deep graph reachability. While raw Acc@5 drops on lexical pools ($-5$ tasks on depth BM25) from decoding truncation, backfilling preserves baseline recall parity (+2 tasks on graphs, $p=0.6875$). Paired with Perron's high-throughput CSR engine, these findings show code graphs require test-time compute or localized structural provenance to unlock local agent performance.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] Z. Chen et al. *LocAgent.* ACL 2025.
- [5] C. S. Xia et al. *Agentless.* FSE 2025.
- [6] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [7] J. Klicpera et al. *Predict then Propagate.* ICLR 2019.
- [8] B. Fatemi, J. Halcrow, and B. Perozzi. *Talk like a graph.* ICLR 2024.
- [9] C. Sanford et al. *Understanding transformer reasoning capabilities via graph algorithms.* NeurIPS 2024.
- [10] J. Dong, B. Fatemi, B. Perozzi, L. F. Yang, and A. Tsitsulin. *Don't Forget to Connect! Improving RAG with Graph-based Reranking.* arXiv:2405.18414, 2024.
