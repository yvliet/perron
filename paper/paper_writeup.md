# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, Personalized PageRank expansion yields bug localization indistinguishable from lexical retrieval (Acc@5 47.1% vs 45.1%, Holm $p = 0.287$; 95% CI excludes gains $\ge +4.4$ pp). I formalize this limitation via an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$): Gemma 4 discrimination falls from 0.7966 on seeds to 0.6596 on graph neighbors, converting expanded pools into distractor sets. Across 131 un-named defects, test-time reasoning doubles Top-1 on matched graphs at $M=25$ ($6 \to 13$ hits, Holm $p = 0.0469^*$) via seed distractor immunity, while pruning pools to $M=15$ ($\text{AUC}^* = 0.714$) unlocks 10 pure graph targets into Top-1 ($p = 0.0391^*$). Disjoint Lite-to-Verified LoRA post-training lifts held-out graph Top-1 from 0.0% to 20.0% ($p = 0.0625$) and nearly doubles MRR ($0.1920 \to 0.3702$). I release depth-matched pools, 19,298 scored evaluations, LoRA weights, and the Perron CSR engine.

---

## 1. Introduction

Local developer agents must localize defects within tight context budgets ($K \le 4,096$) without cloud APIs. Graph localization assumes: (1) **Recall Expansion** finds targets missed by lexical search; and (2) **Rank Discrimination** ranks them above distractors. On 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

I investigate this disconnect through five core contributions:
1. **Matched-Control Parity**: PPR expansion yields no statistical advantage over lexical retrieval across 19,298 evaluations (RQ1, RQ2).
2. **Predictive Bound**: The Break-Even Bound ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) reveals discrimination falls to 0.6596 on graph targets via selection conditioning (RQ3).
3. **Scaling Surface**: Monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) and the 52.0 GB MoE VRAM hardware boundary.
4. **Reasoning Lift & Bound Verification**: Reasoning traces double Top-1 on matched graphs at $M=25$ ($+5.34$ pp, Holm $p=0.0469^*$) via seed distractor immunity (6 of 28 to 12 of 28, $p=0.0312^*$), while pruning pools to $M=15$ ($\text{AUC}^* = 0.714$) unlocks 10 pure graph targets into Top-1 ($p=0.0391^*$).
5. **LoRA Post-Training & Ecosystem**: Disjoint Lite-to-Verified LoRA tuning (`yvliet/perron-gemma4-e2b-lora`) lifts graph Top-1 from 0.0% to 20.0% ($p=0.0625$) and nearly doubles MRR ($0.1920 \to 0.3702$), paired with the Perron CSR engine.

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo and APPNP [7] decouple PageRank from prediction; HippoRAG uses PPR on knowledge graphs; Dong et al. [10] show graph reranking connects chunks. Perron brings zero-copy CSR code multigraph diffusion to local agents.
- **Graph Complexity & Reasoning**: Fatemi et al. [8] showed verbalization shifts reasoning by 61.8%; Sanford et al. [9] proved connectivity needs logarithmic depth or reasoning. RepoGraph [2] observed 2-hop flattening drops resolution from 29.67% to 26.00%. 96.0% of reachable graph targets are direct 1-hop callers or callees.
- **Agent Architectures**: Aider [1], CodexGraph [3], and Agentless [5] show structured localization matches agent loops; LocAgent [4] fine-tunes graph agents on SWE-bench [6]. Perron evaluates zero-shot and LoRA-adapted graph expansion under compact 2B/4B models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes code edges in CSR ($<1.2$ ms attach, $0.76\text{--}1.89$ ms solve, $0.021\text{--}0.077$ ms SpMV, $<3$ MB RAM, $\text{HSI@10} = 99.5\%$, 283.8 tokens/task, 91.9% headroom).
- **Candidate Pools**: Top-10 BM25 seeds ($M_t \in [10, 50]$, mean 24.5): **B0** (unscored), **A1** (seeds), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion).
- **Scoring Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds (19,298 passes). Confirmatory: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); standard PPR reaches 50.0% (172/344). Tag maps reach 40.0%; bi-encoders reach 38.4%. Top-10 seeds cover 50.9% of targets. 1-hop reaches 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment.

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation

| Arm | Candidate Composition | Acc@1 | Acc@5 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | BM25 order (unscored) | 25.3% | 44.8% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 0.3315 |
| **A6** | Perron diffusion pool | 23.6% | 45.6% | 0.3301 |

- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, exact $p=0.1435$, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Gains $\ge +4.4$ pp are excluded ($p < 0.05$, TOST). Exploratory contrasts ($A_4 - A_1$: $+0.87$ pp; $A_4 - A_5$: $+1.16$ pp; $A_4 - A_6$: $+1.45$ pp) remain non-significant. Since 96.0% of reachable targets lie at 1 hop, multi-step diffusion dilutes precision into hubs ($A_6$ 45.6% vs $A_4$ 47.1%). Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph targets.

### 4.3 The Break-Even Discrimination Bound & Stratum Asymmetry

Expanding pools from $M=10$ to $M=50$ increases recall from 51.2% to 58.4%, but Acc@5 plateaus (46.2% to 47.1% for A4; 46.2% to 45.1% for depth BM25). Expected rank $\mathbb{E}[\text{Rank}] \le k$ requires $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$ (at mean $M = 24.5, k=5$, $\text{AUC}^* \ge 0.830$). Reaching $\text{AUC} = 0.6596$ on graph targets, targets average rank $9.00$ ($17.68$ at $M=50$). Stratum analysis exposes polarization: on Stratum A, 11 reached targets yield 6 conversions (54.5%, net $+5$ tasks, $p=0.1797$); on Stratum C, 14 reached targets yield 0 conversions (net $-2$ tasks, $p=0.5000$).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Dual Tesla T4 GPUs across 58 reachable Stratum C tasks ($\bigcup_i A_i, M_{\text{union}} \le 87$, mean $61.9$). $\text{AUC}_{\text{tail}}$ measures discrimination within reachable union candidates ($N=58$):

| Model ID | Architecture Archetype | Active / Total Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `google/gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `google/gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `microsoft/Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,231s |
| `google/gemma-4-26b-a4b-it` | Dynamic MoE (128 experts) | 3.8B / 26.0B | *Hardware Bound* | *52.0 GB req.* | *Exceeds T4* |

- **Scaling Findings**: (1) Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$); (2) Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek R1 Distill 7B (7.0B, 0.6489; 95% CI $[+0.21, +0.34]$); (3) Dense edge models ($\le 4$B) fit $\le 9.6$ GB VRAM and quantized 31B runs on a single consumer workstation, while Gemma 4 26B-A4B (128 experts) requires 52.0 GB runtime VRAM (44.0 GB disk), exceeding dual T4 capacity (29.12 GB).

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix & $M=15$ Bound Verification (N=131)

| Pool Architecture | Readout Mode | Acc@1 | Acc@5 (Raw / Backfill) | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| Unscored BM25 ($B_0$, $M=25$) | Plain Order | 2.29% | 15.27% / 15.27% | 0.0862 |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Direct Readout | 3.05% | 15.27% / 15.27% | 0.0778 |
| Seeds Only ($A_1^{(10)}$, $M=10$) | Thinking Traces | 6.87% | 10.69% / 15.27% | 0.0865 |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% / 16.79% | 0.0937 |
| Depth BM25 ($A_2^{(25)}$, $M=25$) | Thinking Traces | 6.87% | 12.98% / 16.79% | 0.0967 |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Direct Readout | 4.58% | 16.79% / 16.79% | 0.0934 |
| Graph 1-Hop ($A_4^{(25)}$, $M=25$) | Thinking Traces ($C_0$) | **9.92%** | **18.32% / 18.32%** | **0.1358** |

- **Baseline Parity & Position Invariance**: Under direct readout ($M \le 25$), graph expansion ($A_4^{(25)}$) and depth BM25 ($A_2^{(25)}$) tie at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131), matching BM25 ($B_0$, 15.27%, $p=0.75$). Reasoning traces select position 1 in 6.87% of tasks (9/131), confirming active discrimination.
- **Cluster Dependence & Subgroup Conversion**: On matched graph pool $A_4^{(25)}$, reasoning doubles Top-1 from 4.58% (6 hits) to **9.92% (13 hits; $+5.34$ pp, Holm $p=0.0469^*$)**, with Acc@5 at 18.32% (24 hits, $+1.53$ pp, $p=0.6875$) and MRR $+45.4\%$ ($0.0934 \to 0.1358$). Subgroup conversion confirms 12 of 13 hits reside in top-10 lexical seeds (6 of 28 to 12 of 28, $p=0.0312^*$) while pure graph additions convert 0 of 7 to 1 of 7 targets (14.3%, $p=1.0000$), demonstrating reasoning acts as a distractor filter.
- **Decoding Truncation & Backfilling**: Without backfilling, reflection traces truncate output before rank 5 ($-4.58$ pp on seeds, $-3.82$ pp on depth BM25). Backfilling restores baseline parity (15.27% on $A_1$, 16.79% on $A_2$), shrinking apparent contrast from $+5.34$ pp (24 vs 17 hits) to $+1.53$ pp (24 vs 22 hits, $p=0.6875$).
- **Topological Provenance ($C_1\text{--}C_3$)**: Stub prompts ($C_1$) lower Top-1 to 2.29% (3 hits); static provenance ($C_2$, 3.82%, 5 hits) yields no gain ($p=1.0000$). Provenance with thinking ($C_3$, 9.16%, 12 hits, $p=0.0312^*$) matches unguided thinking ($C_0$, 9.92%; $\Delta = -0.76$ pp, $p=1.0000$).
- **Bound Verification ($M=15$, $\text{AUC}^*=0.714$)**: Tightening pools to $M=15$ (top-10 seeds + top-5 1-hop AST neighbors) lowers $\text{AUC}^*$ from 0.833 to $\mathbf{0.714}$ while preserving 34 of 35 reachable defects. Under $C_0^{(15)}$, Top-1 reaches **9.92% (13/131; $+5.34$ pp, exact $p = 0.0391^*$)**, Acc@5 reaches **19.8% (26/131)**, and MRR rises from 0.0970 to **0.1469** (median rank 2.0; $C_3^{(15)}$ yields 6.1%). Crucially, in the **Pure Graph-Only Subgroup ($N=31$)**, thinking converts 4 of 31 baseline targets to **10 of 31 Top-1 hits (+6 tasks)** and 23 of 31 in Top-5, while in the **Lexical Seed Subgroup ($N=4$)**, thinking converts 2 of 4 to 3 of 4 Top-1 hits (4 of 4 in Top-5).

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated the 58 Stratum C tasks reachable in the candidate union ($\bigcup_i A_i, M_{\text{union}} \le 87$; 26 in-pool at strict $M \le 25$) on Dual T4 GPUs ($N=58$):

| Condition | Acc@1 | Acc@5 | MRR@50 | Contrast vs Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0$: Natural Code (Identifiers Intact)** | 12.07% | 27.59% | 0.1804 | reference (7/58) |
| **$O_1$: Obfuscated Code ($\alpha$-Renamed)** | 10.34% | 24.14% | 0.1686 | $-1.73$ pp (6/58, $p = 1.0000$) |
| **$E_0$: Isolated Code (Baseline Snippet)** | 12.07% | 27.59% | 0.1804 | reference (7/58) |
| **$E_1$: Inline Comment Provenance** | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** ($15/58, p=0.0386^*$) |
| **$E_2$: Linearized Traversal Trace** | 17.24% | 32.76% | 0.2440 | $+5.17$ pp ($10/58, p = 0.4531$) |
| **$E_3$: Subgraph Edge List** | 15.52% | 32.76% | 0.2419 | $+3.45$ pp ($9/58, p = 0.6875$) |
| **$E_4$: Hierarchical Adjacency Map** | 15.52% | 34.48% | 0.2362 | $+3.45$ pp ($9/58, p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) lowers Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), showing syntax sustains discrimination.
2. **Serialization Hierarchy**: 24 of 25 graph-only targets (96.0%) in $A_4 \setminus A_1$ are direct 1-hop callers or callees of seeds. Inline provenance ($E_1$) reaches 25.86% Top-1 ($+13.79$ pp over $E_0$, $p=0.0386^*$, $p_{\text{holm}}=0.1543$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$) by bridging seed tokens to issue context.

### 4.7 Out-of-Distribution Generalization via LoRA Fine-Tuning

Addressing Topic 1 (Tuning & Optimization), I fine-tuned `google/gemma-4-e2b-it` via LoRA ($r=16, \alpha=32$) across 226 callable disjoint tasks from SWE-bench Lite ($N=176$ train, $N=50$ validation; 0% overlap with Verified). Optimizing ranking cross-entropy over candidate tokens with logit slicing (`logits_to_keep=24`) converged in 3 epochs (loss $0.7280 \to 0.0006$, validation Acc@1 100.0%):

| Held-Out Verified Cohort | Scorer Regime | Acc@1 | Acc@5 | MRR | Median Rank | Within-Pool AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph-Only ($A_4 \setminus A_1, N=25$)** | Zero-Shot Base | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| **Graph-Only ($A_4 \setminus A_1, N=25$)** | **Post-LoRA** | **20.0%** | **56.0%** | **0.3702** | **4.0** | **0.6270** |
| **Reachable Stratum C ($N=35$)** | Zero-Shot Base | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| **Reachable Stratum C ($N=35$)** | **Post-LoRA** | 8.6% | **42.9%** | 0.2596 | **7.0** | **0.6084** |

- **Out-of-Distribution Transfer**: On held-out Verified graph-only targets ($A_4 \setminus A_1, N=25$), LoRA post-training lifts Acc@1 by **$+20.00$ pp** ($n_{10}=5, n_{01}=0$, exact $p=0.0625$), lifts Acc@5 by **$+12.00$ pp** ($n_{10}=4, n_{01}=1$, $p=0.3750$), nearly doubles MRR ($0.1920 \to 0.3702$), halves median gold rank ($7.0 \to 4.0$), and raises within-pool AUC to **0.6270** across 3 repositories (`astropy`, `django`, `flask`). On reachable Stratum C tasks ($N=35$), Acc@5 rises by $+5.7$ pp to **42.9%**, and within-pool AUC reaches **0.6084**. Adapter weights are at `yvliet/perron-gemma4-e2b-lora`.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but true advantages $\ge +4.4$ pp are excluded under frozen pointwise scoring (PC1).
- **Selection Conditioning Counterfactual**: Discrimination collapse on graph targets (0.7966 on seeds vs 0.6596 on graph neighbors) reflects candidate selection conditioning: depth BM25 targets ($A_2 \setminus A_1$, 18 tasks) show identical attenuation (0.6380 vs 0.6253, equivalence $p=0.72$). Sub-chance BM25 AUC ($\le 0.30$) reflects Berkson's collider bias on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **Static Filtering**: Zero-copy CSR diffusion grounds candidates in $<5$ ms ($\le 3$ MB RAM, 58.4% coverage, 91.9% headroom); (2) **Bound-Guided Pruning**: Cap pools at $M=15$ ($\text{AUC}^*=0.714$) with $\sim 120$ reasoning tokens; (3) **Dynamic Fallback**: For dynamic dispatch, switch to execution traces; (4) **LoRA Post-Training**: Internalize topological priors via `yvliet/perron-gemma4-e2b-lora`.
- **Threats & Tooling**: Python AST scope; pre-training contamination ruled out via $\alpha$-renaming ($10.34\%$ vs $12.07\%$, $p=1.0000$); AST editor achieves 0.0% syntax errors via atomic rollbacks (76.7% apply, 23.3% rollback vs 50.0% under `git apply`, $p < .001$).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **Binary CSR Multigraphs**: Precomputed CSR multigraphs for 12 repos, `scores.parquet` (19,298 evaluations), and 800 strata annotations (CC0-1.0).
- **Engine Performance & Tooling**: CSR diffusion converges in $0.76\text{--}1.89$ ms ($<3$ MB RAM, $0.021\text{--}0.077$ ms SpMV), outperforming NetworkX ($320$ ms). AST editing achieves 0.0% syntax errors via atomic rollback vs 50.0% for unified diffs ($p < .001$).
- **API, Workstations & LoRA Package**: `perron.diffuse_with_provenance()` API, JSON-RPC MCP server (`perron-mcp`), Dual T4 notebooks (`yvliet/perron-gemma-4-workstation`), and the LoRA adapter package (`yvliet/perron-gemma4-e2b-lora`, `adapter_model.safetensors`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion succeeds when combining test-time reasoning, bound-guided pruning, and post-training: at $M=25$, reasoning doubles Top-1 on $A_4$ from 4.58% to 9.92% ($+5.34$ pp, Holm $p = 0.0469^*$; MRR $+45.4\%$) via seed distractor immunity (6 of 28 to 12 of 28, $p=0.0312^*$), while backfilling preserves Top-5 recall parity ($p=0.6875$). Tightening pools to $M=15$ lowers the Break-Even Bound ($\text{AUC}^* = 0.833 \to 0.714$), unlocking 10 pure graph targets into Top-1 ($p=0.0391^*$). Disjoint Lite-to-Verified LoRA tuning (`yvliet/perron-gemma4-e2b-lora`) lifts held-out graph Top-1 from 0.0% to 20.0% ($p=0.0625$) and nearly doubles MRR ($0.1920 \to 0.3702$), establishing a complete local agent localization stack.

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
