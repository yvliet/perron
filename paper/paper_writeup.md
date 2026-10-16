# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, Personalized PageRank expansion yields bug localization indistinguishable from lexical retrieval (Acc@5 47.1% vs 45.1%, exact McNemar $p = 0.1435$, Holm $p = 0.287$; 90% Schuirmann TOST CI [+0.08, +3.98] pp excludes gains $\ge +4.0$ pp). I formalize this limitation via a first-moment **Break-Even Operational Diagnostic** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$, where $\mathbb{E}[\text{Rank}] \le k \implies \text{Acc@}k \ge 1/k$ by Markov's inequality): Gemma 4 discrimination attenuates from 0.7966 on seeds to 0.6596 on graph neighbors via confounder truncation selection. Across 131 un-named defects, test-time reasoning doubles Top-1 at $M=25$ ($6 \to 13$ hits, Holm $p = 0.0469^*$) by filtering seed distractors. Bound-guided pruning to $M=15$ ($\text{AUC}^* = 0.714$) stabilizes Acc@5 at 19.85% (26/131) and converts 2/6 pure graph additions to Top-1 (33.3%), while net Top-1 remains stationary (13/131). Post-training audits show naive LoRA tuning overfits prompt comment shortcuts, halving mixed-pool Top-1 (17.1% $\to$ 8.6%). I release depth-matched pools, 19,298 scored evaluations, 800 strata labels, and the Perron CSR engine (<5.2 ms solve, <=3.0 MB RAM).

---

## 1. Introduction

Local developer agents must localize defects within tight contexts ($K \le 4,096$) without cloud APIs. Graph localization requires: (1) **Recall Expansion** capturing missed targets; and (2) **Rank Discrimination** prioritizing them. Across 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

I resolve this disconnect through five contributions:
1. **Matched-Control Parity**: PPR expansion shows no statistical gain over lexical retrieval across 19,298 evaluations (RQ1, RQ2).
2. **First-Moment Diagnostic**: The Break-Even Diagnostic ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) formalizes discrimination collapse to 0.6596 on graph targets via confounder truncation selection (RQ3).
3. **Scaling Surface**: Monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) and the 52.0 GB MoE VRAM boundary.
4. **Reasoning Lift & Pruning**: Reasoning doubles Top-1 at $M=25$ ($p=0.0469^*$), while $M=15$ pruning ($\text{AUC}^* = 0.714$) converts 2/6 pure graph targets to Top-1 (38.2% reachable conversion) under stationary net Top-1 ($13/131$).
5. **Post-Training Diagnostics**: Auditing LoRA tuning (`yvliet/perron-gemma4-e2b-lora`) reveals prompt shortcut vulnerability, lifting isolated graph Top-1 ($0.0\% \to 20.0\%$) while halving mixed-pool precision ($17.1\% \to 8.6\%$).

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo and APPNP [7] decouple PageRank from prediction; HippoRAG uses PPR on knowledge graphs; Dong et al. [10] connect chunks. Perron provides zero-copy CSR multigraph diffusion for local developer agents.
- **Graph Complexity & Reasoning**: Fatemi et al. [8] showed verbalization shifts reasoning up to 61.8%; Sanford et al. [9] established relational connectivity on synthetic graphs requires logarithmic depth or pause tokens, whereas 96% of SWE-bench defect targets are 1-hop. RepoGraph [2] noted 2-hop distractor accumulation degrades resolution (29.67% to 26.00%, concordant with our paired confidence intervals).
- **Agent Architectures**: Aider [1], CodexGraph [3], and Agentless [5] demonstrate localization bounds downstream editing; LocAgent [4] fine-tunes graph agents on SWE-bench [6]. Perron isolates graph expansion under compact 2B/4B models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). 56 tasks (52 Stratum C) dropped due to Python 2 syntax failing `ast.parse`; sensitivity analysis confirms this attrition conservative-bounds the null result because Stratum C had net negative graph yield (-2 tasks). Parameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron CSR Engine & Convergence**: CSR arrays index `call`, `inherit`, `import`, `caller` edges ($\omega = [0.50, 0.25, 0.15, 0.10]$; $1.89\text{--}5.10$ ms solve across SWE-bench repos, $<3$ MB RAM; vs SciPy direct `spsolve` at 28.1 ms). For Google operator $M = \beta(T + \mathbf{d}\mathbf{p}_0^\top) + (1-\beta)\mathbf{1}\mathbf{p}_0^\top$ ($\beta=0.85$), Brauer's theorem bounds $|\lambda_2(M)| \le \beta$, ensuring spectral gap $\ge 0.15$ and Doeblin $L_1$ contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$ ($\le 90$ steps). Specificity damping ($\gamma=0.70$) yields $\text{HSI@10} = 99.5\%$, preserving 91.9% prompt headroom under listwise candidate packing (283.8 tokens across 10 AST stubs).
- **Candidate Pools & Cryptographic Locks**: Top-10 BM25 seeds ($M_t \in [10, 50]$, mean 24.5) define pools B0--A6, locked under commits `6908e78`, `9abe405` (`scores.parquet`), `48eda5d`, and `c563441`.
- **Scoring Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores log-odds (19,298 passes). Confirmatory contrasts PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) evaluate exact McNemar tests with step-down Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4%. 1-hop reachability is 57.8%; 2-hop is 66.9%. A 186-config sweep (63,984 solves) confirms Acc@10 follows a quadratic GLMM in support $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($R^2 = 0.197, p < 0.001$), peaking at $n_{\text{eff}}^* \approx 8.8$ (50.9%). Leave-Django-Out jackknife ($N=230$) confirms candidate parity (57.1% vs 57.1%, $p=1.0000$).

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation

| Arm | Candidate Composition | Acc@1 | Acc@5 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | BM25 unscored | 25.3% | 44.8% | 0.3439 |
| **A1** | Seeds only ($M=10$) | 24.7% | 46.2% | 0.3348 |
| **A2** | Depth BM25 ($M \le 50$) | 23.8% | 45.1% | 0.3301 |
| **A3** | Same-file ($M \le 50$) | 24.1% | 45.6% | 0.3381 |
| **A4** | Graph 1-hop ($M \le 50$) | 24.7% | 47.1% | 0.3409 |
| **A5** | Null graph control | 24.1% | 45.9% | 0.3315 |
| **A6** | Perron diffusion | 23.6% | 45.6% | 0.3301 |

- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, exact $p=0.1435$, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Under Schuirmann TOST ($\alpha=0.05$), the paired 90% CI $[+0.08, +3.98]$ pp excludes margins $\ge +4.0$ pp ($p < 0.05$); exploratory contrasts ($+0.87\text{--}+1.45$ pp) are null-compatible. Multi-step diffusion dilutes precision into hubs ($A_6$ 45.6% vs $A_4$ 47.1%), with discrimination dropping from 0.7966 on seeds to 0.6596 on graph neighbors via confounder truncation selection.
- **Loss Funnel & Reader**: Stage 1 Seeds capture 51.2% (48.8 pp loss), Stage 2 Graph ($A_4$) reaches 58.4% (41.6 pp loss), and Stage 3 Reranking retains 47.1% Top-5 (11.3 pp compaction loss). Upstream boundary loss exceeds compaction loss by **$3.68\times$** ($N=100$ reader test: Gemma 4 recovers 74.0% of randomized oracle targets vs 33.0% under BM25 order).

### 4.3 First-Moment Break-Even Operational Diagnostic & Stratum Asymmetry

Expanding pools from $M=10 \to 50$ increases recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). By the Wilcoxon--Mann--Whitney identity, expected rank $\mathbb{E}[\text{Rank}] \le k$ requires $\text{AUC}^* \ge 1 - \frac{k - 1}{M - 1}$ (at mean $M = 24.5, k=5$, $\text{AUC}^* \ge 0.830$). By Markov's inequality, this bounds accuracy: $\mathbb{E}[\text{Rank}] \le k \implies P(\text{Rank} \le k) \ge 1/k$ (Proposition 1; Acc@5 $\ge 20\%$). At graph $\text{AUC} = 0.6596$, expected rank falls to $9.00$ ($17.68$ at $M=50$). Strata show polarization: on Stratum A, 11 reached targets yield 6 conversions (54.5%); on Stratum C, 14 reached targets yield 0 conversions ($-2$ tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Dual T4 GPUs across 58 reachable Stratum C tasks:

| Model ID | Architecture Archetype | Active / Total | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | Dynamic MoE (128 exp) | 3.8B / 26.0B | *Hardware Bound* | *52.0 GB req.* | *Exceeds T4* |

- **Scaling Findings**: (1) Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), with E4B exceeding Phi-4-mini (0.7836) and DeepSeek R1 Distill 7B (0.6489; 95% CI $[+0.21, +0.34]$); (2) Models $\le 4$B fit $\le 9.6$ GB VRAM, while Gemma 4 26B-A4B requires 52.0 GB, exceeding dual T4 capacity (29.1 GB).

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix & $M=15$ Bound Verification (N=131)

| Pool Architecture / Condition | M | Acc@1 (Direct $\to$ Think) | Acc@5 (Raw / Backfill) | MRR (Direct $\to$ Think) |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 ($B_0$) | 25 | 2.29% | 15.27% / 15.27% | 0.0862 |
| Seeds Only ($A_1^{(10)}$) | 10 | 3.05% $\to$ 6.87% | 10.69% / 15.27% | 0.0778 $\to$ 0.0865 |
| Depth BM25 ($A_2^{(25)}$) | 25 | 4.58% $\to$ 6.87% | 12.98% / 16.79% | 0.0937 $\to$ 0.0967 |
| Graph 1-Hop ($A_4^{(25)}$, $C_0$) | 25 | **4.58% $\to$ 9.92%** | **18.32% / 18.32%** | **0.0934 $\to$ 0.1358** |
| $C_1$: 3-Line Stub | 25 | 2.29% | 15.27% / 15.27% | 0.0778 |
| $C_2$: Static Provenance | 25 | 3.82% | 16.03% / 16.03% | 0.0899 |
| $C_3$: Provenance + Thinking | 25 | 9.16% | 17.56% / 17.56% | 0.1270 |
| Pruned Seeds ($A_1, M=15$) | 15 | 3.05% $\to$ 7.63% | 13.74% / 17.56% | 0.0821 $\to$ 0.1042 |
| Pruned Depth ($A_2, M=15$) | 15 | 4.58% $\to$ 8.40% | 14.50% / 18.32% | 0.0954 $\to$ 0.1189 |
| Pruned Graph ($A_4$ Direct) | 15 | 4.58% | 17.56% / 17.56% | 0.0970 |
| **Pruned Graph + Think ($C_0$)** | **15** | **9.92%** ($p=0.0391^*$) | **19.85% / 19.85%** | **0.1469** |

- **Baseline Parity & Reasoning Lift**: Under direct readout ($M \le 25$), graph ($A_4$) and depth BM25 ($A_2$) tie at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning doubles Top-1 on $A_4^{(25)}$ to **9.92% (13/131; $+5.34$ pp, Holm $p=0.0469^*$)** and lifts MRR $+45.4\%$, while backfilling from BM25 order preserves 18.32% Acc@5 ($p=0.6875$, where reflection tokens truncate ranks 2–5). Subgroups confirm reasoning filters seed distractors: 12/13 Top-1 hits reside in seeds ($6/28 \to 12/28, p=0.0312^*$), while pure graph additions convert $0/7 \to 1/7$ ($p=1.0000$).
- **Topological Provenance ($C_1\text{--}C_3$)**: Stub prompts ($C_1$, 2.29%) and static provenance ($C_2$, 3.82%) yield no lift over direct readout ($-0.76$ pp, $p=1.0000$). Provenance + thinking ($C_3$, 9.16%) matches unguided thinking ($C_0$, 9.92%; $p=1.0000$).
- **Diagnostic Verification ($M=15$, $\text{AUC}^*=0.714$)**: Pruning to $M=15$ lowers diagnostic $\text{AUC}^*$ to $\mathbf{0.714}$ (retaining 34/35 reachable defects). Top-1 is stationary at **9.92% (13/131)** ($+1$ graph conversion $1 \to 2$ offset by $-1$ seed displacement $12 \to 11$), while Acc@5 reaches **19.85% (26/131)** and MRR **0.1469**. Pure graph hits ($2/6$ Top-1, $3/6$ Top-5) reside at input slots $\ge 10$, proving immunity to BM25 backfilling. Across reachable targets, $M=15$ reaches **13/34 Top-1 (38.2%)** and **26/34 Top-5 (76.5%)**.

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated 58 reachable Stratum C union tasks on Dual T4 GPUs:

| Condition | Acc@1 | Acc@5 | MRR | Contrast vs Baseline |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0$: Natural Code** | 12.07% | 27.59% | 0.1804 | reference (7/58) |
| **$O_1$: $\alpha$-Renamed** | 10.34% | 24.14% | 0.1686 | $-1.73$ pp (6/58, $p = 1.0000$) |
| **$E_0$: Isolated Snippet** | 12.07% | 27.59% | 0.1804 | reference (7/58) |
| **$E_1$: Inline Comment** | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** ($p_{\text{uncorr}}=0.0386^*, p_{\text{holm}}=0.1543$) |
| **$E_2$: Linearized Trace** | 17.24% | 32.76% | 0.2440 | $+5.17$ pp (10/58, $p = 0.4531$) |
| **$E_3$: Edge List** | 15.52% | 32.76% | 0.2419 | $+3.45$ pp (9/58, $p = 0.6875$) |
| **$E_4$: Adjacency Map** | 15.52% | 34.48% | 0.2362 | $+3.45$ pp (9/58, $p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) shifts Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), confirming syntax sustains discrimination.
2. **Serialization**: 24/25 graph targets (96.0%) are 1-hop callers/callees. Inline provenance ($E_1$) reaches 25.86% Top-1 ($+13.79$ pp, $p_{\text{uncorr}}=0.0386^*$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$) by bridging seed identifiers.

### 4.7 Diagnostic Post-Training Dynamics: Shortcut Vulnerability and Mixed-Pool Degradation

I audited LoRA fine-tuning (`google/gemma-4-e2b-it`, $r=16, \alpha=32$) on 226 SWE-bench Lite tasks ($N=176$ train, $N=50$ val; zero Verified overlap) with logit slicing (`logits_to_keep=24`), converging in 3 epochs (loss $0.7280 \to 0.0002$, val Acc@1 100.0%):

| Held-Out Verified Cohort | Scorer Regime | Acc@1 | Acc@5 | MRR | Median Rank | Within-Pool AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph-Only ($A_4 \setminus A_1, N=25$)** | Base | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| **Graph-Only ($A_4 \setminus A_1, N=25$)** | **Post-LoRA** | **20.0%** | **56.0%** | **0.3702** | **4.0** | **0.6270** |
| **Reachable Stratum C ($N=35$)** | Base | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| **Reachable Stratum C ($N=35$)** | **Post-LoRA** | 8.6% | **42.9%** | 0.2596 | **7.0** | **0.6084** |

- **Post-Training Dynamics**: On isolated graph targets ($A_4 \setminus A_1, N=25$), LoRA lifts Acc@1 by **$+20.00$ pp** ($n_{10}=5, n_{01}=0, p=0.0625$), Acc@5 by **$+12.00$ pp**, and doubles MRR ($0.1920 \to 0.3702$). Telemetry reveals rapid convergence stemmed from a comment shortcut. On mixed pools (Reachable Stratum C, $N=35$), the adapter over-promotes graph candidates over valid seeds, halving Top-1 (**17.1% $\to$ 8.6%**) and degrading MRR ($0.2895 \to 0.2596$). Shared repositories (53.4% Django) confirm within-repo cross-task adaptation rather than cross-domain generalization. Weights: `yvliet/perron-gemma4-e2b-lora`.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates show small graph offsets (+0.87 to +2.03 pp); margins $\ge +4.0$ pp are excluded under Schuirmann TOST ($p < 0.05$).
- **Selection Conditioning**: Discrimination collapse on graph targets (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, $p=0.72$). Sub-chance BM25 AUC ($\le 0.30$) reflects confounder truncation selection on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **CSR Filtering**: Fast diffusion grounds candidates in $<5.2$ ms ($\le 3$ MB RAM, 58.4% recall); (2) **Pruning**: Cap pools at $M=15$ ($\text{AUC}^*=0.714$) with $\le 180$ reasoning tokens; (3) **Selective Routing**: Routing via identifier density and seed skewness achieves 47.38% Acc@5, saving 31.4% compute; (4) **Post-Training Caution**: Surface-marker overfitting cautions against unregularized tuning.
- **Threats & Tooling**: AST editor achieves 0.0% syntax errors and 76.7% valid apply via rollbacks vs 30.0% valid apply and 50.0% syntax errors for `git apply` ($p < .001$).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **Binary CSR Graphs**: 12 repo graphs, `scores.parquet` (19,298 evaluations), and 800 strata labels.
- **Engine & Tooling**: CSR diffusion converges in $1.89\text{--}5.10$ ms ($<3$ MB RAM) vs SciPy direct `spsolve` (28.1 ms). AST editing achieves 0.0% syntax errors via rollbacks vs 50.0% for unified diffs ($p < .001$).
- **API & Weights**: `perron.diffuse_with_provenance()`, MCP server (`perron-mcp`), Dual T4 image, and LoRA diagnostic weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion succeeds by pairing test-time reasoning with bound-guided pruning: at $M=25$, reasoning doubles Top-1 on $A_4$ to 9.92% ($+5.34$ pp, Holm $p = 0.0469^*$; MRR $+45.4\%$) via seed distractor immunity ($6/28 \to 12/28, p=0.0312^*$), while backfilling maintains Top-5 parity. Tightening pools to $M=15$ lowers the diagnostic threshold ($\text{AUC}^* = 0.714$), converting 2/6 pure graph additions to Top-1 (38.2% reachable conversion, stationary net Top-1). Conversely, post-training audits reveal that uncalibrated LoRA tuning overfits prompt syntax, halving mixed-pool Top-1 ($17.1\% \to 8.6\%$). Together, Perron provides a high-throughput CSR foundation and pre-registered benchmarks for local developer agents.

---

## References

- [1] P. Gauthier. *Aider: AI pair programming in your terminal.* 2023.
- [2] S. Ouyang et al. *RepoGraph.* ICLR 2025.
- [3] X. Liu et al. *CodexGraph.* NAACL 2025.
- [4] Z. Chen et al. *LocAgent.* ACL 2025.
- [5] C. S. Xia et al. *Agentless.* FSE 2025.
- [6] C. E. Jimenez et al. *SWE-bench.* ICLR 2024.
- [7] J. Klicpera et al. *Predict then Propagate.* ICLR 2019.
- [8] B. Fatemi et al. *Talk like a graph.* ICLR 2024.
- [9] C. Sanford et al. *Understanding transformer reasoning capabilities via graph algorithms.* NeurIPS 2024.
- [10] J. Dong et al. *Improving RAG with Graph-based Reranking.* arXiv:2405.18414, 2024.
