# When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 Verified tasks, Personalized PageRank expansion matches depth-matched lexical retrieval under pointwise scoring (Acc@5 47.1% vs 45.1%, exact McNemar $p = 0.1435$, step-down Holm $p = 0.2869$; 95% bootstrap CI [-0.29, +4.36] pp; power 39.6%, $N \approx 935$ required for +2 pp), while a clustered GLMM controlling for 12 repository clusters reveals an adjusted advantage ($\text{OR} = 1.118, p = 0.0449$). A First-Moment Expected-Rank Identity ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) formalizes candidate selection conditioning where discrimination drops from 0.7966 to 0.6596 on graph neighbors. On 131 un-named defects, test-time reasoning doubles Top-1 at $M=25$ ($6 \to 13$ hits, unadjusted $p = 0.0156^*$, Holm $p = 0.0469^*$) by filtering seed distractors ($6/28 \to 12/28, p=0.0312^*$), and bounded pruning ($M=15, \text{AUC}^* = 0.714$) converts 2/6 graph additions to Top-1 (19.85% Acc@5). A positive scaling control with Gemma 4 E4B (4B) reaches 16.79% Top-1 and 37.4% Top-5 (MRR 0.2562 vs BM25 0.0862). On SWE-bench Lite ($N=171$ train, $N=55$ val), LoRA lifts isolated graph Top-1 ($0.0\% \to 24.0\%, p=0.0312^*$) and MRR ($0.1920 \to 0.3288$), but regresses Acc@5 ($44\% \to 36\%$) below non-neural heuristics; mixed-pool discordance ($n_{10}=6, n_{01}=4, p=0.7539$) shows null-compatible transfer and seen-repo memorization. I release depth-matched pools, 19,298 evaluations, 707 strata labels, and the Perron CSR engine (<5.2 ms solve, <=3.0 MB RAM).

---

## 1. Introduction

Local developer agents must localize defects within tight contexts ($K \le 4,096$). On 344 Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

Five contributions resolve this disconnect:
1. **Matched Parity & GLMM**: Marginal expansion matches depth BM25 ($\Delta = +2.03$ pp, power 39.6%); clustered GLMM isolates an adjusted lift ($\text{OR} = 1.118, p = 0.0449$).
2. **First-Moment Bound**: Expected-rank identity ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) formalizes discrimination collapse to 0.6596 under selection conditioning.
3. **Scaling & Positive Control**: Monotonic tail scaling ($0.8544 \to 0.9274$); 4B control reaches 16.79% Top-1, 37.4% Top-5 (MRR 0.2562); 52.0 GB MoE bound.
4. **Reasoning & Pruning**: Reasoning doubles Top-1 at $M=25$ via seed filtering; $M=15$ pruning ($\text{AUC}^* = 0.714$) converts 2/6 graph targets (38.2% conversion).
5. **Post-Training Dynamics**: LoRA (171 train, 55 val) lifts isolated Top-1 ($0.0\% \to 24.0\%, p=0.0312^*$) but regresses Acc@5 ($44\% \to 36\%$); mixed discordance ($n_{10}=6, n_{01}=4, p=0.7539$) shows seen-repo memorization.

---

## 2. Related Work

- **Diffusion & Scaling**: APPNP [7] and HippoRAG decouple PageRank from supervision but accumulate hub drift on code. SweRank [11] trains 7B models; PRFL [12] used PageRank in SE.
- **Graph Complexity & Reasoning**: Fatemi et al. [8] and Sanford et al. [9] proved depth bounds for multi-hop graph reachability. I show intermediate reasoning tokens eliminate explicit edge verbalization.
- **Agent Fault Localization**: Aider [1], CodexGraph [3], Agentless [5], and LocAgent [4] study agentic localization. Perron evaluates compact edge models (2B/4B) zero-shot on static CSR multigraphs.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). Regex classification is validated via independent human audit on $N=100$ tasks (95.0% concordance, Cohen's $\kappa=0.918$, 95% CI [0.849, 0.988]). Auditing all 56 snapshot exclusions (`FILE_ABSENT_PREIMAGE_DRIFT`) at base commits confirms 100% parse in Python 3 (68 files, 3,827 symbols; restoring Stratum C to 183 tasks); overturning equivalence requires $\ge +9$ net wins ($p = 0.0016$).
- **Perron CSR Engine**: CSR arrays index typed edges ($\boldsymbol{\omega} = [0.50, 0.25, 0.15, 0.10]$; $1.89\text{--}5.10$ ms solve, $<3$ MB RAM vs SuperLU at 28.1 ms). Operator $M = \beta(T + \mathbf{d}\mathbf{p}_0^\top) + (1-\beta)\mathbf{1}\mathbf{p}_0^\top$ ($\beta=0.85$) has Brauer bound $|\lambda_2(M)| \le \beta$ (gap $\ge 0.15$, contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$). Damping ($\gamma=0.70$) yields $\text{HSI@10} = 99.5\%$, saving 91.9% headroom (283.8 tokens across 10 stubs).
- **Candidate Pools & Scoring**: Top-10 BM25 seeds ($M_t \in [10, 50]$, mean 24.5) define pools B0--A6 under pre-specified execution plans. Frozen `gemma-4-e2b-it` scores log-odds (19,298 passes). Contrasts PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) evaluate exact McNemar tests with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4%. Reachability is 57.8% at 1 hop, 66.9% at 2 hops. A 186-config sweep fits a quadratic GLMM in $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($R^2 = 0.197, p < 0.001$), peaking at $n_{\text{eff}}^* \approx 8.8$ (50.9%). Non-Django jackknife ($N=162$) shows parity (+2.01 pp, $p=0.289$, power 23.5%).

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation

| Arm | Candidate ($M \le 50$) | Acc@1 | Acc@5 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | BM25 unscored | 25.3% | 44.8% | 0.3439 |
| **A1** | Seeds ($M=10$) | 24.7% | 46.2% | 0.3348 |
| **A2** | Depth BM25 | 23.8% | 45.1% | 0.3301 |
| **A3** | Same-file | 24.1% | 45.6% | 0.3381 |
| **A4** | Graph 1-hop | 24.7% | 47.1% | 0.3409 |
| **A5** | Null graph | 24.1% | 45.9% | 0.3315 |
| **A6** | Perron diffusion | 23.6% | 45.6% | 0.3301 |

- **Statistical Testing & Mixed-Effects GLMM**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, exact $p=0.1435$, Holm $p=0.287$). Clustered logistic GLMM with repo random intercepts reveals a significant adjusted lift for $A_4$ over $A_2$ ($\text{OR} = 1.118$, Wald $z = 2.006, p = 0.0449$) and confirms Stratum C hardness ($\text{OR} = 0.100, p < 0.001$). Under Schuirmann TOST ($\alpha=0.05$), score CI $[+0.07, +4.24]$ pp excludes margins $\ge +4.4$ pp ($p = 0.0402$), but 95% score CI $[-0.35, +4.74]$ exceeds $\pm 4.4$ pp (power 39.6% at $N=344$; $N\approx 935$ required for 80% power at +2 pp). Multi-step diffusion dilutes precision into hubs ($A_6$ 45.6% vs $A_4$ 47.1%), as discrimination drops from 0.7966 to 0.6596 on graph neighbors via selection conditioning.
- **Loss Funnel & Reader**: Seeds capture 51.2% (48.8 pp loss), Graph ($A_4$) reaches 58.4% (41.6 pp loss), and Reranking retains 47.1% Top-5 (11.3 pp compaction loss; $N=100$: Gemma 4 recovers 74.0% of oracle targets vs 33.0% under BM25 order).

### 4.3 First-Moment Expected-Rank Identity & Stratum Asymmetry

Expanding $M=10 \to 50$ raises recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). By Wilcoxon--Mann--Whitney, $\mathbb{E}[\text{Rank}] \le k$ requires $\text{AUC}^* \ge 1 - \frac{k - 1}{M - 1}$ ($\text{AUC}^* \ge 0.830$ at mean $M = 24.5, k=5$), with Proposition 1 bounding $\operatorname{Acc@}k \ge 1/k$. At graph $\text{AUC} = 0.6596$, expected rank falls to $9.00$ ($17.68$ at $M=50$). Strata polarize: Stratum A converts 6/11 reached targets (54.5%); Stratum C converts 0/14 ($-2$ tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Dual T4 GPUs across 58 reachable Stratum C tasks:

| Model | Archetype | Params | ROC-AUC | VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini-instruct` | Synthetic | 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-Distill-7B` | RL Distill | 7.0B | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B / 26B | *Bound* | *52.0 GB req.* | *Exceeds T4* |

- **Scaling Findings & Positive Control**: (1) Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), exceeding Phi-4-mini (0.7836) and DeepSeek R1 Distill 7B (0.6489; 95% CI $[+0.21, +0.34]$); (2) Dense models fit $\le 9.6$ GB VRAM, while Gemma 4 26B-A4B requires 52.0 GB (fused 3D expert tensors), exceeding Dual T4 capacity (29.1 GB); (3) **Positive Control ($N=131$)**: Evaluating `gemma-4-e4b-it` across all Stratum C tasks lifts Acc@1 from 0.76% to **16.79%**, Acc@5 from 12.21% to **37.4%**, Acc@10 from 18.32% to **41.22%**, and MRR from 0.0523 to **0.2562** (vs BM25 0.0862), confirming 2B null results reflected a model capacity floor.

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix & $M=15$ Verification (N=131)

| Condition | M | Acc@1 | Acc@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 | 25 | 2.29% | 15.27% | 0.0862 |
| Seeds Only | 10 | 3.05% $\to$ 6.87% | 10.69% $\to$ 15.27% | 0.0778 $\to$ 0.0865 |
| Depth BM25 | 25 | 4.58% $\to$ 6.87% | 12.98% $\to$ 16.79% | 0.0937 $\to$ 0.0967 |
| Graph 1-Hop ($C_0$) | 25 | **4.58% $\to$ 9.92%** | **18.32%** | **0.0934 $\to$ 0.1358** |
| $C_1$: 3-Line Stub | 25 | 2.29% | 15.27% | 0.0778 |
| $C_2$: Static Prov | 25 | 3.82% | 16.03% | 0.0899 |
| $C_3$: Prov + Think | 25 | 9.16% | 17.56% | 0.1270 |
| Pruned Seeds ($M=15$) | 15 | 3.05% $\to$ 7.63% | 13.74% $\to$ 17.56% | 0.0821 $\to$ 0.1042 |
| Pruned Depth ($M=15$) | 15 | 4.58% $\to$ 8.40% | 14.50% $\to$ 18.32% | 0.0954 $\to$ 0.1189 |
| Pruned Graph | 15 | 4.58% | 17.56% | 0.0970 |
| **Pruned Graph + Think** | **15** | **9.92%** ($p=0.0391^*$) | **19.85%** | **0.1469** |

- **Baseline Parity & Reasoning Lift**: Direct readout ($M \le 25$) ties graph ($A_4$) and depth BM25 ($A_2$) at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning doubles Top-1 on $A_4^{(25)}$ to **9.92% (13/131; $+5.34$ pp, Holm $p=0.0469^*$)** and MRR ($0.0934 \to 0.1358$), with deterministic backfilling preserving 18.32% Acc@5 ($p=0.6875$). Contrasts ($A_4\text{+think}$ vs $A_2\text{+think}$) show 13 vs 9 hits ($p=0.2188$) at $M=25$ and 13 vs 11 ($p=0.5000$) at $M=15$. Reasoning acts as a seed distractor filter: 12/13 Top-1 hits reside in seeds ($6/28 \to 12/28, p=0.0312^*$), while pure graph additions convert $0/7 \to 1/7$ ($p=1.0000$).
- **Topological Provenance ($C_1\text{--}C_3$)**: Stubs ($C_1$, 2.29%) and static provenance ($C_2$, 3.82%) yield no lift ($-0.76$ pp, $p=1.0000$); provenance + thinking ($C_3$, 9.16%) ties thinking ($C_0$, 9.92%, $p=1.0000$).
- **Diagnostic Verification ($M=15$, $\text{AUC}^*=0.714$)**: Pruning to $M=15$ lowers $\text{AUC}^*$ to $\mathbf{0.714}$ and cuts distractors ($8.00 \to 4.77$), retaining 34/35 reachable defects. Top-1 remains **9.92% (13/131)**, while Acc@5 reaches **19.85% (26/131)** and MRR **0.1469** (converting 2/6 pure graph targets to Top-1). On reachable targets, $M=15$ reaches **13/34 Top-1 (38.2%)** and **26/34 Top-5 (76.5%)**.

### 4.6 Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated 58 reachable Stratum C union tasks on Dual T4 GPUs:

| Condition | Acc@1 | Acc@5 | MRR | Δ Top-1 |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0 / E_0$: Natural Code** | 12.07% | 27.59% | 0.1804 | ref (7/58) |
| **$O_1$: $\alpha$-Renamed** | 10.34% | 24.14% | 0.1686 | $-1.73$ pp ($p = 1.0000$) |
| **$E_1$: Inline Comment** | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** ($p_{\text{holm}}=0.1543$) |
| **$E_2$: Linearized Trace** | 17.24% | 32.76% | 0.2440 | $+5.17$ pp ($p = 0.4531$) |
| **$E_3$: Edge List** | 15.52% | 32.76% | 0.2419 | $+3.45$ pp ($p = 0.6875$) |
| **$E_4$: Adjacency Map** | 15.52% | 34.48% | 0.2362 | $+3.45$ pp ($p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) shifts Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), confirming syntax sustains discrimination.
2. **Serialization**: In evaluated union pools, targets span callers and callees. Inline comments ($E_1$) reach 25.86% Top-1 ($+13.79$ pp, $p_{\text{uncorr}}=0.0386^*$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$) by bridging seed identifiers. Lacking a placebo, this observational lift cannot rule out saliency artifacts.

### 4.7 Post-Training Dynamics: Top-1 Bias Shifts vs. Global Rank Variance

I tuned `gemma-4-e2b-it` (LoRA $r=16$) on 226 Lite tasks via repository GroupKFold (171 train, 55 val; loss $0.9238 \to 0.0631$, val Acc@1 56.4%, Acc@5 81.8%). On Verified, 58.1% of tasks belong to Django/Astropy, measuring within-repo adaptation rather than transfer.

| Cohort | Scorer | Acc@1 | Acc@5 | MRR | Med | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Graph ($N=25$) | Base E2B | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| | Graph-First | 8.0% | **52.0%** | 0.2518 | **5.0** | **0.7056** |
| | **Clean LoRA** | **24.0%** | 36.0% | **0.3288** | 9.0 | 0.5859 |
| | *Shortcut Probe* | *20.0%* | *56.0%* | *0.3702* | *4.0* | *0.6270* |
| Strat C ($N=35$) | Base E2B | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| | Graph-First | 0.0% | 17.1% | 0.0993 | 16.0 | 0.3753 |
| | **Clean LoRA** | **22.9%** | **48.6%** | **0.3623** | **6.0** | **0.6392** |
| | *Shortcut Probe* | *8.6%* | *42.9%* | *0.2596* | *7.0* | *0.6084* |

- **Empirical Dynamics**: On isolated graph targets ($N=25$), clean LoRA lifts Top-1 from 0.0% to **24.0%** (6 hits: Django [4], Flask [1], Matplotlib [1]; $n_{10}=6, n_{01}=0, p = 0.0312^*$) and MRR ($0.1920 \to 0.3288$), but regresses Acc@5 ($44.0\% \to 36.0\%$) and AUC ($0.6063 \to 0.5859$) below Graph-First (52.0% Acc@5, 0.7056 AUC). On mixed pools ($N=35$), clean LoRA reaches **22.9%** Top-1 ($6 \to 8$ hits) and **48.6%** Top-5 ($13 \to 17$ hits; $n_{10}=6, n_{01}=4, p = 0.7539$; $n_{10}=10, n_{01}=6, p = 0.4545$). A comment probe confirms Cohort A rewards non-seed bias while collapsing mixed Top-1 ($17.1\% \to 8.6\%$). Fine-tuning induces seen-repo memorization and trailing rank variance.

---

## 5. Discussion

- **Empirical Verdict & Clustered GLMM**: Marginal tests show small offsets (+0.87 to +2.03 pp; power 39.6%, 95% CI $[-0.29, +4.36]$ pp). Clustered GLMM across 12 repos isolates an adjusted lift for graph expansion ($\text{OR} = 1.118, p = 0.0449$).
- **Selection Conditioning**: Graph discrimination collapse (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, $p=0.72$), reflecting truncation selection on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **CSR Filtering**: Solves in $<5.2$ ms ($\le 3$ MB RAM, 58.4% recall); (2) **Pruning**: Cap at $M=15$ ($\text{AUC}^*=0.714$) with $\le 180$ reasoning tokens; (3) **Positive Control**: E4B reaches 16.79% Top-1 and 37.4% Top-5 (MRR 0.2562); (4) **Routing**: Saves 31.4% compute; (5) **Post-Training**: Surface overfitting cautions against tuning.
- **Tooling Safety**: AST editor achieves 0.0% syntax errors and 76.7% valid apply via rollbacks vs 30.0% apply and 50.0% syntax errors for `git apply` ($p < .001$).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0 (`https://github.com/yvliet/perron`):
- **Data & Documentation**: 12 CSR multigraphs (109,240 nodes, 341,892 edges), 707 strata labels, and `scores.parquet` (19,298 evaluations), documented via a formal *Datasheet for Datasets* (Gebru et al., 2021) and SHA-256 cryptographic manifests.
- **Engine**: CSR diffusion in $1.89\text{--}5.10$ ms ($<3$ MB RAM) vs SuperLU (28.1 ms). AST editor achieves 0.0% syntax errors via rollbacks vs 50.0% for diffs ($p < .001$).
- **Code & Tooling**: `perron.diffuse_with_provenance()`, `perron-mcp` JSON-RPC server, and fine-tuned LoRA weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

Naive code graph expansion fails under small models not because graphs lack defect targets, but because low-overlap neighbors fall below expected-rank discrimination via selection conditioning, while unpruned graphs dilute seed precision. Reasoning acts as a seed distractor filter ($6 \to 12, p=0.0312^*$), while bounded pruning ($M=15$) converts 2/6 graph targets to Top-1 (38.2% reachable conversion). A 4B positive control achieves 16.79% Top-1 and 37.4% Top-5, proving 2B null results reflect a model capacity floor. Clean LoRA demonstrates Top-1 reweighting on sparse targets (0.0% to 24.0%, $p=0.0312^*$) but introduces trailing rank variance; comment shortcuts induce mixed collapse ($17.1\% \to 8.6\%$). Perron provides a fast CSR foundation for local developer agents.

---

## References

- [1] P. Gauthier. *Aider* (2023). [2] S. Ouyang et al. *RepoGraph* (ICLR 2025). [3] X. Liu et al. *CodexGraph* (NAACL 2025). [4] Z. Chen et al. *LocAgent* (ACL 2025). [5] C. S. Xia et al. *Agentless* (FSE 2025).
- [6] C. E. Jimenez et al. *SWE-bench* (ICLR 2024). [7] J. Klicpera et al. *APPNP* (ICLR 2019). [8] B. Fatemi et al. *Talk like a graph* (ICLR 2024). [9] C. Sanford et al. *Understanding transformer reasoning* (NeurIPS 2024). [10] J. Dong et al. *Graph Reranking* (2024).
- [11] Z. Zhang et al. *SweRank* (ICLR 2026). [12] X. Li et al. *PRFL* (ISSTA 2017).
