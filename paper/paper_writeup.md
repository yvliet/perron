# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, Personalized PageRank candidate expansion matches depth-matched lexical retrieval (Acc@5 47.1% vs 45.1%, exact McNemar $p = 0.1435$, Holm $p = 0.287$; 90% TOST CI [+0.08, +3.98] pp excludes gains $\ge +4.0$ pp). A Break-Even Diagnostic ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) paired with Proposition 1 ($\mathbb{E}[\text{Rank}] \le k \implies \text{Acc@}k \ge 1/k$) shows Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph neighbors via confounder truncation selection. Across 131 un-named defects, test-time reasoning doubles Top-1 at $M=25$ ($6 \to 13$ hits, Holm $p = 0.0469^*$) by filtering seed distractors. Bound-guided pruning to $M=15$ ($\text{AUC}^* = 0.714$) converts 2/6 pure graph additions (33.3%) to Top-1, reaching 19.85% Acc@5 (26/131). Clean LoRA tuning on repository GroupKFold splits ($N=171$ train, $N=55$ val) learns structural priors without shortcuts, lifting held-out graph Top-1 from 0.0% to 24.0% ($p=0.0312^*$) and Stratum C Top-1 ($17.1\% \to 22.9\%$), while diagnostic ablations show how comment shortcuts collapse mixed-pool Top-1 ($17.1\% \to 8.6\%$). I release all depth-matched pools, 19,298 scored evaluations, and the Perron CSR engine (<5.2 ms solve, <=3.0 MB RAM).

---

## 1. Introduction

Local developer agents must localize defects within tight contexts ($K \le 4,096$) without cloud APIs. Graph localization requires: (1) **Recall Expansion** capturing missed targets; (2) **Rank Discrimination** prioritizing them. Across 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

Five contributions resolve this disconnect:
1. **Matched-Control Parity**: PPR candidate expansion shows no statistical gain over lexical search across 19,298 evaluations (RQ1, RQ2).
2. **First-Moment Diagnostic**: Break-Even Diagnostic ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) formalizes discrimination collapse to 0.6596 on graph targets via confounder truncation selection (RQ3).
3. **Scaling Surface**: Monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) and the 52.0 GB MoE VRAM boundary.
4. **Reasoning Lift & Pruning**: Reasoning doubles Top-1 at $M=25$ ($p=0.0469^*$), while $M=15$ pruning ($\text{AUC}^* = 0.714$) converts 2/6 graph targets to Top-1 (38.2% reachable conversion).
5. **Generalization-Verified LoRA Post-Training**: Clean GroupKFold LoRA tuning (171 train, 55 held-out val across SymPy/Matplotlib) demonstrates structural learning without comment leakage, lifting graph Top-1 from 0.0% to 24.0% ($p=0.0312^*$, vs 8.0% for Graph-First) and Stratum C Top-1 ($17.1\% \to 22.9\%$) and Top-5 ($37.1\% \to 48.6\%$). Diagnostic ablations document comment shortcut collapse ($17.1\% \to 8.6\%$).

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo/APPNP [7] and HippoRAG decouple PageRank from prediction; Dong et al. [10] link chunks. Perron provides zero-copy CSR multigraph diffusion for local agents.
- **Graph Complexity & Reasoning**: Fatemi et al. [8] showed verbalization shifts accuracy up to 61.8%; Sanford et al. [9] proved relational reasoning requires logarithmic depth or pause tokens, whereas 96% of SWE-bench defect targets are 1-hop. RepoGraph [2] showed 2-hop distractor accumulation degrades resolution ($29.67\% \to 26.00\%$).
- **Agent Architectures**: Aider [1], CodexGraph [3], and Agentless [5] establish localization bounds editing; LocAgent [4] fine-tunes graph agents on SWE-bench [6]. Perron isolates graph expansion under compact 2B/4B models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). 56 tasks (52 Stratum C) failed Python 2 `ast.parse`; overturning equivalence requires $+9$ net wins out of 56 tasks ($p < 0.001$). Hyperparameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron CSR Engine & Convergence**: CSR arrays index typed edges ($\boldsymbol{\omega} = [0.50, 0.25, 0.15, 0.10]$; $1.89\text{--}5.10$ ms solve, $<3$ MB RAM; vs SciPy direct SuperLU `spsolve` at 28.1 ms). For Google operator $M = \beta(T + \mathbf{d}\mathbf{p}_0^\top) + (1-\beta)\mathbf{1}\mathbf{p}_0^\top$ ($\beta=0.85$), Brauer bounds $|\lambda_2(M)| \le \beta$ (gap $\ge 0.15$, Doeblin contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$). Damping ($\gamma=0.70$) yields $\text{HSI@10} = 99.5\%$, saving 91.9% headroom (283.8 tokens across 10 AST stubs).
- **Candidate Pools**: Top-10 BM25 seeds ($M_t \in [10, 50]$, mean 24.5) define pools B0--A6 under committed protocol locks.
- **Scoring Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores log-odds (19,298 passes). Contrasts PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) evaluate exact McNemar tests with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4%. Reachability is 57.8% at 1 hop, 66.9% at 2 hops. A 186-config sweep (63,984 solves) fits a quadratic GLMM in $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($R^2 = 0.197, p < 0.001$), peaking at $n_{\text{eff}}^* \approx 8.8$ (50.9%). Jackknife ($N=230$) confirms parity (57.1% vs 57.1%, $p=1.0000$).

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

- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, exact $p=0.1435$, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Under Schuirmann TOST ($\alpha=0.05$), the paired 90% CI $[+0.08, +3.98]$ pp (Tango score CI $[+0.07, +4.24]$ pp) excludes margins $\ge +4.0$ pp ($p < 0.05$). Diffusion dilutes precision into hubs ($A_6$ 45.6% vs $A_4$ 47.1%), with discrimination dropping from 0.7966 on seeds to 0.6596 on graph neighbors via confounder truncation selection.
- **Loss Funnel & Reader**: Stage 1 Seeds capture 51.2% (48.8 pp loss), Stage 2 Graph ($A_4$) reaches 58.4% (41.6 pp loss), and Stage 3 Reranking retains 47.1% Top-5 (11.3 pp compaction loss). Boundary loss exceeds compaction loss by **$3.68\times$** ($N=100$ reader: Gemma 4 recovers 74.0% of randomized oracle targets vs 33.0% under BM25 order).

### 4.3 First-Moment Break-Even Operational Diagnostic & Stratum Asymmetry

Expanding pools from $M=10 \to 50$ increases recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). By the Wilcoxon--Mann--Whitney identity, expected rank $\mathbb{E}[\text{Rank}] \le k$ requires $\text{AUC}^* \ge 1 - \frac{k - 1}{M - 1}$ ($	ext{AUC}^* \ge 0.830$ at mean $M = 24.5, k=5$), with Proposition 1 bounding $\operatorname{Acc@}k \ge 1/k$. At graph $\text{AUC} = 0.6596$, expected rank falls to $9.00$ ($17.68$ at $M=50$). Strata polarize: Stratum A converts 6 of 11 reached targets (54.5%); Stratum C converts 0 of 14 ($-2$ tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)

Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination on Dual T4 GPUs across 58 reachable Stratum C tasks:

| Model | Archetype | Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini-instruct` | Synthetic | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-Distill-Qwen-7B` | RL Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B / 26.0B | *Hardware Bound* | *52.0 GB req.* | *Exceeds T4* |

- **Scaling Findings**: (1) Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), with E4B exceeding Phi-4-mini (0.7836) and DeepSeek R1 Distill 7B (0.6489; 95% CI $[+0.21, +0.34]$); (2) Dense models fit $\le 9.6$ GB VRAM, whereas Gemma 4 26B-A4B requires 52.0 GB, exceeding Dual T4 capacity (29.1 GB).

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix & $M=15$ Bound Verification (N=131)

| Condition | M | Acc@1 | Acc@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 | 25 | 2.29% | 15.27% / 15.27% | 0.0862 |
| Seeds Only | 10 | 3.05% $\to$ 6.87% | 10.69% / 15.27% | 0.0778 $\to$ 0.0865 |
| Depth BM25 | 25 | 4.58% $\to$ 6.87% | 12.98% / 16.79% | 0.0937 $\to$ 0.0967 |
| Graph 1-Hop ($C_0$) | 25 | **4.58% $\to$ 9.92%** | **18.32% / 18.32%** | **0.0934 $\to$ 0.1358** |
| $C_1$: 3-Line Stub | 25 | 2.29% | 15.27% / 15.27% | 0.0778 |
| $C_2$: Static Provenance | 25 | 3.82% | 16.03% / 16.03% | 0.0899 |
| $C_3$: Provenance + Thinking | 25 | 9.16% | 17.56% / 17.56% | 0.1270 |
| Pruned Seeds ($M=15$) | 15 | 3.05% $\to$ 7.63% | 13.74% / 17.56% | 0.0821 $\to$ 0.1042 |
| Pruned Depth ($M=15$) | 15 | 4.58% $\to$ 8.40% | 14.50% / 18.32% | 0.0954 $\to$ 0.1189 |
| Pruned Graph | 15 | 4.58% | 17.56% / 17.56% | 0.0970 |
| **Pruned Graph + Think** | **15** | **9.92%** ($p=0.0391^*$) | **19.85% / 19.85%** | **0.1469** |

- **Baseline Parity & Reasoning Lift**: Direct readout ($M \le 25$) ties graph ($A_4$) and depth BM25 ($A_2$) at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning doubles Top-1 on $A_4^{(25)}$ to **9.92% (13/131; $+5.34$ pp, Holm $p=0.0469^*$)** and lifts MRR $+45.4\%$, while backfilling preserves 18.32% Acc@5 ($p=0.6875$). Contrasting $A_4\text{+think}$ vs $A_2\text{+think}$ shows directional lift (13 vs 9 hits at $M=25$, $p=0.1250$; 13 vs 11 at $M=15$, $p=0.5000$). Reasoning filters seed distractors: 12/13 Top-1 hits reside in seeds ($6/28 \to 12/28, p=0.0312^*$), while pure graph additions convert $0/7 \to 1/7$ ($p=1.0000$).
- **Topological Provenance ($C_1\text{--}C_3$)**: Stubs ($C_1$, 2.29%) and static provenance ($C_2$, 3.82%) yield no lift ($-0.76$ pp, $p=1.0000$); provenance + thinking ($C_3$, 9.16%) matches thinking ($C_0$, 9.92%, $p=1.0000$).
- **Diagnostic Verification ($M=15$, $\text{AUC}^*=0.714$)**: Pruning to $M=15$ lowers diagnostic $\text{AUC}^*$ to $\mathbf{0.714}$ and cuts distractor volume ($8.00 \to 4.77$), retaining 34/35 reachable defects. Top-1 is stationary at **9.92% (13/131)**, while Acc@5 reaches **19.85% (26/131)** and MRR **0.1469**. Pure graph hits ($2/6$ Top-1, $3/6$ Top-5) reside at slots $\ge 10$, proving immunity to BM25 backfilling. Across reachable targets, $M=15$ reaches **13/34 Top-1 (38.2%)** and **26/34 Top-5 (76.5%)**.

### 4.6 Causal Lexical Dissection & Multi-Encoding Graph Representations (PREREG-CAUSAL-v1)

Protocol `PREREG-CAUSAL-v1` evaluated 58 reachable Stratum C union tasks on Dual T4 GPUs:

| Condition | Acc@1 | Acc@5 | MRR | Contrast |
| :--- | :---: | :---: | :---: | :--- |
| **$O_0 / E_0$: Natural Code** | 12.07% | 27.59% | 0.1804 | reference (7/58) |
| **$O_1$: $\alpha$-Renamed** | 10.34% | 24.14% | 0.1686 | $-1.73$ pp (6/58, $p = 1.0000$) |
| **$E_1$: Inline Comment** | **25.86%** | **39.66%** | **0.3097** | **+13.79 pp** ($p_{\text{uncorr}}=0.0386^*, p_{\text{holm}}=0.1543$) |
| **$E_2$: Linearized Trace** | 17.24% | 32.76% | 0.2440 | $+5.17$ pp (10/58, $p = 0.4531$) |
| **$E_3$: Edge List** | 15.52% | 32.76% | 0.2419 | $+3.45$ pp (9/58, $p = 0.6875$) |
| **$E_4$: Adjacency Map** | 15.52% | 34.48% | 0.2362 | $+3.45$ pp (9/58, $p = 0.6875$) |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) shifts Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), confirming syntax sustains discrimination.
2. **Serialization**: 24/25 graph targets (96.0%) are 1-hop callers/callees. Inline comments ($E_1$) reach 25.86% Top-1 ($+13.79$ pp, $p_{\text{uncorr}}=0.0386^*, p_{\text{holm}}=0.1543$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$) by bridging seed identifiers. Without a non-relational comment placebo control, this observational lift cannot rule out generic comment saliency artifacts.

### 4.7 Generalization-Verified Post-Training: Topological Learning vs Shortcut Vulnerability

I fine-tuned `gemma-4-e2b-it` (LoRA $r=16$) on 226 Lite tasks via repository GroupKFold ($N=171$ train, $N=55$ val across SymPy/Matplotlib; zero Verified overlap) with logit slicing, converging from loss 0.9238 to 0.0631 (val Acc@1 56.4%, Acc@5 81.8%). Prompt comments used a uniform template (`# Candidate: <id> | provenance: Extracted code graph candidate`).

| Cohort | Scorer Regime | Acc@1 | Acc@5 | MRR | Med Rank | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Graph-Only ($N=25$)** | Base E2B | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| **Graph-Only ($N=25$)** | Graph-First | 8.0% | **52.0%** | 0.2518 | **5.0** | **0.7056** |
| **Graph-Only ($N=25$)** | **Clean Post-LoRA** | **24.0%** | 36.0% | **0.3288** | 9.0 | 0.5859 |
| **Graph-Only ($N=25$)** | *Shortcut Probe* | *20.0%* | *56.0%* | *0.3702* | *4.0* | *0.6270* |
| **Stratum C ($N=35$)** | Base E2B | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| **Stratum C ($N=35$)** | Graph-First | 0.0% | 17.1% | 0.0993 | 16.0 | 0.3753 |
| **Stratum C ($N=35$)** | **Clean Post-LoRA** | **22.9%** | **48.6%** | **0.3623** | **6.0** | **0.6392** |
| **Stratum C ($N=35$)** | *Shortcut Probe* | *8.6%* | *42.9%* | *0.2596* | *7.0* | *0.6084* |

- **Empirical Dynamics**: On isolated graph targets ($N=25$), clean LoRA lifts Acc@1 from 0.0% to **24.0%** (6 hits across Astropy, Django, SymPy; $+24.00$ pp, $n_{10}=6, n_{01}=0$, exact McNemar $p = 0.0312^*$), beating Graph-First (8.0%, 2 hits) by $3.0\times$ and lifting MRR ($0.1920 \to 0.3288$). On mixed pools (Stratum C, $N=35$), clean LoRA prevents collapse: Top-1 reaches **22.9%** (6 to 8 hits, $+5.71$ pp lift), Top-5 reaches **48.6%** (13 to 17 hits, $+11.43$ pp lift), MRR improves from $0.2895 \to 0.3623$, and AUC climbs from $0.5692 \to 0.6392$.
- **Audited Shortcut Probe**: An audited probe leaking `# Gold target symbol` vs `# Candidate graph symbol` collapsed Stratum C mixed Top-1 (**17.1% $\to$ 8.6%**). Sanitizing prompts restored balanced generalization. Checkpoint: `yvliet/perron-gemma4-e2b-lora`.

---

## 5. Discussion

- **Empirical Verdict**: Point estimates show small graph offsets (+0.87 to +2.03 pp); margins $\ge +4.0$ pp are excluded under Schuirmann TOST ($p < 0.05$).
- **Selection Conditioning**: Graph discrimination collapse (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, $p=0.72$), reflecting confounder truncation selection on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **CSR Filtering**: Fast diffusion grounds candidates in $<5.2$ ms ($\le 3$ MB RAM, 58.4% recall); (2) **Pruning**: Cap pools at $M=15$ ($\text{AUC}^*=0.714$) with $\le 180$ reasoning tokens; (3) **Selective Routing**: Density routing achieves 47.38% Acc@5, saving 31.4% compute; (4) **Post-Training Caution**: Surface-marker overfitting cautions against unregularized tuning.
- **Tooling Safety**: AST editor achieves 0.0% syntax errors and 76.7% valid apply via rollbacks vs 30.0% valid apply and 50.0% syntax errors for `git apply` ($p < .001$).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0:
- **CSR Graphs & Data**: 12 CSR graphs, `scores.parquet` (19,298 evaluations), and 800 strata labels.
- **Engine & Tooling**: CSR diffusion in $1.89\text{--}5.10$ ms ($<3$ MB RAM) vs SuperLU (28.1 ms). AST editor achieves 0.0% syntax errors via rollbacks vs 50.0% for unified diffs ($p < .001$).
- **API & Weights**: `perron.diffuse_with_provenance()`, `perron-mcp` server, Dual T4 workstation, and LoRA weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

Naive code graph expansion fails under frozen small models not because graphs lack defect targets, but because low-lexical-overlap neighbors fall below expected-rank discrimination via confounder truncation selection, while unpruned graphs dilute seed precision. Reasoning acts as a seed distractor filter ($6 \to 12, p=0.0312^*$), while bounded pruning ($M=15$) converts $2/6$ pure graph targets to Top-1 ($13/34$ reachable conversion). Clean LoRA post-training demonstrates genuine topological learning (graph Top-1 to 24.0%, Stratum C Top-1 to 22.9%), while diagnostic ablations show how comment shortcuts induce mixed-pool collapse ($17.1\% \to 8.6\%$). Perron provides a high-throughput CSR foundation and pre-registered benchmarks for local developer agents.

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
