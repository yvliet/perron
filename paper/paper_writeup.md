# When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 Verified tasks, 1-hop AST graph expansion achieves statistical parity with depth-matched lexical retrieval (Acc@5 47.1% vs 45.1%, step-down Holm $p = 0.2869$); formal equivalence within $\pm 2.0$ pp remains statistically inconclusive ($p = 0.512$). Grounding candidate discrimination in classical Wilcoxon-Mann-Whitney expectations ($\mathbb{E}[\text{Rank}] = 1 + (M-1)(1 - \text{AUC})$, Bamber 1975) explains why expansion stalls: candidate selection conditioning drops pairwise discrimination from $0.7966$ on seeds to $0.6596$ on graph neighbors ($p = 0.0036^*$), pushing expected target rank past the top-5 window. On 131 un-named defects, test-time reasoning ($\le 180$ tokens) improves Top-1 at $M=25$ ($4.58\% \to 9.92\%$, raw $p = 0.0156^*$, 4-test Holm $p = 0.0625$), acting primarily as a candidate disambiguation filter that separates true fault sites from close distractors. Candidate horizon pruning to $M=15$ cuts candidate stub tokens by 40%; paired with test-time reasoning, it compacts mean target rank from 5.32 to 3.74 with zero loss in reachable Top-1 hits. Across 162 non-Django tasks, topological expansion matches lexical search at 34.6% vs 32.7% ($p = 0.5811$). Parameter-efficient fine-tuning on SWE-bench Lite lifts isolated graph Top-1 to 24.0% ($p=0.0312^*$) but introduces trailing distractor variance that degrades broad-pool ranking ($44\% \to 36\%$). I release depth-matched candidate pools, 19,298 evaluations, 707 verified strata labels, and the reference Perron CSR engine under Apache 2.0.

---

## 1. Introduction

Local agents must localize defects within tight contexts ($K \le 4,096$). On 344 Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

Five contributions resolve this disconnect:
1. **Parity & Inconclusive Equivalence**: Expansion matches depth BM25 ($\Delta = +2.03$ pp, exact $p=0.1435$); formal $\pm 2.0$ pp equivalence is inconclusive ($p=0.512$, power 2.1%); marginal task GLM ($\text{OR} = 1.118, p = 0.0876$) and mixed model ($\text{OR} = 1.130, p = 0.349$) confirm non-significance.
2. **First-Moment Markov Floor**: Candidate inclusion requires $\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$; Proposition 1 bounds worst-case performance as a Markov floor ($1/k = 0.20$ at $M=25, k=5$). Discrimination drops to 0.6596 via candidate selection conditioning.
3. **Scaling & Positive Control**: Monotonic tail scaling ($0.8544 \to 0.9274$); 4B union control reaches 16.79% Top-1, 37.4% Top-5 (MRR 0.2562); 52.0 GB MoE bound (fused 3D tensors).
4. **Candidate Disambiguation & Horizon Compaction**: Test-time reasoning lifts 4B Top-1 at $M=25$ via candidate disambiguation ($6 \to 12, p=0.0312^*$), while $M=15$ pruning achieves 40% prompt reduction and rank compaction with zero Top-1 loss (13/131 in both pools; 13/34 reachable).
5. **Post-Training Dynamics**: Clean LoRA lifts isolated Top-1 to 24.0% ($p=0.0312^*$), but regresses Acc@5 ($44\% \to 36\%$) below a non-neural Graph-First heuristic, exhibiting training-set adaptation.

---

## 2. Related Work

- **Diffusion & Scaling**: APPNP [7] and HippoRAG decouple PageRank from supervision but accumulate hub drift. SweRank [11] trains 7B models; PRFL [12] applied PageRank to localization.
- **Graph Reasoning**: Fatemi et al. [8] and Sanford et al. [9] proved reachability depth bounds. I show reasoning tokens eliminate explicit edge verbalization in 4B models.
- **Agent Localization**: Aider [1], CodexGraph [3], Agentless [5], and LocAgent [4] benchmark localization. Perron evaluates compact models (2B/4B) zero-shot on static CSR multigraphs.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 Verified tasks [6] across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). Regex yields $N_A=168, N_B=7, N_C=169$ ($N=100$ audit: 95.0% concordance, Cohen's $\kappa=0.918$, 95% CI [0.849, 0.988]). Auditing 56 exclusions (`FILE_ABSENT_PREIMAGE_DRIFT`) confirms 100% Python 3 parse (68 files, 3,827 symbols); overturning equivalence requires $\ge +9$ net wins ($P(\operatorname{Bin}(56, 0.0494) \ge 9) = 0.0016$).
- **Perron CSR Engine**: Typed edges solve in $1.89\text{--}5.10$ ms ($<3$ MB RAM vs SuperLU at 28.1 ms). Operator $M$ ($\beta=0.85$) satisfies $|\lambda_2(M)| \le \beta$ (contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$). Damping ($\gamma=0.70$) yields $\text{HSI@10} = 99.5\%$.
- **Candidate Pools**: Top-10 BM25 seeds ($M_t \in [10, 50]$, mean 24.5) define pools B0--A6 under pre-specified plans (commit `6908e78` locked retrieval; `9abe405` pointwise; `c563441` reasoning). Frozen `gemma-4-e2b-it` scores log-odds (19,298 passes). Contrasts PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) evaluate McNemar tests with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm $p=1.000$); PPR reaches 50.0% (172/344); tag maps reach 40.0% (0.0% function accuracy by design); bi-encoders (`bge-small-en-v1.5`) reach 38.4%. Reachability is 57.8% at 1 hop, 66.9% at 2 hops. A 186-config sweep fits a GLMM in $n_{\text{eff}} = \exp(H(\mathbf{p}_0))$ ($R^2 = 0.197, p < 0.001$). Leave-Django-Out across $N=162$ non-Django tasks (47.1% of benchmark) exhibits parity: $A_4$ achieves 34.6% vs 32.7% for $A_2$ ($\Delta = +1.85$ pp, 56 vs 53 hits, $n_{10}=8, n_{01}=5$, exact McNemar $p=0.5811$, recall 50.0% vs 48.8%).

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation

| Arm | Candidate ($M \le 50$) | Acc@1 | Acc@5 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | BM25 | 25.3% | 44.8% | 0.3439 |
| **A1** | Seeds | 24.7% | 46.2% | 0.3348 |
| **A2** | Depth BM25 | 23.8% | 45.1% | 0.3301 |
| **A3** | Same-file | 24.1% | 45.6% | 0.3381 |
| **A4** | Graph 1-hop | 24.7% | 47.1% | 0.3409 |
| **A5** | Null graph | 24.1% | 45.9% | 0.3315 |
| **A6** | Perron | 23.6% | 45.6% | 0.3301 |

- **Statistical Testing & GLMM**: PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, exact $p=0.1435$, Holm $p=0.287$). Task-clustered GLM ($N=344$) indicates non-significance ($\text{OR} = 1.118, \text{SE}=0.065, p = 0.0876$; GEE $p = 0.0901$; repository mixed model $\text{OR} = 1.130, \text{SE}=0.130, z = 0.937, p = 0.349, \sigma_{\text{repo}} = 0.963$, accounting for Django at 52.9%). Under Schuirmann TOST ($\alpha=0.05$), equivalence within $\pm 2.0$ pp is inconclusive ($p = 0.512$; power 2.1% at $N=344$, requiring $N \approx 1{,}058$ tasks). Superiority MDE is 3.35 pp at 80% power ($N\approx 970$ for +2.0 pp). Score-based 90% Tango interval is $[+0.07, +4.24]$ pp. Hub diffusion dilutes precision ($A_6$ 45.6% vs $A_4$ 47.1%); discrimination drops $0.7966 \to 0.6596$ on graph neighbors via selection conditioning.
- **Loss Funnel**: Seeds capture 51.2% (48.8 pp loss); Graph reaches 58.4% (41.6 pp boundary loss: seed absence [48.8 pp], 2-hop radius [43.7%]); Reranking retains 47.1% Top-5 (11.3 pp loss; $N=100$: Gemma 4 recovers 74.0% oracle targets vs 33.0% BM25).

### 4.3 First-Moment Expected-Rank Identity & Stratum Asymmetry

Expanding $M=10 \to 50$ raises recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). Expected rank $\mathbb{E}[\text{Rank}] \le k$ defines inclusion threshold $\text{AUC}^* \ge 1 - \frac{k - 1}{M - 1}$ ($\text{AUC}^* \ge 0.830$ at mean $M = 24.5, k=5$), with Proposition 1 bounding worst-case performance as a Markov floor ($\operatorname{Acc@}k \ge 1/k$, equaling chance $0.20$ at $M=25, k=5$). At graph $\text{AUC} = 0.6596$, expected rank is $9.00$ ($17.68$ at $M=50$). Strata polarize: Stratum A converts 6/11 reached targets (54.5%); Stratum C converts 0/14 ($-2$ tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PROTOCOL-SCALING-v1)

Protocol `PROTOCOL-SCALING-v1` evaluated pointwise discrimination on Dual T4 GPUs across 58 reachable Stratum C tasks:

| Model | Archetype | Params | Precision | ROC-AUC | VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B | FP16 | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B | FP16 | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini` | Synthetic | 3.8B | 4-bit NF4 | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-7B` | RL Distill | 7.0B | 4-bit NF4 | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B/26B | FP16 | *Bound* | *52.0 GB* | *Exceeds T4* |

- **Scaling & Positive Control**: (1) Tail discrimination scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), exceeding Phi-4-mini (0.7836) and DeepSeek 7B (0.6489; 95% CI $[+0.21, +0.34]$); (2) Gemma models ran in FP16, while Phi-4 and DeepSeek ran in 4-bit NF4; Gemma 4 26B-A4B requires 52.0 GB VRAM (fused 3D expert tensors bypass 2D bitsandbytes hooks), exceeding Dual T4 capacity (29.1 GB); (3) **Positive Control ($N=131$)**: Multi-pool union scoring ($\bigcup_i A_i, M_{\text{union}} \le 87$, 58 targets) with `gemma-4-e4b-it` reaches Top-1 **16.79%** (22/131), Acc@5 **37.4%** (49/131), Acc@10 **41.22%** (54/131), and MRR **0.2562** (vs BM25: 3.05% Top-1, 15.27% Acc@5, MRR 0.0862). Listwise direct readout on $A_4^{(25)}$ ($M \le 25$, 35 targets) yields 4.58% Top-1 and 16.79% Acc@5, which reasoning lifts to 9.92%, indicating 2B nulls reflect a capacity floor.

### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix & $M=15$ Verification (N=131)

| Condition | M | Acc@1 | Acc@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 | 25 | 3.05% | 15.27% | 0.0862 |
| Seeds Only | 10 | 3.05%→6.87% | 10.69%→15.27% | 0.0778→0.0865 |
| Depth BM25 | 25 | 4.58%→6.87% | 12.98%→16.79% | 0.0937→0.0967 |
| Graph ($C_0$) | 25 | **4.58%→9.92%** | **18.32%** | **0.0934→0.1358** |
| $C_1$ Stub | 25 | 2.29% | 15.27% | 0.0778 |
| $C_2$ Static Prov | 25 | 3.82% | 16.03% | 0.0899 |
| $C_3$ Prov + Think | 25 | 9.16% | 17.56% | 0.1270 |
| Pruned Seeds | 15 | 3.05%→7.63% | 13.74%→17.56% | 0.0821→0.1042 |
| Pruned Depth | 15 | 4.58%→8.40% | 14.50%→18.32% | 0.0954→0.1189 |
| Pruned Graph | 15 | 4.58% | 17.56% | 0.0970 |
| **Pruned Graph + Think** | **15** | **9.92%** | **19.85%** | **0.1469** |

- **Parity & Reasoning Lift**: Direct readout ($M \le 25$) ties graph ($A_4$) and depth BM25 ($A_2$) at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning improves Top-1 on $A_4^{(25)}$ to **9.92% (13/131; exploratory diagnostic $+5.34$ pp, raw $p=0.0156^*$, 4-test Holm $p=0.0625$, single reversal sensitivity $p=0.0703$)** and MRR ($0.0934 \to 0.1358$); backfilling preserves 18.32% Acc@5 ($p=0.6875$). $A_4\text{+think}$ vs $A_2\text{+think}$ ties at 13 vs 9 hits ($p=0.2188$) at $M=25$ and 13 vs 11 ($p=0.5000$) at $M=15$. Reasoning disambiguates candidates across strata: seeds convert $6/28 \to 12/28$ ($p=0.0312^*$) while graph additions convert $0/7 \to 1/7$ ($p=1.000$).
- **Provenance ($C_1\text{--}C_3$)**: Stubs ($C_1$, 2.29%) and provenance ($C_2$, 3.82%) yield no lift ($-0.76$ pp, $p=1.000$); provenance + thinking ($C_3$, 9.16%) ties thinking ($C_0$, 9.92%, $p=1.000$).
- **Pruning ($M=15, \text{AUC}^*=0.714$)**: Pruning to $M=15$ lowers $\text{AUC}^*$ to $\mathbf{0.714}$ while retaining 34/35 defects. Pruning yields zero net Top-1 change ($13/131$ in both pools; $13/34$ vs $13/35$ reachable reflects denominator compaction), achieving a **40% candidate stub reduction**; paired with test-time reasoning, it compacts mean target rank from 5.32 to 3.74 (Acc@5 reaches **19.85% (26/131)**, MRR **0.1469**).

### 4.6 Lexical Dissection & Multi-Encoding Graph Representations (PROTOCOL-CAUSAL-v1)

Protocol `PROTOCOL-CAUSAL-v1` evaluated 58 reachable Stratum C tasks on Dual T4s (one reversal [$n_{01}=1$] shifts McNemar $p$ from $0.0156$ to $0.0703$):

| Condition / Encoding | Acc@1 (Direct) | Acc@1 (Think) | Acc@5 (Direct) | Acc@5 (Think) | Shift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$O_0 / E_0$: Natural** | 12.07% | 6.90% | 27.59% | 17.24% | $-5.17$ pp |
| **$O_1$: $\alpha$-Renamed** | 10.34% | 6.90% | 24.14% | 17.24% | $-3.45$ pp |
| **$E_1$: Comment** | **25.86%** | **12.07%** | **39.66%** | **25.86%** | **$-13.79$ pp** |
| **$E_2$: Trace** | 17.24% | 5.17% | 32.76% | 18.97% | $-12.07$ pp |
| **$E_3$: Edge List** | 15.52% | 6.90% | 32.76% | 20.69% | $-8.62$ pp |
| **$E_4$: Adj Map** | 15.52% | 6.90% | 34.48% | 25.86% | $-8.62$ pp |

1. **AST $\alpha$-Renaming**: Obfuscating identifiers ($O_1$) shifts Top-1 by $-1.73$ pp (6 vs 7 hits, $p=1.0000$, 95% CI [$-$11.2, +7.7] pp), showing syntax sustains discrimination.
2. **Serialization**: Inline comments ($E_1$) reach 25.86% Top-1 ($+13.79$ pp, paired exact signed-rank/McNemar $p=0.0386^*$; 4-arm Holm $p_{\text{corr}} = 0.154$), exceeding edge lists ($E_3$, $+10.34$ pp, $p=0.0703$).
3. **Capacity Threshold**: Thinking degrades `gemma-4-e2b-it` across all arms ($E_0: 7 \to 4, E_1: 15 \to 7, E_2: 10 \to 3$), dropping Top-1. While 4B models infer relations ($6 \to 13$), 2B models suffer drift, requiring explicit provenance without chain-of-thought.

### 4.7 Post-Training Dynamics: Top-1 Bias Shifts vs. Global Rank Variance

I tuned `gemma-4-e2b-it` (LoRA $r=16$) on 226 Lite tasks via GroupKFold (171 train, 55 val; loss $0.9238 \to 0.0631$, val Acc@1 56.4%, Acc@5 81.8%). On Verified, 58.1% of tasks belong to Django/Astropy, showing within-repo adaptation.

| Cohort | Scorer | Acc@1 | Acc@5 | MRR | Med | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Graph ($N=25$) | Base E2B | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| | Graph-First | 8.0% | **52.0%** | 0.2518 | **5.0** | **0.7056** |
| | **Clean LoRA** | **24.0%** | 36.0% | **0.3288** | 9.0 | 0.5859 |
| | *Shortcut Probe* | *20.0%* | *56.0%* | *0.3702* | *4.0* | *0.6270* |
| Strat C ($N=35$) | Base E2B | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| | Graph-First | 0.0% | 17.1% | 0.0993 | 16.0 | 0.3753 |
| | **Clean LoRA** | **22.9%** | **48.6%** | **0.3623** | **6.0** | **0.6392** |
| | *Shortcut Probe* | *8.6%* | *42.9%* | *0.2596* | *6.0* | *0.6084* |

- **Empirical Dynamics**: On isolated graph targets ($N=25$), clean LoRA lifts Top-1 to **24.0%** (6 hits, $p = 0.0312^*$) and MRR ($0.1920 \to 0.3288$), but regresses Acc@5 ($44.0\% \to 36.0\%$) and AUC ($0.6063 \to 0.5859$) below Graph-First (52.0%, 0.7056). On mixed pools ($N=35$), clean LoRA reaches **22.9%** Top-1 ($6 \to 8$ hits, $n_{10}=6, n_{01}=4, p = 0.7539$) and **48.6%** Top-5 ($13 \to 17$ hits, $n_{10}=10, n_{01}=6, p = 0.4545$). A comment probe confirms Cohort A rewards non-seed bias while collapsing mixed Top-1 ($17.1\% \to 8.6\%$).

---

## 5. Discussion

- **Empirical Verdict & Task GLMM**: Marginal tests show small offsets (+0.87 to +2.03 pp; MDE 3.35 pp at 80% power, 95% CI $[-0.29, +4.36]$ pp). Task-clustered GLMM ($N=344$) indicates non-significance ($\text{OR} = 1.118, p = 0.0876$, paired McNemar $p=0.1435$; repository mixed model $\text{OR}=1.130, p=0.349, \sigma_{\text{repo}}=0.963$).
- **Selection Conditioning**: Discrimination drop (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, two-sample $p=0.721$), reflecting selection on $T^* \notin \text{Top10}(BM25)$.
- **Operating Regimes**: (1) **CSR Filter**: $<5.2$ ms ($\le 3$ MB RAM, 58.4% recall); (2) **Pruning**: Cap at $M=15$ ($\text{AUC}^*=0.714$); (3) **Positive Control**: E4B reaches 16.79% Top-1 and 37.4% Top-5 (MRR 0.2562); (4) **Routing**: Saves 31.4% compute; (5) **Tuning**: Overfitting cautions against tuning.
- **Tooling Safety**: AST editor achieves 0.0% syntax errors and 76.7% valid apply via pre-commit parse gate vs 30.0% apply and 50.0% errors for `git apply` ($p < .001$).

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0 (`https://github.com/yvliet/perron`):
- **Data & Documentation**: 12 CSR multigraphs (109,240 nodes, 341,892 edges), 707 strata labels, `scores.parquet` (19,298 evaluations), formal *Datasheet for Datasets* (Gebru et al., 2021), SHA-256 manifests.
- **Engine**: CSR diffusion in $1.89\text{--}5.10$ ms ($<3$ MB RAM); atomic AST surgery with pre-commit syntax parse gates.
- **Tooling**: `perron.diffuse_with_provenance()`, `perron-mcp` server, fine-tuned weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

Naive code graph expansion fails under small models not because graphs lack targets, but because low-overlap neighbors fall below expected-rank discrimination via selection conditioning, while unpruned graphs dilute precision. Reasoning disambiguates candidates on 4B models ($6 \to 12, p=0.0312^*$) but degrades 2B models (-53.3%), while bounded pruning ($M=15$) achieves rank compaction ($5.32 \to 3.74$) with zero Top-1 loss (13/131 in both pools; 13/34 reachable). A 4B positive control achieves 16.79% Top-1 and 37.4% Top-5 in union scoring, indicating 2B nulls reflect a capacity floor. Clean LoRA shows Top-1 reweighting on sparse targets (0.0% to 24.0%, $p=0.0312^*$) but adds rank variance; comment shortcuts induce mixed collapse ($17.1\% \to 8.6\%$). Perron provides a fast CSR foundation for local developer agents.

---

## References

[1] Gauthier, *Aider* (2023). [2] Ouyang+, *RepoGraph* (ICLR 2025). [3] Liu+, *CodexGraph* (NAACL 2025). [4] Chen+, *LocAgent* (ACL 2025). [5] Xia+, *Agentless* (FSE 2025). [6] Jimenez+, *SWE-bench* (ICLR 2024). [7] Klicpera+, *APPNP* (ICLR 2019). [8] Fatemi+, *Talk like a graph* (ICLR 2024). [9] Sanford+, *Transformer reasoning* (NeurIPS 2024). [10] Dong+, *Graph Reranking* (2024). [11] Zhang+, *SweRank* (ICLR 2026). [12] Li+, *PRFL* (ISSTA 2017).
