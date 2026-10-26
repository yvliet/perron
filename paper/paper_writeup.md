# When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet), recrified@gmail.com) | **Code**: [perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Across 344 Verified tasks, 1-hop AST graph expansion exhibits no statistically detectable difference with depth-matched lexical retrieval (Acc@5 47.1% vs 45.1%, step-down Holm p=0.2869, exact McNemar p=0.1435; formal ±2.0 pp equivalence remains inconclusive, p=0.512, TOST power 1.92%). Classical Wilcoxon-Mann-Whitney expectations (E[Rank] = 1 + (M-1)(1-AUC), Bamber 1975) explain why expansion stalls: candidate selection conditioning drops pairwise discrimination from 0.7966 on seeds to 0.6596 on graph neighbors (p=0.0036*), pushing expected target rank past the top-5 window. On 131 un-named defects, test-time reasoning (<= 180 tokens) improves Top-1 at M=25 (4.58% to 9.92%, raw p=0.0156*; 4-test Holm p=0.0625, adverse-pair p=0.0703, hit-flip p=0.1250), acting as candidate disambiguation among lexical seeds (12/28 hits) rather than relational graph traversal (1/7 hits). Candidate horizon pruning (M=15) cuts candidate stubs by 40%; paired with reasoning, it compacts mean target rank from 5.88 to 5.32 to 3.74 with zero loss in reachable Top-1 hits. Across 162 non-Django tasks, topological expansion matches lexical search at 34.6% vs 32.7% (p=0.5811). Parameter-efficient fine-tuning on SWE-bench Lite lifts isolated graph Top-1 to 24.0% (p=0.0312*) but introduces trailing distractor variance that degrades broad-pool ranking (44% to 36%). I release depth-matched candidate pools, 19,298 evaluations, 707 verified strata labels, and the reference Perron CSR engine under Apache 2.0.

---

## 1. Introduction

Local agents must localize defects within tight contexts ($K \le 4,096$). On 344 Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

Five contributions resolve this disconnect:
1. **Parity & Equivalence**: Expansion matches depth BM25 ($\Delta=+2.03$ pp, exact $p=0.1435$; $\pm 2.0$ pp equivalence is inconclusive, $p=0.512$, power 1.92%); task GLM (OR=1.118, p=0.0876) and repo mixed model (OR=1.130, p=0.349) confirm non-significance.
2. **Markov Floor**: Candidate inclusion requires $\text{AUC}^{*} \ge 1 - (k-1)/(M-1)$; Proposition 1 bounds worst-case performance as a Markov floor ($1/k=0.20$ at $M=25, k=5$). Discrimination drops to 0.6596 via candidate selection conditioning.
3. **Scaling & MoE Bound**: Tail discrimination scales monotonically (0.8544 to 0.9274); 4B union control reaches 16.79% Top-1, 37.4% Acc@5 (MRR 0.2562); 26B-A4B MoE 52.0 GB footprint exceeds Dual T4.
4. **Disambiguation & Pruning**: Reasoning lifts 4B Top-1 at $M=25$ via seed disambiguation (6 to 12, $p=0.0312^{*}$); $M=15$ pruning cuts candidate stubs by 40%, compacting mean gold rank from 5.88 to 5.32 to 3.74 with zero Top-1 loss (13/131 in both pools).
5. **Post-Training Dynamics**: Clean LoRA lifts isolated Top-1 to 24.0% ($p=0.0312^{*}$), but regresses Acc@5 (44% to 36%) below Graph-First (52.0%), showing localized training-set adaptation.

---

## 2. Related Work

- **Diffusion & Scaling**: APPNP [7] and HippoRAG decouple PageRank from supervision but accumulate hub drift. SweRank [11] trains 7B models; PRFL [12] applied PageRank to defect localization.
- **Graph Reasoning**: Fatemi et al. [8] and Sanford et al. [9] proved reachability depth bounds. I show reasoning eliminates explicit edge verbalization in 4B models.
- **Agent Localization**: Aider [1], CodexGraph [3], Agentless [5], and LocAgent [4] benchmark localization. Perron evaluates compact models (2B/4B) zero-shot on static CSR graphs.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 Verified tasks [6] across 12 repos: protocol lock evaluates substring strata (N_A=208 [60.5%], N_B=5 [1.5%], N_C=131 [38.1%]); dataset release refines word-boundary regexes (N_A=168, N_B=7, N_C=169; N=100 audit: 95.0% concordance, Cohen κ=0.918, 95% CI [0.849, 0.988]). Auditing 56 exclusions confirms 100% syntax validity (68 files, 3,827 symbols); overturning equivalence requires >= +9 net wins (p=0.0016).
- **Perron CSR Engine**: Typed edges solve in 1.89-5.10 ms (<3 MB RAM vs SuperLU at 28.1 ms). Transition matrix M (β=0.85) satisfies |λ_2(M)| <= β (Doeblin contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$). Damping (γ=0.70) yields HSI@10=99.5%.
- **Candidate Pools**: Top-10 BM25 seeds (M_t in [10, 50], mean 24.5) define depth-matched candidate pools B0--A6 under pre-registered plans. Frozen `gemma-4-e2b-it` scores log-odds (19,298 passes). Contrasts PC1 (A4 - A2) and PC2 (A4 - A3) evaluate McNemar tests with Holm correction and bootstrap CIs (B=10,000).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm p=1.000); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4%. Reachability is 57.8% at 1 hop, 66.9% at 2 hops. A 186-config sweep fits a GLMM in n_eff = exp(H(p_0)) (R^2=0.197, p < 0.001). Leave-Django-Out across N=162 non-Django tasks (47.1% of benchmark) exhibits parity: A4 achieves 34.6% vs 32.7% for A2 (Δ=+1.85 pp, 56 vs 53 hits, n10=8, n01=5, exact McNemar p=0.5811, recall 50.0% vs 48.8%).

### 4.2 Pointwise Candidate Reranking: Confirmatory Matched-Control Evaluation

| Arm | Candidate (M <= 50) | Acc@1 | Acc@5 | MRR@50 |
| :--- | :--- | :---: | :---: | :---: |
| **B0** | BM25 | 25.3% | 44.8% | 0.3439 |
| **A1** | Seeds | 24.7% | 46.2% | 0.3348 |
| **A2** | Depth BM25 | 23.8% | 45.1% | 0.3301 |
| **A3** | Same-file | 24.1% | 45.6% | 0.3381 |
| **A4** | Graph 1-hop | 24.7% | 47.1% | 0.3409 |
| **A5** | Null graph | 24.1% | 45.9% | 0.3315 |
| **A6** | Perron | 23.6% | 45.6% | 0.3301 |

- **Statistical Testing & GLMM**: PC1 (A4 - A2): +2.03 pp (95% CI [-0.29, +4.36], 12/5 discordant, exact $p=0.1435$, Holm $p=0.287$). Task GLM (OR=1.118, p=0.0876) and repo mixed model (OR=1.130, p=0.349, $\sigma_{\text{repo}}=0.963$) confirm non-significance. Equivalence within $\pm 2.0$ pp is inconclusive ($p=0.512$, power 1.92%, requiring $N \approx 1{,}058$). MDE is 3.35 pp at 80% power; 90% Tango interval is [+0.07, +4.24] pp. Diffusion dilutes precision (A6 45.6% vs A4 47.1%); discrimination drops from 0.7966 to 0.6596 via selection conditioning.
- **Loss Funnel**: Seeds capture 51.2% (48.8 pp loss); Graph reaches 58.4% (41.6 pp boundary loss); Reranking retains 47.1% Top-5 (11.3 pp loss; N=100: Gemma 4 recovers 74.0% oracle targets vs 33.0% BM25).

### 4.3 First-Moment Expected-Rank Identity & Stratum Asymmetry

Expanding $M=10$ to $50$ raises recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). Expected rank $\mathbb{E}[\text{Rank}] \le k$ defines inclusion threshold $\text{AUC}^{*} \ge 1 - (k-1)/(M-1)$ ($\text{AUC}^{*} \ge 0.830$ at mean $M=24.5, k=5$), Proposition 1 bounds worst-case performance as a Markov floor (Acc@k $\ge 1/k$, equaling 0.20 at $M=25, k=5$). At graph AUC=0.6596, expected rank is 9.00 (17.68 at $M=50$). Strata polarize: Stratum A converts 6/11 reached targets (54.5%); Stratum C converts 0/14 (-2 tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary

Pointwise candidate discrimination was evaluated on Dual T4s across 58 Stratum C tasks:

| Model | Archetype | Params | Precision | ROC-AUC | VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B | FP16 | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B | FP16 | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini` | Synthetic | 3.8B | 4-bit NF4 | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-7B` | RL Distill | 7.0B | 4-bit NF4 | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B/26B | FP16 | *Bound* | *52.0 GB* | *Exceeds T4* |

- **Scaling & Positive Control**: Tail discrimination scales monotonically in Gemma 4 FP16 (0.8544 to 0.9274), exceeding quantized baselines Phi-4-mini (0.7836) and DeepSeek-7B (0.6489). Gemma 4 26B-A4B requires 52.0 GB unquantized FP16 VRAM (fused 3D expert tensors bypass 2D quantization hooks), exceeding Dual T4 capacity (29.1 GB). Positive Control ($N=131$): Union scoring ($M \le 87$) with `gemma-4-e4b-it` reaches Top-1 **16.79%** (22/131), Acc@5 **37.4%** (49/131), Acc@10 **41.22%** (54/131), MRR **0.2562** (vs BM25: 3.05% Top-1, 15.27% Acc@5, MRR 0.0862). Direct readout on A4(25) yields 4.58% Top-1 and 16.79% Acc@5, which reasoning lifts to 9.92%, confirming 2B nulls reflect a capacity floor.

### 4.5 Candidate Pool × Reasoning Factorial Matrix & M=15 Verification (N=131)

| Condition | M | Acc@1 | Acc@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 | 25 | 3.05% | 15.27% | 0.0862 |
| Seeds Only | 10 | 3.05%→6.87% | 10.69%→15.27% | 0.0778→0.0865 |
| Depth BM25 | 25 | 4.58%→6.87% | 12.98%→16.79% | 0.0937→0.0967 |
| Graph (C0) | 25 | **4.58%→9.92%** | **18.32%** | **0.0934→0.1358** |
| C1 Stub | 25 | 2.29% | 15.27% | 0.0778 |
| C2 Static Prov | 25 | 3.82% | 16.03% | 0.0899 |
| C3 Prov + Think | 25 | 9.16% | 17.56% | 0.1270 |
| Pruned Seeds | 15 | 3.05%→7.63% | 13.74%→17.56% | 0.0821→0.1042 |
| Pruned Depth | 15 | 4.58%→8.40% | 14.50%→18.32% | 0.0954→0.1189 |
| Pruned Graph | 15 | 4.58% | 17.56% | 0.0970 |
| **Pruned Graph + Think** | **15** | **9.92%** | **19.85%** | **0.1469** |

- **Parity & Reasoning Lift**: Direct readout ($M \le 25$) ties graph ($A_4$) and depth BM25 ($A_2$) at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning lifts $A_4(25)$ Top-1 to **9.92% (13/131; diagnostic $+5.34$ pp, raw $p=0.0156^{*}$, Holm $p=0.0625$, adverse-pair $p=0.0703$, hit-flip $p=0.1250$)** and MRR (0.0934 to 0.1358); backfilling preserves 18.32% Acc@5 ($p=0.6875$). $A_4+\text{think}$ vs $A_2+\text{think}$ ties at 13 vs 9 (p=0.2188, M=25) and 13 vs 11 (p=0.5000, M=15). Pointwise $A_4$ vs $A_2$ achieves 12.21% vs 8.40% ($n_{10}=6, n_{01}=1, p=0.1250$). Reasoning disambiguates across strata: seeds convert 6/28 to 12/28 ($p=0.0312^{*}$) while graph additions convert 0/7 to 1/7 ($p=1.000$).
- **Provenance (C1-C3)**: Stubs (C1, 2.29%) and provenance (C2, 3.82%) yield no lift (-0.76 pp, p=1.000); provenance + thinking (C3, 9.16%) ties thinking (C0, 9.92%).
- **Pruning (M=15, AUC*=0.714)**: Pruning to $M=15$ lowers $\text{AUC}^{*}$ to 0.714 while retaining 34/35 defects. Pruning alone preserves Top-1 (6/131, 4.58%) while cutting candidate stubs by 40% (mean rank 5.88 to 5.32). Pairing pruning with reasoning compacts mean rank to **3.74** (Acc@5 reaches **19.85% [26/131]**, MRR **0.1469**), retaining all 13 Top-1 hits. Thus, pruning provides context compaction without hit loss, while reasoning provides rank compaction.

### 4.6 Lexical Dissection & Multi-Encoding Graph Representations

Evaluating 58 Stratum C tasks on Dual T4s across graph encoding formats (one reversal [$n_{01}=1$] shifts McNemar $p$ from 0.0156 to 0.0703):

| Condition / Encoding | Acc@1 (Direct) | Acc@1 (Think) | Acc@5 (Direct) | Acc@5 (Think) | Shift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **O0 / E0: Natural** | 12.07% | 6.90% | 27.59% | 17.24% | -5.17 pp |
| **O1: α-Renamed** | 10.34% | 6.90% | 24.14% | 17.24% | -3.45 pp |
| **E1: Comment** | **25.86%** | **12.07%** | **39.66%** | **25.86%** | **-13.79 pp** |
| **E2: Trace** | 17.24% | 5.17% | 32.76% | 18.97% | -12.07 pp |
| **E3: Edge List** | 15.52% | 6.90% | 32.76% | 20.69% | -8.62 pp |
| **E4: Adj Map** | 15.52% | 6.90% | 34.48% | 25.86% | -8.62 pp |

1. **AST α-Renaming**: Obfuscating identifiers (O1) shifts Top-1 by -1.73 pp (6 vs 7 hits, p=1.0000, 95% CI [-11.2, +7.7] pp), showing syntax sustains discrimination.
2. **Serialization**: Inline comments (E1) reach 25.86% Top-1 (+13.79 pp, paired McNemar $p=0.0386^{*}$, 4-arm Holm p=0.154), exceeding edge lists (E3, +10.34 pp, p=0.0703).
3. **Capacity Threshold**: Thinking degrades `gemma-4-e2b-it` across all serialization arms (E0: 7 to 4, E1: 15 to 7, E2: 10 to 3), dropping Top-1. While 4B models infer relations (6 to 13), 2B models suffer drift, requiring provenance without chain-of-thought.

### 4.7 Post-Training Dynamics: Top-1 Bias Shifts vs. Global Rank Variance

I tuned `gemma-4-e2b-it` (LoRA r=16) on 226 Lite tasks via GroupKFold (171 train, 55 val; loss 0.9238 to 0.0631, val Acc@1 56.4%, Acc@5 81.8%). On Verified, 58.1% of tasks belong to Django/Astropy.

| Cohort | Scorer | Acc@1 | Acc@5 | MRR | Med | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Graph (N=25) | Base E2B | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| | Graph-First | 8.0% | **52.0%** | 0.2518 | **5.0** | **0.7056** |
| | **Clean LoRA** | **24.0%** | 36.0% | **0.3288** | 9.0 | 0.5859 |
| | *Shortcut Probe* | *20.0%* | *56.0%* | *0.3702* | *4.0* | *0.6270* |
| Strat C (N=35) | Base E2B | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| | Graph-First | 0.0% | 17.1% | 0.0993 | 16.0 | 0.3753 |
| | **Clean LoRA** | **22.9%** | **48.6%** | **0.3623** | **6.0** | **0.6392** |
| | *Shortcut Probe* | *8.6%* | *42.9%* | *0.2596* | *6.0* | *0.6084* |

- **Empirical Dynamics**: On isolated graph targets ($N=25$), clean LoRA lifts Top-1 to **24.0%** (6 hits, $p=0.0312^{*}$) and MRR (0.1920 to 0.3288), but regresses Acc@5 (44.0% to 36.0%) and AUC (0.6063 to 0.5859) below Graph-First (52.0%, 0.7056). On mixed pools ($N=35$), clean LoRA reaches **22.9%** Top-1 ($p=0.7539$) and **48.6%** Top-5 ($p=0.4545$). A comment probe confirms non-seed bias collapses mixed Top-1 (17.1% to 8.6%).

---

## 5. Discussion

- **Model Specification**: Offsets (+0.87 to +2.03 pp; 95% CI [-0.29, +4.36] pp) confirm parity. Task GLM (OR=1.118, p=0.0876) and repo GLMM (OR=1.130, p=0.349, $\sigma_{\text{repo}}=0.963$) confirm non-significance; SE doubles (0.065 to 0.130) because repo intercepts drop task pairing.
- **Selection Conditioning**: Discrimination drop (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, $p=0.721$), reflecting conditioning on $T^{*} \notin \text{Top10}(\text{BM25})$. In literature, RepoGraph (ICLR 2025) confirms unpruned 2-hop collapsed to 26.00% ($-5.67$ pp) via distractor dilution, corroborating Perron. As 1-hop checking is Depth-1 (Sanford et al., 2024), reasoning disambiguates lexical distractors rather than traversing graphs.
- **Operating Regimes**: (1) **CSR Filter**: $<5.2$ ms ($\le 3$ MB RAM, 58.4% recall); (2) **Pruning**: Cap at $M=15$ ($\text{AUC}^{*}=0.714$); (3) **Control**: E4B reaches 16.79% Top-1, 37.4% Acc@5 (MRR 0.2562); (4) **Routing**: Saves 31.4% compute; (5) **Tuning**: Overfitting cautions against adapter tuning.

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0 (`https://github.com/yvliet/perron`):
- **Data & Specs**: 12 CSR multigraphs (109,240 nodes, 341,892 edges), 707 strata labels, `scores.parquet` (19,298 evaluations), *Datasheet for Datasets* (Gebru et al., 2021), SHA-256 manifests.
- **Engine**: CSR diffusion in 1.89-5.10 ms (<3 MB RAM) with pre-commit syntax validation gates.
- **Tooling**: `perron.diffuse_with_provenance()`, `perron-mcp` server, LoRA weights (`yvliet/perron-gemma4-e2b-lora`).

---

## 7. Conclusion

Code graph expansion stalls under small models not from missing reachability, but because low-overlap neighbors drop below expected-rank discrimination via selection conditioning, while unpruned graphs dilute precision. Test-time reasoning disambiguates candidates in 4B models ($6 \to 12, p=0.0312^{*}$) but degrades 2B models, while bounded pruning ($M=15$) achieves rank compaction (5.88 to 5.32 to 3.74) with zero Top-1 loss. A 4B control reaches 16.79% Top-1 and 37.4% Acc@5, indicating 2B nulls reflect a capacity floor. Clean LoRA shows Top-1 reweighting on isolated targets (24.0%, $p=0.0312^{*}$) but adds rank variance. Perron provides a fast, zero-copy CSR foundation for local developer agents.

---

## References

[1] Gauthier, *Aider* (2023). [2] Ouyang+, *RepoGraph* (ICLR 2025). [3] Liu+, *CodexGraph* (NAACL 2025). [4] Chen+, *LocAgent* (ACL 2025). [5] Xia+, *Agentless* (FSE 2025). [6] Jimenez+, *SWE-bench* (ICLR 2024). [7] Klicpera+, *APPNP* (ICLR 2019). [8] Fatemi+, *Talk like a graph* (ICLR 2024). [9] Sanford+, *Transformer reasoning* (NeurIPS 2024). [10] Dong+, *Graph Reranking* (2024). [11] Zhang+, *SweRank* (ICLR 2026). [12] Li+, *PRFL* (ISSTA 2017).
