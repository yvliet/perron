# When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet), recrified@gmail.com) | **Code**: [perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Local code graph agents (2B-4B) face hub absorption and distractor dilution. Across 344 Verified tasks, Perron's CSR engine (1.89-5.10 ms, HSI@10=99.5%) suppresses hubs, yet unpruned 1-hop expansion under frozen 2B scoring matches depth-matched lexical retrieval (Acc@5 47.1% vs 45.1%, step-down Holm p=0.2869, exact McNemar p=0.1435; ±2.0 pp equivalence inconclusive, p=0.512, TOST power 1.92%). Wilcoxon-Mann-Whitney expectations (E[Rank] = 1 + (M-1)(1-AUC), Bamber 1975) explain this plateau: candidate selection conditioning drops discrimination from 0.7966 on seeds to 0.6596 on graph neighbors ($p=0.0036^*$), pushing expected target rank past top-5. Scale-matched regimes unlock graph context: on 2B models, zero-cost inline provenance comments lift reachable Top-1 from 12.07% to 25.86%; on 4B models across 131 un-named defects, horizon pruning (M=15, $\text{AUC}^*=0.714$, -40% stubs) with short reasoning (<=180 tokens) lifts Top-1 from 4.58% to 9.92% (raw $p=0.0156^*$, Holm p=0.0625) disambiguating seeds (12/28 hits) over graph additions (1/7 hits), compacting mean rank from 5.88 to 3.74 with zero Top-1 loss. Across 162 non-Django tasks, expansion matches lexical search (34.6% vs 32.7%, p=0.5811). Clean LoRA lifts isolated listwise Top-1 to 24.0% ($p=0.0312^*$) but degrades broad ranking (44% to 36%). I release depth-matched pools, 19,298 evaluations, 707 strata labels, and Perron CSR under Apache 2.0.

---

## 1. Introduction

Offline agents (2B-4B, $K \le 4,096$) need structural context without cloud APIs. Across 344 Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall, yet unpruned top-5 accuracy matches depth BM25 (47.1% vs 45.1%, Holm $p=0.287$).

Five contributions define the operating regime unlocking local code graphs:
1. **Matched Controls & Inconclusive Equivalence**: Expansion shows no detectable difference with depth BM25 ($\Delta=+2.03$ pp, exact $p=0.1435$; $\pm 2.0$ pp equivalence inconclusive, $p=0.512$, power 1.92%); task GLM (OR=1.118, p=0.0876) and repo GLMM (OR=1.130, p=0.349) confirm non-significance.
2. **Expected-Rank Horizon Threshold & Selection Conditioning**: Maintaining expected rank within top-$k$ ($\mathbb{E}[\text{Rank}] \le k$) requires within-pool discrimination $\text{AUC}^{*} \ge 1 - (k-1)/(M-1)$ (bounded by Proposition 1's Markov floor, $\text{Acc@}k \ge 1/k$). Discrimination drops to 0.6596 via non-seed selection conditioning ($T^{*} \notin \text{Top10}(\text{BM25})$, matching depth BM25 at $p=0.721$), necessitating horizon pruning ($M=15, \text{AUC}^{*}=0.714$).
3. **Scaling & MoE Bound**: Tail discrimination scales monotonically (0.8544 to 0.9274); 4B union reaches 16.79% Top-1, 37.4% Acc@5 (MRR 0.2562); 26B-A4B (52.0 GB) exceeds Dual T4.
4. **Disambiguation & Pruning**: Reasoning lifts 4B Top-1 at $M=25$ via seed disambiguation (6 to 12, $p=0.0312^{*}$); $M=15$ pruning cuts stubs 40%, compacting mean rank from 5.88 to 5.32 to 3.74 with zero Top-1 loss (13/131).
5. **Post-Training Dynamics**: Clean LoRA lifts isolated Top-1 to 24.0% ($p=0.0312^{*}$), but regresses Acc@5 (44% to 36%) below Graph-First (52.0%), showing localized adaptation.

---

## 2. Related Work

- **Diffusion & Scaling**: APPNP [7] and HippoRAG decouple PageRank from supervision; PRFL [12] and SweRank [11] evaluate fault localization. Unweighted walks ($\beta=0.85$) funnel $\beta^3=61.4\%$ of $t \ge 3$ mass into high-in-degree sinks (9.2% contamination, HSI@10=90.8%), while undirected hops shortcut through roots. Perron resolves both via directed CSR diffusion with specificity damping ($\gamma=0.70$, HSI@10=99.5%).
- **Graph Reasoning & Agents**: Fatemi et al. [8] and Sanford et al. [9] proved reachability bounds. Aider [1], CodexGraph [3], Agentless [5], and LocAgent [4] benchmark localization. Perron evaluates 2B/4B models zero-shot on static CSR graphs, showing reasoning eliminates edge verbalization in 4B models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 Verified tasks [6] across 12 repos: substring strata (N_A=208 [60.5%], N_B=5 [1.5%], N_C=131 [38.1%]); regexes (N_A=168, N_B=7, N_C=169; N=100 expert boundary audit: 95.0% concordance, Cohen κ=0.918, 95% CI [0.849, 0.988]). Auditing 56 exclusions confirms 100% syntax validity (68 files, 3,827 symbols); overturning equivalence requires >= +9 net wins (p=0.0016).
- **Perron CSR Engine**: Solves in 1.89-5.10 ms (<3 MB RAM vs SuperLU 28.1 ms). Transition matrix M (β=0.85) satisfies |λ_2(M)| <= β (Doeblin contraction $\|\boldsymbol{\pi}^{(t)} - \boldsymbol{\pi}_\infty\|_1 \le 2\beta^t$); damping (γ=0.70) yields HSI@10=99.5%.
- **Candidate Pools**: Top-10 BM25 seeds (M_t in [10, 50], mean 24.5) define depth-matched pools B0--A6. Frozen `gemma-4-e2b-it` scores log-odds (19,298 passes). Contrasts PC1 (A4 - A2) and PC2 (A4 - A3) evaluate McNemar tests with Holm correction and bootstrap CIs (B=10,000).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm p=1.000); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4%. Reachability is 57.8% at 1 hop, 66.9% at 2 hops. A 186-config sweep fits a GLMM in n_eff = exp(H(p_0)) (R^2=0.197, p < 0.001). Leave-Django-Out on N=162 non-Django tasks (47.1% benchmark) shows no detectable difference: A4 reaches 34.6% vs 32.7% for A2 (Δ=+1.85 pp, 56 vs 53 hits, n10=8, n01=5, exact McNemar p=0.5811, recall 50.0% vs 48.8%).

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

- **Statistical Testing & GLMM**: PC1 (A4 - A2): +2.03 pp (95% CI [-0.29, +4.36], 12/5 discordant, exact $p=0.1435$, Holm $p=0.287$). Task GLM (OR=1.118, p=0.0876) and repo GLMM (OR=1.130, p=0.349, $\sigma_{\text{repo}}=0.963$) confirm non-significance. At 4.94% paired discordance, $\pm 2.0$ pp TOST requires $N \approx 1{,}058$ (>500 Verified ceiling; $p=0.512$, power 1.92%); at $N=344$, 3.35 pp MDE and 90% Tango [+0.07, +4.24] pp bound gain below +4.24 pp. Diffusion dilutes precision (A6 45.6% vs A4 47.1%); discrimination drops from 0.7966 to 0.6596 via selection conditioning.
- **Loss Funnel**: Seeds capture 51.2% (48.8 pp loss); Graph reaches 58.4% (41.6 pp loss); Reranking retains 47.1% Top-5 (11.3 pp loss; N=100: Gemma 4 recovers 74.0% oracle vs 33.0% BM25).

### 4.3 First-Moment Expected-Rank Identity & Stratum Asymmetry

Expanding $M=10$ to $50$ raises recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). Expected rank $\mathbb{E}[\text{Rank}] \le k$ defines threshold $\text{AUC}^{*} \ge 1 - (k-1)/(M-1)$ ($\text{AUC}^{*} \ge 0.830$ at mean $M=24.5, k=5$); Proposition 1 bounds worst-case rank as a Markov floor (Acc@k $\ge 1/k$, equaling 0.20 at $M=25, k=5$). At graph AUC=0.6596, expected rank at mean pool $\bar{M}=24.5$ ($\bar{M}-1=23.5$) is $1 + 23.5(1-0.6596) = 9.00$ (17.68 at $M=50$); Stratum A converts 6/11 targets (54.5%), whereas Stratum C converts 0/14 (-2 tasks).

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary

Pointwise candidate discrimination on Dual T4s across 58 Stratum C tasks:

| Model | Archetype | Params | Precision | ROC-AUC | VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B | FP16 | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B | FP16 | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini` | Synthetic | 3.8B | 4-bit NF4 | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-7B` | RL Distill | 7.0B | 4-bit NF4 | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B/26B | FP16 | *Bound* | *52.0 GB* | *Exceeds T4* |

- **Scaling & Positive Control**: Tail discrimination scales monotonically in Gemma 4 FP16 (0.8544 to 0.9274), exceeding quantized Phi-4-mini (0.7836) and DeepSeek-7B (0.6489). Gemma 4 26B-A4B requires 52.0 GB FP16 VRAM (3D expert weights bypass 2D hooks), exceeding Dual T4 (29.1 GB). Positive Control ($N=131$): Union scoring ($M \le 87$) with `gemma-4-e4b-it` reaches Top-1 **16.79%** (22/131), Acc@5 **37.4%** (49/131), Acc@10 **41.22%** (54/131), MRR **0.2562** (vs BM25: 3.05% Top-1, 15.27% Acc@5, MRR 0.0862). Direct A4(25) yields 4.58% Top-1 (9.92% with reasoning), confirming 2B nulls reflect a capacity floor.

### 4.5 Candidate Pool × Reasoning Factorial Matrix & M=15 Verification (N=131)

*Denominator Key*: $N=344$ Verified; $N=131$ Stratum C; $N=58$ union ($M \le 87$); $N=35$ at $M=25$ ($28$ seeds + $7$ graph); $N=34$ at $M=15$ ($28$ seeds + $6$ graph).

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

- **Direct Readout & Reasoning Lift**: Direct readout ($M \le 25$) ties graph ($A_4$) and depth BM25 ($A_2$) at 16.79% Acc@5 (22/131) and 4.58% Top-1 (6/131). Reasoning lifts $A_4(25)$ Top-1 to **9.92% (13/131; diagnostic $+5.34$ pp, raw $p=0.0156^{*}$, 4-test Holm $p=0.0625$)** and MRR (0.0934 to 0.1358); backfilling preserves 18.32% Acc@5 ($p=0.6875$); 1 adverse pair ($p=0.0703$) or flip ($p=0.1250$) crosses $\alpha=0.05$. $A_4+\text{think}$ vs $A_2+\text{think}$ shows no detectable difference at 13 vs 9 (p=0.2188, M=25) and 13 vs 11 (p=0.5000, M=15); pointwise $A_4$ vs $A_2$ reaches 12.21% vs 8.40% ($n_{10}=6, n_{01}=1, p=0.1250$). Reasoning disambiguates seeds (6/28 to 12/28, $p=0.0312^{*}$) over graph additions (0/7 to 1/7, $p=1.000$).
- **Provenance (C1-C3)**: Stubs (C1, 2.29%) and provenance (C2, 3.82%) yield no lift (-0.76 pp, p=1.000); provenance + thinking (C3, 9.16%) ties thinking (C0, 9.92%).
- **Pruning ($M=15$, $\text{AUC}^*=0.714$)**: Pruning to $M=15$ lowers $\text{AUC}^{*}$ to 0.714 while retaining 34/35 defects, preserving Top-1 (6/131, 4.58%) while cutting stubs 40% (mean rank 5.88 to 5.32). Pairing pruning with reasoning compacts mean rank to **3.74** (Acc@5 **19.85% [26/131]**, MRR **0.1469**), retaining all 13 Top-1 hits: pruning compacts context; reasoning compacts rank.

### 4.6 Lexical Dissection & Multi-Encoding Graph Representations

To isolate representation from upstream retrieval misses on Stratum C (where 73 tasks lack the target), Table 6 evaluates `gemma-4-e2b-it` across $N=58$ union-reachable tasks (target presence shifts baseline from 6/131 [4.58%] in Table 5 to 7/58 [12.07%]; one discordant reversal [$n_{01}=1$] shifts McNemar $p$ from 0.0156 to 0.0703):

| Condition / Encoding | Acc@1 (Direct) | Acc@1 (Think) | Acc@5 (Direct) | Acc@5 (Think) | Shift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **O0 / E0: Natural** | 12.07% | 6.90% | 27.59% | 17.24% | -5.17 pp |
| **O1: α-Renamed** | 10.34% | 6.90% | 24.14% | 17.24% | -3.45 pp |
| **E1: Comment** | **25.86%** | **12.07%** | **39.66%** | **25.86%** | **-13.79 pp** |
| **E2: Trace** | 17.24% | 5.17% | 32.76% | 18.97% | -12.07 pp |
| **E3: Edge List** | 15.52% | 6.90% | 32.76% | 20.69% | -8.62 pp |
| **E4: Adj Map** | 15.52% | 6.90% | 34.48% | 25.86% | -8.62 pp |

1. **AST α-Renaming**: Obfuscating identifiers (O1) shifts Top-1 by -1.73 pp (6 vs 7 hits, p=1.0000, 95% CI [-11.2, +7.7] pp); syntax sustains discrimination.
2. **Serialization**: Inline comments (E1) reach 25.86% Top-1 (+13.79 pp, paired McNemar $p=0.0386^{*}$, 4-arm Holm p=0.154), beating edge lists (E3, +10.34 pp, p=0.0703).
3. **Capacity Threshold**: Thinking degrades `gemma-4-e2b-it` across serialization arms (E0: 7 to 4, E1: 15 to 7, E2: 10 to 3). While 4B models infer relations (6 to 13), 2B models suffer drift, requiring provenance without chain-of-thought.

### 4.7 Post-Training Dynamics: Top-1 Bias Shifts vs. Global Rank Variance

Tuning `gemma-4-e2b-it` (LoRA r=16) on 226 Lite tasks via GroupKFold (171 train, 55 val; loss 0.9238 to 0.0631, val Acc@1 56.4%, Acc@5 81.8%) yields 58.1% Django/Astropy on Verified.

| Cohort | Scorer | Acc@1 | Acc@5 | MRR | Med | AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Graph (N=25) | Base E2B (Listwise) | 0.0% | 44.0% | 0.1920 | 7.0 | 0.6063 |
| | Graph-First | 8.0% | **52.0%** | 0.2518 | **5.0** | **0.7056** |
| | **Clean LoRA** | **24.0%** | 36.0% | **0.3288** | 9.0 | 0.5859 |
| | *Shortcut Probe* | *20.0%* | *56.0%* | *0.3702* | *4.0* | *0.6270* |
| Strat C (N=35) | Base E2B (Listwise) | 17.1% | 37.1% | 0.2895 | 9.0 | 0.5692 |
| | Graph-First | 0.0% | 17.1% | 0.0993 | 16.0 | 0.3753 |
| | **Clean LoRA** | **22.9%** | **48.6%** | **0.3623** | **6.0** | **0.6392** |
| | *Shortcut Probe* | *8.6%* | *42.9%* | *0.2596* | *6.0* | *0.6084* |

- **Empirical Dynamics**: On isolated graph targets ($N=25$), clean LoRA lifts listwise Top-1 to **24.0%** (6 hits, $p=0.0312^{*}$; pointwise E2B ranks $\ge 2$ at Top-1) and MRR (0.1920 to 0.3288), but drops Acc@5 (44.0% to 36.0%) and AUC (0.6063 to 0.5859) below Graph-First (52.0%, 0.7056). On mixed pools ($N=35$), clean LoRA reaches **22.9%** Top-1 ($p=0.7539$) and **48.6%** Top-5 ($p=0.4545$); a comment probe collapses mixed Top-1 (17.1% to 8.6%).

---

## 5. Discussion

- **Why Naive Expansion Stalls Under Frozen 2B Scoring**: Offsets (+0.87 to +2.03 pp; 95% CI [-0.29, +4.36] pp) show no detectable lift over depth BM25 across task GLMs (OR=1.118, p=0.0876) and repo GLMMs (OR=1.130, p=0.349, $\sigma_{\text{repo}}=0.963$, SE doubles [0.065 to 0.130] as repo intercepts drop task pairing). Design Rule: Never append unpruned neighborhoods to 2B prompts without truncation and relational bridging.
- **Selection Conditioning**: Discrimination drop (0.7966 vs 0.6596) mirrors depth BM25 (0.6380 vs 0.6253, $p=0.721$), reflecting conditioning on $T^{*} \notin \text{Top10}(\text{BM25})$. RepoGraph [2] unpruned 2-hop collapsed to 26.00% ($-5.67$ pp) via distractor dilution. Depth-matched controls ($N=344$, 19,298 passes) isolate topology from cardinality: as 1-hop verification is Depth-1 [9], reasoning disambiguates lexical distractors over traversing graphs.
- **Practical Operating Regimes by Gemma 4 Scale**: (1) **CPU Pre-Filtering (<5.2 ms, <=3 MB RAM)**: Specificity-damped CSR diffusion ($\gamma=0.70$, HSI@10=99.5%), selective routing ($\pi^*$, saving 31.4% compute; 58.4% recall); (2) **Edge Tier (`gemma-4-e2b-it`, 6698.9 MB VRAM)**: Inline provenance comments (E1, lifting Top-1 from 12.07% to 25.86%); avoid reasoning ($-42.9\%$ drift) and LoRA (drops Acc@5 to 36%); (3) **Workstation Tier (`gemma-4-e4b-it`, 9648.4 MB VRAM)**: Prune to $M=15$ ($\text{AUC}^{*}=0.714$, $-40\%$ stubs) with reasoning ($\le 180$ tokens) compacting mean rank from $5.88 \to 3.74$ (Top-1 $4.58\% \to 9.92\%$, union Acc@5 $37.4\%$); (4) **MoE Tier (`gemma-4-26b-a4b-it`, 52.0 GB FP16)**: Requires ahead-of-time AWQ; fused 3D expert weights ([128, d_in, d_out]) bypass 2D hooks.

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0 (`https://github.com/yvliet/perron`):
- **Data & Specs**: 12 CSR multigraphs (109,240 nodes, 341,892 edges), 707 strata labels, *Datasheet for Datasets*, `scores.parquet` (19,298 E2B and 8,113 E4B runs), SHA-256 manifests.
- **Engine & Tooling**: CSR diffusion in 1.89-5.10 ms (<3 MB RAM) with syntax gates, `diffuse_with_provenance()`, `perron-mcp` server, and LoRA (`yvliet/perron-gemma4-e2b-lora`).
- **Notebooks**: [Gemma 4 Workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation) and [Quickstart Demo](https://www.kaggle.com/code/yvliet/perron-spectral-code-graph-quickstart-demo).

---

## 7. Conclusion

Perron defines the operating regime for local code graphs: post-walk damping (1.89-5.10 ms, HSI@10=99.5%) prevents hub collapse, while unpruned expansion stalls on 2B models as low-overlap neighbors drop below expected-rank discrimination ($\text{AUC}=0.6596 < 0.830$) via selection conditioning. Unlocking graph context requires horizon pruning ($M=15$, $-40\%$ stubs) paired with 4B reasoning ($6 \to 12$ seed Top-1, $p=0.0312^{*}$, compacting mean rank from 5.88 to 3.74 with zero Top-1 loss) or 2B inline provenance (12.07% to 25.86% Top-1). A 4B control reaches 16.79% Top-1 and 37.4% Acc@5, confirming 2B nulls reflect a capacity floor. Clean LoRA lifts isolated Top-1 (24.0%, $p=0.0312^{*}$) but adds rank variance. Perron provides a fast CSR foundation for local developer agents.

---

## References

[1] Gauthier, *Aider* (2023). [2] Ouyang+, *RepoGraph* (ICLR 2025). [3] Liu+, *CodexGraph* (NAACL 2025). [4] Chen+, *LocAgent* (ACL 2025). [5] Xia+, *Agentless* (FSE 2025). [6] Jimenez+, *SWE-bench* (ICLR 2024). [7] Klicpera+, *APPNP* (ICLR 2019). [8] Fatemi+, *Talk like a graph* (ICLR 2024). [9] Sanford+, *Transformer reasoning* (NeurIPS 2024). [10] Dong+, *Graph Reranking* (2024). [11] Zhang+, *SweRank* (ICLR 2026). [12] Li+, *PRFL* (ISSTA 2017).
