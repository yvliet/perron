# When Does Graph Expansion Help Small Code Models Localize Bugs? An Empirical Audit of Reranking, Distractor Dilution, and Test-Time Reasoning on Gemma 4

**Author**: Sultan Haikal ([@yvliet](https://github.com/yvliet), recrified@gmail.com) | **Code**: [perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

---

## Abstract

Local code graph agents (2B-4B) face hub absorption and distractor dilution. Perron's CSR engine (1.89-5.10 ms, <=3 MB RAM, HSI@10=99.5%, |λ₂|<=0.85, ||π^(t)-π_∞||₁<=2β^t) achieves **Level-2 Virtual Namespace Invariance** under SchrödingerRepo perturbations [13]: while BM25 collapses to 0.29%, Perron preserves 34.88% Acc@5 (+34.59 pp for PPR, +34.30 pp for Hybrid, p=0.0000\*), beating Agentless AST skeletons [5] by +11.34 pp (47.09% vs 35.76%, p=0.0000\*). On UTBoost clean Stratum C tasks [14] (N=75), graph expansion lifts Acc@5 by +8.00 pp over depth BM25 (14.67% vs 6.67%, p=0.0312\*). On natural code, selection conditioning limits 2B unassisted expansion (47.1% vs 45.1%, p=0.2869; AUC 0.7966 to 0.6596, p=0.0036\*). Scale-matched regimes unlock compact agents: Pointwise E4B on pruned A4[:15] pools dominates unassisted scoring (22.90% Acc@5, 12.21% Top-1, MRR 0.1740, p=0.0129\*); candidate pruning (M=15, AUC\*=0.714) with short reasoning (<=180 tokens) compacts mean rank from 5.88 to 3.74 with zero Top-1 loss; and 2B inline comments lift reachable Top-1 from 12.07% to 25.86%. I release 12 multigraphs, 19,298 evaluations, 707 strata labels, and Perron CSR under Apache 2.0.

---

## 1. Introduction

Offline agents (2B-4B, K≤4,096) need structural context without cloud APIs. Across 344 Verified tasks with frozen `gemma-4-e2b-it`, 1-hop expansion adds +7.3 pp recall, yet top-5 accuracy matches depth BM25 (47.1% vs 45.1%, p=0.287).

Five contributions define the operating regime unlocking local code graphs:
1. **Level-2 Virtual Namespace Invariance**: Under SchrödingerRepo obfuscation [13], Perron preserves 34.88% Acc@5 (+34.59 pp for PPR, +34.30 pp for Hybrid over BM25 at 0.29%, p=0.0000\*), retaining 94.5% natural performance and exceeding Agentless AST skeletons [5] by +11.34 pp (47.09% vs 35.76%, p=0.0000\*).
2. **Clean Stratum C Lift via UTBoost**: UTBoost de-biasing [14] isolates N=75 clean un-named tasks where 1-hop expansion lifts Acc@5 by +8.00 pp over depth BM25 (14.67% vs 6.67%, p=0.0312\*); on natural code unassisted 2B expansion matches depth BM25 (47.1% vs 45.1%, p=0.2869).
3. **Pointwise E4B Dominance vs. Listwise Collapse**: Pointwise E4B on A4[:15] pools reaches 22.90% Acc@5, 12.21% Top-1, MRR 0.1740 (p=0.0129\*). Listwise ranking suffered 90.8% syntax failure, collapsing to 2.29% without Qiao backfill [15], with severe positional decay (75.0% in slots 1..5 vs 0.0% in slots 11..15) [16].
4. **Dense Bi-Encoder Boundaries & Selection Conditioning**: Dense bi-encoders (CodeRankEmbed 137M, 49.62% Acc@5) hit linear fusion penalties (-17.56 pp, Bellam et al. [17]). Selection conditioning drops neighbor discrimination to AUC=0.6596, requiring horizon pruning (M=15, AUC\*=0.714) with reasoning to compact rank from 5.88 to 3.74.
5. **Post-Training Dynamics & Generalization Split**: Clean LoRA lifts isolated Top-1 to 24.0% (p=0.0312\*) and reaches 58.33% Acc@5 on seen repos, but collapses to 15.38% Acc@5 and 0.4846 ROC-AUC on unseen repos (N=13, Biderman et al. [18]), demonstrating memorization over traversal.

---

## 2. Related Work

- **Diffusion & Scaling**: APPNP [7] and HippoRAG decouple PageRank from supervision; PRFL [12] and SweRank [11] evaluate fault localization. Unweighted walks (β=0.85) funnel 61.4% of mass into sinks (9.2% contamination, HSI@10=90.8%), while undirected hops shortcut through roots. Perron resolves both via directed CSR diffusion with specificity damping (γ=0.70, HSI@10=99.5%).
- **Graph Reasoning & Agents**: Fatemi et al. [8] and Sanford et al. [9] proved reachability bounds. Aider [1], CodexGraph [3], Agentless [5], and LocAgent [4] benchmark localization. Perron evaluates 2B/4B models zero-shot on static CSR graphs; reasoning eliminates edge verbalization in 4B models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 Verified tasks [6] across 12 repos: substring strata (Named N\_A=208, Traceback N\_B=5, Un-Named N\_C=131); regex strata (N\_A=168, N\_B=7, N\_C=169; N=100 audit: 95.0% concordance, Cohen κ=0.918). 56 exclusions confirm 100% syntax validity (68 files, 3,827 symbols); overturning equivalence requires >= +9 net wins (p=0.0016).
- **Perron CSR Engine**: Solves in 1.89-5.10 ms (<3 MB RAM). Stochastic M (β=0.85) satisfies |λ₂(M)|<=β (Doeblin contraction ||π^(t)-π_∞||₁<=2β^t); damping (γ=0.70) yields HSI@10=99.5%. In 10/12 repos, degrees violate log\_n C\_p < 1.0 (Gleich et al., reaching 1.2579 in Xarray), driving PageRank into hub absorption without damping. Inter-file transitions exhibit disassortativity (α\*=-0.2730, Kloumann et al.), explaining negative weights on T³ (-0.2428).
- **Candidate Pools**: Top-10 BM25 seeds (mean M=24.5) define depth-matched pools B0--A6 scored via frozen `gemma-4-e2b-it` (19,298 passes). Contrasts PC1 (A4 - A2) and PC2 (A4 - A3) evaluate McNemar tests with Holm correction (B=10,000).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability

Across 344 tasks, BM25 reaches 50.9% Fn Acc@10 (175/344); Perron reaches 50.3% (173/344, Holm p=1.000); PPR reaches 50.0% (172/344); tag maps reach 40.0%; bi-encoders reach 38.4% (bge-small raw 38.4% vs pooled 65.41%), while CodeRankEmbed 137M reaches 49.62% on Stratum C. Linear fusion degrades dense recall by -17.56 pp (Bellam et al. [17]). Reachability is 57.8% at 1 hop, 66.9% at 2 hops. Leave-Django-Out on N=162 tasks shows no detectable difference: A4 reaches 34.6% vs 32.7% for A2 (Δ=+1.85 pp, n10=8, n01=5, p=0.5811, recall 50.0% vs 48.8%).

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

- **Statistical Testing & GLMM**: PC1 (A4 - A2): +2.03 pp (95% CI [-0.29, +4.36], exact p=0.1435, Holm p=0.287). Task GLM (OR=1.118, p=0.0876) and repo GLMM (OR=1.130, p=0.349, σ\_repo=0.963) confirm non-significance. ±2.0 pp TOST is inconclusive (p=0.512, power 1.92%). Tango 90% CI [+0.07, +4.24] pp bounds gain below +4.24 pp. Diffusion dilutes precision (A6 45.6% vs A4 47.1%); selection conditioning drops discrimination from 0.7966 to 0.6596.
- **Pipeline Attrition**: Seeds capture 51.2% (48.8 pp loss); Graph reaches 58.4% (41.6 pp loss); Reranking retains 47.1% Top-5 (11.3 pp loss; N=100: Gemma 4 recovers 74.0% oracle vs 33.0% BM25).

### 4.3 First-Moment Expected-Rank Identity & Stratum Asymmetry

Expanding M=10 to 50 raises recall from 51.2% to 58.4%, yet Acc@5 plateaus (46.2% to 47.1% on A4; 46.2% to 45.1% on depth BM25). Expected rank E[Rank]≤k requires AUC\*≥1-(k-1)/(M-1) (AUC\*≥0.830 at mean M=24.5, k=5), bounded by Proposition 1's Markov floor (Acc@k≥1/k=0.20). At graph AUC=0.6596, expected rank is 1+23.5(1-0.6596)=9.00 (17.68 at M=50); Stratum A converts 6/11 targets (54.5%), whereas Stratum C converts 0/14.

### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary

Pointwise candidate discrimination on Dual T4s across 58 Stratum C tasks:

| Model | Archetype | Params | Precision | ROC-AUC | VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `gemma-4-e2b-it` | Dense PLE | 2.0B | 4-bit NF4 / FP16 | 0.8544 | 6698.9 MB | 1,798s |
| `gemma-4-e4b-it` | Dense PLE | 4.0B | 4-bit NF4 / FP16 | **0.9274** | 9648.4 MB | 6,250s |
| `Phi-4-mini` | Synthetic | 3.8B | 4-bit NF4 | 0.7836 | 4135.9 MB | 2,854s |
| `DeepSeek-R1-7B` | RL Distill | 7.0B | 4-bit NF4 | 0.6489 | 6594.6 MB | 5,231s |
| `gemma-4-26b-a4b-it` | MoE (128 exp) | 3.8B/26B | FP16 | *Bound* | *52.0 GB* | *Exceeds T4* |

- **Scaling & Precision Profile**: All models operate in 4-bit NF4 storage with FP16 compute kernels. Gemma 4 tail discrimination scales monotonically (0.8544 to 0.9274), exceeding Phi-4-mini (0.7836) and DeepSeek-7B (0.6489); 26B-A4B requires 52.0 GB FP16 VRAM (3D weights bypass 2D hooks), exceeding Dual T4 (29.1 GB). Pointwise E4B on A4[:15] dominates unassisted reranking: **22.90% Acc@5**, **12.21% Top-1**, MRR **0.1740** (p=0.0129\*). Under SchrödingerRepo perturbations [13], Perron achieves **Level-2 Invariance** (+34.59 pp for PPR [34.88% vs 0.29%], +34.30 pp for Hybrid [34.59% vs 0.29%], p=0.0000\*), retaining 34.88% Acc@5. On UTBoost clean tasks [14] (N=75), graph expansion lifts Acc@5 by +8.00 pp over depth BM25 (14.67% vs 6.67%, p=0.0312\*), beating Agentless [5] by +11.34 pp (47.09% vs 35.76%, p=0.0000\*). Positive Control (N=131): Union scoring reaches Top-1 **16.79%** (22/131), Acc@5 **37.4%**, MRR **0.2562** (vs BM25: 3.05% Top-1, 15.27% Acc@5).

### 4.5 Candidate Pool × Reasoning Factorial Matrix & M=15 Verification (N=131)

*Key*: N=344 Verified; N=131 Stratum C; N=58 union; N=35 at M=25 (28 seeds + 7 graph); N=34 at M=15 (28 seeds + 6 graph).

| Condition | M | Acc@1 | Acc@5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| Unscored BM25 | 25 | 3.05% | 15.27% | 0.0862 |
| Seeds Only | 10 | 3.05%→6.87% | 15.27%→10.69% (15.27%) | 0.0778→0.0865 |
| Depth BM25 | 25 | 4.58%→6.87% | 16.79%→12.98% (16.79%) | 0.0937→0.0967 |
| Graph (C0) | 25 | **4.58%→9.92%** | **18.32%** | **0.0934→0.1358** |
| C1 Stub | 25 | 2.29% | 15.27% | 0.0778 |
| C2 Static Prov | 25 | 3.82% | 16.03% | 0.0899 |
| C3 Prov + Think | 25 | 9.16% | 17.56% | 0.1270 |
| Pruned Seeds | 15 | 3.05%→7.63% | 13.74%→17.56% | 0.0821→0.1042 |
| Pruned Depth | 15 | 4.58%→8.40% | 14.50%→18.32% | 0.0954→0.1189 |
| Pruned Graph | 15 | 4.58% | 17.56% | 0.0970 |
| **Pruned Graph + Think** | **15** | **9.92%** | **19.85%** | **0.1469** |

- **Direct Readout, Qiao Backfill & Reasoning Lift**: Direct readout (M≤25) ties graph (A4) and depth BM25 (A2) at 16.79% Acc@5 and 4.58% Top-1. Listwise ranking suffered 90.8% syntax failure (119/131 tasks emitting 0 candidates), collapsing to 2.29% without Qiao backfill [15] (p < 0.001), with severe positional decay (75.0% in slots 1..5 vs 0.0% in slots 11..15) [16]. Pointwise E4B avoids decay, dominating at 22.90% Acc@5 and 12.21% Top-1 (p=0.0129\*). Reasoning lifts A4(25) Top-1 to **9.92% (13/131; diagnostic +5.34 pp, p=0.0156\*, Holm p=0.0625)** and MRR (0.0934 to 0.1358); backfill preserves 18.32% Acc@5. Reasoning disambiguates seeds (6/28 to 12/28, p=0.0312\*) over graph additions (0/7 to 1/7).
- **Provenance (C1-C3)**: Stubs (C1, 2.29%) and provenance comments (C2, 3.82%) yield no unassisted lift; provenance + thinking (C3, 9.16%) matches thinking alone (C0, 9.92%).
- **Pruning (M=15, AUC\*=0.714)**: Pruning to M=15 retains 34/35 defects, cutting stubs 40% (mean rank 5.88 to 5.32). Reasoning compacts mean rank to **3.74** (Acc@5 **19.85%**, MRR **0.1469**), retaining all 13 Top-1 hits: pruning compacts context; reasoning compacts rank.

### 4.6 Lexical Dissection & Multi-Encoding Graph Representations

To isolate representation from retrieval misses on Stratum C, Table 6 evaluates `gemma-4-e2b-it` across N=58 reachable tasks (baseline 7/58 [12.07%]):

| Condition / Encoding | Acc@1 (Direct) | Acc@1 (Think) | Acc@5 (Direct) | Acc@5 (Think) | Shift |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **O0 / E0: Natural** | 12.07% | 6.90% | 27.59% | 17.24% | -5.17 pp |
| **O1: α-Renamed** | 10.34% | 6.90% | 24.14% | 17.24% | -3.45 pp |
| **E1: Comment** | **25.86%** | **12.07%** | **39.66%** | **25.86%** | **-13.79 pp** |
| **E2: Trace** | 17.24% | 5.17% | 32.76% | 18.97% | -12.07 pp |
| **E3: Edge List** | 15.52% | 6.90% | 32.76% | 20.69% | -8.62 pp |
| **E4: Adj Map** | 15.52% | 6.90% | 34.48% | 25.86% | -8.62 pp |

1. **AST α-Renaming**: Obfuscating identifiers (O1) shifts Top-1 by -1.73 pp (p=1.0000); syntax sustains discrimination.
2. **Serialization**: Inline comments (E1) reach 25.86% Top-1 (+13.79 pp, p=0.0386\*), beating edge lists (E3, +10.34 pp, p=0.0703).
3. **Capacity Threshold**: Thinking degrades 2B models (E1: 15 to 7), while lifting 4B models (6 to 13); 2B models require provenance without reasoning.

### 4.7 Post-Training Dynamics: Top-1 Bias Shifts vs. Global Rank Variance

Tuning `gemma-4-e2b-it` (LoRA r=16) on 226 Lite tasks (loss 0.9238 to 0.0631, val Acc@1 56.4%) reaches 58.33% Acc@5 on seen repos, but regresses to 15.38% Acc@5 and 0.4846 ROC-AUC on unseen repos (N=13, Biderman et al. [18]).

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

- **Empirical Dynamics**: On isolated graph targets (N=25), clean LoRA lifts listwise Top-1 to **24.0%** (6 hits, p=0.0312\*) and MRR (0.1920 to 0.3288), but drops Acc@5 (44.0% to 36.0%) below Graph-First (52.0%). On mixed pools (N=35), clean LoRA reaches **22.9%** Top-1 and **48.6%** Top-5; comment probes collapse mixed Top-1 (17.1% to 8.6%). Unseen evaluation exposes cross-codebase limits: 15.38% Acc@5 and 0.4846 ROC-AUC confirm that LoRA memorizes repo layouts rather than transferable traversal rules.

---

## 5. Discussion

- **Why Naive Expansion Stalls Under Frozen 2B Scoring**: Offsets (+0.87 to +2.03 pp; 95% CI [-0.29, +4.36] pp) show no detectable lift over depth BM25 across task GLMs (OR=1.118) and repo GLMMs (OR=1.130, σ\_repo=0.963). Never append unpruned neighborhoods to 2B prompts without truncation and relational bridging.
- **Selection Conditioning & Structural Bounds**: Discrimination drop (0.7966 vs 0.6596) mirrors depth BM25 (p=0.721). Degrees in 10/12 repos violate log\_n C\_p < 1.0 (reaching 1.2579 in Xarray), driving PageRank into hub absorption without damping (γ=0.70). Disassortative inter-file transitions (α\*=-0.2730) explain negative weights on T³ (-0.2428). Reasoning disambiguates lexical distractors over traversing graphs.
- **Practical Operating Regimes by Gemma 4 Scale**: (1) **CPU Pre-Filtering (<5.2 ms, <=3 MB RAM)**: Damped diffusion (γ=0.70, HSI@10=99.5%), selective routing (π\*, saving 31.4% compute; 58.4% recall); (2) **Edge Regime (`gemma-4-e2b-it`, 6698.9 MB VRAM)**: Inline comments (E1, Top-1 12.07% to 25.86%); avoid reasoning (-42.9% drift) and LoRA; (3) **Workstation Regime (`gemma-4-e4b-it`, 9648.4 MB VRAM)**: Pointwise E4B on pruned A4[:15] is primary (22.90% Acc@5, 12.21% Top-1, MRR 0.1740, p=0.0129\*), eliminating 90.8% syntax failure and Qiao backfill; alternatively, prune to M=15 (AUC\*=0.714) with short reasoning (≤180 tokens) compacting rank 5.88 to 3.74 (Top-1 4.58%→9.92%); (4) **MoE Regime (`gemma-4-26b-a4b-it`, 52.0 GB FP16)**: Requires official QAT (qat-w4a16-ct); fused 3D expert weights bypass 2D hooks.

---

## 6. Community Resource Toolkit & Specification

Perron is released under Apache 2.0 (`https://github.com/yvliet/perron`):
- **Data & Specs**: 12 CSR multigraphs (101,397 nodes, 535,377 edges), 707 strata labels, `scores.parquet` (19,298 E2B and 8,113 E4B runs), SHA-256 manifests.
- **Engine & Tooling**: CSR diffusion in 1.89-5.10 ms (<3 MB RAM), `diffuse_with_provenance()`, MCP server (`python -m perron.mcp_server`), and LoRA adapter (`yvliet/perron-gemma4-e2b-lora`).
- **Notebooks**: [Gemma 4 Workstation](https://www.kaggle.com/code/yvliet/perron-gemma-4-workstation) and [Quickstart Demo](https://www.kaggle.com/code/yvliet/perron-spectral-code-graph-quickstart-demo).

---

## 7. Conclusion

Perron establishes the operational boundary for local repository code graphs. While unassisted 2B expansion on natural code stalls via selection conditioning (AUC=0.6596<0.830; 47.1% vs 45.1%, p=0.2869), topological diffusion demonstrates decisive advantages under structural stress: **Level-2 Virtual Namespace Invariance** preserves 34.88% Acc@5 under SchrödingerRepo perturbations (+34.59 pp for PPR, +34.30 pp for Hybrid over BM25 at 0.29%, p=0.0000\*), and beats Agentless AST skeletons by +11.34 pp (47.09% vs 35.76%, p=0.0000\*). On UTBoost clean tasks (N=75), graph expansion lifts Acc@5 by +8.00 pp over depth BM25 (14.67% vs 6.67%, p=0.0312\*). On workstations, Pointwise E4B scoring on pruned A4[:15] pools achieves dominance (22.90% Acc@5, 12.21% Top-1, MRR 0.1740, p=0.0129\*), eliminating listwise syntax failure (90.8%) and positional decay. Perron provides a fast (<5.2 ms), mathematically grounded CSR foundation for offline agents on consumer hardware.

---

## References

[1] Gauthier, *Aider* (2023). [2] Ouyang+, *RepoGraph* (ICLR 2025). [3] Liu+, *CodexGraph* (NAACL 2025). [4] Chen+, *LocAgent* (ACL 2025). [5] Xia+, *Agentless* (FSE 2025). [6] Jimenez+, *SWE-bench* (ICLR 2024). [7] Klicpera+, *APPNP* (ICLR 2019). [8] Fatemi+, *Talk like a graph* (ICLR 2024). [9] Sanford+, *Transformer reasoning* (NeurIPS 2024). [10] Dong+, *Graph Reranking* (2024). [11] Reddy+, *SweRank* (ICLR 2026). [12] Li+, *PRFL* (ISSTA 2017). [13] Chen+, *SchrödingerRepo* (ICSE 2024). [14] Yu+, *UTBoost* (NeurIPS 2024). [15] Qiao+, *Generative Ranking Backfill* (CIKM 2024). [16] Liu+, *Lost in the Middle* (TACL 2024). [17] Bellam+, *Dense Limits* (2025). [18] Biderman+, *Cross-Codebase* (2024).
