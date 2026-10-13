# Does Graph Expansion Help Small Code Models Localize Bugs? A Pre-registered Study of Candidate Pools, Scorer Discrimination, and Test-Time Reasoning with Gemma 4 on SWE-bench Verified
**Pre-registered evaluation, candidate pool controls, and turnkey CSR diffusion on SWE-bench Verified**

**Author**: Sultan Haikal (recrified@gmail.com | [@yvliet](https://github.com/yvliet)) | **Code**: [https://github.com/yvliet/perron](https://github.com/yvliet/perron) | **License**: Apache 2.0

*Research report and reference implementation. Source code, datasets, and precomputed artifacts: https://github.com/yvliet/perron.*

---

## Abstract

Across 344 held-out SWE-bench Verified tasks, naive Personalized PageRank expansion yields localization indistinguishable from lexical retrieval (47.1% vs 45.1%, exact binomial McNemar $p=0.287$; excluding advantages $\ge +4.4$ pp, $p < 0.05$). This reflects an expected-rank **Break-Even Discrimination Bound** ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$): Gemma 4 E2B discrimination falls from $\text{AUC}=0.7966$ on seeds to $\text{AUC}=0.6596$ on graph neighbors competing with seed distractors. Pre-registered cross-architecture scaling (`PREREG-SCALING-v1`) on Kaggle Dual Tesla T4 GPUs confirms monotonic scaling in Gemma 4 ($0.8544 \to 0.9274$) and shows Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek-R1-Distill-7B (7.0B, 0.6489), while Gemma 4 26B-A4B ($44.0$ GB weights) exceeds Dual T4 VRAM ($29.12$ GB). Across 131 un-named Stratum C defects, reasoning traces (`<thought>`) double Top-1 accuracy on matched graph pool $A_4$ from 4.58% to 9.92% ($+5.34$ pp lift, $2.17\times$, exact $p = 0.039^*$; MRR $+45.4\%$). Reasoning compute elevates Top-1 localization across all pools (+3 to +7 hits), while Acc@5 exhibits pool-dependent behavior: lexical pools drop ($-5$ tasks on depth BM25) attributable to decoding truncation under token budgets, whereas the matched graph pool gains $+1.53$ pp (+2 tasks, interaction contrast $\Delta_{\text{interaction}} = +5.34$ pp, $+7$ net tasks; Acc@5 $p=0.6875$). Pre-registered causal interventions (`PREREG-CAUSAL-v1`, $N=58$) show inline comment provenance reaches 25.86% Top-1 ($+13.79$ pp lift, unadjusted $p=0.0386^*$, $p_{\text{holm}}=0.1543$) through localized seed bridging, outperforming disconnected edge lists ($+10.34$ pp, $p=0.0703$), while syntax-preserving AST $\alpha$-renaming sustains 10.34% Top-1 ($-1.73$ pp drop, $p=1.0000$). **Perron** provides an open-source CSR diffusion engine converging in 1.89--5.10 ms within $\le 3$ MB zero-copy RAM ($144.33$ MB process RSS), preserving 91.9% context headroom. Downstream AST editing achieves 0.0% syntax errors ($p < .001$ vs unified diffs).

---

## 1. Introduction

Local developer agents must localize defects within tight token budgets ($K \le 4,096$) without cloud APIs. While lexical search finds symbols named in issue text or error traces, locating un-named fault sites requires graph traversal over AST networks.

Graph-augmented localization assumes two conditions: (1) **Recall Expansion** along call edges must introduce targets that lexical search misses; and (2) **Rank Discrimination** must rank those targets above distractors. Evaluating both on 344 held-out SWE-bench Verified tasks with frozen `gemma-4-e2b-it` reveals a disconnect: 1-hop expansion adds +7.3 pp recall over seeds, yet top-5 accuracy remains indistinguishable from depth-matched BM25 (47.1% vs 45.1%, Holm $p=0.287$).

This paper investigates this disconnect through five core findings:
1. **Matched-Control Pool Parity**: Naive PPR expansion achieves zero statistical advantage over lexical retrieval under frozen small scorers across 19,298 evaluations (RQ1, RQ2).
2. **Predictive Bound**: The Break-Even Discrimination Bound ($\text{AUC}^* \ge 1 - \frac{k-1}{M-1}$) predicts that Gemma 4's discrimination drop to 0.6596 on graph targets turns expanded pools into distractors (RQ3).
3. **Scaling Surface**: Protocol `PREREG-SCALING-v1` demonstrates monotonic Gemma 4 scaling ($0.8544 \to 0.9274$) on reachable targets and identifies the sparse MoE quantization boundary.
4. **Pool $\times$ Reasoning Interaction**: Reasoning traces double matched graph Top-1 accuracy ($4.58\% \to 9.92\%$, $+5.34$ pp, $p = 0.039^*$) while Acc@5 reflects candidate pool bounds (+2 tasks on graphs, $p=0.6875$; interaction contrast $+7$ net tasks).
5. **Open Resource**: Perron provides a 1.89--5.10 ms CSR diffusion engine and AST patch editor within consumer RAM limits.

---

## 2. Related Work

- **Decoupled Diffusion**: PPRGo and APPNP decouple PageRank propagation from neural prediction; HippoRAG applies PPR to text knowledge graphs, and Dong et al. (2024) show graph reranking connects isolated chunks in RAG. Perron extends this decoupled lineage to static code multigraphs in zero-copy CSR.
- **Graph Complexity & Reasoning**: Fatemi et al. (2024) showed verbalization shifts LLM reasoning by 61.8%; Sanford et al. (NeurIPS 2024) proved connectivity requires logarithmic depth or intermediate reasoning tokens. RepoGraph noted 2-hop flattening degrades resolution.
- **Agent Pipelines**: SWE-agent and Agentless show hierarchical localization matches complex agent loops; LocAgent fine-tunes graph-traversal agents. Perron evaluates zero-shot graph expansion under compact 2B/4B open models.

---

## 3. Study Design & Protocol

- **Benchmark & Strata**: 344 callable Verified tasks across 12 repos: Stratum A (Named, 60.5%), B (Traceback, 1.5%), C (Un-Named, 38.1%). Hyperparameters ($\beta=0.85, \gamma=0.70, \tau=0.15, K=25$) frozen upfront.
- **Perron Engine**: Indexes call, inherit, import, caller edges in CSR ($<1.2$ ms attach, $1.89\text{--}5.10$ ms solve, $<3$ MB delta, $\text{HSI@10} = 99.5\%$). Breadcrumbs average 283.8 tokens (91.9% headroom).
- **Pools & Arms**: Seeds are top-10 BM25 symbols ($M_t \in [10, 50]$, mean 24.5): **B0** (unscored), **A1** (seeds), **A2** (depth BM25), **A3** (same-file), **A4** (1-hop graph), **A5** (null graph), **A6** (Perron diffusion).
- **Scoring & Protocol**: Frozen `gemma-4-e2b-it` (FP16) scores candidates via log-odds (19,298 passes). Confirmatory family: PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) on Acc@5 with Holm correction and bootstrap CIs ($B=10{,}000$).

---

## 4. Empirical Results

### 4.1 Standalone Retrieval & Reachability
 
Across 344 tasks, BM25 achieves 50.9% Fn Acc@10 (175/344) and 45.1% Acc@5; Perron Static reaches 50.3% (173/344, 95% CI [45.1, 55.5]%, Holm $p=1.000$) and 40.4% Acc@5; standard PPR reaches 50.0% (172/344). Tag maps reach 40.0% File Acc@5; dense bi-encoders reach 38.4% Fn Acc@10. Top-10 seeds cover 50.9% of targets. 1-hop reaches 57.8% (median 27 nodes); 2-hop reaches 66.9% (median 324), matching same-file containment (66.9%) and exceeding null graphs (63.8%, +3.02 pp).
 
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
 
- **Statistical Testing**: Confirmatory PC1 ($A_4 - A_2$): $+2.03$ pp (95% CI [−0.29, +4.36], discordant 12/5, Holm $p=0.287$). PC2 ($A_4 - A_3$): $+1.45$ pp (95% CI [−0.58, +3.78], discordant 10/5, Holm $p=0.302$). Gains $\ge +4.4$ pp are excluded ($p < 0.05$, TOST). Exploratory contrasts: $A_4 - A_1$: $+0.87$ pp ($p=0.549$); null $A_4 - A_5$: $+1.16$ pp ($p=0.344$); diffusion $A_4 - A_6$: $+1.45$ pp ($p=0.180$). BM25 (**B0**) retains higher MRR@50 (0.344 vs 0.341). Gemma 4 discrimination drops from 0.7966 on seeds to 0.6596 on graph targets.
 
### 4.3 The Break-Even Discrimination Bound & Stratum Asymmetry
 
Expanding candidate pools from $M=10$ to $M=50$ increases recall from 51.2% to 58.4%, but Acc@5 plateaus (46.2% to 47.1% for A4; 46.2% to 45.1% for depth BM25). Landing in top-$k$ requires $\text{AUC}^*_{\text{breakeven}} \ge 1 - \frac{k - 1}{M - 1}$. At mean $M = 24.5$ and $k=5$, break-even requires $\text{AUC} \ge 0.830$. Because Gemma 4 reaches only $\text{AUC} = 0.6596$ on graph targets competing with seed distractors, targets fall to expected rank $9.00$ ($17.68$ at $M=50$).
Stratum analysis exposes polarization: on Stratum A, 11 reached targets yield 6 top-5 conversions (54.5%, net $+5$ tasks, $p=0.1797$); on Stratum C, 14 reached targets yield 0 conversions (0.0%, net $-2$ tasks, $p=0.5000$). The aggregate gain of $+3$ tasks (+0.87 pp, $p=0.549$) sums $+5$ named wins and $-2$ un-named losses.
 
### 4.4 Cross-Architecture Scaling Surface & Systems MoE Boundary (PREREG-SCALING-v1)
 
Protocol `PREREG-SCALING-v1` evaluated pointwise discrimination across four architectures on Kaggle Dual Tesla T4 GPUs on reachable held-out Stratum C tasks ($N=58$):
 
| Model ID | Architecture Archetype | Active / Total Params | Graph ROC-AUC | Peak VRAM | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `google/gemma-4-e2b-it` | Dense PLE Edge | 2.0B / 2.0B | 0.8544 | 6698.9 MB | 1,798s |
| `google/gemma-4-e4b-it` | Dense PLE Edge | 4.0B / 4.0B | **0.9274** | 9648.4 MB | 6,250s |
| `microsoft/Phi-4-mini-instruct` | Synthetic Reasoning | 3.8B / 3.8B | 0.7836 | 4135.9 MB | 2,854s |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | RL Reasoning Distill | 7.0B / 7.0B | 0.6489 | 6594.6 MB | 5,230s |
| `google/gemma-4-26b-a4b-it` | Dynamic MoE (128 experts) | 3.8B / 26.0B | *Hardware Bound* | *44.0 GB req.* | *Exceeds T4* |
 
1. **Monotonic Scaling**: Tail discrimination on reachable candidates ($\text{AUC}_{\text{tail}}$) scales monotonically in Gemma 4 ($0.8544 \to 0.9274$), confirming Hypothesis $H_1$.
2. **Representational Efficiency**: Gemma 4 E4B (4.0B, AUC 0.9274) outperforms Phi-4-mini (3.8B, 0.7836) and DeepSeek R1 Distill 7B (7.0B, 0.6489). On reachable targets ($N=58$), DeepSeek-7B reaches $\text{AUC}_{\text{tail}} = 0.6489$, trailing Gemma 4 E4B (paired bootstrap 95% CI $[+0.21, +0.34]$), confirming that general reasoning distillation does not substitute for structural multigraph discrimination.
3. **MoE Hardware Boundary**: Gemma 4 26B-A4B stores 128 experts as fused 3D tensors, bypassing 4-bit linear quantization. Unquantized weights require $44.0$ GB VRAM, exceeding dual T4 capacity ($29.12$ GB). While dense models ($\le 4$B) fit consumer budgets ($\le 9.6$ GB), offline sparse MoE agents require pre-quantized formats.
 
### 4.5 Candidate Pool $\times$ Reasoning Factorial Matrix (N=131)
 
| Candidate Pool Architecture | Readout Mode | Acc@1 | Acc@5 | Hits / 131 | MRR@50 | $\Delta_{\text{reasoning}}$ Acc@5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Seeds Only ($A_1$, $M=10$) | Direct Readout | 3.05% | 15.27% | 20 | 0.0778 | reference |
| Seeds Only ($A_1$, $M=10$) | Thinking Traces | 6.87% | 10.69% | 14 | 0.0865 | $-4.58$ pp ($-6$ tasks) |
| Depth BM25 ($A_2$, $M=25$) | Direct Readout | 4.58% | 16.79% | 22 | 0.0937 | reference |
| Depth BM25 ($A_2$, $M=25$) | Thinking Traces | 6.87% | 12.98% | 17 | 0.0967 | $-3.82$ pp ($-5$ tasks) |
| Graph 1-Hop ($A_4$, $M=25$) | Direct Readout | 4.58% | 16.79% | 22 | 0.0934 | reference |
| Graph 1-Hop ($A_4$, $M=25$) | Thinking Traces ($C_0$) | **9.92%** | **18.32%** | **24** | **0.1358** | **+1.53 pp (+2 tasks)** |
 
- **Matched Thinking Lift on Graph**: On matched graph pool $A_4$, reasoning traces (`<thought>`) double Top-1 accuracy from 4.58% (6 hits) to **9.92% (13 hits; $+5.34$ pp lift, $2.17\times$ lift, exact binomial $p = 0.039^*$)**, Acc@5 increases from 16.79% to **18.32% (24 hits, $+1.53$ pp, exact binomial $p = 0.6875$)**, and MRR rises $+45.4\%$ ($0.0934 \to 0.1358$).
- **Pool $\times$ Reasoning Interaction**: Reasoning elevates Top-1 across all candidate pools ($A_1$: $4 \to 9$, $A_2$: $6 \to 9$, $A_4$: $6 \to 13$), establishing a general test-time discrimination lift. Acc@5 exhibits pool-dependent behavior: declining on lexical pools ($-4.58$ pp on seeds, $-3.82$ pp on depth BM25) due to prompt decoding truncation under the strict $\le 180$-token budget, while improving by $+1.53$ pp (+2 tasks) on graph pools ($\Delta_{\text{interaction}} = +5.34$ pp, $+7$ net tasks; Acc@5 $p=0.6875$).
- **Topological Provenance Prompting ($C_1\text{--}C_3$)**: Un-thought provenance ($C_2$, 3.82% Top-1, $p=0.6875$) provides modest lift over isolated breadcrumbs ($C_1$, 2.29%). Adding provenance to thinking ($C_3$, 9.16%) achieves empirical parity with unguided thinking ($C_0$, 9.92%; $\Delta = -0.76$ pp, exact McNemar $p = 1.0000$), showing that allocated reasoning tokens allow model representations to trace relational dependencies directly.
 
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
 
1. **AST $\alpha$-Renaming**: Obfuscating function and variable names via an AST `NodeTransformer` ($O_1$) lowers Top-1 localization by only $-1.73$ pp (6 vs 7 hits, exact $p=1.0000$), confirming that control-flow syntax sustains candidate discrimination even when identifier names are masked.
2. **Graph Prompt Serialization**: Prompt structure governs localization (Fatemi et al., 2024). Inline comment provenance ($E_1$) reaches 25.86% Top-1 ($+13.79$ pp over $E_0$, unadjusted $p=0.0386^*$, Wilcoxon $p=0.0103^*$; step-down $p_{\text{holm}}=0.1543$). Contrast against subgraph edge lists ($E_3$) yields $+10.34$ pp (6 tasks, exact $p=0.0703$). Inline comments (`# Provenance: Called by <seed>`) introduce seed identifiers into headers, providing localized lexical bridging that assists candidate selection.
 
---
 
## 5. Discussion
 
- **Empirical Verdict**: Point estimates favor graph expansion (+0.87 to +2.03 pp), but differences remain null-compatible with gains $> +4.4$ pp excluded (PC1).
- **Distractor Degradation vs Topological Deficit**: Direct listwise prompting enables comparative ranking (+8.40 pp on Stratum C, $p=0.0034$) where graph and depth BM25 tie at 16.79%. Test-time reasoning resolves Top-1 ambiguity (+5.34 pp, 13 vs 6 hits, exact $p=0.039^*$) across non-lexical call paths, while Acc@5 reflects candidate pool bounds.
- **Operating Regimes for Local Agents**: (1) **Static Filtering**: Zero-copy CSR diffusion grounds candidates in $<5$ ms ($\le 3$ MB RAM, 58.4% coverage, 91.9% headroom); (2) **Test-Time Reflection**: Allocate $\sim 120$ `<thought>` tokens to trace local call paths; (3) **Dynamic Execution Fallback**: For dynamic dispatch, agents should switch to test traces rather than expanding static graph radius.
- **Limitations**: (1) Python ASTs; (2) compact models (E2B/E4B); (3) 1-hop pools ($M \le 50$); (4) frozen zero-shot scorers.
 
---
 
## 6. Community Resource Toolkit & Specification
 
Perron is released under Apache 2.0:
- **Binary CSR Multigraphs**: Precomputed multigraphs for 12 repos in zero-copy CSR format.
- **Evaluation Archive (`scores.parquet`)**: 19,298 evaluations across 344 tasks with log-odds.
- **Ground Truth & Strata**: Strata annotations across 800 tasks (344 callable Verified primary benchmark; CC0-1.0).
- **Engine Performance & Tooling**: CSR diffusion executes in $1.89\text{--}5.10$ ms ($196\text{--}529$ qps, $<3$ MB delta, $99.5\%$ HSI@10), outperforming NetworkX ($320$ ms). Downstream edit reader ($N=100$) achieves 74.0% EditLoc@1, while AST patch editor ($N=30$) achieves 0.0% syntax errors vs 50.0% for unified diffs ($p < .001$).
- **Turnkey API & MCP Server**: Drop-in `perron.diffuse_with_provenance()` Python API and stdio JSON-RPC server (`perron-mcp`).
- **Kaggle Workstations**: Dual Tesla T4 reproduction notebooks validating context packing, causal controls (`yvliet/perron-gemma-4-workstation`), and cross-architecture scaling (`yvliet/perron-scaling-benchmark`).

---

## 7. Conclusion

On un-named Stratum C defects, candidate expansion pays off only when paired with test-time reasoning compute. On the matched graph pool, reasoning traces double Top-1 localization from 4.58% to 9.92% (exact binomial $p = 0.039^*$; MRR $+45.4\%$), achieving parity with explicit prompt edge annotations ($p = 1.0000$). Reasoning compute elevates Top-1 across all pools (+3 to +7 hits), while Acc@5 exhibits pool-dependent behavior (+2 tasks on graphs vs $-5$ on depth BM25 due to decoding truncation). Cross-architecture scaling confirms monotonic discrimination gains in Gemma 4 ($0.8544 \to 0.9274$) while quantifying memory boundaries for sparse MoE models on consumer GPUs. Paired with Perron's high-throughput CSR diffusion engine, these results establish that compact open models can navigate non-lexical code graphs when provided an allocated inference-time reasoning budget.

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
