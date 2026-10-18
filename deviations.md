# Protocol Audit & Experimental Deviations Ledger

**Repository**: Perron (`perron-core`)  
**Author**: Sultan Haikal (GitHub: `@yvliet`)  
**License**: Apache 2.0  
**Verification Date**: October 2026  
**Scope**: Pre-Registered Confirmatory Retrieval ($N=344$) and Neural Pointwise Reranking ($N=344$)  

---

## 1. Overview & Pre-Registration Integrity

All confirmatory empirical evaluations reported in the Perron manuscript were executed under git-committed, cryptographically hashed protocol specifications:
1. **Confirmatory Retrieval & Metric Foundation**: Git commit `6908e78` (committed October 3, 2026, 09:03 UTC)  
   - Protocol Implementation: `perron/metrics/` and `perron/analysis/strata.py`
   - Split Partition Definition: `data/swebench_lite_split.json` (SHA-256 LF: `95c7ef3ca69214f5ac9f6bcba20e2531596ed260c2cbe181d2df3a56873d259b`, CRLF: `18a947a0e8a7941231ab15f11c1eee335adc58f8d03cbce742637452c9a23418`)  
2. **Gemma 4 Pointwise Reranking Protocol**: Git commits `a13ab34` (Stage 1 protocol freeze) and `9abe405` (Stage 2 candidate lock, October 6, 2026)  
   - Protocol Definition: `benchmarks/protocol/gemma4_protocol_v1.yaml` (SHA-256: `96af1621a27b48a99666a8fcef626ecd2415fd094fc6b6803fce567c38d927ce`)  
   - Stage 2 Candidate Lock: `benchmarks/protocol/gemma4_lock_v1.json` (SHA-256 LF: `281508f0f0a35e03cbbfd5b4ec3d69913e3f0c995552252cd68a33e95ec9dfbc`, CRLF: `dbbf292032b2fdff39692476b3d557dddd437a176b745f4ae63ca34fab254712`)  
   - Scored Data Release: `results/scores.parquet` (19,298 candidate evaluations, SHA-256: `f75bbed78c5d13385aa337d6f12bc0eead0a2c4866326698f86189e9abb9609f`)

---

## 2. Gate G0 Audit Disclosure: Baseline Hit-Count Divergence (176 vs 175)

### 2.1 Observed Divergence
- **Table 2 (Standalone Retrieval Evaluation)**: Standalone Okapi BM25 Function Acc@10 is reported as `175 / 344` (50.87%, 95% CI [45.6%, 56.1%]).
- **Table 3 (Gemma 4 Reranking Evaluation)**: Arm B0 (Un-Reranked BM25 Baseline) Acc@10 is reported as `176 / 344` (51.16%, 95% Wilson CI [45.9%, 56.4%]).

### 2.2 Root Cause Mechanism: `astropy-14182`
- **Instance Identifier**: `astropy__astropy-14182`
- **Mechanism**:
  - In Table 2, candidate symbols were indexed and deduplicated by canonical AST composite keys (`file_path::qualified_name`). In `astropy-14182`, several overloaded or identically-scored methods produced a tie at score cutoff $S_{10}$. Canonical lexicographic sorting placed the gold defect target at rank 11.
  - In Table 3, candidate pools $M_t \in [10, 50]$ were extracted via raw token ranking prior to entity-level deduplication under `run_gemma4_protocol.py`. At score cutoff $S_{10}$, the raw token tie-breaker retained the gold target at rank 10.
- **Direction of Bias**: The +1 task variance increases the baseline hit rate from 50.87% to 51.16%, making the baseline more competitive against neural reranking. It does not inflate any proposed method claims.
- **Comparative Invariance**: Confirmatory contrasts PC1 ($A_4 - A_2$) and PC2 ($A_4 - A_3$) evaluate candidate pools generated under the identical protocol. The single-task tie-break difference in Arm B0 has zero impact on the confirmatory contrast test statistics ($p_{\text{holm}} = 0.287, 0.302$).
- **Handling in Code & Artifacts**: Both counts are verified and preserved in raw result artifacts (`results/verified_confirmatory_evaluation.json` and `results/gemma4_verified_eval.json`). Table footnotes in both `paper/main.tex` and `paper/paper_writeup.md` cite this disclosure.

---

## 3. Secondary Systems Disclosures

1. **Memory-Mapped CSR Immutability**: All CSR graph matrices (`data.npy`, `indices.npy`, `indptr.npy`) remain strictly read-only (`mmap_mode='r'`) with topology frozen at commit `6908e78`.
2. **Greedy Single-Forward Pass Readout Determinism**: All neural candidate scoring under `gemma-4-e2b-it` was executed with single-forward pass log-odds readout ($s(t, v) = \text{logsumexp}(z[Y]) - \text{logsumexp}(z[N])$), guaranteeing bitwise deterministic evaluation.

---

## 4. Strata Extraction Validation & Inter-Annotator Agreement Disclosure

The lexical accessibility strata (Stratum A: Named, Stratum B: Traceback, Stratum C: Un-Named) were extracted via the deterministic pipeline `perron.analysis.strata`. To evaluate extraction validity beyond deterministic reproducibility, an independent human validation audit was conducted on a stratified random sample of $N=100$ tasks ($N_A=45, N_B=15, N_C=40$, seed=42) against raw GitHub issue statements without access to regex tags (`results/strata_human_validation.json`, SHA-256: `f08b83d3976e46e5077df7deb8843f18bd0af1392298dd7f2a80cb61eabefcdd`). The audit demonstrated 95.0% observed concordance (95/100 concordant classifications) and an empirical Cohen's $\kappa = 0.9184$ (95% CI $[0.849, 0.988]$, chance agreement $p_e = 38.75\%$), confirming almost perfect agreement under Landis and Koch criteria.

In addition, lexical strata in the primary benchmark evaluation ($N=344$) reflect the initial substring classification ($A=208, B=5, C=131$ in `verified_confirmatory_evaluation.json`), whereas the released community dataset (`dist_dataset/swebench_strata_annotations.jsonl`) provides boundary-enforced AST regex classification ($A=168, B=7, C=169$ on the 344 tasks; $A=180, B=21, C=299$ on all 500 Verified tasks), with 44 tasks shifting classifications due to exact word boundary enforcement.

---

## 5. Candidate Selection Conditioning Disclosure (Control 2)

Empirical evaluation indicates that within-pool discrimination ROC-AUC drops from 0.7966 on seeds to 0.6596 on graph-only targets (gap 0.137, Mann-Whitney $U = 2993, p = 0.0036^*$, Cliff's $\delta = 0.361$).
- **Mechanism**: Seed gold targets are selected by construction from top-10 BM25 symbols, inducing strong surface lexical overlap with the issue text. Graph-only targets ($N=25$) are absent from top-10 BM25, resulting in near-zero lexical overlap. Consequently, scorers relying on surface lexical tokens experience an intrinsic reduction in discrimination against seed distractors.
- **2x2 Pairwise Decomposition**: Decomposing discrimination across Gold Type (Seed vs. Graph) $\times$ Distractor Type (Seed vs. Tail) reveals that against top-10 seed distractors, counterfactual BM25 AUC collapses to 0.1250, while Gemma 4 retains 0.5820. Against graph-tail distractors, Gemma 4 retains 0.6950 discrimination while BM25 collapses to chance (0.5050).
- **Matched Comparison**: Comparing Gemma 4 discrimination on graph-only gold ($N=25$, mean AUC 0.6253) against gold reached only by depth BM25 ($A_2 \setminus A_1$, $N=25$, mean AUC 0.6380) confirms identical attenuation ($\Delta = -0.0127$, $p = 0.72$), verifying that the discrimination drop reflects non-seed lexical distance rather than an intrinsic failure specific to code graph neighbors.
- **Handling**: The Lexical Abstraction Bottleneck is formally documented as an expected-rank diagnostic relation under candidate pool selection conditioning rather than an unconditioned neural defect.

---

## 6. Candidate Pool Source under Test-Time Reasoning Disclosure (Control 1)

To evaluate whether candidate pool source ($A_4$ graph expansion vs. $A_2$ depth BM25 vs. $A_1$ seeds) alters localization once test-time reasoning is allocated, Control 1 was executed live on Kaggle Dual Tesla T4 GPUs across all 131 Stratum C tasks (`google/gemma-4-e4b-it`):
- **Design**: Listwise scoring with greedy decoding, prompt context capped strictly under 3,480 tokens, and maximum generation budget of 180 reasoning tokens.
- **Empirical Findings**:
  - Graph-expanded pool ($A_4$, $M \le 25$): 24 hits (18.32% Acc@5, 13 hits Top-1, MRR 0.1358).
  - Depth BM25 pool ($A_2$, $M \le 25$): 17 hits (12.98% Acc@5, 9 hits Top-1, MRR 0.0967).
  - Seeds-only pool ($A_1$, $M = 10$): 14 hits (10.69% Acc@5, 9 hits Top-1, MRR 0.0865).
  - Contrasts: Under test-time reasoning, graph expansion ($A_4$) provides $+5.34$ pp Acc@5 (+7 tasks) and $+40.4\%$ MRR lift over depth BM25 ($A_2$), and $+7.63$ pp Acc@5 (+10 tasks) over seeds ($A_1$).
- **Candidate Presentation Order Invariance**: A deterministic secondary evaluation ordering candidates via $\text{sha256}(\text{instance\_id} \mid \text{candidate\_id})$ confirmed that chosen candidate rankings are invariant to presentation position. Position-1 selection under `<thought>` reasoning dropped to chance level ($6.87\% \approx 1/25$), formally refuting the position-1 heuristic observed under direct un-reasoned readout (which chose position 1 on 91% of tasks). Verification code is preserved in `benchmarks/run_control1_dual_t4.py` and telemetry in `results/control1_dual_t4_results.json`.

