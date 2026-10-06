# Protocol Audit & Experimental Deviations Ledger

**Repository**: Perron (`perron-core`)  
**Author**: Sultan Haikal (GitHub: `@yvliet`)  
**License**: Apache 2.0  
**Verification Date**: October 2026  
**Scope**: Pre-Registered Confirmatory Retrieval ($N=344$) and Neural Pointwise Reranking ($N=344$)  

---

## 1. Overview & Pre-Registration Integrity

All confirmatory empirical evaluations reported in the Perron manuscript were executed under pre-registered, cryptographically sealed protocol specifications:
1. **Confirmatory Retrieval Protocol**: Git commit `6908e78`  
   - Protocol SHA-256: `89416e40d8591ef14a38343118ee493a3fd17724a87ad82361fc37cae9d3cecb`  
   - Task Split SHA-256: `95c7ef3c42875b16955fc18698bfb64d1f5cb598686cfbb8fcf05a90d3d578eb`  
2. **Gemma 4 Pointwise Reranking Protocol**:  
   - Protocol Definition: `benchmarks/protocol/gemma4_protocol_v1.yaml` (SHA-256: `96af162121e784576ee8a8929e0618774780516baae9d4f0df3e5e4faea8ef8e`)  
   - Stage 2 Cryptographic Lock: `gemma4_lock_v1.json` (SHA-256: `dbbf292023bd042b31174688174542ee904351b8d6f51cb32098b9f71c49b6b7`)  
   - Scored Data Release: `data_release/scores.parquet` (19,298 candidate evaluations)  

All pre-registered protocol gates (Gates G0 through G5) passed deterministically.

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
