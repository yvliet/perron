# SWE-bench Lite Code Graph Release Validation Report

**Date**: October 3, 2026  
**Scope**: All 300 SWE-bench Lite instance multigraphs  
**Status**: PASS  

---

## 1. Summary Statistics

| Metric | Value |
| :--- | :--- |
| Total Instances Audited | 300 |
| Instances Passing All Gates | 300 (100.0%) |
| Instances Failing Gates | 0 |
| Total AST Nodes | 4,876,848 |
| Total Directed Multigraph Edges | 41,591,756 |
| Mean Nodes per Repository | 16256.2 |
| Mean Edges per Repository | 138639.2 |
| Gold Target Symbols Verified | 285 |
| Gold Target Symbols Matched in AST | 285 (100.0%) |

---

## 2. Invariant Verification Gates

1. **SHA-256 Digest Integrity**: Every numpy array and parquet table matches manifest digest bit-for-bit.
2. **Topological Bounds**: Strictly zero dangling edge indices ($0 \le \text{col} < N$).
3. **Row-Stochastic Normalization**: All non-sink rows sum to $1.0 \pm 10^{-4}$; all sink rows flagged with $d_u = 1.0$.
4. **Symbol Table Uniqueness**: Node IDs in `nodes.parquet` are unique, strictly contiguous $[0, N-1]$.

---

## 3. Detailed Failure Ledger

No validation failures detected. All instances passed 100% of topological and integrity gates.
