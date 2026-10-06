# Gemma 4 Pre-Registered Protocol Evaluation Results

**Sample**: N = 344 callable SWE-bench Verified tasks across 12 canonical repositories.

## Table T-A: Retrieval and Reranking Accuracy Across Arms

| Arm | Description | Acc@1 (95% CI) | Acc@3 (95% CI) | Acc@5 (95% CI) | Acc@10 (95% CI) | MRR@50 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B0** | BM25 Baseline | 25.3% [21.0, 30.1] | 38.7% [33.7, 43.9] | 44.8% [39.6, 50.0] | 51.2% [45.9, 56.4] | 0.3439 |
| **B1** | Perron Static Baseline | 22.7% [18.6, 27.4] | 35.2% [30.3, 40.4] | 40.4% [35.4, 45.7] | 50.3% [45.0, 55.5] | 0.3155 |
| **A1** | Rerank Seeds S (M=10) | 24.7% [20.4, 29.5] | 39.5% [34.5, 44.8] | 46.2% [41.0, 51.5] | 51.2% [45.9, 56.4] | 0.3348 |
| **A2** | Rerank Depth-Matched BM25 | 23.8% [19.6, 28.6] | 38.1% [33.1, 43.3] | 45.1% [39.9, 50.3] | 51.2% [45.9, 56.4] | 0.3301 |
| **A3** | Rerank Same-File Matched | 24.1% [19.9, 28.9] | 39.2% [34.2, 44.5] | 45.6% [40.5, 50.9] | 52.0% [46.8, 57.3] | 0.3381 |
| **A4** | Rerank Graph 1-Hop | 24.7% [20.4, 29.5] | 39.2% [34.2, 44.5] | 47.1% [41.9, 52.4] | 52.3% [47.0, 57.5] | 0.3409 |
| **A5** | Rerank Null Graph 1-Hop | 24.1% [19.9, 28.9] | 39.0% [34.0, 44.2] | 45.9% [40.7, 51.2] | 51.2% [45.9, 56.4] | 0.3315 |
| **A6** | Rerank Perron Diffusion | 23.6% [19.4, 28.3] | 38.7% [33.7, 43.9] | 45.6% [40.5, 50.9] | 52.6% [47.3, 57.8] | 0.3301 |

## Table T-B: Pre-Registered Confirmatory Contrasts (Tier 1)

**Primary Endpoint**: Acc@5 (Strict function match) | **Verdict**: `NOT_DEMONSTRATED`

| Contrast | Endpoint | Delta (pp) | 95% Bootstrap CI | n10 / n01 | p (Exact) | p (Holm) | Classification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PC1 (A4 - A2)** | Acc@5 | +2.03 | [-0.29, +4.36] | 12 / 5 | 0.1435 | 0.2869 | `NULL_COMPATIBLE` |
| **PC2 (A4 - A3)** | Acc@5 | +1.45 | [-0.58, +3.78] | 10 / 5 | 0.3018 | 0.3018 | `NULL_COMPATIBLE` |

## Table T-C: Candidate Pool Diagnostics

- **Candidate Pool Size (M_t)**: Mean 24.5, Median 22.0 (Range: [10, 50])
- **A3 Same-File Fill Rate**: 2.0% tasks required filling (Mean fill: 0.1 symbols)
- **A5 Null Graph Fill Rate**: 70.1% tasks required filling (Mean fill: 5.2 symbols)
- **Jaccard Similarity with Graph Pool (A4)**: J(A4, A2) = 0.38, J(A4, A3) = 0.419
- **Graph-Unique Candidates**: 11.4 symbols/task (3907 total)

## Table T-E: Downstream Edit-Location Mini-Test (N=100, Descriptive)

| Arm | Candidate Context | ContextHit@5 | EditLoc@1 | EditLoc@3 | Conversion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D0** | BM25 Top-5 | 35.0% | 33.0% | 35.0% | 0.943 |
| **D1** | A4 Graph Top-5 | 37.0% | 31.0% | 37.0% | 0.838 |
| **D2** | A3 Control Top-5 | 38.0% | 30.0% | 37.0% | 0.789 |
| **D3** | Oracle Context (Transmission) | 100.0% | 74.0% | 85.0% | 0.740 |

- **Reader Transmission Rate (D3)**: 74.0% (Chance-adjusted: 67.5%)
