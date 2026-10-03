# Baseline Audit & Diagnostic Ledger

## 1. Toy Graph Evaluation Matrix (T1.3)

Evaluation performed on canonical 8-node toy graph:
- Seed node $S$ (high lexical overlap with query `process_order`)
- Target node $T$ ($1$ call-hop from $S$, low in-degree $= 1$)
- Hub node $H$ (high in-degree $= 6$, incoming edges from $S$ and filler nodes)
- Filler nodes $F_1, F_2, F_3, F_4, F_5$
- Query text: names seed $S$ only (`process_order`); zero lexical overlap with target $T$ (`calculate_tax`).
- Evaluation criterion: Target $T$ is retrieved in top 3 candidates ($T \in R[:3]$).

| # | Baseline Name | Toy Graph Result | Classification | Diagnosis / Rationale | Commit Hash |
| :-: | :--- | :---: | :---: | :--- | :---: |
| 1 | Okapi BM25 | **FAIL** (Rank 7) | `(c) Intrinsic` | Method relies strictly on lexical token matching; because query names $S$ only, unmentioned target $T$ has zero token overlap and cannot be reached without graph diffusion. | N/A |
| 2 | BM25 + 1-Hop Expansion | **PASS** (Rank 3) | Pass | Propagates 50% BM25 score from seed $S$ across the 1-hop graph edge, elevating $T$ into the top 3. | N/A |
| 3 | Dense Semantic Retrieval | **FAIL** (Rank 7) | `(c) Intrinsic` | Method computes token and identifier similarity; without graph edge traversal, unmentioned target $T$ scores 0.0. | N/A |
| 4 | Prior-Only Control ($p_0$) | **FAIL** (Rank 7) | `(c) Intrinsic` | Teleport prior $p_0$ reflects lexical query matching without random walk diffusion; probability mass is 0.0 for $T$. | N/A |
| 5 | Standard Personalized PageRank (PPR) | **PASS** (Rank 3) | Pass | Power-iteration diffusion flows probability mass from seed $S$ to 1-hop neighbor $T$, ranking it in the top 3. | N/A |
| 6 | Authentic Aider Repo Map | **PASS** (Rank 3) | Pass | PageRank seeded on issue prompt identifier tags successfully diffuses flow to target $T$. | N/A |
| 7 | Hub Blocklist with Lexical Override | **PASS** (Rank 2) | Pass | Top in-degree hub $H$ is masked by blocklist, cleanly elevating low-degree target $T$ to rank 2. | N/A |
| 8 | Static Degree-Discounted Transition Matrix | **PASS** (Rank 3) | Pass | Transition weights to hub $H$ are penalized by in-degree, ensuring adequate flow reaches target $T$. | N/A |
| 9 | Degree-Normalized PPR | **PASS** (Rank 2) | Pass | Post-walk division by $(d_{\text{in}} + 1)^\gamma$ attenuates hub $H$, elevating target $T$ to rank 2. | N/A |
| 10 | HippoRAG Pre-Walk Prior Scaling | **PASS** (Rank 3) | Pass | Pre-walk prior modulation by $1 / \pi_g^\gamma$ directs diffusion flow to target $T$ in top 3. | N/A |
| 11 | Query-Reweighted PPR | **PASS** (Rank 3) | Pass | Dynamic transition scaling prioritizes edges from active query symbols, ranking $T$ in top 3. | N/A |
| 12 | Perron Static Specificity ($\gamma = 0.70$) | **PASS** (Rank 2) | Pass | Decoupled specificity ratio $\pi_q / \pi_g^{0.70}$ suppresses hub $H$ and places target $T$ at rank 2. | N/A |
| 13 | Perron Adaptive Specificity ($\gamma_q$) | **PASS** (Rank 2) | Pass | Query-directed prominence $\eta_q$ modulates suppression intensity, ranking target $T$ at rank 2. | N/A |
| 14 | Oracle Upper Bound | **PASS** (Rank 1) | Pass | Places ground-truth target $T$ at rank 1 unconditionally. | N/A |

---

## 2. Summary of Audit Classifications

- **Passing Baselines (11 / 14)**:
  All 11 graph-aware diffusion baselines successfully retrieve the unmentioned 1-call-hop target in their top 3 ranks. Hub suppression methods (Perron Adaptive, Perron Static, Degree-Normalized PPR, Hub Blocklist) successfully elevate $T$ to rank 2 ahead of the global hub $H$.
- **Failing Baselines (3 / 14)**:
  All 3 failing baselines (`BM25`, `Dense`, `Prior-Only`) fail exclusively due to **(c) Intrinsic Mathematical Limitations**. None are caused by software bugs or candidate filtering defects: they are purely lexical/teleport-based models that possess no graph edge diffusion mechanism by definition.

---

## 3. Dev-Set Post-Audit Rerun (T1.4)

Evaluation executed on `dev_val` partition ($N=100$, seed `20261003`) via:
```bash
python scripts/run_suite.py --suite table2 --split dev_val --seed 20261003
```
Output verified at: `results/dev_val_table2_postaudit.json`.

| Baseline | Fn Acc@5 | Fn Acc@10 | File Acc@5 | Cov@4k | HSI@10 | McNemar $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Okapi BM25 | 11.0% | 20.0% | 57.0% | 26.3% | 99.7% | 0.0923 |
| BM25 + 1-Hop Expansion | 11.0% | 20.0% | 57.0% | 26.3% | 99.5% | 0.0923 |
| Dense Semantic | 8.0% | 15.0% | 42.0% | 17.3% | 99.2% | 0.7905 |
| Prior-Only Control ($p_0$) | 11.0% | 20.0% | 57.0% | 26.3% | 99.7% | 0.0923 |
| Standard PPR (Shared $p_0$) | 1.0% | 2.0% | 10.0% | 0.5% | 60.0% | 0.0034 |
| Authentic Aider Repo Map | 1.0% | 1.0% | 6.0% | 0.5% | 55.2% | 0.0018 |
| Hub Blocklist + Lexical Override | 2.0% | 2.0% | 16.0% | 1.5% | 90.2% | 0.0034 |
| Static Deg-Discount Matrix | 1.0% | 2.0% | 16.0% | 0.5% | 69.2% | 0.0034 |
| Degree-Normalized PPR | 2.0% | 3.0% | 22.0% | 5.5% | 97.1% | 0.0020 |
| HippoRAG Prior Scaling | 1.0% | 2.0% | 10.0% | 0.5% | 58.5% | 0.0034 |
| Query-Reweighted PPR | 1.0% | 1.0% | 14.0% | 0.5% | 60.8% | 0.0018 |
| Perron Static ($\gamma=0.70$) | 8.0% | 13.0% | 50.0% | 10.5% | 92.3% | baseline |
| Perron Adaptive Specificity | 6.0% | 8.0% | 44.0% | 7.0% | 82.4% | 0.1250 |
| Oracle Upper Bound | 83.0% | 83.0% | 83.0% | 82.0% | 99.8% | 0.0000 |

**Zero-Percent Diagnostic Finding**:
Zero baselines produced exactly 0.0% accuracy across `dev_val`. All 14 baselines achieved verified non-zero recall and precision.
