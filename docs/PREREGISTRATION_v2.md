# Benchmark Preregistration Protocol: SWE-bench Verified Confirmatory Evaluation

> **Document Version**: 2.0.0  
> **Date**: 2026-10-04  
> **Repository**: `yvliet/perron`  
> **Author**: Sultan Haikal (GitHub: `@yvliet`)  
> **Benchmark Target**: SWE-bench Verified Confirmatory Code Graph Retrieval ($N \approx 220$ callable tasks)

---

## 1. Study Objective & Hypothesis Pre-Specification

Following exploratory 2x2 factorial analysis across teleport prior dispersion (dense linear vs. sharp softmax) and graph topology (raw vs. sanitized), this preregistration establishes the confirmatory hypothesis testing protocol on an uncontaminated sample of SWE-bench Verified tasks.

### Confirmatory Hypotheses

- **Hypothesis P1 (Decoupled Specificity vs. Standard PPR at Equal Damping)**:
  Under a sharp teleport prior ($\tau = 0.15$) and sanitized graph topology (test node suppression), Perron Decoupled Specificity ($\beta_q = 0.85, \beta_g = 0.85, \gamma = 0.70$) achieves statistically superior Function Acc@10 compared to Standard PPR evaluated at the identical damping parameter $\beta_q = 0.85$ ($p < 0.05$ under exact two-sided McNemar test).

- **Hypothesis P2 (Lexical Non-Inferiority on Function Localization)**:
  On callable bug localization tasks ($\mathcal{S}_{\text{func}}$), Perron Decoupled Specificity achieves non-inferior Function Acc@10 relative to Okapi BM25 (margin $\Delta \ge -2.0$ pp).

- **Hypothesis P3 (Structural Complementarity & Lexical Stratification)**:
  Perron and BM25 exhibit orthogonal failure modes across lexical strata:
  - **Stratum A (Direct Name Matches)**: BM25 achieves superior localization where issue text explicitly names target identifiers.
  - **Stratum C (Multi-Hop Unnamed Invocations)**: Perron significantly outperforms BM25 by propagating query mass across call and inheritance paths without requiring identifier lexical overlap ($p < 0.05$).
  - **Reciprocal Rank Fusion**: RRF fusion ($k = 60$) of Perron and BM25 strictly outperforms either single modality alone on overall Function Acc@10.

---

## 2. Frozen Hyperparameters (Calibrated on Exploratory Pool)

All hyperparameters are permanently frozen prior to the confirmatory execution on SWE-bench Verified:

| Parameter | Frozen Value | Architectural Role |
| :--- | :--- | :--- |
| $\beta_q$ | $0.85$ | Query random walk teleport damping parameter |
| $\beta_g$ | $0.85$ | Decoupled global stationary PageRank damping parameter |
| $\gamma$ | $0.70$ | Static Specificity ratio exponent: $\pi_q(v) / (\pi_g(v) + \epsilon)^\gamma$ |
| Prior Mode | `softmax` | Sharp top-$K$ softmax teleport prior: $\tau = 0.15, K = 25$ |
| Hub Adaptivity $\tau_{\text{lex}}$ | $0.20$ | Threshold relative prominence for relaxing $\gamma$ on intentional hub targets |
| Test Filter | `True` | Strict $-1\text{e}9$ masking of test nodes in retrieval rankings (0% top-10 contamination) |
| Max Iterations | $100$ | Power iteration convergence ceiling ($\text{tol} = 1\text{e}-6$) |

---

## 3. Dataset Specification & Contamination Firewall

### Target Sample: SWE-bench Verified ($N \approx 220$ Callable Tasks)
- Drawn from `princeton-nlp/SWE-bench_Verified` across the 12 canonical repositories (`astropy`, `django`, `matplotlib`, `seaborn`, `flask`, `requests`, `xarray`, `pylint`, `pytest`, `scikit-learn`, `sphinx`, `sympy`).
- Instances are filtered to exclude the 300 instances belonging to SWE-bench Lite (which served as the exploratory analysis pool).
- Instances are partitioned into:
  1. **Primary Set $\mathcal{S}_{\text{func}}$ ($N \approx 220$)**: Verified bug fixes touching callable function or method AST entities.
  2. **Secondary Set $\mathcal{S}_{\text{file}}$**: Configuration, test file, or module-level variable fixes evaluated exclusively for container File Acc@k.

---

## 4. Pre-Registered Metrics & Statistical Protocol

### Endpoints
1. **Primary Endpoint**:
   - **Function Acc@10** on $\mathcal{S}_{\text{func}}$: Indicator that at least one gold function symbol appears in top-10 retrieved symbols.
2. **Secondary Endpoints**:
   - **Function Acc@5** on $\mathcal{S}_{\text{func}}$.
   - **File Acc@5** across all evaluated instances.
   - **Hub Suppression Index (HSI@10)**: Proportion of top-10 retrieved symbols outside the repository top-25 hubs.
   - **Stratum C Function Acc@10**: Retrieval accuracy specifically on instances where issue descriptions omit gold function names.

### Statistical Testing Protocol
1. **Paired Bootstrap Confidence Intervals**:
   - 10,000 paired repo-stratified bootstrap resamples ($B = 10,000$, seeded deterministically).
   - Preserves repository cluster proportions across all resamples.
   - Reports empirical 95% bootstrap confidence intervals for all means and pairwise differences $\Delta = \text{Score}_{\text{Perron}} - \text{Score}_{\text{Baseline}}$.
2. **Exact Paired Significance Testing**:
   - Two-sided exact McNemar test on discordant pairs ($n_{10}$ vs. $n_{01}$) between Perron and each competing baseline.
3. **Multiple Testing Correction**:
   - Step-down Holm-Bonferroni correction applied across the family of 13 baseline comparisons against Perron to control family-wise error rate $\alpha = 0.05$.
4. **Deterministic Tie-Breaking**:
   - All retrieval score ties are broken deterministically by ascending integer AST symbol ID.
