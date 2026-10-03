# Benchmark Preregistration Protocol: Code Graph Retrieval Evaluation

> **Document Version**: 1.0.0  
> **Date**: 2026-10-03  
> **Repository**: `yvliet/perron`  
> **Author**: Sultan Haikal (GitHub: `@yvliet`)  
> **Benchmark Target**: SWE-bench Lite Task-Grounded Code Graph Retrieval

---

## 1. Study Objective & Hypothesis Pre-Specification

This document pre-registers the empirical evaluation design, statistical endpoints, hypotheses, and analysis protocols for evaluating graph diffusion algorithms on SWE-bench Lite repository graphs.

### Hypotheses
- **Hypothesis 1 (Primary Function-Level Retrieval)**: Perron Adaptive Specificity ($\gamma_q$ with scale-invariant relative hub prominence $\eta_q$) achieves non-inferior Function Acc@10 compared to the strongest tuned baseline on $\mathcal{S}_{\text{func}}$, while significantly outperforming Hub Blocklist with Lexical Override on intentional hub-gold defect targets.
- **Hypothesis 2 (Spectral Decoupling Equivalence)**: Post-walk decoupling Specificity produces ranking outcomes statistically indistinguishable from dynamic Query-Reweighted PPR ($p > 0.05$ under exact McNemar test), while eliminating Copy-on-Write page duplication across parallel agent workers ($W \to 1$ OS page cache sharing).

---

## 2. Dataset Partitioning & Contamination Firewall

To guarantee 100% blinded evaluation and prevent exploratory data snooping, the 300 SWE-bench Lite tasks are partitioned into a strict tri-partition manifest (`data/swebench_lite_split.json`):

1. **`dev_pilot` ($N = 50$)**:
   - The 50 exploratory instances used during initial parameter discovery (6 Requests + 44 SymPy).
   - Quarantined strictly to historical sanity checking. Zero instances enter the held-out split.
2. **`dev_val` ($N = 100$)**:
   - Uncontaminated validation split stratified across remaining instances.
   - Used for calibration of baseline hyperparameters ($k_1, b, \gamma, \tau_{\text{lex}}$).
3. **`heldout` ($N = 150$)**:
   - Strictly blinded test split evaluated exactly once under frozen parameters.
   - Pinned by SHA-256 digests of task cache and gold annotations.

---

## 3. Pre-Registered Metrics & Analysis Sets

### Evaluation Sets
- **Primary Callable Set $\mathcal{S}_{\text{func}}$ ($N = 227$, 75.7%)**:
  - Tasks with at least one verified callable AST function or method node in base-commit syntax trees.
  - Mandatory target set for symbol-level metrics.
- **Secondary File-Level Set $\mathcal{S}_{\text{file}}$ ($N = 73$, 24.3%)**:
  - Tasks whose human diffs modify module-level constants, configuration tables, or class body declarations without touching callable functions.
  - Evaluated strictly for container-level File Acc@k.

### Endpoints
1. **Primary Endpoint**:
   - **Function Acc@10** on $\mathcal{S}_{\text{func}}$: Percentage of tasks where at least one ground-truth function symbol appears in top-10 retrieved symbols.
2. **Secondary Endpoints**:
   - **File Acc@5** across all 300 instances.
   - **Context Coverage@4096**: Proportion of gold symbols retained within a 4,096-token context budget.
   - **Context Purity@25**: Proportion of top-25 retrieved symbols that are gold-relevant.
   - **Relative HSI ($P_{99.5}$)**: Proportion of top-10 retrieved symbols that avoid the top 0.5% in-degree hubs of the repository.

---

## 4. Statistical Testing Protocol

1. **Confidence Intervals**:
   - 10,000 paired repo-stratified bootstrap resamples ($B = 10,000$).
   - Within each repository stratum $r$, resample $N_r$ instances with replacement.
   - Report empirical 95% bootstrap confidence intervals for all means and pairwise differences $\Delta = \text{Score}_{\text{Perron}} - \text{Score}_{\text{Baseline}}$.
2. **Hypothesis Testing**:
   - Exact two-sided McNemar paired test on discordant binary outcomes ($n_{10}$ vs $n_{01}$) between Perron and each competing baseline.
3. **Multiple Testing Correction**:
   - Step-down Holm-Bonferroni correction applied across all pairwise comparisons against Perron to maintain family-wise error rate $\alpha = 0.05$.
4. **Deterministic Tie-Breaking**:
   - All score ties during ranking are broken deterministically by ascending AST symbol ID.
