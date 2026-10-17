# Historical Task AST Recovery Report (N=56 Excluded Tasks)

**Author**: Sultan Haikal (`@yvliet`)  
**Tasks Audited**: 56  
**Recovery Success Rate**: 56/56 (100.0%)  

## 1. Executive Forensic Synthesis

The manuscript previously attributed the exclusion of 56 tasks to 'Python 2 syntax drift'. Our empirical audit refutes this claim:
- **Git Show at Base Commit**: 68/68 Python files (100.0%) exist at their respective task `base_commit`.
- **Standard Python 3 `ast.parse`**: 100.0% of recovered source files parse without syntax errors.
- **Total Recoverable Callable Symbols**: 3827 function, class, and method declarations extracted.
- **Gold Target Matching**: 56/56 tasks successfully map to their ground truth target symbols.

## 2. Population & Strata Breakdown

| Repository | Task Count | Stratum C (Un-named) | Stratum B | Stratum A |
| :--- | :---: | :---: | :---: | :---: |
| `sympy/sympy` | 39 | 36 | 3 | 0 |
| `scikit-learn/scikit-learn` | 9 | 8 | 1 | 0 |
| `pydata/xarray` | 4 | 4 | 0 | 0 |
| `django/django` | 3 | 3 | 0 | 0 |
| `pylint-dev/pylint` | 1 | 1 | 0 | 0 |

## 3. Methodological Implication for Stratum C Population

- Previous Stratum C evaluated cohort: $N = 131$.
- Recovered Stratum C drift tasks: $N = 52$ (plus 4 in Stratum B).
- **Complete Stratum C Cohort Population**: $N = 183$ tasks.
- Restoring these tasks eliminates the 28.4% cohort attrition penalty flagged by peer review, providing an unconfounded, closed population evaluation.
