# Publication-Grade GLMM & Clustered Regression Models: SWE-bench Verified

**Author**: Sultan Haikal (`@yvliet`)  
**Generated**: 2026-10-17T14:24:35.094338+00:00  
**Dataset**: SWE-bench Verified ($N=344$ tasks, 12 repository clusters, 8 experimental arms)  

## 1. Primary Arm Contrast Model (Acc@5, Cluster: Repository, Ref: A2 Depth-BM25)

| Parameter / Contrast | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | +0.9058 | 0.4237 | +2.138 | 0.0325 | 2.474 [1.078, 5.676] |
| `Arm: A1` | +0.0638 | 0.0741 | +0.860 | 0.3898 | 1.066 [0.922, 1.233] |
| `Arm: A3` | +0.0319 | 0.0358 | +0.890 | 0.3735 | 1.032 [0.962, 1.108] |
| `Arm: A4` | +0.1115 | 0.0556 | +2.006 | 0.0449 | 1.118 [1.003, 1.247] |
| `Arm: A5` | +0.0478 | 0.0546 | +0.876 | 0.3810 | 1.049 [0.943, 1.167] |
| `Arm: A6` | +0.0319 | 0.0654 | +0.488 | 0.6257 | 1.032 [0.908, 1.174] |
| `Arm: B0` | -0.0160 | 0.0969 | -0.165 | 0.8691 | 0.984 [0.814, 1.190] |
| `Arm: B1` | -0.2568 | 0.1411 | -1.820 | 0.0687 | 0.773 [0.587, 1.020] |
| `Stratum: B` | -1.9157 | 0.4674 | -4.099 | 0.0000 | 0.147 [0.059, 0.368] |
| `Stratum: C` | -2.3036 | 0.4149 | -5.552 | 0.0000 | 0.100 [0.044, 0.225] |

## 2. Direct Graph Contrasts (Acc@5, Cluster: Repository, Ref: A4 Spectral Graph)

| Parameter / Contrast | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | +1.0173 | 0.4450 | +2.286 | 0.0222 | 2.766 [1.156, 6.616] |
| `Arm: A2` | -0.1115 | 0.0556 | -2.006 | 0.0449 | 0.894 [0.802, 0.997] |
| `Arm: A3` | -0.0796 | 0.0614 | -1.297 | 0.1946 | 0.923 [0.819, 1.042] |
| `Arm: A5` | -0.0637 | 0.0484 | -1.315 | 0.1885 | 0.938 [0.853, 1.032] |
| `Arm: A1` | -0.0477 | 0.0692 | -0.690 | 0.4901 | 0.953 [0.833, 1.092] |
| `Arm: A6` | -0.0796 | 0.0594 | -1.341 | 0.1800 | 0.923 [0.822, 1.038] |
| `Arm: B0` | -0.1275 | 0.0933 | -1.366 | 0.1720 | 0.880 [0.733, 1.057] |
| `Arm: B1` | -0.3684 | 0.1608 | -2.291 | 0.0220 | 0.692 [0.505, 0.948] |
| `Stratum: B` | -1.9157 | 0.4674 | -4.099 | 0.0000 | 0.147 [0.059, 0.368] |
| `Stratum: C` | -2.3036 | 0.4149 | -5.552 | 0.0000 | 0.100 [0.044, 0.225] |

## 3. Binomial Bayesian Mixed GLM (Repository Random Intercepts)

| Parameter | Type | Posterior Mean | Posterior SD | SD (95% Credible Interval) |
| :--- | :---: | :---: | :---: | :---: |
| `Intercept` | M | +0.5064 | 0.0463 | — |
| `Arm: A1` | M | +0.0706 | 0.1304 | — |
| `Arm: A3` | M | +0.0362 | 0.1305 | — |
| `Arm: A4` | M | +0.1221 | 0.1303 | — |
| `Arm: A5` | M | +0.0534 | 0.1305 | — |
| `Arm: A6` | M | +0.0362 | 0.1305 | — |
| `Arm: B0` | M | -0.0156 | 0.1307 | — |
| `Arm: B1` | M | -0.2763 | 0.1317 | — |
| `Stratum: B` | M | -2.0623 | 0.3071 | — |
| `Stratum: C` | M | -2.5059 | 0.0700 | — |
| `repo` | V | -0.0382 | 0.2003 | 0.963 [0.645, 1.437] |

## 4. Causal 2x2 Factorial Interaction Model (Stratum C, N=524 obs)

### Acc@1 Interaction (Top-1 Hit)

| Factor / Term | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | -3.7534 | 0.6322 | -5.937 | 0.0000 | 0.023 [0.007, 0.081] |
| `provenance` | +0.5266 | 0.2987 | +1.763 | 0.0779 | 1.693 [0.943, 3.040] |
| `thinking` | +1.5477 | 0.4939 | +3.134 | 0.0017 | 4.701 [1.785, 12.376] |
| `provenance:thinking` | -0.6151 | 0.2254 | -2.728 | 0.0064 | 0.541 [0.347, 0.841] |

### Acc@5 Interaction (Top-5 Hit)

| Factor / Term | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | -1.7138 | 0.2311 | -7.417 | 0.0000 | 0.180 [0.115, 0.283] |
| `provenance` | +0.0578 | 0.0451 | +1.281 | 0.2001 | 1.060 [0.970, 1.158] |
| `thinking` | +0.2190 | 0.0848 | +2.582 | 0.0098 | 1.245 [1.054, 1.470] |
| `provenance:thinking` | -0.1097 | 0.0367 | -2.986 | 0.0028 | 0.896 [0.834, 0.963] |
