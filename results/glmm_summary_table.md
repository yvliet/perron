# Publication-Grade GLMM & Clustered Regression Models: SWE-bench Verified

**Author**: Sultan Haikal (`@yvliet`)  
**Generated**: 2026-10-18T01:56:07.586711+00:00  
**Dataset**: SWE-bench Verified ($N=344$ tasks, 12 repository clusters, 8 experimental arms)  

## 1. Primary Arm Contrast Model (Acc@5, Cluster: Repository, Ref: A2 Depth-BM25)

| Parameter / Contrast | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | +0.9058 | 0.1612 | +5.619 | 0.0000 | 2.474 [1.804, 3.393] |
| `Arm: A1` | +0.0638 | 0.0638 | +1.000 | 0.3172 | 1.066 [0.941, 1.208] |
| `Arm: A3` | +0.0319 | 0.0553 | +0.577 | 0.5639 | 1.032 [0.926, 1.151] |
| `Arm: A4` | +0.1115 | 0.0653 | +1.708 | 0.0876 | 1.118 [0.984, 1.270] |
| `Arm: A5` | +0.0478 | 0.0479 | +0.999 | 0.3176 | 1.049 [0.955, 1.152] |
| `Arm: A6` | +0.0319 | 0.0553 | +0.576 | 0.5643 | 1.032 [0.926, 1.151] |
| `Arm: B0` | -0.0160 | 0.0920 | -0.174 | 0.8622 | 0.984 [0.822, 1.179] |
| `Arm: B1` | -0.2568 | 0.1187 | -2.165 | 0.0304 | 0.773 [0.613, 0.976] |
| `Stratum: B` | -1.9157 | 0.8372 | -2.288 | 0.0221 | 0.147 [0.029, 0.760] |
| `Stratum: C` | -2.3036 | 0.2366 | -9.735 | 0.0000 | 0.100 [0.063, 0.159] |

## 2. Direct Graph Contrasts (Acc@5, Cluster: Repository, Ref: A4 Spectral Graph)

| Parameter / Contrast | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | +1.0173 | 0.1622 | +6.271 | 0.0000 | 2.766 [2.013, 3.801] |
| `Arm: A2` | -0.1115 | 0.0653 | -1.708 | 0.0876 | 0.894 [0.787, 1.017] |
| `Arm: A3` | -0.0796 | 0.0617 | -1.290 | 0.1971 | 0.923 [0.818, 1.042] |
| `Arm: A5` | -0.0637 | 0.0502 | -1.269 | 0.2045 | 0.938 [0.850, 1.035] |
| `Arm: A1` | -0.0477 | 0.0528 | -0.904 | 0.3660 | 0.953 [0.860, 1.057] |
| `Arm: A6` | -0.0796 | 0.0476 | -1.674 | 0.0942 | 0.923 [0.841, 1.014] |
| `Arm: B0` | -0.1275 | 0.0842 | -1.514 | 0.1300 | 0.880 [0.746, 1.038] |
| `Arm: B1` | -0.3684 | 0.1142 | -3.225 | 0.0013 | 0.692 [0.553, 0.865] |
| `Stratum: B` | -1.9157 | 0.8372 | -2.288 | 0.0221 | 0.147 [0.029, 0.760] |
| `Stratum: C` | -2.3036 | 0.2366 | -9.735 | 0.0000 | 0.100 [0.063, 0.159] |

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
| `Intercept` | -3.7534 | 0.5880 | -6.383 | 0.0000 | 0.023 [0.007, 0.074] |
| `provenance` | +0.5266 | 0.3763 | +1.399 | 0.1617 | 1.693 [0.810, 3.540] |
| `thinking` | +1.5477 | 0.5760 | +2.687 | 0.0072 | 4.701 [1.520, 14.536] |
| `provenance:thinking` | -0.6151 | 0.3570 | -1.723 | 0.0849 | 0.541 [0.269, 1.088] |

### Acc@5 Interaction (Top-5 Hit)

| Factor / Term | Log-Odds ($\beta$) | Cluster-Robust SE | Wald $z$ | $p$-value | Odds Ratio (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Intercept` | -1.7138 | 0.2446 | -7.008 | 0.0000 | 0.180 [0.112, 0.291] |
| `provenance` | +0.0578 | 0.0580 | +0.997 | 0.3188 | 1.060 [0.946, 1.187] |
| `thinking` | +0.2190 | 0.1339 | +1.636 | 0.1019 | 1.245 [0.958, 1.619] |
| `provenance:thinking` | -0.1097 | 0.0776 | -1.413 | 0.1575 | 0.896 [0.770, 1.043] |
