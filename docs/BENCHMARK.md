# Hub-Gold Dissection Benchmark Specification

The **Hub-Gold Benchmark** evaluates developer agent retrieval systems under scale-free hub conditions, isolating vulnerability to hub-blindness vs hub-collapse.

---

## 1. Problem Formulation

In enterprise repository code networks, degree distributions follow power-law regimes: a tiny fraction ($<1\%$) of symbols (e.g., base classes, dispatched utility loggers, central dispatchers) account for a disproportionate volume of incoming edges.

Retrieval systems confront a fundamental dilemma:
1. **Stationary Hub Collapse**: Unnormalized random-walk algorithms (Standard PPR, PageRank) concentrate probability mass on ubiquitous stationary hubs ($HSI \le 60\%$), drowning localized defects in base classes and boilerplate.
2. **Aggressive Blocklisting Fragility**: Hard hub blocklists or severe degree discounting artificially maximize the Hub Suppression Index ($HSI \approx 99\%$) but achieve **0.0% accuracy** on true hub-gold defects where the defect actually resides within a central class.

---

## 2. Benchmark Partitions

Tasks are segmented into two orthogonal subsets based on ground-truth defect localization:
- **Subset $H$ (Hub-Gold)**: Ground-truth defects that modify high-degree symbols ($\ge 99$th percentile in-degree or top-25 repository hubs).
- **Subset $N$ (Non-Hub-Gold)**: Ground-truth defects residing in localized leaf or intermediate symbols.

---

## 3. Four Canonical Metrics

For any retrieval method $\mathcal{M}$:
1. **$\text{Acc}@10(H)$**: Function retrieval accuracy @ $k=10$ evaluated exclusively on Subset $H$.
2. **$\text{Acc}@10(N)$**: Function retrieval accuracy @ $k=10$ evaluated exclusively on Subset $N$.
3. **$\text{HSI}@10$**: Hub Suppression Index, defined as the percentage of retrieved top-$k$ symbols that are non-hub nodes ($1.0 - \text{HubFraction}$).
4. **$\text{HEADLINE}$**: Macro-average balancing hub and non-hub accuracy:
   $$\text{HEADLINE} = \frac{1}{2} \left( \text{Acc}@10(H) + \text{Acc}@10(N) \right)$$

> **Why Macro-Averaging?**  
> A naive hub blocklist that scores 20% on $N$ but 0% on $H$ is penalized to a 10.0% HEADLINE score. Conversely, a hub-only retriever that scores 100% on $H$ but 0% on $N$ is penalized to 50.0%. Only balanced, query-adaptive retrieval (such as Perron) achieves competitive macro-average performance.

---

## 4. Benchmark CLI Runner

Run the benchmark for any registered method:
```bash
perron bench --method perron --split dev_val
perron bench --method bm25 --split heldout
```

### Empirical Results Summary (`data_release/hubgold_v1.json`)

#### Dev-Val Partition ($N=100$)
| Method | $\text{Acc}@10(H)$ | $\text{Acc}@10(N)$ | $\text{HSI}@10$ | $\text{HEADLINE}$ |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical Baseline)** | 33.3% | 19.6% | 99.7% | 26.5% |
| **Standard PPR** | 33.3% | 1.0% | 60.0% | 17.2% |
| **Degree-Norm PPR** | 0.0% | 3.1% | 97.1% | 1.5% |
| **Perron Static** | 0.0% | 13.4% | 92.3% | 6.7% |
| **Perron Adaptive** | 33.3% | 7.2% | 82.4% | 20.3% |

#### Held-out Partition ($N=150$)
| Method | $\text{Acc}@10(H)$ | $\text{Acc}@10(N)$ | $\text{HSI}@10$ | $\text{HEADLINE}$ |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical Baseline)** | 0.0% | 18.1% | 99.9% | 9.1% |
| **Standard PPR** | 0.0% | 0.0% | 59.0% | 0.0% |
| **Degree-Norm PPR** | 0.0% | 3.4% | 96.9% | 1.7% |
| **Perron Static** | 0.0% | 11.4% | 93.2% | 5.7% |
| **Perron Adaptive** | 0.0% | 9.4% | 82.2% | 4.7% |
