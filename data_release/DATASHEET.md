# Datasheet for the Perron SWE-bench Lite Code Graph Dataset

Following the **Datasheets for Datasets** framework proposed by Gebru et al. (2021).

---

## 1. Motivation

- **For what purpose was the dataset created?**  
  Modern autonomous developer agents suffer from context window saturation and reasoning degradation when navigating large enterprise repositories. Static repository call graphs follow scale-free degree distributions where central dispatchers, loggers, and base classes ("hubs") attract massive incoming connectivity. Naive random walk or dense retrieval algorithms either collapse into stationary hubs ($HSI \le 60\%$) or adopt fragile blocklists that fail on true hub defects. This dataset provides a standardized, pre-indexed topological substrate of Compressed Sparse Row (CSR) multigraphs for benchmarking and deploying high-throughput spectral diffusion models without runtime AST compilation overhead.

- **Who created the dataset and on behalf of which entity?**  
  Created by **Sultan Haikal** (`@yvliet`) as part of the Perron research project and the Google Gemma 4 Developer Agent research initiative.

- **Who funded the creation of the dataset?**  
  Independent research project.

---

## 2. Composition

- **What do the instances that comprise the dataset represent?**  
  The dataset comprises **300 task instances** corresponding to the canonical SWE-bench Lite benchmark across 12 mature Python repositories (`astropy`, `django`, `matplotlib`, `seaborn`, `flask`, `requests`, `xarray`, `pylint`, `pytest`, `scikit-learn`, `sphinx`, `sympy`).

- **How many instances are there in total?**  
  300 instances partitioned into three disjoint splits:
  - `dev_pilot` ($N = 50$): exploratory calibration subset.
  - `dev_val` ($N = 100$): hyperparameter tuning and model validation.
  - `heldout` ($N = 150$): blind evaluation split.

- **What data does each instance contain?**  
  Each instance directory contains:
  1. `csr_data.npy`: Float32 non-zero edge weights of the row-stochastic transition matrix $T$.
  2. `csr_indices.npy`: Int32 target column indices.
  3. `csr_indptr.npy`: Int32 row pointers (length $N + 1$).
  4. `csr_dangling.npy`: Float32 indicator vector for zero-outdegree sink nodes ($d_u = 1.0$).
  5. `nodes.parquet`: Comprehensive symbol table containing node identifiers, qualified names, source file paths, line intervals, in/out degrees, test indicators, and hub flags (`is_hub_top25`, `is_hub_p99`).
  6. `manifest.json`: Instance metadata and SHA-256 integrity digests for all arrays.
  
  In aggregate, the dataset spans **4,876,848 AST nodes** and **41,591,756 directed multigraph edges**.

- **Is any confidential or personally identifiable information (PII) present?**  
  No. All code and AST definitions are derived strictly from public open-source software repositories.

---

## 3. Collection Process

- **How was the data collected?**  
  Each repository was checked out at the exact immutable `base_commit` specified in SWE-bench Lite. Codebases were parsed into Abstract Syntax Trees using Python's standard `ast` module. Multigraph edges capture four semantic relations:
  - Direct function and method calls ($\omega_{\text{call}} = 0.50$)
  - Class inheritance and MRO relations ($\omega_{\text{inherit}} = 0.25$)
  - Module import dependencies ($\omega_{\text{import}} = 0.15$)
  - Inverted caller references ($\omega_{\text{caller}} = 0.10$)

- **Who was involved in data collection and what were their roles?**  
  Automated static analysis tooling developed by Sultan Haikal.

---

## 4. Preprocessing and Cleaning

- **What preprocessing or cleaning steps were applied?**  
  1. **Row-Stochastic Normalization**: Edge weights out of each node $u$ are normalized such that $\sum_v T_{uv} = 1.0$ for all non-sink nodes.
  2. **Dangling Sink Handling**: Nodes with zero out-degree are assigned $d_u = 1.0$ and reconnected via uniform teleportation during random-walk diffusion.
  3. **Deterministic Symbol Hashing**: Symbol identifiers and node IDs are contiguous integers $[0, N-1]$.

---

## 5. Uses & Limitations

- **What are the intended uses of the dataset?**  
  - Benchmarking code retrieval algorithms (sparse, dense, graph diffusion) on software engineering tasks.
  - Developing and evaluating local developer agents under strict memory and token budgets ($K \le 4,096$).
  - Analyzing hub suppression and scale-free graph navigation.

- **What are non-intended uses?**  
  - Generative code completion without AST validation.
  - Code attribution or author de-anonymization.

---

## 6. Distribution & Licensing

- **How is the dataset distributed?**  
  - Public repository: `data_release/` directory in Perron.
  - Kaggle Datasets: `yvliet/perron-swebench-graphs`.
  - Specification: MLCommons Croissant 1.0 (`croissant.json`).

- **What is the license of the dataset?**  
  - The derived multigraph topology, transition matrices, symbol metadata, and gold target labels are released under the **Apache-2.0 License**.
  - **Clean-Room License Compliance**: Raw source code bodies are excluded from the distribution bundle to prevent copyleft aggregation issues with upstream GPL components (such as Pylint). On-demand source text reconstruction is supported via `scripts/reconstruct_dataset.py`.

---

## 7. Maintenance

- **Who supports and maintains the dataset?**  
  Sultan Haikal (GitHub: `@yvliet`, `yvliet@users.noreply.github.com`).
- **How can errors or errata be reported?**  
  Via GitHub Issues at `https://github.com/yvliet/perron`.
