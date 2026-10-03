# Authoritative Evaluation Metrics Specification

This document formalizes the authoritative benchmark evaluation metrics implemented in `perron.metrics` and used across all empirical evaluations in Perron.

---

## 1. Core Mathematical Definitions

### 1.1 Function Accuracy at $K$ ($\text{Function Acc}@K$)
Given a ranked candidate list of retrieved symbols $R = (s_1, s_2, \dots, s_N)$ and a set of ground-truth modified gold function symbols $G$:

$$\text{Function Acc}@K = \begin{cases} 1.0 & \text{if } \exists\, s \in R[:K] \text{ such that } s \in G \\ 0.0 & \text{otherwise} \end{cases}$$

- A task instance is scored as a **Hit** ($1.0$) if **ANY** gold function symbol appears within the top-$K$ retrieved candidates.
- Evaluated at standard cutoffs: $K \in \{1, 5, 10, 25\}$.

### 1.2 File Accuracy at $K$ ($\text{File Acc}@K$)
Given a ranked list of candidate files $F_R = (f_1, f_2, \dots, f_M)$ and a set of gold modified files $F_G$:

$$\text{File Acc}@K = \begin{cases} 1.0 & \text{if } \exists\, f \in F_R[:K] \text{ such that } f \text{ matches any } g \in F_G \\ 0.0 & \text{otherwise} \end{cases}$$

- File matching is path-normalized across OS separators (`/` vs `\`) and matches exact paths or relative path segment suffixes.
- Evaluated at standard cutoffs: $K \in \{1, 3, 5\}$.

### 1.3 Context Coverage at 4,096 Tokens ($\text{Coverage}@4\text{k}$)
Given ranked retrieved symbols $R$, each with estimated token footprint $c(s)$, packed iteratively under a hard context window budget $B = 4,096$ tokens:

$$R_{\text{packed}} = \left(s_1, \dots, s_m \mid \sum_{i=1}^m c(s_i) \le B \text{ and } \sum_{i=1}^{m+1} c(s_i) > B\right)$$

$$\text{Coverage}@4\text{k} = \frac{|\{s \in R_{\text{packed}} \mid s \in G\}|}{|G|}$$

- Measures the fraction of unique gold functions whose AST breadcrumb definitions fit inside the model's single-turn context budget.
- If $G = \emptyset$, $\text{Coverage}@4\text{k} = 0.0$.

### 1.4 Hub Suppression Index at Top-10 ($\text{HSI}@10$)
Given the top-10 retrieved items $R[:10]$ and the set of top-25 global graph in-degree hubs $H_{25}$:

$$\text{HSI}@10 = 1.0 - \frac{|R[:10] \cap H_{25}|}{10.0}$$

- Evaluated in the interval $[0.0, 1.0]$.
- $\text{HSI}@10 = 1.0$ indicates zero hub congestion in the top 10 positions.
- $\text{HSI}@10 = 0.0$ indicates the top 10 positions are completely congested by global hubs.

---

## 2. Tie-Breaking Protocol

When two or more candidates produce identical score values from a retrieval algorithm:
1. **Stable Ordering**: Implementations must use stable sort algorithms (`kind="stable"` in NumPy or Python's Timsort).
2. **Deterministic Priority**: If scores are identical, initial symbol registration index (topological / declaration order in the graph catalog) is strictly preserved.
3. Random or hash-dependent ordering is forbidden.

---

## 3. Gold Function Extraction from Human Patches

Ground-truth modified AST symbols are extracted from human unified diff patches via `benchmarks.gold_patch_parser.GoldPatchParser`:

1. **Unified Diff Parsing**:
   - Each patch is parsed into file paths and hunk header tuples: `(file_path, old_start_line, old_line_count)`.
   - The original pre-image line interval is defined as:
     $$I_{\text{hunk}} = [\text{old\_start\_line},\; \text{old\_start\_line} + \max(0, \text{old\_line\_count} - 1)]$$
2. **AST Interval Intersection**:
   - For every symbol $s$ in the repository AST symbol table belonging to `file_path` with line span $[L_{\text{start}}(s), L_{\text{end}}(s)]$:
     $$I_{\text{hunk}} \cap [L_{\text{start}}(s), L_{\text{end}}(s)] \neq \emptyset$$
3. **Innermost Symbol Resolution**:
   - When multiple AST symbols overlap a diff hunk (e.g., an enclosing module, an outer class, and an inner member method):
     - Priority ordering prioritizes `function` and `method` symbols over `class` or `module` nodes.
     - Among symbols of the same category, the innermost symbol with the narrowest span ($L_{\text{end}} - L_{\text{start}}$) is uniquely selected.
   - Pure additions ($\text{old\_line\_count} = 0$) select the enclosing scope enclosing $\text{old\_start\_line}$.

---

## 4. Invariant Edge-Case Handling Rules

| Edge Case | Metric Behavior | Rationale |
| :--- | :--- | :--- |
| **Empty Gold ($G = \emptyset$)** | $\text{Acc} = 0.0$, $\text{Coverage} = 0.0$, $\text{HSI} = 1.0$ | Cannot achieve a hit when no defect exists to localize. |
| **Duplicate IDs in Retrieved** | Evaluated on slice; deduplicated for set hits. | Slicing preserves exact rank position without double counting hits. |
| **$K > \text{Candidate Pool Size}$** | Evaluates all available candidates without out-of-bounds error. | System handles small repositories gracefully. |
| **Gold Not in Graph Catalog** | $\text{Acc} = 0.0$, $\text{Coverage} = 0.0$ | Safely treated as an unretrieved miss rather than an exception. |
| **Zero Token Budget ($B \le 0$)** | $\text{Coverage} = 0.0$ | No context can be delivered. |
