# Pre-Registration Specification: Causal Lexical Intervention & Multi-Encoding Graph Representations

- **Protocol Identifier**: `PREREG-CAUSAL-v1`
- **Author**: Sultan Haikal (GitHub: `@yvliet`)
- **Affiliation**: Independent Research / Gemma 4 Developer Agent Track 2 Submission
- **Timestamp**: October 12, 2026
- **Status**: Pre-Registered & Locked Prior to Experimental Execution
- **Target Venue**: Google DeepMind Developer Agent Competition Track 2 ($35,000 Prize Pool)
- **Primary Evaluator Alignment**: Graph-ML Peer Review Criteria (Bryan Perozzi, Benedek Rózemberczki, Mikhail Galkin)

---

## 1. Context & Theoretical Motivation

In Phase 1 and Phase 2, we established two foundational empirical findings:
1. **The Lexical Abstraction Bottleneck**: Compact language models exhibit a severe discrimination drop between lexical seed candidates (AUC $0.7966\text{--}0.8544$) and graph-expanded neighbors (AUC $0.6596$, Mann-Whitney $U = 2993, p = 0.0036^*$).
2. **The Break-Even Discrimination Bound**: When candidate pools expand from $M=3$ to $M=25$, the minimum discrimination required to sustain bug localization jumps from $0.50$ to $0.875$.
3. **Reasoning Compute Multiplier**: Allocating test-time `<thought>` tokens raises Top-1 accuracy on Stratum C from $2.29\%$ to $9.92\%$ (exact binomial McNemar $p = 0.0063^*$).

### The Dual Unresolved Critiques
- **Critique A (Correlational vs. Causal Lexical Mechanism)**: Is the collapse on graph neighbors driven causally by the absence of surface lexical overlap, or is it an artifact of structural complexity in non-seed code?
- **Critique B (Prompt Encoding Sensitivity / *Talk Like a Graph*)**: Does graph provenance fail without thinking compute merely because a comment-style annotation (`# Provenance: ...`) is sub-optimal? Would alternative graph serializations (Edge Lists, Adjacency Maps, Linearized Traversal Paths) allow the model to trace relationships zero-shot?

This protocol defines the formal experimental design, hypotheses, and decision rules to resolve both critiques via controlled intervention.

---

## 2. Experimental Suite Architecture

All experiments are executed on Google Gemma 4 E2B (`google/gemma-4-e2b-it`) using unpadded single-candidate inference with `torch.float16`, `logits_to_keep=1`, and `expandable_segments:True` on Kaggle Dual Tesla T4 GPUs across the 58 held-out Stratum C tasks from `data_release/stratum_c_eval_bundle.json`.

### Experiment A: Causal Lexical Intervention (Identifier Obfuscation)
We apply a syntax-preserving AST $\alpha$-renaming transformation (`perron.reranker.obfuscation.IdentifierObfuscator`) that replaces user-defined function, parameter, and variable names with neutral synthetic tokens (`fn_1`, `v_1`, `Class_1`) while keeping Python built-ins, control flow, and AST structure intact.

- **Condition $O_0$ (Natural Code / Baseline)**: Original code breadcrumbs with identifiers intact.
- **Condition $O_1$ (Identifier Obfuscated)**: All user identifiers $\alpha$-renamed; surface lexical tokens eliminated ($O_{\text{lex}} = 0$).
- **Condition $O_2$ (Obfuscated + Thinking Mode)**: $\alpha$-renamed breadcrumbs evaluated under greedy test-time reasoning compute ($\le 180$ `<thought>` tokens).

### Experiment B: Multi-Encoding Graph Representations (*Talk Like a Graph* Suite)
Following the graph prompt taxonomy of Fatemi, Halcrow, and Perozzi (ICLR 2024), we evaluate five distinct representations of repository dependency topology:

- **Condition $E_0$ (Isolated Code / Baseline)**: Code breadcrumbs with file and line range headers only.
- **Condition $E_1$ (Inline Comment Provenance / Current Perron)**: Comment header `# Provenance: Structural call graph neighbor...`.
- **Condition $E_2$ (Linearized Call-Path Traversal)**: Directed traversal trace `[Call Trace]: seed_entry -> intermediate_func -> candidate_target`.
- **Condition $E_3$ (Structural Subgraph Edge List)**: Explicit relational preamble listing directed multigraph edges `### Repository Call Graph Edges: (u, CALLS, v)...`.
- **Condition $E_4$ (Hierarchical Adjacency Map)**: Incident neighbor dictionary `### Symbol Dependency Adjacency: u -> [v, w]...`.

Each encoding $E_0 \dots E_4$ is evaluated under both standard readout and test-time reasoning compute (`<thought>`).

---

## 3. Pre-Registered Hypotheses & Decision Rules

### Hypothesis A.1: Causal Lexical Dependency ($H_{obf,1}$)
- **Prediction**: Stripping surface lexical tokens via AST $\alpha$-renaming ($O_1$) causes candidate discrimination on seed files to drop significantly toward the non-lexical ceiling ($\Delta \text{AUC} \le -0.12, p < 0.01$).
- **Falsification Condition**: If $O_1$ discrimination is statistically indistinguishable from $O_0$ ($\Delta \text{AUC} \le 0.02, p > 0.10$), $H_{obf,1}$ is falsified; compact LLMs do not rely on surface lexical cues, and the bottleneck is attributed to other structural factors.

### Hypothesis A.2: Reasoning Recovers Structural Invariant ($H_{obf,2}$)
- **Prediction**: Allocating test-time `<thought>` tokens on obfuscated code ($O_2$) recovers a statistically significant portion of Top-1 accuracy over un-reasoned obfuscated code ($O_1$) by enabling internal self-attention over AST syntax.

### Hypothesis B.1: Prompt Serialization Invariance ($H_{enc,1}$)
- **Prediction**: Without test-time reasoning compute, none of the structured graph serializations ($E_2, E_3, E_4$) significantly surpass the isolated code baseline ($E_0$) on Top-1 accuracy (two-sided McNemar exact $p > 0.10$). Prompt encoding alone cannot overcome the Lexical Abstraction Bottleneck.

### Hypothesis B.2: Traversal Trace Synergizes with Thinking Compute ($H_{enc,2}$)
- **Prediction**: Under test-time reasoning compute, linearized traversal traces ($E_2$) and edge lists ($E_3$) achieve superior ranking efficiency (MRR) compared to isolated code ($E_0$) by constraining the search space of the model's Chain-of-Thought reflections.

---

## 4. Statistical Verification Gates & Receipt Schema

All evaluations log:
- **Pointwise discrimination**: ROC-AUC, mean log-odds margin ($s(t, v_{\text{gold}}) - \bar{s}(t, v_{\text{distractor}})$).
- **Listwise ranking**: Acc@1, Acc@3, Acc@5, Acc@10, MRR@50.
- **Statistical significance**: Exact binomial McNemar test for paired binary outcomes; Wilcoxon signed-rank test for paired continuous rank shifts.
- **Telemetry**: Peak VRAM (MB), per-task inference latency (s), prompt token count, and reasoning token length.
- **Artifacts**: Dumped deterministically to `results/causal_ablation_results.json`.
