# Perron Empirical Claims Ledger

This ledger binds all empirical assertions made in `paper/paper_writeup.md` and `paper/main.tex` to authoritative serialized result files, following the empirical verification protocol.

---

## 1. Primary Empirical Claims Ledger

| Claim Identifier | Manuscript Assertion Text | Authoritative Evidence File | JSON Field / Line Anchor | Metric Value | Verification Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `CLAIM-01` | Perron achieves 14.4% Function Acc@10 on heldout callable set ($S_{\text{func}}$, $N=118$) | `paper/metrics_macros.tex`, `data/retrieval_sweep_results.json` | `\PerronPrimaryFnAccTen`, `subsets.heldout_s_func.perron_static.fn_acc_10` | 14.4% [8.5%, 21.2%] | **Supported** |
| `CLAIM-02` | Degree-Normalized PPR achieves 4.2% Fn Acc@10; exact McNemar $p = 0.0042$, Holm $p = 0.0251$ | `paper/metrics_macros.tex`, `data/retrieval_sweep_results.json` | `\DegNormPPRFnAccTen`, `statistical_tests.heldout_s_func_primary.perron_vs_deg_ppr` | 4.2% ($p = 0.0042$, $p_{\text{holm}} = 0.0251$) | **Supported** |
| `CLAIM-03` | Standard PPR, Aider Repo Maps, Hub Blocklists, and Query-Reweighted PPR collapse to 0.0% | `paper/metrics_macros.tex`, `data/retrieval_sweep_results.json` | `\StandardPPRFnAccTen`, `\AiderFnAccTen`, `\HubBlocklistFnAccTen` | 0.0% ($p < 0.001$) | **Supported** |
| `CLAIM-04` | BM25 achieves 22.9% Fn Acc@10 on callable set and 18.0% on full heldout ($N=150$) | `results/heldout_table2_v2_postaudit.json`, `results/heldout_fusion.json` | `summary.bm25.metrics.fn_acc_10.mean` | 18.00% (full), 22.9% (callable) | **Supported** |
| `CLAIM-05` | Lexical parity: best weighted fusion selects zero graph weight ($w=0.0$, pure BM25) at 18.0%; Reciprocal Rank Fusion reaches 16.7% | `results/heldout_fusion.json` | `selected_w`, `summary.weighted_w_0.0.metrics.fn_acc_10.mean`, `summary.rrf_60.metrics.fn_acc_10.mean` | 18.00% ($w=0.0$), 16.67% (RRF) | **Supported** |
| `CLAIM-06` | Stratum C (unnamed defect targets, $N=112$) yields 8.04% for Perron vs 8.93% for BM25 | `results/heldout_strata.json` | `strata.C.methods.perron_static.fn_acc_10.mean` | 8.04% vs 8.93% | **Supported** |
| `CLAIM-07` | Perron AST patcher achieves 76.7% valid apply with 0.0% syntax errors (23.3% rollback) | `data/editor_eval_results.json`, `paper/metrics_macros.tex` | `\PerronASTApplyRate`, `\PerronASTSyntaxErrorRate` | 76.7% apply, 0.0% syntax error | **Supported** |
| `CLAIM-08` | Raw sed introduces 46.7% syntax errors; git apply suffers 20.0% hunk rejections | `data/editor_eval_results.json`, `paper/metrics_macros.tex` | `\SedSyntaxErrorRate`, `\GitApplyHunkRejectRate` | 46.7% sed err, 20.0% git reject | **Supported** |
| `CLAIM-09` | Perron Engine runs at 144.33 MB peak RSS, preserving 6.86 GB available headroom on 16GB host | `results/memory_measured.json` | `process_memory.peak_process_rss_mb`, `hardware_host.available_system_ram_gb` | 144.33 MB RSS, 6.86 GB headroom | **Supported** |
| `CLAIM-10` | Full dataset spans 300 instances, 4,876,848 AST nodes, and 41,591,756 directed multigraph edges | `data_release/VALIDATION.md`, `facts.json` | `dataset.total_ast_nodes`, `dataset.total_multigraph_edges` | 4,876,848 nodes, 41,591,756 edges | **Supported** |
| `CLAIM-11` | Brauer rank-1 perturbation establishes invariant spectral gap $1 - \vert\lambda_2\vert \ge 0.15$ | `paper/main.tex`, `paper/paper_writeup.md` | Section 3.3, `\beta = 0.85` | $\vert\lambda_2\vert \le 0.85$, gap $\ge 0.15$ | **Supported** |
| `CLAIM-12` | Power iteration reaches $\epsilon = 10^{-6}$ in $\le \lceil \ln 10^{-6} / \ln 0.85 \rceil = 86$ iterations (71 on Requests, 68 on SymPy) | `facts.json` | `spectral_convergence.max_iterations_eps_1e_6`, `spectral_convergence.requests_power_iterations`, `spectral_convergence.sympy_power_iterations` | 86 bound; 71 iters (2.7 ms), 68 iters (6.6 ms) | **Supported** |

---

## 2. Empirical Scope & Methodological Invariants

- **Lexical Baseline Invariant**: Pure lexical BM25 (18.0%) outperforms or matches graph diffusion on full SWE-bench Lite ($N=150$). Perron is framed honestly as a high-throughput CSR graph diffusion, hub suppression, and context-packing substrate.
- **Host Execution Boundary Invariant**: Zero SWE repair claims without verified execution (`tests_pass: null` preserved). Engine memory reported strictly from authentic host telemetry (`results/memory_measured.json`).
