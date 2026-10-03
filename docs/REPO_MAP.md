# Perron Repository Map & System Inventory

## 1. Core Library Modules (`perron/`)

| Path | Purpose | Tests |
| :--- | :--- | :--- |
| `perron/__init__.py` | Package initialization exporting core public APIs (`PerronGraph`, `PersonalizedPageRank`, `calculate_specificity_scores`, etc.). | `tests/test_graph.py`, `tests/test_mcp_server.py`, `tests/test_perron.py` |
| `perron/agent.py` | Autonomous developer agent orchestration loop and Gemma 4 model lifecycle manager handling multi-turn tool interaction. | `tests/test_agent.py`, `tests/test_agent_history.py`, `tests/test_agent_runtime.py`, `tests/test_integration.py`, `tests/test_patch.py` |
| `perron/backends/__init__.py` | Inference backend registration, dispatch, and factory abstractions. | `tests/test_graph.py`, `tests/test_mcp_server.py` |
| `perron/backends/base.py` | Abstract base class `ModelBackend` and dataclasses (`ChatMessage`, `GenerationConfig`) for model inference. | `tests/test_agent.py`, `tests/test_agent_runtime.py` |
| `perron/backends/gguf_backend.py` | Lightweight GGUF / llama.cpp memory-mapped inference backend for local quantized Gemma 4 execution. | None (requires optional `llama-cpp-python` runtime) |
| `perron/backends/replay.py` | Deterministic offline replay backend `OfflineReplayBackend` for recorded evaluations and mock test runs. | `tests/test_agent.py`, `tests/test_agent_runtime.py`, `tests/test_patch.py` |
| `perron/backends/transformers_backend.py` | PyTorch and Hugging Face Transformers local inference backend for Gemma 4. | None (requires live GPU/PyTorch environment) |
| `perron/backends/vllm_backend.py` | High-throughput vLLM and OpenAI-compatible endpoint inference backend for Gemma 4. | None (requires active vLLM HTTP server) |
| `perron/cli.py` | Command-line interface (`perron`) for graph indexing, search, AST slicing, and context packing. | `tests/test_packaging_and_cli.py` |
| `perron/diffusion.py` | High-throughput Personalized PageRank power-iteration solver over row-stochastic CSR sparse matrices. | `tests/test_adversarial_fuzzer.py`, `tests/test_boundary_invariants.py`, `tests/test_perron.py`, `tests/test_polyglot.py` |
| `perron/editor.py` | AST-grounded search-and-replace code editor with fuzzy indentation and scope matching. | `tests/test_adversarial_fuzzer.py`, `tests/test_boundary_invariants.py`, `tests/test_patch.py`, `tests/test_perron.py` |
| `perron/graph.py` | AST parser, symbol table extractor, and zero-copy CSR graph builder for Python repositories. | `tests/test_agent.py`, `tests/test_graph.py`, `tests/test_retriever.py`, `tests/test_perron.py` |
| `perron/matrix.py` | Sparse matrix compression, row-normalization, dangling-node compensation, and memory mapping. | `tests/test_adversarial_fuzzer.py`, `tests/test_boundary_invariants.py`, `tests/test_perron.py`, `tests/test_polyglot.py` |
| `perron/mcp_server.py` | Model Context Protocol (MCP) stdio server exposing graph search, AST slicing, and context packing tools. | `tests/test_mcp_server.py`, `tests/test_packaging_and_cli.py` |
| `perron/packer.py` | Context-budgeted knapsack subgraph packing engine assembling ranked definitions within strict token limits. | `tests/test_adversarial_fuzzer.py`, `tests/test_boundary_invariants.py`, `tests/test_perron.py` |
| `perron/patch.py` | Unified diff generation, hunk offset handling, and patch calculation for SWE-bench evaluation. | `tests/test_integration.py`, `tests/test_patch.py` |
| `perron/polyglot/__init__.py` | Polyglot multi-language AST extraction interfaces and registry. | `tests/test_graph.py`, `tests/test_mcp_server.py` |
| `perron/polyglot/treesitter.py` | Tree-sitter AST parser and symbol table extractor for TypeScript and JavaScript repositories. | `tests/test_polyglot.py` |
| `perron/retriever.py` | Hybrid lexical (BM25) and spectral graph retrieval engine generating teleportation priors. | `tests/test_retriever.py` |
| `perron/runner.py` | Headless autonomous agent runner and execution driver for benchmarking workflows. | `tests/test_agent_runtime.py`, `tests/test_integration.py` |
| `perron/specificity.py` | Decoupled specificity ratio calculation engine with adaptive query-directed hub attenuation. | `tests/test_adversarial_fuzzer.py`, `tests/test_baselines_and_ablation.py`, `tests/test_boundary_invariants.py`, `tests/test_perron.py`, `tests/test_polyglot.py` |
| `perron/telemetry/exporter.py` | Agent trajectory recorder and dataset compiler for fine-tuning serialization (SFT/DPO). | `tests/test_integration.py` |
| `perron/tester.py` | Targeted test locator and runner executing pytest suites within edited worktrees. | `tests/test_adversarial_fuzzer.py`, `tests/test_boundary_invariants.py`, `tests/test_integration.py`, `tests/test_perron.py` |
| `perron/worktree.py` | Ephemeral git worktree context manager ensuring isolated execution during agent code edits. | `tests/test_agent_runtime.py`, `tests/test_integration.py` |

---

## 2. Benchmarks & Evaluation Harnesses (`benchmarks/`)

| Path | Purpose | Tests |
| :--- | :--- | :--- |
| `benchmarks/baseline_retrievers.py` | Complete suite of 14 retrieval baselines and graph algorithms on CSR matrices. | `tests/test_baselines_and_ablation.py` |
| `benchmarks/compile_real_graphs.py` | Offline pipeline compiling Requests and SymPy repositories into binary zero-copy mmap CSR matrices. | None (offline data compilation) |
| `benchmarks/compute_pareto_frontier.py` | Empirical Pareto frontier and Optuna parameter sensitivity surface generator. | None (analytical script) |
| `benchmarks/diagnostic.py` | Benchmark diagnostic suite testing CSR validity, Perron diffusion, and spectral convergence. | None (standalone diagnostic) |
| `benchmarks/eval_harness.py` | Task-grounded benchmark evaluation harness computing Hit@k, MRR, Precision, and Recall against gold patches. | `tests/test_baselines_and_ablation.py`, `tests/test_evaluation_invariants.py` |
| `benchmarks/evaluate_editor_interfaces.py` | Empirical evaluation comparing search-and-replace, unified diff, and full rewrite editing interfaces. | None (benchmark script) |
| `benchmarks/extract_real_graph.py` | CLI extractor transforming local repository checkouts into Perron AST graphs. | None (CLI tool) |
| `benchmarks/generate_rigorous_artifacts.py` | Publication-grade statistical analysis and visualization generator for empirical artifacts. | None (visualization script) |
| `benchmarks/gold_patch_parser.py` | Canonical parser converting SWE-bench gold diffs into ground-truth modified AST node IDs. | `tests/test_evaluation_invariants.py` |
| `benchmarks/hub_gold_ablation.py` | Ablation analysis script isolating hub-gold nodes from non-hub-gold nodes across retrieval baselines. | None (ablation script) |
| `benchmarks/kaggle_gemma_runner.py` | Automated Kaggle GPU batch runner evaluating Gemma 4 on SWE-bench Lite instances. | None (remote runner) |
| `benchmarks/profile_csr_scaling.py` | Synthetic 1M-node CSR scaling profiler measuring multi-worker shared memory virtualization. | None (profiling benchmark) |
| `benchmarks/profile_hardware_telemetry.py` | Hardware telemetry profiler capturing CPU, RAM, GPU VRAM, and per-step latency. | None (profiling script) |
| `benchmarks/repair_eval.py` | Deterministic AST scaffold and agent repair evaluation harness. | None (evaluation harness) |
| `benchmarks/rigorous_significance_test.py` | Rigorous statistical significance testing engine running bootstrap and permutation tests. | None (statistical tool) |
| `benchmarks/run_real_evaluation.py` | End-to-end evaluation runner executing the Perron agent across real SWE-bench instances. | `tests/test_agent_runtime.py` |
| `benchmarks/run_retrieval_sweep.py` | Full 300-instance batch retrieval sweep running 14 baselines on SWE-bench Lite. | None (evaluation script) |
| `benchmarks/run_swebench_lite.py` | Automated batch evaluation runner orchestrating SWE-bench Lite problem solving. | None (batch runner) |
| `benchmarks/stat_engine.py` | Statistical analysis engine calculating bootstrap confidence intervals, Cliff's delta, and p-values. | None (library utility) |
| `benchmarks/swebench_loader.py` | SWE-bench Lite dataset streaming adapter and local JSONL cache loader. | `tests/test_integration.py` |
| `benchmarks/telemetry.py` | Telemetry collection and benchmark ingestion utility tracking runtime performance. | None (telemetry tool) |
| `benchmarks/tune_hyperparameters.py` | Multi-objective Bayesian hyperparameter optimization searching optimal gamma and beta values. | None (optimization script) |
| `benchmarks/viz_engine.py` | Scientific visualization engine rendering publication-grade PDF and PNG plots. | None (plotting engine) |

---

## 3. Automation & Publication Scripts (`scripts/`)

| Path | Purpose | Tests |
| :--- | :--- | :--- |
| `scripts/build_paper.py` | Master publication pipeline compiling LaTeX, generating figures, and synchronizing macros. | None (build orchestrator) |
| `scripts/compile.py` | Multi-pass pdflatex and bibtex compilation harness with automatic rerun resolution. | None (compilation tool) |
| `scripts/export_swebench_graphs.py` | Pre-indexed SWE-bench Lite call graph export pipeline serializing CSR matrices. | None (export utility) |
| `scripts/generate_figures.py` | Canonical figure generator rendering all publication plots from verified empirical data. | None (figure script) |
| `scripts/generate_kaggle_notebook.py` | Generator compiling the Kaggle Gemma 4 workstation notebook and kernel metadata. | None (generator script) |
| `scripts/generate_notebook.py` | Showcase notebook generator building `perron_showcase.ipynb`. | None (generator script) |
| `scripts/generate_paper_macros.py` | Script injecting empirical results from JSON into `paper/metrics_macros.tex`. | None (macro generator) |
| `scripts/publish_pypi.py` | Distribution packaging, twine verification, and PyPI release automation script. | None (release script) |
| `scripts/reconstruct_dataset.py` | Dataset hydration tool validating and reconstructing `swebench-lite-codegraphs`. | `tests/test_dataset_reconstruction.py` |
| `scripts/run_local_gemma4.py` | Live local Gemma 4 model inference runner testing transformers and GGUF runtimes. | None (interactive runner) |
| `scripts/sanitize_metadata.py` | Sanitizer ensuring graph metadata paths are strictly relative to the repository root. | None (sanitizer script) |
| `scripts/verify_paper.py` | Comprehensive paper verification tool checking word counts, macro consistency, and assets. | None (verification script) |

---

## 4. Test Suite (`tests/`)

| Path | Purpose | Target Modules Tested |
| :--- | :--- | :--- |
| `tests/test_adversarial_fuzzer.py` | Maximum-entropy adversarial fuzzing suite testing edge-case graphs and inputs. | `perron.diffusion`, `perron.editor`, `perron.matrix`, `perron.packer`, `perron.specificity`, `perron.tester` |
| `tests/test_agent.py` | Unit tests for agent lifecycle, state transitions, tool execution, and prompt assembly. | `perron.agent`, `perron.backends.base`, `perron.backends.replay`, `perron.graph` |
| `tests/test_agent_history.py` | Unit tests for history compaction, context budgeting, and XML thinking tag preservation. | `perron.agent` |
| `tests/test_agent_runtime.py` | Runtime tests for inference backends, prefix-cache stability, and agent loops. | `perron.agent`, `perron.backends.base`, `perron.backends.replay`, `benchmarks.run_real_evaluation` |
| `tests/test_baselines_and_ablation.py` | Unit tests verifying numerical accuracy, ranking order, and invariants of retrieval baselines. | `benchmarks.baseline_retrievers`, `benchmarks.eval_harness`, `perron.specificity` |
| `tests/test_boundary_invariants.py` | Boundary invariance battery testing mathematical properties under extreme limits. | `perron.diffusion`, `perron.editor`, `perron.matrix`, `perron.packer`, `perron.specificity`, `perron.tester` |
| `tests/test_dataset_reconstruction.py` | Verification tests for dataset hydration, checksum verification, and license compliance. | `scripts.reconstruct_dataset` |
| `tests/test_evaluation_invariants.py` | Invariant tests for gold patch parsing, Hit@k calculation, and metrics consistency. | `benchmarks.eval_harness`, `benchmarks.gold_patch_parser` |
| `tests/test_graph.py` | Tests for AST parsing, symbol table generation, and zero-copy CSR compilation. | `perron.graph` |
| `tests/test_integration.py` | End-to-end integration tests verifying agent execution, patching, telemetry, and evaluation. | `benchmarks.swebench_loader`, `perron.agent`, `perron.patch`, `perron.runner`, `perron.telemetry.exporter`, `perron.tester` |
| `tests/test_mcp_server.py` | Unit and integration tests for Model Context Protocol server tools, prompts, and resources. | `perron.mcp_server` |
| `tests/test_packaging_and_cli.py` | Packaging manifest integrity, version synchronization, and CLI entrypoint tests. | `perron.cli`, packaging manifests |
| `tests/test_patch.py` | Unit tests for unified diff calculation, hunk application, and file creation. | `perron.agent`, `perron.backends.replay`, `perron.editor`, `perron.patch` |
| `tests/test_perron.py` | Comprehensive test suite for diffusion, specificity, matrix operations, packing, and editing. | `perron.diffusion`, `perron.editor`, `perron.matrix`, `perron.packer`, `perron.specificity`, `perron.tester` |
| `tests/test_polyglot.py` | Unit and integration tests for Tree-sitter TypeScript and JavaScript AST extraction. | `perron.polyglot.treesitter`, `perron.diffusion`, `perron.matrix`, `perron.specificity` |
| `tests/test_retriever.py` | Unit tests for BM25 indexing, query tokenization, and prior vector computation. | `perron.graph`, `perron.retriever` |

---

## 5. All 14 Baseline Implementations

All 14 retrieval baselines are implemented in `benchmarks/baseline_retrievers.py` inside the class `RepositoryRetrievalEngine`, with a unified dispatcher `RepositoryRetrievalEngine.retrieve_by_name`:

| # | Baseline Name | File Path | Function Name |
| :-: | :--- | :--- | :--- |
| 1 | Okapi BM25 | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_bm25` |
| 2 | BM25 + 1-Hop Expansion | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_bm25_1hop` |
| 3 | Dense Semantic Retrieval | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_dense` |
| 4 | Prior-Only Control ($p_0$ without diffusion) | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_prior_only` |
| 5 | Standard Personalized PageRank (PPR) | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_standard_ppr` |
| 6 | Authentic Aider Repo Map | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_aider_repomap` |
| 7 | Hub Blocklist with Lexical Override | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_hub_blocklist_lexical` |
| 8 | Static Degree-Discounted Transition Matrix | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_static_deg_discount` |
| 9 | Degree-Normalized PPR | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_degree_normalized_ppr` |
| 10 | HippoRAG Pre-Walk Prior Scaling | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_hipporag` |
| 11 | Query-Reweighted PPR | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_query_reweighted_ppr` |
| 12 | Perron Static Specificity ($\gamma = 0.70$) | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_perron_static` |
| 13 | Perron Adaptive Specificity ($\gamma_q$ with prominence) | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_perron` |
| 14 | Oracle Upper Bound | `benchmarks/baseline_retrievers.py` | `RepositoryRetrievalEngine.retrieve_oracle` |

*Unified Dispatcher*: `benchmarks/baseline_retrievers.py` -> `RepositoryRetrievalEngine.retrieve_by_name(method_name, query_text, p_0, pi_query, gold_symbol_ids)`.

---

## 6. Key Functional Component Directory

| Component | Authoritative File Paths | Authoritative Functions / Classes |
| :--- | :--- | :--- |
| **Graph Building** | `perron/graph.py`<br>`perron/polyglot/treesitter.py`<br>`benchmarks/extract_real_graph.py` | `PerronGraph`, `build_graph_from_repo`<br>`TreeSitterExtractor`<br>`extract_repository_graph` |
| **Retriever** | `perron/retriever.py`<br>`benchmarks/baseline_retrievers.py`<br>`perron/diffusion.py`<br>`perron/specificity.py` | `HybridRetriever`<br>`RepositoryRetrievalEngine`<br>`personalized_pagerank_power_iteration`<br>`calculate_specificity_scores` |
| **Packer** | `perron/packer.py` | `ContextPacker`, `pack_subgraph_context` |
| **Editor** | `perron/editor.py`<br>`perron/patch.py` | `ASTRangeEditor`, `apply_ast_replacement`<br>`create_patch`, `apply_patch` |
| **Backends** | `perron/backends/base.py`<br>`perron/backends/replay.py`<br>`perron/backends/transformers_backend.py`<br>`perron/backends/vllm_backend.py`<br>`perron/backends/gguf_backend.py` | `ModelBackend`, `ChatMessage`<br>`OfflineReplayBackend`<br>`TransformersBackend`<br>`VLLMBackend`<br>`GGUFBackend` |
| **MCP Server** | `perron/mcp_server.py` | `PerronMCPServer`, `serve_stdio` |
| **Polyglot** | `perron/polyglot/treesitter.py`<br>`perron/polyglot/__init__.py` | `TreeSitterExtractor`, `extract_polyglot_symbols` |
| **Dataset Scripts** | `benchmarks/swebench_loader.py`<br>`benchmarks/gold_patch_parser.py`<br>`scripts/reconstruct_dataset.py`<br>`scripts/export_swebench_graphs.py` | `SWEBenchLoader`<br>`GoldPatchParser`<br>`reconstruct_dataset`<br>`export_graphs` |
| **Split File** | `data/swebench_lite_split.json` | Tri-partition: `dev_pilot` (50), `dev_val` (100), `heldout` (150). Total 300 instances. Disjoint partitions. |
| **Evaluation Scripts** | `benchmarks/run_retrieval_sweep.py`<br>`benchmarks/run_swebench_lite.py`<br>`benchmarks/run_real_evaluation.py`<br>`benchmarks/eval_harness.py`<br>`benchmarks/evaluate_editor_interfaces.py`<br>`benchmarks/hub_gold_ablation.py`<br>`benchmarks/repair_eval.py`<br>`benchmarks/diagnostic.py`<br>`benchmarks/rigorous_significance_test.py`<br>`benchmarks/compute_pareto_frontier.py` | Full empirical benchmark evaluation suite for retrieval, code editing, and end-to-end task completion. |

---

## 7. Plan Path Cross-References & Discrepancies

| Plan Reference | Repository Real Path | Status / Plan Usage |
| :--- | :--- | :--- |
| `data/swebench_lite_split.json` | `data/swebench_lite_split.json` | Exists and verified. Partition sizes: 50 pilot / 100 dev_val / 150 heldout. SHA256: `18a947a0e8a7941231ab15f11c1eee335adc58f8d03cbce742637452c9a23418`. |
| `Makefile` | `Makefile` | Defined in task T0.2; runs `pytest -q` and `python scripts/check_facts.py`. |
| `tests/test_split_frozen.py` | `tests/test_split_frozen.py` | Defined in task T0.2; validates hardcoded SHA256 hash and partition integrity. |
| `RUNLOG.md` | `RUNLOG.md` | Defined in task T0.2; markdown table logging all benchmark and held-out runs. |
| `facts.json` | `facts.json` | Defined in task T0.2; skeleton facts database for paper verifier. |
| `scripts/check_facts.py` | `scripts/check_facts.py` | Defined in task T0.2; assertion gate verifying numerical consistency in paper. |
| `perron/results.py` | `perron/results.py` | Defined in task T0.3; standardized results serialization helper capturing G5 metadata. |
| `tests/test_results.py` | `tests/test_results.py` | Defined in task T0.3; unit test verifying G5 metadata keys in serialized results. |
| `scripts/run_heldout.py` | `scripts/run_heldout.py` | Defined in task G3; dedicated gate for held-out evaluation logging to `RUNLOG.md`. |
| `results/heldout_table2_v1_preaudit.json` | `results/heldout_table2_v1_preaudit.json` | Defined in task G8; archived pre-audit held-out baseline result file. |
| `scripts/build_instance_graphs.py` | `scripts/export_swebench_graphs.py` | Phase 4 graph builder mapped to existing authoritative export pipeline with `--cleanup-clones`. |
| `scripts/make_gold.py` | `benchmarks/gold_patch_parser.py` | Phase 4 gold label generation mapped to existing gold parser generating `data_release/gold.jsonl`. |
| `perron/bench.py` | `benchmarks/hub_gold_ablation.py` | Phase 4 hub-gold benchmark mapped to existing ablation evaluation harness. |
