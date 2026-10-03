# Changelog

All notable changes to the `perron-core` library and dataset resource are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), adhering to Semantic Versioning.

---

## [0.3.0] - 2026-10-03

### Added
- **SWE-bench Lite Multigraph Resource Release**: Exported 300 pre-indexed zero-copy CSR transition matrices and AST symbol tables in `data_release/graphs/` (4.8M nodes, 41.5M edges).
- **MLCommons Croissant 1.0 Specification**: Formalized `data_release/croissant.json` metadata specification with zero-warning `mlcroissant` validation.
- **Datasheet for Datasets**: Formalized `data_release/DATASHEET.md` following Gebru et al. standards.
- **Hub-Gold Dissection Benchmark**: Implemented `perron bench` CLI command and `perron/bench.py` evaluating `Acc@10(H)`, `Acc@10(N)`, `HSI@10`, and `HEADLINE` metrics.
- **High-Throughput Graph Loaders**: Added `perron.loaders` module supporting `InstanceGraph`, NetworkX `to_networkx` and `from_networkx` conversions, and lazy PyG integration.
- **Deterministic Validation Suite**: Added `scripts/validate_release.py` and `tests/test_graph_determinism.py`.
- **Model Context Protocol Integration Test**: Added end-to-end stdio subprocess validation (`tests/test_mcp_stdio.py`) and wire trace (`docs/mcp_transcript.json`).
- **Turnkey Evaluation CLI**: Added `perron eval` command to execute registered baseline evaluation suites.

### Changed
- **Metric Harmonization**: Resolved empty gold boundary condition in evaluation harness to strictly return 0.0 coverage.
- **Clean-Room License Architecture**: Verified zero-source-text distribution to eliminate copyleft aggregation risks with upstream GPL components.
- **Test Isolation**: Parameterized instance output paths in evaluation probes to guarantee zero working tree mutations during `pytest` runs.

---

## [0.2.1] - 2026-10-02

### Added
- Model Context Protocol (MCP) server adhering to 2024-11-05 JSON-RPC specification over stdio.
- Formal Brauer rank-1 spectral bounds ($|\lambda_2| \le \beta = 0.85$).
- Query-level hub adaptivity ($\gamma(q)$).

---

## [0.2.0] - 2026-10-01

### Added
- Initial open-source release of Perron CSR spectral diffusion engine.
- Bipartite call and caller graph construction.
- Power-iteration Personalized PageRank solver.
