# Perron Empirical Claims Ledger

This ledger binds all empirical assertions made in `paper/paper_writeup.md` and `paper/main.tex` to authoritative serialized result files, following the empirical verification protocol.

---

## 1. Primary Empirical Claims Ledger

Claims are defined in `paper/claims_spec.json` and checked by `scripts/check_facts.py`.

---

## 2. Empirical Scope & Methodological Invariants

- **Lexical Baseline Invariant**: Pure lexical BM25 (18.0%) outperforms or matches graph diffusion on full SWE-bench Lite ($N=150$). Perron is framed honestly as a high-throughput CSR graph diffusion, hub suppression, and context-packing substrate.
- **Host Execution Boundary Invariant**: Zero SWE repair claims without verified execution (`tests_pass: null` preserved). Engine memory reported strictly from authentic host telemetry (`results/memory_measured.json`).
