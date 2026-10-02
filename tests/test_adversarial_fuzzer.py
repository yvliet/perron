"""
Perron Maximum-Entropy Adversarial Fuzzing Suite.

Probes edge-case AST syntax, graph topologies, scale-free limits,
and subprocess isolation boundaries to actively break system invariants.
"""

from __future__ import annotations

import ast
import pytest
from collections import deque
import concurrent.futures
from pathlib import Path
import tempfile
import time
import numpy as np
from scipy.sparse import csr_matrix

from perron.matrix import (
    build_static_transition_matrix,
    save_mmap_csr,
    load_mmap_csr,
    close_mmap_csr,
)
from perron.diffusion import (
    compute_softmax_teleport_prior,
    personalized_pagerank_power_iteration,
)
from perron.specificity import (
    compute_global_pagerank,
    calculate_specificity_scores,
)
from perron.packer import (
    ASTContextSymbol,
    pack_context_subgraphs,
    format_hierarchical_context,
)
from perron.editor import (
    apply_symbol_edit,
    find_block_match,
    insert_imports_safely,
)
from perron.tester import (
    run_targeted_test,
)


def run_fuzzer():
    failures = []
    print("======================================================================")
    print("STARTING PERRON MAXIMUM-ENTROPY ADVERSARIAL FUZZING HARNESS")
    print("======================================================================")

    # --------------------------------------------------------------------
    # PROBE 1: Editor Empty String, Non-existent Targets, and Trailing Spaces
    # --------------------------------------------------------------------
    print("\n[PROBE 1] Fuzzing Editor: Empty target, whitespace mismatch, no trailing newline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "empty_target.py"
        f.write_text("x = 1\ny = 2\n", encoding="utf-8")
        
        # Test 1.1: Empty old_str
        ok, msg = apply_symbol_edit(f, "", "z = 3")
        if ok:
            failures.append("Editor allowed empty old_str replacement!")
        print("  -> Empty old_str rejected:", not ok, f"({msg})")

        # Test 1.2: File without trailing newline
        f2 = Path(tmpdir) / "no_newline.py"
        f2.write_bytes(b"def foo():\n    return 42") # No trailing newline
        ok2, msg2 = apply_symbol_edit(f2, "return 42", "return 100")
        f2_bytes = f2.read_bytes()
        print("  -> File without trailing newline edited ok:", ok2)
        if f2_bytes.endswith(b"\n"):
            print("  [NOTE] File gained trailing newline during replacement.")

        # Test 1.3: Target with subtle trailing spaces vs file without
        f3 = Path(tmpdir) / "trailing_space.py"
        f3.write_text("class Foo:\n    def bar(self):   \n        return 1\n", encoding="utf-8")
        # Query has no trailing spaces on 'def bar(self):'
        ok3, msg3 = apply_symbol_edit(f3, "def bar(self):\n    return 1", "def bar(self):\n    return 2")
        print("  -> Trailing space mismatch resolution:", ok3, msg3)
        if not ok3:
            failures.append(f"Editor failed on trailing space mismatch: {msg3}")

    # --------------------------------------------------------------------
    # PROBE 2: Editor Exotic Python 3.10+ AST Constructs & Syntax Bomb Rollback
    # --------------------------------------------------------------------
    print("\n[PROBE 2] Fuzzing Editor: Walrus operators, nested async, syntax bomb rollback...")
    with tempfile.TemporaryDirectory() as tmpdir:
        exotic_source = (
            "async def outer():\n"
            "    if (n := len([x async for x in gen()])) > 0:\n"
            "        data = f'''prefix {n} suffix'''\n"
            "        match data:\n"
            "            case str(s) if 'prefix' in s:\n"
            "                return lambda cb=lambda x: x * 2: cb(n)\n"
            "            case _:\n"
            "                return None\n"
        )
        f_exotic = Path(tmpdir) / "exotic.py"
        f_exotic.write_text(exotic_source, encoding="utf-8")

        # Edit inner lambda
        old_lambda = "return lambda cb=lambda x: x * 2: cb(n)"
        new_lambda = "return lambda cb=lambda x: x * 4: cb(n)"
        ok_l, msg_l = apply_symbol_edit(f_exotic, old_lambda, new_lambda)
        print("  -> Exotic walrus + async comprehension + match/case edit:", ok_l, msg_l)
        if not ok_l:
            failures.append(f"Failed to edit exotic AST construct: {msg_l}")

        # Syntax bomb: replacement has illegal syntax
        f_pre = f_exotic.read_text(encoding="utf-8")
        bad_syntax = "return (unbalanced_paren"
        ok_bad, msg_bad = apply_symbol_edit(f_exotic, new_lambda, bad_syntax)
        f_post = f_exotic.read_text(encoding="utf-8")
        print("  -> Syntax bomb rejected:", not ok_bad, f"({msg_bad})")
        if ok_bad:
            failures.append("Editor accepted code with SyntaxError!")
        if f_pre != f_post:
            failures.append("Editor corrupted disk file on syntax verification failure!")

    # --------------------------------------------------------------------
    # PROBE 3: Diffusion Extreme Topologies & Giant Power-Law Graphs
    # --------------------------------------------------------------------
    print("\n[PROBE 3] Fuzzing Diffusion: Massive scale-free graph (|V|=25,000), single mega-hub...")
    n_giant = 25000
    rng = np.random.default_rng(12345)
    
    # 1 mega hub at node 0 connected to 10,000 nodes
    mega_hub = 0
    spokes = rng.choice(np.arange(1, n_giant), size=10000, replace=False)
    call_edges = [(s, mega_hub) for s in spokes]
    caller_edges = [(mega_hub, s) for s in spokes[:2000]] # Asymmetric callers

    # Chain 20000 -> 20001 -> ... -> 24999 (Long DAG chain)
    for c in range(20000, 24999):
        call_edges.append((c, c + 1))

    t0 = time.perf_counter()
    t_giant, d_giant = build_static_transition_matrix(n_giant, call_edges, caller_edges)
    t_build = time.perf_counter() - t0
    print(f"  -> Built |V|={n_giant}, |E|={t_giant.nnz} in {t_build*1000:.2f}ms")

    # Extreme similarity vector with NaNs, Infs, and negative values
    sims = rng.normal(0, 1, size=n_giant)
    sims[0] = 500.0 # mega hub high similarity
    sims[100] = np.nan
    sims[200] = np.inf
    sims[300] = -np.inf
    sims[400:500] = -1e10

    t1 = time.perf_counter()
    p_0_giant = compute_softmax_teleport_prior(sims, np.arange(n_giant), n_giant, tau=0.05, top_k=25)
    t_p0 = time.perf_counter() - t1
    print(f"  -> Computed prior p_0 in {t_p0*1000:.2f}ms: sum={np.sum(p_0_giant):.6f}, NaNs={np.isnan(p_0_giant).any()}")
    if not np.isclose(np.sum(p_0_giant), 1.0):
        failures.append(f"Giant p_0 sum is not 1.0: {np.sum(p_0_giant)}")

    t2 = time.perf_counter()
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_giant = personalized_pagerank_power_iteration(t_giant, d_giant, p_0_giant, beta=0.85, max_iter=25)
    t_ppr = time.perf_counter() - t2
    print(f"  -> Ran PPR on |V|={n_giant} in {t_ppr*1000:.2f}ms: sum={np.sum(pi_giant):.6f}, min={np.min(pi_giant):.8e}")
    if not np.isclose(np.sum(pi_giant), 1.0, atol=1e-4):
        failures.append(f"Giant PPR sum drifted from 1.0: {np.sum(pi_giant)}")
    if (pi_giant < 0).any():
        failures.append(f"Giant PPR has negative probabilities: min={np.min(pi_giant)}")

    # --------------------------------------------------------------------
    # PROBE 4: Packer Token Budget Starvation & Single-Symbol Overrun
    # --------------------------------------------------------------------
    print("\n[PROBE 4] Fuzzing Packer: Budget=1, Symbol token count=5,000, 50 Disjoint Clusters...")
    symbols_fuzz = {}
    # Symbol 0 costs 5,000 tokens (larger than budget)
    symbols_fuzz[0] = ASTContextSymbol(0, "giant_fn", "giant.py", 1, 100, "def giant():\n    pass\n", token_count=5000)
    # Symbol 1 costs 10 tokens
    symbols_fuzz[1] = ASTContextSymbol(1, "small_fn", "small.py", 1, 5, "def small():\n    pass\n", token_count=10)

    spec_fuzz = np.array([100.0, 50.0], dtype=np.float32)
    adj_fuzz = csr_matrix((2, 2), dtype=np.float32)

    # Test 4.1: token_budget = 100 (smaller than giant symbol)
    packed_under, prompt_under = pack_context_subgraphs(symbols_fuzz, spec_fuzz, adj_fuzz, token_budget=100)
    packed_ids = [s.node_id for s in packed_under]
    print(f"  -> Budget=100 with 5000-token anchor: selected nodes = {packed_ids}")
    if 0 in packed_ids:
        failures.append("Packer selected symbol whose cost exceeded token_budget!")
    if 1 not in packed_ids:
        failures.append("Packer failed to pick small_fn after skipping oversized anchor!")

    # Test 4.2: 50 Disjoint clusters with token_budget = 3480
    n_clusters = 50
    symbols_disjoint = {}
    spec_disjoint = np.zeros(n_clusters * 2, dtype=np.float32)
    rows, cols = [], []
    for c in range(n_clusters):
        u = c * 2
        v = c * 2 + 1
        symbols_disjoint[u] = ASTContextSymbol(u, f"fn_{u}", f"mod_{c}.py", 1, 5, f"def fn_{u}(): pass", token_count=30)
        symbols_disjoint[v] = ASTContextSymbol(v, f"fn_{v}", f"mod_{c}.py", 6, 10, f"def fn_{v}(): pass", token_count=30)
        spec_disjoint[u] = float(n_clusters - c) # Decreasing specificity
        spec_disjoint[v] = float(n_clusters - c) * 0.5
        rows.extend([u, v])
        cols.extend([v, u])

    adj_disjoint = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_clusters*2, n_clusters*2))
    packed_disjoint, prompt_disjoint = pack_context_subgraphs(symbols_disjoint, spec_disjoint, adj_disjoint, token_budget=3480)
    print(f"  -> 50 Disjoint clusters packed: {len(packed_disjoint)} symbols selected without crash.")
    if len(packed_disjoint) == 0:
        failures.append("Packer returned 0 symbols on 50 disjoint clusters!")

    # --------------------------------------------------------------------
    # PROBE 5: Tester Massive Stderr/Stdout Buffer Exhaustion & Fast Timeouts
    # --------------------------------------------------------------------
    print("\n[PROBE 5] Fuzzing Tester: 1MB stdout burst, tight timeout deadlock check...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 5.1: Test that prints massive output to verify pipe buffers don't deadlock
        f_spam = Path(tmpdir) / "test_spam.py"
        f_spam.write_text(
            "import sys\n"
            "def test_spam_output():\n"
            "    for _ in range(10000):\n"
            "        print('SPAM_LINE_' * 10)\n"
            "    assert True\n",
            encoding="utf-8",
        )
        res_spam = run_targeted_test(f_spam, timeout_seconds=10.0)
        print("  -> Massive output test passed:", res_spam.passed, f"({len(res_spam.output)} chars)")
        if not res_spam.passed:
            failures.append(f"Tester failed on massive stdout burst: {res_spam.output[:200]}")

        # Test 5.2: Test that sleeps past timeout
        f_sleep = Path(tmpdir) / "test_sleep.py"
        f_sleep.write_text(
            "import time\n"
            "def test_sleep_hang():\n"
            "    time.sleep(10)\n"
            "    assert True\n",
            encoding="utf-8",
        )
        t_start = time.monotonic()
        res_sleep = run_targeted_test(f_sleep, timeout_seconds=1.0)
        elapsed_sleep = time.monotonic() - t_start
        print(f"  -> Timeout kill in {elapsed_sleep:.2f}s: timed_out={res_sleep.timed_out}, passed={res_sleep.passed}")
        if not res_sleep.timed_out:
            failures.append("Tester failed to detect timeout on hanging test!")
        if elapsed_sleep > 4.0:
            failures.append(f"Tester took too long to kill process tree: {elapsed_sleep:.2f}s")

    # --------------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------------
    print("\n======================================================================")
    print(f"FUZZING HARNESS SUMMARY: {len(failures)} FAILURES DETECTED")
    print("======================================================================")
    for f in failures:
        print("  [FAIL] " + f)
    
    return len(failures) == 0


def test_adversarial_fuzzer():
    """Pytest discovery entrypoint for maximum-entropy fuzzing battery."""
    assert run_fuzzer(), "Adversarial fuzzer encountered failures."


if __name__ == "__main__":
    success = run_fuzzer()
    if not success:
        exit(1)
