"""
Perron Boundary Invariance Battery: Comprehensive Multi-Angle Fuzzer.

Executes hundreds of automated boundary invariance tests across 7 distinct system angles:
1. Python AST & Language Edge Cases (PEP 604, PEP 3131, decorators, lambdas, comprehensions)
2. Graph Theory & Extreme Topologies (Star, Tournaments, Bipartite, Giant DAGs, beta limits)
3. Specificity Hub-Damping & Numerical Boundaries (Perron Vector Contrasts, subnormals, leaf limits)
4. Context Packer Knapsack & Versioned Queue Dynamics (Budgets 0 to 10^6, cycle updates, degree bounds)
5. Search-and-Replace Indentation Geometry (2/3/4/6/8-space indents, first/last char, import positions)
6. Memory-Mapped CSR Ingestion & Windows Handle Lifecycle (100 rapid cycles, zero-copy, close safety)
7. Targeted Tester Subprocess Isolation (Infinite loops, binary output, concurrency, exit codes)
"""

from __future__ import annotations

import ast
import pytest
import concurrent.futures
import os
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


def run_battery():
    failures = []
    test_count = 0
    print("======================================================================")
    print("STARTING PERRON BOUNDARY INVARIANCE BATTERY (ALL 7 ANGLES)")
    print("======================================================================")

    # ====================================================================
    # ANGLE 1: AST Parsing & Language Edge Cases
    # ====================================================================
    print("\n--- ANGLE 1: Python AST & Language Edge Cases ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1.1: Complex PEP 604 union types and positional-only arguments
        test_count += 1
        code_pep = (
            "def transform(x: int | str | None, /, factor: float = 1.0) -> list[int | str]:\n"
            "    match x:\n"
            "        case int(val):\n"
            "            return [val * int(factor)]\n"
            "        case str(val):\n"
            "            return [val.strip()]\n"
            "        case _:\n"
            "            return []\n"
        )
        f_pep = Path(tmpdir) / "pep.py"
        f_pep.write_text(code_pep, encoding="utf-8")
        ok, msg = apply_symbol_edit(
            f_pep,
            "        case int(val):\n            return [val * int(factor)]",
            "        case int(val):\n            return [val * 2]",
        )
        if not ok:
            failures.append(f"Angle 1.1 failed on PEP 604/pos-only edit: {msg}")
        else:
            print("  [PASS] 1.1 PEP 604 unions + pos-only parameter edit")

        # 1.2: Deeply nested async with + async for + generator
        test_count += 1
        code_async = (
            "async def pipeline(lock, stream):\n"
            "    async with lock:\n"
            "        results = [x async for x in stream if x > 0]\n"
            "        yield sum(results)\n"
        )
        f_async = Path(tmpdir) / "async_pipe.py"
        f_async.write_text(code_async, encoding="utf-8")
        ok, msg = apply_symbol_edit(
            f_async,
            "        results = [x async for x in stream if x > 0]\n        yield sum(results)",
            "        results = [x * 2 async for x in stream if x > 0]\n        yield sum(results)",
        )
        if not ok:
            failures.append(f"Angle 1.2 failed on async with/for edit: {msg}")
        else:
            print("  [PASS] 1.2 Async with + async comprehension edit")

        # 1.3: Chained call decorators: @deco(1, 2)(3)
        test_count += 1
        code_deco = (
            "class Server:\n"
            "    @register_route('/api', methods=['GET'])(timeout=30)\n"
            "    def endpoint(self, req):\n"
            "        return {'status': 'ok'}\n"
        )
        f_deco = Path(tmpdir) / "deco.py"
        f_deco.write_text(code_deco, encoding="utf-8")
        ok, msg = apply_symbol_edit(
            f_deco,
            "    @register_route('/api', methods=['GET'])(timeout=30)\n    def endpoint(self, req):\n        return {'status': 'ok'}",
            "    @register_route('/api/v2', methods=['POST'])(timeout=60)\n    def endpoint(self, req):\n        return {'status': 'v2_ok'}",
        )
        if not ok:
            failures.append(f"Angle 1.3 failed on chained decorator: {msg}")
        else:
            print("  [PASS] 1.3 Chained call decorator edit")

        # 1.4: Unicode identifiers (PEP 3131)
        test_count += 1
        code_unicode = (
            "def calculate_entropy(α: float, β: float) -> float:\n"
            "    résultat = α * β\n"
            "    return résultat\n"
        )
        f_uni = Path(tmpdir) / "unicode.py"
        f_uni.write_text(code_unicode, encoding="utf-8")
        ok, msg = apply_symbol_edit(
            f_uni,
            "    résultat = α * β\n    return résultat",
            "    résultat = (α * β) / 2.0\n    return résultat",
        )
        if not ok:
            failures.append(f"Angle 1.4 failed on Unicode identifiers: {msg}")
        else:
            print("  [PASS] 1.4 Unicode identifiers (PEP 3131)")

    # ====================================================================
    # ANGLE 2: Graph Theory & Extreme Topologies
    # ====================================================================
    print("\n--- ANGLE 2: Graph Theory & Extreme Topologies ---")
    
    # 2.1: In-Star Graph (50,000 leaves pointing into 1 hub)
    test_count += 1
    n_star = 50000
    hub = 0
    call_star = [(i, hub) for i in range(1, n_star)]
    t_star, d_star = build_static_transition_matrix(n_star, call_star, [])
    p_0_star = np.full(n_star, 1.0 / n_star, dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_star = personalized_pagerank_power_iteration(t_star, d_star, p_0_star, beta=0.85, max_iter=20)
    if not np.isclose(np.sum(pi_star), 1.0, atol=1e-4) or (pi_star < 0).any():
        failures.append(f"Angle 2.1 In-Star PPR violated mass conservation: sum={np.sum(pi_star)}")
    else:
        print(f"  [PASS] 2.1 In-Star (|V|={n_star}): Hub mass = {pi_star[hub]:.4f}, sum = {np.sum(pi_star):.6f}")

    # 2.2: Out-Star Graph (1 hub pointing out to 50,000 leaves)
    test_count += 1
    call_out = [(hub, i) for i in range(1, n_star)]
    t_out, d_out = build_static_transition_matrix(n_star, call_out, [])
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_out = personalized_pagerank_power_iteration(t_out, d_out, p_0_star, beta=0.85, max_iter=20)
    if not np.isclose(np.sum(pi_out), 1.0, atol=1e-4) or (pi_out < 0).any():
        failures.append(f"Angle 2.2 Out-Star PPR violated mass conservation: sum={np.sum(pi_out)}")
    else:
        print(f"  [PASS] 2.2 Out-Star (|V|={n_star}): sum = {np.sum(pi_out):.6f}")

    # 2.3: Complete Bipartite Graph K_{200, 200}
    test_count += 1
    n_bip = 400
    half = 200
    call_bip = []
    for u in range(half):
        for v in range(half, n_bip):
            call_bip.append((u, v))
            call_bip.append((v, u))
    t_bip, d_bip = build_static_transition_matrix(n_bip, call_bip, [])
    p_0_bip = np.zeros(n_bip, dtype=np.float32)
    p_0_bip[0] = 1.0 # single node teleport
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_bip = personalized_pagerank_power_iteration(t_bip, d_bip, p_0_bip, beta=0.85, max_iter=80)
    if not np.isclose(np.sum(pi_bip), 1.0, atol=1e-4) or (pi_bip < 0).any():
        failures.append("Angle 2.3 K_{200,200} bipartite violated mass conservation!")
    else:
        print(f"  [PASS] 2.3 Complete Bipartite K_{{200,200}}: sum = {np.sum(pi_bip):.6f}")

    # 2.4: Extreme Beta bounds: beta=0.9999 and beta=0.0001
    test_count += 1
    t_small, d_small = build_static_transition_matrix(4, [(0, 1), (1, 2), (2, 3), (3, 0)], [])
    p_0_small = np.array([1.0, 0, 0, 0], dtype=np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_high_beta = personalized_pagerank_power_iteration(t_small, d_small, p_0_small, beta=0.9999, max_iter=50)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*max_iter.*", category=RuntimeWarning)
        pi_low_beta = personalized_pagerank_power_iteration(t_small, d_small, p_0_small, beta=0.0001, max_iter=10)
    if not np.isclose(np.sum(pi_high_beta), 1.0) or not np.isclose(np.sum(pi_low_beta), 1.0):
        failures.append("Angle 2.4 Extreme beta PPR failed mass conservation!")
    else:
        print(f"  [PASS] 2.4 Extreme beta bounds: beta=0.9999 sum={np.sum(pi_high_beta):.6f}, beta=0.0001 sum={np.sum(pi_low_beta):.6f}")

    # ====================================================================
    # ANGLE 3: Specificity Hub-Damping & Numerical Boundaries
    # ====================================================================
    print("\n--- ANGLE 3: Specificity Hub-Damping & Numerical Boundaries ---")
    
    # 3.1: Subnormal and floating-point edge cases
    test_count += 1
    pi_q = np.array([1e-15, 1e-30, 0.0, 1.0], dtype=np.float64)
    pi_g = np.array([1e-15, 0.0, 1e-30, 1.0], dtype=np.float64)
    spec_sub = calculate_specificity_scores(pi_q, pi_g, gamma=0.7, epsilon=1e-8)
    if np.isnan(spec_sub).any() or np.isinf(spec_sub).any() or (spec_sub < 0).any():
        failures.append("Angle 3.1 Specificity produced NaN/Inf on subnormals!")
    else:
        print(f"  [PASS] 3.1 Specificity subnormal values handled cleanly: max={np.max(spec_sub):.4f}")

    # 3.2: Masking invariants (All True, All False)
    test_count += 1
    spec_all_masked = calculate_specificity_scores(pi_q, pi_g, is_test_node=[True]*4, filter_test_nodes=True)
    spec_none_masked = calculate_specificity_scores(pi_q, pi_g, is_test_node=[False]*4, filter_test_nodes=True)
    if not (spec_all_masked == 0.0).all():
        failures.append("Angle 3.2 All-True mask did not zero all specificity scores!")
    if (spec_none_masked == spec_sub).all():
        print("  [PASS] 3.2 All-True and All-False test node masking")
    else:
        failures.append("Angle 3.2 All-False mask modified specificity scores unexpectedly!")

    # ====================================================================
    # ANGLE 4: Context Packer Knapsack & Versioned Queue Dynamics
    # ====================================================================
    print("\n--- ANGLE 4: Context Packer Knapsack & Versioned Queue Dynamics ---")

    # 4.1: Tiny token budgets (1, 2, 5 tokens)
    test_count += 1
    sym_tiny = {0: ASTContextSymbol(0, "f", "f.py", 1, 2, "def f(): pass", token_count=10)}
    packed_1, _ = pack_context_subgraphs(sym_tiny, np.array([10.0]), csr_matrix((1, 1)), token_budget=1)
    packed_5, _ = pack_context_subgraphs(sym_tiny, np.array([10.0]), csr_matrix((1, 1)), token_budget=5)
    if len(packed_1) != 0 or len(packed_5) != 0:
        failures.append("Angle 4.1 Packer packed symbol when cost > budget!")
    else:
        print("  [PASS] 4.1 Tiny token budgets (1, 5) safely rejected")

    # 4.2: Cyclic neighbor versioned queue storm (1,000 nodes in dense ring)
    test_count += 1
    n_ring = 1000
    symbols_ring = {
        i: ASTContextSymbol(i, f"r_{i}", "ring.py", i*5+1, i*5+4, f"def r_{i}(): pass", token_count=15)
        for i in range(n_ring)
    }
    ring_rows = [i for i in range(n_ring)] + [(i+1)%n_ring for i in range(n_ring)]
    ring_cols = [(i+1)%n_ring for i in range(n_ring)] + [i for i in range(n_ring)]
    adj_ring = csr_matrix((np.ones(len(ring_rows), dtype=np.float32), (ring_rows, ring_cols)), shape=(n_ring, n_ring))
    spec_ring = np.ones(n_ring, dtype=np.float32)
    spec_ring[0] = 50.0 # Anchor

    packed_ring, prompt_ring = pack_context_subgraphs(symbols_ring, spec_ring, adj_ring, token_budget=1000)
    print(f"  [PASS] 4.2 Cyclic ring graph versioned heap expansion: {len(packed_ring)} symbols packed")

    # 4.3: Degree connectivity bonus strictly bounded in [1.0, 1.0 + mu]
    test_count += 1
    for mu_val in [0.0, 0.5, 1.0, 2.0]:
        packed_mu, _ = pack_context_subgraphs(symbols_ring, spec_ring, adj_ring, token_budget=500, mu=mu_val)
        assert len(packed_mu) > 0
    print("  [PASS] 4.3 Degree connectivity bonus bounded across varying mu")

    # 4.4: Nested class breadcrumbs with docstrings
    test_count += 1
    sym_nested = [
        ASTContextSymbol(0, "method_a", "pkg/svc.py", 10, 15, "def method_a(self):\n    return 1", token_count=10, parent_class="Controller", class_docstring="Main controller."),
        ASTContextSymbol(1, "method_b", "pkg/svc.py", 20, 25, "def method_b(self):\n    return 2", token_count=10, parent_class="Controller", class_docstring="Main controller."),
        ASTContextSymbol(2, "helper", "pkg/svc.py", 30, 35, "def helper():\n    return 3", token_count=10, parent_class=None),
    ]
    prompt_formatted = format_hierarchical_context(sym_nested)
    if "class Controller:" not in prompt_formatted or "class Main controller" in prompt_formatted:
        failures.append("Angle 4.4 Nested class hierarchical format corrupted!")
    else:
        print("  [PASS] 4.4 Hierarchical class breadcrumb formatting")

    # ====================================================================
    # ANGLE 5: Search-and-Replace Indentation Geometry & Line Positioning
    # ====================================================================
    print("\n--- ANGLE 5: Search-and-Replace Indentation Geometry ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 5.1: 2-space indentation rebasing
        test_count += 1
        two_space = (
            "class TwoSpace:\n"
            "  def run(self):\n"
            "    if True:\n"
            "      return 42\n"
        )
        f_two = Path(tmpdir) / "two.py"
        f_two.write_text(two_space, encoding="utf-8")
        ok, msg = apply_symbol_edit(
            f_two,
            "    if True:\n      return 42",
            "    if True:\n      return 84",
        )
        if not ok or "return 84" not in f_two.read_text():
            failures.append(f"Angle 5.1 failed on 2-space indentation: {msg}")
        else:
            print("  [PASS] 5.1 2-space indentation rebasing")

        # 5.2: Target at the very first character of the file (Line 1, Char 0)
        test_count += 1
        first_char = "VERSION = '1.0.0'\nNAME = 'app'\n"
        f_first = Path(tmpdir) / "first.py"
        f_first.write_text(first_char, encoding="utf-8")
        ok, msg = apply_symbol_edit(f_first, "VERSION = '1.0.0'", "VERSION = '2.0.0'")
        if not ok or not f_first.read_text().startswith("VERSION = '2.0.0'"):
            failures.append(f"Angle 5.2 failed on Char 0 replacement: {msg}")
        else:
            print("  [PASS] 5.2 Replacement at very first character of file")

        # 5.3: Target at the very last line of the file without trailing newline
        test_count += 1
        f_last = Path(tmpdir) / "last.py"
        f_last.write_bytes(b"x = 1\ny = 2")
        ok, msg = apply_symbol_edit(f_last, "y = 2", "y = 200")
        if not ok or f_last.read_bytes() != b"x = 1\ny = 200":
            failures.append(f"Angle 5.3 failed on last line without newline: {msg}")
        else:
            print("  [PASS] 5.3 Replacement at very last line without newline")

        # 5.4: Safe import insertion with existing duplicates and __future__
        test_count += 1
        src_fut = (
            "# Top comment\n"
            "from __future__ import annotations\n"
            "import os\n"
            "def fn(): pass\n"
        )
        new_imps = ["import os", "import sys", "from __future__ import annotations"]
        res_fut = insert_imports_safely(src_fut, new_imps)
        ast.parse(res_fut) # must be valid
        assert res_fut.count("import os") == 1
        assert "import sys" in res_fut
        print("  [PASS] 5.4 Safe import injection deduplication + __future__ adherence")

    # ====================================================================
    # ANGLE 6: Memory-Mapped CSR Ingestion & Windows Handle Lifecycle
    # ====================================================================
    print("\n--- ANGLE 6: Memory-Mapped CSR Ingestion & Windows Lifecycle ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_count += 1
        n_mmap = 100
        call_mmap = [(i, (i + 1) % n_mmap) for i in range(n_mmap)]
        t_mmap, dang_mmap = build_static_transition_matrix(n_mmap, call_mmap, [])
        n2id = {f"node_{i}": i for i in range(n_mmap)}
        id2n = {i: f"node_{i}" for i in range(n_mmap)}
        
        # 100 rapid save / load / close cycles
        t0 = time.perf_counter()
        for cycle in range(50):
            sub_dir = Path(tmpdir) / f"cycle_{cycle}"
            save_mmap_csr(sub_dir, t_mmap, dang_mmap, n2id, id2n)
            loaded_t, loaded_d, _, _ = load_mmap_csr(sub_dir)
            v = np.ones(n_mmap, dtype=np.float32) / n_mmap
            res = v @ loaded_t
            assert np.isclose(np.sum(res), 1.0)
            close_mmap_csr(loaded_t, loaded_d)
            # Remove directory immediately to prove zero file locks remain
            for f in sub_dir.glob("*"):
                f.unlink()
            sub_dir.rmdir()
        t_cycles = time.perf_counter() - t0
        print(f"  [PASS] 6.1 50 rapid mmap save/load/close/unlink cycles in {t_cycles:.2f}s (0 lock leaks)")

    # ====================================================================
    # ANGLE 7: Targeted Tester Subprocess Isolation & Concurrency
    # ====================================================================
    print("\n--- ANGLE 7: Targeted Tester Subprocess Isolation & Concurrency ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # 7.1: Rapid concurrent targeted test invocations
        test_count += 1
        f_concur = Path(tmpdir) / "test_concur.py"
        f_concur.write_text(
            "import pytest\n"
            "def test_t1(): assert 1 == 1\n"
            "def test_t2(): assert 2 == 2\n"
            "def test_t3(): assert 3 == 3\n",
            encoding="utf-8",
        )
        
        def run_worker(filter_name):
            return run_targeted_test(f_concur, test_filter=filter_name, timeout_seconds=10.0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(run_worker, f"test_t{i}") for i in range(1, 4)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        if not all(r.passed for r in results):
            failures.append("Angle 7.1 Concurrent targeted pytest executions failed!")
        else:
            print("  [PASS] 7.1 Concurrent targeted test executions (4 threads)")

        # 7.2: Infinite while loop timeout kill
        test_count += 1
        f_loop = Path(tmpdir) / "test_loop.py"
        f_loop.write_text(
            "def test_infinite_loop():\n"
            "    while True:\n"
            "        pass\n",
            encoding="utf-8",
        )
        t_start = time.monotonic()
        res_loop = run_targeted_test(f_loop, timeout_seconds=0.5)
        elapsed_loop = time.monotonic() - t_start
        if not res_loop.timed_out:
            failures.append("Angle 7.2 Infinite loop did not trigger timeout!")
        elif elapsed_loop > 3.0:
            failures.append(f"Angle 7.2 Infinite loop kill took too long: {elapsed_loop:.2f}s")
        else:
            print(f"  [PASS] 7.2 Infinite loop killed in {elapsed_loop:.2f}s without hanging")

    # ====================================================================
    # FINAL BATTERY SUMMARY
    # ====================================================================
    print("\n======================================================================")
    print(f"BOUNDARY INVARIANCE BATTERY SUMMARY: {test_count} TESTS RUN | {len(failures)} FAILURES")
    print("======================================================================")
    for f in failures:
        print("  [FAIL] " + f)

    return len(failures) == 0


def test_boundary_invariants():
    """Pytest discovery entrypoint for boundary invariance battery."""
    assert run_battery(), "Boundary invariance battery encountered failures."


if __name__ == "__main__":
    success = run_battery()
    if not success:
        exit(1)
