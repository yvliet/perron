"""
Comprehensive Test Suite for Perron Modules.

Tests matrix ingestion, diffusion, specificity scoring, context packing,
AST-grounded editing, and targeted test execution.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import numpy as np
from scipy.sparse import csr_matrix

from perron.matrix import (
    build_static_transition_matrix,
    close_mmap_csr,
    load_mmap_csr,
    save_mmap_csr,
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
)
from perron.editor import (
    apply_symbol_edit,
    apply_multi_file_patch,
    insert_imports_safely,
    SymbolEditSpec,
)
from perron.tester import (
    run_targeted_test,
)


def test_matrix_construction_and_mmap():
    num_nodes = 5
    call_edges = [(0, 1), (0, 2), (1, 3)]
    caller_edges = [(2, 3)]

    t_matrix, dangling = build_static_transition_matrix(
        num_nodes=num_nodes,
        call_edges=call_edges,
        caller_edges=caller_edges,
        lambda_call=1.0,
        lambda_caller=0.5,
    )

    assert t_matrix.shape == (5, 5)
    assert np.isclose(t_matrix[0, 1], 0.5)
    assert np.isclose(t_matrix[0, 2], 0.5)
    assert dangling[3] == 1.0
    assert dangling[4] == 1.0
    assert dangling[0] == 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        node_to_id = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        id_to_node = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"}

        save_mmap_csr(tmpdir, t_matrix, dangling, node_to_id, id_to_node)
        loaded_t, loaded_dang, loaded_n2id, loaded_id2n = load_mmap_csr(tmpdir)

        assert loaded_t.shape == t_matrix.shape
        np.testing.assert_allclose(loaded_t.toarray(), t_matrix.toarray())
        np.testing.assert_allclose(loaded_dang, dangling)
        assert loaded_n2id["a"] == 0
        assert loaded_id2n[3] == "d"

        # Explicitly close memory map handles on Windows before directory exit
        close_mmap_csr(loaded_t, loaded_dang)


def test_diffusion_numerical_stability_and_convergence():
    num_nodes = 4
    sims = [100.0, 95.0, 80.0, 50.0]
    nodes = [0, 1, 2, 3]

    p_0 = compute_softmax_teleport_prior(
        similarities=sims,
        node_indices=nodes,
        num_nodes=num_nodes,
        tau=0.05,
        top_k=3,
    )

    assert np.isclose(np.sum(p_0), 1.0)
    assert not np.isnan(p_0).any()
    assert not np.isinf(p_0).any()
    assert p_0[3] == 0.0
    assert p_0[0] > p_0[1]

    t_matrix = csr_matrix(np.array([[0.0, 1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    dangling = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    pi = personalized_pagerank_power_iteration(
        t_matrix=t_matrix,
        dangling=dangling,
        p_0=p_0,
        beta=0.85,
        max_iter=100,
    )

    assert np.isclose(np.sum(pi), 1.0)
    assert (pi >= 0.0).all()


def test_specificity_scoring_and_hub_damping():
    pi_global = np.array([0.5, 0.01, 0.01, 0.01], dtype=np.float32)
    pi_query = np.array([0.2, 0.15, 0.01, 0.01], dtype=np.float32)
    is_test = [False, False, True, False]

    scores = calculate_specificity_scores(
        pi_query=pi_query,
        pi_global=pi_global,
        gamma=0.7,
        is_test_node=is_test,
        filter_test_nodes=True,
    )

    assert scores[1] > scores[0]
    assert scores[2] == 0.0


def test_context_packer_and_ast_breadcrumbs():
    symbols = {
        0: ASTContextSymbol(
            node_id=0,
            name="process_item",
            file_path="src/engine.py",
            start_line=20,
            end_line=35,
            code="    def process_item(self, item):\n        return item.clean()",
            token_count=15,
            parent_class="Engine",
            class_docstring="Core execution engine.",
        ),
        1: ASTContextSymbol(
            node_id=1,
            name="clean",
            file_path="src/item.py",
            start_line=10,
            end_line=18,
            code="def clean(self):\n    return self.val.strip()",
            token_count=12,
        ),
    }

    scores = np.array([10.5, 8.2], dtype=np.float32)
    adj = csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.float32))

    packed, prompt = pack_context_subgraphs(
        symbols=symbols,
        specificity_scores=scores,
        adjacency_matrix=adj,
        token_budget=100,
    )

    assert len(packed) == 2
    assert "class Engine:" in prompt
    assert '"""Core execution engine."""' in prompt
    assert "def process_item" in prompt
    assert "def clean" in prompt


def test_editor_safe_import_placement():
    source_with_future = (
        '"""Module docstring."""\n'
        'from __future__ import annotations\n\n'
        'import os\n\n'
        'def run():\n'
        '    return os.name\n'
    )

    new_imports = ["import sys", "from typing import List"]
    updated = insert_imports_safely(source_with_future, new_imports)

    ast.parse(updated)

    lines = updated.splitlines()
    assert lines[0] == '"""Module docstring."""'
    assert lines[1] == 'from __future__ import annotations'
    assert any("import sys" in l for l in lines)
    assert any("from typing import List" in l for l in lines)


def test_editor_scoped_multiline_rebasing():
    file_content = (
        "class Worker:\n"
        "    def execute(self, val):\n"
        "        if val > 0:\n"
        "            return False\n"
        "        return True\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "worker.py"
        test_file.write_text(file_content, encoding="utf-8")

        # Multiline replacement with nested block (even if dedented in input query)
        old_str = "if val > 0:\n    return False"
        new_str = (
            "if val > 0:\n"
            "    adjusted = val * 2\n"
            "    if adjusted > 10:\n"
            "        return True\n"
            "    return False"
        )

        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str=old_str,
            new_str=new_str,
            start_line=2,
            end_line=5,
        )

        assert success, msg
        updated_content = test_file.read_text(encoding="utf-8")

        tree = ast.parse(updated_content)
        assert tree is not None
        assert "adjusted = val * 2" in updated_content
        assert "            adjusted = val * 2" in updated_content
        assert "            if adjusted > 10:" in updated_content
        assert "                return True" in updated_content


def test_targeted_test_runner():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_sample.py"
        test_file.write_text(
            "def test_pass():\n"
            "    assert 1 + 1 == 2\n\n"
            "def test_fail():\n"
            "    assert 1 == 2\n",
            encoding="utf-8",
        )

        res_pass = run_targeted_test(test_file, test_filter="test_pass", timeout_seconds=10.0)
        assert res_pass.passed, f"Test failed with output: {res_pass.output}"
        assert res_pass.exit_code == 0
        assert not res_pass.timed_out

        res_fail = run_targeted_test(test_file, test_filter="test_fail", timeout_seconds=10.0)
        assert not res_fail.passed
        assert res_fail.exit_code != 0

        res_no_match = run_targeted_test(test_file, test_filter="test_nonexistent", timeout_seconds=10.0)
        assert not res_no_match.passed
        assert res_no_match.exit_code == 5
        assert "exit code 5" in res_no_match.output


def test_component_budget_partitioning_and_zero_spec_isolation():
    # Two disconnected components: C1 = {0, 1}, C2 = {2, 3}, plus test node 4 with spec 0.0
    symbols = {
        0: ASTContextSymbol(
            node_id=0, name="c1_main", file_path="pkg/c1.py",
            start_line=1, end_line=10, code="def c1_main(): pass", token_count=50,
        ),
        1: ASTContextSymbol(
            node_id=1, name="c1_sub", file_path="pkg/c1.py",
            start_line=11, end_line=20, code="def c1_sub(): pass", token_count=50,
        ),
        2: ASTContextSymbol(
            node_id=2, name="c2_main", file_path="pkg/c2.py",
            start_line=1, end_line=10, code="def c2_main(): pass", token_count=50,
        ),
        3: ASTContextSymbol(
            node_id=3, name="c2_sub", file_path="pkg/c2.py",
            start_line=11, end_line=20, code="def c2_sub(): pass", token_count=50,
        ),
        4: ASTContextSymbol(
            node_id=4, name="test_routing", file_path="pkg/test_pkg.py",
            start_line=1, end_line=10, code="def test_routing(): pass", token_count=50,
        ),
    }

    # Edges: 0 <-> 1, 2 <-> 3, and 0 -> 4 (test node)
    adj = csr_matrix((
        [1.0, 1.0, 1.0, 1.0, 1.0],
        ([0, 1, 2, 3, 0], [1, 0, 3, 2, 4])
    ), shape=(5, 5), dtype=np.float32)

    specificity = np.array([10.0, 5.0, 8.0, 4.0, 0.0], dtype=np.float32)

    packed, prompt = pack_context_subgraphs(
        symbols=symbols,
        specificity_scores=specificity,
        adjacency_matrix=adj,
        token_budget=400,
    )

    packed_ids = {s.node_id for s in packed}
    # Verify multi-component fairness: both C1 anchor (0) and C2 anchor (2) are packed
    assert 0 in packed_ids
    assert 2 in packed_ids
    # Verify strict test node exclusion: test node 4 with 0.0 specificity is NEVER packed
    assert 4 not in packed_ids
    assert "test_routing" not in prompt


def test_editor_docstring_with_license_and_shebang_header():
    source_with_headers = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "# Copyright 2026 Sultan Haikal\n"
        "# MIT License\n"
        '"""Module docstring explaining core logic."""\n'
        "from __future__ import annotations\n\n"
        "def compute():\n"
        "    return 42\n"
    )

    new_imports = ["import math", "from typing import Dict"]
    updated = insert_imports_safely(source_with_headers, new_imports)

    ast.parse(updated)
    lines = updated.splitlines()

    # Verify shebang, encoding, license header preserved at the top
    assert lines[0] == "#!/usr/bin/env python3"
    assert lines[1] == "# -*- coding: utf-8 -*-"
    assert lines[2] == "# Copyright 2026 Sultan Haikal"
    assert lines[3] == "# MIT License"
    assert lines[4] == '"""Module docstring explaining core logic."""'
    assert lines[5] == "from __future__ import annotations"

    # Verify imports placed after docstring and __future__
    import_indices = [i for i, l in enumerate(lines) if "import math" in l or "typing" in l]
    assert all(idx > 5 for idx in import_indices)


def test_editor_windows_crlf_preservation():
    crlf_source = (
        "class Handler:\r\n"
        "    def run(self):\r\n"
        "        value = 1\r\n"
        "        return value\r\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "handler.py"
        test_file.write_bytes(crlf_source.encode("utf-8"))

        old_str = "value = 1\r\nreturn value"
        new_str = "value = 42\nreturn value * 2"

        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str=old_str,
            new_str=new_str,
            start_line=2,
            end_line=4,
        )

        assert success, msg
        raw_bytes = test_file.read_bytes()
        # Verify CRLF is preserved consistently throughout the entire file
        assert b"\r\n" in raw_bytes
        # Ensure no lone LF byte without preceding CR
        content_str = raw_bytes.decode("utf-8")
        assert "\r\n" in content_str
        assert content_str.count("\r\n") == len(content_str.splitlines())


def test_editor_preindented_search_and_replace_preservation():
    source = (
        "class Engine:\n"
        "    def execute(self):\n"
        "        pass\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "engine.py"
        test_file.write_text(source, encoding="utf-8")

        # Test both pre-indented new_str and dedented new_str with pre-indented old_str
        old_str = "    def execute(self):\n        pass"
        new_str = "    def execute(self):\n        return True"

        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str=old_str,
            new_str=new_str,
            start_line=2,
            end_line=3,
        )

        assert success, msg
        content = test_file.read_text(encoding="utf-8")
        ast.parse(content)
        assert "    def execute(self):" in content
        assert "        return True" in content


def test_specificity_numerical_underflow_and_validation():
    import pytest

    # Underflow with negative drift should not trigger NaN or warning
    pi_q = np.array([0.1, 0.2], dtype=np.float32)
    pi_g = np.array([-1e-7, 0.01], dtype=np.float32)
    scores = calculate_specificity_scores(pi_q, pi_g, gamma=0.7, epsilon=1e-8)
    assert not np.isnan(scores).any()
    assert scores[0] > 0.0
    assert scores[1] > 0.0

    # Dimension mismatch validation
    with pytest.raises(ValueError, match="Dimension mismatch"):
        calculate_specificity_scores(np.array([0.1]), np.array([0.1, 0.2]))

    # Invalid epsilon validation
    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        calculate_specificity_scores(pi_q, pi_q, epsilon=0.0)

    # Invalid gamma validation
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        calculate_specificity_scores(pi_q, pi_q, gamma=-0.5)

    # Test mask length mismatch validation
    with pytest.raises(ValueError, match="Length of is_test_node"):
        calculate_specificity_scores(pi_q, pi_q, is_test_node=[True])


def test_diffusion_validation_and_sanitization():
    import pytest

    # Sanitization of NaNs in similarity scores
    sims = [0.9, np.nan, 0.4]
    nodes = [0, 1, 2]
    p_0 = compute_softmax_teleport_prior(sims, nodes, num_nodes=3, tau=0.05)
    assert not np.isnan(p_0).any()
    assert np.isclose(np.sum(p_0), 1.0)
    assert p_0[0] > p_0[2] > p_0[1]

    # Out of bounds node index validation
    with pytest.raises(IndexError, match="outside"):
        compute_softmax_teleport_prior([0.5], [5], num_nodes=3)

    # Invalid tau validation
    with pytest.raises(ValueError, match="tau must be strictly positive"):
        compute_softmax_teleport_prior([0.5], [0], num_nodes=3, tau=-0.1)

    # Invalid top_k validation
    with pytest.raises(ValueError, match="top_k must be positive"):
        compute_softmax_teleport_prior([0.5, 0.2], [0, 1], num_nodes=3, top_k=0)

    # Matrix dimension check in power iteration
    t_mat = csr_matrix(np.zeros((3, 3), dtype=np.float32))
    dang = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError, match="p_0 length"):
        personalized_pagerank_power_iteration(t_mat, dang, p_0=np.ones(2))

    with pytest.raises(ValueError, match="beta must satisfy"):
        personalized_pagerank_power_iteration(t_mat, dang, p_0=np.ones(3), beta=1.5)


def test_weakly_connected_component_clustering_and_small_budget():
    # Only unidirectional edge: 0 -> 1
    adj = csr_matrix(([1.0], ([0], [1])), shape=(2, 2), dtype=np.float32)
    # Node 1 has higher specificity than Node 0, so candidate_indices will visit 1 before 0!
    specificity = np.array([5.0, 10.0], dtype=np.float32)
    symbols = {
        0: ASTContextSymbol(node_id=0, name="caller_fn", file_path="pkg/mod.py", start_line=1, end_line=5, code="def caller_fn(): pass", token_count=10),
        1: ASTContextSymbol(node_id=1, name="callee_fn", file_path="pkg/mod.py", start_line=6, end_line=10, code="def callee_fn(): pass", token_count=10),
    }

    # Tight token budget (smaller than 150)
    packed, prompt = pack_context_subgraphs(
        symbols=symbols,
        specificity_scores=specificity,
        adjacency_matrix=adj,
        token_budget=60,
    )

    packed_ids = {s.node_id for s in packed}
    # Unidirectional edge 0 -> 1 should still cluster 0 and 1 together
    assert 1 in packed_ids
    assert 0 in packed_ids


def test_matrix_validation_and_null_safe_mmap_close():
    import pytest

    # Null-safe close_mmap_csr with None
    close_mmap_csr(None, None)

    # Empty matrix
    t_mat, dang = build_static_transition_matrix(0, [], [])
    assert t_mat.shape == (0, 0)
    assert len(dang) == 0

    # Negative lambdas validation
    with pytest.raises(ValueError, match="must be non-negative"):
        build_static_transition_matrix(2, [(0, 1)], [], lambda_call=-1.0)

    # Canonical CSR indices check
    t_mat, dang = build_static_transition_matrix(3, [(0, 2), (0, 1)], [])
    assert t_mat.has_sorted_indices
    assert t_mat.has_canonical_format


def test_packer_class_indentation_and_negative_budget():
    # Test method body with 0 initial indentation placed in a class
    symbols = {
        0: ASTContextSymbol(
            node_id=0,
            name="run",
            file_path="src/worker.py",
            start_line=10,
            end_line=12,
            code="def run(self):\n    return True",  # 0 indentation on first line
            token_count=15,
            parent_class="Worker",
            class_docstring="Worker class.",
        )
    }

    packed, prompt = pack_context_subgraphs(
        symbols=symbols,
        specificity_scores=np.array([10.0], dtype=np.float32),
        adjacency_matrix=csr_matrix((1, 1), dtype=np.float32),
        token_budget=100,
    )

    assert len(packed) == 1
    assert "class Worker:" in prompt
    assert "    def run(self):" in prompt
    assert "        return True" in prompt

    # Test non-positive token budgets
    p_zero, pr_zero = pack_context_subgraphs(symbols, np.array([10.0]), csr_matrix((1, 1)), token_budget=0)
    assert len(p_zero) == 0
    assert pr_zero == ""

    p_neg, pr_neg = pack_context_subgraphs(symbols, np.array([10.0]), csr_matrix((1, 1)), token_budget=-50)
    assert len(p_neg) == 0
    assert pr_neg == ""


def test_tester_nonexistent_file_and_exit_code_diagnostics():
    res = run_targeted_test("nonexistent_test_path_12345.py")
    assert not res.passed
    assert res.exit_code == 4
    assert "not found" in res.output
    assert not res.timed_out


def test_editor_python310_match_case_with_bounding_newlines():
    match_code = (
        "def handle_event(event):\n"
        "    match event:\n"
        "        case {'type': 'click', 'pos': (x, y)}:\n"
        "            return x + y\n"
        "        case {'type': 'keypress', 'key': key}:\n"
        "            return key\n"
        "        case _:\n"
        "            return None\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "handler.py"
        test_file.write_text(match_code, encoding="utf-8")

        # Multi-line edit with triple-quote bounding newlines
        old_str = (
            "\n"
            "        case {'type': 'click', 'pos': (x, y)}:\n"
            "            return x + y\n"
        )
        new_str = (
            "\n"
            "        case {'type': 'click', 'pos': (x, y)}:\n"
            "            return (x + y) * 2\n"
        )

        ok, msg = apply_symbol_edit(
            file_path=test_file,
            old_str=old_str,
            new_str=new_str,
            start_line=3,
            end_line=5,
        )
        assert ok, msg
        updated = test_file.read_text(encoding="utf-8")
        tree = ast.parse(updated)
        assert tree is not None
        assert "return (x + y) * 2" in updated
        assert "        case {'type': 'click'" in updated


def test_editor_async_def_and_stacked_decorators():
    source = (
        "class AsyncProcessor:\n"
        "    @classmethod\n"
        "    @wrap_handler(timeout=10)\n"
        "    async def stream_data(cls, items):\n"
        "        async for it in items:\n"
        "            yield it\n"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "async_proc.py"
        test_file.write_text(source, encoding="utf-8")

        old_str = "        async for it in items:\n            yield it"
        new_str = "        async for it in items:\n            yield it.upper()"

        ok, msg = apply_symbol_edit(
            file_path=test_file,
            old_str=old_str,
            new_str=new_str,
            start_line=4,
            end_line=6,
        )
        assert ok, msg
        content = test_file.read_text(encoding="utf-8")
        ast.parse(content)
        assert "yield it.upper()" in content


def test_editor_inverted_line_range_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"
        test_file.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")

        ok, msg = apply_symbol_edit(
            file_path=test_file,
            old_str="y = 2",
            new_str="y = 42",
            start_line=10,
            end_line=2,
        )
        assert not ok
        assert "Invalid line range" in msg


def test_packer_matrix_shape_mismatch():
    import pytest

    symbols = {0: ASTContextSymbol(0, "fn", "f.py", 1, 5, "def fn(): pass", 10)}
    spec = np.array([1.0, 2.0], dtype=np.float32)
    adj = csr_matrix((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        pack_context_subgraphs(symbols, spec, adj, token_budget=100)


def test_matrix_negative_num_nodes_validation():
    import pytest

    with pytest.raises(ValueError, match="num_nodes must be non-negative"):
        build_static_transition_matrix(-5, [], [])


def test_diffusion_nan_parameter_validation():
    import pytest

    # NaN tau
    with pytest.raises(ValueError, match="tau must be strictly positive"):
        compute_softmax_teleport_prior([0.5], [0], num_nodes=2, tau=float("nan"))

    # Non-positive max_iter
    t_mat = csr_matrix(np.eye(2, dtype=np.float32))
    dang = np.zeros(2, dtype=np.float32)
    p_0 = np.array([0.5, 0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="max_iter must be strictly positive"):
        personalized_pagerank_power_iteration(t_mat, dang, p_0, max_iter=0)

    # NaN or non-positive tol
    with pytest.raises(ValueError, match="tol must be strictly positive"):
        personalized_pagerank_power_iteration(t_mat, dang, p_0, tol=float("nan"))


def test_specificity_nan_parameter_validation():
    import pytest

    pi = np.array([0.5, 0.5], dtype=np.float32)

    with pytest.raises(ValueError, match="epsilon must be strictly positive"):
        calculate_specificity_scores(pi, pi, epsilon=float("nan"))

    with pytest.raises(ValueError, match="gamma must be non-negative"):
        calculate_specificity_scores(pi, pi, gamma=float("nan"))


def test_bipartite_diffusion_spectral_gap_and_conservation():
    # Pure 2-node bipartite cycle
    t_mat = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
    dangling = np.zeros(2, dtype=np.float32)
    p_0 = np.array([1.0, 0.0], dtype=np.float32)

    pi = personalized_pagerank_power_iteration(
        t_matrix=t_mat,
        dangling=dangling,
        p_0=p_0,
        beta=0.85,
        max_iter=100,
    )

    # Sum of probability vector is strictly conserved at 1.0
    assert np.isclose(np.sum(pi), 1.0)
    assert (pi >= 0.0).all()
    # Analytical solution: pi_0 = 1 / (1 + beta) = 0.540541, pi_1 = beta / (1 + beta) = 0.459459
    assert np.isclose(pi[0], 1.0 / (1.0 + 0.85), atol=1e-3)
    assert np.isclose(pi[1], 0.85 / (1.0 + 0.85), atol=1e-3)


def test_tester_async_pytest_support():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_async_suite.py"
        test_file.write_text(
            "import pytest\n"
            "@pytest.mark.asyncio\n"
            "async def test_coroutine_pass():\n"
            "    assert 42 == 42\n",
            encoding="utf-8",
        )

        res = run_targeted_test(test_file, timeout_seconds=10.0)
        assert res.passed, f"Async test failed: {res.output}"
        assert res.exit_code == 0


def test_concurrent_mmap_reads_thread_safety():
    import concurrent.futures

    with tempfile.TemporaryDirectory() as tmpdir:
        n = 50
        call_edges = [(i, (i + 1) % n) for i in range(n)]
        t, dang = build_static_transition_matrix(n, call_edges, [])
        n2id = {f"n_{i}": i for i in range(n)}
        id2n = {i: f"n_{i}" for i in range(n)}
        save_mmap_csr(tmpdir, t, dang, n2id, id2n)

        errors = []

        def worker():
            try:
                for _ in range(10):
                    loaded_t, loaded_d, _, _ = load_mmap_csr(tmpdir)
                    v = np.ones(n, dtype=np.float32) / n
                    res = v @ loaded_t
                    assert np.isclose(np.sum(res), 1.0)
                    close_mmap_csr(loaded_t, loaded_d)
            except Exception as e:
                errors.append(str(e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(worker) for _ in range(4)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0, f"Concurrent mmap read errors: {errors}"


def test_editor_exact_lf_and_crlf_preservation():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Unix LF preservation on any OS
        test_lf = Path(tmpdir) / "unix_lf.py"
        test_lf.write_bytes(b"def run():\n    return True\n")
        ok, msg = apply_symbol_edit(test_lf, "return True", "return False")
        assert ok, msg
        raw_lf = test_lf.read_bytes()
        assert b"\r\n" not in raw_lf
        assert raw_lf == b"def run():\n    return False\n"

        # 2. Windows CRLF preservation without double-CR
        test_crlf = Path(tmpdir) / "win_crlf.py"
        test_crlf.write_bytes(b"def run():\r\n    return True\r\n")
        ok_cr, msg_cr = apply_symbol_edit(test_crlf, "return True", "return False")
        assert ok_cr, msg_cr
        raw_crlf = test_crlf.read_bytes()
        assert b"\r\r\n" not in raw_crlf
        assert raw_crlf == b"def run():\r\n    return False\r\n"


def test_packer_parameter_validation():
    import pytest

    symbols = {0: ASTContextSymbol(0, "fn", "f.py", 1, 5, "def fn(): pass", 10)}
    spec = np.array([1.0], dtype=np.float32)
    adj = csr_matrix((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="mu must be non-negative"):
        pack_context_subgraphs(symbols, spec, adj, mu=-0.1)

    with pytest.raises(ValueError, match="top_candidates_limit must be strictly positive"):
        pack_context_subgraphs(symbols, spec, adj, top_candidates_limit=0)


def test_diffusion_duplicate_node_indices_deduplication():
    # Duplicate node 1 with different similarity scores
    sims = [0.9, 0.4]
    nodes = [1, 1]
    p_0 = compute_softmax_teleport_prior(sims, nodes, num_nodes=3, tau=0.05)

    assert np.isclose(np.sum(p_0), 1.0)
    assert p_0[1] == 1.0
    assert p_0[0] == 0.0
    assert p_0[2] == 0.0


def test_adaptive_power_iteration_convergence():
    import pytest
    # Graph with cycle requiring multiple iterations
    t_mat = csr_matrix(np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32))
    dang = np.zeros(4, dtype=np.float32)
    p_0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 1. Converges cleanly with max_iter=90
    pi = personalized_pagerank_power_iteration(t_mat, dang, p_0, beta=0.85, max_iter=90, tol=1e-6)
    assert np.isclose(np.sum(pi), 1.0)
    assert pi[0] > pi[1] > pi[2] > pi[3]

    # 2. Emits RuntimeWarning if max_iter is too low to converge
    with pytest.warns(RuntimeWarning, match="without converging"):
        pi_warn = personalized_pagerank_power_iteration(t_mat, dang, p_0, beta=0.85, max_iter=2, tol=1e-8)
        assert np.isclose(np.sum(pi_warn), 1.0)


def test_multi_file_patch_atomic_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "mod1.py"
        f2 = Path(tmpdir) / "mod2.py"

        f1.write_text("def fn1():\n    return 1\n", encoding="utf-8")
        f2.write_text("def fn2():\n    return 2\n", encoding="utf-8")

        edits = [
            SymbolEditSpec(
                file_path=f1,
                old_str="return 1",
                new_str="return 10",
                start_line=2,
                end_line=2,
            ),
            SymbolEditSpec(
                file_path=f2,
                old_str="return 2",
                new_str="return 20",
                start_line=2,
                end_line=2,
            ),
        ]

        success, msg, modified = apply_multi_file_patch(edits)
        assert success
        assert len(modified) == 2
        assert "return 10" in f1.read_text(encoding="utf-8")
        assert "return 20" in f2.read_text(encoding="utf-8")
        ast.parse(f1.read_text(encoding="utf-8"))
        ast.parse(f2.read_text(encoding="utf-8"))


def test_multi_file_patch_rollback_on_syntax_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "mod1.py"
        f2 = Path(tmpdir) / "mod2.py"

        orig_f1 = "def fn1():\n    return 1\n"
        orig_f2 = "def fn2():\n    return 2\n"
        f1.write_text(orig_f1, encoding="utf-8")
        f2.write_text(orig_f2, encoding="utf-8")

        # Edit 1 is valid, but Edit 2 injects a fatal syntax error
        edits = [
            SymbolEditSpec(
                file_path=f1,
                old_str="return 1",
                new_str="return 10",
                start_line=2,
                end_line=2,
            ),
            SymbolEditSpec(
                file_path=f2,
                old_str="return 2",
                new_str="def broken(: return syntax_error",
                start_line=2,
                end_line=2,
            ),
        ]

        success, msg, modified = apply_multi_file_patch(edits)
        assert not success
        assert "AST syntax error" in msg
        assert len(modified) == 0

        # CRITICAL ATOMICITY CHECK: file 1 must NOT be modified on disk!
        assert f1.read_text(encoding="utf-8") == orig_f1
        assert f2.read_text(encoding="utf-8") == orig_f2


def test_multi_file_patch_rollback_on_missing_target():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = Path(tmpdir) / "mod1.py"
        f2 = Path(tmpdir) / "mod2.py"

        orig_f1 = "def fn1():\n    return 1\n"
        orig_f2 = "def fn2():\n    return 2\n"
        f1.write_text(orig_f1, encoding="utf-8")
        f2.write_text(orig_f2, encoding="utf-8")

        edits = [
            SymbolEditSpec(
                file_path=f1,
                old_str="return 1",
                new_str="return 10",
                start_line=2,
                end_line=2,
            ),
            SymbolEditSpec(
                file_path=f2,
                old_str="non_existent_target_string",
                new_str="return 20",
                start_line=2,
                end_line=2,
            ),
        ]

        success, msg, modified = apply_multi_file_patch(edits)
        assert not success
        assert "Could not locate target string" in msg
        assert len(modified) == 0

        # Invariant: Neither file modified on disk
        assert f1.read_text(encoding="utf-8") == orig_f1
        assert f2.read_text(encoding="utf-8") == orig_f2


def test_multi_file_patch_with_dict_specs():
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / "dict_test.py"
        f.write_text("def test():\n    x = 1\n    return x\n", encoding="utf-8")

        dict_edits = [
            {
                "file_path": str(f),
                "old_str": "x = 1",
                "new_str": "x = 100",
                "start_line": 2,
                "end_line": 2,
            }
        ]

        success, msg, modified = apply_multi_file_patch(dict_edits)
        assert success
        assert len(modified) == 1
        assert "x = 100" in f.read_text(encoding="utf-8")


def test_packer_bidirectional_ancestor_reachability():
    """
    Verifies that callee anchors can traverse in-edges to reach caller ancestors
    in the same connected component, preventing orphan nodes.
    """
    from scipy.sparse import csr_matrix
    from perron.packer import ASTContextSymbol, pack_context_subgraphs

    # Directed call chain: 0 -> 1 -> 2 -> 3 -> 4
    # Node 4 has highest specificity (callee anchor)
    # Nodes 0, 1, 2, 3 are caller ancestors
    symbols = {
        0: ASTContextSymbol(0, "caller_root", "app.py", 1, 5, "def caller_root(): pass", token_count=20),
        1: ASTContextSymbol(1, "caller_sub1", "app.py", 6, 10, "def caller_sub1(): pass", token_count=20),
        2: ASTContextSymbol(2, "caller_sub2", "app.py", 11, 15, "def caller_sub2(): pass", token_count=20),
        3: ASTContextSymbol(3, "caller_direct", "app.py", 16, 20, "def caller_direct(): pass", token_count=20),
        4: ASTContextSymbol(4, "callee_anchor", "app.py", 21, 25, "def callee_anchor(): pass", token_count=20),
    }

    row = np.array([0, 1, 2, 3])
    col = np.array([1, 2, 3, 4])
    data = np.ones(4, dtype=np.float32)
    adj = csr_matrix((data, (row, col)), shape=(5, 5))

    # Node 4 is highest specificity, followed by ancestors
    specificity = np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)

    # Token budget ample for all 5 nodes
    packed, prompt = pack_context_subgraphs(
        symbols=symbols,
        specificity_scores=specificity,
        adjacency_matrix=adj,
        token_budget=500,
    )

    packed_ids = {s.node_id for s in packed}
    # Invariant: Bidirectional expansion must reach caller ancestors 0, 1, 2, 3
    assert 4 in packed_ids, "Anchor callee must be packed"
    assert 3 in packed_ids, "Immediate caller must be packed via reverse edge"
    assert 2 in packed_ids, "2-hop caller ancestor must be packed"
    assert 1 in packed_ids, "3-hop caller ancestor must be packed"
    assert 0 in packed_ids, "Root caller ancestor must be packed"


def test_packer_nested_class_breadcrumbs():
    """
    Verifies that nested class parent names (e.g. Outer.Inner) format
    as valid, parseable Python syntax.
    """
    from perron.packer import ASTContextSymbol, format_hierarchical_context

    symbols = [
        ASTContextSymbol(
            node_id=0,
            name="method",
            file_path="models.py",
            start_line=10,
            end_line=12,
            code="def method(self):\n    return True\n",
            token_count=15,
            parent_class="Outer.Inner",
            class_docstring="Inner class docstring.",
        )
    ]

    prompt = format_hierarchical_context(symbols)
    assert "class Outer:" in prompt
    assert "class Inner:" in prompt
    # Invariant: Output must be valid, executable Python syntax parseable by ast.parse()
    # Strip comment header lines before parsing
    code_lines = [l for l in prompt.splitlines() if not l.startswith("#")]
    valid_source = "\n".join(code_lines) + "\n"
    tree = ast.parse(valid_source)
    assert len(tree.body) > 0


def test_editor_tab_indentation():
    """
    Verifies search-and-replace on tab-indented files without TabError.
    """
    source = "def foo():\n\tx = 1\n\treturn x\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "tabs.py"
        test_file.write_text(source, encoding="utf-8")

        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str="x = 1",
            new_str="x = 2",
            start_line=2,
            end_line=2,
        )

        assert success, msg
        content = test_file.read_text(encoding="utf-8")
        # Invariant: Must not raise TabError
        tree = ast.parse(content)
        assert "\tx = 2" in content
        assert "    x = 2" not in content  # pure tabs preserved


def test_tester_pytest_nodeid_support():
    """
    Verifies run_targeted_test support for Pytest NodeID syntax.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_selector.py"
        test_file.write_text(
            "def test_target_pass():\n    assert 1 == 1\n\n"
            "def test_target_fail():\n    assert 1 == 2\n",
            encoding="utf-8",
        )

        # Run only the passing test via NodeID syntax
        node_id = f"{test_file}::test_target_pass"
        result = run_targeted_test(test_file=node_id, cwd=tmpdir, timeout_seconds=10.0)

        assert result.passed, f"Expected pass, got output: {result.output}"
        assert result.exit_code == 0
        assert "test_target_pass PASSED" in result.output
        assert "test_target_fail" not in result.output


def test_close_mmap_csr_exported_and_null_safe():
    """
    Verifies close_mmap_csr is exported from top-level package and null-safe.
    """
    from perron import close_mmap_csr
    # Should not raise exception
    close_mmap_csr(None, None)


def test_nested_class_breadcrumbs_valid_ast():
    """
    Verifies that nested class breadcrumbs produce 100% syntactically valid Python
    that ast.parse() can parse without SyntaxError.
    """
    from perron.packer import format_hierarchical_context

    sym = ASTContextSymbol(
        node_id=1,
        name="Outer.Inner.compute",
        file_path="service/engine.py",
        start_line=45,
        end_line=50,
        code="    def compute(self, x):\n        return x * 2\n",
        token_count=20,
        parent_class="Outer.Inner",
        class_docstring="Inner engine container.",
    )

    context_str = format_hierarchical_context([sym])
    assert "class Outer:" in context_str
    assert "    class Inner:" in context_str
    assert "        def compute(self, x):" in context_str

    # Must be valid Python AST
    tree = ast.parse(context_str)
    assert tree is not None


def test_editor_full_file_fallback_on_line_drift():
    """
    Verifies that editor Tier 2 full-file search recovers unique targets
    when LLM-predicted line numbers drift far beyond window_slack.
    """
    code_lines = [f"# Line {i}\n" for i in range(1, 100)]
    code_lines[75] = "def target_function(val):\n"
    code_lines[76] = "    return val + 1\n"
    raw_code = "".join(code_lines)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "drift.py"
        test_file.write_text(raw_code, encoding="utf-8")

        # LLM estimated lines 5-10, but target is at line 76 (drift > 60 lines)
        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str="return val + 1",
            new_str="return val + 42",
            start_line=5,
            end_line=10,
        )

        assert success, msg
        updated = test_file.read_text(encoding="utf-8")
        assert "return val + 42" in updated
        assert ast.parse(updated) is not None


def test_zero_node_global_pagerank_and_specificity():
    """
    Verifies that empty 0-node matrices do not trigger ZeroDivisionError.
    """
    empty_t = csr_matrix((0, 0), dtype=np.float32)
    empty_d = np.empty(0, dtype=np.float32)

    pi_global = compute_global_pagerank(empty_t, empty_d)
    assert len(pi_global) == 0

    spec = calculate_specificity_scores(pi_global, pi_global)
    assert len(spec) == 0


def test_packer_column_0_comment_rebasing():
    """
    Verifies that indentation rebasing handles column-0 comments without double indentation.
    """
    from perron.packer import ASTContextSymbol, format_hierarchical_context

    sym = ASTContextSymbol(
        node_id=0,
        name="do_work",
        file_path="worker.py",
        start_line=1,
        end_line=5,
        code="    def do_work(self):\n# unindented comment\n        return True\n",
        token_count=10,
        parent_class="Worker",
    )

    formatted = format_hierarchical_context([sym])
    assert "class Worker:" in formatted
    assert "def do_work" in formatted
    # Must parse as valid Python AST
    assert ast.parse(formatted) is not None


def test_editor_latin1_fallback_encoding():
    """
    Verifies that files with non-UTF8 Latin-1 characters can be edited cleanly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "latin1.py"
        # Write bytes containing Latin-1 character (0xe9 = é in latin-1)
        test_file.write_bytes(b"# Copyright \xe9\ndef greet():\n    return 'hello'\n")

        success, msg = apply_symbol_edit(
            file_path=test_file,
            old_str="return 'hello'",
            new_str="return 'hello world'",
        )

        assert success, msg
        content = test_file.read_text(encoding="latin-1")
        assert "return 'hello world'" in content


def test_multi_edit_reverse_line_order_no_drift():
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "multi_func.py"
        target.write_text(
            "def func_a():\n"
            "    return 1\n\n"
            "def func_b():\n"
            "    return 2\n\n"
            "def func_c():\n"
            "    return 3\n",
            encoding="utf-8"
        )

        edits = [
            SymbolEditSpec(
                file_path=str(target),
                old_str="return 1",
                new_str="return 10",
                start_line=1,
                end_line=3,
                add_imports=["import math"],
            ),
            SymbolEditSpec(
                file_path=str(target),
                old_str="return 3",
                new_str="return 30",
                start_line=7,
                end_line=9,
                add_imports=["import sys"],
            ),
        ]

        success, msg, modified = apply_multi_file_patch(edits)
        assert success, f"Expected multi-edit to succeed, got: {msg}"
        content = target.read_text(encoding="utf-8")
        assert "return 10" in content
        assert "return 30" in content
        assert "import math" in content
        assert "import sys" in content
        ast.parse(content)


def test_packer_tab_indentation_ast_valid():
    from perron.packer import format_hierarchical_context
    sym = ASTContextSymbol(
        node_id=0,
        name="execute",
        file_path="service.py",
        start_line=2,
        end_line=4,
        code="\tdef execute(self):\n\t\tval = 1\n\t\treturn val\n",
        token_count=15,
        parent_class="Service",
        class_docstring="Core service handler.",
    )
    context = format_hierarchical_context([sym])
    parsed = ast.parse(context)
    assert parsed is not None














