"""
Tests for perron.graph real AST repository extraction and zero-copy CSR compilation.
"""

import ast
import tempfile
from pathlib import Path
import numpy as np
import pytest

from perron import (
    ASTSymbolNode,
    extract_repository_graph,
    compile_and_save_repository_graph,
    load_mmap_csr,
    close_mmap_csr,
    compute_global_pagerank,
)


def test_graph_composite_keying_zero_symbol_loss():
    """
    Verifies that identical function names across different files are never dropped.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "pkg").mkdir()
        (root / "pkg" / "mod1.py").write_text("def init():\n    return 'mod1'\n", encoding="utf-8")
        (root / "pkg" / "mod2.py").write_text("def init():\n    return 'mod2'\n", encoding="utf-8")

        symbols, call_edges, caller_edges, can_to_id, id_to_can = extract_repository_graph(root)

        # Invariant: Both init() functions must be extracted
        init_symbols = [s for s in symbols if s.qualified_name == "init"]
        assert len(init_symbols) == 2
        file_paths = {s.file_path for s in init_symbols}
        assert file_paths == {"pkg/mod1.py", "pkg/mod2.py"}
        assert len(can_to_id) == 2


def test_graph_verbatim_code_slicing():
    """
    Verifies that AST symbols retain their verbatim source code slices rather than dummy stubs.
    """
    source = (
        "class Calculator:\n"
        "    '''A simple calculator.'''\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        # add implementation\n"
        "        return a + b\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "calc.py").write_text(source, encoding="utf-8")

        symbols, _, _, _, _ = extract_repository_graph(root)
        method_sym = next(s for s in symbols if s.qualified_name == "Calculator.add")

        assert "return a + b" in method_sym.code
        assert "# add implementation" in method_sym.code
        assert "pass" not in method_sym.code
        assert method_sym.parent_class == "Calculator"


def test_graph_import_aware_call_resolution():
    """
    Verifies that calls across files are resolved via import tables.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "utils.py").write_text(
            "def helper(x):\n    return x * 2\n",
            encoding="utf-8"
        )
        (root / "app.py").write_text(
            "from utils import helper\n\n"
            "def run():\n"
            "    val = helper(10)\n"
            "    return val\n",
            encoding="utf-8"
        )

        symbols, call_edges, caller_edges, can_to_id, id_to_can = extract_repository_graph(root)

        run_id = can_to_id["app.py::run"]
        helper_id = can_to_id["utils.py::helper"]

        # Call edge from run -> helper
        assert (run_id, helper_id) in call_edges
        # Caller edge from helper -> run
        assert (helper_id, run_id) in caller_edges


def test_compile_and_save_repository_graph_mmap_roundtrip():
    """
    Verifies end-to-end compilation, binary serialization, and zero-copy mmap loading.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        out_dir = Path(tmpdir) / "graph_out"

        (repo_dir / "service.py").write_text(
            "class Service:\n"
            "    def execute(self):\n"
            "        return self.step()\n"
            "    def step(self):\n"
            "        return 1\n",
            encoding="utf-8"
        )

        t_matrix, dangling, can_to_id, id_to_can, syms = compile_and_save_repository_graph(
            repo_dir, out_dir
        )

        assert t_matrix.shape[0] == len(syms)
        assert len(can_to_id) == len(syms)

        # Load back via zero-copy mmap
        loaded_t, loaded_d, loaded_node_to_id, loaded_id_to_node = load_mmap_csr(out_dir)
        assert loaded_t.shape == t_matrix.shape
        assert len(loaded_node_to_id) == len(syms)

        # Run global pagerank on loaded matrix
        pi_global = compute_global_pagerank(loaded_t, loaded_d)
        assert np.isclose(np.sum(pi_global), 1.0, atol=1e-5)

        close_mmap_csr(loaded_t, loaded_d)


def test_graph_decorator_slicing_and_start_line():
    """
    Verifies that decorated classes and functions retain their decorators in code slices.
    """
    source = (
        "class Model:\n"
        "    @classmethod\n"
        "    def create(cls, name):\n"
        "        return cls(name)\n\n"
        "    @property\n"
        "    def label(self):\n"
        "        return 'label'\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "model.py").write_text(source, encoding="utf-8")

        symbols, _, _, can_to_id, _ = extract_repository_graph(root)
        create_sym = next(s for s in symbols if s.qualified_name == "Model.create")
        label_sym = next(s for s in symbols if s.qualified_name == "Model.label")

        assert "@classmethod" in create_sym.code
        assert create_sym.start_line == 2  # Decorator line
        assert "classmethod" in create_sym.decorators

        assert "@property" in label_sym.code
        assert label_sym.start_line == 6  # Decorator line
        assert "property" in label_sym.decorators


def test_matches_module_path_prevents_substring_collision():
    """
    Verifies that import matching does not cause false-positive substring collisions.
    """
    from perron.graph import _matches_module_path

    # False-positive substring candidates that must be rejected
    assert not _matches_module_path("audio/player.py", "io")
    assert not _matches_module_path("bios/driver.py", "os")
    assert not _matches_module_path("actions/io_handler.py", "io")

    # True positive module matches that must be accepted
    assert _matches_module_path("io.py", "io")
    assert _matches_module_path("lib/io.py", "io")
    assert _matches_module_path("io/__init__.py", "io")
    assert _matches_module_path("perron/backends/base.py", "perron/backends/base")
    assert _matches_module_path("perron/backends/base.py", "perron/backends")


def test_graph_cross_class_method_isolation(tmp_path: Path):
    """
    Verifies that calling foreign object methods (e.g. db.connect()) inside
    a class method does not hijack edges into ClassName.connect().
    """
    src = (
        "class Database:\n"
        "    def connect(self):\n"
        "        pass\n\n"
        "class NetworkClient:\n"
        "    def connect(self):\n"
        "        pass\n"
        "    def fetch(self, db):\n"
        "        db.connect()\n"
        "        self.connect()\n"
    )
    test_file = tmp_path / "client.py"
    test_file.write_text(src, encoding="utf-8")

    symbols, call_edges, caller_edges, can_to_id, _ = extract_repository_graph(tmp_path)
    sym_by_name = {s.qualified_name: s.node_id for s in symbols}

    fetch_id = sym_by_name["NetworkClient.fetch"]
    client_conn_id = sym_by_name["NetworkClient.connect"]
    db_conn_id = sym_by_name["Database.connect"]

    # (fetch_id, client_conn_id) should exist because of self.connect()
    assert (fetch_id, client_conn_id) in call_edges

    # db.connect() should NOT be resolved to NetworkClient.connect()
    # Count how many calls from fetch_id to client_conn_id: should be exactly 1, not 2!
    client_calls = [v for u, v in call_edges if u == fetch_id and v == client_conn_id]
    assert len(client_calls) == 1, "db.connect() was hijacked into NetworkClient.connect()"


def test_extract_repository_graph_with_max_files(tmp_path: Path):
    """
    Verifies that max_files slices file traversal to exact file budget.
    """
    for i in range(10):
        (tmp_path / f"mod_{i}.py").write_text(f"def func_{i}(): return {i}\n", encoding="utf-8")

    syms, _, _, _, _ = extract_repository_graph(tmp_path, max_files=3)
    # Should only have extracted symbols from 3 files
    unique_files = {s.file_path for s in syms}
    assert len(unique_files) == 3


def test_extract_repository_graph_multiplex_mro(tmp_path: Path):
    """
    Verifies that extract_repository_graph with return_multiplex=True extracts
    MRO inheritance and import edges across classes and modules.
    """
    base_src = (
        "class BaseHandler:\n"
        "    def execute(self):\n"
        "        return 'base'\n"
    )
    derived_src = (
        "from base import BaseHandler\n\n"
        "class DerivedHandler(BaseHandler):\n"
        "    def run(self):\n"
        "        return self.execute()\n"
    )
    (tmp_path / "base.py").write_text(base_src, encoding="utf-8")
    (tmp_path / "derived.py").write_text(derived_src, encoding="utf-8")

    res = extract_repository_graph(tmp_path, return_multiplex=True)
    assert len(res) == 7
    symbols, call_edges, caller_edges, can_to_id, id_to_can, inherit_edges, import_edges = res

    run_id = can_to_id["derived.py::DerivedHandler.run"]
    exec_id = can_to_id["base.py::BaseHandler.execute"]
    assert (run_id, exec_id) in inherit_edges
    assert (run_id, exec_id) not in call_edges
    assert (exec_id, run_id) not in caller_edges
    # Strict transpose duality: caller_edges must exactly match reversed call_edges
    assert set(caller_edges) == {(v, u) for (u, v) in call_edges}


def test_inheritance_edges_do_not_leak_into_call_edges(tmp_path: Path):
    """
    Verifies that class inheritance edges (Base -> Derived) populate
    exclusively inherit_edges and do not leak into call_edges or caller_edges.
    """
    src = (
        "class Base:\n"
        "    pass\n\n"
        "class Derived(Base):\n"
        "    pass\n"
    )
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")

    res = extract_repository_graph(tmp_path, return_multiplex=True)
    symbols, call_edges, caller_edges, can_to_id, _, inherit_edges, _ = res

    derived_id = can_to_id["mod.py::Derived"]
    base_id = can_to_id["mod.py::Base"]

    # Class-level inheritance edge must be in inherit_edges
    assert (derived_id, base_id) in inherit_edges

    # Inheritance MUST NOT be in call_edges or caller_edges
    assert (derived_id, base_id) not in call_edges
    assert (base_id, derived_id) not in caller_edges
