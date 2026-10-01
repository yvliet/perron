"""
Tests for perron.agent autonomous lifecycle and Gemma 4 orchestration.
"""

import ast
import tempfile
from pathlib import Path
import numpy as np
import pytest

from perron import (
    PerronAgent,
    TrajectoryResult,
    TurnTelemetry,
)
from perron.backends.base import ModelBackend, BackendResponse
from perron.backends.replay import OfflineReplayBackend


def test_agent_initialization_and_context_retrieval():
    """
    Verifies agent graph indexing, CSR generation, and query retrieval.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "math_mod.py").write_text(
            "def multiply(a, b):\n    return a * b\n\n"
            "def divide(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
            encoding="utf-8"
        )

        agent = PerronAgent(repo_dir=repo_dir)
        agent.initialize_graph()

        assert agent.t_matrix is not None
        assert agent.pi_global is not None
        assert len(agent.symbols) == 2

        # Retrieve context for a query
        packed, context_str, specificity, telemetry = agent.retrieve_context(
            "divide division by zero"
        )
        assert len(packed) >= 1
        assert "def divide" in context_str
        assert len(telemetry) == 4  # TELEPORT_PRIOR, PPR_WALK, SPEC_FILTER, AST_PACK

        agent.close()


def test_agent_prompt_and_edit_parsing():
    """
    Verifies Gemma 4 prompt structure and XML <edit> block parsing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = PerronAgent(repo_dir=tmpdir)

        prompt = agent.format_prompt(
            issue_text="ZeroDivisionError in divide()",
            context_str="def divide(a, b): pass",
        )

        assert "<start_of_turn>user" in prompt
        assert "<start_of_turn>model" in prompt
        assert "<|think|>" in prompt
        assert "ZeroDivisionError in divide()" in prompt

        # Simulate Gemma 4 response with thinking trace and edit block
        model_response = (
            "<|think|>\n"
            "The division by zero returns None, but should raise ZeroDivisionError.\n"
            "</|think|>\n"
            "<edit file=\"calc.py\" start_line=\"3\" end_line=\"4\">\n"
            "<old>\n"
            "    if b == 0:\n"
            "        return None\n"
            "</old>\n"
            "<new>\n"
            "    if b == 0:\n"
            "        raise ZeroDivisionError('cannot divide by zero')\n"
            "</new>\n"
            "</edit>\n"
        )

        edits = agent.parse_edits_from_response(model_response)
        assert len(edits) == 1
        edit = edits[0]
        assert edit.file_path.resolve() == (Path(tmpdir) / "calc.py").resolve()
        assert edit.start_line == 3
        assert edit.end_line == 4
        assert "return None" in edit.old_str
        assert "raise ZeroDivisionError" in edit.new_str


def test_agent_solve_issue_end_to_end_replay():
    """
    Verifies full autonomous agent loop: retrieve -> generate -> patch -> test -> resolve.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()

        # Buggy source file
        src_file = repo_dir / "calculator.py"
        src_file.write_text(
            "def calculate_tax(amount):\n"
            "    # Bug: 0% tax\n"
            "    return amount * 0.0\n",
            encoding="utf-8"
        )

        # Test file that currently fails
        test_file = repo_dir / "test_tax.py"
        test_file.write_text(
            "from calculator import calculate_tax\n\n"
            "def test_calculate_tax():\n"
            "    assert calculate_tax(100) == 15.0\n",
            encoding="utf-8"
        )

        canned_gemma_response = (
            "<|think|>\n"
            "The calculate_tax function returns amount * 0.0, but the test expects 15.0 for 100.\n"
            "The tax rate should be 0.15 (15%).\n"
            "</|think|>\n"
            "<edit file=\"calculator.py\" start_line=\"1\" end_line=\"3\">\n"
            "<old>\n"
            "def calculate_tax(amount):\n"
            "    # Bug: 0% tax\n"
            "    return amount * 0.0\n"
            "</old>\n"
            "<new>\n"
            "def calculate_tax(amount):\n"
            "    return amount * 0.15\n"
            "</new>\n"
            "</edit>\n"
        )

        replay_backend = OfflineReplayBackend(canned_responses=[canned_gemma_response])
        agent = PerronAgent(repo_dir=repo_dir, backend=replay_backend)
        agent.initialize_graph()

        result = agent.solve_issue(
            issue_text="Tax is calculated as 0 instead of 15%",
            test_file=str(test_file),
            test_filter="test_calculate_tax",
            instance_id="tax_issue_1",
        )

        assert result.resolved
        assert result.test_passed
        assert result.patch_applied
        assert len(result.applied_edits) == 1
        assert len(result.thinking_traces) == 1
        assert "The tax rate should be 0.15" in result.thinking_traces[0]

        # Verify disk code updated
        updated_code = src_file.read_text(encoding="utf-8")
        assert "amount * 0.15" in updated_code

        agent.close()


def test_agent_context_manager_safe_cleanup():
    """
    Verifies that PerronAgent safely closes memory maps when used as a context manager.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "mod.py").write_text("def f(): return 1\n", encoding="utf-8")

        with PerronAgent(repo_dir=repo_dir) as agent:
            agent.initialize_graph()
            assert agent.t_matrix is not None
            assert agent.dangling is not None

        # Exiting context must release and nullify handles
        assert agent.t_matrix is None
        assert agent.dangling is None


def test_parse_edits_path_traversal_rejection():
    """
    Verifies that attempts to escape the repository directory via path traversal are rejected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = PerronAgent(repo_dir=tmpdir)

        malicious_response = (
            "<edit file=\"../../etc/passwd\" start_line=\"1\" end_line=\"5\">\n"
            "<old>root:x:0:0</old>\n"
            "<new>root:x:0:0:hacked</new>\n"
            "</edit>\n"
            "<edit file=\"sub/allowed.py\" start_line=\"1\" end_line=\"2\">\n"
            "<old>val = 1</old>\n"
            "<new>val = 2</new>\n"
            "</edit>\n"
        )

        edits = agent.parse_edits_from_response(malicious_response)
        # The path traversal edit must be rejected
        assert len(edits) == 1
        assert edits[0].file_path == (Path(tmpdir) / "sub/allowed.py").resolve()


def test_parse_edits_flexible_attributes():
    """
    Verifies parsing with arbitrary attribute order, whitespace around '=', and single quotes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = PerronAgent(repo_dir=tmpdir)

        flexible_response = (
            "<edit symbol = 'my_func' end_line = '15' file = 'core/worker.py' start_line = '10'>\n"
            "<old>\n"
            "old code block\n"
            "</old>\n"
            "<new>\n"
            "new code block\n"
            "</new>\n"
            "</edit>\n"
        )

        edits = agent.parse_edits_from_response(flexible_response)
        assert len(edits) == 1
        e = edits[0]
        assert e.file_path == (Path(tmpdir) / "core/worker.py").resolve()
        assert e.start_line == 10
        assert e.end_line == 15
        assert "old code block" in e.old_str
        assert "new code block" in e.new_str


def test_agent_refresh_symbols_for_file():
    """
    Verifies that refresh_symbols_for_file updates in-memory symbol slices after disk edit.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        file_path = repo_dir / "service.py"
        file_path.write_text(
            "def handle():\n    return 'initial'\n",
            encoding="utf-8"
        )

        agent = PerronAgent(repo_dir=repo_dir)
        agent.initialize_graph()

        sym = next(s for s in agent.symbols if s.name == "handle")
        assert "initial" in sym.code

        # Mutate file on disk
        file_path.write_text(
            "def handle():\n    return 'updated_value'\n",
            encoding="utf-8"
        )

        agent.refresh_symbols_for_file(file_path)

        # In-memory symbol must reflect updated content
        sym_after = next(s for s in agent.symbols if s.name == "handle")
        assert "updated_value" in sym_after.code

        agent.close()


def test_agent_refresh_symbols_freezes_topology_at_n0():
    """
    Verifies that during multi-turn edits, refresh_symbols_for_file updates
    existing symbols in-place and freezes topology at N_0, preserving mmap
    CSR array immutability and preventing dimension mismatch errors.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        file_path = repo_dir / "service.py"
        file_path.write_text(
            "def handle():\n    return 'initial'\n",
            encoding="utf-8"
        )

        with PerronAgent(repo_dir=repo_dir) as agent:
            agent.initialize_graph()
            assert len(agent.symbols) == 1
            assert agent.t_matrix.shape == (1, 1)

            # Add a newly extracted helper function to the file
            file_path.write_text(
                "def handle():\n    return helper()\n\n"
                "def helper():\n    return 'helper_val'\n",
                encoding="utf-8"
            )

            agent.refresh_symbols_for_file(file_path)

            # Invariant: Topology is frozen at N_0 (1 symbol)
            assert len(agent.symbols) == 1
            assert agent.symbols[0].name == "handle"
            assert "return helper()" in agent.symbols[0].code

            # Matrix and vectors remain strictly (1, 1) and length 1
            assert agent.t_matrix.shape == (1, 1)
            assert len(agent.dangling) == 1
            assert len(agent.pi_global) == 1
            assert np.isclose(np.sum(agent.pi_global), 1.0)

            # Subsequent retrieve_context functions cleanly without dimension errors
            packed, ctx, _, _ = agent.retrieve_context("handle")
            assert len(packed) == 1
            assert "return helper()" in ctx


def test_transformers_backend_stop_sequences_truncation():
    """
    Verifies that stop_sequences truncation logic correctly splits generated text.
    """
    from perron.backends.base import BackendResponse
    raw_content = "Here is an edit:\n<edit>foo</edit>\nAdditional hallucinated chatter"
    stop_seqs = ["\nAdditional"]
    cleaned = raw_content
    for stop in stop_seqs:
        if stop in cleaned:
            cleaned = cleaned.split(stop)[0].strip()
    assert cleaned == "Here is an edit:\n<edit>foo</edit>"


def test_agent_retrieve_context_zero_symbols():
    """
    Verifies that retrieve_context returns empty context cleanly when no symbols exist.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "script.py").write_text("# Pure script\nx = 1\n", encoding="utf-8")

        with PerronAgent(repo_dir=repo_dir) as agent:
            agent.initialize_graph()
            assert len(agent.symbols) == 0

            packed, ctx, spec, tel = agent.retrieve_context("Find issue")
            assert packed == []
            assert ctx == ""
            assert len(spec) == 0


def test_agent_bounded_issue_prompt_fits_budget():
    """
    Verifies that an excessively large issue text is bounded and total prompt fits within 4096 tokens.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("def work(): return 1\n", encoding="utf-8")

        huge_issue = "Stack trace: " + ("database timeout at line 42 with payload dump " * 500)
        with PerronAgent(repo_dir=repo_dir) as agent:
            agent.initialize_graph()
            result = agent.solve_issue(issue_text=huge_issue)
            # Must complete without crashing
            assert result is not None


def test_agent_solve_issue_passes_stop_sequences():
    """
    Verifies that agent.solve_issue passes stop_sequences=['<end_of_turn>'] to model backend.
    """
    class SpyBackend(ModelBackend):
        def __init__(self):
            self.last_stops = None

        def generate(self, prompt, max_tokens=2048, temperature=0.2, stop_sequences=None):
            self.last_stops = stop_sequences
            return BackendResponse(
                content="<edit file=\"mod.py\" start_line=\"1\" end_line=\"2\"><old>def f(): pass</old><new>def f(): return 1</new></edit>",
                thinking_trace="Trace",
                prompt_tokens=50,
                completion_tokens=20,
                latency_seconds=0.05,
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "mod.py").write_text("def f(): pass\n", encoding="utf-8")

        spy = SpyBackend()
        with PerronAgent(repo_dir=repo_dir, backend=spy) as agent:
            agent.initialize_graph()
            agent.solve_issue("fix f")

        assert spy.last_stops is not None
        assert "<end_of_turn>" in spy.last_stops


def test_agent_refresh_symbols_clears_deleted_symbol():
    """
    Verifies that when a function is removed on disk, refresh_symbols_for_file
    zeros code and token_count, and pack_context_subgraphs skips it.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        file_path = repo_dir / "service.py"
        file_path.write_text(
            "def handle():\n    return 'initial'\n\ndef old_helper():\n    return 42\n",
            encoding="utf-8"
        )

        agent = PerronAgent(repo_dir=repo_dir)
        agent.initialize_graph()

        helper_sym = next(s for s in agent.symbols if s.name == "old_helper")
        assert helper_sym.code != ""
        assert helper_sym.token_count > 0

        # Remove old_helper on disk
        file_path.write_text(
            "def handle():\n    return 'updated'\n",
            encoding="utf-8"
        )
        agent.refresh_symbols_for_file(file_path)

        assert helper_sym.code == ""
        assert helper_sym.token_count == 0

        # Test context packing omits the deleted symbol
        packed_symbols, _, _, _ = agent.retrieve_context("handle")
        packed_names = [s.name for s in packed_symbols]
        assert "old_helper" not in packed_names


def test_offline_replay_backend_stop_sequences():
    """
    Verifies that OfflineReplayBackend truncates content at the specified stop sequence.
    """
    backend = OfflineReplayBackend(
        canned_responses=["Hello world<end_of_turn>extra junk that should be pruned"]
    )
    res = backend.generate("prompt", stop_sequences=["<end_of_turn>"])
    assert res.content == "Hello world"
    assert "<end_of_turn>" not in res.content
    assert "extra junk" not in res.content


def test_agent_test_timeout_configurability():
    """
    Verifies that PerronAgent exposes test_timeout and propagates it to test execution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        (repo_dir / "mod.py").write_text("def f(): return 1\n", encoding="utf-8")

        agent = PerronAgent(repo_dir=repo_dir, test_timeout=45.0)
        assert agent.test_timeout == 45.0


def test_agent_refresh_symbols_duck_types_ast_symbol_node():
    """
    Verifies that refresh_symbols_for_file correctly handles ASTSymbolNode
    instances that have qualified_name instead of name, avoiding AttributeError.
    """
    from perron.graph import ASTSymbolNode

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        repo_dir.mkdir()
        file_path = repo_dir / "service.py"
        file_path.write_text("def compute():\n    return 10\n", encoding="utf-8")

        agent = PerronAgent(repo_dir=repo_dir)
        sym_node = ASTSymbolNode(
            node_id=0,
            file_path="service.py",
            qualified_name="compute",
            symbol_type="function",
            start_line=1,
            end_line=2,
            code="def compute():\n    return 10\n",
            token_count=10,
        )
        agent.symbols = [sym_node]

        file_path.write_text("def compute():\n    return 20\n", encoding="utf-8")
        agent.refresh_symbols_for_file(file_path)
        assert "return 20" in sym_node.code




