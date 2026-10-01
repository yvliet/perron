"""
Unit tests for perron/patch.py, unified diff calculation, and new file creation.
"""

import ast
import tempfile
from pathlib import Path
from perron.editor import apply_multi_file_patch, _apply_single_edit_to_text, SymbolEditSpec
from perron.patch import generate_unified_diff, compute_git_patch
from perron.agent import PerronAgent
from perron.backends.replay import OfflineReplayBackend


def test_create_new_file_via_multi_file_patch():
    """
    Verifies that apply_multi_file_patch allows creating a new file from scratch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "repo"
        repo.mkdir()
        new_file = repo / "pkg" / "module.py"

        spec = SymbolEditSpec(
            file_path=new_file,
            old_str="",
            new_str="def greet():\n    return 'hello'\n",
        )
        ok, msg, committed = apply_multi_file_patch([spec])
        assert ok
        assert new_file.is_file()
        assert "hello" in new_file.read_text(encoding="utf-8")


def test_docstring_multiline_indentation_preservation():
    """
    Verifies that replacement code blocks containing docstrings with column 0 do not double indent.
    """
    original = "class Service:\n    def execute(self):\n        pass\n"
    new_block = (
        "        '''Execute method.\n"
        "Starts operation.\n"
        "        '''\n"
        "        return True\n"
    )
    ok, modified, msg = _apply_single_edit_to_text(
        raw_content=original,
        old_str="        pass\n",
        new_str=new_block,
        start_line=3,
        end_line=3,
    )
    assert ok
    parsed = ast.parse(modified)
    assert parsed is not None
    assert "                return True" not in modified
    assert "        return True" in modified


def test_generate_unified_diff():
    """
    Verifies unified diff generation from file snapshot dictionaries.
    """
    orig = {Path("lib/core.py"): "val = 1\n"}
    mod = {Path("lib/core.py"): "val = 2\n"}
    diff = generate_unified_diff(orig, mod)
    assert "--- a/lib/core.py" in diff
    assert "+++ b/lib/core.py" in diff
    assert "-val = 1" in diff
    assert "+val = 2" in diff


def test_agent_rollback_on_test_failure():
    """
    Verifies that when rollback_on_test_failure=True, failed turns restore disk state.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir) / "repo"
        repo.mkdir()
        file_path = repo / "core.py"
        original_code = "def compute():\n    return 42\n"
        file_path.write_text(original_code, encoding="utf-8")

        test_path = repo / "test_core.py"
        test_path.write_text("from core import compute\ndef test_compute():\n    assert compute() == 100\n", encoding="utf-8")

        # Turn 1: Proposes broken patch returning 99
        # Turn 2: Proposes patch replacing original 'return 42' with 'return 100'
        canned_turn1 = "<edit file=\"core.py\" start_line=\"2\" end_line=\"2\"><old>    return 42\n</old><new>    return 99\n</new></edit>"
        canned_turn2 = "<edit file=\"core.py\" start_line=\"2\" end_line=\"2\"><old>    return 42\n</old><new>    return 100\n</new></edit>"

        backend = OfflineReplayBackend(canned_responses=[canned_turn1, canned_turn2])

        with PerronAgent(repo_dir=repo, backend=backend, max_turns=2, rollback_on_test_failure=True) as agent:
            agent.initialize_graph()
            result = agent.solve_issue(
                issue_text="Make compute return 100",
                test_file=str(test_path),
            )

        assert result.resolved
        assert "return 100" in file_path.read_text(encoding="utf-8")
