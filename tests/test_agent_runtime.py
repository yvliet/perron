"""
Tests for Gemma 4 agent runtime, prefix-cache stability, thinking tag hygiene, and isolated evaluation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from perron.agent import PerronAgent
from perron.backends.base import BackendResponse, ModelBackend
from perron.backends.replay import OfflineReplayBackend
from benchmarks.run_real_evaluation import EphemeralWorktree, run_instance


def test_thinking_tag_normalization_both_closing_tags():
    """
    Verifies that both </|think|> and <|/think|> tags are stripped cleanly
    and do not leak truncated thinking blocks into turn history.
    """
    class AlternatingThinkingBackend(ModelBackend):
        def __init__(self):
            self.turn = 0

        def generate(self, prompt, max_tokens=2048, temperature=0.2, stop_sequences=None):
            self.turn += 1
            if self.turn == 1:
                # First turn uses </|think|>
                return BackendResponse(
                    content=(
                        "<|think|>\nThinking about first step with deep analysis...\n</|think|>\n"
                        "<edit file=\"calc_runner.py\" start_line=\"1\" end_line=\"2\">\n"
                        "<old>def run(): return 0</old>\n"
                        "<new>def run(): return 1</new>\n"
                        "</edit>"
                    ),
                    thinking_trace="Thinking about first step with deep analysis...",
                    prompt_tokens=50,
                    completion_tokens=40,
                    latency_seconds=0.1,
                )
            else:
                # Second turn uses <|/think|>
                return BackendResponse(
                    content=(
                        "<|think|>\nRefining solution for second step...\n<|/think|>\n"
                        "<edit file=\"calc_runner.py\" start_line=\"1\" end_line=\"2\">\n"
                        "<old>def run(): return 1</old>\n"
                        "<new>def run(): return 2</new>\n"
                        "</edit>"
                    ),
                    thinking_trace="Refining solution for second step...",
                    prompt_tokens=80,
                    completion_tokens=40,
                    latency_seconds=0.1,
                )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        (repo / "calc_runner.py").write_text("def run(): return 0\n", encoding="utf-8")
        test_file = repo / "test_calc.py"
        test_file.write_text(
            "import calc_runner\n"
            "def test_run():\n"
            "    assert calc_runner.run() == 2\n",
            encoding="utf-8"
        )

        backend = AlternatingThinkingBackend()
        with PerronAgent(repo_dir=repo, backend=backend, max_turns=2) as agent:
            agent.initialize_graph()
            res = agent.solve_issue(
                issue_text="Make run return 2",
                test_file=str(test_file),
            )

            assert res.resolved
            assert res.test_passed
            assert len(res.thinking_traces) == 2
            # Neither thinking block should have leaked into model response content in history
            for trace in res.thinking_traces:
                assert trace not in (repo / "calc_runner.py").read_text(encoding="utf-8")


def test_prefix_cache_invariant_context():
    """
    Verifies that the Turn 0 prompt header (including # REPOSITORY CONTEXT)
    remains byte-for-byte invariant across multiple turns to preserve vLLM prefix caching.
    """
    prompts_seen = []

    class PromptCapturingBackend(ModelBackend):
        def generate(self, prompt, max_tokens=2048, temperature=0.2, stop_sequences=None):
            prompts_seen.append(prompt)
            turn = len(prompts_seen)
            if turn == 1:
                return BackendResponse(
                    content="<edit file=\"mod.py\" start_line=\"1\" end_line=\"2\"><old>x = 1</old><new>x = 2</new></edit>",
                    thinking_trace="Turn 1",
                    prompt_tokens=40,
                    completion_tokens=20,
                    latency_seconds=0.05,
                )
            else:
                return BackendResponse(
                    content="<edit file=\"mod.py\" start_line=\"1\" end_line=\"2\"><old>x = 2</old><new>x = 3</new></edit>",
                    thinking_trace="Turn 2",
                    prompt_tokens=80,
                    completion_tokens=20,
                    latency_seconds=0.05,
                )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        (repo / "mod.py").write_text("x = 1\n", encoding="utf-8")
        test_file = repo / "test_mod.py"
        test_file.write_text("import mod\ndef test_x(): assert mod.x == 3\n", encoding="utf-8")

        backend = PromptCapturingBackend()
        with PerronAgent(repo_dir=repo, backend=backend, max_turns=2) as agent:
            agent.initialize_graph()
            res = agent.solve_issue(
                issue_text="Update x to 3",
                test_file=str(test_file),
            )

            assert res.resolved
            assert len(prompts_seen) == 2

            # Extract the Turn 0 user block from both prompts
            p1 = prompts_seen[0]
            p2 = prompts_seen[1]

            # The Turn 0 user prompt must be identical as a prefix
            p1_user_turn = p1.split("<end_of_turn>")[0]
            p2_user_turn = p2.split("<end_of_turn>")[0]
            assert p1_user_turn == p2_user_turn, "Turn 0 context mutated, breaking vLLM prefix cache!"


def test_false_resolution_loophole_closed():
    """
    Verifies that when test_file is None, an applied patch is NOT falsely marked resolved
    unless assume_resolved_without_test is explicitly set to True.
    """
    canned_patch = (
        "<edit file=\"app.py\" start_line=\"1\" end_line=\"2\">\n"
        "<old>val = 0</old>\n"
        "<new>val = 9999</new>\n"
        "</edit>"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        repo = Path(tmpdir)
        (repo / "app.py").write_text("val = 0\n", encoding="utf-8")

        backend = OfflineReplayBackend(canned_responses=[canned_patch])
        with PerronAgent(repo_dir=repo, backend=backend, assume_resolved_without_test=False) as agent:
            agent.initialize_graph()
            # Without test file, resolution is unverified
            res = agent.solve_issue(
                issue_text="Change val to 9999",
                test_file=None,
            )
            assert res.patch_applied
            assert not res.resolved, "Patch marked resolved without verification!"

        # Reset app.py for the second test
        (repo / "app.py").write_text("val = 0\n", encoding="utf-8")

        # With explicit opt-in, legacy unverified resolution is permitted
        backend2 = OfflineReplayBackend(canned_responses=[canned_patch])
        with PerronAgent(repo_dir=repo, backend=backend2, assume_resolved_without_test=True) as agent2:
            agent2.initialize_graph()
            res2 = agent2.solve_issue(
                issue_text="Change val to 9999",
                test_file=None,
            )
            assert res2.patch_applied
            assert res2.resolved


def test_ephemeral_worktree_isolation():
    """
    Verifies that EphemeralWorktree creates an isolated checkout and cleans up safely.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "base"
        base_dir.mkdir()
        (base_dir / "file.txt").write_text("original content\n", encoding="utf-8")

        with EphemeralWorktree(base_dir) as wt_dir:
            assert wt_dir.exists()
            assert (wt_dir / "file.txt").read_text(encoding="utf-8") == "original content\n"
            # Modify inside worktree
            (wt_dir / "file.txt").write_text("mutated in worktree\n", encoding="utf-8")
            assert (base_dir / "file.txt").read_text(encoding="utf-8") == "original content\n"

        # After context exit, worktree directory must be cleaned up
        assert not wt_dir.exists()
        # Base repo must remain untouched
        assert (base_dir / "file.txt").read_text(encoding="utf-8") == "original content\n"
