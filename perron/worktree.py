"""
Perron Ephemeral Git Worktree Context Manager.
Provides isolated ephemeral git worktrees for safe evaluation and testing.
"""

from __future__ import annotations

from perron.tester import EphemeralWorktree

__all__ = ["EphemeralWorktree"]
