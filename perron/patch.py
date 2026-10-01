"""
Unified diff calculation and patch generation for SWE-bench and autonomous agent benchmarks.
"""

from __future__ import annotations

import difflib
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def compute_git_patch(
    repo_dir: Path,
    base_commit: Optional[str] = "HEAD",
    initial_snapshots: Optional[Dict[Path, bytes]] = None,
) -> str:
    """
    Computes unified git diff of all modified and untracked files relative to base_commit.
    Falls back to difflib if git binary is unavailable or execution fails.
    """
    repo = Path(repo_dir).resolve()
    try:
        # Stage newly created files as intent-to-add, excluding cache and index artifacts
        subprocess.run(
            ["git", "add", "-N", "--", ".", ":(exclude).perron_index*", ":(exclude)__pycache__*", ":(exclude).pytest_cache*"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            check=False,
        )
        cmd = ["git", "diff", base_commit] if base_commit else ["git", "diff"]
        res = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout:
            return res.stdout
    except Exception:
        pass

    # Deterministic fallback via difflib if initial snapshots are provided
    if initial_snapshots:
        orig_files: Dict[Path, str] = {}
        mod_files: Dict[Path, str] = {}
        for p, snap in initial_snapshots.items():
            orig_files[p] = snap.decode("utf-8", errors="replace")
            if p.is_file():
                mod_files[p] = p.read_text(encoding="utf-8", errors="replace")
            else:
                mod_files[p] = ""
        diff = generate_unified_diff(orig_files, mod_files, repo_root=repo)
        if diff:
            return diff

    return ""


def generate_unified_diff(
    original_files: Dict[Path, str],
    modified_files: Dict[Path, str],
    repo_root: Optional[Path] = None,
) -> str:
    """
    Generates a unified diff string from dictionaries of file contents.
    """
    diff_lines: List[str] = []
    all_paths = sorted(set(original_files.keys()).union(set(modified_files.keys())))

    for path in all_paths:
        orig = original_files.get(path, "")
        mod = modified_files.get(path, "")
        if orig == mod:
            continue

        rel_path = (
            str(path.relative_to(repo_root)).replace("\\", "/")
            if repo_root and repo_root in path.parents
            else str(path).replace("\\", "/")
        )
        orig_lines = orig.splitlines(keepends=True)
        mod_lines = mod.splitlines(keepends=True)

        hunk = difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="\n",
        )
        diff_lines.extend(hunk)

    return "".join(diff_lines)
