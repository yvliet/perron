"""
Perron Targeted Test Runner Module.

Executes targeted pytest invocations with cross-platform process-group timeout enforcement.
Shields against lingering zombie test processes across Windows and Linux environments.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestResult:
    """Execution outcome of a targeted test run."""
    passed: bool
    exit_code: int
    duration_seconds: float
    timed_out: bool
    output: str


class EphemeralWorktree:
    """
    Context manager that isolates evaluation inside a clean git worktree or
    temporary checkout, ensuring that file edits and test runs never contaminate
    the base repository across benchmark runs.
    """
    def __init__(self, base_repo: Path | str, commit_or_branch: str = "HEAD"):
        self.base_repo = Path(base_repo).resolve()
        self.commit = commit_or_branch
        self.worktree_dir: Optional[Path] = None
        self._is_git = (self.base_repo / ".git").exists() or (self.base_repo / ".git").is_file()

    def __enter__(self) -> Path:
        if not self._is_git:
            self.worktree_dir = Path(tempfile.mkdtemp(prefix="perron_iso_"))
            shutil.copytree(self.base_repo, self.worktree_dir, dirs_exist_ok=True)
            return self.worktree_dir

        self.worktree_dir = Path(tempfile.mkdtemp(prefix="perron_wt_"))
        try:
            cmd = ["git", "worktree", "add", "--detach", str(self.worktree_dir), self.commit]
            subprocess.run(cmd, cwd=str(self.base_repo), check=True, capture_output=True, text=True)
            return self.worktree_dir
        except Exception:
            shutil.rmtree(self.worktree_dir, ignore_errors=True)
            self.worktree_dir = Path(tempfile.mkdtemp(prefix="perron_iso_"))
            shutil.copytree(self.base_repo, self.worktree_dir, dirs_exist_ok=True)
            return self.worktree_dir

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.worktree_dir and self.worktree_dir.exists():
            import gc
            import stat
            gc.collect()

            def _on_rm_error(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass

            if self._is_git:
                try:
                    subprocess.run(
                        ["git", "worktree", "remove", "--force", str(self.worktree_dir)],
                        cwd=str(self.base_repo),
                        check=False,
                        capture_output=True,
                    )
                    subprocess.run(
                        ["git", "worktree", "prune"],
                        cwd=str(self.base_repo),
                        check=False,
                        capture_output=True,
                    )
                except Exception:
                    pass
            try:
                shutil.rmtree(self.worktree_dir, onerror=_on_rm_error)
            except Exception:
                shutil.rmtree(self.worktree_dir, ignore_errors=True)


def _kill_process_tree(pid: int, is_windows: bool) -> None:
    """Helper to forcefully terminate process group tree across platforms."""
    if is_windows:
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5.0,
            )
        except Exception:
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            pass


def run_targeted_test(
    test_file: Path | str,
    test_filter: Optional[str] = None,
    cwd: Optional[Path | str] = None,
    timeout_seconds: float = 30.0,
    extra_args: Optional[List[str]] = None,
    disable_plugins: Optional[List[str]] = None,
    isolation: str = "none",
) -> TestResult:
    """
    Run targeted pytest execution with platform-aware process-group timeout.

    Prevents whole-suite multi-hour execution by isolating specific failure reproduction targets.
    Supports physical file paths as well as Pytest NodeID selectors (e.g. tests/test_x.py::test_case).
    Enforces robust tree-kill on timeout:
      - Windows: taskkill /F /T /PID
      - Linux: os.killpg(SIGKILL)
    """
    raw_target = str(test_file)
    target_cwd = Path(cwd).resolve() if cwd else Path.cwd()
    if "::" in raw_target:
        file_part, selector = raw_target.split("::", 1)
        p = Path(file_part)
        file_path = (target_cwd / p).resolve() if not p.is_absolute() else p.resolve()
        test_target = f"{file_path}::{selector}"
    else:
        p = Path(test_file)
        file_path = (target_cwd / p).resolve() if not p.is_absolute() else p.resolve()
        test_target = str(file_path)

    if not cwd:
        target_cwd = file_path.parent

    if isolation == "git_worktree":
        with EphemeralWorktree(target_cwd) as iso_dir:
            return run_targeted_test(
                test_file=test_file,
                test_filter=test_filter,
                cwd=iso_dir,
                timeout_seconds=timeout_seconds,
                extra_args=extra_args,
                disable_plugins=disable_plugins,
                isolation="none",
            )

    if not file_path.is_file():
        return TestResult(
            passed=False,
            exit_code=4,
            duration_seconds=0.0,
            timed_out=False,
            output=f"[ERROR: Test file not found: {file_path}]",
        )

    effective_timeout = max(0.1, timeout_seconds)

    cmd = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        test_target,
        "-v",
        "-p", "no:cacheprovider",
    ]
    if disable_plugins:
        for plugin in disable_plugins:
            cmd.extend(["-p", f"no:{plugin}"])
    if test_filter:
        cmd.extend(["-k", test_filter])
    if extra_args:
        cmd.extend(extra_args)

    is_windows = os.name == "nt"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    target_str = str(target_cwd)
    env["PYTHONPATH"] = f"{target_str}{os.pathsep}{existing_pp}" if existing_pp else target_str

    kwargs = {
        "cwd": target_str,
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
        "text": True,
    }

    if is_windows:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["preexec_fn"] = os.setsid

    start_time = time.monotonic()
    timed_out = False
    output_str = ""
    exit_code = -1
    proc = None

    try:
        proc = subprocess.Popen(cmd, **kwargs)
        try:
            out, _ = proc.communicate(timeout=effective_timeout)
            output_str = out
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_tree(proc.pid, is_windows)
            try:
                proc.kill()
            except Exception:
                pass

            try:
                out, _ = proc.communicate(timeout=2.0)
                output_str = (out or "") + f"\n[ERROR: Test timed out after {effective_timeout}s]"
            except Exception:
                output_str = f"[ERROR: Test timed out after {effective_timeout}s]"

    except Exception as e:
        output_str = f"[EXECUTION ERROR: {str(e)}]"
        exit_code = 1
    finally:
        if proc and proc.poll() is None:
            _kill_process_tree(proc.pid, is_windows)
            try:
                proc.kill()
            except Exception:
                pass

    # Annotate pytest exit codes for agent diagnostics
    if exit_code == 1:
        output_str += "\n[DIAGNOSTIC: Pytest exit code 1 - Test assertions or runtime failure occurred]"
    elif exit_code == 2:
        output_str += "\n[DIAGNOSTIC: Pytest exit code 2 - Interrupted by user or signal]"
    elif exit_code == 3:
        output_str += "\n[DIAGNOSTIC: Pytest exit code 3 - Internal pytest engine error]"
    elif exit_code == 4:
        output_str += "\n[DIAGNOSTIC: Pytest exit code 4 - Command line usage error]"
    elif exit_code == 5:
        output_str += "\n[DIAGNOSTIC: Pytest exit code 5 - No tests matched selection filter]"

    duration = time.monotonic() - start_time
    passed = (exit_code == 0) and not timed_out

    return TestResult(
        passed=passed,
        exit_code=exit_code,
        duration_seconds=duration,
        timed_out=timed_out,
        output=output_str,
    )
