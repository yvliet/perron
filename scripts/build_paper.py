#!/usr/bin/env python3
"""
scripts/build_paper.py - Master Unified Build & Publication Pipeline for Perron.

Single command orchestrator for:
    - Generating all publication-grade figures (HTML/Chrome & Matplotlib)
    - Compiling paper/main.tex -> paper/main.pdf via standalone Tectonic
    - Running rigorous paper verification gates
    - Executing full test suites
    - Launching local PDF reader or hot-reload watcher

Usage:
    python scripts/build_paper.py --all         # Full pipeline: Figures -> Compile -> Verify -> Test
    python scripts/build_paper.py --figures     # Regenerate all publication figures
    python scripts/build_paper.py --compile     # Compile LaTeX to PDF
    python scripts/build_paper.py --verify      # Run academic quality gates
    python scripts/build_paper.py --test        # Run pytest test suite
    python scripts/build_paper.py --read        # Open compiled PDF in Chrome / system viewer
    python scripts/build_paper.py --watch       # Hot-reload watcher (recompiles on edit)
    python scripts/build_paper.py --server      # Start local HTTP viewer (localhost:8080)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def step_generate_figures(fig_arg="all"):
    print("\n" + "=" * 70)
    print("STEP: GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "generate_figures.py"), "--fig", str(fig_arg)]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print("[FAIL] Figure generation failed.")
        return False
    return True

def step_compile_latex():
    print("\n" + "=" * 70)
    print("STEP: COMPILING LATEX PAPER (paper/main.tex -> paper/main.pdf)")
    print("=" * 70)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "compile.py"), "paper/main.tex"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print("[FAIL] LaTeX compilation failed.")
        return False
    return True

def step_verify_quality_gates():
    print("\n" + "=" * 70)
    print("STEP: VERIFYING ACADEMIC QUALITY GATES")
    print("=" * 70)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "verify_paper.py")]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print("[FAIL] Quality gates verification failed.")
        return False
    return True

def step_run_tests():
    print("\n" + "=" * 70)
    print("STEP: RUNNING UNIT & INTEGRATION TEST SUITE")
    print("=" * 70)
    cmd = [sys.executable, "-m", "pytest", "-q"]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        print("[FAIL] Test suite failed.")
        return False
    return True

def step_read_paper(watch=False, server=False, port=8080):
    reader_script = REPO_ROOT / "paper" / "read_paper.py"
    cmd = [sys.executable, str(reader_script)]
    if watch:
        cmd.append("--watch")
    elif server:
        cmd.extend(["--server", "--port", str(port)])
    subprocess.run(cmd, cwd=str(REPO_ROOT))

def main():
    parser = argparse.ArgumentParser(
        description="Master Unified Build & Publication Pipeline for Perron.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/build_paper.py --all
    python scripts/build_paper.py --compile
    python scripts/build_paper.py --figures
    python scripts/build_paper.py --verify
    python scripts/build_paper.py --read
    python scripts/build_paper.py --watch
        """
    )
    parser.add_argument("--all", action="store_true", help="Run full pipeline: Figures -> Compile -> Verify -> Test")
    parser.add_argument("--figures", action="store_true", help="Generate all publication-grade figures")
    parser.add_argument("--fig", dest="fig_id", type=str, default=None, help="Generate a specific figure (1-5)")
    parser.add_argument("--compile", action="store_true", help="Compile LaTeX document via standalone Tectonic")
    parser.add_argument("--verify", action="store_true", help="Verify paper figures, citations, privacy, and typography")
    parser.add_argument("--test", action="store_true", help="Run repository pytest test suite")
    parser.add_argument("--read", action="store_true", help="Open main.pdf in Chrome or system default PDF viewer")
    parser.add_argument("--watch", action="store_true", help="Launch live hot-reloading watcher for LaTeX & figures")
    parser.add_argument("--server", action="store_true", help="Launch local HTTP reader server")
    parser.add_argument("--port", type=int, default=8080, help="Port for local HTTP server (default: 8080)")

    args = parser.parse_args()

    # Default to --compile if no arguments provided
    if not any([args.all, args.figures, args.fig_id, args.compile, args.verify, args.test, args.read, args.watch, args.server]):
        print("No specific action selected. Running default: --compile")
        args.compile = True

    if args.watch:
        step_read_paper(watch=True)
        return 0

    if args.server:
        step_read_paper(server=True, port=args.port)
        return 0

    if args.all:
        if not step_generate_figures("all"): return 1
        if not step_compile_latex(): return 1
        if not step_verify_quality_gates(): return 1
        if not step_run_tests(): return 1
        print("\n" + "=" * 70)
        print("ALL PIPELINE STAGES PASSED! PERRON PAPER IS FULLY PUBLICATION-READY.")
        print("=" * 70)
        return 0

    success = True
    if args.figures or args.fig_id:
        success = step_generate_figures(args.fig_id or "all") and success

    if args.compile:
        success = step_compile_latex() and success

    if args.verify:
        success = step_verify_quality_gates() and success

    if args.test:
        success = step_run_tests() and success

    if args.read:
        step_read_paper()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
