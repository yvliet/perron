#!/usr/bin/env python3
"""
Turnkey distribution build and PyPI verification/publishing script for perron-core.
Ensures clean sdist and wheel build, verifies metadata, and securely calls twine.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = REPO_ROOT / "dist"


def clean_dist():
    if DIST_DIR.exists():
        print(f"Cleaning previous build artifacts in {DIST_DIR}...")
        shutil.rmtree(DIST_DIR)


def build_package():
    print("Building sdist and bdist_wheel via python -m build...")
    subprocess.run([sys.executable, "-m", "build"], cwd=str(REPO_ROOT), check=True)


def check_twine():
    print("Verifying package distributions via twine check...")
    files = [str(f) for f in DIST_DIR.iterdir() if f.is_file()]
    subprocess.run([sys.executable, "-m", "twine", "check"] + files, cwd=str(REPO_ROOT), check=True)


def upload_pypi(test: bool = False):
    target = "testpypi" if test else "pypi"
    print(f"Uploading to {target}...")
    cmd = [sys.executable, "-m", "twine", "upload"]
    if test:
        cmd.extend(["--repository", "testpypi"])
    files = [str(f) for f in DIST_DIR.iterdir() if f.is_file()]
    cmd.extend(files)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    parser = argparse.ArgumentParser(description="Build and publish perron-core to PyPI.")
    parser.add_argument("--build-only", action="store_true", help="Only build and verify dist/, do not upload.")
    parser.add_argument("--test", action="store_true", help="Upload to TestPyPI instead of production PyPI.")
    args = parser.parse_args()

    clean_dist()
    build_package()
    check_twine()

    if args.build_only:
        print("\n[SUCCESS] Distribution packages built and verified in dist/:")
        for f in DIST_DIR.iterdir():
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
        print("\nTo publish manually, run:")
        print("  python -m twine upload dist/*")
        return

    upload_pypi(test=args.test)
    print("\n[SUCCESS] perron-core published successfully!")


if __name__ == "__main__":
    main()
