#!/usr/bin/env python3
"""
Patch notebooks/kaggle_gemma4_workstation.ipynb for Option A4-2:
Live Pointwise E4B Reasoning & AST Surgical Repair Showcase on Free Kaggle 2x T4 GPUs.

License: Apache-2.0
Author: Sultan Haikal (GitHub: @yvliet)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch Kaggle workstation notebook.")
    parser.add_argument("--mode", default="full", choices=["full", "a4_2"], help="Patching mode")
    args = parser.parse_args()

    build_script = PROJECT_ROOT / "scratch" / "build_a4_2_workstation.py"
    if not build_script.exists():
        raise FileNotFoundError(f"Build script not found: {build_script}")

    # Execute build script
    result = subprocess.run([sys.executable, str(build_script)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Build script failed:\n{result.stderr}\n{result.stdout}")

    print(result.stdout.strip())
    print("Patched notebooks/kaggle_gemma4_workstation.ipynb with Option A4-2 live showcase and AST editor successfully!")


if __name__ == "__main__":
    main()
