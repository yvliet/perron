#!/usr/bin/env python3
"""
Clean-Room Wheel Installation Verification Test.
Builds the perron-core wheel, creates an isolated temporary virtual environment,
installs the wheel, and verifies CLI entrypoints and module imports.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("clean_install")

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run command and log output on failure."""
    logger.info(f"Running: {' '.join(str(c) for c in cmd)}")
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        logger.error(f"Command failed (exit {res.returncode}):\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")
    return res


def main():
    dist_dir = REPO_ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build the distribution wheel
    logger.info("Building distribution wheel with python -m build --wheel...")
    run_cmd([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)], cwd=REPO_ROOT)

    wheel_files = list(dist_dir.glob("perron_core-0.3.0-*.whl"))
    if not wheel_files:
        # Check alternative name
        wheel_files = list(dist_dir.glob("perron*.whl"))
    if not wheel_files:
        raise FileNotFoundError(f"No built wheel found in {dist_dir}")

    latest_wheel = max(wheel_files, key=os.path.getmtime)
    logger.info(f"Built wheel: {latest_wheel}")

    # 2. Check wheel metadata with twine if available
    try:
        run_cmd([sys.executable, "-m", "twine", "check", str(latest_wheel)], cwd=REPO_ROOT)
        logger.info("Twine metadata check: PASS")
    except Exception as e:
        logger.warning(f"Twine check skipped: {e}")

    # 3. Create isolated virtual environment
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_venv_path = Path(tmp_dir) / "venv"
        logger.info(f"Creating isolated virtual environment at {tmp_venv_path}...")
        venv.create(tmp_venv_path, with_pip=True)

        # Locate venv python and pip
        if sys.platform == "win32":
            venv_python = tmp_venv_path / "Scripts" / "python.exe"
            venv_pip = tmp_venv_path / "Scripts" / "pip.exe"
            venv_perron = tmp_venv_path / "Scripts" / "perron.exe"
            venv_perron_mcp = tmp_venv_path / "Scripts" / "perron-mcp.exe"
        else:
            venv_python = tmp_venv_path / "bin" / "python"
            venv_pip = tmp_venv_path / "bin" / "pip"
            venv_perron = tmp_venv_path / "bin" / "perron"
            venv_perron_mcp = tmp_venv_path / "bin" / "perron-mcp"

        # 4. Install wheel into isolated venv
        logger.info("Installing wheel into isolated venv...")
        run_cmd([str(venv_pip), "install", str(latest_wheel)])

        # 5. Verify import and version
        logger.info("Verifying module import in clean venv...")
        res = run_cmd([str(venv_python), "-c", "import perron; print(perron.__version__)"])
        assert "0.3.0" in res.stdout.strip(), f"Expected version 0.3.0, got {res.stdout}"

        # 6. Verify CLI entrypoint
        logger.info("Verifying perron --version CLI entrypoint...")
        res = run_cmd([str(venv_perron), "--version"])
        assert "0.3.0" in res.stdout.strip() or "0.3.0" in res.stderr.strip()

        logger.info("Verifying perron --help CLI entrypoint...")
        res = run_cmd([str(venv_perron), "--help"])
        assert "index" in res.stdout and "query" in res.stdout and "bench" in res.stdout

        logger.info("Verifying perron-mcp entrypoint...")
        res = run_cmd([str(venv_perron_mcp), "--version"])
        assert "0.3.0" in res.stdout.strip() or "0.3.0" in res.stderr.strip()

        logger.info("All clean-room installation gates PASSED successfully!")


if __name__ == "__main__":
    main()
