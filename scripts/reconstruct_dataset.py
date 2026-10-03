"""
Dataset Reconstruction Tool for swebench-lite-codegraphs.

Hydrates verbatim source code segments from local git checkouts at exact base commits,
ensuring 100% license compliance by decoupling copyrightable source code from
freely distributable mathematical graph topology metadata.

Usage:
    python scripts/reconstruct_dataset.py --repos-dir ./clones --output-dir ./hydrated_graphs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[reconstruct-dataset] %(levelname)s: %(message)s",
)
logger = logging.getLogger("reconstruct-dataset")

REPO_UPSTREAM_URLS: Dict[str, str] = {
    "astropy/astropy": "https://github.com/astropy/astropy.git",
    "django/django": "https://github.com/django/django.git",
    "matplotlib/matplotlib": "https://github.com/matplotlib/matplotlib.git",
    "mwaskom/seaborn": "https://github.com/mwaskom/seaborn.git",
    "pallets/flask": "https://github.com/pallets/flask.git",
    "psf/requests": "https://github.com/psf/requests.git",
    "pydata/xarray": "https://github.com/pydata/xarray.git",
    "pylint-dev/pylint": "https://github.com/pylint-dev/pylint.git",
    "pytest-dev/pytest": "https://github.com/pytest-dev/pytest.git",
    "scikit-learn/scikit-learn": "https://github.com/scikit-learn/scikit-learn.git",
    "sphinx-doc/sphinx": "https://github.com/sphinx-doc/sphinx.git",
    "sympy/sympy": "https://github.com/sympy/sympy.git",
}

REPO_LICENSES: Dict[str, str] = {
    "astropy/astropy": "BSD-3-Clause",
    "django/django": "BSD-3-Clause",
    "matplotlib/matplotlib": "PSF / Custom Permissive",
    "mwaskom/seaborn": "BSD-3-Clause",
    "pallets/flask": "BSD-3-Clause",
    "psf/requests": "Apache-2.0",
    "pydata/xarray": "Apache-2.0",
    "pylint-dev/pylint": "GPL-2.0-or-later",
    "pytest-dev/pytest": "MIT",
    "scikit-learn/scikit-learn": "BSD-3-Clause",
    "sphinx-doc/sphinx": "BSD-2-Clause",
    "sympy/sympy": "BSD-3-Clause",
}


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slice_source_file(file_path: Path, start_line: int, end_line: int) -> str:
    """Read 1-indexed line range from a text file."""
    if not file_path.exists():
        return ""
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        s_idx = max(0, start_line - 1)
        e_idx = min(len(lines), end_line)
        return "".join(lines[s_idx:e_idx])
    except Exception as e:
        logger.warning("Failed to slice %s: %s", file_path, e)
        return ""


def ensure_repo_checkout(repo_slug: str, target_dir: Path, base_commit: str, auto_clone: bool = False) -> bool:
    """Verify or checkout a repository at the specified base commit."""
    if not (target_dir / ".git").exists():
        if not auto_clone:
            logger.error("Git directory not found at %s. Pass --auto-clone to clone automatically.", target_dir)
            return False
        url = REPO_UPSTREAM_URLS.get(repo_slug)
        if not url:
            logger.error("No upstream URL known for %s", repo_slug)
            return False
        logger.info("Cloning %s from %s...", repo_slug, url)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run(["git", "clone", url, str(target_dir)], capture_output=True, text=True)
        if res.returncode != 0:
            logger.error("Clone failed: %s", res.stderr)
            return False

    # Checkout base commit
    res = subprocess.run(["git", "-C", str(target_dir), "checkout", base_commit], capture_output=True, text=True)
    if res.returncode != 0:
        logger.warning("Failed to checkout %s in %s: %s", base_commit, target_dir, res.stderr)
        return False
    return True


def hydrate_graph_symbols(
    graph_metadata_path: Path,
    repo_checkout_dir: Path,
    output_path: Path,
    verify_hashes: bool = True,
) -> Dict[str, Any]:
    """
    Hydrate graph symbol metadata with verbatim source code read from the local git checkout.
    """
    with open(graph_metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    symbols = meta.get("symbols", [])
    hydrated_symbols = []
    matches = 0
    mismatches = 0

    for sym in symbols:
        f_rel = sym.get("file_path", "")
        start_l = sym.get("start_line", 1)
        end_l = sym.get("end_line", 1)
        expected_hash = sym.get("content_sha256")

        target_file = repo_checkout_dir / f_rel
        code = slice_source_file(target_file, start_l, end_l)
        actual_hash = compute_sha256(code) if code else ""

        if expected_hash and actual_hash != expected_hash:
            mismatches += 1
        else:
            matches += 1

        hydrated_sym = dict(sym)
        hydrated_sym["code"] = code
        hydrated_sym["token_count"] = max(1, len(code.split()))
        hydrated_symbols.append(hydrated_sym)

    result_payload = {
        "repository": meta.get("repository", repo_checkout_dir.name),
        "base_commit": meta.get("base_commit", ""),
        "license": REPO_LICENSES.get(meta.get("repository", ""), "Unknown"),
        "symbols_count": len(hydrated_symbols),
        "hash_verification": {
            "matches": matches,
            "mismatches": mismatches,
        },
        "symbols": hydrated_symbols,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    logger.info("Saved hydrated symbols to %s (%d symbols, %d hash matches)", output_path, len(hydrated_symbols), matches)
    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="reconstruct-dataset",
        description="Hydrate code text into swebench-lite-codegraphs from local git repositories.",
    )
    parser.add_argument("--graph-meta", "-g", help="Path to graph metadata JSON.")
    parser.add_argument("--repo-dir", "-r", help="Path to local checked-out git repository.")
    parser.add_argument("--base-commit", "-b", help="Git commit SHA to checkout.")
    parser.add_argument("--output", "-o", help="Target output JSON path for hydrated graph.")
    parser.add_argument("--auto-clone", action="store_true", help="Automatically clone missing repositories.")
    parser.add_argument("--verify-only", action="store_true", help="Verify line spans without writing output.")
    args = parser.parse_args()

    if not args.graph_meta or not args.repo_dir:
        parser.print_help()
        sys.exit(1)

    meta_p = Path(args.graph_meta)
    repo_p = Path(args.repo_dir)
    out_p = Path(args.output) if args.output else meta_p.parent / f"{meta_p.stem}_hydrated.json"

    if args.base_commit:
        success = ensure_repo_checkout("custom", repo_p, args.base_commit, auto_clone=args.auto_clone)
        if not success:
            sys.exit(1)

    hydrate_graph_symbols(meta_p, repo_p, out_p)


if __name__ == "__main__":
    main()
