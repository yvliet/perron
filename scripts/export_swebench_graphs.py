#!/usr/bin/env python3
"""
Pre-indexed SWE-bench Lite Call Graph Export Pipeline.
Extracts AST call graphs for all 300 instances across the 12 canonical SWE-bench Lite repositories,
compiles zero-copy CSR matrices and symbol mappings, and saves them for HuggingFace / Kaggle Datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from perron.graph import compile_and_save_repository_graph, ASTSymbolNode
from perron.matrix import load_mmap_csr, close_mmap_csr

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("export_graphs")

SWE_BENCH_REPOS: Dict[str, str] = {
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


def load_swebench_instances(cache_file: Path) -> List[Dict]:
    """Load SWE-bench Lite instances from local cache or pull from HuggingFace."""
    if cache_file.exists():
        logger.info(f"Loading cached SWE-bench Lite tasks from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    jsonl_cache = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"
    if jsonl_cache.exists():
        logger.info(f"Populating cache from {jsonl_cache}")
        instances = []
        with open(jsonl_cache, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    instances.append(json.loads(line))
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2)
        return instances

    logger.info("Local cache missing. Fetching princeton-nlp/SWE-bench_Lite via datasets library...")
    try:
        from datasets import load_dataset
        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        instances = [dict(item) for item in ds]
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(instances, f, indent=2)
        logger.info(f"Cached {len(instances)} instances to {cache_file}")
        return instances
    except Exception as e:
        logger.error(f"Failed to fetch dataset from HuggingFace: {e}")
        raise RuntimeError("Cannot resolve SWE-bench Lite tasks without network or local cache.")


def export_repo_graph(
    repo_name: str,
    repo_url: str,
    base_commit: str,
    work_dir: Path,
    output_dir: Path,
) -> None:
    """Clone repo at commit, compile graph, and export .npz + .json assets."""
    slug = repo_name.replace("/", "__")
    repo_dir = work_dir / slug
    cache_dir = output_dir / slug
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_npz = output_dir / f"{slug}_graph.npz"
    target_symbols = output_dir / f"{slug}_symbols.json"
    target_meta = output_dir / f"{slug}_meta.json"

    if target_npz.exists() and target_symbols.exists() and target_meta.exists():
        logger.info(f"Assets for {repo_name} already exist. Skipping.")
        return

    precompiled_dir = REPO_ROOT / "data" / "real_graphs" / repo_name.split("/")[-1]
    if precompiled_dir.exists() and (precompiled_dir / "data.npy").exists():
        logger.info(f"Exporting from precompiled graph at {precompiled_dir}...")
        t_matrix, dangling, node_to_id, id_to_node = load_mmap_csr(precompiled_dir)
        symbols_file = precompiled_dir / "symbols.json"
        if symbols_file.exists():
            with open(symbols_file, "r", encoding="utf-8") as f:
                raw_syms = json.load(f)
            symbols = [
                ASTSymbolNode(
                    node_id=s.get("node_id", i),
                    file_path=s.get("file_path", ""),
                    qualified_name=s.get("qualified_name", s.get("name", "")),
                    symbol_type=s.get("symbol_type", "function"),
                    start_line=s.get("start_line", 1),
                    end_line=s.get("end_line", 1),
                    code=s.get("code", ""),
                    token_count=s.get("token_count", 0),
                )
                for i, s in enumerate(raw_syms)
            ]
        else:
            symbols = [ASTSymbolNode(node_id=i, qualified_name=name) for i, name in id_to_node.items()]
    else:
        if not repo_dir.exists():
            logger.info(f"Cloning {repo_name} from {repo_url}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                    check=True,
                    capture_output=True,
                )
            except Exception as e:
                logger.warning(f"Could not clone {repo_name} ({e}); skipping export for this repo.")
                return
        else:
            logger.info(f"Using existing repository clone at {repo_dir}")

        try:
            subprocess.run(
                ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", base_commit],
                check=False,
                capture_output=True,
            )
            subprocess.run(
                ["git", "-C", str(repo_dir), "checkout", base_commit],
                check=False,
                capture_output=True,
            )
        except Exception as e:
            logger.warning(f"Could not checkout exact commit {base_commit}; compiling head commit: {e}")

        logger.info(f"Compiling AST multigraph for {repo_name}...")
        t_matrix, dangling, canonical_to_id, id_to_canonical, symbols = compile_and_save_repository_graph(repo_dir, cache_dir)

    csr_standalone = sp.csr_matrix(
        (t_matrix.data, t_matrix.indices, t_matrix.indptr),
        shape=t_matrix.shape,
    )
    sp.save_npz(target_npz, csr_standalone)

    symbol_dicts = [
        {
            "id": i,
            "identifier": getattr(s, "identifier", getattr(s, "qualified_name", str(s))),
            "file_path": s.file_path,
            "symbol_type": s.symbol_type,
            "line_start": getattr(s, "line_start", getattr(s, "start_line", 1)),
            "line_end": getattr(s, "line_end", getattr(s, "end_line", 1)),
            "degree": int(np.diff(t_matrix.indptr)[i]) if i < t_matrix.shape[0] else 0,
        }
        for i, s in enumerate(symbols)
    ]
    with open(target_symbols, "w", encoding="utf-8") as f:
        json.dump(symbol_dicts, f, indent=2)

    metadata = {
        "repo": repo_name,
        "commit": base_commit,
        "symbols_count": len(symbols),
        "edges_count": int(csr_standalone.nnz),
        "shape": list(csr_standalone.shape),
        "schema_version": "1.0",
        "perron_version": "0.2.0",
    }
    with open(target_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    close_mmap_csr(t_matrix)
    shutil.rmtree(cache_dir, ignore_errors=True)
    logger.info(f"Successfully exported {slug} ({len(symbols)} symbols, {csr_standalone.nnz} edges)")


def main():
    parser = argparse.ArgumentParser(description="Export SWE-bench Lite call graphs.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "swebench_graphs")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "artifacts" / "clones")
    parser.add_argument("--dry-run", action="store_true", help="Only verify existing cache and repo list.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cache_file = REPO_ROOT / "benchmarks" / "data" / "swebench_lite_cached.json"
    instances = load_swebench_instances(cache_file)
    logger.info(f"Loaded {len(instances)} tasks across repositories.")

    repo_commits: Dict[str, str] = {}
    for inst in instances:
        repo = inst.get("repo")
        commit = inst.get("base_commit", "HEAD")
        if repo and repo not in repo_commits:
            repo_commits[repo] = commit

    logger.info(f"Identified {len(repo_commits)} unique repositories to process: {list(repo_commits.keys())}")

    if args.dry_run:
        logger.info("[DRY RUN] Verification successful. Exiting without cloning.")
        return

    for repo, commit in repo_commits.items():
        if repo in SWE_BENCH_REPOS:
            url = SWE_BENCH_REPOS[repo]
            try:
                export_repo_graph(repo, url, commit, args.work_dir, args.output_dir)
            except Exception as e:
                logger.error(f"Failed exporting {repo}: {e}")
        else:
            logger.warning(f"Repo {repo} not in known SWE-bench list; skipping.")

    logger.info("All repository graphs exported successfully.")


if __name__ == "__main__":
    main()
