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
    cleanup_clone: bool = False,
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
    if cleanup_clone and repo_dir.exists():
        logger.info(f"Cleaning up clone at {repo_dir} to reclaim disk space...")
        shutil.rmtree(repo_dir, ignore_errors=True)
    logger.info(f"Successfully exported {slug} ({len(symbols)} symbols, {csr_standalone.nnz} edges)")


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 hash of a file."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


_REPO_CACHE: Dict[str, Tuple[sp.csr_matrix, List[Dict]]] = {}


def load_cached_repo_graph(repo_name: str, repo_graphs_dir: Path) -> Tuple[sp.csr_matrix, List[Dict]]:
    """Load and cache repository CSR matrix and symbol records."""
    slug = repo_name.replace("/", "__")
    if slug in _REPO_CACHE:
        return _REPO_CACHE[slug]

    target_npz = repo_graphs_dir / f"{slug}_graph.npz"
    target_symbols = repo_graphs_dir / f"{slug}_symbols.json"

    if not target_npz.exists() or not target_symbols.exists():
        raise FileNotFoundError(f"Missing precompiled graph for repository {repo_name} at {target_npz}")

    csr = sp.load_npz(target_npz)
    with open(target_symbols, "r", encoding="utf-8") as f:
        symbols = json.load(f)

    _REPO_CACHE[slug] = (csr, symbols)
    return csr, symbols


def build_instance_graph(
    instance: Dict,
    repo_graphs_dir: Path,
    target_dir: Path,
) -> None:
    """
    Build per-instance graph directory:
    csr_data.npy, csr_indices.npy, csr_indptr.npy, csr_dangling.npy,
    nodes.parquet, manifest.json.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    target_dir.mkdir(parents=True, exist_ok=True)

    repo_name = instance.get("repo", "")
    instance_id = instance.get("instance_id", "")
    base_commit = instance.get("base_commit", "HEAD")

    csr, symbols = load_cached_repo_graph(repo_name, repo_graphs_dir)
    num_nodes = len(symbols)

    data = csr.data.astype(np.float32)
    indices = csr.indices.astype(np.int32)
    indptr = csr.indptr.astype(np.int32)

    row_sums = np.diff(indptr)
    dangling = (row_sums == 0).astype(np.float32)

    in_degrees = np.bincount(indices, minlength=num_nodes).astype(np.int64)
    out_degrees = row_sums.astype(np.int64)

    if num_nodes > 0:
        k_top = min(25, num_nodes)
        top25_thresh = np.partition(in_degrees, -k_top)[-k_top]
        p99_thresh = np.percentile(in_degrees, 99.0)
    else:
        top25_thresh = 0
        p99_thresh = 0

    # Write CSR components
    np.save(target_dir / "csr_data.npy", data)
    np.save(target_dir / "csr_indices.npy", indices)
    np.save(target_dir / "csr_indptr.npy", indptr)
    np.save(target_dir / "csr_dangling.npy", dangling)

    # Build Parquet table for nodes
    ids: List[int] = []
    kinds: List[str] = []
    names: List[str] = []
    files: List[str] = []
    start_lines: List[int] = []
    end_lines: List[int] = []
    is_tests: List[bool] = []
    is_hub_top25s: List[bool] = []
    is_hub_p99s: List[bool] = []

    for i, s in enumerate(symbols):
        fpath = str(s.get("file_path", ""))
        qname = str(s.get("identifier", s.get("qualified_name", s.get("name", ""))))
        kind = str(s.get("symbol_type", "function"))

        s_val = s.get("line_start")
        if s_val is None:
            s_val = s.get("start_line")
        sline = int(s_val) if s_val is not None else 1

        e_val = s.get("line_end")
        if e_val is None:
            e_val = s.get("end_line")
        eline = int(e_val) if e_val is not None else sline

        in_deg = int(in_degrees[i])

        f_lower = fpath.lower()
        is_test = bool(
            "test" in f_lower or f_lower.startswith("tests/") or f_lower.startswith("test_")
        )

        ids.append(int(s.get("id", i)))
        kinds.append(kind)
        names.append(qname)
        files.append(fpath)
        start_lines.append(sline)
        end_lines.append(eline)
        is_tests.append(is_test)
        is_hub_top25s.append(bool(in_deg >= top25_thresh and top25_thresh > 0))
        is_hub_p99s.append(bool(in_deg >= p99_thresh and p99_thresh > 0))

    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(kinds, type=pa.string()),
            pa.array(names, type=pa.string()),
            pa.array(files, type=pa.string()),
            pa.array(start_lines, type=pa.int64()),
            pa.array(end_lines, type=pa.int64()),
            pa.array(in_degrees, type=pa.int64()),
            pa.array(out_degrees, type=pa.int64()),
            pa.array(is_tests, type=pa.bool_()),
            pa.array(is_hub_top25s, type=pa.bool_()),
            pa.array(is_hub_p99s, type=pa.bool_()),
        ],
        names=[
            "id",
            "kind",
            "qualified_name",
            "file",
            "start_line",
            "end_line",
            "in_degree",
            "out_degree",
            "is_test",
            "is_hub_top25",
            "is_hub_p99",
        ],
    )
    pq.write_table(table, target_dir / "nodes.parquet")

    # Compute SHA256 hashes
    hashes = {
        "csr_data.npy": compute_file_sha256(target_dir / "csr_data.npy"),
        "csr_indices.npy": compute_file_sha256(target_dir / "csr_indices.npy"),
        "csr_indptr.npy": compute_file_sha256(target_dir / "csr_indptr.npy"),
        "csr_dangling.npy": compute_file_sha256(target_dir / "csr_dangling.npy"),
        "nodes.parquet": compute_file_sha256(target_dir / "nodes.parquet"),
    }

    manifest = {
        "instance_id": instance_id,
        "repo": repo_name,
        "base_commit": base_commit,
        "perron_version": "0.3.0",
        "node_count": num_nodes,
        "edge_count": len(indices),
        "sha256": hashes,
    }

    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def export_all_instance_graphs(
    instances: List[Dict],
    repo_graphs_dir: Path,
    data_release_dir: Path,
    resume: bool = True,
) -> Tuple[int, int]:
    """
    Export all per-instance graphs into data_release/graphs/<instance_id>/.
    Handles resume and logs failures to data_release/failures.jsonl.
    """
    graphs_dir = data_release_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    failures_file = data_release_dir / "failures.jsonl"

    success_count = 0
    failure_count = 0
    failures: List[Dict] = []

    required_files = [
        "csr_data.npy",
        "csr_indices.npy",
        "csr_indptr.npy",
        "csr_dangling.npy",
        "nodes.parquet",
        "manifest.json",
    ]

    for inst in instances:
        instance_id = inst.get("instance_id", "")
        if not instance_id:
            continue

        target_dir = graphs_dir / instance_id
        if resume and target_dir.exists():
            if all((target_dir / f).exists() for f in required_files):
                success_count += 1
                continue

        try:
            build_instance_graph(inst, repo_graphs_dir, target_dir)
            success_count += 1
        except Exception as e:
            failure_count += 1
            err_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"Failed exporting graph for {instance_id}: {err_msg}")
            failures.append({"instance_id": instance_id, "error": err_msg})

    if failures:
        with open(failures_file, "w", encoding="utf-8") as f:
            for fail in failures:
                f.write(json.dumps(fail) + "\n")
    elif failures_file.exists():
        failures_file.unlink()

    return success_count, failure_count


def main():
    parser = argparse.ArgumentParser(description="Export SWE-bench Lite call graphs.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "swebench_graphs")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "artifacts" / "clones")
    parser.add_argument("--cleanup-clones", action="store_true", help="Remove cloned repo working directory after successful export to save disk.")
    parser.add_argument("--export-instances", action="store_true", help="Export per-instance graphs into data_release/graphs/.")
    parser.add_argument("--data-release-dir", type=Path, default=REPO_ROOT / "data_release")
    parser.add_argument("--dry-run", action="store_true", help="Only verify existing cache and repo list.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cache_file = REPO_ROOT / "benchmarks" / "data" / "swebench_lite_cached.json"
    instances = load_swebench_instances(cache_file)
    logger.info(f"Loaded {len(instances)} tasks across repositories.")

    if args.export_instances:
        logger.info(f"Exporting per-instance graphs for {len(instances)} instances to {args.data_release_dir / 'graphs'}...")
        succ, fail = export_all_instance_graphs(
            instances,
            repo_graphs_dir=args.output_dir,
            data_release_dir=args.data_release_dir,
            resume=True,
        )
        logger.info(f"Per-instance export complete: {succ} succeeded, {fail} failed.")
        return

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
                export_repo_graph(repo, url, commit, args.work_dir, args.output_dir, cleanup_clone=args.cleanup_clones)
            except Exception as e:
                logger.error(f"Failed exporting {repo}: {e}")
        else:
            logger.warning(f"Repo {repo} not in known SWE-bench list; skipping.")

    logger.info("All repository graphs exported successfully.")


if __name__ == "__main__":
    main()

