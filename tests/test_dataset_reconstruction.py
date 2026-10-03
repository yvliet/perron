"""
Unit tests for scripts/reconstruct_dataset.py (dataset hydration and license compliance).
"""

import json
from pathlib import Path
import pytest

from scripts.reconstruct_dataset import (
    compute_sha256,
    slice_source_file,
    hydrate_graph_symbols,
    REPO_LICENSES,
)


def test_slice_source_file(tmp_path: Path):
    sample_file = tmp_path / "sample.py"
    sample_file.write_text(
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "line 5\n",
        encoding="utf-8",
    )

    # 1. Slice lines 2 to 4
    sliced = slice_source_file(sample_file, 2, 4)
    assert sliced == "line 2\nline 3\nline 4\n"

    # 2. Slice out-of-bounds
    sliced_oob = slice_source_file(sample_file, 4, 10)
    assert sliced_oob == "line 4\nline 5\n"

    # 3. Non-existent file
    assert slice_source_file(tmp_path / "missing.py", 1, 5) == ""


def test_hydrate_graph_symbols(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    
    src_file = repo_dir / "calc.py"
    code_content = "def add(a, b):\n    return a + b\n"
    src_file.write_text(code_content, encoding="utf-8")
    
    code_hash = compute_sha256(code_content)
    
    meta_path = tmp_path / "meta.json"
    meta_payload = {
        "repository": "psf/requests",
        "base_commit": "abc1234",
        "symbols": [
            {
                "symbol_id": 1,
                "file_path": "calc.py",
                "identifier": "add",
                "symbol_type": "function",
                "start_line": 1,
                "end_line": 2,
                "content_sha256": code_hash,
            }
        ]
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f)

    out_path = tmp_path / "hydrated.json"
    res = hydrate_graph_symbols(meta_path, repo_dir, out_path)

    assert res["symbols_count"] == 1
    assert res["license"] == "Apache-2.0"
    assert res["hash_verification"]["matches"] == 1
    assert res["hash_verification"]["mismatches"] == 0
    assert res["symbols"][0]["code"] == code_content
    assert res["symbols"][0]["token_count"] == len(code_content.split())
    assert out_path.exists()
