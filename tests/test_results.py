"""
Unit tests for perron/results.py and G5 metadata compliance.

Enforces:
Every results file includes: git sha, python version, package versions,
seed, split name, hardware (CPU, RAM, GPU), UTC timestamp.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np

from perron.results import (
    collect_experiment_metadata,
    load_results,
    save_results,
)


def test_g5_metadata_keys_present() -> None:
    """Verify that saved results contain all required G5 metadata keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_eval_results.json"
        payload = {
            "mrr": 0.85,
            "hit_at_5": 0.92,
            "arr": np.array([1, 2, 3]),
            "np_val": np.float64(3.1415),
        }

        saved_path = save_results(
            path=out_file,
            results_data=payload,
            split_name="dev_val",
            seed=20261003,
        )

        assert saved_path.is_file(), f"Output file does not exist: {saved_path}"

        with open(saved_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assert all G5 required keys in snake_case format
        required_g5_keys = [
            "git_sha",
            "python_version",
            "package_versions",
            "seed",
            "split_name",
            "hardware",
            "utc_timestamp",
        ]
        for key in required_g5_keys:
            assert key in data, f"Missing required G5 key: {key}"

        # Assert all G5 required keys in verbatim text format
        verbatim_g5_keys = [
            "git sha",
            "python version",
            "package versions",
            "split name",
            "UTC timestamp",
        ]
        for key in verbatim_g5_keys:
            assert key in data, f"Missing verbatim G5 key: {key}"

        # Hardware subkeys (CPU, RAM, GPU)
        assert isinstance(data["hardware"], dict), "hardware must be a dictionary"
        for hw_key in ["cpu", "ram", "gpu"]:
            assert hw_key in data["hardware"], f"Missing hardware subkey: {hw_key}"

        # Field validation
        assert data["seed"] == 20261003
        assert data["split_name"] == "dev_val"
        assert len(data["git_sha"]) > 0
        assert len(data["python_version"]) > 0
        assert isinstance(data["package_versions"], dict)
        assert len(data["utc_timestamp"]) > 0

        # Result data payload preserved
        assert data["mrr"] == 0.85
        assert data["hit_at_5"] == 0.92
        assert data["arr"] == [1, 2, 3]


def test_load_results_roundtrip() -> None:
    """Verify load_results properly parses serialized JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "subdir" / "roundtrip.json"
        save_results(
            path=out_file,
            results_data={"accuracy": 0.99},
            split_name="heldout",
            seed=42,
        )

        loaded = load_results(out_file)
        assert loaded["split_name"] == "heldout"
        assert loaded["seed"] == 42
        assert loaded["accuracy"] == 0.99
        assert "git_sha" in loaded
        assert "hardware" in loaded
