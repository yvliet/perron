"""
Perron Standardized Results Writer & Experiment Metadata Serialization (G5).

Guarantees every result artifact captures:
- git sha
- python version
- package versions
- seed
- split name
- hardware (CPU, RAM, GPU)
- UTC timestamp
"""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Optional, Union


class ResultsJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling NumPy types, sets, and Path objects."""

    def default(self, o: Any) -> Any:
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except ImportError:
            pass

        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (set, frozenset)):
            return sorted(list(o))
        if isinstance(o, (datetime.date, datetime.datetime)):
            return o.isoformat()

        return super().default(o)


def get_git_sha() -> str:
    """Retrieve current Git commit SHA, or 'unknown' if not in a repository."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def get_package_versions() -> Dict[str, str]:
    """Capture versions of installed core packages."""
    packages = [
        "perron-core",
        "perron",
        "numpy",
        "scipy",
        "pytest",
        "torch",
        "transformers",
        "tree-sitter",
        "psutil",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pass
    return versions


def get_hardware_info() -> Dict[str, Any]:
    """Capture hardware configuration (CPU, RAM, GPU)."""
    # CPU
    cpu_info = platform.processor() or platform.machine() or "unknown"

    # RAM
    ram_bytes: Union[int, str] = "unknown"
    try:
        import psutil
        ram_bytes = int(psutil.virtual_memory().total)
    except Exception:
        pass

    # GPU
    gpu_info = "None"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = str(torch.cuda.get_device_name(0))
    except Exception:
        pass

    if gpu_info == "None":
        try:
            smi_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if smi_out:
                gpu_info = smi_out.split("\n")[0].strip()
        except Exception:
            pass

    return {
        "cpu": cpu_info,
        "ram": ram_bytes,
        "gpu": gpu_info,
        "CPU": cpu_info,
        "RAM": ram_bytes,
        "GPU": gpu_info,
    }


def collect_experiment_metadata(split_name: str, seed: int = 20261003) -> Dict[str, Any]:
    """
    Assemble the complete G5 metadata payload.

    Contains both canonical snake_case and verbatim literal G5 keys.
    """
    git_sha = get_git_sha()
    py_ver = sys.version
    pkg_vers = get_package_versions()
    hw = get_hardware_info()
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    return {
        # Canonical snake_case keys
        "git_sha": git_sha,
        "python_version": py_ver,
        "package_versions": pkg_vers,
        "seed": seed,
        "split_name": split_name,
        "hardware": hw,
        "utc_timestamp": ts,
        # Verbatim G5 literal keys
        "git sha": git_sha,
        "python version": py_ver,
        "package versions": pkg_vers,
        "split name": split_name,
        "UTC timestamp": ts,
    }


def save_results(
    path: Union[str, Path],
    results_data: Any,
    split_name: str,
    seed: int = 20261003,
    **extra_metadata: Any,
) -> Path:
    """
    Save evaluation results with strict G5 metadata to JSON.

    Parameters
    ----------
    path : str or Path
        Destination path for the results JSON file.
    results_data : Any
        Evaluation metrics, summary dictionary, or records payload.
    split_name : str
        Name of dataset partition (e.g., 'dev_val', 'dev_pilot', 'heldout').
    seed : int, optional
        Pseudorandom seed used for the run (default 20261003).
    **extra_metadata : Any
        Additional custom metadata to record.

    Returns
    -------
    Path
        Absolute Path to the written JSON file.
    """
    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = collect_experiment_metadata(split_name=split_name, seed=seed)
    if extra_metadata:
        metadata.update(extra_metadata)

    payload: Dict[str, Any] = {}
    # Embed G5 metadata at root
    payload.update(metadata)
    payload["metadata"] = metadata

    # Embed results
    if isinstance(results_data, dict):
        for k, v in results_data.items():
            if k not in payload:
                payload[k] = v
        payload["results"] = results_data
        payload["data"] = results_data
    else:
        payload["results"] = results_data
        payload["data"] = results_data

    temp_path = target_path.with_suffix(f"{target_path.suffix}.tmp.{os.getpid()}")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=ResultsJSONEncoder)
        temp_path.replace(target_path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

    return target_path


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and return serialized results dictionary."""
    target_path = Path(path).resolve()
    with open(target_path, "r", encoding="utf-8") as f:
        return json.load(f)
