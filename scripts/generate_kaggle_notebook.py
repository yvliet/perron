"""
Generator for the Kaggle Gemma 4 Workstation Notebook and Kernel Metadata.
Option A4-2: Live Pointwise E4B Reasoning & AST Surgical Repair Showcase.

Configured for Kaggle's dual Tesla T4 GPUs (29.1 GB combined VRAM).
Author: Sultan Haikal (@yvliet)
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def build_kaggle_notebook() -> None:
    # Load cells from build script or define cleanly
    nb_path = NOTEBOOKS_DIR / "kaggle_gemma4_workstation.ipynb"
    if nb_path.exists():
        with open(nb_path, "r", encoding="utf-8") as f:
            existing_nb = json.load(f)
        cells = existing_nb.get("cells", [])
    else:
        cells = []

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    target_nb = NOTEBOOKS_DIR / "kaggle_gemma4_workstation.ipynb"
    with open(target_nb, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated Kaggle notebook: {target_nb}")

    # Generate Kaggle kernel metadata
    metadata = {
        "id": "yvliet/perron-gemma-4-workstation",
        "title": "perron-gemma-4-workstation",
        "code_file": "kaggle_gemma4_workstation.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "yvliet/perron-swebench-graphs",
            "yvliet/swebench-strata-annotations",
            "yvliet/perron-gemma4-benchmark"
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [
            "google/gemma-4/Transformers/gemma-4-e4b-it/1"
        ],
    }

    target_meta = NOTEBOOKS_DIR / "kernel-metadata.json"
    with open(target_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Generated Kaggle kernel metadata: {target_meta}")


if __name__ == "__main__":
    build_kaggle_notebook()
