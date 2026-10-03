"""
Hardware and Memory Telemetry Profiling for Gemma 4 & Perron.

Measures actual system specifications, resident memory footprint (RSS),
memory-mapped CSR virtualization efficiency, and retrieval latency on host hardware.
Outputs verified telemetry to `results/hardware_telemetry_measured.json`.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict
import numpy as np
import psutil

from perron.retriever import CodeGraph, PerronRetriever

logging.basicConfig(level=logging.INFO, format="[hw-telemetry] %(levelname)s: %(message)s")
logger = logging.getLogger("hw-telemetry")


def profile_hardware() -> Dict[str, Any]:
    process = psutil.Process(os.getpid())
    base_rss_mb = process.memory_info().rss / (1024 * 1024)
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    logger.info("Profiling on %s %s (Python %s)...", platform.system(), platform.release(), platform.python_version())
    logger.info("Total System RAM: %.2f GB (Available: %.2f GB)", total_ram_gb, available_ram_gb)
    logger.info("Base Process RSS: %.2f MB", base_rss_mb)

    # Measure CSR mmap loading on real graph assets
    real_graph_dir = Path(__file__).resolve().parent.parent / "data" / "real_graphs" / "requests"
    mmap_rss_delta_mb = 0.0
    load_time_ms = 0.0
    query_time_ms = 0.0

    if real_graph_dir.exists():
        t0 = time.perf_counter()
        # Mock repo dir to requests cache dir
        graph = CodeGraph.from_directory(real_graph_dir.parent.parent, real_graph_dir)
        load_time_ms = (time.perf_counter() - t0) * 1000.0

        rss_after_mmap = process.memory_info().rss / (1024 * 1024)
        mmap_rss_delta_mb = max(0.0, rss_after_mmap - base_rss_mb)

        # Measure query latency
        retriever = PerronRetriever(graph, gamma=0.70, beta=0.85)
        t_q0 = time.perf_counter()
        _ = retriever.query("session request connection pool timeout", max_tokens=4096)
        query_time_ms = (time.perf_counter() - t_q0) * 1000.0

        graph.close()

    telemetry = {
        "host_hardware": {
            "platform": platform.platform(),
            "cpu_architecture": platform.machine(),
            "cpu_count": psutil.cpu_count(logical=True),
            "total_physical_ram_gb": round(total_ram_gb, 2),
            "available_physical_ram_gb": round(available_ram_gb, 2),
            "python_version": platform.python_version(),
        },
        "measured_runtime_rss": {
            "base_process_rss_mb": round(base_rss_mb, 2),
            "mmap_csr_rss_delta_mb": round(mmap_rss_delta_mb, 2),
            "mmap_cold_start_ms": round(load_time_ms, 3),
            "retrieval_query_ms": round(query_time_ms, 2),
        },
    }

    out_file = Path(__file__).resolve().parent.parent / "results" / "hardware_telemetry_measured.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)

    logger.info("Hardware telemetry recorded: %s", telemetry["measured_runtime_rss"])
    return telemetry


if __name__ == "__main__":
    profile_hardware()
