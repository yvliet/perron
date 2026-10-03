#!/usr/bin/env python3
"""CPU inference telemetry for Gemma 4 E4B (GGUF, llama-cpp-python), or a fail-closed record.

Model lookup: environment variable PERRON_GEMMA_GGUF, else exactly one file in
models/ whose name matches 'gemma[-_]?4.*e4b' (case-insensitive) and ends in .gguf.
Without a model file or llama_cpp, writes {"status": "not_run", ...} and exits 0;
the paper then states that no model was executed. Only the file name and SHA-256
of the model are recorded, never its absolute path.

Protocol: prompts of exactly K tokens (K in 1024, 2048, 3480, 4096) built from the
problem statements of the first five held-out tasks (sorted by instance_id);
time-to-first-token depends on prompt length, not content. Per (K, task): one
warm-up run and three measured runs, greedy decoding, 128 generated tokens with
EOS suppressed, KV cache reset before every run. RSS and available memory are
sampled every 50 ms in a background thread during each run.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil  # noqa: E402

from perron.results import get_git_sha  # noqa: E402

OUT = PROJECT_ROOT / "results" / "live_inference_telemetry.json"
BUDGETS = (1024, 2048, 3480, 4096)
N_TASKS, N_WARMUP, N_REPEATS, N_GEN = 5, 1, 3, 128


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def locate_gemma_gguf() -> Tuple[Optional[Path], List[str]]:
    reasons: List[str] = []
    env = os.environ.get("PERRON_GEMMA_GGUF")
    if env:
        candidates = [Path(env)]
    else:
        models = PROJECT_ROOT / "models"
        candidates = sorted(p for p in models.glob("*.gguf") if re.search(r"gemma[-_]?4.*e4b", p.name, re.I)) if models.is_dir() else []
    if len(candidates) != 1:
        reasons.append(f"expected exactly one Gemma 4 E4B GGUF file, found {len(candidates)}")
    elif not candidates[0].is_file():
        reasons.append("model path is not a file")
    elif candidates[0].stat().st_size < 2 ** 30:
        reasons.append("model file is smaller than 1 GiB")
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        reasons.append("llama_cpp is not importable")
    return (candidates[0] if not reasons else None), reasons


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def task_texts() -> List[Tuple[str, str]]:
    split = json.loads((PROJECT_ROOT / "data" / "swebench_lite_split.json").read_text(encoding="utf-8"))
    ids = sorted(split["heldout_instances"])[:N_TASKS]
    with open(PROJECT_ROOT / "data" / "swebench_lite_cache.jsonl", encoding="utf-8") as f:
        rows = {r["instance_id"]: r for r in (json.loads(l) for l in f if l.strip())}
    return [(i, rows[i].get("problem_statement") or "") for i in ids]


def exact_prompt(llm, text: str, k: int) -> List[int]:
    base = llm.tokenize(text.encode("utf-8"), add_bos=False) or llm.tokenize(b"def f(): pass", add_bos=False)
    tokens = [llm.token_bos()]
    while len(tokens) < k:
        tokens.extend(base)
    return tokens[:k]


def measure(llm, prompt: List[int]) -> Dict[str, Any]:
    peak_rss, min_avail, stop = [0], [psutil.virtual_memory().available], threading.Event()
    proc = psutil.Process()

    def sampler() -> None:
        while not stop.is_set():
            peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
            min_avail[0] = min(min_avail[0], psutil.virtual_memory().available)
            time.sleep(0.05)

    llm.reset()
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    stamps: List[float] = []
    for _ in llm(prompt, max_tokens=N_GEN, temperature=0.0, top_k=1, stream=True,
                 logit_bias={int(llm.token_eos()): -100.0}):
        stamps.append(time.perf_counter())
    stop.set()
    th.join()
    ttft = stamps[0] - t0
    return {"ttft_s": ttft, "prefill_tok_s": len(prompt) / ttft,
            "decode_tok_s": (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0,
            "n_gen_chunks": len(stamps), "peak_rss_gib": peak_rss[0] / 2 ** 30, "min_avail_gib": min_avail[0] / 2 ** 30}


def agg(values: List[float]) -> Dict[str, float]:
    mean = statistics.fmean(values)
    return {"median": statistics.median(values), "mean": mean, "min": min(values), "max": max(values),
            "cv": (statistics.stdev(values) / mean) if len(values) > 1 and mean else 0.0}


def main() -> int:
    model, reasons = locate_gemma_gguf()
    base = {"git_sha": get_git_sha(), "utc_timestamp": now_utc()}
    if model is None:
        OUT.write_text(json.dumps({"status": "not_run", "reasons": reasons, **base}, indent=2) + "\n", encoding="utf-8")
        print(f"not_run: {reasons}")
        return 0
    import llama_cpp
    n_threads = psutil.cpu_count(logical=False) or 1
    llm = llama_cpp.Llama(model_path=str(model), n_ctx=max(BUDGETS) + N_GEN + 64, n_threads=n_threads,
                          n_gpu_layers=0, use_mmap=True, seed=20261003, verbose=False)
    runs: List[Dict[str, Any]] = []
    for k in BUDGETS:
        for iid, text in task_texts():
            prompt = exact_prompt(llm, text, k)
            for _ in range(N_WARMUP):
                measure(llm, prompt)
            for rep in range(N_REPEATS):
                runs.append({"K": k, "instance_id": iid, "repeat": rep, **measure(llm, prompt)})
    summary: Dict[str, Any] = {}
    for k in BUDGETS:
        rk = [r for r in runs if r["K"] == k]
        summary[f"K_{k}"] = {m: agg([r[m] for r in rk]) for m in ("ttft_s", "prefill_tok_s", "decode_tok_s", "peak_rss_gib")}
    summary["overall"] = {"peak_rss_gib_max": max(r["peak_rss_gib"] for r in runs),
                          "min_avail_gib_min": min(r["min_avail_gib"] for r in runs)}
    out = {"status": "measured", "model_file_name": model.name, "model_sha256": sha256_file(model),
           "backend": "llama-cpp-python", "backend_version": llama_cpp.__version__, "n_threads": n_threads,
           "n_gpu_layers": 0, "use_mmap": True, "rss_includes_mapped_weights": True,
           "host": {"cpu_physical": psutil.cpu_count(logical=False), "ram_total_gib": psutil.virtual_memory().total / 2 ** 30},
           "protocol": {"budgets": list(BUDGETS), "tasks": N_TASKS, "warmup": N_WARMUP, "repeats": N_REPEATS, "n_gen": N_GEN},
           "runs": runs, "summary": summary, **base}
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"measured: {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
