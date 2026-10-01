"""
SWE-bench Lite Dataset Loader & Streaming Adapter.

Streams real-world repository issue instances from Hugging Face (princeton-nlp/SWE-bench_Lite)
or ingests from local cached JSONL files for offline air-gapped evaluation execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "swebench_lite_cache.jsonl"


def normalize_swebench_record(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardizes a raw SWE-bench Lite record into normalized Perron evaluation fields.
    """
    instance_id = raw.get("instance_id", "unknown_instance")
    repo = raw.get("repo", "")
    base_commit = raw.get("base_commit", "HEAD")
    problem_statement = raw.get("problem_statement", "")

    # Parse test specifications
    fail_to_pass = raw.get("FAIL_TO_PASS", [])
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except Exception:
            fail_to_pass = [fail_to_pass] if fail_to_pass else []

    pass_to_pass = raw.get("PASS_TO_PASS", [])
    if isinstance(pass_to_pass, str):
        try:
            pass_to_pass = json.loads(pass_to_pass)
        except Exception:
            pass_to_pass = [pass_to_pass] if pass_to_pass else []

    golden_patch = raw.get("patch", "")
    test_patch = raw.get("test_patch", "")
    created_at = raw.get("created_at", "")
    version = raw.get("version", "")

    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "issue_text": problem_statement,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "patch": golden_patch,
        "test_patch": test_patch,
        "created_at": created_at,
        "version": version,
    }


def stream_hf_swebench_lite(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Streams instances from Hugging Face datasets library without full upfront download.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Streaming SWE-bench Lite requires the 'datasets' package. "
            "Install it via 'pip install datasets'."
        ) from e

    ds = load_dataset(dataset_name, split=split, streaming=True)
    count = 0
    for raw in ds:
        yield normalize_swebench_record(raw)
        count += 1
        if limit and count >= limit:
            break


def load_cached_swebench_lite(cache_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads normalized SWE-bench Lite instances from a local JSONL cache file.
    """
    if not cache_path.is_file():
        raise FileNotFoundError(f"Local SWE-bench cache not found at: {cache_path}")

    instances: List[Dict[str, Any]] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            instances.append(normalize_swebench_record(raw))
            if limit and len(instances) >= limit:
                break
    return instances


def load_swebench_lite(
    split: str = "test",
    cache_path: Optional[Path] = None,
    limit: Optional[int] = None,
    use_cache_first: bool = True,
) -> List[Dict[str, Any]]:
    """
    Tiered loader: uses local JSONL cache if present, otherwise streams from Hugging Face.
    """
    actual_cache = cache_path or DEFAULT_CACHE_PATH

    # 1. Try local cache if requested and exists
    if use_cache_first and actual_cache.is_file():
        try:
            return load_cached_swebench_lite(actual_cache, limit=limit)
        except Exception as e:
            print(f"[WARN] Failed to read from cache {actual_cache}: {e}. Falling back to Hugging Face stream.")

    # 2. Stream from Hugging Face
    try:
        instances = list(stream_hf_swebench_lite(split=split, limit=limit))
        return instances
    except Exception as e:
        # If offline and cache exists, try cache regardless
        if actual_cache.is_file():
            print(f"[WARN] Hugging Face stream failed ({e}); using offline cache {actual_cache}.")
            return load_cached_swebench_lite(actual_cache, limit=limit)
        raise RuntimeError(
            f"Unable to load SWE-bench Lite from Hugging Face ({e}) and no local cache found at {actual_cache}."
        ) from e


def cache_swebench_lite(
    output_path: Optional[Path] = None,
    split: str = "test",
    limit: Optional[int] = None,
) -> Path:
    """
    Downloads and caches SWE-bench Lite instances to a local JSONL file for offline execution.
    """
    dest = output_path or DEFAULT_CACHE_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Caching SWE-bench Lite ({split}) to {dest}...")
    count = 0
    with open(dest, "w", encoding="utf-8") as f:
        for item in stream_hf_swebench_lite(split=split, limit=limit):
            f.write(json.dumps(item) + "\n")
            count += 1
            if count % 25 == 0:
                print(f"  -> Cached {count} instances...")

    print(f"Finished caching {count} SWE-bench Lite instances to {dest}.")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Perron SWE-bench Lite Dataset Ingestion Tool")
    parser.add_argument("--cache", action="store_true", help="Download and cache SWE-bench Lite to local JSONL")
    parser.add_argument("--output", type=str, default=str(DEFAULT_CACHE_PATH), help="Destination JSONL path")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "train"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of instances to load or cache")
    parser.add_argument("--summary", action="store_true", help="Print summary of cached instances")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()

    if args.cache:
        cache_swebench_lite(output_path=out_path, split=args.split, limit=args.limit)
    elif args.summary or not args.cache:
        try:
            instances = load_swebench_lite(split=args.split, cache_path=out_path, limit=args.limit)
            print(f"Successfully loaded {len(instances)} SWE-bench Lite instances.")
            if instances:
                first = instances[0]
                print(f"Sample Instance: {first['instance_id']}")
                print(f"  Repo: {first['repo']}")
                print(f"  Base Commit: {first['base_commit']}")
                print(f"  Fail to pass tests: {len(first['fail_to_pass'])}")
        except Exception as e:
            print(f"Error loading SWE-bench Lite: {e}")


if __name__ == "__main__":
    main()
