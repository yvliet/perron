"""
Canonical Patch-to-AST Gold Node Extractor & Dataset Splitter.

Extracts ground-truth modified files and AST symbols from human unified diff patches
across all 300 SWE-bench Lite task instances. Partitions tasks into a stratified 150/150
dev/held-out split with deterministic random seed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def get_repo_slug(repo_name: str) -> str:
    """Convert repository name (e.g. 'psf/requests') to filesystem slug ('psf__requests')."""
    return repo_name.replace("/", "__")


def file_path_matches(f_patch: str, s_file: str) -> bool:
    """Match relative file paths strictly using exact match or path segment suffix."""
    f_p = f_patch.replace("\\", "/").lstrip("/")
    s_f = s_file.replace("\\", "/").lstrip("/")
    return f_p == s_f or f_p.endswith("/" + s_f) or s_f.endswith("/" + f_p)


def parse_unified_diff(patch_str: str) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    """
    Parse a unified diff string.
    Returns:
        files: List of unique modified file paths.
        hunks: List of (file_path, old_start_line, old_line_count) tuples.
    """
    files: Set[str] = set()
    hunks: List[Tuple[str, int, int]] = []
    current_file: Optional[str] = None

    for line in patch_str.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                f = parts[3]
                if f.startswith("b/"):
                    f = f[2:]
                current_file = f
                files.add(f)
        elif line.startswith("@@") and current_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                hunks.append((current_file, old_start, old_count))

    return sorted(list(files)), hunks


class GoldPatchParser:
    """
    Extracts gold files, symbol nodes, and hub status for SWE-bench Lite tasks
    using exact AST pre-image line range interval intersection.
    """

    def __init__(
        self,
        graphs_dir: str | Path = "artifacts/swebench_graphs",
        tasks_file: str | Path = "data/swebench_lite_cache.jsonl",
    ):
        self.graphs_dir = Path(graphs_dir)
        self.tasks_file = Path(tasks_file)
        self.repo_symbols: Dict[str, List[Dict[str, Any]]] = {}
        self.repo_hub_ids: Dict[str, Set[int]] = {}
        self._load_repo_metadata()

    def _load_repo_metadata(self) -> None:
        """Load symbols and precompute top-25 hub IDs for each repository."""
        for p in self.graphs_dir.glob("*_symbols.json"):
            slug = p.stem.replace("_symbols", "")
            try:
                with open(p, "r", encoding="utf-8") as f:
                    symbols = json.load(f)
                self.repo_symbols[slug] = symbols
                # Compute top-25 in-degree hubs
                sorted_by_degree = sorted(
                    symbols, key=lambda s: s.get("degree", 0), reverse=True
                )
                top_25 = {s["id"] for s in sorted_by_degree[:25]}
                self.repo_hub_ids[slug] = top_25
            except Exception as e:
                print(f"Warning: Failed loading symbols for {slug}: {e}")

    def extract_gold_for_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Extract gold files, gold symbol IDs, and hub status for a single task."""
        repo = task.get("repo", "")
        instance_id = task.get("instance_id", "")
        patch = task.get("patch", "")
        slug = get_repo_slug(repo)

        symbols = self.repo_symbols.get(slug, [])
        hub_ids = self.repo_hub_ids.get(slug, set())

        gold_files, hunks = parse_unified_diff(patch)
        gold_symbol_ids: Set[int] = set()
        gold_symbol_names: Set[str] = set()
        is_hub_gold = False

        for f_patch, start, count in hunks:
            hunk_start = start
            hunk_end = start + max(0, count - 1)
            candidates: List[Dict[str, Any]] = []

            for s in symbols:
                if not file_path_matches(f_patch, s.get("file_path", "")):
                    continue
                s_start = s.get("line_start", 0)
                s_end = s.get("line_end", 0)
                if s_start <= 0 or s_end < s_start:
                    continue

                if count == 0:
                    # Pure addition at start
                    intersects = (s_start <= hunk_start <= s_end)
                else:
                    intersects = not (hunk_end < s_start or hunk_start > s_end)

                if intersects:
                    candidates.append(s)

            if candidates:
                type_prio = {"function": 0, "method": 0, "class": 1}
                candidates.sort(
                    key=lambda sym: (
                        type_prio.get(sym.get("symbol_type", ""), 2),
                        sym.get("line_end", 0) - sym.get("line_start", 0),
                    )
                )
                innermost = candidates[0]
                sym_id = innermost["id"]
                gold_symbol_ids.add(sym_id)
                ident = innermost.get("identifier", "")
                if ident:
                    gold_symbol_names.add(ident)
                if sym_id in hub_ids:
                    is_hub_gold = True

        return {
            "instance_id": instance_id,
            "repo": repo,
            "gold_files": gold_files,
            "gold_symbol_ids": sorted(list(gold_symbol_ids)),
            "gold_symbol_names": sorted(list(gold_symbol_names)),
            "is_hub_gold": is_hub_gold,
            "has_function_target": len(gold_symbol_ids) > 0,
        }

    def build_all_gold_labels(self) -> Dict[str, Dict[str, Any]]:
        """Extract gold labels across all tasks and return dictionary keyed by instance_id."""
        labels: Dict[str, Dict[str, Any]] = {}
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                task = json.loads(line)
                info = self.extract_gold_for_task(task)
                labels[info["instance_id"]] = info
        return labels

    def save_gold_labels(self, out_path: str | Path = "data/swebench_lite_gold_labels.json") -> None:
        """Save extracted gold labels to JSON."""
        labels = self.build_all_gold_labels()
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)
        print(f"Saved {len(labels)} gold labels to {out_p}")

    def export_gold_jsonl(
        self,
        out_path: str | Path = "data_release/gold.jsonl",
        strata_file: str | Path = "data/strata.json",
        split_file: str | Path = "data/swebench_lite_split.json",
    ) -> None:
        """
        Export authoritative gold labels to data_release/gold.jsonl with:
        instance_id, repo, gold_files, gold_functions, gold_in_degree_percentile,
        is_hub_gold, stratum, and split.
        """
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        strata_map: Dict[str, str] = {}
        strata_p = Path(strata_file)
        if strata_p.exists():
            with open(strata_p, "r", encoding="utf-8") as f:
                strata_map = json.load(f)

        split_map: Dict[str, str] = {}
        split_p = Path(split_file)
        if split_p.exists():
            with open(split_p, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            for iid in split_data.get("dev_pilot_instances", []):
                split_map[iid] = "pilot"
            for iid in split_data.get("dev_val_instances", []):
                split_map[iid] = "dev_val"
            for iid in split_data.get("heldout_instances", []):
                split_map[iid] = "heldout"

        labels = self.build_all_gold_labels()

        repo_degrees: Dict[str, List[int]] = {}
        repo_sym_by_id: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for slug, syms in self.repo_symbols.items():
            repo_degrees[slug] = sorted(s.get("degree", 0) for s in syms)
            repo_sym_by_id[slug] = {s["id"]: s for s in syms}

        with open(out_p, "w", encoding="utf-8") as f:
            for instance_id, info in sorted(labels.items()):
                repo = info.get("repo", "")
                slug = get_repo_slug(repo)
                gold_sym_ids = info.get("gold_symbol_ids", [])

                degs = repo_degrees.get(slug, [])
                sym_map = repo_sym_by_id.get(slug, {})
                if gold_sym_ids and degs:
                    max_gold_deg = max(sym_map.get(sid, {}).get("degree", 0) for sid in gold_sym_ids)
                    pct = round((sum(1 for d in degs if d <= max_gold_deg) / len(degs)) * 100.0, 2)
                else:
                    pct = 0.0

                record = {
                    "instance_id": instance_id,
                    "repo": repo,
                    "gold_files": info.get("gold_files", []),
                    "gold_functions": info.get("gold_symbol_names", []),
                    "gold_in_degree_percentile": pct,
                    "is_hub_gold": bool(info.get("is_hub_gold", False) or pct >= 99.0),
                    "stratum": strata_map.get(instance_id, "C"),
                    "split": split_map.get(instance_id, "heldout"),
                }
                f.write(json.dumps(record) + "\n")
        print(f"Exported {len(labels)} gold records to {out_p}")


def build_tri_partition_split(
    tasks_file: str | Path = "data/swebench_lite_cache.jsonl",
    labels_file: str | Path = "data/swebench_lite_gold_labels.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build a tri-partition split (Dev-Pilot N=50, Dev-Val N=100, Held-out N=150)
    with SHA-256 integrity pinning.
    """
    import hashlib
    import random

    rng = random.Random(seed)
    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    # Isolate original 50 exploratory pilot tasks (6 Requests + first 44 SymPy)
    req_instances = [i["instance_id"] for i in tasks if "requests" in i["repo"]]
    sym_instances = [i["instance_id"] for i in tasks if "sympy" in i["repo"]]
    dev_pilot = sorted(req_instances + sym_instances[: 50 - len(req_instances)])
    assert len(dev_pilot) == 50, f"Expected 50 pilot tasks, got {len(dev_pilot)}"

    pilot_set = set(dev_pilot)
    remaining_tasks = [t for t in tasks if t["instance_id"] not in pilot_set]
    assert len(remaining_tasks) == 250

    # Stratify remaining 250 tasks into 100 dev_val and 150 heldout proportionally by repo
    repo_to_tasks: Dict[str, List[str]] = {}
    for t in remaining_tasks:
        repo_to_tasks.setdefault(t["repo"], []).append(t["instance_id"])

    dev_val: List[str] = []
    heldout: List[str] = []

    for repo in sorted(repo_to_tasks.keys()):
        insts = sorted(repo_to_tasks[repo])
        rng.shuffle(insts)
        n = len(insts)
        # Allocate 40% (100/250) to dev_val, 60% (150/250) to heldout
        n_val = int(round(n * (100.0 / 250.0)))
        dev_val.extend(insts[:n_val])
        heldout.extend(insts[n_val:])

    # Exact balancing to 100 dev_val and 150 heldout
    while len(dev_val) > 100:
        heldout.append(dev_val.pop())
    while len(dev_val) < 100:
        dev_val.append(heldout.pop())

    dev_val = sorted(dev_val)
    heldout = sorted(heldout)

    assert len(set(dev_pilot) & set(heldout)) == 0, "Contamination: pilot in heldout!"
    assert len(set(dev_val) & set(heldout)) == 0, "Contamination: dev_val in heldout!"
    assert len(set(dev_pilot) & set(dev_val)) == 0, "Overlap in dev partitions!"

    # Compute SHA-256 hashes
    with open(tasks_file, "rb") as f:
        cache_sha256 = hashlib.sha256(f.read()).hexdigest()

    labels_p = Path(labels_file)
    if labels_p.exists():
        with open(labels_p, "rb") as f:
            labels_sha256 = hashlib.sha256(f.read()).hexdigest()
    else:
        labels_sha256 = ""

    split_manifest = {
        "manifest_version": "2.0.0",
        "description": "SWE-bench Lite Tri-Partition Split with Contamination Firewall",
        "seed": seed,
        "sha256_cache": cache_sha256,
        "sha256_gold_labels": labels_sha256,
        "dev_pilot_count": len(dev_pilot),
        "dev_val_count": len(dev_val),
        "dev_count": len(dev_pilot) + len(dev_val),
        "heldout_count": len(heldout),
        "dev_pilot_instances": dev_pilot,
        "dev_val_instances": dev_val,
        "heldout_instances": heldout,
        # Backward compatibility alias: dev_instances = dev_pilot + dev_val (150 instances)
        "dev_instances": sorted(dev_pilot + dev_val),
    }
    return split_manifest


def build_stratified_split(
    tasks_file: str | Path = "data/swebench_lite_cache.jsonl",
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Backward-compatibility wrapper returning (dev_instances, heldout_instances)."""
    manifest = build_tri_partition_split(tasks_file=tasks_file, seed=seed)
    return manifest["dev_instances"], manifest["heldout_instances"]


if __name__ == "__main__":
    parser = GoldPatchParser()
    parser.save_gold_labels()
    split_info = build_tri_partition_split()
    with open("data/swebench_lite_split.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print("Saved tri-partition split configuration to data/swebench_lite_split.json")
