"""
Canonical Patch-to-AST Gold Node Extractor & Dataset Splitter.

Extracts ground-truth modified files and AST symbols from human unified diff patches
across all 300 SWE-bench Lite task instances. Partitions tasks into a stratified 150/150
dev/held-out split with deterministic random seed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
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


def parse_patch_modified_files(patch_str: str) -> Dict[str, Set[int]]:
    """
    Parse a unified diff string into a mapping of file_path -> set of modified line numbers in the pre-image.
    """
    files: Dict[str, Set[int]] = {}
    current_file: Optional[str] = None
    for line in patch_str.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                f = parts[3]
                if f.startswith("b/"):
                    f = f[2:]
                current_file = f
        elif line.startswith("@@") and current_file:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                lines = set(range(old_start, old_start + max(1, old_count)))
                files.setdefault(current_file, set()).update(lines)
    return files


class EnclosingSymbolVisitor(ast.NodeVisitor):
    """AST visitor collecting classes, functions, and methods with their line spans."""
    def __init__(self):
        self.symbols: List[Dict[str, Any]] = []
        self.scope_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef):
        self.scope_stack.append(node.name)
        qname = ".".join(self.scope_stack)
        start_line = node.lineno
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)
        end_line = getattr(node, "end_lineno", node.lineno)
        self.symbols.append({
            "name": qname,
            "type": "class",
            "start_line": start_line,
            "end_line": end_line,
            "span": end_line - start_line,
        })
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node)

    def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        qname = f"{'.'.join(self.scope_stack)}.{node.name}" if self.scope_stack else node.name
        sym_type = "method" if self.scope_stack else "function"
        start_line = node.lineno
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)
        end_line = getattr(node, "end_lineno", node.lineno)
        self.symbols.append({
            "name": qname,
            "type": sym_type,
            "start_line": start_line,
            "end_line": end_line,
            "span": end_line - start_line,
        })
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()


def find_enclosing_symbol(tree: ast.AST, target_line: int) -> Optional[str]:
    """Find the innermost enclosing symbol (function/method preferred over class) for a line."""
    visitor = EnclosingSymbolVisitor()
    visitor.visit(tree)
    candidates = [
        s for s in visitor.symbols
        if s["start_line"] <= target_line <= s["end_line"]
    ]
    if not candidates:
        return None
    type_prio = {"function": 0, "method": 0, "class": 1}
    candidates.sort(key=lambda s: (type_prio.get(s["type"], 2), s["span"]))
    return candidates[0]["name"]


def extract_gold_symbols_from_base_commit(
    repo_path: Path,
    base_commit: str,
    patch_str: str,
) -> Set[str]:
    """
    Extract canonical symbol keys '(file_path, qualname)' from base_commit AST.
    Uses git show base_commit:file to eliminate cross-commit line drift.
    """
    modified_files = parse_patch_modified_files(patch_str)
    gold_qualnames: Set[str] = set()
    if not repo_path.exists() or not base_commit:
        return gold_qualnames

    for file_path, line_numbers in modified_files.items():
        if not file_path.endswith(".py"):
            continue
        try:
            res = subprocess.run(
                ["git", "show", f"{base_commit}:{file_path}"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
            if res.returncode == 0 and res.stdout.strip():
                tree = ast.parse(res.stdout)
                for target_line in line_numbers:
                    qualname = find_enclosing_symbol(tree, target_line)
                    if qualname:
                        gold_qualnames.add(f"{file_path}::{qualname}")
        except Exception:
            continue
    return gold_qualnames


class GoldPatchParser:
    """
    Extracts gold files, symbol nodes, and hub status for SWE-bench Lite tasks
    using base-commit AST qualified-name extraction and snapshot reconciliation.
    """

    def __init__(
        self,
        graphs_dir: str | Path = "artifacts/swebench_graphs",
        tasks_file: str | Path = "data/swebench_lite_cache.jsonl",
        clones_dir: str | Path = "artifacts/clones",
    ):
        self.graphs_dir = Path(graphs_dir)
        self.tasks_file = Path(tasks_file)
        self.clones_dir = Path(clones_dir)
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
        base_commit = task.get("base_commit", "")
        instance_id = task.get("instance_id", "")
        patch = task.get("patch", "")
        slug = get_repo_slug(repo)
        repo_path = self.clones_dir / slug

        symbols = self.repo_symbols.get(slug, [])
        hub_ids = self.repo_hub_ids.get(slug, set())

        gold_files, hunks = parse_unified_diff(patch)
        gold_symbol_ids: Set[int] = set()
        gold_symbol_names: Set[str] = set()
        is_hub_gold = False
        unresolvable_symbols: List[str] = []
        resolution_method = "unresolvable"

        # 1. Primary: Base-Commit AST Qualified Name Extraction
        base_qualnames = extract_gold_symbols_from_base_commit(repo_path, base_commit, patch)
        if base_qualnames:
            for item in base_qualnames:
                f_path, qname = item.split("::", 1)
                matched = False
                for s in symbols:
                    if file_path_matches(f_path, s.get("file_path", "")):
                        ident = s.get("identifier", "")
                        if ident == qname or qname.endswith("." + ident) or ident.endswith("." + qname):
                            gold_symbol_ids.add(s["id"])
                            gold_symbol_names.add(ident)
                            matched = True
                            if s["id"] in hub_ids:
                                is_hub_gold = True
                if not matched:
                    unresolvable_symbols.append(item)
            if gold_symbol_ids:
                resolution_method = "base_commit_ast"

        # 2. Secondary fallback: Snapshot line-span intersection
        if not gold_symbol_ids and hunks:
            for f_patch, start, count in hunks:
                hunk_start = start
                hunk_end = start + max(0, count - 1)
                candidates: List[Dict[str, Any]] = []

                for s in symbols:
                    if not file_path_matches(f_patch, s.get("file_path", "")):
                        continue
                    s_start = s.get("line_start") or 0
                    s_end = s.get("line_end") or 0
                    if s_start <= 0 or s_end < s_start:
                        continue

                    if count == 0:
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
                            (sym.get("line_end") or 0) - (sym.get("line_start") or 0),
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
                    resolution_method = "snapshot_line"

        return {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "gold_files": gold_files,
            "gold_symbol_ids": sorted(list(gold_symbol_ids)),
            "gold_symbol_names": sorted(list(gold_symbol_names)),
            "is_hub_gold": is_hub_gold,
            "has_function_target": len(gold_symbol_ids) > 0,
            "resolution_method": resolution_method,
            "unresolvable_symbols": unresolvable_symbols,
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
    import argparse

    cli_parser = argparse.ArgumentParser(description="Perron Gold Patch AST Parser")
    cli_parser.add_argument(
        "--validate-keys",
        action="store_true",
        help="Validate AST qualname key extraction across benchmark tasks",
    )
    cli_parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Export data_release/gold.jsonl",
    )
    cli_parser.add_argument(
        "--save-labels",
        action="store_true",
        help="Save data/swebench_lite_gold_labels.json",
    )
    cli_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process",
    )
    args = cli_parser.parse_args()

    parser = GoldPatchParser()

    if args.validate_keys:
        print("[VALIDATION] Running AST qualname key extraction validation...")
        tasks = []
        with open(parser.tasks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        if args.limit:
            tasks = tasks[: args.limit]

        total = len(tasks)
        has_func = 0
        method_counts: Dict[str, int] = {"base_commit_ast": 0, "snapshot_line": 0, "unresolvable": 0}
        ast_matches = 0
        unmatched_total = 0

        for idx, task in enumerate(tasks):
            info = parser.extract_gold_for_task(task)
            m = info.get("resolution_method", "unresolvable")
            method_counts[m] = method_counts.get(m, 0) + 1
            if info.get("has_function_target"):
                has_func += 1
                if m == "base_commit_ast":
                    ast_matches += 1
            unmatched_total += len(info.get("unresolvable_symbols", []))
            if (idx + 1) % 50 == 0 or idx + 1 == total:
                print(f"  Processed {idx + 1}/{total} tasks... (AST matches: {ast_matches}/{has_func})")

        ast_match_rate = (ast_matches / has_func * 100.0) if has_func > 0 else 0.0
        print("\n=== AST Qualname Extraction Validation Results ===")
        print(f"Total tasks evaluated: {total}")
        print(f"Tasks with function target (S_func): {has_func} ({has_func / total * 100.0:.1f}%)")
        print(f"Non-callable / unresolvable stratum: {total - has_func} ({(total - has_func) / total * 100.0:.1f}%)")
        print(f"Resolution methods: {method_counts}")
        print(f"Base-commit AST match rate on callable tasks: {ast_matches}/{has_func} ({ast_match_rate:.2f}%)")
        print(f"Total unresolvable symbols recorded: {unmatched_total}")
        assert ast_match_rate >= 80.0, f"Expected >= 80% AST match rate, got {ast_match_rate:.2f}%"
        print("[VALIDATION PASSED] AST keying verified without line drift.")
    elif args.export_jsonl:
        parser.export_gold_jsonl()
    elif args.save_labels:
        parser.save_gold_labels()
    else:
        # Default behavior: run validation
        print("No specific action requested; running key validation by default (pass --help for options).")
        parser.save_gold_labels()

