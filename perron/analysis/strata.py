"""
Deterministic regex-based instance stratification (T2.1).

Labels each instance by how findable it is lexically in issue text:
- A_named (A): Issue text contains the exact gold function name (or qualified name or Class.method).
- B_trace (B): Not A, but a traceback frame in the issue points to a gold file.
- C_unnamed (C): Neither A nor B.

Used for empirical analysis only, never for retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Set, Tuple

STRATUM_A = "A"
STRATUM_B = "B"
STRATUM_C = "C"

STRATA_LABELS = {
    STRATUM_A: "A_named",
    STRATUM_B: "B_trace",
    STRATUM_C: "C_unnamed",
}

# Regex for standard Python traceback lines: File "...", line 123
TRACEBACK_FRAME_REGEX = re.compile(
    r'File\s+["\']?([^"\'\r\n]+\.py)["\']?,\s+line\s+\d+',
    re.IGNORECASE,
)

# Regex for pytest / console trace frames: path/to/file.py:123: in func
PYTEST_FRAME_REGEX = re.compile(
    r'(?:^|\n)\s*([a-zA-Z0-9_\-\./\\]+\.py):(\d+):',
    re.MULTILINE,
)

# Regex for traceback arrow pointers: --> 123 in path/to/file.py
ARROW_FRAME_REGEX = re.compile(
    r'(?:-->|-{2,}>)\s*\d+\s+in\s+([a-zA-Z0-9_\-\./\\]+\.py)',
    re.IGNORECASE,
)


def extract_candidate_function_names(gold_symbol_names: List[str]) -> Set[str]:
    """
    Extract set of candidate exact function names, qualified names, and Class.methods.
    """
    candidates: Set[str] = set()
    for sym in gold_symbol_names:
        if not sym or not isinstance(sym, str):
            continue
        cleaned = sym.strip()
        if not cleaned:
            continue
        candidates.add(cleaned)
        # If Class.method or qualname with dots
        if "." in cleaned:
            parts = cleaned.split(".")
            # terminal function / method name
            candidates.add(parts[-1])
            # Class.method if more than 2 parts
            if len(parts) >= 2:
                candidates.add(f"{parts[-2]}.{parts[-1]}")
    return {c for c in candidates if c}


def matches_named_function(issue_text: str, gold_symbol_names: List[str]) -> bool:
    """
    Check if issue text contains the exact gold function name (or qualified name or Class.method).
    Uses strict identifier boundary checks.
    """
    if not issue_text or not gold_symbol_names:
        return False

    candidates = extract_candidate_function_names(gold_symbol_names)
    for cand in candidates:
        # Strict identifier boundary: not preceded or followed by word chars
        pattern = rf"(?<![a-zA-Z0-9_]){re.escape(cand)}(?![a-zA-Z0-9_])"
        if re.search(pattern, issue_text):
            return True
    return False


def normalize_file_path(p: str) -> str:
    """Normalize file path to POSIX forward slashes, stripping drive letters."""
    norm = p.replace("\\", "/").strip()
    # Strip drive letter like C:/ if present
    if len(norm) > 2 and norm[1] == ":" and norm[2] == "/":
        norm = norm[2:]
    return norm.lstrip("/")


def matches_traceback_gold_file(issue_text: str, gold_files: List[str]) -> bool:
    """
    Check if a traceback frame in the issue text points to any gold file.
    """
    if not issue_text or not gold_files:
        return False

    norm_gold_files = [normalize_file_path(gf) for gf in gold_files if gf]
    if not norm_gold_files:
        return False

    # Extract all file paths from traceback frames
    frame_paths: List[str] = []
    frame_paths.extend(TRACEBACK_FRAME_REGEX.findall(issue_text))
    frame_paths.extend([m[0] for m in PYTEST_FRAME_REGEX.findall(issue_text)])
    frame_paths.extend(ARROW_FRAME_REGEX.findall(issue_text))

    for raw_frame in frame_paths:
        norm_frame = normalize_file_path(raw_frame)
        for gf in norm_gold_files:
            # Check exact suffix match (e.g. /opt/repo/django/models.py ends with django/models.py)
            if norm_frame.endswith(gf):
                return True
            # Or if gold file path is within frame path
            if gf in norm_frame:
                return True
            # Or if relative frame path matches tail of gold file
            if norm_frame in gf and len(norm_frame.split("/")) >= 2:
                return True
            # Match basename + parent folder if both present
            gf_parts = gf.split("/")
            frame_parts = norm_frame.split("/")
            if len(gf_parts) >= 2 and len(frame_parts) >= 2:
                if gf_parts[-1] == frame_parts[-1] and gf_parts[-2] == frame_parts[-2]:
                    return True

    return False


def classify_stratum(
    issue_text: str,
    gold_files: List[str],
    gold_symbol_names: List[str],
) -> str:
    """
    Classify an instance into Stratum A, B, or C:
    - A: issue text contains the exact gold function name.
    - B: not A, but a traceback frame in the issue points to a gold file.
    - C: neither A nor B.
    """
    if matches_named_function(issue_text, gold_symbol_names):
        return STRATUM_A
    if matches_traceback_gold_file(issue_text, gold_files):
        return STRATUM_B
    return STRATUM_C


def label_all_instances(
    tasks_path: Path,
    labels_path: Path,
) -> Dict[str, str]:
    """
    Load tasks and gold labels, and generate {instance_id: 'A' | 'B' | 'C'}.
    """
    with open(tasks_path, "r", encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]

    with open(labels_path, "r", encoding="utf-8") as f:
        gold_labels = json.load(f)

    strata_map: Dict[str, str] = {}
    for task in tasks:
        iid = task.get("instance_id", "")
        if not iid:
            continue
        issue_text = task.get("problem_statement") or ""
        gold_info = gold_labels.get(iid, {})
        gold_files = gold_info.get("gold_files", [])
        gold_symbols = gold_info.get("gold_symbol_names", [])

        stratum = classify_stratum(issue_text, gold_files, gold_symbols)
        strata_map[iid] = stratum

    return strata_map


def generate_strata_file(
    tasks_path: Path | None = None,
    labels_path: Path | None = None,
    output_path: Path | None = None,
) -> Tuple[Path, Dict[str, int]]:
    """
    Generate data/strata.json and return output path and stratum counts.
    """
    root = Path(__file__).resolve().parent.parent.parent
    if tasks_path is None:
        tasks_path = root / "data" / "swebench_lite_cache.jsonl"
    if labels_path is None:
        labels_path = root / "data" / "swebench_lite_gold_labels.json"
    if output_path is None:
        output_path = root / "data" / "strata.json"

    strata_map = label_all_instances(tasks_path, labels_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(strata_map, f, indent=2, sort_keys=True)

    counts = {
        STRATUM_A: sum(1 for s in strata_map.values() if s == STRATUM_A),
        STRATUM_B: sum(1 for s in strata_map.values() if s == STRATUM_B),
        STRATUM_C: sum(1 for s in strata_map.values() if s == STRATUM_C),
    }

    return output_path, counts


if __name__ == "__main__":
    out_p, counts = generate_strata_file()
    print(f"Generated strata map at: {out_p}")
    print(f"Stratum Counts (N={sum(counts.values())}):")
    print(f"  Stratum A (A_named):   {counts[STRATUM_A]} ({counts[STRATUM_A] / sum(counts.values()) * 100:.1f}%)")
    print(f"  Stratum B (B_trace):   {counts[STRATUM_B]} ({counts[STRATUM_B] / sum(counts.values()) * 100:.1f}%)")
    print(f"  Stratum C (C_unnamed): {counts[STRATUM_C]} ({counts[STRATUM_C] / sum(counts.values()) * 100:.1f}%)")
