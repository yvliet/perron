"""
Perron AST-Grounded Editor Module.

Executes anchor-bounded search-and-replace edits with syntax validation,
AST-guided import placement, and block-preserving indentation rebasing.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple, Union


def insert_imports_safely(source_code: str, new_imports: List[str]) -> str:
    """
    Insert new imports safely into Python source code.
    Places imports AFTER module docstring and __future__ statements,
    preventing SyntaxError: from __future__ imports must occur at beginning of file.
    """
    if not new_imports:
        return source_code

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return "\n".join(new_imports) + "\n" + source_code

    existing_import_lines = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            existing_import_lines.add(ast.unparse(node).strip())

    filtered_imports = [imp.strip() for imp in new_imports if imp.strip() not in existing_import_lines]
    if not filtered_imports:
        return source_code

    lines = source_code.splitlines(keepends=True)
    insert_line_idx = 0

    # Handle shebang or encoding cookies at top of file
    if lines:
        if lines[0].startswith("#!") or "coding" in lines[0]:
            insert_line_idx = 1
            if len(lines) > 1 and "coding" in lines[1]:
                insert_line_idx = 2

    body_idx = 0
    # 1. Check if first AST statement is module docstring
    if tree.body and isinstance(tree.body[0], ast.Expr):
        val = tree.body[0].value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            end_line = getattr(tree.body[0], "end_lineno", tree.body[0].lineno)
            insert_line_idx = max(insert_line_idx, end_line)
            body_idx = 1

    # 2. Advance past all from __future__ import ... statements
    while body_idx < len(tree.body):
        stmt = tree.body[body_idx]
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            end_line = getattr(stmt, "end_lineno", stmt.lineno)
            insert_line_idx = max(insert_line_idx, end_line)
            body_idx += 1
        else:
            break

    import_block = "\n".join(filtered_imports) + "\n"
    if insert_line_idx < len(lines):
        lines.insert(insert_line_idx, import_block)
    else:
        lines.append("\n" + import_block)

    return "".join(lines)


def find_block_match(
    window_content: str,
    target_str: str,
) -> Tuple[Optional[str], Union[int, str], int]:
    """
    Find matching block within window content, supporting exact verbatim matching
    and indentation-tolerant relative block matching. Preserves tab vs space characters.

    Returns:
        (matched_substring, base_indentation_prefix, count_of_matches)
    """
    # 1. Exact match attempt
    exact_count = window_content.count(target_str)
    if exact_count == 1:
        match_offset = window_content.find(target_str)
        leading_newlines = len(target_str) - len(target_str.lstrip("\n"))
        first_line_start_in_win = match_offset + leading_newlines
        line_start = (
            window_content.rfind("\n", 0, first_line_start_in_win) + 1
            if "\n" in window_content[:first_line_start_in_win]
            else 0
        )
        preceding_prefix = window_content[line_start:first_line_start_in_win]

        if preceding_prefix.strip() == "":
            if len(preceding_prefix) > 0:
                # Target begins after line-start indentation.
                # Expand matched string to include line-start indentation to prevent duplicate indentation on rebase.
                base_indent = preceding_prefix
                matched_block = window_content[line_start : match_offset + len(target_str)]
                return matched_block, base_indent, 1
            else:
                # Target begins at line_start. Extract existing base indentation of the target block.
                target_lines = target_str.splitlines()
                non_empty = [l for l in target_lines if l.strip()]
                base_indent = non_empty[0][:len(non_empty[0]) - len(non_empty[0].lstrip())] if non_empty else ""
                return target_str, base_indent, 1
        else:
            # Inline expression replacement inside a line
            return target_str, "", 1

    # 2. Indentation-tolerant line-by-line block matching
    win_lines = window_content.splitlines(keepends=True)
    target_lines = target_str.splitlines()
    non_empty = [l for l in target_lines if l.strip()]
    if not non_empty:
        return None, "", 0

    common_target_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    norm_targets = [l[common_target_indent:] if l.strip() else "" for l in target_lines]

    matches = []
    num_target_lines = len(norm_targets)

    for i in range(len(win_lines) - num_target_lines + 1):
        if win_lines[i].strip() != norm_targets[0].strip():
            continue

        cand_base_prefix = win_lines[i][:len(win_lines[i]) - len(win_lines[i].lstrip())]
        cand_base_indent = len(cand_base_prefix)
        matched = True

        for k in range(1, num_target_lines):
            curr_target = norm_targets[k]
            curr_win = win_lines[i + k]

            if not curr_target.strip():
                if curr_win.strip():
                    matched = False
                    break
            else:
                if curr_win.strip() != curr_target.strip():
                    matched = False
                    break

                target_rel = len(curr_target) - len(curr_target.lstrip())
                win_rel = (len(curr_win) - len(curr_win.lstrip())) - cand_base_indent
                if win_rel != target_rel:
                    matched = False
                    break

        if matched:
            matched_slice = "".join(win_lines[i : i + num_target_lines])
            matches.append((matched_slice, cand_base_prefix))

    if len(matches) == 1:
        return matches[0][0], matches[0][1], 1
    elif len(matches) > 1:
        return None, "", len(matches)

    return None, "", 0


@dataclass
class SymbolEditSpec:
    """
    Specification for a single scoped symbol replacement edit.
    """
    file_path: Path | str
    old_str: str
    new_str: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    add_imports: Optional[List[str]] = None
    window_slack: int = 15


def _apply_single_edit_to_text(
    raw_content: str,
    old_str: str,
    new_str: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    add_imports: Optional[List[str]] = None,
    window_slack: int = 15,
) -> Tuple[bool, str, str]:
    """
    In-memory application of a scoped search-and-replace edit.

    Returns:
        (success: bool, modified_content_or_original: str, message: str)
    """
    if not raw_content and (not old_str or not old_str.strip()):
        return True, new_str, "New file created."

    if not old_str or not old_str.strip():
        return False, raw_content, "Target string 'old_str' cannot be empty or purely whitespace."

    target_newline = "\r\n" if "\r\n" in raw_content else "\n"
    lines = raw_content.splitlines(keepends=True)
    num_lines = len(lines)

    if start_line is not None and end_line is not None:
        if start_line > end_line:
            return False, raw_content, f"Invalid line range: start_line ({start_line}) must be <= end_line ({end_line})."
        search_start = max(0, start_line - 1 - window_slack)
        search_end = min(num_lines, end_line + window_slack)
    else:
        search_start = 0
        search_end = num_lines

    window_content = "".join(lines[search_start:search_end])
    norm_window = window_content.replace("\r\n", "\n")
    norm_old = old_str.replace("\r\n", "\n")

    matched_target, base_indent, match_count = find_block_match(norm_window, norm_old)

    if match_count == 0:
        # Tier 2 Fallback: If line window search yielded 0 matches, attempt full-file recovery
        if start_line is not None and end_line is not None:
            full_norm = raw_content.replace("\r\n", "\n")
            f_matched, f_indent, f_count = find_block_match(full_norm, norm_old)
            if f_count == 1:
                matched_target = f_matched
                base_indent = f_indent
                match_count = f_count
                search_start = 0
                search_end = num_lines
                norm_window = full_norm
            elif f_count > 1:
                return False, raw_content, f"Could not locate target string within scope lines [{search_start+1}, {search_end}], and full-file fallback was ambiguous ({f_count} matches)."
            else:
                return False, raw_content, f"Could not locate target string within scope lines [{search_start+1}, {search_end}] or anywhere in the file."
        else:
            return False, raw_content, f"Could not locate target string in file."

    if match_count > 1:
        return False, raw_content, f"Target string matched {match_count} times within scope. Ambiguous replacement rejected."

    if isinstance(base_indent, str):
        indent_prefix = base_indent
    else:
        indent_prefix = " " * base_indent

    new_lines = new_str.splitlines(keepends=True)
    first_non_empty = next((l for l in new_lines if l.strip()), None)

    if first_non_empty:
        lead_ws = first_non_empty[:len(first_non_empty) - len(first_non_empty.lstrip())]
        if lead_ws == indent_prefix:
            # new_str is already indented to the target prefix
            reindented_new = new_str
        elif lead_ws:
            # Rebase relative to first non-empty line
            reindented_parts = []
            for l in new_lines:
                if l.startswith(lead_ws):
                    reindented_parts.append(indent_prefix + l[len(lead_ws):])
                else:
                    reindented_parts.append(l)
            reindented_new = "".join(reindented_parts)
        else:
            # new_str is unindented at column 0
            dedented_new = textwrap.dedent(new_str)
            if "\t" in indent_prefix:
                dedented_new = dedented_new.replace("    ", "\t")
            reindented_new = textwrap.indent(dedented_new, indent_prefix)
    else:
        reindented_new = new_str

    # Match newline ending of target
    if matched_target.endswith("\n") and not reindented_new.endswith("\n"):
        reindented_new += "\n"

    updated_window = norm_window.replace(matched_target, reindented_new, 1)

    modified_content = (
        "".join(lines[:search_start])
        + updated_window
        + "".join(lines[search_end:])
    )

    if add_imports:
        modified_content = insert_imports_safely(modified_content, add_imports)

    # Normalize newlines uniformly to preserve original file line-ending convention
    modified_content = modified_content.replace("\r\n", "\n").replace("\n", target_newline)
    return True, modified_content, "Edit applied in-memory."


def apply_symbol_edit(
    file_path: Path | str,
    old_str: str,
    new_str: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    add_imports: Optional[List[str]] = None,
    window_slack: int = 15,
) -> Tuple[bool, str]:
    """
    Apply a scoped search-and-replace edit to a single file on disk with AST validation.

    Guarantees:
    1. Scope-bounded matching: Searches within [start_line - slack, end_line + slack].
    2. Indentation-tolerant block matching: Resolves dedented queries cleanly.
    3. Block-preserving indentation: Uses textwrap.dedent and textwrap.indent.
    4. Syntax verification: Executes ast.parse() before saving to disk.
    5. Safe import insertion: Preserves docstrings and __future__ imports.
    """
    path = Path(file_path)
    if not path.is_file():
        return False, f"File not found: {file_path}"

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            raw_content = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", newline="") as f:
            raw_content = f.read()

    success, modified_content, msg = _apply_single_edit_to_text(
        raw_content=raw_content,
        old_str=old_str,
        new_str=new_str,
        start_line=start_line,
        end_line=end_line,
        add_imports=add_imports,
        window_slack=window_slack,
    )
    if not success:
        return False, msg

    try:
        ast.parse(modified_content)
    except SyntaxError as e:
        return False, f"Syntax verification failed: {e.msg} at line {e.lineno}"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            encoding="utf-8",
            newline="",
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(modified_content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception as e:
        if temp_path and temp_path.exists():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False, f"Atomic file write failed: {e}"

    return True, "Edit applied and verified successfully."


def apply_multi_file_patch(
    edits: Sequence[SymbolEditSpec | dict],
) -> Tuple[bool, str, List[Path]]:
    """
    Apply a transactional batch of symbol edits across one or more files.
    Enforces all-or-nothing atomicity: if any edit fails matching, ambiguity,
    AST syntax verification, or disk I/O, all modified files are rolled back to original snapshots.

    Returns:
        (success: bool, message: str, modified_paths: List[Path])
    """
    if not edits:
        return False, "No edits provided in batch.", []

    # 1. Normalize specs and verify file presence
    specs: List[SymbolEditSpec] = []
    for idx, e in enumerate(edits):
        if isinstance(e, SymbolEditSpec):
            specs.append(e)
        elif isinstance(e, dict):
            specs.append(SymbolEditSpec(
                file_path=e["file_path"],
                old_str=e["old_str"],
                new_str=e["new_str"],
                start_line=e.get("start_line"),
                end_line=e.get("end_line"),
                add_imports=e.get("add_imports"),
                window_slack=e.get("window_slack", 15),
            ))
        else:
            return False, f"Invalid edit specification at index {idx}: expected SymbolEditSpec or dict.", []

    target_paths = {}
    original_exists: Dict[Path, bool] = {}
    for s in specs:
        p = Path(s.file_path).resolve()
        is_file = p.is_file()
        original_exists[p] = is_file
        if not is_file:
            if s.old_str and s.old_str.strip():
                return False, f"Target file not found: {s.file_path}", []
        target_paths[p] = True

    # 2. Read snapshots of all unique files for rollback safety
    staged_contents: Dict[Path, str] = {}
    original_snapshots: Dict[Path, bytes] = {}
    for p in target_paths:
        if original_exists[p]:
            try:
                original_snapshots[p] = p.read_bytes()
                try:
                    staged_contents[p] = original_snapshots[p].decode("utf-8")
                except UnicodeDecodeError:
                    staged_contents[p] = original_snapshots[p].decode("latin-1")
            except Exception as e:
                return False, f"Failed reading snapshot of {p}: {e}", []
        else:
            original_snapshots[p] = b""
            staged_contents[p] = ""

    # 3. Sequentially apply edits in memory per file
    # To prevent coordinate drift across multiple edits in the same file:
    # (a) Group edits by file.
    # (b) Apply code edits in descending order of start_line (bottom-to-top), so modifying a lower span never shifts line indices of higher spans.
    # (c) Aggregate and insert any new imports at the top of the file *after* all code replacements are complete, avoiding top-of-file line shifts during editing.
    from collections import defaultdict
    file_to_specs = defaultdict(list)
    for s in specs:
        file_to_specs[Path(s.file_path).resolve()].append(s)

    for p, file_specs in file_to_specs.items():
        current_text = staged_contents[p]
        # Aggregate all imports needed for this file
        aggregated_imports: List[str] = []
        for s in file_specs:
            if s.add_imports:
                for imp in s.add_imports:
                    if imp not in aggregated_imports:
                        aggregated_imports.append(imp)

        # Sort code edits in descending order of start_line (bottom-to-top)
        # Edits without start_line are applied last (start_line=-1)
        sorted_specs = sorted(
            file_specs,
            key=lambda item: item.start_line if item.start_line is not None else -1,
            reverse=True,
        )

        for s in sorted_specs:
            success, next_text, msg = _apply_single_edit_to_text(
                raw_content=current_text,
                old_str=s.old_str,
                new_str=s.new_str,
                start_line=s.start_line,
                end_line=s.end_line,
                add_imports=None,  # Handled in single aggregated pass below
                window_slack=s.window_slack,
            )
            if not success:
                return False, f"Edit failed on {p.name}: {msg}", []
            current_text = next_text

        # Apply aggregated imports safely to the top of the file
        if aggregated_imports:
            current_text = insert_imports_safely(current_text, aggregated_imports)

        staged_contents[p] = current_text

    # 4. AST validation for all modified Python files
    for p, text in staged_contents.items():
        if p.suffix == ".py":
            try:
                ast.parse(text)
            except SyntaxError as e:
                return False, f"Transaction aborted: AST syntax error in {p.name}: {e.msg} at line {e.lineno}", []

    # 5. Commit all staged files atomically to disk with rollback guarantee
    committed_paths: List[Path] = []
    for p, text in staged_contents.items():
        temp_path = None
        try:
            if not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=p.parent,
                delete=False,
                encoding="utf-8",
                newline="",
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(text)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, p)
            committed_paths.append(p)
        except Exception as e:
            if temp_path and temp_path.exists():
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            # Rollback all already-committed files to original snapshots
            for rb_path in committed_paths:
                try:
                    if not original_exists.get(rb_path, True):
                        if rb_path.exists():
                            rb_path.unlink()
                    else:
                        rb_path.write_bytes(original_snapshots[rb_path])
                except Exception:
                    pass
            return False, f"Transaction aborted while writing {p.name} (all {len(committed_paths)} modified files rolled back): {e}", []

    return True, f"Successfully committed {len(specs)} edits across {len(committed_paths)} files.", committed_paths
