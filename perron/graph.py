"""
Perron Real Repository AST Graph Extractor.

Extracts real call and caller dependency graphs from Python codebases,
constructs row-stochastic CSR transition matrices, and serializes
zero-copy memory-mapped binary representations.

Key Invariants:
1. Composite Keying: Symbols are identified by (file_path, qualified_name),
   preventing duplicate function or method name collisions across files.
2. Verbatim Code Slicing: Functions and classes retain their exact source code
   slices via ast.get_source_segment() instead of dummy stubs.
3. Import-Aware Edge Resolution: Tracks import and import-from bindings
   per module to resolve cross-file function and class invocations.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from scipy.sparse import csr_matrix

from perron.matrix import build_static_transition_matrix, save_mmap_csr, load_mmap_csr
from perron.packer import ASTContextSymbol


@dataclass
class ASTSymbolNode:
    """
    Rich representation of an extracted AST symbol node.
    """
    node_id: int = 0
    file_path: str = ""
    qualified_name: str = ""
    symbol_type: str = "function"  # "function" | "class" | "method"
    start_line: int = 1
    end_line: int = 1
    code: str = ""
    token_count: int = 0
    parent_class: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    identifier: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    source_code: Optional[str] = None
    calls: List[str] = field(default_factory=list)
    base_classes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.identifier is not None and not self.qualified_name:
            self.qualified_name = self.identifier
        elif self.qualified_name and self.identifier is None:
            self.identifier = self.qualified_name

        if self.line_start is not None and self.start_line == 1:
            self.start_line = self.line_start
        elif self.start_line != 1 and self.line_start is None:
            self.line_start = self.start_line

        if self.line_end is not None and self.end_line == 1:
            self.end_line = self.line_end
        elif self.end_line != 1 and self.line_end is None:
            self.line_end = self.end_line

        if self.source_code is not None and not self.code:
            self.code = self.source_code
        elif self.code and self.source_code is None:
            self.source_code = self.code

        if not self.token_count and self.code:
            self.token_count = max(1, len(self.code) // 4)

    @property
    def canonical_key(self) -> str:
        return f"{self.file_path}::{self.qualified_name}"

    def to_context_symbol(self) -> ASTContextSymbol:
        return ASTContextSymbol(
            node_id=self.node_id,
            name=self.qualified_name,
            file_path=self.file_path,
            start_line=self.start_line,
            end_line=self.end_line,
            code=self.code,
            token_count=self.token_count,
            parent_class=self.parent_class,
            class_docstring=self.docstring if self.symbol_type == "class" else None,
        )


class ImportTable:
    """
    Tracks imported symbols and module aliases for a single source file.
    """
    def __init__(self, file_path: str, repo_root: Path):
        self.file_path = file_path.replace("\\", "/")
        self.repo_root = repo_root
        # alias -> (resolved_file_rel_path, target_symbol_name)
        self.imported_symbols: Dict[str, Tuple[str, str]] = {}
        # module_alias -> resolved_file_or_dir_rel_path
        self.imported_modules: Dict[str, str] = {}

    def register_import(self, node: ast.Import):
        for alias in node.names:
            asname = alias.asname or alias.name
            mod_path = alias.name.replace(".", "/")
            self.imported_modules[asname] = mod_path

    def register_import_from(self, node: ast.ImportFrom):
        module_str = node.module or ""
        level = node.level or 0

        # Resolve relative import paths
        if level > 0:
            cur_file = Path(self.file_path)
            # level 1 is same directory, level 2 is parent directory
            base_parts = list(cur_file.parent.parts)
            pop_count = level - 1
            if pop_count > 0 and len(base_parts) >= pop_count:
                base_parts = base_parts[:-pop_count]
            prefix = "/".join(base_parts)
            if module_str:
                resolved_base = f"{prefix}/{module_str.replace('.', '/')}" if prefix else module_str.replace(".", "/")
            else:
                resolved_base = prefix
        else:
            resolved_base = module_str.replace(".", "/")

        for alias in node.names:
            asname = alias.asname or alias.name
            orig_name = alias.name
            self.imported_symbols[asname] = (resolved_base, orig_name)


class FileASTCollector(ast.NodeVisitor):
    """
    Extracts class and function definitions, source slices, and raw calls from a Python file.
    """
    def __init__(self, file_path: str, source_text: str, repo_root: Path):
        self.file_path = file_path.replace("\\", "/")
        self.source_text = source_text
        self.source_lines = source_text.splitlines(keepends=True)
        self.repo_root = repo_root
        self.import_table = ImportTable(file_path, repo_root)

        self.symbols_raw: List[dict] = []
        self.raw_calls: List[Tuple[str, str]] = []  # (caller_qname, callee_name)

        self.class_stack: List[str] = []
        self.current_function: Optional[str] = None

    def visit_Import(self, node: ast.Import):
        self.import_table.register_import(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.import_table.register_import_from(node)
        self.generic_visit(node)

    def _extract_code(self, node: ast.AST) -> str:
        first_line = getattr(node, "lineno", 1)
        if hasattr(node, "decorator_list") and node.decorator_list:
            first_line = min(first_line, node.decorator_list[0].lineno)
            end_line = getattr(node, "end_lineno", first_line + 1)
            return "".join(self.source_lines[first_line - 1:end_line])

        # Prefer standard library exact segment extraction
        segment = ast.get_source_segment(self.source_text, node)
        if segment:
            return segment + ("\n" if not segment.endswith("\n") else "")
        # Fallback to line numbers
        end_line = getattr(node, "end_lineno", first_line + 1)
        return "".join(self.source_lines[first_line - 1:end_line])

    def _extract_decorators(self, decorator_list: List[ast.expr]) -> List[str]:
        decs = []
        for d in decorator_list:
            if isinstance(d, ast.Name):
                decs.append(d.id)
            elif isinstance(d, ast.Attribute):
                decs.append(d.attr)
            elif isinstance(d, ast.Call) and isinstance(d.func, (ast.Name, ast.Attribute)):
                name = d.func.id if isinstance(d.func, ast.Name) else d.func.attr
                decs.append(name)
        return decs

    def visit_ClassDef(self, node: ast.ClassDef):
        parent_class = ".".join(self.class_stack) if self.class_stack else None
        self.class_stack.append(node.name)
        qname = ".".join(self.class_stack)

        doc = ast.get_docstring(node)
        code = self._extract_code(node)
        tokens = max(15, int(len(code) / 3.5))

        start_line = node.lineno
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)

        self.symbols_raw.append({
            "file_path": self.file_path,
            "qualified_name": qname,
            "symbol_type": "class",
            "start_line": start_line,
            "end_line": getattr(node, "end_lineno", node.lineno + 10),
            "code": code,
            "token_count": tokens,
            "parent_class": parent_class,
            "docstring": doc,
            "decorators": self._extract_decorators(node.decorator_list),
        })

        # Extract inheritance hierarchy edges
        base_names: List[str] = []
        for base in node.bases:
            b_name = None
            if isinstance(base, ast.Name):
                b_name = base.id
            elif isinstance(base, ast.Attribute):
                chain = self._unwrap_attribute_chain(base)
                b_name = chain if chain else base.attr
            if b_name:
                base_names.append(b_name)

        self.symbols_raw[-1]["bases"] = base_names

        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_func(node, is_async=True)

    def _handle_func(self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool):
        parent_class = ".".join(self.class_stack) if self.class_stack else None
        func_qname = f"{parent_class}.{node.name}" if parent_class else node.name
        sym_type = "method" if parent_class else "function"

        prev_func = self.current_function
        self.current_function = func_qname

        doc = ast.get_docstring(node)
        code = self._extract_code(node)
        tokens = max(15, int(len(code) / 3.5))

        start_line = node.lineno
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)

        self.symbols_raw.append({
            "file_path": self.file_path,
            "qualified_name": func_qname,
            "symbol_type": sym_type,
            "start_line": start_line,
            "end_line": getattr(node, "end_lineno", node.lineno + 10),
            "code": code,
            "token_count": tokens,
            "parent_class": parent_class,
            "docstring": doc,
            "decorators": self._extract_decorators(node.decorator_list),
        })

        self.generic_visit(node)
        self.current_function = prev_func

    def _unwrap_attribute_chain(self, node: ast.expr) -> Optional[str]:
        parts = []
        curr = node
        while isinstance(curr, ast.Attribute):
            parts.append(curr.attr)
            curr = curr.value
        if isinstance(curr, ast.Name):
            parts.append(curr.id)
            return ".".join(reversed(parts))
        return None

    def visit_Call(self, node: ast.Call):
        if self.current_function:
            callee_name = None
            if isinstance(node.func, ast.Name):
                callee_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                chain = self._unwrap_attribute_chain(node.func)
                callee_name = chain if chain else node.func.attr
            if callee_name:
                self.raw_calls.append((self.current_function, callee_name))
        self.generic_visit(node)


def _matches_module_path(candidate_file: str, module_path: str) -> bool:
    """
    Check if a candidate file path matches the given module path safely,
    avoiding accidental substring matches (e.g. 'io' matching 'audio/player.py').
    """
    clean_mod = module_path.strip("/").replace("\\", "/")
    if not clean_mod:
        return False
    clean_file = candidate_file.strip("/").replace("\\", "/")

    # 1. Exact module file match (e.g. "perron/matrix.py" matches "perron/matrix")
    if clean_file == f"{clean_mod}.py" or clean_file.endswith(f"/{clean_mod}.py"):
        return True
    # 2. Package __init__.py match (e.g. "perron/__init__.py" matches "perron")
    if clean_file == f"{clean_mod}/__init__.py" or clean_file.endswith(f"/{clean_mod}/__init__.py"):
        return True
    # 3. Directory prefix match (e.g. file is within "perron/backends/")
    if clean_file.startswith(f"{clean_mod}/") or f"/{clean_mod}/" in clean_file:
        return True
    return False


def extract_repository_graph(
    repo_dir: Path,
    include_patterns: Optional[List[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    max_files: Optional[int] = None,
    return_multiplex: bool = False,
) -> Union[
    Tuple[List[ASTSymbolNode], List[Tuple[int, int]], List[Tuple[int, int]], Dict[str, int], Dict[int, str]],
    Tuple[List[ASTSymbolNode], List[Tuple[int, int]], List[Tuple[int, int]], Dict[str, int], Dict[int, str], List[Tuple[int, int]], List[Tuple[int, int]]],
]:
    """
    Extracts complete AST call graph from a repository directory with composite keying.

    Returns:
        (symbols, call_edges, caller_edges, canonical_to_id, id_to_canonical) or
        (symbols, call_edges, caller_edges, canonical_to_id, id_to_canonical, inherit_edges, import_edges) if return_multiplex is True.
    """
    repo_dir = repo_dir.resolve()
    if exclude_dirs is None:
        exclude_dirs = {
            ".git", "__pycache__", "venv", ".venv", "build", "dist",
            ".pytest_cache", ".eggs", "egg-info", "node_modules", ".idea",
            ".perron_index", "real_graphs_out", "scratch"
        }

    python_files: List[Path] = []
    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith(".egg-info")]
        for f in files:
            if f.endswith(".py"):
                p = Path(root) / f
                python_files.append(p)

    python_files.sort()
    if max_files is not None and max_files > 0:
        python_files = python_files[:max_files]

    all_collectors: List[FileASTCollector] = []
    symbols: List[ASTSymbolNode] = []
    canonical_to_id: Dict[str, int] = {}
    id_to_canonical: Dict[int, str] = {}

    # File and suffix indexes for fast call resolution
    file_to_symbols: Dict[str, List[int]] = {}
    suffix_to_ids: Dict[str, List[int]] = {}
    qualified_to_ids: Dict[str, List[int]] = {}

    # Pass 1: Parse AST and collect symbols
    for py_file in python_files:
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except Exception:
            try:
                source = py_file.read_text(encoding="latin-1")
                tree = ast.parse(source, filename=str(py_file))
            except Exception:
                continue

        rel_path = str(py_file.relative_to(repo_dir)).replace("\\", "/")
        collector = FileASTCollector(rel_path, source, repo_dir)
        collector.visit(tree)
        all_collectors.append(collector)

        for item in collector.symbols_raw:
            nid = len(symbols)
            sym = ASTSymbolNode(
                node_id=nid,
                file_path=item["file_path"],
                qualified_name=item["qualified_name"],
                symbol_type=item["symbol_type"],
                start_line=item["start_line"],
                end_line=item["end_line"],
                code=item["code"],
                token_count=item["token_count"],
                parent_class=item["parent_class"],
                docstring=item["docstring"],
                decorators=item["decorators"],
            )
            symbols.append(sym)
            can_key = sym.canonical_key
            canonical_to_id[can_key] = nid
            id_to_canonical[nid] = can_key

            file_to_symbols.setdefault(sym.file_path, []).append(nid)
            suffix_to_ids.setdefault(sym.qualified_name.split(".")[-1], []).append(nid)
            qualified_to_ids.setdefault(sym.qualified_name, []).append(nid)

    # Class inheritance map: class_key -> list of base class names
    class_bases_map: Dict[str, List[str]] = {}
    for collector in all_collectors:
        for item in collector.symbols_raw:
            if item["symbol_type"] == "class" and "bases" in item:
                c_key = f"{item['file_path']}::{item['qualified_name']}"
                class_bases_map[c_key] = item["bases"]

    # Pass 2: Resolve caller-callee, inheritance MRO, and import edges
    call_edges_set: Set[Tuple[int, int]] = set()
    caller_edges_set: Set[Tuple[int, int]] = set()
    inherit_edges_set: Set[Tuple[int, int]] = set()
    import_edges_set: Set[Tuple[int, int]] = set()

    for collector in all_collectors:
        file_path = collector.file_path
        import_table = collector.import_table

        for caller_qname, callee_name in collector.raw_calls:
            caller_key = f"{file_path}::{caller_qname}"
            u = canonical_to_id.get(caller_key)
            if u is None:
                continue

            v = None
            caller_sym = symbols[u]
            is_inherit_call = False

            # 1. Intra-class call resolution (self.method() or cls.method() or unqualified call)
            if caller_sym.parent_class:
                is_self_or_cls = callee_name.startswith(("self.", "cls."))
                is_unqualified = "." not in callee_name
                if is_self_or_cls or is_unqualified:
                    attr_name = callee_name.split(".")[-1]
                    target_qname = f"{caller_sym.parent_class}.{attr_name}"
                    target_key = f"{file_path}::{target_qname}"
                    if target_key in canonical_to_id:
                        v = canonical_to_id[target_key]
                    else:
                        # MRO Base Class Resolution: Check parent class hierarchy
                        p_class_key = f"{file_path}::{caller_sym.parent_class}"
                        bases = class_bases_map.get(p_class_key, [])
                        for b_name in bases:
                            # Try intra-file base class
                            base_target = f"{file_path}::{b_name}.{attr_name}"
                            if base_target in canonical_to_id:
                                v = canonical_to_id[base_target]
                                is_inherit_call = True
                                break
                            # Try cross-file imported base class
                            if b_name in import_table.imported_symbols:
                                mod_pfx, orig_b = import_table.imported_symbols[b_name]
                                b_actual = orig_b or b_name
                                for c_id in qualified_to_ids.get(f"{b_actual}.{attr_name}", ()):
                                    if _matches_module_path(symbols[c_id].file_path, mod_pfx):
                                        v = c_id
                                        is_inherit_call = True
                                        break
                            if v is not None:
                                break

            # 2. Intra-file function call resolution
            if v is None:
                intra_file_key = f"{file_path}::{callee_name}"
                if intra_file_key in canonical_to_id:
                    v = canonical_to_id[intra_file_key]

            # 3. Import-aware cross-file resolution
            if v is None:
                base_callee = callee_name.split(".")[0]
                if base_callee in import_table.imported_symbols:
                    mod_prefix, orig_sym = import_table.imported_symbols[base_callee]
                    target_name = orig_sym or base_callee
                    for candidate_id in qualified_to_ids.get(target_name, ()):
                        cand_sym = symbols[candidate_id]
                        if _matches_module_path(cand_sym.file_path, mod_prefix):
                            v = candidate_id
                            import_edges_set.add((u, v))
                            break

                elif "." in callee_name and base_callee in import_table.imported_modules:
                    mod_path = import_table.imported_modules[base_callee]
                    attr_name = callee_name.split(".", 1)[1]
                    for candidate_id in qualified_to_ids.get(attr_name, ()):
                        cand_sym = symbols[candidate_id]
                        if _matches_module_path(cand_sym.file_path, mod_path):
                            v = candidate_id
                            import_edges_set.add((u, v))
                            break

            # 4. Unambiguous repository-wide suffix match fallback
            if v is None:
                parts = callee_name.split(".")
                if len(parts) == 1 or (len(parts) == 2 and parts[0] in ("self", "cls")):
                    short_name = parts[-1]
                    candidates = suffix_to_ids.get(short_name, [])
                    if len(candidates) == 1:
                        v = candidates[0]

            if v is not None and u != v:
                if is_inherit_call:
                    inherit_edges_set.add((u, v))
                else:
                    call_edges_set.add((u, v))
                    caller_edges_set.add((v, u))

        # Add explicit class-to-base inheritance edges
        for item in collector.symbols_raw:
            if item["symbol_type"] == "class" and "bases" in item:
                c_id = canonical_to_id.get(f"{file_path}::{item['qualified_name']}")
                if c_id is None:
                    continue
                for b_name in item["bases"]:
                    for cand_id in suffix_to_ids.get(b_name.split(".")[-1], []):
                        if symbols[cand_id].symbol_type == "class" and cand_id != c_id:
                            inherit_edges_set.add((c_id, cand_id))

    # Add hierarchical class-to-method containment edges
    for sym in symbols:
        if sym.parent_class:
            class_key = f"{sym.file_path}::{sym.parent_class}"
            p_id = canonical_to_id.get(class_key)
            if p_id is not None and p_id != sym.node_id:
                call_edges_set.add((p_id, sym.node_id))
                caller_edges_set.add((sym.node_id, p_id))

    # Deterministic sorting of all relation edge tuples
    call_edges = sorted(list(call_edges_set))
    caller_edges = sorted(list(caller_edges_set))
    inherit_edges = sorted(list(inherit_edges_set))
    import_edges = sorted(list(import_edges_set))

    if return_multiplex:
        return symbols, call_edges, caller_edges, canonical_to_id, id_to_canonical, inherit_edges, import_edges
    return symbols, call_edges, caller_edges, canonical_to_id, id_to_canonical


def compile_and_save_repository_graph(
    repo_dir: Path,
    output_dir: Path,
    lambda_call: float = 1.0,
    lambda_caller: float = 0.2,
    use_multiplex: bool = True,
    omega_call: float = 0.50,
    omega_inherit: float = 0.25,
    omega_import: float = 0.15,
    omega_caller: float = 0.10,
) -> Tuple[csr_matrix, np.ndarray, Dict[str, int], Dict[int, str], List[ASTSymbolNode]]:
    """
    Extracts AST graph, builds static CSR transition matrix, and saves zero-copy mmap binaries.
    Supports Perron 2.0 Multiplex Spectral Diffusion across calls, inheritance MRO, imports, and callers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    res = extract_repository_graph(repo_dir, return_multiplex=True)
    symbols, call_edges, caller_edges, canonical_to_id, id_to_canonical, inherit_edges, import_edges = res
    num_nodes = len(symbols)

    if use_multiplex:
        t_matrix, dangling = build_static_transition_matrix(
            num_nodes=num_nodes,
            call_edges=call_edges,
            caller_edges=caller_edges,
            inherit_edges=inherit_edges,
            import_edges=import_edges,
            omega_call=omega_call,
            omega_inherit=omega_inherit,
            omega_import=omega_import,
            omega_caller=omega_caller,
        )
    else:
        t_matrix, dangling = build_static_transition_matrix(
            num_nodes=num_nodes,
            call_edges=call_edges,
            caller_edges=caller_edges,
            lambda_call=lambda_call,
            lambda_caller=lambda_caller,
        )

    save_mmap_csr(output_dir, t_matrix, dangling, canonical_to_id, id_to_canonical)

    # Compute and save stationary PageRank distribution for static call graph
    from perron.specificity import compute_global_pagerank
    pi_global = compute_global_pagerank(t_matrix, dangling, beta=0.85, max_iter=100)
    np.save(output_dir / "pi_global.npy", pi_global.astype(np.float64))

    # Save full symbol representations with code segments
    symbols_json_path = output_dir / "id_to_symbol.json"
    with open(symbols_json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in symbols], f, indent=2)

    return t_matrix, dangling, canonical_to_id, id_to_canonical, symbols
