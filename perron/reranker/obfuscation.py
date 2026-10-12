"""
perron.reranker.obfuscation - Syntax-Preserving AST Identifier Obfuscator.

Implements causal alpha-renaming of user-defined symbols (functions, parameters,
local variables, classes) while preserving language keywords, built-ins, control
flow, and AST structure intact.

Used in causal perturbation studies to evaluate model sensitivity to surface lexical
tokens versus topological code relationships.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import ast
import re
from typing import Dict, Set, Tuple


class IdentifierObfuscator(ast.NodeTransformer):
    """
    Transforms a Python AST by systematically alpha-renaming all user-defined
    identifiers to neutral synthetic labels (fn_1, v_1, Class_1).
    """

    PYTHON_BUILTINS: Set[str] = {
        "self", "cls", "True", "False", "None", "print", "len", "range",
        "int", "str", "float", "list", "dict", "set", "tuple", "bool", "bytes",
        "super", "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
        "Exception", "ValueError", "TypeError", "KeyError", "AttributeError",
        "IndexError", "RuntimeError", "StopIteration", "NotImplementedError",
        "open", "zip", "enumerate", "any", "all", "min", "max", "sum", "sorted",
        "map", "filter", "reversed", "iter", "next", "abs", "round", "repr"
    }

    def __init__(self) -> None:
        super().__init__()
        self.mapping: Dict[str, str] = {}
        self.fn_count: int = 0
        self.var_count: int = 0
        self.cls_count: int = 0

    def _get_synthetic_name(self, original: str, kind: str) -> str:
        if original in self.PYTHON_BUILTINS or (original.startswith("__") and original.endswith("__")):
            return original
        if original not in self.mapping:
            if kind == "fn":
                self.fn_count += 1
                self.mapping[original] = f"fn_{self.fn_count}"
            elif kind == "cls":
                self.cls_count += 1
                self.mapping[original] = f"Class_{self.cls_count}"
            else:
                self.var_count += 1
                self.mapping[original] = f"v_{self.var_count}"
        return self.mapping[original]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.name = self._get_synthetic_name(node.name, "fn")
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node.name = self._get_synthetic_name(node.name, "fn")
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.name = self._get_synthetic_name(node.name, "cls")
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg not in ("self", "cls"):
            node.arg = self._get_synthetic_name(node.arg, "var")
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id not in self.PYTHON_BUILTINS and not (node.id.startswith("__") and node.id.endswith("__")):
            node.id = self._get_synthetic_name(node.id, "var")
        return node


def obfuscate_code_snippet(source: str) -> Tuple[str, Dict[str, str]]:
    """
    Alpha-renames user-defined identifiers in a Python snippet.

    Returns:
        Tuple of (obfuscated_source, identifier_mapping).
        If the snippet is incomplete syntax, falls back to a regex word-masker.
    """
    if not source or not source.strip():
        return source, {}

    try:
        tree = ast.parse(source)
        transformer = IdentifierObfuscator()
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)
        obfuscated = ast.unparse(new_tree)
        return obfuscated, transformer.mapping
    except Exception:
        # Fallback for partial/truncated code snippets that fail ast.parse
        mapping: Dict[str, str] = {}
        counter = [0]

        def repl(match: re.Match[str]) -> str:
            word = match.group(0)
            if word in IdentifierObfuscator.PYTHON_BUILTINS or word.startswith("__"):
                return word
            if word not in mapping:
                counter[0] += 1
                mapping[word] = f"ident_{counter[0]}"
            return mapping[word]

        pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
        obfuscated = re.sub(pattern, repl, source)
        return obfuscated, mapping
