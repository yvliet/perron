"""
Tree-sitter TypeScript and JavaScript AST Extractor for Perron.
Extracts symbols, function declarations, class definitions, and directed call/inheritance edges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from perron.graph import ASTSymbolNode

logger = logging.getLogger("perron.polyglot")

try:
    import tree_sitter
    import tree_sitter_typescript

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


class TreeSitterExtractor:
    """Extracts symbols and edges from TypeScript/JavaScript repositories."""

    def __init__(self):
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "Tree-sitter dependencies not installed. "
                "Install polyglot extras: pip install perron-core[polyglot]"
            )
        self.language = tree_sitter.Language(tree_sitter_typescript.language_typescript())
        try:
            self.parser = tree_sitter.Parser(self.language)
        except (TypeError, ValueError):
            self.parser = tree_sitter.Parser()
            self.parser.language = self.language

    def extract_file(self, file_path: Path, repo_root: Path) -> List[ASTSymbolNode]:
        """Parse a single TypeScript/JavaScript file into ASTSymbolNode objects."""
        try:
            with open(file_path, "rb") as f:
                code_bytes = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return []

        tree = self.parser.parse(code_bytes)
        root_node = tree.root_node
        rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

        symbols: List[ASTSymbolNode] = []
        self._traverse_tree(root_node, rel_path, code_bytes, symbols)
        return symbols

    def _traverse_tree(
        self,
        node: tree_sitter.Node,
        rel_path: str,
        code_bytes: bytes,
        symbols: List[ASTSymbolNode],
        parent_class: Optional[str] = None,
    ) -> None:
        node_type = node.type

        # 1. Function and Method Declarations
        if node_type in ("function_declaration", "method_definition", "arrow_function"):
            identifier = self._get_node_name(node, code_bytes)
            if identifier:
                full_name = f"{parent_class}.{identifier}" if parent_class else identifier
                source = code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
                calls = self._extract_call_identifiers(node, code_bytes)

                symbols.append(
                    ASTSymbolNode(
                        node_id=len(symbols),
                        identifier=full_name,
                        file_path=rel_path,
                        symbol_type="function" if node_type != "method_definition" else "method",
                        line_start=node.start_point[0] + 1,  # 1-indexed line
                        line_end=node.end_point[0] + 1,
                        source_code=source,
                        docstring=None,
                        calls=calls,
                        base_classes=[],
                    )
                )

        # 2. Class and Interface Declarations
        elif node_type in ("class_declaration", "interface_declaration"):
            class_name = self._get_node_name(node, code_bytes)
            if class_name:
                bases = self._extract_heritage_clauses(node, code_bytes)
                source = code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

                symbols.append(
                    ASTSymbolNode(
                        node_id=len(symbols),
                        identifier=class_name,
                        file_path=rel_path,
                        symbol_type="class",
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        source_code=source,
                        docstring=None,
                        calls=[],
                        base_classes=bases,
                    )
                )
                for child in node.children:
                    self._traverse_tree(child, rel_path, code_bytes, symbols, parent_class=class_name)
                return

        for child in node.children:
            self._traverse_tree(child, rel_path, code_bytes, symbols, parent_class=parent_class)

    def _get_node_name(self, node: tree_sitter.Node, code_bytes: bytes) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node:
            return code_bytes[name_node.start_byte : name_node.end_byte].decode("utf-8", errors="replace")
        for child in node.children:
            if child.type in ("identifier", "property_identifier", "type_identifier"):
                return code_bytes[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
        return None

    def _extract_call_identifiers(self, node: tree_sitter.Node, code_bytes: bytes) -> List[str]:
        calls: Set[str] = set()

        def _walk(n: tree_sitter.Node):
            if n.type == "call_expression":
                fn_node = n.child_by_field_name("function")
                if fn_node:
                    if fn_node.type == "identifier":
                        calls.add(code_bytes[fn_node.start_byte : fn_node.end_byte].decode("utf-8", errors="replace"))
                    elif fn_node.type == "member_expression":
                        prop = fn_node.child_by_field_name("property")
                        if prop:
                            calls.add(code_bytes[prop.start_byte : prop.end_byte].decode("utf-8", errors="replace"))
            for c in n.children:
                _walk(c)

        _walk(node)
        return sorted(calls)

    def _extract_heritage_clauses(self, node: tree_sitter.Node, code_bytes: bytes) -> List[str]:
        bases: List[str] = []
        for child in node.children:
            if child.type == "class_heritage":
                for clause in child.children:
                    if clause.type in ("extends_clause", "implements_clause"):
                        for base in clause.children:
                            if base.type in ("identifier", "type_identifier"):
                                bases.append(code_bytes[base.start_byte : base.end_byte].decode("utf-8", errors="replace"))
        return bases


def extract_typescript_graph(repo_dir: Path) -> List[ASTSymbolNode]:
    """Scan directory and extract symbols for all .ts and .js files."""
    extractor = TreeSitterExtractor()
    all_symbols: List[ASTSymbolNode] = []
    for ext in ("*.ts", "*.tsx", "*.js", "*.jsx"):
        for path in repo_dir.rglob(ext):
            if "node_modules" in path.parts or ".git" in path.parts:
                continue
            all_symbols.extend(extractor.extract_file(path, repo_dir))
    return all_symbols
