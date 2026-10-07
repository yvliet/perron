"""
perron.reranker.provenance - Structural Call-Path Provenance Resolver and Renderer.

Resolves causal call-graph and AST dependency provenance connecting candidate
symbols back to issue seed locations, addressing the Lexical Abstraction Bottleneck
where compact language models fail to recognize non-lexical graph-expanded targets.

Author: Sultan Haikal (@yvliet)
License: Apache-2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class EdgeType(str, Enum):
    SEED = "seed"
    CALLED_BY_SEED = "called_by_seed"
    CALLS_SEED = "calls_seed"
    SUBCLASSES_SEED = "subclasses_seed"
    BASE_CLASS_OF_SEED = "base_class_of_seed"
    IMPORTED_BY_SEED = "imported_by_seed"
    IMPORTS_SEED = "imports_seed"
    CO_LOCATED = "co_located"
    LEXICAL_NEIGHBOR = "lexical_neighbor"
    UNKNOWN = "unknown"

    def to_human_label(self) -> str:
        mapping = {
            EdgeType.SEED: "Issue seed retrieved via lexical search",
            EdgeType.CALLED_BY_SEED: "Called by seed function",
            EdgeType.CALLS_SEED: "Calls seed function",
            EdgeType.SUBCLASSES_SEED: "Subclasses seed class",
            EdgeType.BASE_CLASS_OF_SEED: "Base class of seed class",
            EdgeType.IMPORTED_BY_SEED: "Imported by seed module",
            EdgeType.IMPORTS_SEED: "Imports seed module",
            EdgeType.CO_LOCATED: "Co-located in same file as seed",
            EdgeType.LEXICAL_NEIGHBOR: "Lexical depth neighbor",
            EdgeType.UNKNOWN: "Graph neighbor",
        }
        return mapping.get(self, "Graph neighbor")


@dataclass(frozen=True)
class SymbolProvenance:
    edge_type: EdgeType
    seed_symbol_id: Optional[int] = None
    seed_identifier: Optional[str] = None
    seed_file_path: Optional[str] = None
    hops: int = 1

    def format_comment(self) -> str:
        """Formats a clean inline header comment for prompt inclusion."""
        if self.edge_type == EdgeType.SEED:
            return "# Provenance: Direct lexical seed from issue query"
        if self.seed_identifier:
            file_hint = f" ({self.seed_file_path})" if self.seed_file_path else ""
            label = self.edge_type.to_human_label()
            return f"# Provenance: {label} '{self.seed_identifier}'{file_hint}"
        if self.edge_type == EdgeType.CO_LOCATED:
            file_hint = f" in '{self.seed_file_path}'" if self.seed_file_path else ""
            return f"# Provenance: Co-located in same file as seed{file_hint}"
        return f"# Provenance: {self.edge_type.to_human_label()}"


class ProvenanceResolver:
    """
    Resolves exact structural relations between candidates and seed symbols
    using precomputed AST multigraph layer arrays.
    """

    def __init__(self, graphs_dir: Path | str = "artifacts/swebench_graphs"):
        self.graphs_dir = Path(graphs_dir)
        self._layers_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._symbols_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_repo_data(self, repo_slug: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[List[Dict[str, Any]]]]:
        if repo_slug not in self._layers_cache:
            layers_path = self.graphs_dir / f"{repo_slug}_layers.npz"
            symbols_path = self.graphs_dir / f"{repo_slug}_symbols.json"
            if not layers_path.exists() or not symbols_path.exists():
                return None, None
            try:
                npz = np.load(layers_path)
                self._layers_cache[repo_slug] = {k: npz[k] for k in npz.files}
                self._symbols_cache[repo_slug] = json.loads(symbols_path.read_text(encoding="utf-8"))
            except Exception:
                return None, None
        return self._layers_cache[repo_slug], self._symbols_cache[repo_slug]

    def resolve(
        self,
        candidate_symbol_id: int,
        seed_symbol_ids: Set[int] | Sequence[int],
        repo_slug: str,
        candidate_file_path: Optional[str] = None,
    ) -> SymbolProvenance:
        """
        Resolves the highest-priority structural relation between candidate and seeds.
        Priority:
        1. Direct seed
        2. Direct call (called by seed or calls seed)
        3. Inheritance (subclass or base class)
        4. Import relation
        5. Co-located file
        """
        seeds_set = set(seed_symbol_ids)
        if candidate_symbol_id in seeds_set:
            return SymbolProvenance(edge_type=EdgeType.SEED, seed_symbol_id=candidate_symbol_id, hops=0)

        layers, symbols = self._load_repo_data(repo_slug)
        if not layers or not symbols:
            return SymbolProvenance(edge_type=EdgeType.UNKNOWN)

        # 1. Check call edges: [u, v] means u calls v
        if "call_edges" in layers:
            for u, v in layers["call_edges"]:
                if u in seeds_set and v == candidate_symbol_id:
                    s_sym = symbols[u] if u < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.CALLED_BY_SEED,
                        seed_symbol_id=int(u),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )
                elif v in seeds_set and u == candidate_symbol_id:
                    s_sym = symbols[v] if v < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.CALLS_SEED,
                        seed_symbol_id=int(v),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )

        # 2. Check inheritance: [u, v] means u inherits from v
        if "inherit_edges" in layers:
            for u, v in layers["inherit_edges"]:
                if u in seeds_set and v == candidate_symbol_id:
                    s_sym = symbols[u] if u < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.BASE_CLASS_OF_SEED,
                        seed_symbol_id=int(u),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )
                elif v in seeds_set and u == candidate_symbol_id:
                    s_sym = symbols[v] if v < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.SUBCLASSES_SEED,
                        seed_symbol_id=int(v),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )

        # 3. Check import edges: [u, v] means u imports v
        if "import_edges" in layers:
            for u, v in layers["import_edges"]:
                if u in seeds_set and v == candidate_symbol_id:
                    s_sym = symbols[u] if u < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.IMPORTED_BY_SEED,
                        seed_symbol_id=int(u),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )
                elif v in seeds_set and u == candidate_symbol_id:
                    s_sym = symbols[v] if v < len(symbols) else {}
                    return SymbolProvenance(
                        edge_type=EdgeType.IMPORTS_SEED,
                        seed_symbol_id=int(v),
                        seed_identifier=s_sym.get("identifier"),
                        seed_file_path=s_sym.get("file_path"),
                        hops=1,
                    )

        # 4. Check file co-location with any seed
        if candidate_file_path:
            for s_id in seeds_set:
                if s_id < len(symbols):
                    if symbols[s_id].get("file_path") == candidate_file_path:
                        return SymbolProvenance(
                            edge_type=EdgeType.CO_LOCATED,
                            seed_symbol_id=s_id,
                            seed_identifier=symbols[s_id].get("identifier"),
                            seed_file_path=candidate_file_path,
                            hops=1,
                        )

        return SymbolProvenance(edge_type=EdgeType.UNKNOWN)


class CallPathProvenanceRenderer:
    """
    Wraps existing breadcrumbs or raw candidate dictionaries with structural
    call-path provenance metadata.
    """

    def __init__(self, resolver: Optional[ProvenanceResolver] = None):
        self.resolver = resolver or ProvenanceResolver()

    def format_breadcrumb_with_provenance(
        self,
        base_breadcrumb: str,
        provenance: SymbolProvenance,
    ) -> str:
        """
        Injects provenance header comment into candidate breadcrumb.
        """
        prov_comment = provenance.format_comment()
        lines = base_breadcrumb.splitlines()
        # Find where to inject: after the '# File: ...' line if present
        if lines and lines[0].startswith("# File:"):
            return lines[0] + "\n" + prov_comment + "\n" + "\n".join(lines[1:])
        return prov_comment + "\n" + base_breadcrumb
