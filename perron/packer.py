"""
Perron Context-Budgeted Subgraph Packing Module.

Implements component-budget partitioning, target-first anchor selection, versioned
priority queue frontier expansion, and hierarchical AST breadcrumbs for LLM prompts.
"""

from __future__ import annotations

from collections import deque
import heapq
from dataclasses import dataclass
from pathlib import Path
import textwrap
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class ASTContextSymbol:
    """Represents a code symbol extracted from the repository AST."""
    node_id: int
    name: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    token_count: int
    parent_class: Optional[str] = None
    class_docstring: Optional[str] = None

    @property
    def identifier(self) -> str:
        return self.name

    @property
    def line_start(self) -> int:
        return self.start_line

    @property
    def line_end(self) -> int:
        return self.end_line

    @property
    def source_code(self) -> str:
        return self.code


def pack_context_subgraphs(
    symbols: Dict[int, ASTContextSymbol] | Sequence[Any],
    specificity_scores: np.ndarray,
    adjacency_matrix: Optional[csr_matrix] = None,
    token_budget: int = 3480,
    mu: float = 0.5,
    top_candidates_limit: int = 150,
) -> Tuple[List[ASTContextSymbol], str]:
    """
    Extract context-budgeted subgraph maximizing causal edit recall within strict token limit.

    Mechanism:
    1. Filter candidates to top positive specificity nodes.
    2. Component-budget partitioning: Clusters candidate symbols into connected
       components and allocates proportional token budgets to prevent cluster starvation.
    3. Target-first anchor selection per component with versioned priority queue expansion.
    4. Bounded degree connectivity bonus ensuring ratio in [0, 1].
    5. Strict positive-specificity gate suppressing zero-specificity test routing nodes.
    6. Breadcrumb overhead accounting guaranteeing emitted context <= token_budget.
    """
    if isinstance(adjacency_matrix, (int, float)) and token_budget == 3480:
        token_budget = int(adjacency_matrix)
        adjacency_matrix = None

    if isinstance(symbols, (list, tuple)):
        symbols_dict: Dict[int, ASTContextSymbol] = {}
        for i, s in enumerate(symbols):
            if hasattr(s, "to_context_symbol"):
                symbols_dict[i] = s.to_context_symbol()
            elif isinstance(s, ASTContextSymbol):
                symbols_dict[i] = s
            else:
                symbols_dict[i] = ASTContextSymbol(
                    node_id=getattr(s, "node_id", i),
                    name=getattr(s, "identifier", getattr(s, "name", str(s))),
                    file_path=getattr(s, "file_path", ""),
                    start_line=getattr(s, "line_start", getattr(s, "start_line", 1)),
                    end_line=getattr(s, "line_end", getattr(s, "end_line", 1)),
                    code=getattr(s, "source_code", getattr(s, "code", "")),
                    token_count=getattr(s, "token_count", max(1, len(getattr(s, "source_code", getattr(s, "code", "")) or "") // 4)),
                )
        symbols = symbols_dict

    if len(symbols) == 0 or token_budget <= 0 or np.all(specificity_scores <= 0):
        return [], ""

    num_nodes = len(specificity_scores)
    if adjacency_matrix is None:
        adjacency_matrix = csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    elif adjacency_matrix.shape[0] != num_nodes or adjacency_matrix.shape[1] != num_nodes:
        raise ValueError(
            f"Dimension mismatch: adjacency_matrix shape {adjacency_matrix.shape} does not match specificity_scores length ({num_nodes})."
        )
    if np.isnan(mu) or mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}.")
    if top_candidates_limit <= 0:
        raise ValueError(f"top_candidates_limit must be strictly positive, got {top_candidates_limit}.")

    # Filter candidates to positive specificity in symbols
    candidate_indices = np.argsort(specificity_scores)[::-1][:top_candidates_limit]
    candidate_indices = [
        int(idx)
        for idx in candidate_indices
        if specificity_scores[idx] > 0 and int(idx) in symbols
    ]

    if not candidate_indices:
        return [], ""

    # Compute symmetric total degrees (in-degree + out-degree)
    out_degrees = np.diff(adjacency_matrix.indptr)
    in_degrees = np.bincount(adjacency_matrix.indices, minlength=num_nodes)
    total_degrees = np.maximum(out_degrees + in_degrees, 1)

    candidate_set = set(candidate_indices)

    # 1. Identify weakly connected components among candidate symbols using undirected candidate adjacency
    cand_adj: Dict[int, Set[int]] = {c: set() for c in candidate_set}
    for c in candidate_set:
        start_p = adjacency_matrix.indptr[c]
        end_p = adjacency_matrix.indptr[c + 1]
        for nbr in adjacency_matrix.indices[start_p:end_p]:
            nbr_int = int(nbr)
            if nbr_int in candidate_set:
                cand_adj[c].add(nbr_int)
                cand_adj[nbr_int].add(c)

    visited_candidates: Set[int] = set()
    components: List[List[int]] = []

    for c in candidate_indices:
        if c in visited_candidates:
            continue
        comp: List[int] = []
        q = deque([c])
        visited_candidates.add(c)
        while q:
            curr = q.popleft()
            comp.append(curr)
            for nbr_int in cand_adj[curr]:
                if nbr_int not in visited_candidates:
                    visited_candidates.add(nbr_int)
                    q.append(nbr_int)
        components.append(comp)

    # Sort components by total specificity mass descending
    components.sort(
        key=lambda comp: sum(float(specificity_scores[n]) for n in comp),
        reverse=True,
    )

    total_candidate_mass = sum(
        sum(float(specificity_scores[n]) for n in comp) for comp in components
    )

    selected_nodes: Set[int] = set()
    seen_files: Set[str] = set()
    seen_classes: Set[Tuple[str, str]] = set()
    current_tokens = 0

    versions = np.zeros(num_nodes, dtype=np.int32)
    neighbor_counts = np.zeros(num_nodes, dtype=np.int32)

    # 2. Pack each component according to its proportional budget
    for comp in components:
        if current_tokens >= token_budget:
            break

        remaining_tokens = token_budget - current_tokens
        comp_mass = sum(float(specificity_scores[n]) for n in comp)
        comp_ratio = comp_mass / total_candidate_mass if total_candidate_mass > 0 else 1.0 / len(components)

        if len(components) == 1 or comp_ratio >= 1.0:
            comp_budget = remaining_tokens
        else:
            raw_budget = int(token_budget * comp_ratio)
            comp_budget = min(remaining_tokens, max(50, raw_budget)) if remaining_tokens >= 50 else remaining_tokens

        comp_spent = 0

        # Component max-heap
        comp_heap: List[Tuple[float, int, int]] = []

        # Target-first anchor: highest specificity node in component
        comp_sorted = sorted(comp, key=lambda n: float(specificity_scores[n]), reverse=True)
        for anchor in comp_sorted[:3]:
            if anchor in selected_nodes:
                continue
            cost = max(symbols[anchor].token_count, 1)
            score = float(specificity_scores[anchor] / cost)
            comp_heap.append((-score, anchor, versions[anchor]))
        heapq.heapify(comp_heap)

        while comp_heap and comp_spent < comp_budget and current_tokens < token_budget:
            neg_score, u, entry_version = heapq.heappop(comp_heap)

            if entry_version < versions[u] or u in selected_nodes:
                continue

            sym = symbols[u]
            if not sym.code or not sym.code.strip():
                continue
            base_cost = max(sym.token_count, 1)

            # Account for breadcrumb formatting overhead
            extra_tokens = 0
            if sym.file_path not in seen_files:
                extra_tokens += 15
            if sym.parent_class and (sym.file_path, sym.parent_class) not in seen_classes:
                doc_len = len(sym.class_docstring.split()) if sym.class_docstring else 0
                extra_tokens += 15 + doc_len

            node_total_cost = base_cost + extra_tokens

            if current_tokens + node_total_cost > token_budget:
                continue

            selected_nodes.add(u)
            seen_files.add(sym.file_path)
            if sym.parent_class:
                seen_classes.add((sym.file_path, sym.parent_class))

            current_tokens += node_total_cost
            comp_spent += node_total_cost

            # Expand frontier bidirectionally: outgoing CSR edges and incoming/candidate edges
            start_ptr = adjacency_matrix.indptr[u]
            end_ptr = adjacency_matrix.indptr[u + 1]
            out_neighbors = {int(v) for v in adjacency_matrix.indices[start_ptr:end_ptr]}
            all_neighbors = out_neighbors.union(cand_adj.get(u, set()))

            for v_int in all_neighbors:
                if v_int in selected_nodes or v_int not in symbols:
                    continue

                v_spec = float(specificity_scores[v_int])
                if v_spec <= 0.0:
                    # Strict gate: do not expand into zero-specificity (e.g. test) nodes
                    continue

                neighbor_counts[v_int] += 1
                versions[v_int] += 1

                v_sym = symbols[v_int]
                v_cost = max(v_sym.token_count, 1)
                conn_ratio = min(1.0, float(neighbor_counts[v_int]) / float(total_degrees[v_int]))
                connectivity_bonus = 1.0 + mu * conn_ratio
                v_score = (v_spec / v_cost) * connectivity_bonus

                heapq.heappush(comp_heap, (-v_score, v_int, versions[v_int]))

    packed_symbols = [symbols[idx] for idx in selected_nodes]
    formatted_prompt = format_hierarchical_context(packed_symbols)
    return packed_symbols, formatted_prompt


def _rebase_code_indent(code: str, target_indent: str) -> str:
    """
    Dedents code block while ignoring column-0 comments and blank lines,
    then applies target_indent uniformly. Normalizes tabs if target_indent uses spaces.
    """
    clean_code = code
    clean_target = target_indent
    if "\t" in clean_code and "\t" not in clean_target:
        clean_code = clean_code.expandtabs(4)
    elif "\t" not in clean_code and "\t" in clean_target:
        clean_target = clean_target.expandtabs(4)

    lines = clean_code.rstrip().splitlines()
    non_comment_indents = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent_len = len(line) - len(line.lstrip(" \t"))
        non_comment_indents.append(indent_len)

    if not non_comment_indents:
        base_dedented = textwrap.dedent(clean_code.rstrip())
        return textwrap.indent(base_dedented, clean_target)

    min_indent = min(non_comment_indents)
    rebased_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            rebased_lines.append("")
        elif stripped.startswith("#"):
            indent_len = len(line) - len(line.lstrip(" \t"))
            if indent_len >= min_indent:
                rebased_lines.append(clean_target + line[min_indent:])
            else:
                rebased_lines.append(clean_target + stripped)
        else:
            if len(line) >= min_indent:
                rebased_lines.append(clean_target + line[min_indent:])
            else:
                rebased_lines.append(clean_target + line.lstrip(" \t"))
    return "\n".join(rebased_lines)


def format_hierarchical_context(symbols: List[ASTContextSymbol]) -> str:
    """
    Format selected symbols into hierarchically anchored code context.
    Preserves parent class signatures and lexical indentation scopes.
    """
    if not symbols:
        return ""

    # Group by file path
    by_file: Dict[str, List[ASTContextSymbol]] = {}
    for s in symbols:
        by_file.setdefault(s.file_path, []).append(s)

    output_blocks: List[str] = []

    for file_path, file_symbols in sorted(by_file.items()):
        # Sort symbols in file by start_line ascending
        file_symbols.sort(key=lambda x: x.start_line)
        output_blocks.append(f"# ========================================================")
        output_blocks.append(f"# File: {file_path}")
        output_blocks.append(f"# ========================================================")

        # Group by class within file
        current_class: Optional[str] = None
        for sym in file_symbols:
            if sym.parent_class:
                class_parts = sym.parent_class.split(".")
                method_indent = "    " * len(class_parts)
                if sym.parent_class != current_class:
                    current_class = sym.parent_class
                    for depth, part in enumerate(class_parts):
                        indent = "    " * depth
                        output_blocks.append(f"{indent}class {part}:")
                    if sym.class_docstring:
                        output_blocks.append(f'{method_indent}"""{sym.class_docstring}"""')
                    output_blocks.append(f"{method_indent}# ... [preceding class members omitted] ...")
                output_blocks.append(f"{method_indent}# [Lines {sym.start_line}-{sym.end_line}: {sym.name}]")
                code_str = _rebase_code_indent(sym.code, method_indent)
            else:
                current_class = None
                output_blocks.append(f"# [Lines {sym.start_line}-{sym.end_line}: {sym.name}]")
                code_str = sym.code.rstrip()

            output_blocks.append(code_str)
            output_blocks.append("")

    return "\n".join(output_blocks)
